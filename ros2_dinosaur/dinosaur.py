# General imports
import argparse
import os
import numpy as np
import torch
from torch.utils.cpp_extension import CUDA_HOME
from PIL import Image, ImageDraw, ImageFont
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ImageMsg, CameraInfo
from std_msgs.msg import String
from geometry_msgs.msg import PoseArray, Pose
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge, CvBridgeError
from pathlib import Path
import urllib.request

# Grounding DINO
if CUDA_HOME is not None:
    os.environ['CUDA_HOME'] = CUDA_HOME
else:
    print('CUDA_HOME is not set. This may cause errors with groundingdino.')
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import annotate, predict
from huggingface_hub import hf_hub_download

# segment anything
from segment_anything import build_sam, SamPredictor 
import numpy as np


# depth estimation
from transformers import pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print("Using CUDA version:", torch.version.cuda)
    print("Using GPU:", torch.cuda.get_device_name(0))


def load_model_hf(repo_id, filename, ckpt_config_filename):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model


def calculate_3d_position(x, y, depth, camera_info):
    """
    Calculate the 3D position in camera coordinates from normalized 2D image coordinates and depth.

    Parameters:
    - x, y: float, normalized image coordinates (0 to 1) relative to image width and height
    - depth: float, depth in meters
    - camera_info: sensor_msgs/CameraInfo message containing camera parameters

    Returns:
    - (X, Y, Z): Tuple of float, 3D position in camera coordinates
    """
    if camera_info is None:
        # Assume ideal pinhole camera model if no camera_info is provided
        camera_info = CameraInfo()
        camera_info.width = 640
        camera_info.height = 480
        camera_info.k = np.array([400, 0, 320, 0, 400, 240, 0, 0, 1], dtype=np.float64)

    # Extract the intrinsic matrix K from camera_info
    K = np.array(camera_info.k).reshape(3, 3)
    
    # Convert normalized coordinates to pixel coordinates
    u = x * camera_info.width
    v = y * camera_info.height

    # Convert pixel coordinates to normalized camera coordinates
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    Xc = (u - cx) * depth / fx
    Yc = (v - cy) * depth / fy
    Zc = depth

    print(f"Xc: {Xc}, Yc: {Yc}, Zc: {Zc}, u: {u}, v: {v}, depth: {depth}, fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")

    return float(Xc), float(Yc), float(Zc)


class ObjectDetectionAndLocalizationNode(Node):
    def __init__(self, image_topic: str, depth_topic: str, camera_info_topic: str, frame_id: str):
        super().__init__('object_detection_and_localization_node')
        self.frame_id = frame_id

        # load models
        self.groundingdino_model = load_model_hf("ShilongLiu/GroundingDINO", "groundingdino_swinb_cogcoor.pth", "GroundingDINO_SwinB.cfg.py")
        sam_path = Path('~/.ros/obj_detection/').expanduser() / 'sam_vit_h_4b8939.pth'
        sam_path.parent.mkdir(parents=True, exist_ok=True)
        if not sam_path.exists():
            def show_progress(block_num, block_size, total_size):
                print(f"Downloading SAM model: {block_num * block_size / total_size * 100:.2f}%", end='\r')
            urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth', sam_path, show_progress)
            print()
        sam = build_sam(checkpoint=sam_path)
        sam.to(device=device)
        self.sam_predictor = SamPredictor(sam)

        self.subscription_video = self.create_subscription(
            ImageMsg,
            image_topic,
            self.video_stream_callback,
            10)
        if depth_topic is not None:
            self.subscription_depth = self.create_subscription(
                ImageMsg,
                depth_topic,
                self.depth_stream_callback,
                10)
        if camera_info_topic is not None:
            self.subscription_camera_info = self.create_subscription(
                CameraInfo,
                camera_info_topic,
                self.camera_info_callback,
                10)
        self.annotated_image_publisher = self.create_publisher(ImageMsg, 'annotated_images', 10)
        self.object_positions_publisher = self.create_publisher(PoseArray, 'object_positions', 10)
        self.marker_publisher = self.create_publisher(MarkerArray, 'visualization_marker_array', 10)

        self.bridge = CvBridge()
        self.last_depth_image = np.ones((1,1,1))
        self.last_image = np.zeros((1,1,3))

        # assume ideal pinhole camera model
        self.camera_info = None


    def video_stream_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"Could not convert ROS Image to OpenCV: {e}")
            return
        image_source = np.asarray(cv_image)
        self.last_image = image_source
        self.publish_result()


    def depth_stream_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        except CvBridgeError as e:
            self.get_logger().error(f"Could not convert ROS Image to OpenCV: {e}")
            return
        depth_image = np.asarray(cv_image)
        self.last_depth_image = depth_image


    def publish_offset_vectors(self, boxes, phrases, depths):
        pose_array = PoseArray()
        pose_array.header.frame_id = self.frame_id
        marker_array = MarkerArray()

        for i, (box, phrase, depth) in enumerate(zip(boxes, phrases, depths)):
            # Assuming box format is [center_x, center_y, width, height]
            pose = Pose()
            center_x = box[0] + 0.5 * box[2]
            center_y = box[1] + 0.5 * box[3]
            pose.position.x, pose.position.y, pose.position.z = calculate_3d_position(center_x, center_y, depth, self.camera_info)
            pose_array.poses.append(pose)

            # Create a marker for each object
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.type = Marker.TEXT_VIEW_FACING
            marker.pose = pose
            marker.text = phrase
            marker.scale.z = 0.1  # Text size
            marker.color.a = 1.0  # Alpha
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.id = np.random.randint(0, 1000000)
            marker.lifetime = rclpy.time.Duration(seconds=2).to_msg()
            marker_array.markers.append(marker)

        self.object_positions_publisher.publish(pose_array)
        self.marker_publisher.publish(marker_array)


    @torch.no_grad()
    def publish_result(self):
        TEXT_PROMPT = "cat"
        BOX_TRESHOLD = 0.3
        TEXT_TRESHOLD = 0.25

        image_source = self.last_image
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image, _ = transform(Image.fromarray(image_source), None)

        boxes, logits, phrases = predict(
            model=self.groundingdino_model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        # annotated_frame = annotated_frame[...,::-1] # BGR to RGB
        
        def show_mask(mask, image, random_color=True):
            if random_color:
                color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
            else:
                color = np.array([30/255, 144/255, 255/255, 0.6])
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            
            annotated_frame_pil = Image.fromarray(image).convert("RGBA")
            mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

            return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))


        self.sam_predictor.set_image(image_source)
        H, W, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)
        try:
            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes,
                multimask_output = False,
            )
        except Exception as e:
            self.get_logger().error(f"Error while segmenting: {e}")
            return
        masks = masks.cpu()
        annotated_frame_with_mask = annotated_frame
        for mask in masks:
            annotated_frame_with_mask = show_mask(mask[0], annotated_frame_with_mask)

        # To publish the annotated frame, convert it back to a ROS Image and publish
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame_with_mask, encoding="rgba8")
        self.annotated_image_publisher.publish(annotated_msg)

        # calculate depth
        depth_image = self.last_depth_image
        depth_image_scaled = torch.nn.functional.interpolate(torch.from_numpy(depth_image).unsqueeze(0), size=(H, W), mode='bicubic', align_corners=False)[0,0,:,:]
        depths = [depth_image_scaled[mask[0]].mean().item() for mask in masks]

        # publish the offset vectors
        self.publish_offset_vectors(boxes, phrases, depths)


    def camera_info_callback(self, msg):
        self.camera_info = msg


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_topic', type=str, required=True, help='ros2 topic name for the image stream')
    parser.add_argument('-d', '--depth_topic', type=str, required=False, help='ros2 topic name for the depth stream', default=None)
    parser.add_argument('-c', '--camera_info_topic', type=str, required=False, help='ros2 topic name for the camera info', default=None)
    parser.add_argument('-f', '--frame_id', type=str, required=False, help='frame id for the camera', default="camera_link")
    cmdline_args = parser.parse_args()

    rclpy.init(args=args)
    node = ObjectDetectionAndLocalizationNode(cmdline_args.image_topic, cmdline_args.depth_topic, cmdline_args.camera_info_topic, cmdline_args.frame_id)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()
