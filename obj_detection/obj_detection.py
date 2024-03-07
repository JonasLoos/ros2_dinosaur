# General imports
import argparse
import os
import copy
import numpy as np
import torch
from torch.utils.cpp_extension import CUDA_HOME
from PIL import Image, ImageDraw, ImageFont
import io
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2
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
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, load_image, predict
from huggingface_hub import hf_hub_download
import supervision as sv

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np


# depth estimation
from transformers import pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print("Using CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))


def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model


class ObjectDetectionAndLocalizationNode(Node):
    def __init__(self):
        super().__init__('object_detection_and_localization')

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
            Image,
            'video_stream',
            self.video_stream_callback,
            10)
        self.subscription_depth = self.create_subscription(
            Image,
            'depth_stream',
            self.depth_stream_callback,
            10)
        self.publisher_ = self.create_publisher(String, 'detected_objects_positions', 10)
        self.annotated_image_publisher = self.create_publisher(Image, 'annotated_images', 10)

        # self.bridge = CvBridge()

        # Initialize your Segment Anything and Grounding DINO instances here

        def video_stream_callback(self, msg):
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except CvBridgeError as e:
                self.get_logger().error(f"Could not convert ROS Image to OpenCV: {e}")
                return

            TEXT_PROMPT = "dog"
            BOX_TRESHOLD = 0.3
            TEXT_TRESHOLD = 0.25
            
            _, image = load_image(cv_image)  # usually the argument should be the filename

            boxes, logits, phrases = predict(
                model=self.groundingdino_model, 
                image=image, 
                caption=TEXT_PROMPT, 
                box_threshold=BOX_TRESHOLD, 
                text_threshold=TEXT_TRESHOLD
            )
            
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
            
            H, W, _ = cv_image.shape
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

            # To publish the annotated frame, convert it back to a ROS Image and publish
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame_with_mask, encoding="rgba8")
            self.annotated_image_publisher.publish(annotated_msg)


    def depth_stream_callback(self, msg):
        # Similar to video_stream_callback, but process depth stream here for position estimation
        
        # For now, we'll skip the depth processing details.
        # You'd typically use cv_bridge to convert the depth image and use it for localization
        detected_objects_positions = [(42,42,42)]

        # After processing, publish positions
        positions_msg = String()
        positions_msg.data = str(detected_objects_positions)  # Convert your positions list to string
        self.publisher_.publish(positions_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionAndLocalizationNode()
    rclpy.spin(node)
    rclpy.shutdown()
