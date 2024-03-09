# ROS2 DINOSAUR

> [!WARNING]
> This package is still under early development and not yet ready for use.

DINOSAUR stands for: gounding **DINO** with **S**egment **A**nything for **U**niversal object **R**ecognition.

The goal of this package is to provide a ROS2 node for object detection and position estimation using a camera feed and depth information.

It uses [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) for object detection, and [Segment Anything](https://segment-anything.com/) for segmentation to estimate the position of the detected objects by using given depth information.


## Installation

Assuming Python `3.10`.


### Install requirements

You need cuda toolkit 11.7 (11.8 should also work).

You can check your version of cuda toolkit with:

```bash
nvcc --version
```

If you don't have cuda toolkit 11.8, you can install it by following the instructions from https://developer.nvidia.com/cuda-11-8-0-download-archive. The network installation is probably simpler. Instead of installing `cuda` (last line) it might be better to just install `cuda-toolkit-11-7` or `cuda-toolkit-11-8`.

You can find information about your system with:

```bash
lsb_release -a && uname -io
```

Then install torch:

```bash
pip install torch==1.13.0+cu117 torchvision==0.14+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

Then install the rest of the requirements:

```bash
pip install \
    transformers \
    accelerate \
    scipy \
    safetensors \
    cv-bridge \
    git+https://github.com/facebookresearch/segment-anything.git \
    git+https://github.com/IDEA-Research/GroundingDINO.git
```


### Build using colcon

```bash
colcon build --symlink-install --packages-select obj_detection
```


### Source

```bash
source install/setup.bash
```


## Usage

Example usage:

```bash
ros2 run ros2_dinosaur dinosaur -i '/video_stream' -d '/depth_stream' -f 'camera_id'
```

Note that the node will download the model weights from the internet (several GB) the first time it is run. The weights will be saved in `~/.ros/obj_detection/` (sam) and the default huggingface cache directory (groundingdino, usually `~/.cache/huggingface/hub/`).

Arguments:

* `-q` or `--query`: query to search for in the image (required)
* `-i` or `--image_topic`: ros2 topic name for the image stream (required)
* `-d` or `--depth_topic`: ros2 topic name for the depth stream
* `-c` or `--camera_info_topic`: ros2 topic name for the camera info
* `-f` or `--frame_id`: frame id for the camera (default: camera_link)
* `-u` or `--update_on`: update on receiving `image`, `depth` or `both` (default: `both`, choices: `image`, `depth`, `both`)


## Common errors

Duing installation:

* error when installing groundindino: cuda-toolkit version might not fit pytorch version. Make sure you have the correct versions installed

During running:

* Groundingdino using cpu: cuda-toolkit probably wasn't correctly installed when installing groundindino. Try to reinstall.
* CUDA_HOME is not set: cuda-toolkit might not be installed.

