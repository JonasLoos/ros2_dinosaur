


## Installation

Assuming Python `3.10`.

### Install requirements:

```bash
# conda install -c conda-forge cudatoolkit-dev -y  # probably conda isn't the way
sudo apt-get install nvidia-cuda-toolkit
pip install torch==1.13.0+cu117 torchvision==0.14+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

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

### Run

Note that the script will download the model weights from the internet (several GB) the first time it is run.

```bash
ros2 run obj_detection obj_detection
```
