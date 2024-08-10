# AirShot: Efficient Few-Shot Detection for Autonomous Exploration 

#### IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS2024)
#### Zihan Wang, Bowen Li, Chen Wang, and Sebastian Scherer*

## Abstract
Few-shot object detection has drawn increasing attention in the field of robotic exploration, where robots are required to find unseen objects with a few online provided examples. Despite recent efforts have been made to yield online processing capabilities, slow inference speeds of low-powered robots fail to meet the demands of real-time detection-making them impractical for autonomous exploration. Existing methods still face performance and efficiency challenges, mainly due to unreliable features and exhaustive class loops. In this work, we propose a new paradigm AirShot, and discover that, by fully exploiting the valuable correlation map, AirShot can result in a more robust and faster few-shot object detection system, which is more applicable to robotics community. The core module Top Prediction Filter (TPF) can operate on multi-scale correlation maps in both the training and inference stages. During training, TPF supervises the generation of a more representative correlation map, while during inference, it reduces looping iterations by selecting top-ranked classes, thus cutting down on computational costs with better performance. Surprisingly, this dual functionality exhibits general effectiveness and efficiency on various off-the-shelf models. Exhaustive experiments on COCO2017, VOC2014, and SubT datasets demonstrate that TPF can significantly boost the efficacy and efficiency of most off-the-shelf models, achieving up to 36.4\% precision improvements along with 56.3\% faster inference speed. We also opensource the DARPA Subterranean (SubT) Dataset for Few-shot Object Detection.

## TODO
- [x] Release SubT Dataset
- [x] Release Pre-trained Checkpoints
- [ ] Prepare Website w/ Videos
- [ ] Release Code(Coming in May...)

## DARPA Subterranean (SubT) Dataset for Few-shot Object Detection
Access the data and annotations through the following link:
[Dataset](https://drive.google.com/drive/folders/1KoFmW8W2biI3FOEREe57tKVmYa6yupCW?usp=sharing)

## Pre-trained Checkpoints

Access the pre-trained checkpoints and data through the following link:
[Pre-trained Checkpoints](https://drive.google.com/file/d/1T-Vrkv42HjcL0RDfJRnbaMuLy74juBVl/view?usp=sharing)

## Dataset Preparation(Credit: Bowen Li)
We provide official implementation here to reproduce the results **w/o** fine-tuning of ResNet101 backbone on:
- [x] COCO-2017 validation
- [x] VOC-2012 validation dataset

### 1. Download official datasets

[MS COCO 2017](https://cocodataset.org/#home)

[PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)

[COCO format VOC annotations](https://s3.amazonaws.com/images.cocodataset.org/external/external_PASCAL_VOC.zip)

Expected dataset Structure:

```shell
coco/
  annotations/
    instances_{train,val}2017.json
    person_keypoints_{train,val}2017.json
  {train,val}2017/
```

```shell
VOC20{12}/
  annotations/
  	json files
  JPEGImages/
```

### 2. Generate supports 

Download and unzip support (COCO json files) [MEGA](https://mega.nz/file/QEETwCLJ#A8m0R7NhJ-MUNuT1fhzEgRIg6t5R69u5rAaBHTsqgUw)/[BaiduNet](https://pan.baidu.com/s/1cFtwrWAwTotwZKbXYyzjEA)(pwd:1134) in

```shell
datasets/
  coco/
    new_annotations/
```

Download and unzip support (VOC json files) [MEGA](https://mega.nz/file/BBcjjYwY#1S3Utg99D_WyfzN5qq0UfeuFrlh7Eum2jZs9U7GHhJY)/[BaiduNet](https://pan.baidu.com/s/1vPZmKKue4CAZQVzOnBUs-A)(pwd:1134) in

```shell
datasets/
  voc/
    new_annotations/
```

Run the script

```shell
cd datasets
bash generate_support_data.sh
```

You may modify 4_gen_support_pool_10_shot.py line 190, 213, and 269 with different shots (default is 1 shot).





## Usage

Detailed instructions on how to use AirShot, including command line options and example commands.

