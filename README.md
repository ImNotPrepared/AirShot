# AirShot
## Abstract
Few-shot object detection has drawn increasing attention in the field of robotic exploration, where robots are required to find unseen objects with a few online provided examples. Despite recent efforts have been made to yield online processing capabilities, slow inference speeds of low-powered robots fail to meet the demands of real-time detection-making them impractical for autonomous exploration. Existing methods still face performance and efficiency challenges, mainly due to unreliable features and exhaustive class loops. In this work, we propose a new paradigm AirShot, and discover that, by fully exploiting the valuable correlation map, AirShot can result in a more robust and faster few-shot object detection system, which is more applicable to robotics community. The core module Top Prediction Filter (TPF) can operate on multi-scale correlation maps in both the training and inference stages. During training, TPF supervises the generation of a more representative correlation map, while during inference, it reduces looping iterations by selecting top-ranked classes, thus cutting down on computational costs with better performance. Surprisingly, this dual functionality exhibits general effectiveness and efficiency on various off-the-shelf models. Exhaustive experiments on COCO2017, VOC2014, and SubT datasets demonstrate that TPF can significantly boost the efficacy and efficiency of most off-the-shelf models, achieving up to 36.4\% precision improvements along with 56.3\% faster inference speed. We also opensource the DARPA Subterranean (SubT) Dataset for Few-shot Object Detection.



## DARPA Subterranean (SubT) Dataset for Few-shot Object Detection
Access the data and annotations through the following link:
[Dataset](https://drive.google.com/drive/folders/1KoFmW8W2biI3FOEREe57tKVmYa6yupCW?usp=sharing)

## Pre-trained Checkpoints

Access the pre-trained checkpoints and data through the following link:
[Pre-trained Checkpoints](https://drive.google.com/file/d/1T-Vrkv42HjcL0RDfJRnbaMuLy74juBVl/view?usp=sharing)

## Usage

Detailed instructions on how to use AirShot, including command line options and example commands.

