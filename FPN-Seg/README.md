# Semantic Segmentation using FPN
from:
[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)

This repository is a semantic segmentation part implement of [Kaiming He, Panoptic Feature Pyramid Networks
](https://arxiv.org/abs/1901.02446).

##dataset
Train on Cityscapes Dataset



## Training

### Prepare data
- for Cityscapes dataset, make directory "Cityscapes" and put "gtFine" in "Cityscapes/gtFine_trainvaltest" folder, put "test", "train", "val" in "Cityscapes/leftImg8bit" foloder.

### Train the network

train with Cityscapes(default) dataset:
change to your own CityScapes dataset path in mypath.py, then run:

```
python train_val.py --dataset Cityscapes --save_dir /path/to/run
```

## Test
Test with Cityscapes dataset(val), run:
```
python test.py --dataset Cityscapes --exp_dir /path/to/experiment_x
```
If you want to plot the color semantic segmentation prediction of the test input color image, please set --plot=True, for example:
```
python test.py --dataset Cityscapes --exp_dir /path/to/experiment_x --plot True
```
## Inference
if you want to inference the pciture you want,put the picture in the project path(the same directory with train_val.py),run:
```
sh text.sh
```
input:
[!image]https://github.com/Andy-zhujunwen/FPN-Semantic-segmentation/blob/master/FPN-Seg/s1.jpeg
output:
[!image]https://github.com/Andy-zhujunwen/FPN-Semantic-segmentation/blob/master/FPN-Seg/testjpg.jpeg

## Acknowledgment
[FCN-pytorch](https://github.com/pochih/FCN-pytorch)

[pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)

[pytorch-fpn](https://github.com/kuangliu/pytorch-fpn)

[fpn.pytorch](https://github.com/jwyang/fpn.pytorch)
