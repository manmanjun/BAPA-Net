# A Pytorch Implementation of BAPA-Net: Boundary Adaptation and Prototype Alignment for Cross-domain Semantic Segmentation (ICCV 2021 oral)

## Requirements
`pip3 install -r requirements.txt`


## Usage

- Download datasets

- Example of testing a model with domain adaptation with CityScapes as target domain(***class_num=16***)
    `python3 evaluateUDA.py --model-path *checkpoint.pth* --class-num 16`

## Checkpoints

We provide the checkpoints at [Google Drive](https://drive.google.com/drive/folders/1d6guGc5gw6jrkxNJ25NPzx2fgH4LK4Ox?usp=sharing).

## Citation
```
@InProceedings{Liu_2021_ICCV,
author = {Liu, Yahao and Deng, Jihong and Gao, Xinchen and Li, Wen and Duan, Lixin},
title = {BAPA-Net: Boundary Adaptation and Prototype Alignment for Cross-domain Semantic Segmentation},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
year = {2021}
}
```
## Acknowledgements
Some codes are adapted from [DACS](https://github.com/vikolss/DACS) and [UDADT](https://github.com/SHI-Labs/Unsupervised-Domain-Adaptation-with-Differential-Treatment). We thank them for their excellent projects.

## Contact
- lyhaolive@gmail.com