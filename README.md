# LT-DS
[**ECCV 2022**] Repo for our paper "**Tackling Long-Tailed Category Distribution Under Domain Shifts**" 

[[project](https://xiaogu.site/LTDS)] [[dataset](./dataset/)] [[paper](http://arxiv.org/abs/2207.10150)]

<p align="center"><img src="misc/eccv2022_gif.gif" alt="figure1" width=100%/></p>

## Abstract
Machine learning models fail to perform well on real-world applications when 
1) <span style="color:darkred">**LT**</span>: the category distribution P(Y) of the training dataset suffers from long-tailed distribution; 
2) <span style="color:darkgreen">**DS**</span>: the test data is drawn from different conditional distributions P(X|Y). 

Existing approaches cannot handle the scenario where both issues exist, which however is common for real-world applications. In this study, we took a step forward and looked into the problem of long-tailed classification under domain shifts. By taking both the categorical distribution bias and conditional distribution shifts into account, we designed three novel core functional blocks including Distribution Calibrated Classification Loss, Visual-Semantic Mapping and Semantic-Similarity Guided Augmentation. Furthermore, we adopted a meta-learning framework which integrates the three blocks to improve domain generalization on unseen target domains.

## Dataset 
We provide two datasets for benchmarking <span style="color:darkred">**LT**</span>-<span style="color:darkgreen">**DS**</span> (Long-Tailed Under Domain shifts) algorithms. 
Due to the license issue, we only provided instructions on how to create the corresponding datasets. 
Please follow [here](./dataset/).

## Install Env 
```commandline
conda env create -f requirement.yml
```

## Training 
### AWA2-LTS
```commandline
python train/trainer.py --cfg config/exp/awa2.yaml 
```

### ImageNet-LTS
```commandline
python train/trainer.py --cfg config/exp/imagenet.yaml
```


## TODO
- [x] Add requirements
- [x] Add evaluation scripts
- [ ] Add imbalanced Bbselines


## Citation
If you find our paper/code useful, please consider citing:
```bibtex
@inproceedings{gu2022tackling,
  title={Tackling Long-Tailed Category Distribution Under Domain Shifts},
  author={Gu, Xiao and Guo, Yao and Li, Zeju and Qiu, Jianing and Dou, Qi and Liu, Yuxuan and Lo, Benny and Yang, Guang-Zhong},
  booktitle={ECCV},
  year={2022}
  }
```

## Acknowledgement
Our codes are inspired by the following repos:
[[OpenDG-DAML](https://github.com/thuml/OpenDG-DAML)] [[BagofTricks-LT
](https://github.com/zhangyongshun/BagofTricks-LT)] [[ISDA](https://github.com/blackfeather-wang/ISDA-for-Deep-Networks)].