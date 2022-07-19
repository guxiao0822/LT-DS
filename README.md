# LT-DS
Repo for our paper "Tackling Long-Tailed Category Distribution Under Domain Shifts" (ECCV 2022)

[[project page](https://xiaogu.site/LTDS)] [[dataset](./dataset/)]

<p align="center"><img src="misc/figure1.jpg" alt="figure1" width=90%/></p>


## Dataset 
We provided two datasets for benchmarking LT-DS problems. 
Due to the license issue, we only provided instructions on how to create the corresponding datasets. 
Please follow [here](./dataset/).


## Training 
### AWA2-LTS
```commandline
python train/trainer.py --cfg config/exp/awa2.yaml 
```

### ImageNet-LTS
```commandline
python train/trainer.py --cfg config/exp/imagenet.yaml
```

## Evaluation
We provide the code for evaluation, calculating metrics including Acc, Acc-U, H. e.g., 
```commandline
python train/trainer.py --cfg config/exp/awa2.yaml MODE test
```

## TODO
- [ ] Add requirements
- [ ] Add PACS-ODG experiments
- [ ] Add Imbalanced Baselines


## Citation
If you find our paper useful, please consider citing:
```
@inproceedings{gu2022tackling,
  title={Tackling Long-Tailed Category Distribution Under Domain Shifts},
  author={Gu, Xiao and Guo, Yao and Li, Zeju and Qiu, Jianing and Dou, Qi and Liu, Yuxuan and Lo, Benny and Yang, Guang-Zhong},
  booktitle={ECCV},
  year={2022}
```

## Acknowledgement
Our codes are inspired from [OpenDG-DAML](https://github.com/thuml/OpenDG-DAML).