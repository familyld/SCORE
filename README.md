# SCORE

A PyTorch implementation for our paper "False Correlation Reduction for Offline Reinforcement Learning" published on IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI). Our code is built off of [TD3-BC](https://github.com/sfujim/TD3_BC) and [rlkit](https://github.com/rail-berkeley/rlkit).

Link to our paper:
- IEEE TPAMI: [https://ieeexplore.ieee.org/document/9518443](https://ieeexplore.ieee.org/document/9518443)
- arXiv: [https://arxiv.org/abs/2110.12468](https://arxiv.org/abs/2110.12468)

## Prerequisites

- PyTorch 1.4.0 with Python 3.7 
- MuJoCo 2.00 with mujoco-py 2.0.2.13
- [d4rl](https://github.com/rail-berkeley/d4rl) 1.1 or higher (with v2 datasets)
- [rlkit](https://github.com/rail-berkeley/rlkit) 0.2.1

## Usage

For training SCORE on `Envname` (e.g. `walker2d-medium-v2`), run:

```
python main.py --env_name=Envname --version=VersionName --gpu=0 
```

The results are collected in `./output/Envname/SCORE(VersionName)/`, where

- `debug.log` records the log data,
- `params.pkl` records the final model parameters,
- `progress.csv` records the log data in the `.csv` format for analysis purpose,
- `variant.json` records the hyperparameters.

# Bibtex

```
@article{deng2023score,
  author={Deng, Zhihong and Fu, Zuyue and Wang, Lingxiao and Yang, Zhuoran and Bai, Chenjia and Zhou, Tianyi and Wang, Zhaoran and Jiang, Jing},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={False Correlation Reduction for Offline Reinforcement Learning}, 
  year={2023},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TPAMI.2023.3328397}}
```
