# Robust Multi-tab Website Fingerprinting Attacks in the Wild

<p align="center">
<img src=".\figs\ARES.png" height = "180" alt="" align=center />
<br><br>
</p>


This repository contains the source code and datasets for our paper "Robust Multi-tab Website Fingerprinting Attacks in the Wild" (Published in IEEE S&P 2023).


If you want to cite the repo, you can use our [paper](https://www.computer.org/csdl/proceedings-article/sp/2023/933600b005/1NrbYpaG652).
```bibtex
@INPROCEEDINGS {multitab-wf-datasets,
author = {X. Deng and Q. Yin and Z. Liu and X. Zhao and Q. Li and M. Xu and K. Xu and J. Wu},
booktitle = {2023 IEEE Symposium on Security and Privacy (SP)},
title = {Robust Multi-tab Website Fingerprinting Attacks in the Wild},
year = {2023},
}
```

**[News]** We release a website fingerprint attack library ([link](https://github.com/Xinhao-Deng/Website-Fingerprinting-Library)) that includes implementations of 11 advanced DL-based WF attacks.


## Prerequisites

We prototype attacks using Pytorch 2.0.1 and Python 3.8. For convenience, we recommend running the following command.

```
conda create --name <env> --file requirements.txt
```


## Datasets

We collect our Tor browsing datasets under the real multi-tab scenario. You can download the dataset via the [link](https://drive.google.com/file/d/1akeBzeGLfnzgmD0Qt196WshwgbsYMnnS/view?usp=sharing).

You can load the dataset using numpy.

```python
import numpy as np

inpath = "example.npz"
data = np.load(inpath)
dir_array = data["direction"]  # Sequence of packet direction
time_array = data["time"] # Sequence of packet timestamps
label = data["label"]  # labels
```

Note that we have recently improved the quality of our dataset. We built a new image classification model based on ResNet, which effectively filters out failed website traffic using screenshots. We will report the latest experimental results in the extended journal version.

## Usage

### Prepare Data

- Download datasets and place it in the folder `./datasets`

- Divide the dataset into training, validation, and test sets. 
For example, for the 2-tab dataset collected in the closed-world, you can execute the following command.

```
python scripts/dataset_split.py -i datasets/closed_2tab.npz -o datasets/processed/closed_2tab
```

### Training

We take the training of ARES on a 2-tab dataset in the closed-world as an example.

```sh
python train.py -d closed-2tab -g 0 -l ARES
```

Training separate models for each website is costly. We use [torch.nn.MultiLabelSoftMarginLoss](https://pytorch.org/docs/stable/generated/torch.nn.MultiLabelSoftMarginLoss.html) to achieve a similar effect. This loss function creates a criterion that optimizes a multi-label one-versus-all loss based on max-entropy.

Specifically, you can use [TensorBoard](https://pytorch.org/docs/stable/tensorboard.html) to visualize the training process.

```
tensorboard --logdir=runs
```

<p align="center">
<img src=".\figs/tensorboard.png" height = "180" alt="" align=center />
<br><br>
</p>

Note that, benefiting from the Transformer architecture, ARES's performance gradually improves with an increase in epochs, even experiencing slight improvements beyond 500 epochs.


### Evaluation

We take the evaluation of ARES on a 2-tab dataset in the closed-world as an example.

```sh
python eval.py -d closed_2tab -g 0 -m ARES
```

> You can directly download the trained ARES parameter file (with the random seed set to 1018) [link](https://drive.google.com/drive/folders/1wKMlky_G-x_1IxJg6YgnKt3GFFyTOPrK?usp=sharing).


## Contact

If you have any questions or suggestions, feel free to contact:

- [Xinhao Deng](https://xinhao-deng.github.io/) (dengxh23@mails.tsinghua.edu.cn)

