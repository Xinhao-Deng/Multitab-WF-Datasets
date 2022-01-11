# Robust Multi-tab Website Fingerprinting Attacks in the Wild

This repository contains the source code and datasets for our paper "Robust Multi-tab Website Fingerprinting Attacks in the Wild" (Published in IEEE S&P 2023).


If you find our repository useful, please cite the corresponding paper:

```bibtex
@inproceedings{deng2023robust,
  title={Robust multi-tab website fingerprinting attacks in the wild},
  author={Deng, Xinhao and Yin, Qilei and Liu, Zhuotao and Zhao, Xiyuan and Li, Qi and Xu, Mingwei and Xu, Ke and Wu, Jianping},
  booktitle={2023 IEEE Symposium on Security and Privacy (SP)},
  pages={1005--1022},
  year={2023},
  organization={IEEE}
}
```

## Extension

We improved the design of ARES and enhanced the dataset quality after the paper was published. The extended version of the paper is available at the following [link](https://arxiv.org/pdf/2501.12622).


We summarize the key improvements in the extended version as follows.
- **Improved traffic feature extraction.** We capture traffic aggregation features at both the packet level and burst level, enhancing ARES's robustness against noise and temporal interference caused by multi-tab browsing and WF defenses.
- **Reduced model overhead.** We optimize the multi-label one-versus-all loss based on maximum entropy to reduce the number of Trans-WF models, thereby enhancing the practicality of ARES.
- **Improved dataset quality.** We employed a ResNet-based image classification model to filter out failed page load screenshots, which helps reduce noisy traffic in the datasets and enhances the reliability of our experiments. We re-ran all experiments using the improved datasets.
- **More comprehensive evaluation.** We conducted a more comprehensive evaluation by comparing ARES with four WF attacks: Var-CNN (PETS’19), RF (Security’23), NetCLR (CCS’23), and TMWF (CCS’23). Moreover, we performed extra experiments to further assess ARES's effectiveness.

## Implementation

We implemented the ARES prototype based on WFlib. Details on the datasets and implementation can be found at this [link](https://github.com/Xinhao-Deng/Website-Fingerprinting-Library).

## Contact
If you have any questions or suggestions, feel free to contact:

- [Xinhao Deng](https://xinhao-deng.github.io/) (dengxh23@mails.tsinghua.edu.cn)