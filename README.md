# pVoxel

![Licence](https://img.shields.io/github/license/fuchuanpu/pVoxel)
![Last](https://img.shields.io/github/last-commit/fuchuanpu/pVoxel)
![Language](https://img.shields.io/github/languages/count/fuchuanpu/pVoxel)

_Point cloud analysis_ based false postive (FP) identification for machine learning based malicious traffic detection systems.

___Point Cloud Analysis for ML-Based Malicious Traffic Detection:
Reducing Majorities of False Positive Alarms___  
In Proceedings of the 2023 ACM SIGSAC Conference on
Computer and Communications Security ([CCS'23](https://www.sigsac.org/ccs/CCS2023/)).  
[Chuanpu Fu](https://www.fuchuanpu.cn), [Qi Li](https://sites.google.com/site/qili2012), [Ke Xu](http://www.thucsnet.org/xuke.html) and [Jianping Wu](https://www.cs.tsinghua.edu.cn/info/1126/3582.htm). 

This repository provides a simplified demo for the paper, which is easy to reproduce. 

> Please find proofs in the [full version paper](./CCS23_pVoxel_longVersion.pdf). 

## __0x00__ Environment
AWS EC2 c4.4xlarge, 100GB SSD, canonical `Ubuntu` 22.04 LTS (amd64, 3/3/2023).


## __0x01__ Software
`start.sh` is an all-in-one script to build and run this demo:

```bash
git clone https://github.com/fuchuanpu/pVoxel.git
cd pVoxel
chmod +x start.sh && ./start.sh
```

## __0x02__ Reference
``` bibtex
@inproceedings{CCS23-pVoxel,
  author    = {Chuanpu Fu and
               others},
  title     = {Point Cloud Analysis for ML-Based Malicious Traffic Detection: Reducing Majorities of False Positive Alarms},
  booktitle = {CCS},
  publisher = {ACM},
  year      = {2023}
}
```

## __0x02__ Maintainer
[Chuanpu Fu](fcp20@tsinghua.edu.cn)
