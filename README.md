# Learning Causal Alignment For Reliable Disease Diagnosis (ICLR 2025)

## Summary

#### If you happen to use or modify this code, please remember to cite our paper:

    @inproceedings{
    liu2025learning,
    title={Learning Causal Alignment for Reliable Disease Diagnosis},
    author={Mingzhou Liu and Ching-Wen Lee and Xinwei Sun and Xueqing Yu and Yu QIAO and Yizhou Wang},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=ozZG5FXuTV}
    }
## Table with Results 
| Methodology          | Precision of CAM(LIDC) | Precision of CAM(DDSM) | Classification Accuracy(LIDC) | Classification Accuracy(DDSM) |
| -------------------- | ---------------------- | ---------------------- | ----------------------------- | ----------------------------- |
| Ross et al. (2017)   | 0.034 (0.06)           | 0.084 (0.11)           | 0.656 (0.00)                  | 0.559 (0.05)                  |
| Zhang et al. (2018)  | 0.068 (0.11)           | 0.110 (0.13)           | 0.381 (0.03)                  | 0.581 (0.00)                  |
| Brendel & Bethge     | 0.048 (0.04)           | 0.090 (0.04)           | 0.358 (0.00)                  | 0.592 (0.00)                  |
| Rieger et al. (2020) | 0.041 (0.05)           | **0.232 (0.17)**       | 0.343 (0.00)                  | 0.586 (0.01)                  |
| Chang et al. (2021)  | **0.074 (0.03)**       | 0.119 (0.07)           | 0.503 (0.08)                  | 0.496 (0.08)                  |
| Oracle classifier    | 1.000 (0.00)           | 1.000 (0.00)           | 0.789 (0.00)                  | 0.726 (0.01)                  |
| Ours                 | **0.751 (0.03)**       | **0.805 (0.06)**       | **0.722 (0.00)**              | **0.656 (0.00)**              |


## Installation

To install this repo, first create your virtual environment in Python3.8, for example this way:

```
python3.8 -m venv my_venv
```

And then install the requirements as follows:

```
pip install -r requirements.txt
```

**Notes:**

Please make sure your Torchopt version==0.7.0 

Please make sure your Python version is consistent with the Pytorch, Torch Geometric, and CUDA versions you are going to install.

## Running the experiments

`sh .\DDSM\run.sh`

`sh .\LIDC\run.sh`

### Data Preprocess

##### Data Format in cache_path

- **Image (`\*_img.pt`)**
  - Shape: `(batch_size, m, n)`
  - Stores the mass images.
- **Latent Code (`\*_z.pt`)**
  - Shape: `(batch_size, channel, w, h)`
  - Stores the image's latent representation.
- **Node ID (`\*_nid.pt`)** or **Jit ID (`\*_jitid.pt`)**
  - Shape: `(batch_size,)`
  - Stores the mass node IDs.
- **Labels (`\*_label.pt`)**
  - Shape: `(batch_size, 6)`
  - Stores the mass attributes labels.
- **Mask (`\*_mask.pt`)**
  - Shape: `(batch_size, m, n)`
  - Stores the binary mask for each image (1 for mass region, 0 for background).

##### ccce_file

Stores mass attributes ccce score csv(cite from **Conditional Counterfactual Causal Effect for Individual Attribution**) 

Code for CCCE ```https://github.com/LLily0703/CCCE```

##### DDSM Data Preprocess code

- Raw data preprocess

  `python DDSM\preprocess\test_db.py`(cite from https://github.com/jbustospelegri/breast_cancer_diagnosis)

  `python DDSM\preprocess\jit.py`

  `python DDSM\add_crossmask_manually.py`

- auto encoder get latent code

  get DDSM image latent code by auto encoder

##### LIDC Data Preprocess code

- Raw data preprocess

  follow README.md in preprocess directory

- auto encoder get latent code

  get DDSM image latent code by auto encoder

