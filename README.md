# StegGNN

This is the repository for the paper titled [StegGNN: Learning Graphical Representation for Image Steganography](https://openreview.net/forum?id=UpCzCC4hCn)

<img src="https://github.com/AGAMPANDEYY/StegGNN/blob/main/assets/StegGNN.png">

## Abstract 

Image steganography refers to embedding secret messages within cover images while maintaining imperceptibility. Recent advances in deep learning—primarily driven by Convolutional Neural Networks (CNNs) and architectures such as inverse neural networks, autoencoders, and generative adversarial networks—have led to notable progress. However, these frameworks are primarily built on CNN architectures, which treat images as regular grids and are limited by their receptive field size and a bias toward spatial locality. In parallel, Graph Neural Networks (GNNs) have recently demonstrated strong adaptability in several computer vision tasks, achieving state-of-the-art performance with architectures such as Vision GNN (ViG). This work moves in that direction and introduces StegGNN—a novel autoencoder-based, cover-agnostic image steganography framework based on GNNs. By modeling images as graph structures, our approach leverages the representational flexibility of GNNs over the grid-based rigidity of conventional CNNs. We conduct extensive experiments on standard benchmark datasets to evaluate visual quality and imperceptibility. Our results show that our GNN-based
method performs comparably to existing CNN benchmarks. These findings suggest that GNNs provide a promising al-
ternative representation for steganographic embedding and open the field of deep learning-based steganography to further exploration of GNN-based architectures.

## Setup

### 1. Install dependencies

See `setup/environment.yml` and run:

```bash
bash setup/install.sh
```

2. Prepare datasets
Download and organize datasets:

``` bash
bash setup/download_datasets.sh
```

Ensure your data directory has:

```kotlin
data/
├── div2k/
├── coco/
└── imagenet_subset/
```

Modify paths in:
```bash 
configs/steggnn.yaml
```
3. Training
   
```bash
python train.py --config configs/steggnn.yaml
```

4. Evaluation
```bash
python evaluate.py \
  --model_path checkpoints/best_model.pth \
  --dataset div2k
```

Outputs PSNR, SSIM, LPIPS, and steganalysis AUC metrics.

----

## Experiments

<img src="https://github.com/AGAMPANDEYY/StegGNN/blob/main/assets/comparison.png" >

### Cover/Stego Image Quality (DIV2K)


<div align="center">


| Method   | PSNR (dB) | SSIM | LPIPS |
|----------|-----------|------|--------|
| HiDDeN   | 28.45     | 0.93 | 0.13   |
| Baluja   | 28.47     | 0.93 | 0.13   |
| UDH      | 34.35     | 0.94 | 0.02   |
| HiNet    | 42.89     | 0.99 | 0.00   |
| **StegGNN** | **41.65** | **0.98** | **0.00** |

</div>

### Secret Image Reconstruction Quality (DIV2K)


<div align="center">

| Method   | PSNR (dB) | SSIM | LPIPS |
|----------|-----------|------|--------|
| HiDDeN   | 27.79     | 0.87 | 0.11   |
| Baluja   | 28.25     | 0.91 | 0.13   |
| UDH      | 33.30     | 0.94 | 0.04   |
| HiNet    | 31.27     | 0.96 | 0.00   |
| **StegGNN** | **27.64** | **0.87** | **0.13** |

</div>

### Steganalysis Resistance

- **StegExpose**: AUC ≈ 0.57 (near-random)
- **SRNet**: Over 100 cover/stego pairs needed to reach >95% detection accuracy

Refer to the paper for full experimental curves.

----

## Dataset

We evaluate StegGNN on three publicly available image datasets:

- **DIV2K**: Used for both training and evaluation. Contains 800 high-resolution images in the training set. During training, we randomly crop `256×256` patches from these images with horizontal and vertical flipping for data augmentation.
- **COCO**: A subset of 1000 cover-secret image pairs is randomly sampled from the COCO dataset for testing.
- **ImageNet**: A subset of 1000 cover-secret image pairs is randomly sampled from ImageNet for evaluation.

All evaluation images are resized to `256×256` using bilinear interpolation to ensure cover and secret images have the same dimensions.

### Directory Structure

Organize datasets under the `data/` directory as follows:

```kotlin 
data/
├── div2k/
│ ├── train/
│ └── val/
├── coco/
│ └── images/
└── imagenet_subset/
└── images/
```
- You must manually download the datasets from their official websites:
  - DIV2K: [https://data.vision.ee.ethz.ch/cvl/DIV2K/](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
  - COCO: [https://cocodataset.org/](https://cocodataset.org/)
  - ImageNet: [http://www.image-net.org/](http://www.image-net.org/)

Ensure that all datasets are resized or preprocessed to 256×256 resolution before training or evaluation.

----

## Acknowledgment 

This repository builds upon several foundational works in deep image steganography and graph neural networks.

We acknowledge the following open-source projects for providing baselines, architectural inspiration, or tools:

- [HiDDeN (ECCV 2018)](https://github.com/isl-org/HiDDeN) for the CNN-based autoencoder framework.
- [HiNet (ICCV 2021)](https://github.com/BRIAREUSdotio/HiNet) for invertible neural network design and comparison baselines.
- [UDH (NeurIPS 2020)](https://github.com/chaoningzhang/Universal-Deep-Hiding) for pioneering cover-agnostic steganography with deep learning.
- [StegExpose](https://github.com/b3dk7/StegExpose) for statistical steganalysis.
- [SRNet](https://github.com/nerdslab/SRNet) for deep learning-based steganalysis.

We also thank the authors of [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/), [COCO](https://cocodataset.org/), and [ImageNet](http://www.image-net.org/) for making their datasets publicly available.

This implementation was developed for academic research purposes only.

----

## Citations

Please cite the paper if you use this code:

```bibtex
@inproceedings{steggnn2025,
  title = {StegGNN: Learning Graphical Representation for Image Steganography},
  author = {Anonymous},
  booktitle = {The IEEE/CVF International Conference on Computer Vision (ICCV)},
  year = {2025}
}

```

## Contact

For questions, please raise an issue or contact the authors via email or raise GitHub Issues.
