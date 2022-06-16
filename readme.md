# CCNeRF

This repository contains the official implementation for the paper: __[Compressible-composable NeRF via Rank-residual Decomposition](https://arxiv.org/abs/2205.14870)__.

We also provide a slightly different implementation in the [torch-ngp](https://github.com/ashawkey/torch-ngp) framework, which has an interactive GUI and maybe better for experience!

### [Project Page](https://ashawkey.github.io/ccnerf/) | [Arxiv](https://arxiv.org/abs/2205.14870) | [Torch-ngp implementation](https://github.com/ashawkey/torch-ngp)

![teaser](assets/teaser.png)

### Installation

Tested on Ubuntu with Python >= 3.6 and PyTorch >= 1.8.0.

```bash
git clone https://github.com/ashawkey/CCNeRF.git
cd CCNeRF
pip install -r requirements.txt 
```

### Datasets

You can download the following datasets and put them under `./data`

* [Synthetic-NeRF](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)
* [Tanks&Temples](https://dl.fbaipublicfiles.com/nsvf/dataset/TanksAndTemple.zip)

### Quick start

To reproduce the scene in teaser, simply run: 
```bash 
bash run.sh
```

### Train & Test on a single object / scene

To generate config files for all objects:

```bash
cd configs

# modify the config template in this file.
python gen_config.py 
```

To train and test on a single object:

```bash
# train and test on lego
python train.py --config configs/lego_hybrid.txt

# test with a pretrained checkpoint
python train.py --config configs/lego_hybrid.txt --render_only 1 # choose the default ckpt
python train.py --config configs/lego_hybrid.txt --render_only 1 --ckpt path/to/ckpt # speficy ckpt path
```

By default, we test and report at all compression levels (groups), which may take some time to finish.

### Compose multiple objects / scenes

To compose multiple pretrained objects in to a scene, we can modify the composition settings (model checkpoint and transformation matrix) in `compose.py`.
We provide some composed scenes as examples too:

```python
# load model
chair = load_model('./log/chair_hybrid/chair_hybrid_5.th', 'CCNeRF')
# scale and translation
T0 = np.array([
    [0.6, 0, 0, 0.8],
    [0, 0.6, 0, 0],
    [0, 0, 0.6, 0],
    [0, 0, 0, 1],
])
# rotation
R0 = np.eye(4)
R0[:3, :3] = Rot.from_euler('zyx', [-90, 0, 0], degrees=True).as_matrix()
T0 = T0 @ R0
# compose to the scene
tensorf.compose(chair, T0, R0[:3, :3])
```

The config file is still needed to provide testing camera poses.
`--ckpt none` means we are going to compose on an empty scene, else we will compose on the hotdog scene, which is not desired for the current example.
```bash
python compose.py --config configs/hotdog_hybrid.txt --ckpt none
```

### Citation

If you find the code useful for your research, please use the following `BibTeX` entry:
```
@article{tang2022compressible,
  title={Compressible-composable NeRF via Rank-residual Decomposition},
  author={Tang, Jiaxiang and Chen, Xiaokang and Wang, Jingbo and Zeng, Gang},
  journal={arXiv preprint arXiv:2205.14870},
  year={2022}
}
```

### Acknowledgement
We would like to thank [TensoRF](https://github.com/apchenstu/TensoRF) authors for the great framework!