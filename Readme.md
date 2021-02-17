# Weakly Supervised Learning of Rigid 3D Scene Flow 
This repository provides code and data to train and evaluate a weakly supervised method for rigid 3D scene flow estimation. It represents the official implementation of the paper:

### [Weakly Supervised Learning of Rigid 3D Scene Flow](add_arxiv_link)
[Zan Gojcic](https://zgojcic.github.io/), [Or Litany](https://orlitany.github.io/), [Andreas Wieser](https://baug.ethz.ch/departement/personen/mitarbeiter/personen-detail.MTg3NzU5.TGlzdC82NzksLTU1NTc1NDEwMQ==.html), [Leonidas J. Guibas](https://geometry.stanford.edu/member/guibas/), [Tolga Birdal](http://tbirdal.me/)\
| [IGP ETH Zurich](https://igp.ethz.ch/) | [Nvidia Toronto AI Lab](https://nv-tlabs.github.io/) | [Guibas Lab Stanford University](https://geometry.stanford.edu/index.html) |

For more information, please see the [project webpage](https://3dsceneflow.github.io/)

![WSR3DSF](assets/network_architecture.jpg?raw=true)


### Environment Setup

> Note: the code in this repo has been tested on Ubuntu 16.04/20.04 with Python 3.7, CUDA 10.1/10.2, PyTorch 1.7.1 and MinkowskiEngine 0.5.1. It may work for other setups, but has not been tested.


Before proceding, make sure CUDA is installed and set up correctly. 

After cloning this reposiory you can proceed by setting up and activating a virual environment with Python 3.7. If you are using a different version of cuda (10.1) change the pytorch installation instruction accordingly.

```bash
export CXX=g++-7
conda config --append channels conda-forge
conda create --name rigid_3dsf python=3.7
source activate rigid_3dsf
conda install --file requirements.txt
conda install -c open3d-admin open3d=0.9.0.0
conda install -c intel scikit-learn
conda install pytorch==1.7.1 torchvision cudatoolkit=10.1 -c pytorch
```
You can then proceed and install [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) library for sparse tensors:

```bash
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps
```
Our repository also includes a pytorch implementation of [Chamfer Distance](https://github.com/chrdiller/pyTorchChamferDistance) in `./utils/chamfer_distance` which will be compiled on the first run. 

In order to test if Pytorch and MinkwoskiEngine are installed correctly please run
```bash
python -c "import torch, MinkowskiEngine"
```
which should run without an error message.

### Data

We provide the preprocessed data of *flying_things_3d* (108GB), *stereo_kitti* (500MB), *lidar_kitti* (~160MB), *semantic_kitti* (78GB), and *waymo_open* (50GB) used for training and evaluating our model.

To download a single dataset please run:

```bash
bash ./scripts/download_data.sh name_of_the_dataset
```

To download all datasets simply run:

```bash
bash ./scripts/download_data.sh
```
The data will be downloaded and extracted to `./data/name_of_the_dataset/`.

### Pretrained models

We provide the checkpoints of the models trained on *flying_things_3d* or *semantic_kitti*, which we use in our main evaluations.

To download these models please run:

```bash
bash ./scripts/download_pretrained_models.sh
```

Additionally, we provide all the models used in the ablation studies and the model fine tuned on *waymo_open*.

To download these models please run:

```bash
bash ./scripts/download_pretrained_models_ablations.sh
```

All the models will be downloaded and extracted to `./logs/dataset_used_for_training/`.

### Evaluation with pretrained models

Our method with pretrained weights can be evaluated using the `./eval.py` script. The configuration parameters of the evaluation can be set with the `*.yaml` configuration files located in `./configs/eval/`. We provide a configuration file for each dataset used in our paper. For all evaluations please first download the pretrained weights and the corresponding data. Note, if the data or pretrained models are saved to a non-default path the config files also has to be adapted accordingly.

#### *FlyingThings3D*

To evaluate our backbone + scene flow head on *FlyingThings3d* please run:

```shell
python eval.py ./configs/eval/eval_flying_things_3d.yaml
```
This should recreate the results from the Table 1 of our paper (EPE3D: 0.052 m).

#### *stereoKITTI*

To evaluate our backbone + scene flow head on *stereoKITTI* please run:

```shell
python eval.py ./configs/eval/eval_stereo_kitti.yaml
```
This should again recreate the results from the Table 1 of our paper (EPE3D: 0.042 m).

#### *lidarKITTI*

To evaluate our full weakly supervised method on *lidarKITTI* please run:

```shell
python eval.py ./configs/eval/eval_lidar_kitti.yaml
```
This should recreate the results for Ours++ on *lidarKITTI* (w/o ground) from the Table 2 of our paper (EPE3D: 0.094 m). To recreate other results on *lidarKITTI* please change the `./configs/eval/eval_lidar_kitti.yaml` file accordingly.


#### *semanticKITTI*

To evaluate our full weakly supervised method on *semanticKITTI* please run:

```shell
python eval.py ./configs/eval/eval_semantic_kitti.yaml
```
This should recreate the results of our full model on *semanticKITTI* (w/o ground) from the Table 4 of our paper. To recreate other results on *semanticKITTI* please change the `./configs/eval/eval_semantic_kitti.yaml` file accordingly.

#### *waymo open*

To evaluate our fine-tuned model on *waymo open* please run:

```shell
python eval.py ./configs/eval/eval_waymo_open.yaml
```
This should recreate the results for Ours++ (fine-tuned) from the Table 9 of the appendix. To recreate other results on *waymo open* please change the `./configs/eval/eval_waymo_open.yaml` file accordingly.


### Training our method from scratch

Our method can be trained using the `./train.py` script. The configuration parameters of the training process can be set using the config files located in `./configs/train/`.

#### Training our backbone with full supervision on *FlyingThings3D*

To train our backbone network and scene flow head under full supervision (corresponds to Sec. 4.3 of our paper) please run: 

```shell
python train.py ./configs/train/train_fully_supervised.yaml
```

The checkpoints and tensorboard data will be saved to `./logs/logs_FlyingThings3D_ME`. If you run out of GPU memory with the default setting please adapt the `batch_size` and `acc_iter_size` in the `./configs/default.yaml` to e.g. 4 and 2, respectively.

#### Training under weak supervision on *semanticKITTI*

To train our full method under weak supervision on *semanticKITTI* please run

```shell
python train.py ./configs/train/train_weakly_supervised.yaml
```

The checkpoints and tensorboard data will be saved to `./logs/logs_SemanticKITTI_ME`. If you run out of GPU memory with the default setting please adapt the `batch_size` and `acc_iter_size` in the `./configs/default.yaml` to e.g. 4 and 2, respectively.

### Citation

If you found this code or paper useful, please consider citing:

```shell
@inproceedings{gojcic2021wslr3Dsf,
	title={Weakly {S}upervised {L}earning of {R}igid {3D} {S}cene {F}low},
	author={Gojcic, Zan and Litany, Or and Wieser, Andreas and Guibas, Leonidas J and Birdal, Tolga},
	booktitle={add arxiv info},
	year={2021}
}
```
### Contact
If you run into any problems or have questions, please create an issue or contact [Zan Gojcic](zgojcic@ethz.ch).


### Acknowledgments
In this project we use parts of the official implementations of: 

- [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)
- [MultiviewReg](https://github.com/zgojcic/3D_multiview_reg)
- [RPMNet](https://github.com/yewzijian/RPMNet)
- [FLOT](https://github.com/valeoai/FLOT)

 We thank the respective authors for open sourcing their methods.