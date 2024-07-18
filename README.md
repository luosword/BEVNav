# Learning Bird's Eye View Scene Graph and Knowledge-Inspired  Policy for Embodied Visual Navigation

![](assets/main_ps2.png)

> This repository is an official PyTorch implementation of paper:<br>
> [Learning Bird's Eye View Scene Graph and Knowledge-Inspired  Policy for Embodied Visual Navigation].<br>




## Abstract
With the assistance of scene graphs, embodied
agents in visual navigation tasks can infer the object location
based on commonsense knowledge about unknown environ-
ments. Most existing methods primarily focus on constructing
2D scene graphs using panoramic observations or encoding
semantic information in 2D representations using large language
models. However, these methods hinder to perceive 3D scene
geometry, prone to ignore the surrounding details by ambiguous
selection of observations and lack grounding interaction with
the environment, which generalizes poorly to novel objects or
unseen environments. We propose BevNav framework to solve
these issues by three parts: (i) we introduce a novel Bird’s Eye
View (BEV) scene graph that utilizes multi-view 2D information
transformed into 3D under the supervision of 3D detection to
encode scene layouts and geometric clues. (ii) we propose BEV-
BLIP contrastive learning that aligns the BEV and language
grounding inputs transferring constrain commonsense knowledge
in pre-trained models without other training in the environments.
(iii) we design BEV-based view search navigation policy, which
encourages representations that encode the semantics, relation-
ships, and positional information of objects. This policy leverages
the topological relations of locally collected BEV representations
to infer invisible objects. Utilizing BevSG, the agent can predict
a BEV graph decision score, for more accurate action prediction,
thus improving generalization ability. Extensive experiments
demonstrate that BevSG shows promising results on Gibson,
HM3D, and ProcTHOR, which exhibits higher success rates
than existing graph-based and LLM-based method, indicating the
feasibility of BevSG and commonsense knowledge from language
models leading efficient semantic exploration

## HM3D Installation
The implementation of BEV Detection is built on [MMDetection3D v0.17.1](https://github.com/open-mmlab/mmdetection3d). Please follow [BEVFormer](https://github.com/fundamentalvision/BEVFormer) for installation. 

The implementation of VN is built on the latest version of [Matterport3D simulators](https://github.com/peteanderson80/Matterport3DSimulator):
```
export PYTHONPATH=Matterport3DSimulator/build:$PYTHONPATH
```
## AI2THOR Installation
For AI2THOR environment, we follow the implements ([https://github.com/nuoxu/AKGVP])
Many thanks to the contributors for their great efforts.


## Setup
- Clone the repository and move into the top-level directory `cd AKGVP`
- Create conda environment. `conda env create -f environment.yml`
- Activate the environment. `conda activate akgvp`
- Our settings of dataset follows previous works, please refer to [HOZ](https://github.com/sx-zhang/HOZ.git) and [L-sTDE](https://github.com/sx-zhang/Layout-based-sTDE.git) for AI2THOR.
- After placing the dataset, use CLIP to generate image features. `python create_image_feat.py`
- For zero-shot navigation, lines 70-73 in `runners/a3c_train.py` can be enabled. In this way, certain categories will be filtered during the training.

## Training and Evaluation
### Train the AKGVP model with Reinforcement Learning and CLIP features
```shell
python main.py \
      --title AKGVPModel \
      --model AKGVPModel \
      --workers 4 \
      --gpu-ids 0 \
      --images-file-name clip_featuremap.hdf5
```
### Evaluate the AKGVP model
```shell
python full_eval.py \
        --title AKGVPModel \
        --model AKGVPModel \
        --results-json AKGVPModel.json \
        --gpu-ids 0 \
        --images-file-name clip_featuremap.hdf5 \
        --save-model-dir trained_models
```
### Visualization
```shell
python visualization.py
```

## Matterport3D Dataset Preparation
The dataset is based on indoor RGB images from [Matterport3D](https://niessner.github.io/Matterport/). Please fill and sign the [Terms of Use](http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf) agreement form and send it to matterport3d@googlegroups.com to request access to the dataset. 

Note that we use the undistorted_color_images for BEV Detection. Camera parameters (word-to-pixel matrix) are from undistorted_camera_parameters. The 3D box annotations can be available in mp3dbev/data. For VLN, please follow [VLN-DUET](https://github.com/cshizhe/VLN-DUET) for more details, including processed annotations, features and pretrained models of REVERIE, R2R and R4R datasets.



## Extracting Features
Please follow the [scripts](https://github.com/cshizhe/VLN-HAMT/tree/main/preprocess) to extract visual features for both undistorted_color_images (for BEV Detection) and matterport_skybox_images (for VLN, optional). Note that all the ViT features of undistorted_color_images should be used (not only the [CLS] token, about 130 GB). Please note this line since different version of [timm](https://github.com/huggingface/pytorch-image-models) models have different output: 
```
b_fts = model.forward_features(images[k: k+args.batch_size])
```

## BEV Detection
```shell
cd mp3dbev/
# multi-gpu train
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=${PORT:id} ./tools/dist_train.sh ./projects/configs/bevformer/mp3dbev.py 4

# multi-gpu test
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=${PORT:id} ./tools/dist_test.sh ./projects/configs/bevformer/mp3dbev.py ./path/to/ckpts.pth 4

# inference for BEV features
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=${PORT:id} ./tools/dist_test.sh ./projects/configs/bevformer/getbev.py ./path/to/ckpts.pth 4
```
Please also see train and inference for the detailed [usage](https://github.com/open-mmlab/mmdetection3d) of MMDetection3D.

## VN Training
```shell
cd bsg_vln
# train & infer
cd map_nav_src
bash scripts/run_bev.sh 
```



## Acknowledgement
We thank the developers of these excellent open source projects: [MMDetection3D](https://github.com/open-mmlab/mmdetection3d), [BEVFormer](https://github.com/fundamentalvision/BEVFormer/tree/master), [DUET](https://github.com/cshizhe/VLN-DUET), [HAMT](https://github.com/cshizhe/VLN-HAMT), [ETPNav](https://github.com/MarSaKi/ETPNav), [MP3D Simulator](https://github.com/peteanderson80/Matterport3DSimulator), [VLNBERT](https://github.com/YicongHong/Recurrent-VLN-BERT). Many thanks to the reviewers for their valuable comments.
and paper  ICCV 2023. ([arXiv 2308.04758](https://arxiv.org/abs/2308.04758))
 BEV-CLIP: Multi-modal BEV Retrieval Methodology for Complex Scene in
Autonomous Driving (CVPR2024)
 Aligning Knowledge Graph with Visual Perception for Object-goal Navigation (ICRA 2024)

## Contact
This repository is currently maintained by Jian Luo, Jie Yang, Bo Cai,  Kang Zhou, Jian Zhang
