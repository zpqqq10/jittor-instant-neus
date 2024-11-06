# Jittor-Instant-NeuS

[Xiuchao Wu](https://xchaowu.github.io/), Xin Zhang, Peiquan Zhang, Zhenyang Li, [Weiwei Xu](http://www.cad.zju.edu.cn/home/weiweixu/weiweixu_en.htm)

## Introduction
Jittor-Instant-NeuS is a NeuS work based on [Jittor](https://github.com/Jittor/jittor) and [JNeRF](https://github.com/Jittor/JNeRF), mostly motivated by [Instant-NSR-pl](https://github.com/bennyguo/instant-nsr-pl). We developed this repo for the 2nd International Algorithm Case Competition (IACC) the Greater Bay Area 2023 (category of three-dimensional reconstruction of objects represented by neural implicit representations), and winned the first prize, among 466 teams. 

## Install
Jittor-Instant-NeuS environment requirements (tested on Ubuntu 22.04 with RTX 4090 and CUDA 11.8):

* System: **Linux**(e.g. Ubuntu/CentOS/Arch), **macOS**, or **Windows Subsystem of Linux (WSL)**
* Python version >= 3.9
* CPU compiler (require at least one of the following)
    * g++ (>=5.4.0)
    * clang (>=8.0)
* GPU compiler (optional)
    * nvcc (>=10.0 for g++ or >=10.2 for clang)
* GPU library: cudnn-dev (recommend tar file installation, [reference link](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-tar))
* GPU supporting:
  * sm arch >= sm_61 (GTX 10x0 / TITAN Xp and above)
  * to use fp16: sm arch >= sm_70 (TITAN V / V100 and above). JNeRF will automatically use original fp32 if the requirements are not meet.
  * to use FullyFusedMLP: sm arch >= sm_75 (RTX 20x0 and above). JNeRF will automatically use original MLPs if the requirements are not meet.

**Install the requirements**

We found that the new dependency requires python >= 3.9, so we recommend using conda to create an environment with python 3.10.

```shell
conda create -n jt python=3.10
python3 -m pip install --user -e .
```
If you have any installation problems for Jittor, please refer to [Jittor](https://github.com/Jittor/jittor)

If you encounter the `~/anaconda3/envs/jt/bin/../lib/libstdc++.so.6: version 'GLIBCXX_3.4.30' not found` error, please use the following commands to solve it:

```shell
# to confirm that the required version exists 
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
# create a soft link
cd ~/anaconda3/envs/jt/bin/../lib/
mv libstdc++.so.6 libstdc++.so.6.old
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 libstdc++.so.6
```

Also, if `import pymeshlab` cause error like `libmeshlab-common.so: undefined symbol: _ZdlPvm, version Qt_5`, you can try `export LD_LIBRARY_PATH=~/anaconda3/envs/jt/lib/python3.10/site-packages/pymeshlab/lib/:$LD_LIBRARY_PATH` from [issue](https://github.com/cnr-isti-vclab/PyMeshLab/issues/341). 

After installation, you can ```import jnerf``` in python interpreter to check if it is successful or not.

## Getting Started

### Datasets

The dataset is provided by the competition, and you can download it from the competition website. The dataset is organized as follows:

```shell
.
├── cameras_sphere.npz
├── image
│   ├── 000.png
│   ├── 001.png
│   ├── 002.png
│   ├── 003.png
│   └── ...
└── mask
    ├── 000.png
    ├── 001.png
    ├── 002.png
    ├── 003.png
    └──...
```

### Config

We organize our configs of Jittor-Instant-NeuS in projects/. You can refer to `./projects/neus/configs/instant_neus_handbag_019.py` to learn how it works.

### Train from scratch

You can set `cal_bbox = True` in the config file to calculate the bounding box of the object in the first run. 

Then for the second run, you can set `cal_bbox = False`, `bbox_corner` and `bbox_size` according to the output of the first run. The mesh will be saved in the `./logs/` folder.

```shell
python tools/run_net.py --config-file ./projects/neus/configs/instant_neus_handbag_019.py --type instant --task train
```

## Main Features

We adopt some designs from [Instant-NGP](https://github.com/NVlabs/instant-ngp) and [Neuralangelo](https://research.nvidia.com/labs/dir/neuralangelo/) to improve the design of NeuS:

1. hashgrid

We adopt the multi-resolution hashgrid to efficiently encode the geometry feature, which can significantly reduce the memory consumption and improve the reconstruction quality.

2. numerical gradient

We adopt the numerical gradient to calculate the second-order derivative (required by the eikonal loss), which produces a smoother result than analytical gradient more stably.

3. visual hull

We adopt the visual hull to provide a more accurate initialization for the occupancy grid, which can significantly reduce the training time and improve the quality of the final result.

4. progressive optimization

We adopt progressive optimization on the occupancy grid and the hashgrid. We prune invalid vertices in the occupancy grid according to the weights derived from volume rendering, resulting in a more accurate and efficient sampling. On the hashgrid, we progressively activate finer levels of hashgrid to efficiently eliminate noise and artifacts.

5. post decoding

We perform volume rendering on the geometry feature rather than radiance color, which helps to reconstruct dark regions (gradient of which is small and hard to optimize) and reduce training time.

6. curvature loss

We adopt the curvature loss to regularize the surface smoothness, which utilizes the second-order derivative and helps reconstruction under sparse views.

## Acknowledgements

The original implementation comes from the following cool project:
* [Instant-NGP](https://github.com/NVlabs/instant-ngp)
* [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)
* [Eigen](https://github.com/Tom94/eigen) ([homepage](https://eigen.tuxfamily.org/index.php?title=Main_Page))

Their licenses can be seen at `licenses/`, many thanks for their nice work!


## Citation


```
@article{hu2020jittor,
  title={Jittor: a novel deep learning framework with meta-operators and unified graph execution},
  author={Hu, Shi-Min and Liang, Dun and Yang, Guo-Ye and Yang, Guo-Wei and Zhou, Wen-Yang},
  journal={Science China Information Sciences},
  volume={63},
  number={222103},
  pages={1--21},
  year={2020}
}
@article{mueller2022instant,
    author = {Thomas M\"uller and Alex Evans and Christoph Schied and Alexander Keller},
    title = {Instant Neural Graphics Primitives with a Multiresolution Hash Encoding},
    journal = {ACM Trans. Graph.},
    issue_date = {July 2022},
    volume = {41},
    number = {4},
    month = jul,
    year = {2022},
    pages = {102:1--102:15},
    articleno = {102},
    numpages = {15},
    url = {https://doi.org/10.1145/3528223.3530127},
    doi = {10.1145/3528223.3530127},
    publisher = {ACM},
    address = {New York, NY, USA},
}
@inproceedings{mildenhall2020nerf,
  title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
  author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
  year={2020},
  booktitle={ECCV},
}
@inproceedings{li2023neuralangelo,
  title={Neuralangelo: High-Fidelity Neural Surface Reconstruction},
  author={Li, Zhaoshuo and M\"uller, Thomas and Evans, Alex and Taylor, Russell H and Unberath, Mathias and Liu, Ming-Yu and Lin, Chen-Hsuan},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition ({CVPR})},
  year={2023}
}
```
