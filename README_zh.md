# Jittor-Instant-NeuS

[吴秀超](https://xchaowu.github.io/), 张鑫, 张沛全, 李镇洋, [许威威](http://www.cad.zju.edu.cn/home/weiweixu/weiweixu_en.htm)

## Introduction
Jittor-Instant-NeuS基于[计图](https://github.com/Jittor/jittor)和[JNeRF](https://github.com/Jittor/JNeRF)开发，参考了[Instant-NSR-pl](https://github.com/bennyguo/instant-nsr-pl). 本项目在第二届粤港澳大湾区国际算法算例大赛（神经隐式表示的物体三维重建赛道）中获一等奖。

## 安装
Jittor-Instant-NeuS环境要求（于RTX 4090使用CUDA11.8的Ubuntu 22.04系统上测试）:

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

**安装依赖**

我们发现最新的JNeRF依赖要求python >= 3.9，所以我们推荐使用conda创建一个python 3.10的环境。

```shell
conda create -n jt python=3.10
python3 -m pip install --user -e .
```
任何关于计图的安装问题，请参考[Jittor](https://github.com/Jittor/jittor)

如果安装时遇到了 `~/anaconda3/envs/jt/bin/../lib/libstdc++.so.6: version 'GLIBCXX_3.4.30' not found`，可以使用以下命令解决

```shell
# 确认要求的版本存在
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
# 创建软链接
cd ~/anaconda3/envs/jt/bin/../lib/
mv libstdc++.so.6 libstdc++.so.6.old
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 libstdc++.so.6
```

如果`import pymeshlab`导致错误`libmeshlab-common.so: undefined symbol: _ZdlPvm, version Qt_5`，可以使用命令`export LD_LIBRARY_PATH=~/anaconda3/envs/jt/lib/python3.10/site-packages/pymeshlab/lib/:$LD_LIBRARY_PATH`，该解决方案来自[issue](https://github.com/cnr-isti-vclab/PyMeshLab/issues/341). 

完成安装后，你可以在python解释器中输入`import jnerf`来检查是否安装成功。

## 开始

### 数据集

数据集来自赛事主办方，可以从赛事官网下载，数据集组织形式如下：

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

### 配置

配置文件放置于projects/中，你可以参考`./projects/neus/configs/instant_neus_handbag_019.py`来学习如何配置。

### 训练

在第一次运行时，你可以在配置文件中设置`cal_bbox = True`来计算物体的边界框。

然后在第二次运行时，你可以设置`cal_bbox = False`，根据第一次运行的输出设置`bbox_corner`和`bbox_size`。重建的网格将会保存在`./logs/`文件夹中。

```shell
python tools/run_net.py --config-file ./projects/neus/configs/instant_neus_handbag_019.py --type instant --task train
```

## 主要特性

我们采用了来自[Instant-NGP](https://github.com/NVlabs/instant-ngp)和[Neuralangelo](https://research.nvidia.com/labs/dir/neuralangelo/)的部分设计来提升效果：

1. hashgrid

我们采用多分辨率哈希网格来高效编码几何特征，这可以显著减少内存消耗并提高重建质量。

2. numerical gradient

我们采用数值梯度来计算eikonal loss要求的二阶导数（eikonal loss所需），可以得到更稳定和平滑的结果。

3. visual hull

我们计算视觉外壳来为占用网格提供更准确的初始化，这可以显著减少训练时间并提高最终结果的质量。

4. progressive optimization

我们在占用网格和哈希网格上采用渐进优化。对于占用网格，我们根据体积渲染得到的权重修剪无效顶点，从而实现更准确和高效的采样。在哈希网格上，我们逐渐激活精细级别的哈希网格，以有效消除噪声和伪影。

5. post decoding

我们对几何特征进行体积渲染而不是颜色，这有助于重建暗区域（较暗区域的导数较小，优化困难）并减少训练时间。

6. curvature loss

我们采用curvature loss来规范表面光滑度，该损失利用二阶导数有助于在稀疏视图下重建。

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
