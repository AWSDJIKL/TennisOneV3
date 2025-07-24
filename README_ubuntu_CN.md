# 运行在Ubuntu环境下的安装教程

本教程运行在Ubuntu2204版本，python环境配置推荐使用Anaconda管理

注意本解决方案受限于使用奥比中光SDK限制不能运行在WSL等没有GUI的系统中

硬件环境：

Intel(R) Core(TM) i9-14900HX

64GB内存

NVIDIA GeForce RTX 5080  GPU

Cuda环境：cuda12.8+cudnn9.7.1

# 1. 环境搭建

### 1.1 conda 虚拟环境搭建

~~~
conda create -n tennis_one python=3.9
conda activate tennis_one
# install pytorch
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
# install utils tool, TrackNetV3 and openpose requirements
pip3 install -r requirements.txt
~~~

### 1.2 编译orbbec的python SDK

可以参考[官方文档](https://orbbec.github.io/pyorbbecsdk/source/2_installation/build_the_package.html)进行操作。

注意：因为本项目推荐使用anaconda管理python环境，因此需要特别指定python路径，可以激活虚拟环境后利用`which python`命令得到虚拟环境的路径
~~~
# 设定使用的python环境
set(Python3_ROOT_DIR "/home/tennisone/anaconda3/envs/tennis_one/bin/python") # 替换为上文查询到的虚拟环境路径
set(pybind11_DIR "${Python3_ROOT_DIR}/lib/python3.9/site-packages/pybind11/share/cmake/pybind11")
# 编译
mkdir build
cd build
cmake -Dpybind11_DIR=`pybind11-config --cmakedir` ..
make -j4
make install
~~~

### 1.3 下载并配置与训练模型

~~~
python download_pretrain_model.py
~~~

# 2. 运行

### 2.1 通过orbbecsdk录制视频，默认设置为30秒一个视频，不间断录制

~~~
python ./multi_device_sync_record.py -dn 2
~~~

参数解释

-dn:摄像机的数量

### 2.2 使用TracknetV3提取视频中的击球片段并切片，并对该击球动作进行分析

~~~
python pipeline_by_vitpose.py --video_file ./video/game3.mp4 --phase test
~~~

参数解释

-video_file:指定需要提取分析的视频路径

-phase:表示测试阶段，无需修改

