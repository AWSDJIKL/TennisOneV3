# 运行在Win10环境下的安装教程

本教程运行在Win10专业版22H2版本，python环境配置推荐使用Anaconda管理

硬件环境：

Intel(R) Core(TM) i9-14900HX

64GB内存

NVIDIA GeForce RTX 4070 Laptop GPU

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

### 1.2 orbbecsdk setup

<details>

#### 1.2.1 安装CMake和Visual Studio

此处提供推荐版本下载链接安装

[CMake-3.17.2-win64-x64](https://drive.google.com/file/d/10IPa59chh4vYpl6Li8kYqxZyY5ueaYNs/view?usp=sharing)

[Visual Studio 2019](https://drive.google.com/file/d/14jAmLRdFaZmASuxTzlXukN6-vLWvrUy-/view?usp=sharing)

#### 1.2.2 配置CMake和Visual Studio

在TennisOneV2/orbbec目录下创建build目录，打开CMake，配置项目路径和build文件夹后点击Configure按钮
![](md_images/cmake1.png)
配置vs版本，选择vs2019，x64版本，使用默认编译配置，点击Finish
![](md_images/cmake2.png)
若出现找不到Python3的错误，需要添加环境变量指定虚拟环境路径
![](md_images/cmake3.png)
点击Add Entry按钮，分别添加Python3_EXECUTABLE和Python3_LIBRARIES，分别是虚拟环境路径下的python.exe和虚拟环境路径下的Lib文件夹，重新点击Configure按钮
![](md_images/cmake4-1.png)
![](md_images/cmake4-2.png)
若出现找不到pybind11的错误，需要修改pybind11_DIR变量的值，指定到虚拟环境下的Lib/site-packages/pybind11/share/cmake/pybind11文件夹，重新点击Configure按钮
![](md_images/cmake5-1.png)
![](md_images/cmake5-2.png)
当出现Configuring done且没有报错后，点击Generate按钮生成项目
![](md_images/cmake6.png)
生成完成后，点击Open Project按钮打开VS2019
![](md_images/cmake7.png)
切换到Release模式，选择pyorbbecsdk项目，右键菜单选择重新生成
![](md_images/vs_1.png)
选择INSTALL项目，右键菜单选择生成
![](md_images/vs_2.png)

复制orbbec/install/lib路径下的所有文件至虚拟环境路径下的Lib\site-packages目录下
![](md_images/image10.png)
![](md_images/image11.png)

</details>


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

