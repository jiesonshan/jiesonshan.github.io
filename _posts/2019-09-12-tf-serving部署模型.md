---
layout:     post
title:      tf-serving 部署模型的方法
subtitle:   tensorflow
date:       2019-09-12
author:     Jieson
header-img: img/post-bg-map.jpg
catalog: true
tags:
    - tensorflow
    - model deployment
---
### <center>tf-serving 部署模型的方法</center>
&#160;&#160;&#160;&#160; Tensorflow Serving 是一个用于机器学习模型 serving 的高性能开源库。它可以将训练好的机器学习模型部署到线上，使用 gRPC 作为接口接受外部调用。更加让人眼前一亮的是，它支持模型热更新与自动模型版本管理。这意味着一旦部署 TensorFlow Serving 后，你再也不需要为线上服务操心，只需要关心你的线下模型训练。
####  环境部署（ubuntu16.04）
1、Docker 安装

   `sudo apt-get install docker`
   
   `sudo apt-get install docker.io`

2、 nvidia-docker 安装

   `wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb`
   
   `sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb`
   
   安装完成后验证是否安装成功
   
3、利用dockerfile建立tf-serving 镜像
   
   `git clone https://github.com/tensorflow/serving.git` 
   
   在文件夹 ./serving/tensorflow_serving/tools/docker/ 里有四个dockerfile文件，本文使用的是 Dockerfile.devel-gpu文件。
   在文件夹 ./serving/ 中创建新文件夹 temp， cp ./serving/tensorflow_serving/tools/docker/Dockerfile.devel-gpu ./serving/temp/Dockerfile
   
   修改Dockerfile文件
   为了加速镜像build 更换国内源与pip源，将sources.list和pip.conf及下载好的bazel安装脚本都放在temp文件夹中
   sources.list
   
    ```
    deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial main restricted universe multiverse
    # deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial main restricted universe multiverse
    deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-updates main restricted universe multiverse
    # deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-updates main restricted universe multiverse
    deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-backports main restricted universe multiverse
    # deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-backports main restricted universe multiverse
    deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-security main restricted universe multiverse
    # deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-security main restricted universe multiverse

    # deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-proposed main restricted universe multiverse
    # deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-proposed main restricted universe multiverse
    ```
    
   pip.conf
   
   `
    [global]
    index-url = https://pypi.tuna.tsinghua.edu.cn/simple 
   `
   
   更换docker镜像源
   
   ```
   vim /lib/systemd/system/docker.service
   ExecStart=/usr/bin/docker daemon -H fd:// 这一行后面加上镜像的选项。
   --registry-mirror=https://docker.mirrors.ustc.edu.cn
   
   ```
   
   在Dockerfile文件中修改相应位置
   
   开始build镜像
   
   ```
   cd temp
   docker build -t tf-serving(tag) ./Dockerfile
   ```
   可能出现的问题： ImportError: libcuda.so.1: cannot open shared object file: No such file or directory
   
   解决方法：
   ```
   LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
   ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
   LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}
   ```
   
   
   
   
   
   
   
   

