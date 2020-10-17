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
   
   ![nvidia-smi](https://note.youdao.com/yws/api/personal/file/WEBda77219cf4b1f4242d12dfb69584b86e?method=download&shareKey=e0f04b56c972bb8729cdd3bbd3a81adb)
   
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
   
   ![dockerfile](https://note.youdao.com/yws/api/personal/file/WEB5be206ec835d87b1e4ed41e233ed6d93?method=download&shareKey=af9218ff76fa4247bbb565c65d43a68b)
   
   ![bazel](https://note.youdao.com/yws/api/personal/file/WEBbef22ba2ab7425fe73251c0d725fafff?method=download&shareKey=1050e8c7c512d8c7fd5ee0dabc77fe9d)
   
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
#### 模型导出
未完，待续...
前面讲述了怎么制作一个tensorflow serving 环境的docker镜像，下篇博客介绍部署及调用。由于tensorflow能够直接保存成savemodel格式，
就不在描述模型导出。
#### 模型部署
请移步 ![tensorflow serving 部署及调用]()
   
#### reference
> [1] https://zhuanlan.zhihu.com/p/23361413

> [2] https://zhuanlan.zhihu.com/p/52096200
