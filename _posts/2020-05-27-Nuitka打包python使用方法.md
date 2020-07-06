---
layout:     post
title:      Nuitka打包python使用方法
subtitle:   安装配置及使用
date:       2020-05-27
author:     Jieson
header-img: img/post-bg-map.jpg
catalog: true
tags:
    - course
    - python
---
### <center>Nuitka打包python使用方法</center>
#### Nuitka 简介
Nuitka是Python编译器，它是用Python编写，对Python解释器的无缝替换或扩展，兼容多个CPython版本。
你可以自由使用所有Python库模块和所有扩展模块。 
Nuitka将Python模块转换为一个C语言程序，然后使用libpython和它自己的静态C文件用与CPython相同的方式执行。
#### 相关软件及配置
1、下载MinGW64 8.1(MinGW编译器比MSVS编译器要快，最低8.1.0版
 
  解压后放在C盘目录下，查询gcc.exe是否有效
  输入gcc.exe --version 检查是否有版本显示
  将c:\mingw64\bin 加入到环境变量中
2、 安装 Nuitka
   pip install Nuitka
#### 使用方法
1、 常用命令参数

--mingw64 #默认为已经安装的vs2017去编译，否则就按指定的比如mingw

--standalone 独立文件，这是必须的

--windows-disable-console 没有CMD控制窗口

--recurse-all 所有的资源文件 这个也选上

-recurse-not-to=numpy,jinja2 不编译的模块，防止速度会更慢

--output-dir=out 生成exe到out文件夹下面去

--show-progress 显示编译的进度，很直观

--show-memory 显示内存的占用

--plugin-enable=pylint-warnings 报警信息

--plugin-enable=qt-plugins 需要加载的PyQT插件

--nofollow-imports  # 所有的import不编译，交给python3x.dll执行

--follow-import-to=need  # need为你需要编译成C/C++的py文件夹命名

2、 命令示例
```buildoutcfg
nuitka --mingw64 --windows-disable-console --standalone --show-progress --show-memory --plugin-enable=qt-plugins --plugin-enable=pylint-warnings --recurse-all --recurse-not-to=numpy --output-dir=out index.py
```

调试模式
```
nuitka --standalone --mingw64 --show-memory --show-progress --nofollow-imports --plugin-enable=qt-plugins --follow-import-to=lib  --output-dir=o main.py
```
release 模式
```buildoutcfg
nuitka --standalone --windows-disable-console --mingw64 --nofollow-imports --show-memory --show-progress --plugin-enable=qt-plugins --follow-import-to=lib --recurse-all --output-dir=o main.py
```
打包完成后将相关的python 包拷贝到exe所在目录

#### pyinstaller 打包
pyinstaller 命令行参数

-h，--help	查看该模块的帮助信息

-F，-onefile	产生单个的可执行文件

-D，--onedir	产生一个目录（包含多个文件）作为可执行程序

-a，--ascii	不包含 Unicode 字符集支持

-d，--debug	产生 debug 版本的可执行文件

-w，--windowed，--noconsolc	指定程序运行时不显示命令行窗口（仅对 Windows 有效）

-c，--nowindowed，--console	指定使用命令行窗口运行程序（仅对 Windows 有效）

-o DIR，--out=DIR	指定 spec 文件的生成目录。如果没有指定，则默认使用当前目录来生成 spec 文件

-p DIR，--path=DIR	设置 Python 导入模块的路径（和设置 PYTHONPATH 环境变量的作用相似）。也可使用路径分隔符（Windows 使用分号，Linux 使用冒号）来分隔多个路径

-n NAME，--name=NAME	指定项目（产生的 spec）名字。如果省略该选项，那么第一个脚本的主文件名将作为 spec 的名字

示例
```buildoutcfg
pyinstaller -F -w main.py # 编译单个文件

pyinstaller -F -w main.py -p depend1.py  -p depend2.py # 多个文件

pyinstaller -F -w main.py --hidden-import lib # 自己的包
```
#### 遇到问题
pyinstaller打包报错： RecursionError: maximum recursion depth exceeded

解决方法：
   
    在第一次打包报时会生成.spec文件， 修改打开该文件
    在文件第二行添加
    ```
    import sys 
    sys.setrecursionlimit(5000)
    ```
    将递归次数设置大一些，然后重新打包
    ```
    pyinstaller *.spec
    ```

#### Reference
> [1]https://zhuanlan.zhihu.com/p/133303836
> [2]https://zhuanlan.zhihu.com/p/137785388
> [3]https://zhuanlan.zhihu.com/p/141810934
> [4]https://github.com/Nuitka/Nuitka


