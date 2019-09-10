- [Tar File Installation](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html#installing-tar)

```
> docker run -it --name gxg -v /dev/shm:/dev/shm -v /home/user/gxg/:/gxg/ nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04 bash

$ apt-get update
$ apt-get install -y python3-dev
$ apt-get install -y python3-setuptools
$ easy_install3 -U pip
$ apt-get install -y git vim wget unzip tree
$ pip3 install tensorflow-gpu pillow pycuda

$ cd /opt
$ mv /gxg/TensorRT-5.1.5.0.Ubuntu-16.04.5.x86_64-gnu.cuda-9.0.cudnn7.5.tar.gz .
$ tar xzvf TensorRT-5.1.5.0.Ubuntu-16.04.5.x86_64-gnu.cuda-9.0.cudnn7.5.tar.gz
$ ls TensorRT-5.1.5.0
TensorRT-Release-Notes.pdf  bin  data  doc  graphsurgeon  include  lib  python  samples  targets  uff
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/TensorRT-5.1.5.0/lib
$ cd /opt/TensorRT-5.1.5.0/python
$ pip3 install tensorrt-5.1.5.0-cp35-none-linux_x86_64.whl
$ cd /opt/TensorRT-5.1.5.0/uff
$ pip3 install uff-0.6.3-py2.py3-none-any.whl
$ which convert-to-uff
/usr/local/bin/convert-to-uff
$ cd /opt/TensorRT-5.1.5.0/graphsurgeon
$ pip3 install graphsurgeon-0.4.1-py2.py3-none-any.whl

$ cd /opt/TensorRT-5.1.5.0
$ tree lib
$ tree include
$ tree data
$ tree bin
```

- [2. “Hello World” For TensorRT](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#mnist_sample)

```
$ export CUDA_INSTALL_DIR=/usr/local/cuda
$ export CUDNN_INSTALL_DIR=/usr/local/cuda
$ cd /opt/TensorRT-5.1.5.0/samples/sampleMNIST
$ make
$ cd ../../bin
$ ./sample_mnist
&&&& RUNNING TensorRT.sample_mnist # ./sample_mnist
[I] Building and running a GPU inference engine for MNIST
[I] Input:
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@%.:@@@@@@@@@@@@
@@@@@@@@@@@@@: *@@@@@@@@@@@@
@@@@@@@@@@@@* =@@@@@@@@@@@@@
@@@@@@@@@@@% :@@@@@@@@@@@@@@
@@@@@@@@@@@- *@@@@@@@@@@@@@@
@@@@@@@@@@# .@@@@@@@@@@@@@@@
@@@@@@@@@@: #@@@@@@@@@@@@@@@
@@@@@@@@@+ -@@@@@@@@@@@@@@@@
@@@@@@@@@: %@@@@@@@@@@@@@@@@
@@@@@@@@+ +@@@@@@@@@@@@@@@@@
@@@@@@@@:.%@@@@@@@@@@@@@@@@@
@@@@@@@% -@@@@@@@@@@@@@@@@@@
@@@@@@@% -@@@@@@#..:@@@@@@@@
@@@@@@@% +@@@@@-    :@@@@@@@
@@@@@@@% =@@@@%.#@@- +@@@@@@
@@@@@@@@..%@@@*+@@@@ :@@@@@@
@@@@@@@@= -%@@@@@@@@ :@@@@@@
@@@@@@@@@- .*@@@@@@+ +@@@@@@
@@@@@@@@@@+  .:-+-: .@@@@@@@
@@@@@@@@@@@@+:    :*@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@

[I] Output:
0: 
1: 
2: 
3: 
4: 
5: 
6: **********
7: 
8: 
9: 

&&&& PASSED TensorRT.sample_mnist # ./sample_mnist
```

- [17. Introduction To Importing Caffe, TensorFlow And ONNX Models Into TensorRT Using Python](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#introductory_parser_samples)

```
$ cd /opt/TensorRT-5.1.5.0/samples/python/introductory_parser_samples
$ python3 onnx_resnet50.py -d /opt/TensorRT-5.1.5.0/data/
Correctly recognized /opt/TensorRT-5.1.5.0/data/resnet50/reflex_camera.jpeg as reflex camera
```

- [18. “Hello World” For TensorRT Using TensorFlow And Python](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#end_to_end_tensorflow_mnist)

```
$ cd /opt/TensorRT-5.1.5.0/samples/python/end_to_end_tensorflow_mnist
$ convert-to-uff models/lenet5.pb
$ python3 sample.py -d /opt/TensorRT-5.1.5.0/data/
Test Case: 3
Prediction: 3
```

## architecture

```
                 train          convert                    tensorrt parser                     tensorrt optimize                    tensorrt inference
in-memory model -------> model ---------> model: uff/onnx -----------------> in-memory model --------------------> in-memory model --------------------> output 
                                                                                                                          ^
                                                                                                                          |
                                                                                                                        input
```

## reference

- [TensorRT安装及使用教程](https://blog.csdn.net/zong596568821xp/article/details/86077553)
- [install and configure TensorRT 4 on ubuntu 16.04](https://kezunlin.me/post/dacc4196/)
- [高性能深度学习支持引擎实战——TensorRT](https://zhuanlan.zhihu.com/p/35657027)