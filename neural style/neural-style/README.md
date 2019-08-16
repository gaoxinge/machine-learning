# anishathalye/neural-style

## run 

```
> docker run -it --name gxg -v /dev/shm:/dev/shm -v /home/user/gxg/:/gxg/ nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04 bash

> apt-get update
> apt-get install -y python3-dev
> apt-get install -y python3-setuptools
> easy_install3 -U pip
> apt-get install -y git
> apt-get install -y vim
> apt-get install -y wget

> git clone https://github.com/anishathalye/neural-style.git
> cd neural-style
> wget http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat
> pip3 install -r requirements.txt

> python3 neural_style.py --content /gxg/input/cat.jpg --styles /gxg/style/star.jpg --output /gxg/output/cat.jpg --style-layer-weight-exp 0.2
```