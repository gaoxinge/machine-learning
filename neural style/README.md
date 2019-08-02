# neural style

- [图像风格化算法综述三部曲](https://zhuanlan.zhihu.com/c_185430820)
- [深度学习实践：使用Tensorflow实现快速风格迁移](https://zhuanlan.zhihu.com/p/24383274)
- [anishathalye/neural-style](https://github.com/anishathalye/neural-style)
- [lengstrom/fast-style-transfer](https://github.com/lengstrom/fast-style-transfer)

```
> docker run -it --name gxg -v /dev/shm:/dev/shm -v /home/user/gxg/:/gxg/ nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04 bash

> apt-get update
> apt-get install -y python3-dev
> apt-get install -y python3-setuptools
> easy_install3 -U pip
> apt-get install -y git
> apt-get install -y vim
> apt-get install -y wget
> apt-get install -y unzip

> git clone https://github.com/lengstrom/fast-style-transfer.git
> cd fast-style-transfer
> pip3 install tensorflow-gpu
> pip3 install pillow
> pip3 install numpy==1.16.4
> pip3 install scipy==0.18.1

> tree /gxg
gxg
├── input
│   └── cat.jpg
├── model
│   ├── la_muse.ckpt
│   ├── rain_princess.ckpt
│   ├── scream.ckpt
│   ├── udnie.ckpt
│   ├── wave.ckpt
│   └── wreck.ckpt
└── output
    └── cat.jpg

> python3 evaluate.py --checkpoint /gxg/model/la_muse.ckpt --in-path /gxg/input/ --out-path /gxg/output/
> python3 evaluate.py --checkpoint /gxg/model/rain_princess.ckpt --in-path /gxg/input/ --out-path /gxg/output/
> python3 evaluate.py --checkpoint /gxg/model/scream.ckpt --in-path /gxg/input/ --out-path /gxg/output/
> python3 evaluate.py --checkpoint /gxg/model/udnie.ckpt --in-path /gxg/input/ --out-path /gxg/output/
> python3 evaluate.py --checkpoint /gxg/model/wave.ckpt --in-path /gxg/input/ --out-path /gxg/output/
> python3 evaluate.py --checkpoint /gxg/model/wreck.ckpt --in-path /gxg/input/ --out-path /gxg/output/
```