# 官方文档

- https://pjreddie.com/darknet/

## Installing Darknet

- set `GPU=1` and `OPENCV=1` in Makefile
- run `make`
- run `./darknet`

## YOLO: Real-Time Object Detection

### yolov2

### yolov2-tiny

```
$ mkdir weights
$ cd weights
$ wget https://pjreddie.com/media/files/yolov2-tiny.weights
$ cd ..
$ ./darknet detect cfg/yolov2-tiny.cfg weights/yolov2-tiny.weights data/dog.jpg
layer     filters    size              input                output
    0 conv     16  3 x 3 / 1   416 x 416 x   3   ->   416 x 416 x  16  0.150 BFLOPs
    1 max          2 x 2 / 2   416 x 416 x  16   ->   208 x 208 x  16
    2 conv     32  3 x 3 / 1   208 x 208 x  16   ->   208 x 208 x  32  0.399 BFLOPs
    3 max          2 x 2 / 2   208 x 208 x  32   ->   104 x 104 x  32
    4 conv     64  3 x 3 / 1   104 x 104 x  32   ->   104 x 104 x  64  0.399 BFLOPs
    5 max          2 x 2 / 2   104 x 104 x  64   ->    52 x  52 x  64
    6 conv    128  3 x 3 / 1    52 x  52 x  64   ->    52 x  52 x 128  0.399 BFLOPs
    7 max          2 x 2 / 2    52 x  52 x 128   ->    26 x  26 x 128
    8 conv    256  3 x 3 / 1    26 x  26 x 128   ->    26 x  26 x 256  0.399 BFLOPs
    9 max          2 x 2 / 2    26 x  26 x 256   ->    13 x  13 x 256
   10 conv    512  3 x 3 / 1    13 x  13 x 256   ->    13 x  13 x 512  0.399 BFLOPs
   11 max          2 x 2 / 1    13 x  13 x 512   ->    13 x  13 x 512
   12 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024  1.595 BFLOPs
   13 conv    512  3 x 3 / 1    13 x  13 x1024   ->    13 x  13 x 512  1.595 BFLOPs
   14 conv    425  1 x 1 / 1    13 x  13 x 512   ->    13 x  13 x 425  0.074 BFLOPs
   15 detection
mask_scale: Using default '1.000000'
Loading weights from weights/yolov2-tiny.weights...Done!
data/dog.jpg: Predicted in 1.282148 seconds.
dog: 82%
car: 74%
bicycle: 59%
```

### yolov3

### yolov3-tiny

```
$ mkdir weights
$ cd weights
$ wget https://pjreddie.com/media/files/yolov3-tiny.weights
$ cd ..
$ ./darknet detect cfg/yolov3-tiny.cfg weights/yolov3-tiny.weights data/dog.jpg
layer     filters    size              input                output
    0 conv     16  3 x 3 / 1   416 x 416 x   3   ->   416 x 416 x  16  0.150 BFLOPs
    1 max          2 x 2 / 2   416 x 416 x  16   ->   208 x 208 x  16
    2 conv     32  3 x 3 / 1   208 x 208 x  16   ->   208 x 208 x  32  0.399 BFLOPs
    3 max          2 x 2 / 2   208 x 208 x  32   ->   104 x 104 x  32
    4 conv     64  3 x 3 / 1   104 x 104 x  32   ->   104 x 104 x  64  0.399 BFLOPs
    5 max          2 x 2 / 2   104 x 104 x  64   ->    52 x  52 x  64
    6 conv    128  3 x 3 / 1    52 x  52 x  64   ->    52 x  52 x 128  0.399 BFLOPs
    7 max          2 x 2 / 2    52 x  52 x 128   ->    26 x  26 x 128
    8 conv    256  3 x 3 / 1    26 x  26 x 128   ->    26 x  26 x 256  0.399 BFLOPs
    9 max          2 x 2 / 2    26 x  26 x 256   ->    13 x  13 x 256
   10 conv    512  3 x 3 / 1    13 x  13 x 256   ->    13 x  13 x 512  0.399 BFLOPs
   11 max          2 x 2 / 1    13 x  13 x 512   ->    13 x  13 x 512
   12 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024  1.595 BFLOPs
   13 conv    256  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 256  0.089 BFLOPs
   14 conv    512  3 x 3 / 1    13 x  13 x 256   ->    13 x  13 x 512  0.399 BFLOPs
   15 conv    255  1 x 1 / 1    13 x  13 x 512   ->    13 x  13 x 255  0.044 BFLOPs
   16 yolo
   17 route  13
   18 conv    128  1 x 1 / 1    13 x  13 x 256   ->    13 x  13 x 128  0.011 BFLOPs
   19 upsample            2x    13 x  13 x 128   ->    26 x  26 x 128
   20 route  19 8
   21 conv    256  3 x 3 / 1    26 x  26 x 384   ->    26 x  26 x 256  1.196 BFLOPs
   22 conv    255  1 x 1 / 1    26 x  26 x 256   ->    26 x  26 x 255  0.088 BFLOPs
   23 yolo
Loading weights from weights/yolov3-tiny.weights...Done!
data/dog.jpg: Predicted in 1.278319 seconds.
dog: 57%
car: 52%
truck: 56%
car: 62%
bicycle: 59%
```

### yolov3-spp

## ImageNet Classification

## Nightmare

## RNNs in Darknet

## DarkGo: Go in Darknet

## Tiny Darknet

```
$ mkdir weights
$ cd weights
$ wget https://pjreddie.com/media/files/tiny.weights
$ cd ..
$ ./darknet classify cfg/tiny.cfg weights/tiny.weights data/dog.jpg
layer     filters    size              input                output
    0 conv     16  3 x 3 / 1   224 x 224 x   3   ->   224 x 224 x  16  0.043 BFLOPs
    1 max          2 x 2 / 2   224 x 224 x  16   ->   112 x 112 x  16
    2 conv     32  3 x 3 / 1   112 x 112 x  16   ->   112 x 112 x  32  0.116 BFLOPs
    3 max          2 x 2 / 2   112 x 112 x  32   ->    56 x  56 x  32
    4 conv     16  1 x 1 / 1    56 x  56 x  32   ->    56 x  56 x  16  0.003 BFLOPs
    5 conv    128  3 x 3 / 1    56 x  56 x  16   ->    56 x  56 x 128  0.116 BFLOPs
    6 conv     16  1 x 1 / 1    56 x  56 x 128   ->    56 x  56 x  16  0.013 BFLOPs
    7 conv    128  3 x 3 / 1    56 x  56 x  16   ->    56 x  56 x 128  0.116 BFLOPs
    8 max          2 x 2 / 2    56 x  56 x 128   ->    28 x  28 x 128
    9 conv     32  1 x 1 / 1    28 x  28 x 128   ->    28 x  28 x  32  0.006 BFLOPs
   10 conv    256  3 x 3 / 1    28 x  28 x  32   ->    28 x  28 x 256  0.116 BFLOPs
   11 conv     32  1 x 1 / 1    28 x  28 x 256   ->    28 x  28 x  32  0.013 BFLOPs
   12 conv    256  3 x 3 / 1    28 x  28 x  32   ->    28 x  28 x 256  0.116 BFLOPs
   13 max          2 x 2 / 2    28 x  28 x 256   ->    14 x  14 x 256
   14 conv     64  1 x 1 / 1    14 x  14 x 256   ->    14 x  14 x  64  0.006 BFLOPs
   15 conv    512  3 x 3 / 1    14 x  14 x  64   ->    14 x  14 x 512  0.116 BFLOPs
   16 conv     64  1 x 1 / 1    14 x  14 x 512   ->    14 x  14 x  64  0.013 BFLOPs
   17 conv    512  3 x 3 / 1    14 x  14 x  64   ->    14 x  14 x 512  0.116 BFLOPs
   18 conv    128  1 x 1 / 1    14 x  14 x 512   ->    14 x  14 x 128  0.026 BFLOPs
   19 conv   1000  1 x 1 / 1    14 x  14 x 128   ->    14 x  14 x1000  0.050 BFLOPs
   20 avg                       14 x  14 x1000   ->  1000
   21 softmax                                        1000
Loading weights from weights/tiny.weights...Done!
data/dog.jpg: Predicted in 0.250622 seconds.
14.51%: malamute
 6.09%: Newfoundland
 5.59%: dogsled
 4.55%: standard schnauzer
 4.05%: Eskimo dog
```

## Train a Classifier on CIFAR-10

## Hardware Guide: Neural Networks on GPUs