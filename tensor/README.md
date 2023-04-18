# tensor

- [overview](./overview.md)
- [deep learning compiler](./deep%20learning%20compiler.md)
- [infrastructure](./infrastructure.md)

## application

### graphics

- [tensorflow/graphics](https://github.com/tensorflow/graphics)
- [facebookresearch/pytorch3d](https://github.com/facebookresearch/pytorch3d)

## task parallel vs data parallel

- [TensorFlow Architecture](https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/extend/architecture.md)
- [Data Parallelism VS Model Parallelism in Distributed Deep Learning Training](https://leimao.github.io/blog/Data-Parallelism-vs-Model-Paralelism/)
- [Model Parallelism in Deep Learning is NOT What You Think](https://medium.com/@esaliya/model-parallelism-in-deep-learning-is-not-what-you-think-94d2f81e82ed)
- [Data parallel and model parallel distributed training with Tensorflow](http://kuozhangub.blogspot.com/2017/08/data-parallel-and-model-parallel.html)

## distributed training

- [Scaling Distributed Machine Learning with the Parameter Server](https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf)
- [分布式机器学习之——Spark MLlib并行训练原理](https://zhuanlan.zhihu.com/p/81784947)
- [一文读懂「Parameter Server」的分布式机器学习训练原理](https://zhuanlan.zhihu.com/p/82116922)
- [Horovod知识储备：将HPC技术带入深度学习之中](https://zhuanlan.zhihu.com/p/89093128)
- [horovod/horovod](https://github.com/horovod/horovod)
- [bytedance/byteps](https://github.com/bytedance/byteps)

## multi-gpu communication

### nccl

- [NCCL 2.0](http://on-demand.gputechconf.com/gtc/2017/presentation/s7155-jeaugey-nccl.pdf)
- [NVIDIA/nccl](https://github.com/NVIDIA/nccl)
- [NCCL Installation Guide](https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html)
- [NVIDIA Collective Communication Library (NCCL) Documentation](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/index.html)
- [如何理解Nvidia英伟达的Multi-GPU多卡通信框架NCCL？](https://www.zhihu.com/question/63219175)

### baidu-allreduce

- [baidu-research/baidu-allreduce](https://github.com/baidu-research/baidu-allreduce)

## deep learning compiler

- [Machine Learning in Compiler Optimisation](https://arxiv.org/pdf/1805.03441.pdf)
- [Deep Learning Compilers](https://ucbrise.github.io/cs294-ai-sys-sp19/assets/lectures/lec12/dl-compilers.pdf)

## intermediate representation

### mlir

- [tensorflow/mlir](https://github.com/tensorflow/mlir)
- [The LLVM Compiler Infrastructure: 2019 European LLVM Developers Meeting](https://llvm.org/devmtg/2019-04/talks.html)
- [MLIR: Multi-Level Intermediate Representation Compiler Infrastructure](https://llvm.org/devmtg/2019-04/slides/Keynote-ShpeismanLattner-MLIR.pdf)
- [TensorFlow Graph Optimizations](https://web.stanford.edu/class/cs245/slides/TFGraphOptimizationsStanford.pdf)    
- [MLIR: A new intermediate representation and compiler framework](https://medium.com/tensorflow/mlir-a-new-intermediate-representation-and-compiler-framework-beba999ed18d)

### halide ir

- [dmlc/HalideIR](https://github.com/dmlc/HalideIR)
