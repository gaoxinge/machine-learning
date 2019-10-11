## architecture

```
                                                             plugin
                                                               |
                 train          convert         parser         v          optimize                    inference
in-memory model -------> model ---------> onnx --------> in-memory model ----------> in-memory model -----------> output 
                                                                                            ^
                                                                                            |
                                                                                          input
```

```
+-------------------------------+
|python tensor library          |
+-------------------------------+
|c/c++ tensor library           |
+-------------------------------+
|programming language           |
+-------------------------------+
|graph ir                       | ---> task parallel
+-------------------------------+
|intermedia representation      | ---> data parallel
+-------------------------------+
|library                        |
+-------------------------------+
|openmp|mpi|opengl|opencl|cuda  |
+-------------------------------+
|os                             |
+-------------------------------+
|hardware                       |
+-------------------------------+
```

## project

- mobile
  - [tensorflow lite](https://www.tensorflow.org/lite)
  - [core ml](https://developer.apple.com/documentation/coreml)
  - [dmlc/tvm](https://github.com/dmlc/tvm)
  - [alibaba/MNN](https://github.com/alibaba/MNN)
  - [Tencent/ncnn](https://github.com/Tencent/ncnn)
  - [XiaoMi/mace](https://github.com/XiaoMi/mace)
- programming language
  - [halide/Halide](https://github.com/halide/Halide)
  - [lift-project/lift](https://github.com/lift-project/lift)
  - [skelcl/skelcl](https://github.com/skelcl/skelcl)
- graph ir
  - task parallel
    - [分布式机器学习之——Spark MLlib并行训练原理](https://zhuanlan.zhihu.com/p/81784947)
    - [一文读懂「Parameter Server」的分布式机器学习训练原理](https://zhuanlan.zhihu.com/p/82116922)
    - [yahoo/TensorFlowOnSpark](https://github.com/yahoo/TensorFlowOnSpark)
    - [ray-project/ray](https://github.com/ray-project/ray)
    - [horovod/horovod](https://github.com/horovod/horovod)
    - [bytedance/byteps](https://github.com/bytedance/byteps)
- intermedia representation
  - [tensorflow/mlir](https://github.com/tensorflow/mlir)
    - [The LLVM Compiler Infrastructure: 2019 European LLVM Developers Meeting](https://llvm.org/devmtg/2019-04/talks.html)
    - [MLIR: Multi-Level Intermediate Representation Compiler Infrastructure](https://llvm.org/devmtg/2019-04/slides/Keynote-ShpeismanLattner-MLIR.pdf)
    - [TensorFlow Graph Optimizations](https://web.stanford.edu/class/cs245/slides/TFGraphOptimizationsStanford.pdf)    
    - [MLIR: A new intermediate representation and compiler framework](https://medium.com/tensorflow/mlir-a-new-intermediate-representation-and-compiler-framework-beba999ed18d)
  - [dmlc/HalideIR](https://github.com/dmlc/HalideIR)

## differential programming

- [可微编程：打开深度学习的黑盒子](https://www.jiqizhixin.com/articles/2018-06-08)
- [梯度下降是最好的程序员：Julia未来将内嵌可微编程系统](https://www.jiqizhixin.com/articles/2019-07-21-3)
- [Differentiable programming](https://en.wikipedia.org/wiki/Differentiable_programming)
- [What is Differentiable Programming?](https://www.quora.com/What-is-Differentiable-Programming)
- [Demystifying Differentiable Programming: Shift/Reset the Penultimate Backpropagator](https://arxiv.org/pdf/1803.10228.pdf)
- [First-Class Automatic Differentiation in Swift: A Manifesto](https://gist.github.com/rxwei/30ba75ce092ab3b0dce4bde1fc2c9f1d)

## probabilistic programming

- [CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)