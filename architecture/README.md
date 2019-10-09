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
|intermedia representation      |
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
- intermedia representation
  - [tensorflow/mlir](https://github.com/tensorflow/mlir)
    - [The LLVM Compiler Infrastructure: 2019 European LLVM Developers Meeting](https://llvm.org/devmtg/2019-04/talks.html)
    - [MLIR: Multi-Level Intermediate Representation Compiler Infrastructure](https://llvm.org/devmtg/2019-04/slides/Keynote-ShpeismanLattner-MLIR.pdf)
    - [TensorFlow Graph Optimizations](https://web.stanford.edu/class/cs245/slides/TFGraphOptimizationsStanford.pdf)    
    - [MLIR: A new intermediate representation and compiler framework](https://medium.com/tensorflow/mlir-a-new-intermediate-representation-and-compiler-framework-beba999ed18d)
  - [dmlc/HalideIR](https://github.com/dmlc/HalideIR)
