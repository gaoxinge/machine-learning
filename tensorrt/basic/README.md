- [高性能深度学习支持引擎实战——TensorRT](https://zhuanlan.zhihu.com/p/35657027)

```
                 train          convert                    tensorrt parser                     tensorrt optimize                    tensorrt inference
in-memory model -------> model ---------> model: uff/onnx -----------------> in-memory model --------------------> in-memory model --------------------> output 
                                                                                                                          ^
                                                                                                                          |
                                                                                                                        input
```