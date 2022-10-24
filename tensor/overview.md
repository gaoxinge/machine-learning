# overview

## overview

- tensor: python/c++
- graph: standalone/distributed
- IR

```
                                                      plugin
                                                        |
                 train          tensorrt parser         v           tensorrt optimize                    tensorrt inference
in-memory model -------> model -----------------> in-memory model --------------------> in-memory model --------------------> output 
                                                                                               ^
                                                                                               |
                                                                                             input
```

## awesome

- [HuaizhengZhang/Awesome-System-for-Machine-Learning](https://github.com/HuaizhengZhang/Awesome-System-for-Machine-Learning)

## open source

### symbolic programming

- [sympy/sympy](https://github.com/sympy/sympy)
- [aesara-devs/aesara](https://github.com/aesara-devs/aesara)
- [Theano/Theano](https://github.com/Theano/Theano)

### deep learning

- [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow)
- [pytorch/pytorch](https://github.com/pytorch/pytorch)
- [google/jax](https://github.com/google/jax)
- [Oneflow-Inc/oneflow](https://github.com/Oneflow-Inc/oneflow)

