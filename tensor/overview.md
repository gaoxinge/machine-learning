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

### differentiable programming

- [可微编程：打开深度学习的黑盒子](https://www.jiqizhixin.com/articles/2018-06-08)
- [梯度下降是最好的程序员：Julia未来将内嵌可微编程系统](https://www.jiqizhixin.com/articles/2019-07-21-3)
- [What is Differentiable Programming?](https://www.quora.com/What-is-Differentiable-Programming)
- [Demystifying Differentiable Programming: Shift/Reset the Penultimate Backpropagator](https://arxiv.org/pdf/1803.10228.pdf)
- [First-Class Automatic Differentiation in Swift: A Manifesto](https://gist.github.com/rxwei/30ba75ce092ab3b0dce4bde1fc2c9f1d)

### deep learning

- [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow)
- [pytorch/pytorch](https://github.com/pytorch/pytorch)
- [google/jax](https://github.com/google/jax)
- [Oneflow-Inc/oneflow](https://github.com/Oneflow-Inc/oneflow)

