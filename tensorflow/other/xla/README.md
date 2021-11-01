## 例子

```python
import tensorflow as tf


@tf.function(jit_compile=True)
def model_fn(x, y, z):
    return x + y * z


model_fn(tf.ones([1, 10]), tf.ones([1, 10]), tf.ones([1, 10]))
```

```
XLA_FLAGS="--xla_dump_to=tmp/ --xla_dump_hlo_as_html" TF_XLA_FLAGS="--tf_xla_auto_jit=2" python main.py
```

## reference

- [初识XLA](https://zhuanlan.zhihu.com/p/87709496)
- [XLA 探究：矩阵乘法](https://zhuanlan.zhihu.com/p/88991966)
- [[腾讯机智]TensorFlow XLA工作原理](https://zhuanlan.zhihu.com/p/98565435)
