## command

```
# generate txt
$ XLA_FLAGS="--xla_dump_to=tmp/" TF_XLA_FLAGS="--tf_xla_auto_jit=2" python main.py
# generate html
$ XLA_FLAGS="--xla_dump_to=tmp/ --xla_dump_hlo_as_html" TF_XLA_FLAGS="--tf_xla_auto_jit=2" python main.py
# generate all pass
$ XLA_FLAGS="--xla_dump_to=tmp/ --xla_dump_hlo_pass_re=.*" TF_XLA_FLAGS="--tf_xla_auto_jit=2" python main.py
# set cuda path
$ XLA_FLAGS="--xla_dump_to=tmp/ --xla_gpu_cuda_data_dir='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8'" TF_XLA_FLAGS="--tf_xla_auto_jit=2" python main.py
# set cpu
$ CUDA_VISIBLE_DEVICES=-1 XLA_FLAGS="--xla_dump_to=tmp/" TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit" python main.py
```
