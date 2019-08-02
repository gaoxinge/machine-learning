- [Installation Guide](https://xgboost.readthedocs.io/en/latest/build.html)
- [XGBoost GPU Support](https://xgboost.readthedocs.io/en/latest/gpu/)
- [Updates to the XGBoost GPU algorithms](https://xgboost.ai/2018/07/04/gpu-xgboost-update.html)

```
> docker build -t xgboost-gxg .
> docker run -it -v /dev/shm:/dev/shm --name gxg xgboost-gxg bash
```