- [Installation Guide](https://xgboost.readthedocs.io/en/latest/build.html)
- [XGBoost GPU Support](https://xgboost.readthedocs.io/en/latest/gpu/)
- [Updates to the XGBoost GPU algorithms](https://xgboost.ai/2018/07/04/gpu-xgboost-update.html)

```
> docker build -t xgboost-gxg .
> docker run -it -v /dev/shm:/dev/shm --name gxg xgboost-gxg bash
# python3 main.py
[0]	train-rmse:21.6024
[1]	train-rmse:19.5552
[2]	train-rmse:17.715
[3]	train-rmse:16.062
[4]	train-rmse:14.5715
[5]	train-rmse:13.2409
[6]	train-rmse:12.0339
[7]	train-rmse:10.9579
[8]	train-rmse:9.97879
[9]	train-rmse:9.10759
[03:23:56] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
[0]	validation_0-rmse:21.6024
[1]	validation_0-rmse:19.5552
[2]	validation_0-rmse:17.715
[3]	validation_0-rmse:16.062
[4]	validation_0-rmse:14.5715
[5]	validation_0-rmse:13.2409
[6]	validation_0-rmse:12.0339
[7]	validation_0-rmse:10.9579
[8]	validation_0-rmse:9.97879
[9]	validation_0-rmse:9.10759
```