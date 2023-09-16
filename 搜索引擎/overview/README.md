## 系统

![](//www.plantuml.com/plantuml/png/PPBFJW8n4CRl-nGDTo1U2J6yWWTl9ARkJf082tIxHcDCY1xW8lynQhYWCI4H5HF_AA9FOxjmyXKiEpJfwedE_Bv-_PZPZCSfutwnO36mENLp1I2ne3UIzTSOeHxnFM0csl217P1Drm7cgPwJTtEXhkARAdbT3PwkGk01PVOEERG8CGLdDJaOozP0oNvcbwBQaOsEEejPFodUJiIaXnPPD-kk9GWKgCVpDdg5jXps-HRMKMVHQYzgJNGEbNkAbTXAIwmZCwTwL4oRcBF4k85aVo9FxvV0ePp-ZH_Kn_EBMxnNpZl0eXWlYsatfWssgGV_9l3zyIBkpiNnA2aUtkUFGopGLKMcEOYcBT8o0yUt8LbMZ55tDzJgmhAMooBfU7nzkOwAZyVLaF07M-5_nU0adb0D7J555jqlsRWneSwI8i5dUp5yjh06XEOUJ10TkwKV8EaGwiYwZtSmLmPSHtsFlm00)

### 爬虫

```python
qs = QueueSet(url0)
while len(qs) > 0:
    url = qs.pop()
    html = fetch(url)
    urls = parse(html)
    store(html, url, urls)
    for url in urls:
        qs.push(url)
```

### 排序

- 相关性
  - 算法
    - tf-idf
- 质量分
  - 算法
    - pagerank
  - 实现
    - mapreduce
- 点击率预估
  - 算法
    - 考虑用户偏好，个性化，添加推荐系统相关的逻辑
