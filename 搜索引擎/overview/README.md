## 系统

![](//www.plantuml.com/plantuml/png/PPBDIWCn58NtUOfBzzMz5Eahk70jiHF6OfssIKOK4SHDkx4_LaMXcof2mVeZ-8CLVHgJwRo5cRaJI7JLvdpdvfuSfYeavJBUQH3EQf96OOD1AJcRwkvx2QY0M33kehMOyNrOZmB6XT62UXJcldRiklJDKV9odGF1AAzjmG41S0qN7JsUwS7OsYJRanlezPrgC4mxVSUvQrYJ5ruWXzxUtgcQzt5laqi7wVCdVVAGWO1A-ZTEqfqjhkvmhbKF6FSn3glvg6GlsAI2_NlHp-URkhzKnzDbzVYnU9xW0MyraMd8lBGj8t0QRHEIbGoLSdargx4Tuz-Y3Fp99NmJ1eJoaX5ib1L8Hz-wT-j1t6bOyZLC9w-aE2Dq-tQ1No-n2w2IGnlPwZY3kQei2intz0i0)

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

### 向量化

- 关键词向量化
- 内容向量化
- 用户向量化

### 文本搜索引擎

- 功能
  - 存放倒排索引
  - 支持布尔检索
- 实现
  - elasticsearch

### 文本向量搜索

- 功能
  - 根据关键词，搜索相关性最高的文章

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
