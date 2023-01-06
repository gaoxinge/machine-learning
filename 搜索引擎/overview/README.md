## 系统

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
- 用户向量化
- 内容向量化

### 数据库

- 功能
  - 存放链接

### 文件系统

- 功能
  - 存放html

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
