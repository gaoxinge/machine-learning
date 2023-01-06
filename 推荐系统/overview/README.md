## 系统

- 推荐: slowTopK: User -> List[Item]
  - 召回: faskTopK: User -> List[Item]
  - 排序: sort: (User, List[Item]) -> List[Item]
    - 生成topK

## 算法

### 协同过滤

- 基于用户
- 基于item

### embedding

- 向量化
  - 评分卡
  - one hot
  - word2vec
  - item2vec
  - MF
  - FM
- 向量搜索

### 点击率预测

- 在线学习: 强化学习
- 离线学习: 监督学习

