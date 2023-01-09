# golang

## overview

```mermaid
flowchart TD;
  subgraph ARROW
  A(arrow array)
  B(arrow table)
  A --> B
  end

  subgraph DATAFRAME
  C(dataframe)
  D(distributed dataframe)
  C --> D
  end

  E(database)
  F(sql)
  
  ARROW --> DATAFRAME
  ARROW --> E
  DATAFRAME --> F
  E --> F
```

## open source

| | standalone | distributed |
|-|------------|-------------|
| [kfultz07/go-dataframe](https://github.com/kfultz07/go-dataframe) | x | |
| [AdikaStyle/go-df](https://github.com/AdikaStyle/go-df) | x | |
