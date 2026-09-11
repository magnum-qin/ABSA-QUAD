# 重复推理成本实验

One fixed English dev workload, no training or test scoring; does not establish all-language or production latency.

| Arm | Kind | n | Seconds mean±SD /64 rows | rows/s | Peak allocated GiB |
|---|---|---:|---:|---:|---:|
| A | single | 15 | 0.232±0.002 | 276.3 | 1.32 |
| A | ensemble | 5 | 0.699±0.002 | 91.5 | 3.43 |
| B | single | 15 | 0.358±0.013 | 179.0 | 1.38 |
| B | ensemble | 5 | 1.073±0.006 | 59.7 | 3.60 |
| C | single | 15 | 0.371±0.017 | 172.4 | 1.38 |
| C | ensemble | 5 | 1.111±0.004 | 57.6 | 3.60 |
