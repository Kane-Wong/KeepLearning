

## 性能分析
### sample1 cpu example
矩阵尺寸：[1024,1024] * [1024,1024] 

| method_cpu  | time(ms) | 
| ----------- |--------- |
| case0       | 8250     |
| case1       | 5474     |

### sample2 CUDA example for block size
矩阵尺寸：[1024,1024] * [1024,1024], 变换不同block size， 测试case0和case1的耗时

| case0(ms)| case1(ms)| block_x | block_y | device  | 
| -------- | -------- |-------- |-------- | ------- |
| 15       | 3.64     | 4       | 4       | TITAN V |
| 60       | 1.267    | 32      | 32      | TITAN V |
| 29       | 1.539    | 16      | 16      | TITAN V |
| 76       | 12.865   | 1024    | 1       | TITAN V |
| 4.19     | 2.117    | 1       | 1024    | TITAN V |
| 3.973    | 2.488    | 1       | 256     | TITAN V |
| 3.876    | 2.492    | 1       | 128     | TITAN V |
| 4.879    | 2.435    | 1       | 32      | TITAN V |
| 4.377    | 1.315    | 4       | 128     | TITAN V |
| 15.91    | 1.264    | 8       | 128     | TITAN V |

### sample3 优化性能
矩阵尺寸：[1024,1024] * [1024,1024] ，case0 和 case1 的最好结果为1.264ms。

| method   | time(ms) | block_xy | other      |  
| -------- |--------- |--------- | ---------- |
| case2    |  0.688   | (32,32)  | block tile |
| case3    |  0.749   | (64,64)  | block tile + thread tile, 每个thread计算8*8个输出值 |

1. 对比case2， case3主要消耗在matA、matB数据加载上，去掉此部分仅耗时0.366ms，而case2去掉此部分耗时0.659ms

### knowledge

1. cpu 算法最里层的加法，`sum += *** ` 5474ms，`r[index] += ***` 8250ms



## reference
1. https://blog.csdn.net/qianqing13579/article/details/127359866