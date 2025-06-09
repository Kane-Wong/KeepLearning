# softmax

## cpu性能
### sample1
matrix size: [1024,1024]

|  cpu(ms)    | method   |      other       |  
| ----------- |--------- |----------------- |
| 19.941      | case0    |                  |
| 26.601      | case1    |                  |
| 36.953      | case2    | online softmax   |

### sample2
matrix size: [1024,2048] 

|  cpu(ms)    | method   |      other       |  
| ----------- |--------- |----------------- |
| 44.282      | case0    |                  |
| 71.303      | case2    |                  |


## gpu性能
### sample1， matrix size: [1024,1024]

| RTX 3090(ms) | kernel(ms) | method   |      other                  |  
| ------------ | ---------- | ---------|---------------------------- |
| 3.449        | 0.774      | case0    | <<<1024, 1>>>               |
| 3.625        |            | case1    | <<<1024, 1024>>>            |
| 3.57         |            | case2    | <<<1024, 32>>>              |
| 3.141        |            | case3    | <<<1024, 1>>>, fun0_online  |
| 4.853        |            | case4    | <<<1024, 32>>>, fun2_online |

### sample2， matrix size: [1024,2048] 

| RTX 3090(ms) | method   |      other       |  
| ------------ |--------- |----------------- |
| 7.443        | case0    | <<<1024, 1>>>    |
| 6.709        | case2    | <<<1024, 32>>>   |
| 7.37         | case3    | <<<1024, 1>>>    |
| 7.85         | case4    | <<<1024, 32>>>   |

### case0 block分布测试

|          | elapsed(ms) | kernel(ms)  |      block       |                                             |
| -------- |------------ |------------ |----------------- | ------------------------------------------- |
| RTX 3090 | 7.443       | 3.91        | <<<1, 1024>>>    | 仅有1个block，82个multiprocessors未充分利用   |
| RTX 3090 | 3.449       | 0.148       | <<<1024, 1>>>    | 一个warp中剩余31个线程空置，硬件资源未充分利用  |
| RTX 3090 | 3.445       | 0.226       | <<<32, 32>>>     | 仅有32个block，82个multiprocessors未充分利用  |
| RTX 3090 | 3.638       | 0.417       | <<<8, 128>>>     | 仅有8个block，82个multiprocessors未充分利用   |
| RTX 3090 | 3.146       | 0.136       | <<<128, 8>>>     | 一个warp中剩余24个线程空置，硬件资源未充分利用  |

1. 将所有线程放在同一block的表现最糟糕。
2. 此方式就warp空置和multiprocessors空置情况来看，无法充分利用计算资源。
3. 整体来看，打满multiprocessors比打满warp效果更好。

## knowledge
1. online softmax 未提升cpu运行速度

## reference
1. https://zhuanlan.zhihu.com/p/704789263
2. https://zhuanlan.zhihu.com/p/719205928
