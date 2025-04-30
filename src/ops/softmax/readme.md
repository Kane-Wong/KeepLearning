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
### sample1
matrix size: [1024,1024]

| RTX 3090(ms) | method   |      other                  |  
| ------------ |--------- |---------------------------- |
| 7.413        | case0    | <<<1, 1024>>>               |
| 3.116        | case1    | <<<1024, 1>>>               |
| 3.625        | case2    | <<<1024, 1024>>>            |
| 3.57         | case3    | <<<1024, 32>>>              |
| 3.141        | case4    | <<<1024, 1>>>, fun0_online  |
| 4.853        | case5    | <<<1024, 32>>>, fun2_online |

### sample2
matrix size: [1024,2048] 

| RTX 3090(ms) | method   |      other       |  
| ------------ |--------- |----------------- |
| 7.443        | case1    | <<<1024, 1>>>    |
| 6.709        | case3    | <<<1024, 32>>>   |
| 7.37         | case4    | <<<1024, 1>>>    |
| 7.85         | case5    | <<<1024, 32>>>   |

## knowledge
1. online softmax 未提升cpu运行速度

## reference
1. https://zhuanlan.zhihu.com/p/704789263
2. https://zhuanlan.zhihu.com/p/719205928
