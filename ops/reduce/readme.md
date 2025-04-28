

## 性能分析
### sample1_reduce_sum
矩阵尺寸：[1024,1024] ，cpu运行耗时 3.91ms

| TITAN V(ms) | method   |      other       |  
| ----------- |--------- |----------------- |
| 0.164       | case0    |                  |
| 0.159       | case1    | warp divegence   |
| 0.14        | case2    | bank conflict    |
| 0.134       | case3    |                  |
| 0.123       | case4    |                  |
| 0.122       | case5    |                  |

| RTX 3090(ms) | method   |      other       |  
| ------------ |--------- |----------------- |
| 0.135        | case3    |                  |
| 0.118        | case5    |                  |
| 0.114        | case6    | __shfl_xor_sync  |


### sample2_reduce_sum
矩阵尺寸：[2048,2048] ，cpu运行耗时 13.195ms

| RTX 3090(ms) | method   |      other       |  
| ------------ |--------- |----------------- |
| 0.132        | case6    |                  |

## knowledge
1. warp divegence: 一个warp是共享相同的指令的，若warp内的线程存在分支，那么会一半线程什么都不做，一半线程执行分支后的条件，
2. bank conflict是发生在同一个warp中，不同的线程对于同一个bank的访问
3. fun5 中 sdata 使用volatile 修饰，否则运行错误，原因：避免编译器优化掉对sdata的访问，强制从内存中读取此数据。

## reference
1. https://zhuanlan.zhihu.com/p/17996548596
