# reduce

## reduce_sum_gpu
### sample1
matrix size: [1024,1024] ，cpu运行耗时 3.91ms

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


### sample2
matrix size: [2048,2048] ，cpu运行耗时 13.195ms

| RTX 3090(ms) | method   |      other       |  
| ------------ |--------- |----------------- |
| 0.132        | case6    |                  |

## reference
1. https://zhuanlan.zhihu.com/p/17996548596
