# Attention

## 编译方式
```
编译
cd src/op/attention 
bash ../../../build.sh

执行
./bin/attention_cpu 0       ## cpu 执行 case 0
./bin/attention_gpu 0 1     ## cpu 执行 case 0, gpu 执行 case 1
```

## cpu性能
### sample1
输入尺寸：Q[1, 12, 128, 64]、K[1, 12, 128, 64]、V[1, 12, 128, 64]   

| method  | time(ms) | 
| ------- |--------- |
| case0   | 30.834   |

## gpu性能
### sample1
输入尺寸：Q[1, 12, 128, 64]、K[1, 12, 128, 64]、V[1, 12, 128, 64]   

| method  | time(ms) | 
| ------- |--------- |
| case0   | 4.412    |


## knowledge
1. __device__ 修饰的代码在同一文件或头文件定义，或加参数`-relocatable-device-code=true`， 否则编译时出现链接错误“ptxas fatal”.

