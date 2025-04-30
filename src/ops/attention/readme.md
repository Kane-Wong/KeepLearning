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

