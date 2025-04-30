# KeepLearning
用于记载常见算子在cpu和gpu平台上部署时的性能优化方法。

## 工程代码结构
```
project       
├─ include  
│  ├─ cpu    
│  └─ cuda    
├─ src   
|  ├─ ops  
|  |  ├─ attention  
|  |  ├─ matmul     
|  |  ├─ reduce     
|  |  └─ softmax      
└─ utils   
```  

## 编译
一键编译     
```
cd src/op/*** 
bash build.sh
```  
单个算子编译     
```
cd src/op/*** 
bash ../../../build.sh
```
可执行文件位于工程目录下`bin`文件夹。