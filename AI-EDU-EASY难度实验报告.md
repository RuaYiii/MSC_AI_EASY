---
title: AI-EDU-EASY的实验报告
date: 2020-12-2 21:30:12
tags: 实验报告
---

# 实验报告

## 准备工作

**使用的库：**

- numpy
- csv
- pathlib
- math  (这个实际没咋用到)

```python
import csv
import numpy as np
from pathlib import Path
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
```

##部分的代码：

**主函数：**

```python
if __name__ == "__main__":
    data,lable=init_data() #返回list
    data,lable= read_data(data,lable)#返回numpy对象
    data_new,data_norm,lable_new,lable_m,lable_min=normalize(data,lable)#数据标准化
    w,b=train(data_new,lable_new)#训练
    w,b=denormalizeweightsBias(w,b,data_norm)#还原参数值
    #差点东西 训练的还是调参的 呜呜呜
    w1,w2,b=test(data,lable,w,b,1,lable_m,lable_min)#测试 以及 输出最终模型
    Model_visualization(w1,w2,b)#模型可视化
```

**初始化数据模块：**

```python
def init_data():
    path="Dataset/mlm.csv" 
    csv_file=open(path,'r')
    xyz_set=csv.reader(csv_file)
    data=[]
    z_ls=[]
    k=0 
    pass
    for x,y,z in xyz_set:
        if k==0:
            k=1
            continue
        data.append([float(x),float(y)])
        z_ls.append(float(z))

    return data,z_ls
```

**标准化模块：**

```python
def normalize(data,lable):  #max-min标准化
    data_new=np.zeros(data.shape)
    feature_nm=data.shape[1]
    data_norm=np.zeros((data.shape[1],2))
    #lable标准化
    lable_new=np.zeros((feature_nm,1))
    lable_max=np.max(lable)
    lable_min=np.min(lable)
    lable_m=lable_max-lable_min
    lable_new=(lable-lable_min)/lable_m
    #标准化data
    for i in range(feature_nm):
        col_i=data[:,i]
        data_max=np.max(col_i)
        data_min=np.min(col_i)
        data_norm[i,0]=data_min
        data_norm[i,1]=data_max-data_min
        data_new[:,i]=(col_i-data_min)/data_norm[i,1]
    return data_new,data_norm,lable_new,lable_m,lable_min
```

**训练模块：**

```python
def train(data,lable):
    input_sz=2
    output_sz=1
    eps=0.1
    eta=0.001
    batch_size=5
    max_epoch=5000
    train_num=1000
    #第一次训练
    w=np.zeros((input_sz,output_sz))
    b=np.zeros((1,output_sz))
    
    max_iteration= math.ceil(train_num/batch_size)
    checkpoint=0.1 #检查点
    checkpoint_iteration = (int)(max_iteration * checkpoint)
    for epoch in range(max_epoch):
        for iteration in range(max_iteration):
            start=iteration*batch_size
            end=start+batch_size
            batch_x=data[start:end,:]  #data
            batch_y=lable[start:end:] #lable   淦哦 特殊情况
            batch_z=np.dot(batch_x,w)+b     #预测值
            #backwardBatch
            dz= batch_z - batch_y
            dw= np.dot(batch_x.T,dz)/batch_size
            db= dz.sum()/batch_size
            w-= dw*eta
            b-= db*eta
            
            total_iteration = epoch * max_iteration + iteration
            if (total_iteration+1) % checkpoint_iteration == 0:
                loss=((np.dot(data,w)+b)- lable)**2
                Loss=loss.sum()/1000/ 2 
                #print(f'dz={dz},loss={Loss} b={b}')
                #print(db)
    return w,b
```

**可视化模块：**

```python
def Model_visualization(w1,w2,b):
    x = np.linspace(0, 100, 10)
    y = np.linspace(0, 100, 10)
    x_mesh, y_mesh = np.meshgrid(x, y, indexing='ij')
    z_mesh = w1 * x_mesh + w2 * y_mesh + b   # 拟合平面方程
    path="Dataset/mlm.csv"

    fig = plt.figure(figsize=(10, 10))
    sub = fig.add_subplot(111, projection='3d')
    sub.plot_surface(x_mesh, y_mesh, z_mesh, color='0.999', alpha=0.4)  # 绘制平面方程
    sub.set_xlabel(r'$x$')
    sub.set_ylabel(r'$y$')
    sub.set_zlabel(r'$z$')
    sub.view_init(elev=45, azim=0)  # 初始观看角度
    x1, y1, z1 = np.loadtxt(path, delimiter=',', skiprows=1, usecols=(0, 1, 2), unpack=True)
    # 导入数据
    sub.scatter(x1, y1, z1, color='r', s=1)  # 绘制散点图
    plt.show()
```

##结果

最终的训练得到的函数以及可视化图像：
z= 4.021290227979238x + 4.021290227979238y + 4.310284011920089
![作业1_可视化](作业1_可视化.png)



使用线性回归方法，前期由于没有进行数据标准化，在训练过程中loss死活下降缓慢
之后直接好多了，这从我们之前写的main()中的注释可以看到
```
#经过第一次迭代后的值  a=1,b=1,c=0 loos=13.651262268662906
#经过第二次迭代后的值  a=395.3620899561479
                    #b=-373.9789681172605
                    #c=27.58882723274058  loss=1.0906218830518108
#经过第三次迭代后的值  a=401.9158795062455
                    #b=-376.35246212932987
                    #c=29.887500814868403  loss=1.0533662679415212
```

同时这次没有写类，很后悔，下一次作业应该不会这样了，没有类管理起来真的很麻烦。


