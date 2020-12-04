import csv
import numpy as np
from pathlib import Path
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

def main():#不要管这个啊啊啊啊啊啊啊是瞎写的
    #已知x、y为自变量  z为应变量且存在线性关系 则设 z=a*x+b*y+c
    a=402.1325753661921
    b=-376.23482160130277
    c=30.061448262222637
    eta=0.05
    #经过第一次迭代后的值  a=1,b=1,c=0 loos=13.651262268662906
    #经过第二次迭代后的值  a=395.3620899561479
                        #b=-373.9789681172605
                        #c=27.58882723274058  loss=1.0906218830518108
    #经过第三次迭代后的值  a=401.9158795062455
                        #b=-376.35246212932987
                        #c=29.887500814868403  loss=1.0533662679415212

    x,y,z=init_data()
    x_max=max(x)
    y_max=max(y)
    x_min=min(x)
    y_min=min(y)
    x_norm=x_max-x_min
    y_norm=y_max-y_min
    for n in range(999) :
        x_new= (x[n]-x_max)/x_norm
        y_new= (y[n]-y_max)/y_norm
        z_f= a*x_new+b*y_new+c
        dz=z[n]-z_f 
        loss= (dz**2)/2
        a+=(dz*x_new)*eta
        b+=(dz*y_new)*eta
        c+=dz*eta
    print(loss)
    print(a,b,c)
def read_data(data,lable):  #读取转numpy数组
    data=np.array(data)
    lable=np.array([lable]).T  #导入不对劲，亲人两行泪
    return data,lable
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
def denormalizeweightsBias(w,b,data_norm):
    w[0,0]=w[0,0]/data_norm[0,1]
    w[1,0]=w[1,0]/data_norm[1,1]
    b[0,0]=b[0,0]-w[0,0]*data_norm[0,0]-w[1,0]*data_norm[1,0]
    return w,b
def test(data,lable,w,b,i,lable_m,lable_min):
    z=np.dot(data[i,:],w)+b
    w1=w[0,0]*lable_m
    w2=w[1,0]*lable_m
    c_=b[0,0]*lable_m+lable_min
    
    #print(f"z= {w[0,0]}x + {w[1,0]}y + {b[0][0]}= {z[0][0]}")
    print(f"呜呜呜 预测的 {(z[0][0]*lable_m)+lable_min}")
    print(f"呜呜呜 实际的 {lable[i][0]}")
    print(f'别人家的结果{4.021347*data[i,0]+-3.770851*data[i,1]+4.302547}')
    print("--------------------------")
    print(f"最终模型：z={w1}x{w1}y+{c_}")
    return w1,w2,c_
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
if __name__ == "__main__":
    data,lable=init_data() #返回list
    data,lable= read_data(data,lable)#返回numpy对象
    data_new,data_norm,lable_new,lable_m,lable_min=normalize(data,lable)#数据标准化
    w,b=train(data_new,lable_new)#训练
    w,b=denormalizeweightsBias(w,b,data_norm)#还原参数值
    #差点东西 训练的还是调参的 呜呜呜
    w1,w2,b=test(data,lable,w,b,1,lable_m,lable_min)#测试 以及 输出最终模型
    Model_visualization(w1,w2,b)#模型可视化
    pass