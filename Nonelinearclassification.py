import numpy as np 
import math
from enum import Enum
import matplotlib.pyplot as plt
import os
from pathlib import Path

class NetType(Enum):
    Fitting = 1
    BinaryClassifier = 2
    MultipleClassifier = 3
class InitialMethod(Enum):
    Zero = 0,
    Normal = 1,
    Xavier = 2,
    MSRA = 3
class HyperParameters(object):  #超参
    def __init__(self,n_input,n_output,n_hidden,eta=0.1,max_epoch=10000,batch_sz=5,
    eps=0.1,net_type=NetType.Fitting,init_method= InitialMethod.Xavier):
        self.num_input = n_input
        self.num_hidden = n_hidden
        self.num_output = n_output
        self.eta = eta
        self.max_epoch = max_epoch

        self.batch_sz = batch_sz
        self.net_type = net_type
        self.init_method = init_method
        self.eps = eps
class NeuralNet(object):
    def __init__(self,hp,nm):
        self.hp=hp
        self.model_name=nm
        self.subfolder=os.getcwd()+"\\"+self.__create_subfloder()
        #print(self.subfolder)  打印路径

        #self.w=np.zeros((self.hp.n_input,self.hp.n_output))
        #self.b=np.zeros((1,self.hp.n_output))

        self.wb1=WB(self.hp.num_input, self.hp.num_hidden, self.hp.init_method, self.hp.eta)
        self.wb1.init_w(self.subfolder, False)
        self.wb2=WB(self.hp.num_hidden, self.hp.num_output, self.hp.init_method, self.hp.eta)
        self.wb2.init_w(self.subfolder, False)

    def __create_subfloder(self):
        if self.model_name!=None:
            path=self.model_name.strip()
            path= path.rstrip("\\")
            isExists = os.path.exists(path)
            if not isExists:
                os.makedirs(path)
            return path
    def load_result(self):
        self.wb1.load_result_v(self.subfolder,"wb1")
        self.wb2.load_result_v(self.subfolder,"wb2")
    def train(self,dataread,checkpoint,need_test): #分类训练
        self.loss_trace= train_history() #历史记录
        self.loss_func= loss_func(self.hp.net_type) #loss对象
        if self.hp.batch_sz==-1:
            self.hp.batch_sz = dataread.num_train
        max_iteration= math.ceil(dataread.num_train/self.hp.batch_sz) #最大迭代数
        checkpoint_iteration= int(max_iteration*checkpoint)
        for epoch in range(self.hp.max_epoch):
            #dataread.shuffle()
            for iteration in range(max_iteration):
                batch_x,batch_y=dataread.get_batch_train(self.hp.batch_sz,iteration)
                batch_a= self.forward(batch_x)
                #正向传播结束
                self.backward(batch_x,batch_y,batch_a)
                #反向传播结束()
                self.updata()# 
                total_iteration = epoch * max_iteration + iteration #计算迭代数                
                if (total_iteration+1)% checkpoint_iteration == 0:
                    self.checkloss(dataread,batch_x,batch_y,epoch,total_iteration)
        self.save_result()
        print("end")
    def forward(self,batch_x):
        self.z1= np.dot(batch_x,self.wb1.w)+self.wb1.b #计算z1
        self.a1= Sigmoid().forward(self.z1) #计算a1
        self.z2= np.dot(self.a1,self.wb2.w)+self.wb2.b  #计算z2
        if self.hp.net_type ==NetType.BinaryClassifier:
            print("哈哈哈哈我没做二元分类")
        elif self.hp.net_type ==NetType.MultipleClassifier:
            self.a2= softmax().forward(self.z2) 
        else:
            self.a2= self.z2
        self.output= self.a2 #最终输出
    def backward(self,batch_x,batch_y,batch_a): #这个要注意
        m= batch_x.shape[0] #样本个数  
        dz2=self.a2-batch_y
        self.wb2.dw= np.dot(self.a1.T,dz2)#除以m防止梯度爆炸
        self.wb2.db= np.sum(dz2,axis=0, keepdims=True)/m
        #对于多样本计算，需要在横轴上做sum，得到平均值
        d1= np.dot(dz2, self.wb2.w.T)
        dz1,_ = Sigmoid().backward(None, self.a1, d1) #?????
        self.wb1.dw= np.dot(batch_x.T, dz1)/m
        self.wb1.db= np.sum(dz1)/m
    def updata(self):
        self.wb1.updata()
        self.wb2.updata()
    def checkloss(self,datareader,train_x,train_y,epoch,total_iteration):
        print(f"epoch={epoch}, total_iteration={total_iteration}")
        self.forward(train_x)
        #计算损失
        loss_train= self.loss_func.checkloss(self.output,train_y)
        accuracy_train= self.__cal_accuracy(self.output,train_y)
        print("loss_train=%.6f, accuracy_train=%f" %(loss_train, accuracy_train))
        #验证损失
        yld_x,yld_y= datareader.GetValidationSet()
        self.forward(yld_x)
        loss_vld= self.loss_func.checkloss(self.output,yld_y)
        accuracy_vld= self.__cal_accuracy(self.output,yld_y)
        print("loss_valid=%.6f, accuracy_valid=%f" %(loss_vld, accuracy_vld))        
        print(f"loss = {loss_train}")
    def __cal_accuracy(self, a, y):
        assert(a.shape == y.shape)
        m = a.shape[0]
        if self.hp.net_type == NetType.Fitting:
            var = np.var(y)
            mse = np.sum((a-y)**2)/m
            r2 = 1 - mse / var
            return r2
        elif self.hp.net_type == NetType.BinaryClassifier:
            b = np.round(a)
            r = (b == y)
            correct = r.sum()
            return correct/m
        elif self.hp.net_type == NetType.MultipleClassifier:
            ra = np.argmax(a, axis=1)
            ry = np.argmax(y, axis=1)
            r = (ra == ry)
            correct = r.sum()
            return correct/m
    def save_result(self):
        self.wb1.save_result_v(self.subfolder,"wb1")
        self.wb2.save_result_v(self.subfolder,"wb2")
class CActivator(object):
    # z = 本层的wx+b计算值矩阵
    def forward(self, z):
        pass
    # z = 本层的wx+b计算值矩阵
    # a = 本层的激活函数输出值矩阵
    # delta = 上（后）层反传回来的梯度值矩阵
    def backward(self, z, a, delta):
        pass
class Sigmoid(CActivator):
    def forward(self, z):
        a = 1.0 / (1.0 + np.exp(-z))
        return a

    def backward(self, z, a, delta):
        da = np.multiply(a, 1-a)
        dz = np.multiply(delta, da)
        return dz, da        
class Logistic(CActivator): #Logistic
    def forward(self, z):
        a = 1.0 / (1.0 + np.exp(-z))
        return a

class Softmax(CActivator): #Softmax
    def forward(self, z):
        shift_z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shift_z)
        a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return a
class train_history(object):
    def __inin__(self):
        print("train_history还没有写！！！")               
class loss_func(object): #loss函数  
    def __init__(self,net_type):
        self.net_type= net_type
    def checkloss(self,A,Y):
        m=Y.shape[0]
        if self.net_type== NetType.Fitting:
            print("Fitting没写！")
        elif self.net_type== NetType.BinaryClassifier:
            print("BinaryClassifier没写！")
        elif self.net_type== NetType.MultipleClassifier: #这次是这个
            loss= self.CE3(A,Y,m) 
        return loss
    def CE3(self, A, Y, count):
        p1 = np.log(A)
        p2 =  np.multiply(Y, p1)
        LOSS = np.sum(-p2) 
        loss = LOSS / count
        return loss
class reader(object): 
    def __init__(self,path):
        self.path=path
        self.num_train=0
        self.num_category=0
        self.num_feature=0
        self.xtrain=None
        self.ytrain=None
        self.xraw=None
        self.yraw=None
        self.xtest_raw=None
        self.ytest_raw=None
        self.xtest=None
        self.ytest=None
    def read_csv_data(self):  #质检合格
        #file=csv.reader(self.path)
        data= np.loadtxt(self.path,delimiter=",",dtype="str",skiprows=1)
        #默认情况下，数据被认为是float类型，因此，我全转str
        self.xraw=data[:,:-1].astype("float")
        self.yraw=data[:,-1:]
        self.xtrain=self.xraw
        self.ytrain=self.yraw
        self.num_train=self.xtrain.shape[0]
        self.num_feature=self.xtrain.shape[1]
        self.num_category=len(np.unique(self.ytrain))#unique挺好用,下面这个就可以注释掉了
        self.label_dict={"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":2}
        #好家伙，我自己分训练集
        self.test_head=int(self.num_train/4)
        self.test_end=int(self.num_train/2)
        self.xtest_raw=self.xraw[self.test_head:self.test_end,:]
        self.ytest_raw=self.yraw[self.test_head:self.test_end,:]
        self.xtest=self.xtest_raw
        self.ytest=self.ytest_raw
    def __normalize_train_x(self,raw_data):
        x_new=np.zeros_like(raw_data)
        self.x_norm = np.zeros((2,self.num_feature))
        for i in range(self.num_feature):
            x=raw_data[:,i]  
            max_v=np.max(x)
            min_v=np.min(x)
            self.x_norm[0,i]=min_v
            self.x_norm[1,i]=max_v-min_v
            x_new[:,i]= (x - self.x_norm[0,i])/self.x_norm[1,i]
        return x_new
    def normalize_x(self):
        x_merge= np.vstack((self.xraw,self.xtest_raw))
        x_merge_norm= self.__normalize_train_x(x_merge)
        train_count= self.xraw.shape[0]
        self.xtrain= x_merge_norm[:train_count,:]
        self.xtest= x_merge_norm[train_count:,:]

    def normalize_y(self,type,base=0):
        '''if type == NetType.Fitting:
            y_merge = np.vstack((self.YTrainRaw, self.YTestRaw))
            y_merge_norm = self.__NormalizeY(y_merge)
            train_count = self.YTrainRaw.shape[0]
            self.YTrain = y_merge_norm[0:train_count,:]
            self.YTest = y_merge_norm[train_count:,:]                
        elif type == NetType.BinaryClassifier:
            self.YTrain = self.__ToZeroOne(self.YTrainRaw, base)
            self.YTest = self.__ToZeroOne(self.YTestRaw, base)'''
        if type == NetType.MultipleClassifier:  #我们这次重点关注 类型为MultipleClassifier的类型
            self.ytrain = self.__To_onehot(self.yraw, base)
            self.ytest = self.__To_onehot(self.ytest_raw, base)
    def normalize_predictdata(self,x_p): #尺度问题————待解决
        x_new_p=np.zeros(x_p.shape)
        for i in range(x_p.shape[1]):
            x_new_p[:,i]=(x_p[:,i]-self.x_norm[i,0])/self.x_norm[i,1]
        return x_new_p
        
    def __To_onehot(self,Y,base=0):
        count=Y.shape[0]
        y_new= np.zeros((count,self.num_category))
        for i in range(count):
            n=self.label_dict[Y[i,0]]
            y_new[i,base-n]=1
        return y_new
    def get_batch_train(self,batch_sz,iteration):
        start= iteration*batch_sz
        end=start+batch_sz
        batch_x=self.xtrain[start:end,:]
        batch_y=self.ytrain[start:end,:]
        return batch_x,batch_y
    def shuffle(self): #这就是样本打乱吗 爱了爱了
        seed= np.random.randint(0,100)
        np.random.seed(seed) #设置seed相同 保证标签值是一一对应的
        xp=np.random.permutation(self.xtrain)
        yp=np.random.permutation(self.ytrain)
        self.xtrain= xp
        self.ytrain= yp
    def GenerateValidationSet(self,k=10):
        self.num_validation = (int)(self.num_train / k)
        self.num_train = self.num_train - self.num_validation
        # validation set
        self.XDev = self.xtrain[0:self.num_validation]
        self.YDev = self.ytrain[0:self.num_validation]
        # train set
        self.XTrain = self.xtrain[self.num_validation:]
        self.YTrain = self.ytrain[self.num_validation:]
    def GetValidationSet(self):
        return self.XDev, self.YDev
class softmax(object):
    def forward(self,z):
        shift_z= z-np.max(z,axis=1,keepdims=True)
        exp_z=np.exp(shift_z)
        a= exp_z/np.sum(exp_z,axis=1,keepdims=True)
        return a 
class WB(object): #WeightsBias
    def __init__(self,n_input, n_output, init_method, eta):
        self.num_input = n_input
        self.num_output = n_output
        self.init_method = init_method
        self.eta = eta
        self.init_filename=f"w_{self.num_input}_{self.num_output}_{self.init_method}_init"
        self.dw=None
        self.db=None    
    def init_w(self,folder,create_new): 
        self.folder=folder
        if create_new:
            self.__create_new()
        else:
            self.__load()
        #self.dw=np.zeros(self.w.shape)
        #self.db=np.zeros(self.b.shape)
    def __create_new(self): #整合创建并储存w,b
        self.w,self.b= self.InitialParameters(self.num_input,self.num_output,self.init_method)
        self.__save()
        print(self.num_input,self.num_output)
    def __save(self): #存储w,b   
        file_name=f"{self.folder}/{self.init_filename}.npz"
        np.savez(file_name,weight=self.w,bias= self.b)
    def __load(self): #读取 要是无就继续创建
        file_name= f"{self.folder}/{self.init_filename}.npz"
        w_file=Path(file_name)
        if w_file.exists():
            #print("你 __LoadInitialValue() 还没写")  
            self.__LoadInitialValue()
        else:
            self.__create_new()
    def __LoadInitialValue(self): # .npz的格式 得解决 
        file_name= f"{self.folder}/{self.init_filename}.npz"
        data=np.load(file_name)
        self.w = data["weight"]
        self.b = data["bias"]
    def load_result_v(self,folder,name):  #加载 错乱
        file_name= f"{folder}/{name}.npz"
        data=np.load(file_name)
        self.w = data["weight"]
        self.b = data["bias"]
    def updata(self):
        self.w= self.w-self.eta*self.dw
        self.b= self.b-self.eta*self.db
    def save_result_v(self,folder,name):
        file_name=f"{folder}/{name}.npz"
        np.savez(file_name,weight=self.w,bias=self.b)
    @staticmethod 
    def InitialParameters(num_input, num_output, method): #返回w,b
        if method == InitialMethod.Zero:
            # zero
            W = np.zeros((num_input, num_output))
        elif method == InitialMethod.Normal:
            # normalize
            W = np.random.normal(size=(num_input, num_output))
        elif method == InitialMethod.MSRA:
            W = np.random.normal(0, np.sqrt(2/num_output), size=(num_input, num_output))
        elif method == InitialMethod.Xavier:
            # xavier
            W = np.random.uniform(-np.sqrt(6/(num_output+num_input)),
                                np.sqrt(6/(num_output+num_input)),
                                size=(num_input, num_output))
        # end if
        B = np.zeros((1, num_output))
        return W, B
if __name__ == "__main__":
    url="Dataset/iris.csv"
    data=reader(url)
    data.read_csv_data()
    data.normalize_y(NetType.MultipleClassifier, base=1) #
    
    #fig= plt.figure(figsize=(6,6))
    data.normalize_x() #这个地方让xtrain的shape变了
    #data.shuffle()
    data.GenerateValidationSet()  
    n_input= data.num_feature
    n_ouput= data.num_category
    n_hidden=3  
    eta,batch_sz,max_epoch=0.1,20,5000
    eps=3
    hp=HyperParameters(n_input,n_ouput,n_hidden,eta,max_epoch,batch_sz,eps,
    NetType.MultipleClassifier, InitialMethod.Xavier)
    net= NeuralNet(hp,"I_don't_know")   
    net.load_result() 
    net.train(data,100,True)
    #可视化
