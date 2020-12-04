import numpy as np 
import math
class HyperParameters(object):  #超参
    def __init__(self,input_sz,output_sz,eta=0.1,max_epoch=1000,batch_sz=5,
    eps=0.1,net_type=3):
        self.input_sz=input_sz
        self.output_sz=output_sz
        self.eta=eta
        self.eps=eps
        self.max_epoch= max_epoch
        self.batch_sz=batch_sz
        self.net_type= net_type

class NeuralNetHelper(object):
    def __init__(self,params):
        self.params=params
        self.w=np.zeros((self.params.input_sz,self.params.output_sz))
        self.b=np.zeros((1,self.params.output_sz))
    def forward_batch(self,x_batch):
        z= np.dot(x_batch,self.w)+self.b
        return z
    def train__Classification(self,dataread,checkpoint=0.1): #分类训练
        loss_func=loss_fn(self.params.net_type)
        loss=10
        if self.params.batch_sz==-1:
            pass
        max_iteration= math.ceil(dataread.num_train/self.params.batch_sz)
        checkpoint_iteration= int(max_iteration*checkpoint)
        for epoch in range(self.params.max_epoch):
            for iteration in range(max_iteration):
                batch_x,batch_y=dataread.get_batch_train(self.params.batch_sz,iteration)
                pass
                
class loss_fn(object): #loss函数  
    def __init__(self,net_type):
        self.net_type= net_type
    def checkloss(self,A,Y):
        Fitting = 1,
        BinaryClassifier = 2,
        MultipleClassifier = 3
        m=Y.shape[0]
        if self.net_type== Fitting:
            pass
        elif self.net_type== BinaryClassifier:
            pass
        elif self.net_type== MultipleClassifier: #这次是这个
            pass
class reader(object):
    def __init__(self,path):
        self.path=path
        self.num_train=0
        self.xtrain=None
        self.ytrain=None
        self.xraw=None
        self.yraw=None
    def read_csv_data(self):
        #file=csv.reader(self.path)
        data= np.loadtxt(self.path,delimiter=",",dtype="str",skiprows=1)
        #默认情况下，数据被认为是float类型，因此，我全转str
        self.xraw=data[:,:-1].astype("float")
        self.yraw=data[:,-1]
        self.xtrain=self.xraw
        self.ytrain=self.yraw
        self.num_train=self.xtrain.shape[0]
        self.label_dict={"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":2}
    def normalize_train_x(self):
        x_new=np.zeros(self.xraw.shape)
        num_feature=self.xraw.shape[1]
        self.x_norm = np.zeros((num_feature,2))
        for i in range(num_feature):
            self.x_norm[i,0]=np.min(self.xraw[:,i])
            self.x_norm[i,1]=np.max(self.xraw[:,i])-np.min(self.xraw[:,i])
            x_new[:,i]=(self.xraw[:,i]-self.x_norm[i,0])/self.x_norm[i,1]
        self.xtrain= x_new
    def normalize_train_y(self): #这里分类就不需要了，所以我改了且合并了To_onehot
        dicts=self.label_dict
        y_new=np.zeros((self.yraw.shape[0],3))
        for i in range(self.yraw.shape[0]):
            y_new[i,dicts[self.yraw[i]]]=1
        self.yraw=y_new
    def normalize_predictdata(self,x_p): #尺度问题————待解决
        x_new_p=np.zeros(x_p.shape)
        for i in range(x_p.shape[1]):
            x_new_p[:,i]=(x_p[:,i]-self.x_norm[i,0])/self.x_norm[i,1]
        return x_new_p
        
    def To_onehot(self,num_category,base=0):
        count=self.yraw.shape[0]
        self.num_category=num_category
        y_new= np.zeros((count,self.num_category))
        for i in range(count):
            n=int(self.yraw[i,0])
            y_new[i,base-n]=1
        return y_new
    def get_batch_train(self,batch_sz,iteration):
        start= iteration*batch_sz
        end=start+batch_sz
        batch_x=self.xtrain[start:end,:]
        batch_y=self.ytrain[start:end,:]
        return batch_x,batch_y
    def shuffle(self): #样本打乱
        seed= np.random.randint(0,100)
        np.random.seed(seed) #设置seed相同 保证标签值是一一对应的
        xp=np.random.permutation(self.xtrain)
        yp=np.random.permutation(self.ytrain)
        self.xtrain= xp
        self.ytrain= yp

class softmax(object):
    def forward(self,z):
        shift_z= z-np.max(z,axis=1,keepdims=True)
        exp_z=np.exp(shift_z)
        a= exp_z/np.sum(exp_z,axis=1,keepdims=True)
        return a 


if __name__ == "__main__":
    url="Dataset/iris.csv"
    data=reader(url)
    data.read_csv_data()
    data.normalize_train_x()
    data.normalize_train_y()
    params= HyperParameters(input_sz=3,output_sz=3,eta=0.1, max_epoch=100, batch_sz=10,
    eps=1e-3,net_type=3)
    net=NeuralNetHelper(params)
    net.train__Classification(data)