import numpy as np 
import csv
class NeuralNetHelper(object):
    def __init__(self,hp):
        self.hp=hp
        self.w=np.zeros(self.hp.input_sz,self.hp.ouput_sz)
        self.b=np.zeros(1,self,hp.output_sz)
    def forward_batch(self,x_batch):
        z= np.dot(x_batch,self.w)+self.b
        pass
        return z
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
        self.xraw=data[:,:-1]
        self.yraw=data[:,-1]
        self.xtrain=self.xraw
        self.ytrain=self.yraw
        self.num_train=self.xtrain.shape[0])
        #现在为止还都是str
        
if __name__ == "__main__":
    url="Dataset/iris.csv"
    data=reader(url)
    data.read_csv_data()