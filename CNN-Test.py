import numpy as np
import numpy as np
import sys,os
sys.path.append(os.pardir)
from ch6.util import im2col
'''
input_data--由(数据量、通道、高、长)的4维数组构成的输入数据
filter_h:滤波器的高
filter_w:滤波器的长
stride:步幅
pad:填充
'''
x1=np.random.rand(1,3,3,7)
coll=im2col(x1,5,5,stride=1,pad=0)
print(coll.shape) #(9,75)

x2=np.random.rand(10,3,7,7)
col2=im2col(x2,5,5,stride=1,pad=0)
print(col2.shape)#(90,75)


class Convolution:
    def __init__(self,W,b,stride=1,pad=0):
        self.W=W
        self.b=b
        self.stride=stride
        self.pad=pad
    def forward(self,x):
        FN,C,FH,FW=self.W.shape
        N,C,H,W=x.shape
        out_h=int(1+(H+2*self.pad-FH)/self.stride)
        out_w=int(1+(W+2*self.pad-FW)/self.stride)
        
        col=im2col(x,FH,FW,self.stride,self.pad)
        col_W=self.W.reshape(FN,-1).T
        out=np.dot(col,col_W)+self.b
        
        out=out.reshape(N,out_h,out_w,-1).transpose(0,3,1,2)
        
        return out
x=np.random.rand(10,1,28,28)
print(x)
