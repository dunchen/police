from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#from sklearn import svm

import random
import pandas as pd

random.seed()
x_dim=150
h_dim=150
h2_dim=50
h3_dim=25
inputx=[[0 for col in range(150)] for row in range(45000)]

pred_loss =nn.MSELoss()
#stop_loss=nn.CrossEntropy()

fil=pd.read_csv('all_type_150_count_d_reg.csv')
n=41582-2
day=0
temp_time=20141001
for i in range(n):
	time=fil.ix[i+2,1].astype(int)
	if time!=temp_time:
		day=day+1
		temp_time=time
	reg=fil.ix[i+2,2]
	if not(pd.isnull(reg)):
		reg2=reg.astype(int)
		count=np.asscalar(fil.ix[i+2,3])
		#print(mon-10,reg2,count)
		inputx[day][reg2]=count/(count+1)
inputx=Variable(torch.FloatTensor(inputx),requires_grad=False).cuda()



class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm = nn.LSTMCell(x_dim,h_dim)

    def forward(self, x):
        #outputs,outputsy = []
        h_t = Variable(torch.zeros(1,h_dim), requires_grad=False).cuda()
        c_t = Variable(torch.zeros(1,h_dim), requires_grad=False).cuda()


        for i in range(10):
            h_t, c_t = self.lstm(inputx[x-(10-i)].cuda(), (h_t, c_t))
        return h_t
        

def trainpred(out,target):
    loss=pred_loss(out,target)
    loss.backward()
    pred_optimizer.step()
    return


model=Sequence().cuda()
pred_optimizer=optim.SGD(model.parameters(),lr=0.1)


x=pred_loss(model(day-1),inputx[day-1]).data[0]
for i in range(day-11): x=x+pred_loss(model(i+10),inputx[i+10]).data[0]
print(x/(day-10))

xt=pred_loss(model(13),inputx[13]).data[0]
for i in range(day): 
	if (i%13)==0: xt=xt+pred_loss(model(i),inputx[i]).data[0]
print(xt/(day/13))


for i in range(30000):
	t=random.randint(10,day)
	if not((t % 13)==0):
		out=model(t)
		pred_optimizer.zero_grad()
		trainpred(out,inputx[t])




x=pred_loss(model(day-1),inputx[day-1]).data[0]
for i in range(day-11): x=x+pred_loss(model(i+10),inputx[i+10]).data[0]
print(x/(day-10))

x=pred_loss(model(13),inputx[13]).data[0]
for i in range(day): 
	if (i%13)==0: x=x+pred_loss(model(i),inputx[i]).data[0]
print(x/(day/13))






'''
class stopsign(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.fc1=nn.Linear(x_dim+h_dim,h2_dim)
        self.fc2=nn.Linear(h2_dim,h3_dim)
        self.fc3=nn.Linear(h3_dim,2)
        self.relu=nn.ReLU()
        self.softmax()=nn.Softmax()     
    
    def forward(self,x):
        ht=self.relu(self.fc1(x))
        ht2=self.relu(self.fc2(ht))
        ht3=self.softmax(self.relu(self.fc3(ht2)))
        return ht3


------------------------------------
		out,outs,outsy,size=model(inputx[i])
		trainstopsign(outs,outsy,size,inputy[i])
        	trainpred(out,y[i])

---------------------------------

        y=0
        i=0
        while random.random()>y[0]:
            h_t, c_t = self.lstm(x[i], (h_t, c_t))
            y=ystop(torch.cat((h_t.data,x[i+1].data))
            outputs+= [h_t]
            outputsy+= [y]
            i=i+1
            outh=h_t

        for j in range(30):
            h_t, c_t = self.lstm(x[i+j], (h_t, c_t))
            y=ystop(torch.cat((h_t.data,x[i+1].data))
            outputs+= [h_t]
            outputsy+= [y]

        return outh,outputs,outputsy,i+29

--------------------------------------------
def trainstopsign(outputs,outputsy,size,target):
    minx=9999
    minloss=9999
    for i in range(size):
        loss=pred_loss(outputs[i],target)
        if loss<minloss:
            minloss=loss
            minx=i
    p1=torch.longTensor(1)
    p1[0]=1
    p1[1]=0
    p2=torch.longTensor(1)
    p2[0]=0
    p2[1]=1

    for i in range(size):
        if minx<i:
            loss=stop_loss(outputsy[i],p1)
            loss.backward()
            stop_optimizer.step()
        else:
            loss=stop_loss(outputsy[i],p2)
            loss.backward()
            stop_optimizer.step()
    return
'''

