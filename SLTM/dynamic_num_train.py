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

#dynamical learning to look back

import random
import pandas as pd

random.seed()
x_dim=150
h_dim=150
h2_dim=50
h3_dim=25
inputx=[[0 for col in range(150)] for row in range(45000)]

pred_loss =nn.MSELoss()
stop_loss=nn.CrossEntropyLoss()

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
                if count<200:
                        inputx[day][reg2]=count/200
                else:
                        inputx[day][reg2]=1
inputx=Variable(torch.FloatTensor(inputx),requires_grad=False)

#print(inputx[34])


def min(a,b):
	if a<b:
		return a
	else:
		return b

def max(a,b):
        if a>b:
                return a
        else:
                return b

class stopsign(nn.Module):
    def __init__(self):
        super(stopsign, self).__init__()
        self.fc1=nn.Linear(x_dim+h_dim,h2_dim)
        self.fc2=nn.Linear(h2_dim,h3_dim)
        self.fc3=nn.Linear(h3_dim,2)
        self.relu=nn.ReLU()
        self.softmax=nn.Softmax()     
    
    def forward(self,x):
        ht=self.relu(self.fc1(x))
        ht2=self.relu(self.fc2(ht))
        ht3=self.softmax(self.relu(self.fc3(ht2)))
        return ht3


class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm = nn.LSTMCell(x_dim,h_dim)

    def forward(self, x):
        outputs = []
        outputsy = []
        h_t = Variable(torch.zeros(1,h_dim), requires_grad=False)
        c_t = Variable(torch.zeros(1,h_dim), requires_grad=False)
        i=0
        y=torch.FloatTensor([[0,1]])   
        while i<30:
            h_t, c_t = self.lstm(inputx[x-i-1], (h_t, c_t))
            y=ystop(torch.cat((h_t,inputx[x-i-1].view(1,150)),1))
            outputs = outputs+ [h_t]
            #outputsy =outputsy+ [y]
            i=i+1
            outh=h_t
            if (random.random()>y[0][0].data).numpy(): break

        for j in range(min(10,x-i)):
            h_t, c_t = self.lstm(inputx[x-i-1-j], (h_t, c_t))
            y=ystop(torch.cat((h_t,inputx[x-i-1-j].view(1,150)),1))
            outputs=outputs+ [h_t]
            #outputsy=outputsy+ [y]
        
        return outh,outputs,i+min(10,x-i)-1

def trainstopsign(x,outputs,size,target):
    minx=9999
    minloss=Variable(torch.FloatTensor([9999]))
    for i in range(size):
        loss=pred_loss(outputs[i],target)
        if (loss.data<minloss.data).numpy():
            minloss=loss
            minx=i

    for i in range(size):
        #p1=Variable(torch.LongTensor([0]),requires_grad=False)
        #p2=Variable(torch.LongTensor([1]),requires_grad=False)
        temp_input=Variable(torch.cat((outputs[i],inputx[x-i-1].view(1,150)),1).data,requires_grad=False)
        y=ystop(temp_input)
        if minx<i:
            loss=stop_loss(y,Variable(torch.LongTensor([0]),requires_grad=False))
            loss.backward()
            stop_optimizer.step()
            stop_optimizer.zero_grad()
        else:
            loss=stop_loss(y,Variable(torch.LongTensor([1]),requires_grad=False))
            loss.backward()
            stop_optimizer.step()
            stop_optimizer.zero_grad()
    return


def trainpred(out,target):
    loss=pred_loss(out,target)
    loss.backward()
    pred_optimizer.step()
    pred_optimizer.zero_grad()
    return


model=Sequence()
ystop=stopsign()
pred_optimizer=optim.SGD(model.parameters(),lr=0.5)
stop_optimizer=optim.Adam(ystop.parameters(),lr=1e-3)


xh,_,_=model(day-1)
x=pred_loss(xh,inputx[day-1]).data[0]
for i in range(day-41):
        xh,_,_=model(i)
        x=x+pred_loss(xh,inputx[i]).data[0]
print(x/(day-41))


for i in range(10000):
	t=random.randint(40,day)
	if not((t % 13)==0):
		pred_optimizer.zero_grad()
		stop_optimizer.zero_grad()
		out,outs,size=model(t)
		
		trainstopsign(t,outs,size,inputx[t])
		trainpred(out,inputx[t])

xh,_,_=model(day-1)
x=pred_loss(xh,inputx[day-1]).data[0]
for i in range(day-41):
        xh,_,_=model(i)
        x=x+pred_loss(xh,inputx[i]).data[0]
print(x/(day-41))

