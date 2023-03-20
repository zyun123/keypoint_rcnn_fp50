import torch
import torch.nn.functional as F
import math

from transformers import LogitsProcessorList

input = torch.randn(5,requires_grad=True)
target = torch.rand(5).random_(2)

print(input)
print(target)

#binary cross entropy 
loss = F.binary_cross_entropy(torch.sigmoid(input),target,reduction = "sum") #默认求均值
print("binary cross entropy loss: " ,loss)

#my binary cross entropy loss
loss1 = 0
input1 = torch.sigmoid(input)
for i in range(5):
    loss1+= -(target[i]*torch.log(input1[i])+(1-target[i])*torch.log(1-input1[i]))

print("my binary cross entropy loss: " ,loss1)



#cross entropy 
output = torch.randn(3,5,requires_grad=True)
label = torch.randint(5,(3,),dtype=torch.int64)
print("output\n",output)
print("label\n",label)
loss_mean = F.cross_entropy(output,label)
loss = F.cross_entropy(output,label,reduction="none")
print("cross entropy loss sum:  " ,loss)
print("cross entropy loss mean:  " ,loss_mean)

#my cross entropy
output = torch.softmax(output,dim=1)

loss1 = 0
for i in range(len(label)):
    loss1+= -torch.log(output[i][label[i]])

print("my cross entropy :", loss1)
