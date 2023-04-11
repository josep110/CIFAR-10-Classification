import torch
from torch import nn
from torch import tensor
from torch import transpose
from torch import cat
from torch import mul
from torch import add
from torch import flatten
from torch import rand
from torch import argmax
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10 # Allows access to CIFAR-10 Validation set.
from torchvision.transforms import ToTensor # Allows access to method for converting datset images to Tensors.
import my_utils as mu


CIFAR10_dataset_train = CIFAR10('./dataset', True, download=True, transform=ToTensor()) # pulls CIFAR-10 training set to dir 'dataset'. Each image converted to Tensor.
CIFAR10_dataset_test = CIFAR10('./dataset', False, transform=ToTensor())                # pulls CIFAR-10 testing set to dir 'dataset'. Each image converted to Tensor.

batchsize = 256

trainingLoader = DataLoader(CIFAR10_dataset_train, batchsize ,shuffle=True, pin_memory=True)  # Creates DataLoader for training data.
testingLoader = DataLoader(CIFAR10_dataset_test, batchsize, pin_memory=True)                  # Creates DataLoader for testing data.

class Block(nn.Module):

  def __init__(self, K, ins, conv_kernal):
    super(Block, self).__init__()
    self.K = K
    self.conv_kernal = conv_kernal
    self.ins = ins
    self.lin = nn.Linear(3,self.K)
    self.SpatialAveragePool = nn.AvgPool2d(self.ins) # takes global average of each input channel.
    self.g = nn.ReLU()                         # activation function applied to S.A.P output.

    self.Convs = nn.ModuleList()

    for i in range(self.K):
      self.Convs.append(nn.Conv2d(3,3,conv_kernal))

    


  def forward(self, X):
    a = self.SpatialAveragePool(X)   

    a = a.view([-1,3])


    a = self.lin(a)
    a = self.g(a)                     # vector a[ai .. ak]

    #o = mul(self.Convs[0](X),a[0][0])    # Conv(k)(X)*a(k)
    o = self.Convs[0](X)

   # print(o.shape)

    for i in range(1,self.K):
      o = add(o, self.Convs[i](X))    # iteritively sums up o.

    return o     





class CIFAR10_Model(nn.Module):
  def __init__(self, block_info):
    super(CIFAR10_Model,self).__init__()

    self.Blocks = nn.ModuleList()

    for i, (ins, conv_kernal) in enumerate(block_info):
      self.Blocks.append(Block(20, ins, conv_kernal))


  def forward(self, X):

    o = self.Blocks[0].forward(X)
    for bl in self.Blocks[1::]:
      o = bl.forward(o)

    o = nn.AvgPool2d(2,10)(o)
    o = o.view(-1,3)
    # print(o.shape)
    o = nn.Linear(3,10)(o)
    o = nn.Softmax(dim=0)(o)

    return o





block_info = ((32,4),(29,8),(22,11),(12,6),(7,6))
test = CIFAR10_Model(block_info)

loss = nn.CrossEntropyLoss() # mean squared error loss.
lr = 0.1 
optimizer = torch.optim.SGD(test.parameters(), lr=lr) # stochastic gradient descent optimizer.



num_epochs = 20

mu.train_ch3(test, trainingLoader, testingLoader, loss, num_epochs, optimizer)
