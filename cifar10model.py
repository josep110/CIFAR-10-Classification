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


CIFAR10_dataset_train = CIFAR10('./dataset', True, download=True, transform=ToTensor()) # pulls CIFAR-10 training set to dir 'dataset'. Each image converted to Tensor.
CIFAR10_dataset_test = CIFAR10('./dataset', False, transform=ToTensor())                # pulls CIFAR-10 testing set to dir 'dataset'. Each image converted to Tensor.

batchsize = 256

trainingLoader = DataLoader(CIFAR10_dataset_train, batchsize ,shuffle=True, pin_memory=True)  # Creates DataLoader for training data.
testingLoader = DataLoader(CIFAR10_dataset_test, batchsize, pin_memory=True)                  # Creates DataLoader for testing data.

class Block(nn.Module):

  def __init__(self, batchsize, K, ins, conv_kernal, channels=3):

    # Each block produces vector [a] of length K by : applying spatial average pooling to input tensor, 
    # , running output though linear layer and activation function.

    # Input tensor is run through K different convolution layers and multiplied with corresponding entry in a, then combined to produce output tensor. 

    super(Block, self).__init__()
    
    self.batchsize = batchsize
    self.K = K
    self.ins = ins
    self.conv_kernal = conv_kernal
    self.channels = channels

    self.map_dim = (self.ins - self.conv_kernal) + 1

    self.ins = ins
    self.lin = nn.Linear(self.channels,self.channels)
    self.SpatialAveragePool = nn.AvgPool2d(self.ins) # takes global average of each input channel.
    self.g = nn.ReLU()                         # activation function applied to S.A.P output.

    self.Convs = nn.ModuleList()

    for i in range(self.K):
      self.Convs.append(nn.Conv2d(self.channels, self.channels, conv_kernal))


  def forward(self, X):

    self.batchsize = X.shape[0]                                    # X = [batch, channels, ins, ins]
    a = self.SpatialAveragePool(X)                # [batch, channels, 1, 1]  
    a = a.view(self.batchsize, self.channels)                 # [batch, channels]   SpatialAveragePool(X)
    

    a = self.lin(a)                   # [batch, channels]  SpatialAveragePool(X)W  -> [a1, a2, a3]
    a = self.g(a)                     # [batch, channels]  g(SpatialAveragePool(X)W) -> [a1, a2, a3] = a
    a = a.view(self.batchsize, self.channels, 1)

    o_components = []

    for k in range(self.K):                           # This loop finds each ak*convk(X) that are combined to form block output signal.
      convk = self.Convs[k](X).view(self.batchsize, -1, 3)
      ak = a[:,k].view(self.batchsize,1,1)
      comp_k = mul(convk, ak)
      o_components.append(comp_k)     


    o = o_components[0]
    o = o.view(self.batchsize, self.channels, self.map_dim, self.map_dim)               # a1Con1(X)
    for i in range(1,self.K):
      o = add(o,o_components[i].view(self.batchsize, self.channels, self.map_dim, self.map_dim))  #  + ... akConvk(X)

    return o
    


class CIFAR10_Model(nn.Module):
  def __init__(self, batchsize, block_info):
    super(CIFAR10_Model,self).__init__()

    self.Blocks = nn.ModuleList()

    for i, (ins, conv_kernal) in enumerate(block_info):
      self.Blocks.append(Block(batchsize, 3, ins, conv_kernal))            # sets up Blocks.


  def forward(self, X):

    o = self.Blocks[0].forward(X) # output of first block

    for bl in self.Blocks[1::]:
      o = bl.forward(o)          # output of subsequent blocks up to final.

    last_block_outdim = o.shape[3]

    o = nn.AvgPool2d(last_block_outdim,10)(o)   # Pools output of last Block.
    o = o.view(-1,3)

    o = nn.Linear(3,10)(o)                      # Linear layer, converts output signal into vector length 10.
    o = nn.Softmax(dim=0)(o)                    # Softmax classification for 10 classes.

    return o


