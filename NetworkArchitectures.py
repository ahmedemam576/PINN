import sys
sys.path.append('..')
import argparse
import os
import numpy as np
import math
import time
import torch
import torch.nn as nn
import torch.autograd
import torch.optim as optim
import scipy.io
from torch.autograd import Variable
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import  matplotlib.pyplot as plt
from scipy.interpolate import griddata
import torch.nn.functional as F
import h5py
import time
from torch.utils.data import Dataset
from activation.tanh import TrainableTanh
from util.mixture_of_experts.moe import MoE

import torch.utils.data.distributed

USE_TRAINABLE_TANH = False


class HelmholtzNet(nn.Module):
    def __init__(self, lb, ub, numFeatures = 100, numLayers = 8, numInFeatures = 4, useGPU = True, activation = torch.tanh):
        """
        This function creates the components of the Neural Network and saves the datasets
        :param x0: Position x at time zero
        :param u0: Real Part of the solution at time 0 at position x
        :param v0: Imaginary Part of the solution at time 0 at position x
        :param tb: Time Boundary
        :param X_f: Training Data for partial differential equation
        :param layers: Describes the structure of Neural Network
        :param lb: Value of the lower bound in space
        :param ub: Value of the upper bound in space
        """
        
        torch.manual_seed(1234)
        super(HelmholtzNet, self).__init__()

        self.numInFeatures = numInFeatures
        self.numFeatures = numFeatures
        self.numLayers = numLayers 
        self.lin_layers = nn.ModuleList()
        self.lb = torch.from_numpy(lb).float()
        self.ub = torch.from_numpy(ub).float()
        
        if(useGPU):
            self.dtype = torch.cuda.FloatTensor
            self.dtype2 = torch.cuda.LongTensor
            self.lb = self.lb.cuda()
            self.ub = self.ub.cuda()
        else:
            self.dtype = torch.FloatTensor
            self.dtype2 = torch.LongTensor

        #self.activation = lambda x: torch.log(1+torch.exp(x)) 
        #self.activation = lambda x: torch.log(1+torch.mul(x,torch.exp(x))) 
        #self.activation = lambda x: torch.max(torch.zeros_like(x),  torch.log(1+torch.mul(x,torch.exp(x))))
        self.activation = activation
        
        if(USE_TRAINABLE_TANH):
            self.activation = nn.ModuleList()
        
        self.init_layers()

 
    def init_layers(self):
        """
        This function creates the torch layers and initialize them with xavier
        :param self:
        :return:
        """
        self.lin_layers.append(nn.Linear(4, self.numFeatures))
        for _ in range(self.numLayers):
            self.lin_layers.append(nn.Linear(self.numFeatures, self.numFeatures))
            if(USE_TRAINABLE_TANH):
                self.activation.append(TrainableTanh(n=10))
            
        self.lin_layers.append(nn.Linear(self.numFeatures, 1))

        for m in self.lin_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
               
         
    def net(self, x, y, z, t):
        """
        Function that calculates the nn output at postion x at time t
        :param x: position
        :param t: time
        :return: Solutions and their gradients
        """

        dim = x.shape[0]
        
        x = Variable(x,requires_grad=True)
        y = Variable(y,requires_grad=True)
        z = Variable(z,requires_grad=True)
        t = Variable(t,requires_grad=True)
        X = torch.cat([x, y, z, t], 1)
        Ex = torch.squeeze(self.forward(X))  
                
        # compute partial derivatives
        grads = torch.ones([dim]).cuda()
        Ex_pds = torch.autograd.grad(Ex, [x,y,z,t], create_graph=True,grad_outputs=grads)
        Ex_x = Ex_pds[0].reshape([dim])
        Ex_y = Ex_pds[1].reshape([dim])
        Ex_z = Ex_pds[2].reshape([dim])
        Ex_t = Ex_pds[3].reshape([dim])
        
        Ex_xx = torch.autograd.grad(Ex_x, x, create_graph=True, grad_outputs=grads)[0]
        del Ex_x
        Ex_yy = torch.autograd.grad(Ex_y, y, create_graph=True, grad_outputs=grads)[0]
        del Ex_y
        Ex_zz = torch.autograd.grad(Ex_z, z, create_graph=True, grad_outputs=grads)[0]
        del Ex_z
        Ex_tt = torch.autograd.grad(Ex_t, t, create_graph=True, grad_outputs=grads)[0]
        del Ex_t
        
        return Ex, Ex_tt, Ex_xx, Ex_yy, Ex_zz
    

    def forward(self, x):
        """
        This function is the forward of a simple multilayer perceptron
        """
        #normalize input in range between -1 and 1 for better training convergence
        x = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        for i in range(0, len(self.lin_layers) - 1):
            x = self.lin_layers[i](x)
            if(USE_TRAINABLE_TANH):
                x = self.activation[i](x)
            else:
                x = self.activation(x)
        return self.lin_layers[-1](x)
        
    
    def net_pde(self, x, y, z, t):
        """
        Calculates the qualit of the pde
        :param x postion x
        :param t time t
        """
        Ex, Ex_tt, Ex_xx, Ex_yy, Ex_zz = self.net(x, y, z, t)
        
        f = Ex_xx + Ex_yy + Ex_zz - Ex_tt

        return Ex, f


    def solution_loss(self, x, y, z, t, gt):
        # forward
        inputX = torch.stack([x, y, z, t], 1)       
        E_t0 = self.forward(inputX)
        loss = torch.mean((E_t0 - gt)**2)

        return loss
    

    def pde_loss(self, x, y, z, t, gt, xf,yf,zf,tf, w = 1):
        # forward
        xx = torch.cat([x,xf.view(-1,1)])
        yy = torch.cat([y,yf.view(-1,1)])
        zz = torch.cat([z,zf.view(-1,1)])
        tt = torch.cat([t,tf.view(-1,1)])

        noElementsAtT0 = x.shape[0]

        Ex, f = self.net_pde(xx, yy, zz, tt)

        E_t0 = Ex[:noElementsAtT0]
        f = f[noElementsAtT0:]
        
        sz_y = (torch.max(y) - torch.min(y) + 1).type(self.dtype2)
        sz_z = (torch.max(z) - torch.min(z) + 1).type(self.dtype2)
        
        loss = w * torch.mean( (E_t0 - gt)**2 ) + torch.mean( f ** 2 ) # torch.abs(f))
        
        return loss


class GatedHelmholtzNet(HelmholtzNet):
    def __init__(self, numLayers, numFeatures, lb, ub, inputSize, useGPU, numExperts = 10, k = 2, nonlinear = True, noisy_gating = False, activation = torch.tanh, outputSize = 1, useFourNet = False):
        super().__init__(lb, ub, numFeatures, numLayers, inputSize)
        self.moe_layer = MoE(inputSize, outputSize, numExperts, numFeatures, numLayers, self.lb, self.ub, activation,non_linear=nonlinear, noisy_gating=noisy_gating, k=k, useFourNet = useFourNet)      

    def init_layers(self):
        return


    def forward(self, x, train=True, loss_coef=1):
        x = torch.squeeze(x)
        x = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        output, load_balance_loss = self.moe_layer(x,train,loss_coef)
        self.load_balance_loss = load_balance_loss
        return output
    
    
    def solution_loss(self, x, y, z, t, gt):
        return super().solution_loss(x,y,z,t,gt) + self.load_balance_loss

    
    def pde_loss(self, x, y, z, t, gt, xf,yf,zf,tf, w = 1):
        return super().pde_loss(x, y, z, t, gt, xf,yf,zf,tf, w) + self.load_balance_loss
    
    
class HelmholtzNetFour(HelmholtzNet):
    def __init__(self, lb, ub, inputSize = 4, numFeatures = 500, numLayers = 8, numInFeatures = 4, useGPU = True, activation = torch.relu):
        super().__init__(lb, ub, numFeatures, numLayers, inputSize)
        
    def init_layers(self):
        """
        This function creates the torch layers and initialize them with xavier
        :param self:
        :return:
        """
        self.in_x = nn.ModuleList()
        self.in_y = nn.ModuleList()
        self.in_z = nn.ModuleList()
        self.in_t = nn.ModuleList()
        lenInput = 1
        noInLayers = 3

        self.in_x.append(nn.Linear(lenInput,self.numFeatures))
        for _ in range(noInLayers):
            self.in_x.append(nn.Linear(self.numFeatures, self.numFeatures)) 

        self.in_y.append(nn.Linear(lenInput,self.numFeatures))
        for _ in range(noInLayers):
            self.in_y.append(nn.Linear(self.numFeatures, self.numFeatures)) 

        self.in_z.append(nn.Linear(lenInput,self.numFeatures))
        for _ in range(noInLayers):
            self.in_z.append(nn.Linear(self.numFeatures, self.numFeatures)) 

        self.in_t.append(nn.Linear(1,self.numFeatures))
        for _ in range(noInLayers):
            self.in_t.append(nn.Linear(self.numFeatures, self.numFeatures)) 

        for m in [self.in_x,self.in_y,self.in_z,self.in_t]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                #nn.init.constant_(m.bias, 0)
        
        
        self.lin_layers.append(nn.Linear(4 * self.numFeatures,self.numFeatures))
        for i in range(self.numLayers):
            inFeatures = self.numFeatures
            self.lin_layers.append(nn.Linear(inFeatures,self.numFeatures))
        inFeatures = self.numFeatures
        self.lin_layers.append(nn.Linear(inFeatures,1))
        for m in self.lin_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x_in):
        x = 2.0 * (x_in - self.lb) / (self.ub - self.lb) - 1.0    
        #x = (x_in - self.lb) / (self.ub - self.lb)
        x_inx = x_in[:,0].view(-1,1)
        x_iny = x_in[:,1].view(-1,1)
        x_inz = x_in[:,2].view(-1,1)
        x_int = x_in[:,3].view(-1,1)

        act_old = self.activation
        act_old = torch.relu
        self.activation = torch.sin


        for i in range(0,len(self.in_x)):
            x_inx = self.in_x[i](x_inx)
            x_inx = self.activation(x_inx)
        #x_inx = self.in_x[-1](x_inx)
        
        for i in range(0,len(self.in_y)):
            x_iny = self.in_y[i](x_iny)
            x_iny = self.activation(x_iny)
        #x_iny = self.in_y[-1](x_iny)

        for i in range(0,len(self.in_z)):
            x_inz = self.in_z[i](x_inz)
            x_inz = self.activation(x_inz)
        #x_inz = self.in_z[-1](x_inz)
            
        for i in range(0,len(self.in_t)):
            x_int = self.in_t[i](x_int)
            x_int = self.activation(x_int)
        #x_int = self.in_t[-1](x_int)
        
        x = torch.cat([x_inx,x_iny,x_inz,x_int],1).type(self.dtype)

        #self.activation = act_old

        for i in range(0,len(self.lin_layers)-1):
            x = self.lin_layers[i](x)
            x = self.activation(x)
        x = self.lin_layers[-1](x)
        
        #x = self.scalingConstant * x
        return x
