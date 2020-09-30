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

import torch.utils.data.distributed

DEBUGMODE = False

def loadFrame(pFile,dataset):
    idxL = pFile.find('simData_')+8
    idxR = pFile.find('.h5')
    frameIdx = pFile[idxL:idxR]
    dataset = '/data/' + frameIdx + "/" + dataset
    print("Loading " + dataset)
    hf = h5py.File(pFile, 'r')
    E = np.array(hf.get(dataset))
    
    dx = hf['/data/{}'.format(frameIdx)].attrs['cell_width']
    dy = hf['/data/{}'.format(frameIdx)].attrs['cell_height']
    dz = hf['/data/{}'.format(frameIdx)].attrs['cell_depth']
    
    #normConstant =  hf['/data/2000/fields/E/x'].attrs['unitSI']
    hf.close()
    return E, dx, dy, dz



class HelmholtzEquationBoxedDataset(Dataset):

    #hqd = HelmholtzEquationDataset(pData=pData, pResults=pResults, box_width = args.boxwidth, noSteps = 1) # pretraining dataset    
    
    def __init__(self,pData, useGPU = True, pResults = "", box_width = 100, noSteps = 100):
        dataForFrame = np.sort(os.listdir(pData))
        # Load data for t0
        Exact, dx, dy, dz = loadFrame(pData + dataForFrame[0], 'fields/E/x')
        
        #Exact = Exact[80:90,:,90:160]
        nx,ny,nz = Exact.shape
        print("Shape of initial value: ", Exact.shape)    

        # Domain bounds
        self.pdeLossBatchSizeFactor = 2
        self.lb = np.array([0, 0, 0, 0])
        self.ub = np.array([nx * dx,
                            ny * dy,
                            nz * dz,
                            1
                  ])
        if(useGPU):
            self.dtype = torch.cuda.FloatTensor
            self.dtype2 = torch.cuda.LongTensor
        else:
            self.dtype = torch.FloatTensor
            self.dtype2 = torch.LongTensor
            
        self.Exact = (self.dtype(Exact))
        
        self.excitingSlices = []

        self.noSteps = noSteps
        self.dt = 1.39e-16 #PIC step size

        self.tmax = self.noSteps * self.dt
        self.box_width = box_width
        self.no_boxes = int(np.ceil(self.Exact.shape[1] / self.box_width))
        

    def __getNormalizedCoords(self, x,y,z,tensorShape):
        xx = self.dtype((2 * (x / (tensorShape[-3] )) - 1)).view(1,1,1,-1,1).requires_grad_()
        yy = self.dtype((2 * (y / (tensorShape[-2] )) - 1)).view(1,1,1,-1,1).requires_grad_()
        zz = self.dtype((2 * (z / (tensorShape[-1] )) - 1)).view(1,1,1,-1,1).requires_grad_() # tensorShape[-1]-1 maybe?
        return torch.cat([zz,yy,xx],-1)
        
        
        
    def __getitem__(self, index):
        x0,y0,t = np.unravel_index(index,[self.Exact.shape[0],self.no_boxes,self.noSteps])

        lb_y = y0*self.box_width
        ub_y = np.min([(y0+1)*self.box_width, self.Exact.shape[1]])
        
        lb_z = 0
        ub_z = self.Exact.shape[2]
        
        idx_y = torch.arange(lb_y, ub_y).cuda().type(self.dtype2)
        idx_z = torch.arange(lb_z, ub_z).cuda().type(self.dtype2)
        
        y0,z0 = torch.meshgrid(idx_y, idx_z)
        y0 = y0.reshape((-1))
        z0 = z0.reshape((-1))
        x0 = x0 * torch.ones([len(z0)]).type(self.dtype2)
        
        t0 = torch.zeros([len(z0)])
        
        Ex0 = (self.Exact[x0,y0,z0])
        
        if(False): 
            #TODO: this sampling results in worse accuracy?!
            dd = self.Exact
            dd = dd[None,...]
            dd = dd[None,...].requires_grad_()
            
            x0 = x0.type(self.dtype)
            y0 = y0.type(self.dtype)
            z0 = z0.type(self.dtype)
            iv = self.__getNormalizedCoords(x0,y0,z0,self.Exact.shape)
            Ex0 = torch.squeeze(torch.nn.functional.grid_sample(dd,iv,mode='bilinear')).t()
            
        
        x0 = x0.type(self.dtype)
        y0 = y0.type(self.dtype)
        z0 = z0.type(self.dtype)
        t0 = t0.type(self.dtype)
                
        xf = x0.type(self.dtype)
        yf = y0.type(self.dtype)
        zf = z0.type(self.dtype)
        #xf = x0 * torch.ones([len(t0)]).type(self.dtype)
        #yf = self.dtype(len(xf), 1).uniform_(lb_y, ub_y)
        #zf = self.dtype(len(xf), 1).uniform_(lb_z, ub_z)
        tf = (t * self.dt) * torch.ones([len(x0)]).type(self.dtype)
        #self.Exact = torch.from_numpy(self.Exact).type(self.dtype)
        return x0,y0,z0,t0,Ex0,xf,yf,zf,tf
        
        
    def __len__(self):
        return int(self.Exact.shape[0] * self.no_boxes * self.noSteps)

    
class HelmholtzDataset(Dataset):
    def __init__(self, pData,file_prefix = "simData_",init_t = 2000,n0 = 1400000,use_gpu=True):
        path_init = pData + file_prefix + str(init_t) + '.h5' # concat filename for initial state
        exact, self.dx, self.dy, self.dz = loadFrame(path_init, 'fields/E/x') # get exact initial state (electrical field)
        #exact = exact[80:90,:,90:160]
        random_state = np.random.RandomState(seed=1234)
        # get domain sampling
        self.nx = exact.shape[0]
        self.ny = exact.shape[1]
        self.nz = exact.shape[2]
        print("Size: ", exact.shape) 
        ##get cell dimensions
        #with h5.File(path_init,'r') as f:
        #    self.dx = f['/data/{}'.format(init_t)].attrs['cell_width']
        #    self.dy = f['/data/{}'.format(init_t)].attrs['cell_height']
        #    self.dz = f['/data/{}'.format(init_t)].attrs['cell_depth']
            
        # create dataset x,y,z, E(x,y,z)
        x = np.arange(self.nx) * self.dx
        y = np.arange(self.ny) * self.dy
        z = np.arange(self.nz) * self.dz
        X,Y,Z = np.meshgrid(x,y,z)
        self.x = X.reshape(-1)
        self.y = Y.reshape(-1)
        self.z = Z.reshape(-1)
        self.t = np.full(self.x.shape[0],0)
        self.Exact = exact
        if (use_gpu):
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

       
        #random points
        idx = random_state.choice(self.x.shape[0],n0)
        self.x = torch.from_numpy(self.x[idx]).type(self.dtype)
        self.y = torch.from_numpy(self.y[idx]).type(self.dtype)
        self.z = torch.from_numpy(self.z[idx]).type(self.dtype)
        self.t = torch.from_numpy(self.t[idx]).type(self.dtype)
        self.Ex_rand = torch.from_numpy(self.Exact).reshape(-1)[idx].type(self.dtype)
        self.dt = 1.39e-16 #PIC step size
        #create lower and upper bound
        self.lb = np.array([0,0,0,0])
        self.ub = np.array([self.nx * self.dx,
                            self.ny * self.dy,
                            self.nz * self.dz,
                            1
                  ])
        self.Exact = torch.from_numpy(exact).type(self.dtype)


    def __getitem__(self,index):
        #tf = (index * self.dt) * torch.ones([1]).type(self.dtype)
        return self.x[index], self.y[index], self.z[index], self.t[index], self.Ex_rand[index], self.x[index], self.y[index], self.z[index], self.t[index] 
        
    
    def __len__(self):
        return len(self.Ex_rand)
