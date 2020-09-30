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
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import h5py

import time
from torch.utils.data import Dataset

import torch.utils.data.distributed
import horovod.torch as hvd
import socket

from NetworkArchitectures import HelmholtzNet, HelmholtzNetFour, GatedHelmholtzNet
from HelmholtzDatasets import HelmholtzDataset as HelmholtzEquationRandomDataset
from HelmholtzDatasets import HelmholtzEquationBoxedDataset as HelmholtzEquationDataset
DEBUGMODE = False

#######################################
#######################################
#######################################
#######################################
def save_checkpoint(model, optimizer, path, epoch):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, path + 'model_' + str(epoch))


def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def writeDynamicsZoomed(model,hqd,pResults,epoch, logwriter = None):
    # write dynamics based on grid interpolation
    dy = hqd.Exact.shape[1] 
    dz = hqd.Exact.shape[2]
       
    y0 = torch.linspace(500, 1250,dy).cuda()
    z0 = torch.linspace(50, 150,dz).cuda()
    y0v, z0v = torch.meshgrid(y0,z0)
    y0v = y0v.contiguous().view([-1,1])
    z0v = z0v.contiguous().view([-1,1])
    
    x0 = int(hqd.Exact.shape[0] / 2)
    x0v = hqd.dtype((x0 * torch.ones([len(z0v)])).cuda()).view([-1,1])
    
    
    for t in range(100): # 100 steps
        t0 = t * hqd.dt
        t0v = hqd.dtype((t0 * torch.ones([len(z0v)])).cuda()).view([-1,1])
        X = torch.cat([x0v,y0v,z0v,t0v],1)

        Ex = model.forward(X).detach().cpu().numpy()    
        Ex = Ex.reshape((dy,dz))
        
        plt.imshow(Ex,interpolation='bilinear', cmap='jet', origin='lower',aspect='auto')
        plt.title('epoch %d @ x=%d, t=%.17fs' % (epoch,x0,t0))
        plt.colorbar()
        plt.show()
        plt.savefig(pResults+"/h_dynamics_zoomed_x%d_t%d_e%d.png" % (x0,t,epoch))
        plt.close()
    

def writeDynamics(model,hqd,pResults,epoch, logwriter = None):
    # write dynamics based on grid interpolation
    dy = hqd.Exact.shape[1] 
    dz = hqd.Exact.shape[2]
       
    y0 = torch.linspace(0, hqd.Exact.shape[1]-1,dy).cuda()
    z0 = torch.linspace(0, hqd.Exact.shape[2]-1,dz).cuda()
    y0v, z0v = torch.meshgrid(y0,z0)
    y0v = y0v.contiguous().view([-1,1])
    z0v = z0v.contiguous().view([-1,1])
    
    x0 = int(hqd.Exact.shape[0] / 2)
    x0v = hqd.dtype((x0 * torch.ones([len(z0v)])).cuda()).view([-1,1])
    
    
    for t in range(100): # 100 steps
        t0 = t * hqd.dt
        t0v = hqd.dtype((t0 * torch.ones([len(z0v)])).cuda()).view([-1,1])
        X = torch.cat([x0v,y0v,z0v,t0v],1)

        Ex = model.forward(X).detach().cpu().numpy()
        Ex = Ex.reshape((dy,dz))
        
        #plt.imshow(Ex,interpolation='bilinear', cmap='jet', origin='lower',aspect='auto')
        #plt.title('epoch %d @ x=%d, t=%.17fs' % (epoch,x0,t0))
        #plt.colorbar()
        #plt.show()
        #plt.savefig(pResults+"/h_dynamics_x%d_t%d_e%d.png" % (x0,t,epoch))
        #plt.close()
        
        fig = plt.figure()
        plt.imshow(Ex,interpolation='bilinear', cmap = 'jet', origin='lower',aspect='auto')
        plt.colorbar()
        logwriter.add_figure('h@x%d_t=%.17fs' % (x0, t),fig,0)
        plt.close(fig)
        
        #if(t>0):
        #    plt.imshow(Ex-Ex_old,interpolation='bilinear', cmap='jet', origin='lower',aspect='auto')
        #    plt.title('epoch %d @ x=%d, t=%.17fs' % (epoch,x0,t0))
        #    plt.colorbar()
        #    plt.show()
        #    plt.savefig(pResults+"/h_diffdynamics_x%d_t%d_e%d.png" % (x0,t,epoch))
        #    plt.close()
        #else:
        #    Ex0 = Ex.copy()
        # Ex_old = Ex.copy()

    #writeDynamicsZoomed(model,hqd,pResults,epoch)


def computeTimeDifference(model,hqd,epoch,t, logwriter = None):
    dy = hqd.Exact.shape[1]
    dz = hqd.Exact.shape[2]

    y0 = torch.linspace(0, hqd.Exact.shape[1]-1,dy).cuda()
    z0 = torch.linspace(0, hqd.Exact.shape[2]-1,dz).cuda()
    y0v, z0v = torch.meshgrid(y0,z0)
    y0v = y0v.contiguous().view([-1,1])
    z0v = z0v.contiguous().view([-1,1])

    x0 = int(hqd.Exact.shape[0] / 2)
    x0v = hqd.dtype((x0 * torch.ones([len(z0v)])).cuda()).view([-1,1])

    t0v = hqd.dtype((torch.zeros([len(z0v)])).cuda()).view([-1,1])
    t1 = t * hqd.dt
    t1v = hqd.dtype((t1 * torch.ones([len(z0v)])).cuda()).view([-1,1])   

    X0 = torch.cat([x0v,y0v,z0v,t0v],1)
    X1 = torch.cat([x0v,y0v,z0v,t1v],1)
    
    Ex0 = model.forward(X0).detach()
    Ex1 = model.forward(X1).detach()

    print("[%d] D@0 -> %.2E = %.16f" % (epoch,t1v[0,0], torch.mean(torch.abs(Ex0 - Ex1))))
    #print("   inx  %.2E, %.2E, %.2E " % (torch.min(model.in_x[0].weight.grad.view(-1)),torch.mean(model.in_x[0].weight.grad.view(-1)),torch.max(model.in_x[0].weight.grad.view(-1))))
    #print("   int  %.2E, %.2E, %.2E " % (torch.min(model.in_t[0].weight.grad.view(-1)),torch.mean(model.in_t[0].weight.grad.view(-1)),torch.max(model.in_t[0].weight.grad.view(-1))))


def writeIntermediateResults(model,hqd,pResults,epoch, logwriter = None):
    # write results based on grid interpolation
    dy = hqd.Exact.shape[1] 
    dz = hqd.Exact.shape[2]
    
    y0 = torch.linspace(0, dy-1,dy).cuda()
    z0 = torch.linspace(0, dz-1,dz).cuda()
    y0v, z0v = torch.meshgrid(y0,z0)
    y0v = y0v.contiguous().view([-1,1])
    z0v = z0v.contiguous().view([-1,1])

    t0v = hqd.dtype((torch.zeros([len(z0v)])).cuda()).view([-1,1])
    
    for x0 in range(hqd.Exact.shape[0]):
        x0v = hqd.dtype((x0 * torch.ones([len(z0v)])).cuda()).view([-1,1])
        X = torch.cat([x0v,y0v,z0v,t0v],1)
        Ex = model.forward(X).detach().cpu().numpy()    
        Ex = Ex.reshape((dy,dz))
        Ex_gt = hqd.Exact[x0,...].detach().cpu().numpy() 
        
        if log_writer:
            fig = plt.figure()
            plt.imshow(Ex,interpolation='bilinear', cmap = 'jet', origin='lower',aspect='auto')
            plt.colorbar()
            logwriter.add_figure('h@x%d/pinn' % x0,fig,epoch)
            plt.close(fig)
            
            if(epoch < 10):
                fig = plt.figure()
                plt.imshow(Ex_gt,interpolation='bilinear', cmap = 'jet', origin='lower',aspect='auto')
                plt.colorbar()
                logwriter.add_figure('h@x%d/gt' % x0,fig,0)
                plt.close(fig)
    
        print("[%d] x=%d EX min %.5f mu %.5f s %.5f max %.5f vs EXr min %.5f mu %.5f s %.5f max %.5f" % (epoch, X[0,0], np.min(Ex),np.mean(Ex),np.std(Ex),np.max(Ex),np.min(Ex_gt),np.mean(Ex_gt),np.std(Ex_gt),np.max(Ex_gt)))


################################################
################################################
################################################
### MAIN
################################################
################################################
################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3D Helmholtz Equation -- Learning')
    parser.add_argument("-pd", "--pdata", type=str, default="/bigdata/hplsim/production/AIPP/Data/LaserEvolution/runs/004_LaserOnly/simOutput/h5/")
    parser.add_argument("-id","--identifier", type=str, default="")
    parser.add_argument('-bw', '--boxwidth', default=150, type=int, help='size of mini batch')
    parser.add_argument('-d', '--depth', default=8, type=int, help='number of layers')
    parser.add_argument('-f', '--features', default=100, type=int, help='number of features')
    parser.add_argument('-w', '--weightInitialState', default=1, type=int, help='weight of initial state to be used in pde loss')
    parser.add_argument('-i_pde', '--iterPDELearning', default=50000, type=int, help='iterations of adam optimizer')
    parser.add_argument('-i_pt',dest='iterPretraining', help='no epochs for pretraining', type=int, default=2000)
    parser.add_argument('-lr_pt','--learningratePretraining', type=float, default=1e-4, help='learning rate for pretraining')
    parser.add_argument('-lr_pde','--learningratePDELearning', type=float, default=1e-6, help='learning rate for PDE learning')
    parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies total batch size.')
    parser.add_argument('--gatedNet',dest='gatedNet', help='use gated network', action='store_true')
    parser.add_argument('--fourInNet',dest='fourInNet', help='use fourIn network', action='store_true')
    parser.add_argument('-k', dest="k", default=2, type=int, help='number of active experts')
    parser.add_argument('-n', dest="numberExperts", default=5, type=int, help='number of experts')
    parser.add_argument('-b', dest="batchsize", default=32000, type=int, help='number of experts')
    args = parser.parse_args() 
    
    
    ################################################
    ### VARIABLES
    ################################################
    pData = args.pdata
    noLayer = args.depth

    myRank = 0

    ### HVD ###
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(2342)
    myRank = hvd.rank()  
 

    ################################################
    ### DATA LOADER
    ################################################
    pResults = 'results/%s_l%d_f%d_g%d_k%d_n%d_4in%d_sinrelu_%.2e' % (args.identifier, noLayer, args.features, args.gatedNet, args.k, args.numberExperts, args.fourInNet, args.learningratePretraining)
    log_writer = SummaryWriter(pResults) if hvd.rank() == 0 else None
    
    
    if(myRank == 0 or DEBUGMODE):
        os.makedirs(pResults, exist_ok=True)
        log_writer.add_text('Setup', '%s' % args, 0)
        print(args)
        print("Writing results to: %s" % pResults)

    #hqd = HelmholtzEquationRandomDataset(pData=pData) # pretraining dataset    
    hqd = HelmholtzEquationDataset(pData=pData, pResults=pResults, box_width = args.boxwidth, noSteps = 1) # pretraining dataset    
    sampler = torch.utils.data.distributed.DistributedSampler(hqd, num_replicas=hvd.size(), rank=hvd.rank())
    hqd_loader = torch.utils.data.DataLoader(dataset=hqd,
                                                    batch_size=args.batchsize,
                                                    sampler=sampler)
   
    ################################################
    ### TRAIN MODEL
    ################################################
    if(args.gatedNet == True):
        activation = torch.relu
        noisy_gating = False
        numExperts = args.numberExperts
        k = args.k
        nonlinear = False 
        model = GatedHelmholtzNet(lb = hqd.lb, ub = hqd.ub, numLayers = noLayer, numFeatures = args.features, numExperts = numExperts, k = k, nonlinear = nonlinear, noisy_gating = noisy_gating, activation = activation, inputSize = 4, useGPU = True, useFourNet = args.fourInNet).cuda()
    elif(args.fourInNet == True):
        model = HelmholtzNetFour(lb = hqd.lb, ub = hqd.ub, numLayers = noLayer, numFeatures = args.features).cuda()
    else:
        model = HelmholtzNet(lb = hqd.lb, ub = hqd.ub, numLayers = noLayer, numFeatures = args.features).cuda()

    for param in model.parameters():
        print(type(param.data), param.size()) 
        
    if(myRank == 0):
        log_writer.add_text("Architecture", "%s" % model, 0)
        
        with open(pResults + "/net.txt", "w") as text_file:
            text_file.write(str(args))
            for param_tensor in model.state_dict():
                 text_file.write("%s%s%s\n" % (str(param_tensor), "\t", str(model.state_dict()[param_tensor].size())))
            text_file.write("\n")
            #for param_tensor in optim.state_dict():
            #     text_file.write("%s%s%s\n" % (str(param_tensor), "\t", str(optim.state_dict()[param_tensor])))
            text_file.write("\n")
 
    
    ################################################
    ### Pretraining w/ 1st order optimizer
    ################################################
    start_time = time.time()
  
    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.       
    optim = optim.Adam(model.parameters(),lr=args.learningratePretraining)   
    optim = hvd.DistributedOptimizer(optim,
                                     named_parameters=model.named_parameters(), compression=compression,
                                     backward_passes_per_step=1)
    
    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optim, root_rank=0)

    if(myRank == 0):
        print("--- Starting pretraining ---")
        
    for epoch in range(args.iterPretraining):
        l_epoch = []                
        for x0,y0,z0,t0,Ex0,xf,yf,zf,tf in hqd_loader:
            optim.zero_grad()
            x0 = x0.view(-1,1)
            y0 = y0.view(-1,1)
            z0 = z0.view(-1,1)
            t0 = t0.view(-1,1)
            Ex0 = Ex0.view(-1,1)
            loss = model.solution_loss(x0,y0,z0,t0,Ex0)
            loss.backward()
            #for param in model.parameters():
            #    if(param.size()[0] == 4):
            #        print (type(param.data), param.size(), param.grad)
            #print(model.moe_layer.w_gate)
            optim.step()
            l_epoch.append(loss.item())
            
        #if(myRank == 0):
        #    computeTimeDifference(model,hqd,epoch,1000)
        ### done iterating over mini-batches
        if(myRank == 0):
            print("[%d] IC loss %.10f / %.2fs" % (epoch, np.mean(l_epoch), time.time() - start_time))
            log_writer.add_scalar('Loss/Pretraining', np.mean(l_epoch), epoch)
            if( (epoch % 100) == 0):
                writeIntermediateResults(model,hqd,pResults,epoch, log_writer)

    
    ### pretraining completed
    if(myRank == 0 and args.iterPretraining > 0):
        torch.save(model.state_dict(), pResults+"/mlp_adam_pt_%d.pt" % (epoch))
        print("------ pretraining completed. Total runtime %.1f s" % (time.time() - start_time))
        print("       saving model to %s" % (pResults+"/mlp_adam_pt_%d.pt" % (epoch)))

        
    ################################################
    ### PDE learning w/ 1st order optimizer
    ################################################
    del hqd, sampler, hqd_loader
    
    start_time = time.time()
    w = args.weightInitialState
    
    hqd_pdeLearning = HelmholtzEquationDataset(pData=pData, pResults=pResults, box_width = args.boxwidth)
    sampler_pdeLearning = torch.utils.data.distributed.DistributedSampler(hqd_pdeLearning, num_replicas=hvd.size(), rank=hvd.rank())
    hqd_loader_pdeLearning = torch.utils.data.DataLoader(dataset=hqd_pdeLearning,
                                                    batch_size=args.batchsize,
                                                    sampler=sampler_pdeLearning)
    
    for paramGroup in optim.param_groups:
        paramGroup['lr'] = args.learningratePDELearning

    if(myRank == 0):
        print("\n--- Starting PDE learning ---")
    for epoch in range(args.iterPDELearning):
        l_epoch = []                
        for x0,y0,z0,t0,Ex0,xf,yf,zf,tf in hqd_loader_pdeLearning:
            optim.zero_grad()
            x0 = x0.view(-1,1)
            y0 = y0.view(-1,1)
            z0 = z0.view(-1,1)
            t0 = t0.view(-1,1)
            Ex0 = Ex0.reshape([-1])
            xf = xf.view(-1,1)
            yf = yf.view(-1,1)
            zf = zf.view(-1,1)
            tf = tf.view(-1,1)

            #print("%.2e - %.2e %s" % (torch.min(tf), torch.max(tf), str(tf.shape)))

            loss = model.pde_loss(x0,y0,z0,t0,Ex0,xf,yf,zf,tf, w)
            loss.backward()
            optim.step()
            l_epoch.append(loss.item())
            
        #if(myRank == 0):
        #    computeTimeDifference(model,hqd_pdeLearning,epoch,1000)
        ### done iterating over mini-batches
        if(myRank == 0):
            print("[%d] PDE loss %.10f / %.2fs" % (epoch, np.mean(l_epoch), time.time() - start_time))
            log_writer.add_scalar('Loss/PDE', np.mean(l_epoch), epoch)
            
#            if( (epoch % 100) == 0):
            writeDynamics(model,hqd_pdeLearning,pResults,epoch, log_writer)
            
            
    if(myRank == 0):
        torch.save(model.state_dict(), pResults+"/mlp_adam_pde_%d.pt" % (epoch))
        print("------ PDE learning completed. Total runtime %.1f s" % (time.time() - start_time))
        print("       saving model to %s" % (pResults+"/mlp_adam_pde_%d.pt" % (epoch)))
