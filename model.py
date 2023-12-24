# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 11:04:42 2018

@author: liuliang
"""
import torch.nn as nn
import torch
import torch.nn.init as init
import numpy as np


class DQN(nn.Module):
    def __init__(self, ACTION_NUMBER, NUM_STEP_ALLOW):
        super(DQN, self).__init__()
        self.ACTION_NUMBER = ACTION_NUMBER
        self.layer1 = nn.Conv2d(in_channels=NUM_STEP_ALLOW+512, out_channels=1024, kernel_size=1, padding=0)        
        self.layer2 = nn.ReLU(inplace=True)  
        
        self.layer3 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, padding=0)        
        self.layer4 = nn.ReLU(inplace=True)  
                 
        self.layer5 = nn.Conv2d(in_channels=1024, out_channels=ACTION_NUMBER, kernel_size=1, padding=0)         
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
                init.constant_(m.bias.data,0.01)
                
    def forward(self, x, state):                
        x = [x,state]        
        x = torch.cat(x,1)
        del state
        
        x = self.layer1(x) 
        x = self.layer2(x) 
        
        x = self.layer3(x) 
        x = self.layer4(x) 
                
        x = self.layer5(x) 
        return x
    
class LibraNet(nn.Module):
    def __init__(self, parameters):
        super(LibraNet, self).__init__()  
        #Weights definition
        Action1 = 1
        Action2 = 2
        Action3 = 3
        Action4 = 5
        Action5 = 7
        Action6 = 999
        self.A = [Action1,Action2,Action3,Action4,Action5,Action6]
        self.A_mat = np.array(self.A)
        
        self.A_mat_h_w = np.expand_dims(np.expand_dims(self.A_mat, 1), 2)
        
        #Inverse discretization vector
        self.class2num = np.zeros(parameters['Interval_N'])
        for i in range(1, parameters['Interval_N']):
            if i == 1:
                lower = 0
            else:
                lower = np.exp((i - 2) * parameters['step_log'] + parameters['start_log'])
            upper = np.exp((i - 1) * parameters['step_log'] + parameters['start_log'])
            self.class2num[i] = (lower + upper) / 2
        
        #Network definition   
        self.DQN = DQN(parameters['ACTION_NUMBER'], parameters['NUM_STEP_ALLOW'])
        self.DQN_fixed_w = DQN(parameters['ACTION_NUMBER'], parameters['NUM_STEP_ALLOW'])
        
        # Initialize the weight
        self.weights_normal_init(self.DQN)
        self.weights_normal_init(self.DQN_fixed_w)
        
    def get_Q(self, feature=None, states=None):
        return self.DQN(feature,states) * 100       # Why need * 100
    
    def get_Q_fixed(self, feature=None, states=None):
        return self.DQN_fixed_w(feature, states) * 100
   
    def weights_normal_init(self, model, dev=0.01):
        if isinstance(model, list):
            for m in model:
                self.weights_normal_init(m, dev)
        else:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):                
                    m.weight.data.normal_(0, dev)
                    if m.bias is not None:
                        m.bias.data.fill_(0)
                elif isinstance(m, nn.ConvTranspose2d):                
                    m.weight.data.normal_(0, dev)
                    if m.bias is not None:
                        m.bias.data.fill_(0)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, dev)

class VGG16_BackBone(nn.Module):
    def __init__(self):
        super(VGG16_BackBone, self).__init__()
        self.layer0 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)        
        self.layer1 = nn.ReLU(inplace=True)        
        self.layer2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.layer3 = nn.ReLU(inplace=True)            
        self.layer4 = nn.MaxPool2d(kernel_size=2, stride=2) 
# =============================================================================        
        self.layer5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.layer6 = nn.ReLU(inplace=True)      
        self.layer7 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)  
        self.layer8 = nn.ReLU(inplace=True)              
        self.layer9 = nn.MaxPool2d(kernel_size=2, stride=2) 
# =============================================================================        
        self.layer10 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1) 
        self.layer11 = nn.ReLU(inplace=True)      
        self.layer12 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1) 
        self.layer13 = nn.ReLU(inplace=True)      
        self.layer14 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1) 
        self.layer15 = nn.ReLU(inplace=True)              
        self.layer16 = nn.MaxPool2d(kernel_size=2, stride=2) 
               
# =============================================================================        
        self.layer17 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1) 
        self.layer18 = nn.ReLU(inplace=True)      
        self.layer19 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.layer20 = nn.ReLU(inplace=True)      
        self.layer21 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1) 
        self.layer22 = nn.ReLU(inplace=True)              
        self.layer23 = nn.MaxPool2d(kernel_size=2, stride=2) 
        
# =============================================================================
        self.layer24 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1) 
        self.layer25 = nn.ReLU(inplace=True)   
        self.layer26 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.layer27 = nn.ReLU(inplace=True)    
        self.layer28 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1) 
        self.layer29 = nn.ReLU(inplace=True)
        self.layer30 = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
                init.constant_(m.bias.data,0.01)
                
    def forward(self, x):
        x = self.layer0(x)         
        x = self.layer1(x)  
        x = self.layer2(x)  
        x = self.layer3(x)  
        x = self.layer4(x)  
        x = self.layer5(x)  
        x = self.layer6(x)  
        x = self.layer7(x)  
        x = self.layer8(x)  
        x = self.layer9(x)  
        x = self.layer10(x)  
        x = self.layer11(x)  
        x = self.layer12(x)  
        x = self.layer13(x)  
        x = self.layer14(x)  
        x = self.layer15(x)  
        x = self.layer16(x)   
                 
        x = self.layer17(x)             
        x = self.layer18(x)  
        x = self.layer19(x) 
        x = self.layer20(x)  
        x = self.layer21(x)  
        x = self.layer22(x)          
        x = self.layer23(x)  
                         
        x = self.layer24(x)  
        x = self.layer25(x)  
        x = self.layer26(x)  
        x = self.layer27(x)  
        x = self.layer28(x)  
        x = self.layer29(x)    
        x = self.layer30(x)  
        return x