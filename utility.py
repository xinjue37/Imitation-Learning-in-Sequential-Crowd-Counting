import numpy as np
import random
import torch
import numpy as np
import cv2
import pandas as pd
import scipy.io as sio
from scipy.io import loadmat
import os
from torchvision import transforms
from torchvision.models import vgg16
import torch.nn as nn


def load_data_in_batch(i, parameters, h, w, img_pt_list, gt_csv_list, device, vgg16, max_x, max_y):
    num_data           = min(parameters["BATCH_SIZE_IMG"], len(img_pt_list) - i*parameters["BATCH_SIZE_IMG"])
    train_img_features = torch.rand(num_data, 512, h, w)
    train_labels       = torch.rand(num_data, h, w)
    actual_count       = torch.rand(num_data)
    
    for j in range(num_data):
        curr_idx     = i*parameters["BATCH_SIZE_IMG"] + j
        if img_pt_list[curr_idx].endswith(".pt"):   # If pt file is passed, directly load the pre-computed image features
            img_feature  = torch.load(img_pt_list[curr_idx])
            train_img_features[j] = img_feature.clone()   # If img file is passed, compute image features by VGG_16
        elif img_pt_list[curr_idx].endswith(".jpg"):
            if vgg16 is None:
                raise("VGG16 model is not passed.")
            else:
                batch_img          = get_img_in_batch(i, parameters, img_pt_list, max_x, max_y)
                train_img_features = vgg16(batch_img.to(device)).cpu().detach().clone()
                break
        else:
            raise("Invalid format passed. Accept format .pt OR .jpg, but directory pass is :", img_pt_list[curr_idx])
    
    for j in range(num_data):
        curr_idx     = i*num_data + j
        if gt_csv_list[curr_idx].endswith(".csv"):
            label        = pd.read_csv(gt_csv_list[curr_idx],
                                    sep=',',
                                    header=None)
            train_labels[j]       = torch.from_numpy(label.to_numpy()).clone()
            actual_count[j]       = train_labels[j].sum()
        elif gt_csv_list[curr_idx].endswith(".mat"):
            label = loadmat(gt_csv_list[curr_idx])
            label = label["image_info"][0][0][0][0][0]
            actual_count[j] = len(label)

    return train_img_features, train_labels, actual_count


def get_img_in_batch(i, parameters, img_pt_list, max_x, max_y):
    num_data           = min(parameters["BATCH_SIZE_IMG"], len(img_pt_list) - i*parameters["BATCH_SIZE_IMG"])

    batch_img = torch.rand(num_data, 3, max_x, max_y)
    for j in range(num_data):
        curr_idx     = i*parameters["BATCH_SIZE_IMG"] + j
        img       = cv2.imread(img_pt_list[curr_idx])
        transform = transforms.Compose([transforms.ToTensor(),   # Convert image to range [0,1] and shape from (3,H,W) to (H,W,3)
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) 
        img       = transform(img)     # Transform the image so it follow the VGG16_Weights.IMAGENET1K_V1.transforms
        
        # Add padding to image so the final image have shape (3, max_x, max_y)
        img_pad   = torch.zeros((3, max_x, max_y)) 
        img_pad[:, :img.shape[1], :img.shape[2]] = img.clone()
        batch_img[j] = img_pad.clone()
    
    return batch_img
                               

def train_model(Libranet, epoch, 
                parameters, 
                replay, optimizer,                    
                train_img_pt_list, train_gt_csv_list, # List of directory that point to .pt file and .csv file
                device,                               # Device used to train the Neural Network
                TRAIN_WITH_OPTIMAL_ACTION=False,      # If True, train the network using 90% optimal action and 10% using argmax(Q)
                LOSS_FUNCTION="QL",                   # Loss function used to train the Deep Q network
                vgg16=None,                           # Pre-trained vgg16
                max_x=1024,                           # Maximum number of pixel along the x-axis
                max_y=1024                            # Maximum number of pixel along the y-axis
                ):

    assert LOSS_FUNCTION in ["QL", "CE", "CB"]
    print(f"Train with {LOSS_FUNCTION} that guided by optimal action? {TRAIN_WITH_OPTIMAL_ACTION}")
    
    # Initialize the variables
    EPSILON     = max(0.1, 1-epoch*0.05) # Decaying epsilon for decrease exploration over time, start with [1, 0.95, 0.9, ..., 0.1]
    loss_train  = 0
    total_mae   = 0
    total_mse   = 0
    number_deal = 0
    number_rest = 0
    h = int(max_x // 32)
    w = int(max_y // 32)

    # Training loop for each batch of image data 
    # If len(train_img_pt_list) < parameters["BATCH_SIZE_IMG"], it will loops for one time
    # Else it loop for [len(train_img_pt_list) // parameters["BATCH_SIZE_IMG"])] times
    for i in range( max(  int(len(train_img_pt_list) // parameters["BATCH_SIZE_IMG"]),   1 )  ):  
        print(f"Generate episode from {i+1} batch of data")
        
        # A) Load the data in a batch
        num_data           = min(parameters["BATCH_SIZE_IMG"], len(train_img_pt_list) - i*parameters["BATCH_SIZE_IMG"])
        train_img_features, train_labels, _ = load_data_in_batch(i, parameters, h, w, 
                                                                 train_img_pt_list, train_gt_csv_list, 
                                                                 device, vgg16, 
                                                                 max_x, max_y)
        
        # B) Initialize some variables for later used
        mask_prev_end = torch.zeros(num_data, h, w, dtype=torch.int8)
        count_rem     = torch.zeros(num_data, h, w, dtype=torch.int16)   # Remainder count
        curr_state    = torch.zeros(num_data, parameters['NUM_STEP_ALLOW'], h, w)
        
        # C) Sample the data from each of the image
        for step_hv in range(0, parameters['NUM_STEP_ALLOW']): 
            Libranet.eval()
            
            # C1. Sample actions 
            ## C1.1 Compute the optimal action on each grid
            optimal_action = torch.zeros(num_data,h, w)
            
            count_after_every_action = count_rem[:,None,:,:] + Libranet.A_mat[None,:,None,None]  # Shape = BATCH_SIZE, NUM_ACTION, h, w
            error_every_action       = abs(train_labels[:,None,:,:] - count_after_every_action)  # Shape = BATCH_SIZE, NUM_ACTION, h, w
            optimal_action           = error_every_action.argmin(axis=1)                         # Shape = BATCH_SIZE, h, w
            error_last               = train_labels - count_rem                             # Shape = BATCH_SIZE, h, w    
            optimal_action[error_last<=parameters['ERROR_SYSTEM']] = parameters['ACTION_NUMBER'] - 1 # Optimal action when train_labels-count_rem <=0 is END

            # Sample action 
            curr_state_torch         = curr_state.permute(0,2,3,1).reshape(-1, parameters['NUM_STEP_ALLOW'], 1, 1).float().to(device) # Reshape to (parameters['BATCH_SIZE_IMG']* h* w, parameters['NUM_STEP_ALLOW'])
            train_img_features_torch = train_img_features.permute(0,2,3,1).reshape(-1, 512, 1, 1).to(device) # Reshape to (parameters["BATCH_SIZE_IMG"] * 32 * 32, 512)
            old_Q                    = Libranet.get_Q(feature=train_img_features_torch, 
                                                      states=curr_state_torch)
            curr_state_torch         = curr_state_torch.detach().cpu()                                           # Move the data to cpu()
            train_img_features_torch = train_img_features_torch.detach().cpu()                                   # Move the data to cpu()
            old_Q                    = old_Q.cpu().detach().reshape((num_data,h,w,parameters["ACTION_NUMBER"]))  # old_Q.shape=[BATCH_SIZE_IMG,32,32,6], 6 = ACTION_NUMBER
            old_Q                    = old_Q.permute((0, 3, 1, 2))                                               # old_Q.shape=[BATCH_SIZE_IMG,6,32,32]
            action_max               = old_Q.argmax(axis=1)                                                      # Action = argmax_a (Q(s,a))
            
            if TRAIN_WITH_OPTIMAL_ACTION:   # Sample action based on 90% optimal action and 10% action based on argmax(Q)
                random_select =  (torch.rand(num_data, h, w) < 0.1).to(torch.int8)   
                action_fusion =  random_select * optimal_action + (1-random_select) * action_max
            else:                           # Sample action based on eps-greedy policy (about (1-EPSILON)% is based on argmax(Q), about EPSILON% is based on random action to favor exploration)     
                action_random = torch.randint(low=0, high=len(Libranet.A), 
                                              size=(num_data, h, w))
                random_select =  (torch.rand(num_data, h, w) < EPSILON).to(torch.int8)      
                action_fusion = random_select * action_random + (1-random_select) * action_max                # Shape = [BATCH_SIZE_IMG, h, w]
            
            
            # C2. Save the next state
            next_state                 = curr_state.clone()
            next_state[:,step_hv, :,:] = action_fusion
            
            
            # C3. Compute the reward
            reward_matrix  = torch.zeros(num_data,h, w)    # Reward matrix
            
            ## C3.1 Compute the current error
            mask_select_end = action_fusion == parameters['ACTION_NUMBER'] - 1 # Mask of selecting the END for current timestep, t 
            mask_select_end = mask_select_end.to(torch.int8)
            mask_now_end    = mask_prev_end | mask_select_end              # Mask of selecting the end for all step [0, 1, ..., t]
            mask_now_end    = mask_now_end.to(torch.int8)
            
            # Libranet.A.shape=(8), action_fusion.shape=(BATCH_SIZE_IMG,h,w), Libranet.A[action_fusion].shape=(BATCH_SIZE, IMG, h,w)
            # Add the action number to count_rem if it is not at the end state
            count_rem  = count_rem + (1 - mask_select_end) * (1 - mask_prev_end) * \
                         (Libranet.A_mat[action_fusion.to(torch.int8)])                              
            error_now  = train_labels - count_rem                                                  # Shape = BATCH_SIZE, h, w            
                        
            ## C3.2 Reward computation
            if step_hv != parameters['NUM_STEP_ALLOW'] - 1:  # If it is not equal to last step
                mask_in_range       = (count_rem <= train_labels * (1 + parameters['ERROR_RANGE'])).to(torch.int8)  # All the count that are currently <= label
                mask_error_decrease = (abs(error_last) > abs(error_now)).to(torch.int8)                                       # All the value that the error increase
                mask_optimal        = (action_fusion == optimal_action).to(torch.int8)                              # All the action that it is equal to optimal solution
                mask_could_end_last = (error_last <= parameters['ERROR_SYSTEM']).to(torch.int8)                     # All the error that <= ERROR_SYSTEM
                
                ##ending reward
                reward_matrix = mask_select_end * mask_could_end_last * 5 + mask_select_end * (1 - mask_could_end_last) * -5
                
                ##guiding reward
                reward_matrix = reward_matrix + (1 - mask_select_end) * mask_in_range * mask_error_decrease * mask_optimal * 3
                reward_matrix = reward_matrix + (1 - mask_select_end) * mask_in_range * mask_error_decrease * (1 - mask_optimal) * 1
                reward_matrix = reward_matrix + (1 - mask_select_end) * mask_in_range * (1 - mask_error_decrease) * -1
                
                ##squeeze guiding reward
                reward_matrix = reward_matrix + (1 - mask_select_end) * (1 - mask_in_range) * mask_error_decrease * mask_optimal * -1
                reward_matrix = reward_matrix + (1 - mask_select_end) * (1 - mask_in_range) * mask_error_decrease * (1 - mask_optimal) * -3
                reward_matrix = reward_matrix + (1 - mask_select_end) * (1 - mask_in_range) * (1 - mask_error_decrease) * -3
            else:
                ##ending reward
                mask_select_end = torch.ones(num_data, h, w)                
                mask_could_end_now = error_now <= parameters['ERROR_SYSTEM']
                mask_could_end_now = mask_could_end_now.to(torch.int8)
                reward_matrix = mask_could_end_now * 5 + (1 - mask_could_end_now) * -5
            
            
            # C4. Update the buffer
            # C4.1 Resize each of the shape to (BATCH_SIZE_IMG *32 *32, -1)
            mask_drop = (torch.rand(num_data,h, w) < 0.5) * (abs(error_last) <= 1) # Random drop some data
            mask_drop = mask_drop.to(torch.int8)
            
            if ((1-mask_prev_end)*(1-mask_drop)).sum()<=1:
                continue
            mask_last_batch = mask_prev_end.flatten()  # Shape=(BATCH_SIZE_IMG * 32 * 32)
            mask_drop       = mask_drop.flatten()      # Shape=(BATCH_SIZE_IMG * 32 * 32)
            mask_all        = torch.logical_or(mask_last_batch==0, mask_drop==0).to(torch.bool)
            
            state_fv        = np.transpose(train_img_features.cpu().numpy(), (0,2,3,1)).reshape(mask_drop.shape[0],-1) # Shape=(BATCH_SIZE_IMG * 32 * 32, 512)
            state_hv        = np.transpose(curr_state, (0,2,3,1)).reshape(mask_drop.shape[0],-1)        # Shape=(BATCH_SIZE_IMG * 32 * 32, parameters['NUM_STEP_ALLOW'])
            next_state_hv   = np.transpose(next_state, (0,2,3,1)).reshape(mask_drop.shape[0],-1)        # Shape=(BATCH_SIZE_IMG * 32 * 32, parameters['NUM_STEP_ALLOW'])
            action          = action_fusion.flatten()                                                   # Shape=(BATCH_SIZE_IMG * 32 * 32)
            optimal_action  = optimal_action.flatten()                                                  # Shape=(BATCH_SIZE_IMG * 32 * 32)
            reward          = reward_matrix.flatten()                                                   # Shape=(BATCH_SIZE_IMG * 32 * 32)
            done            = mask_select_end.flatten()
            
            # C4.2 hard sample mining (Distill large amount of raw data into high quality data)
            state_fv        = state_fv[mask_all]
            state_hv        = state_hv[mask_all]
            next_state_hv   = next_state_hv[mask_all]
            action          = action[mask_all]
            optimal_action  = optimal_action[mask_all]
            reward          = reward[mask_all]
            done            = done[mask_all]
            
            # C4.3 Send the data into buffer
            if not replay.can_sample():  # if buffer is not full
                # print(f"Placing {len(state_fv)} data into buffer")
                replay.put(state_fv, state_hv, action, optimal_action, reward, next_state_hv, done)
            else:                        # if buffer is full      
                # print(f"Backpropagation {len(state_fv)} data from buffer")   
                number_this_batch = len(state_fv)
                point_start = 0
                point_end = 0
                rest = number_this_batch + number_rest
                
                while rest>0:  
                    #train when every 100 samples are sent to buffer 
                    if rest < parameters['TRAIN_SKIP']:  # if number_this_batch + number_rest < parameters['TRAIN_SKIP']=100
                        # Place all the remaining data into the buffer
                        replay.put(state_fv[point_start:number_this_batch,:],
                                   state_hv[point_start:number_this_batch,:],
                                   action[point_start:number_this_batch],
                                   optimal_action[point_start:number_this_batch],
                                   reward[point_start:number_this_batch],
                                   next_state_hv[point_start:number_this_batch,:],
                                   done[point_start:number_this_batch])                        
                        number_rest=rest
                        rest=0
                    else:
                        point_end = min(point_end + parameters['TRAIN_SKIP'] - number_rest, number_this_batch)
                        number_rest = 0
                        
                        # Place some data into the buffer
                        replay.put(state_fv[point_start:point_end,:],
                                   state_hv[point_start:point_end,:],
                                   action[point_start:point_end],
                                   optimal_action[point_start:point_end],
                                   reward[point_start:point_end],
                                   next_state_hv[point_start:point_end,:],
                                   done[point_start:point_end])
                        
                        point_start = point_end                      # Set the start point = curr end point after place data into buffer
                        rest        = number_this_batch - point_end  # The remaining data = number_this_batch - point_end  
                        
                        Libranet.train()     
                        state_fv_batch, state_hv_batch, act_batch, opt_act_batch, rew_batch, next_state_hv_batch, done_mask = replay.out()
                        
                        # Convert the torch.FloatTensor and add 2 more dimension at the back
                        state_fv_batch      = torch.FloatTensor(state_fv_batch).to(device)[..., None, None]
                        state_hv_batch      = torch.FloatTensor(state_hv_batch).to(device)[..., None, None]
                        act_batch           = torch.LongTensor(act_batch).to(device)[..., None, None, None]
                        opt_act_batch       = torch.LongTensor(opt_act_batch).to(device)[..., None, None]
                        rew_batch           = torch.FloatTensor(rew_batch).to(device)[..., None, None, None]
                        next_state_hv_batch = torch.FloatTensor(next_state_hv_batch).to(device)[..., None, None]
                        done_mask           = torch.FloatTensor(done_mask).to(device)[..., None, None, None]
                        
                        # Get the Action-value from fixed weight
                        newQ     = Libranet.get_Q_fixed(feature=state_fv_batch, states=next_state_hv_batch) # Shape=[BATCH_SIZE, ACTION_NUMBER, 1, 1]
                        newQ     = newQ.data.max(1)[0].unsqueeze(1)                                         # Shape=[BATCH_SIZE, 1, 1, 1]
                        target_Q = newQ * parameters['GAMMA'] * (1 - done_mask) + rew_batch                 # argmax(Q(s',a')) + r
                        
                        # Get the Action-value from trainable weight
                        eval_Q     = Libranet.get_Q(feature=state_fv_batch, states=state_hv_batch)          # Shape=[BATCH_SIZE, ACTION_NUMBER, 1, 1]
                        eval_Q_act = eval_Q.gather(1,act_batch)                                             # Q(s,a,w), Shape=[BATCH_SIZE, 1, 1, 1]
                        
                        if LOSS_FUNCTION == "QL":          # Loss function for Q-leaning |argmax(Q(s',a', w_fixed)) + r - Q(s,a,w)|
                            loss = (eval_Q_act - target_Q.detach()).abs().mean()
                        elif LOSS_FUNCTION == "CE":        # Cross entropy loss between the predicted action and target action
                            loss_func = nn.CrossEntropyLoss()  
                            loss = loss_func(eval_Q, opt_act_batch)

                        elif LOSS_FUNCTION == "CB":        # Combination of Q-leaning loss + cross entropy loss
                            loss_func = nn.CrossEntropyLoss()
                            loss = loss_func(eval_Q, opt_act_batch) + (eval_Q_act - target_Q.detach()).abs().mean()
                        
                        optimizer.zero_grad()     # Clear the gradient
                        loss.backward()           # Compute the gradient
                        optimizer.step()          # Update the weight through backpropagation
                        
                        loss_train += loss.item() # Accumulate the loss
                            
                        number_deal = number_deal+1
                                                    
                        Libranet.eval()     
                        
                                     
                 
            curr_state =  next_state.clone()
            mask_prev_end = mask_now_end.clone()   
            if (1-mask_now_end).sum()==0:
                break   
        
    return loss_train/number_deal
            
            
@torch.no_grad()   
def evaluate_model(Libranet, parameters, 
                   img_pt_list, gt_file_list, device,
                   vgg16=None,
                   max_x=1024,
                   max_y=1024
                   ):
    
    Libranet.eval()
    # Initialize the variable for later used
    h = 32
    w = 32
    total_mae = 0
    total_mse = 0
    
    for i in range(max(int (len(img_pt_list) // parameters["BATCH_SIZE_IMG"]), 1)):  
        
        # A) Load the data in a batch
        num_data           = min(parameters["BATCH_SIZE_IMG"], len(img_pt_list))
        train_img_features, _, actual_count = load_data_in_batch(i, parameters, h, w, img_pt_list, gt_file_list, device, vgg16, max_x, max_y)
            
        # B) Initialize some variables for later used
        mask_prev_end = torch.zeros(num_data, h, w, dtype=torch.int8)
        count_rem     = torch.zeros(num_data, h, w, dtype=torch.int16)   # Remainder count
        curr_state    = torch.zeros(num_data, parameters['NUM_STEP_ALLOW'], h, w)
        
        # C) Sample the data from each of the image
        for step_hv in range(0, parameters['NUM_STEP_ALLOW']): 
            
            
            # C1. Sample actions based on eps-greedy policy
            curr_state_torch         = curr_state.permute(0,2,3,1).reshape(-1, parameters['NUM_STEP_ALLOW'], 1, 1).float().to(device) # Reshape to (parameters['BATCH_SIZE_IMG']* h* w, parameters['NUM_STEP_ALLOW'])
            train_img_features_torch = train_img_features.permute(0,2,3,1).reshape(-1, 512, 1, 1).to(device) # Reshape to (num_data * 32 * 32, 512)
            old_Q = Libranet.get_Q(feature=train_img_features_torch, 
                                   states=curr_state_torch)
            curr_state_torch = curr_state_torch.detach().cpu()
            train_img_features_torch = train_img_features_torch.detach().cpu()
            old_Q = old_Q.cpu().detach().reshape((num_data,h,w,parameters["ACTION_NUMBER"]))       # old_Q.shape=[BATCH_SIZE_IMG,6,32,32], 6 = ACTION_NUMBER
            old_Q = old_Q.permute((0, 3, 1, 2))
            
            action_max    = old_Q.argmax(axis=1)
            action_fusion = action_max # Shape = [BATCH_SIZE_IMG, h, w]
            
            # C2. Save the next state
            next_state = curr_state.clone()
            next_state[:,step_hv, :,:] = action_fusion
            
            ## C3 Compute the current error
            mask_select_end = action_fusion == parameters['ACTION_NUMBER'] - 1 # Mask of selecting the END for current timestep, t 
            mask_select_end = mask_select_end.to(torch.int8)
            mask_now_end    = mask_prev_end | mask_select_end              # Mask of selecting the end for all step [0, 1, ..., t]
            mask_now_end    = mask_now_end.to(torch.int8)
            
            # Libranet.A.shape=(8), action_fusion.shape=(BATCH_SIZE_IMG,h,w), Libranet.A[action_fusion].shape=(BATCH_SIZE, IMG, h,w)
            # Add the action number to count_rem if it is not at the end state
            count_rem  = count_rem + (1 - mask_select_end) * (1 - mask_prev_end) * \
                         (Libranet.A_mat[action_fusion.to(torch.int8)])                              
                                     
                 
            curr_state =  next_state.clone()
            mask_prev_end = mask_now_end.clone()   
            if (1-mask_now_end).sum()==0:  # If all end, break the loop
                break
        
        
        pred_count   = count_rem.reshape(num_data, -1).sum(dim=1)
        mae          = abs(actual_count-pred_count).sum()
        mse          = ((actual_count-pred_count)**2).sum()
        total_mae   += mae.item()
        total_mse   += mse.item()
        
    return total_mae/len(img_pt_list), total_mse/len(img_pt_list)
        


def sample_n_unique(sampling_f, batch_size):
    res = []
    while len(res) < batch_size:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res


class ReplayBuffer(object):
    def __init__(self, size, vector_len_fv,vector_len_hv,batch_size):
        
        self.size           = size
        self.batch_size     = batch_size
        self.next_idx       = 0
        self.num_in_buffer  = 0
        self.state_fv       = np.zeros((size, vector_len_fv))
        self.state_hv       = np.zeros((size, vector_len_hv))
        self.action         = np.zeros((size))
        self.optimal_action = np.zeros((size))
        self.reward         = np.zeros((size))
        self.next_state_hv  = np.zeros((size, vector_len_hv))
        self.done           = np.zeros((size))
        self.flag_full      = False

    def can_sample(self):
        return self.flag_full

    def out(self):
        
        assert self.can_sample()
        # Sample an index
        rng = np.random.default_rng()
        idxes = rng.choice(self.size, size=self.batch_size, replace=False)
        # idxes = sample_n_unique(lambda: random.randint(0, self.size  - 2), 
        #                         self.batch_size)
        
        # Sample a batch of data
        state_fv_batch       = self.state_fv[idxes]
        state_hv_batch       = self.state_hv[idxes]
        next_state_hv_batch  = self.next_state_hv[idxes]
        act_batch            = self.action[idxes]
        opt_act_batch        = self.optimal_action[idxes]
        rew_batch            = self.reward[idxes]
        done_mask            = self.done[idxes]

        return state_fv_batch,state_hv_batch, act_batch, opt_act_batch, rew_batch,next_state_hv_batch, done_mask

    def put(self, state_fv,state_hv, action, optimal_action, reward,  next_state_hv,  done):
            
        length=len(state_fv)
        if self.size-self.num_in_buffer>length:    # If the buffer is not full, place the information into buffers
            self.state_fv[self.num_in_buffer:self.num_in_buffer+length,:]      = state_fv
            self.state_hv[self.num_in_buffer:self.num_in_buffer+length,:]      = state_hv
            self.action[self.num_in_buffer:self.num_in_buffer+length]          = action
            self.optimal_action[self.num_in_buffer:self.num_in_buffer+length]  = optimal_action
            self.reward[self.num_in_buffer:self.num_in_buffer+length]          = reward
            self.next_state_hv[self.num_in_buffer:self.num_in_buffer+length,:] = next_state_hv
            self.done[self.num_in_buffer:self.num_in_buffer+length]            = done
            
            self.num_in_buffer=self.num_in_buffer+length
        else:                                      # If the buffer is full 
            self.flag_full = True
            buffer_int     = self.size-self.num_in_buffer      # Number of size left (Total_size - Num_in_buffer) 
            
            # Fill the information until full
            self.state_fv[self.num_in_buffer:self.size,:]       = state_fv[0:buffer_int,:]
            self.state_hv[self.num_in_buffer:self.size,:]       = state_hv[0:buffer_int,:]
            self.action[self.num_in_buffer:self.size]           = action[0:buffer_int]
            self.optimal_action[self.num_in_buffer:self.size]   = optimal_action[0:buffer_int]
            self.reward[self.num_in_buffer:self.size]           = reward[0:buffer_int]
            self.next_state_hv[self.num_in_buffer:self.size,:]  = next_state_hv[0:buffer_int,:]
            self.done[self.num_in_buffer:self.size]             = done[0:buffer_int]
            
            # Replace the first few observation with the information left
            buffer_int2=length-buffer_int                      # Length - (Total_size - Num_in_buffer) 
            self.state_fv[0:buffer_int2,:]      = state_fv[buffer_int:length,:]
            self.state_hv[0:buffer_int2,:]      = state_hv[buffer_int:length,:]
            self.action[0:buffer_int2]          = action[buffer_int:length]
            self.optimal_action[0:buffer_int2]  = optimal_action[buffer_int:length]
            self.reward[0:buffer_int2]          = reward[buffer_int:length]
            self.next_state_hv[0:buffer_int2,:] = next_state_hv[buffer_int:length,:]
            self.done[0:buffer_int2]            = done[buffer_int:length]
            
            self.num_in_buffer = buffer_int2                   # Set number in buffer = Length - (Total_size - Num_in_buffer) ?



