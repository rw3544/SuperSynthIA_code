import numpy as np
import os
import sys
import time
import numpy as np
import zarr
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn.functional as F
from tqdm import tqdm
from torch import autograd, nan
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime

from dataset import *
from utils import *
from encoder import *



from loguru import logger



# train each epoch
def train_epoch(data_loader, model, optimizer, rlrop, epoch, is_train, GLOBAL_DEVICE):
    start_time = time.time()

    losses = 0.0
    torch.set_grad_enabled(is_train)

    # Set model mode
    if is_train:
        model = model.train()
    else:
        model = model.eval()

    pbar = tqdm(data_loader, ncols=50)

    
    for i, batch in enumerate(pbar):

        # X: N,C,H,W   all depadded
        # y: N,C,H,W     with correct shape
        # dmask: N,C,H,W   0 is off-disk
        X = batch['X'][0].float().to(GLOBAL_DEVICE)
        y = batch['y'][0].float().to(GLOBAL_DEVICE)
        mask = batch['mask'][0].to(GLOBAL_DEVICE)
        dmask = batch['disk_mask'][0].to(GLOBAL_DEVICE)
        

        
        # For X, remove all nans and apply disk mask
        X = X.masked_fill_(torch.isnan(X), 0)
        X = X.masked_fill_(~dmask, 0)
        
        # For y, apply mask only as it already contians dmask
        y = y.masked_fill_(~mask, 0)



        
        '''
        print()
        print(f'{X.device, y.device, mask.device}')
        print(X.shape)
        print(y.shape)
        print(mask.shape)
        '''
        

        # Forward + loss
        optimizer.zero_grad()

        pred = model(X)
        #pred = F.log_softmax(pred, dim=1)
        #pred = pred.reshape(1, bins, -1).permute(0,2,1).reshape(-1, bins)
        
        #print((pred-y))
        #print(pred[:,:,200:220, 200:220])


        #with autograd.detect_anomaly():
        #loss_func = torch.nn.MSELoss()
        loss_func = torch.nn.MSELoss(reduction='none')
        loss_tensor = loss_func(pred,y)
        #print(loss_tensor.isnan().sum())
        #print((~loss_tensor.isnan()).sum())
        #mseloss = np.longdouble(0)
        mseloss = (loss_tensor.masked_fill_(~mask, 0))

        #print(mseloss[:,:,200:220, 200:220])

        #print(torch.max(mseloss))
        mseloss = mseloss.sum()
        #print(mseloss)
        #print()
        #print(mseloss)
        
        nonzero_value = mask.sum()
        
        loss = mseloss/nonzero_value
        
        

        if torch.isnan(loss) == True:
            print()
            print(i)
            print()
            quit()

        
        if is_train:
            loss.backward()
            optimizer.step()
        
        # Logging
        losses += float(loss.detach())
        #step = (i + epoch*epoch_len)

        pbar.set_description(
            '{} epoch {}: itr {:<6}/ {}- batch loss: {}- avg loss: {:.4f}- dt {:.4f}'
          .format('TRAIN' if is_train else 'VAL  ', 
                  epoch, i * data_loader.batch_size, len(data_loader) * data_loader.batch_size, # steps
                  loss / data_loader.batch_size, losses / (i+1), # print batch loss and avg loss
                  time.time() - start_time)) # batch time
        

    avg_loss = losses / (i+1)
    if is_train == False:
        rlrop.step(avg_loss)
    
    return avg_loss

# train each epoch
def train_classification_epoch(
    data_loader, 
    model, 
    optimizer, 
    rlrop, 
    epoch, 
    is_train, 
    GLOBAL_DEVICE, 
    bins,
    h_flip,
    h_flip_neg,
    h_flip_AZ):
    start_time = time.time()

    losses = 0.0
    torch.set_grad_enabled(is_train)

    # Set model mode
    if is_train:
        model = model.train()
    else:
        model = model.eval()

    pbar = tqdm(data_loader, ncols=200)

    
    for i, batch in enumerate(pbar):
        
        # X: N,C,H,W   all depadded
        # y: N,C,H,W     with correct shape
        # dmask: N,C,H,W   0 is off-disk
        X = batch['X'][0].float().to(GLOBAL_DEVICE)
        y = batch['y'][0].float().to(GLOBAL_DEVICE)
        mask = batch['mask'][0].to(GLOBAL_DEVICE)
        dmask = batch['disk_mask'][0].to(GLOBAL_DEVICE)
        
        #print()
        # Flip if necessary 
        if h_flip == True:
            # Get random number to decide
            r_num = random.random()
            
            if r_num > 0.5:
                # Flip X and y
                #print('Flip')

                X = torch.flip(X, dims=[3])
                y = torch.flip(y, dims=[3])
                mask = torch.flip(mask, dims=[3])
                dmask = torch.flip(dmask, dims=[3])

                # If Bp, need to flip the sign
                if h_flip_neg == True:
                    assert(h_flip == True)
                    #print('Flip sign')
                    y = -y
                if h_flip_AZ == True:
                    assert(h_flip == True)
                    assert(h_flip_neg == False)
                    y = 180 - y
                    
        
        # For X, remove all nans and apply disk mask
        X = X.masked_fill_(torch.isnan(X), 0)
        X = X.masked_fill_(~dmask, 0)
        
        # For y, apply mask only as it already contians dmask
        
        y = y.masked_fill_(~mask, 0)

        
        y = encoder(y, bins)
        # masked y in (N,C,H,W)
        
        y = y.squeeze()
        bin_mask = mask.squeeze()

        num_channel = y.shape[0]
        y = y[torch.arange(num_channel)[:,None], bin_mask]
        
        
        # Forward + loss
        optimizer.zero_grad()

            
        

        pred = model(X)
        pred = pred.squeeze()
        
        
        pred = pred[torch.arange(num_channel)[:,None], bin_mask]
        
        
        #pred = F.log_softmax(pred, dim=1)
        #pred = pred.reshape(1, bins, -1).permute(0,2,1).reshape(-1, bins)
        
        #print((pred-y))
        #print(pred[:,:,200:220, 200:220])


        #with autograd.detect_anomaly():

        pred = F.log_softmax(pred, dim=0)
        
        pred = pred.permute(1,0)
        y = y.permute(1,0)
        
        loss_func = torch.nn.KLDivLoss(reduction='sum')
        loss = loss_func(pred,y)
        
        
        
        #print(loss_tensor.isnan().sum())
        #print((~loss_tensor.isnan()).sum())
        #mseloss = np.longdouble(0)
        
        #print(mseloss[:,:,200:220, 200:220])
        

        if torch.isnan(loss) == True:
            print()
            print(i)
            print()
            quit()

        
        if is_train:
            loss.backward()
            optimizer.step()
        
        # Logging
        losses += float(loss.detach())
        #step = (i + epoch*epoch_len)

        pbar.set_description(
            '{} epoch {}: itr {:<6}/ {}- batch loss: {}- avg loss: {:.4f}- dt {:.4f}'
          .format('TRAIN' if is_train else 'VAL  ', 
                  epoch, i * data_loader.batch_size, len(data_loader) * data_loader.batch_size, # steps
                  loss / data_loader.batch_size, losses / (i+1), # print batch loss and avg loss
                  time.time() - start_time)) # batch time
        

        
    avg_loss = losses / (i+1)
    if is_train == False:
        rlrop.step(avg_loss)
    
    return avg_loss



# Train the network
def train(
    X_TRAIN_DIR:str,
    X_VAL_DIR:str,
    y_TRAIN_DIR:str,
    y_VAL_DIR:str,
    X_NORM_DIR:str,
    y_NORM_DIR:str,
    disk_mask_TRAIN_DIR: str,
    disk_mask_VAL_DIR: str,
    model,
    optimizer,
    rlrop,
    MODEL_SAVE_PATH,
    GLOBAL_DEVICE,
    lr,
    OBS_TRAIN_DIR_LIST = None,
    OBS_TRAIN_NORM_DIR_LIST = None,
    OBS_VAL_DIR_LIST = None,
    bins = [],
    whether_continue_training = None,
    main_h_flip = False,
    ):
    ########
    # Args:
    #
    #   OBS_DIR_LIST: list of DIR of OBS ZARR
    #   OBS_NORM_DIR_LIST: list of DIR of OBS_NORM.npy
    #   
    ########
    model = model.to(GLOBAL_DEVICE)
    
    train_dataset = None
    VAL_dataset = None
    if OBS_TRAIN_DIR_LIST == None:
        assert(OBS_TRAIN_NORM_DIR_LIST == None)
        assert(OBS_VAL_DIR_LIST == None)
    else:
        # Extra_layer
        assert(OBS_TRAIN_NORM_DIR_LIST != None)
        assert(OBS_VAL_DIR_LIST != None)
    if bins == []:
        train_dataset = HMI_Pipeline(X_TRAIN_DIR, y_TRAIN_DIR, X_NORM_DIR, y_NORM_DIR, disk_mask_TRAIN_DIR, 'train', GLOBAL_DEVICE, False, OBS_TRAIN_DIR_LIST, OBS_TRAIN_NORM_DIR_LIST)
        VAL_dataset = HMI_Pipeline(X_VAL_DIR, y_VAL_DIR, X_NORM_DIR, y_NORM_DIR, disk_mask_VAL_DIR, 'validation', GLOBAL_DEVICE, False, OBS_VAL_DIR_LIST, OBS_TRAIN_NORM_DIR_LIST)
    else:
        train_dataset = HMI_Pipeline(X_TRAIN_DIR, y_TRAIN_DIR, X_NORM_DIR, y_NORM_DIR, disk_mask_TRAIN_DIR, 'train', GLOBAL_DEVICE, False, OBS_TRAIN_DIR_LIST, OBS_TRAIN_NORM_DIR_LIST, True)
        VAL_dataset = HMI_Pipeline(X_VAL_DIR, y_VAL_DIR, X_NORM_DIR, y_NORM_DIR, disk_mask_VAL_DIR, 'validation', GLOBAL_DEVICE, False, OBS_VAL_DIR_LIST, OBS_TRAIN_NORM_DIR_LIST, True)
    
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=12, pin_memory=True)
    val_loader = DataLoader(VAL_dataset, batch_size=1, num_workers=12, pin_memory=True)


    # Meta training loop
    train_losses, val_losses = [], []
    min_loss = sys.maxsize

    
    # Check if load models
    # Automatically load the model with greatest epoch
    have_prev_model = os.path.exists(MODEL_SAVE_PATH)
    if os.path.exists(MODEL_SAVE_PATH) and (whether_continue_training == None):
        arr = sorted(os.listdir(MODEL_SAVE_PATH))
        assert(len(arr) == 1)
        print(arr)
        print()
        print('Model Already exists, continue? [yes]')
        s_in = str(input())
        if s_in != 'yes':
            print('Exit')
            quit()
        whether_continue_training = True
        
    if have_prev_model == True and whether_continue_training != True:
        quit()
    
    
    if have_prev_model == True:
        arr = sorted(os.listdir(MODEL_SAVE_PATH))
        assert(len(arr) == 1)
        MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, arr[0])
    else:    
        MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, str(datetime.now()).replace(" ", "_"))
        utils_make_dir(MODEL_SAVE_PATH)
        param_PATH = os.path.join(MODEL_SAVE_PATH, 'parameter.txt')

            
        with open(param_PATH, 'w') as f:
            weight_decay, eps = None, None 
            
            for param_group in optimizer.param_groups:
                weight_decay = param_group['weight_decay']
                eps = param_group['eps']
            
            f.write(f'lr: {lr}, weight_decay: {weight_decay}, eps={eps}')
            f.write('\n')
            f.write(f'X_TRAIN_DIR: {X_TRAIN_DIR} \n')
            f.write(f'X_VAL_DIR: {X_VAL_DIR} \n')
            f.write(f'y_TRAIN_DIR: {y_TRAIN_DIR} \n')
            f.write(f'y_VAL_DIR: {y_VAL_DIR} \n')
            f.write(f'X_NORM_DIR: {X_NORM_DIR} \n')
            f.write(f'y_NORM_DIR: {y_NORM_DIR} \n')
            f.write(f'MODEL_SAVE_PATH: {MODEL_SAVE_PATH} \n')
            f.write(f'OBS_TRAIN_DIR_LIST: {OBS_TRAIN_DIR_LIST} \n')
            f.write(f'OBS_TRAIN_NORM_DIR_LIST: {OBS_TRAIN_NORM_DIR_LIST} \n')
            f.write(f'OBS_VAL_DIR_LIST: {OBS_VAL_DIR_LIST} \n')
            f.write(str(model))
    
    LATEST_MODEL_DIR = None 
    LATEST_EPOCH_NUM = 0 
    
    # Load from the latest model
    if whether_continue_training == True and have_prev_model == True:
        TRAIN_LOSS_NPY_DIR = os.path.join(MODEL_SAVE_PATH, 'Train_loss.npy')
        prev_train_loss_arr = np.load(TRAIN_LOSS_NPY_DIR)
        VAL_LOSS_NPY_DIR = os.path.join(MODEL_SAVE_PATH, 'Validation_loss.npy')
        prev_val_loss_arr = np.load(VAL_LOSS_NPY_DIR)
        
    
        
        for epoch_num in range(len(prev_train_loss_arr) + 1):
            model_out_path = os.path.join(MODEL_SAVE_PATH, f'epoch={epoch_num}.checkpoint.pth')
            if os.path.exists(model_out_path):
                LATEST_MODEL_DIR = model_out_path  
                LATEST_EPOCH_NUM = epoch_num  
        print(f'Loading model from {LATEST_MODEL_DIR}')
        model.load_state_dict(torch.load(LATEST_MODEL_DIR, map_location=GLOBAL_DEVICE)['model'])
        optimizer.load_state_dict(torch.load(LATEST_MODEL_DIR, map_location='cpu')['optimizer'])
        rlrop = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
        
        min_loss = np.min(prev_val_loss_arr)
        
        LATEST_EPOCH_NUM += 1
        train_losses = prev_train_loss_arr[:LATEST_EPOCH_NUM]
        val_losses = prev_val_loss_arr[:LATEST_EPOCH_NUM]
        train_losses = train_losses.tolist()
        val_losses = val_losses.tolist()
    
    
    # Early stopping
    patience_epoch = 0 
    
    logger.info('Start Training')
    for epoch in range(LATEST_EPOCH_NUM, 10000):
        if bins == []:
            train_losses.append(train_epoch(train_loader, model, optimizer, rlrop, epoch, True, GLOBAL_DEVICE))
            val_losses.append(train_epoch(val_loader, model, optimizer, rlrop, epoch, False, GLOBAL_DEVICE))
        else:
            h_flip_neg = False 
            h_flip_AZ = False 
            if main_h_flip == True and 'spDisambig_Bp' in y_TRAIN_DIR:
                h_flip_neg = True 
            if main_h_flip == True and 'spInv_Field_Azimuth' in y_TRAIN_DIR:
                h_flip_AZ = True
            train_losses.append(train_classification_epoch(train_loader, model, optimizer, rlrop, epoch, True, GLOBAL_DEVICE, bins, main_h_flip, h_flip_neg, h_flip_AZ))
            val_losses.append(train_classification_epoch(val_loader, model, optimizer, rlrop, epoch, False, GLOBAL_DEVICE, bins, main_h_flip, h_flip_neg, h_flip_AZ))
            
        
        logger.info(f'Epoch: {epoch}')
        if val_losses[-1] < min_loss:
            min_loss = val_losses[-1]
            patience_epoch = 0
            model_out_path = os.path.join(MODEL_SAVE_PATH, f'epoch={epoch}.checkpoint.pth')
            torch.save({'model': ((model.to('cpu')).state_dict()), 'optimizer': (optimizer.state_dict())}, model_out_path)
            model = model.to(GLOBAL_DEVICE)
            print('saved model..\t\t\t {}'.format(model_out_path))
        else:
            patience_epoch += 1
            print('--> loss failed to decrease {} epochs..\t\t\tthreshold is {}, {} all..{}'.format(patience_epoch, 6, val_losses, min_loss))
            # Decrease learning rate
            '''
            if patience_epoch == 5 or patience_epoch == 25:
                lr = lr/5
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                
                param_PATH = os.path.join(MODEL_SAVE_PATH, 'Updated_parameter.txt')
                with open(param_PATH, 'a') as f:
                    for param_group in optimizer.param_groups:
                        tmp_lr = param_group['lr']
                        f.write(f'lr: {tmp_lr} ')
                        f.write(f'\n ')
            '''


            if patience_epoch > 50: break
        assert(epoch + 1 == len(train_losses))
        assert(epoch + 1 == len(val_losses))
        train_PATH = os.path.join(MODEL_SAVE_PATH, 'Train_loss.npy')
        np.save(train_PATH, train_losses)

        validation_PATH = os.path.join(MODEL_SAVE_PATH, 'Validation_loss.npy')
        np.save(validation_PATH, val_losses)
        
        plt.figure()
        plt.plot(train_losses, color='r', label='Train_Loss')
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.plot(val_losses, color='black', label='Val_Loss')
        plt.legend()
        plt.savefig(os.path.join(MODEL_SAVE_PATH, "loss.png"))
        plt.close()





