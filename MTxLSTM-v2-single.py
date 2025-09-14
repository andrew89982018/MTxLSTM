#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import json
import pickle
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torchmetrics
#from torchvision.models import resnet18
from senet import resnet18,getResNet
from sklearn.metrics import accuracy_score, f1_score, recall_score
from tqdm import tqdm  # For progress bar
import torch.nn.functional as F
import math
#from torchxlstm import sLSTM, mLSTM, xLSTM 
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)
from FANLayer import FANLayer
from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator


# In[2]:


rootdir="data"
#filehandler = open(os.path.join(rootdir, "spectros_choose.pl"),"rb")
filehandler = open(os.path.join(rootdir, "spectros.pl"),"rb")
spectros=pickle.load(filehandler)
filehandler.close()

#filehandler = open(os.path.join(rootdir, "subjects_choose.pl"),"rb")
filehandler = open(os.path.join(rootdir, "subjects.pl"),"rb")
subjects=pickle.load(filehandler)
filehandler.close()

#filehandler = open(os.path.join(rootdir, "activities_choose.pl"),"rb")
filehandler = open(os.path.join(rootdir, "activities.pl"),"rb")
activities=pickle.load(filehandler)
filehandler.close()
print(len(subjects),len(activities), spectros.shape)

le = LabelEncoder()
subjects=le.fit_transform(subjects)
activities=le.fit_transform(activities)
spectros=np.transpose(spectros, (0, 3, 1,2))


# In[3]:


print(np.unique(subjects,return_counts=True))
print(np.unique(activities,return_counts=True))
subject_num=len(np.unique(subjects))
act_num=len(np.unique(activities))
print(subject_num, act_num)


# In[4]:


x_train, x_test, subject_train, subject_test, act_train, act_test = train_test_split(spectros, subjects, activities,test_size=0.2)


# In[5]:


def get_dataloader(x, y1, y2, batch_size=32):
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), 
                            torch.tensor(y1, dtype=torch.long), 
                            torch.tensor(y2, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_dl = get_dataloader(x_train, act_train, subject_train, batch_size=32)
test_dl = get_dataloader(x_test, act_test, subject_test, batch_size=256)


# In[6]:


def getSummary(model, inputshape):
    with get_accelerator().device('cuda:0'):
        flops, macs, params = get_model_profile(model=model, # model
                                        input_shape=inputshape, 
                                        args=None, 
                                        kwargs=None, 
                                        print_profile=True,
                                        detailed=True,
                                        module_depth=-1, 
                                        top_modules=2,
                                        warm_up=10, 
                                        as_string=True, 
                                        output_file=None,
                                        ignore_modules=None) 
    print('#####################################################')


# In[7]:


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class UniTModel(nn.Module):
    def __init__(self, num_gestures, num_persons, d_model=256,layers=[2,2,2,2],block_num=3,task='gesture'):
        super(UniTModel, self).__init__()

        self.task=task

        layers=layers.copy()
        stages=len(layers)
        while len(layers)<4:
            layers.append(1)

        resnet1 = getResNet(layers,use_senet=False,in_ch=1)
        resnet2 = getResNet(layers,use_senet=False,in_ch=1)

        # Modify the first conv layer to accept single channel input
        resnet1.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet2.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.branch1_before_last = nn.Sequential(
            resnet1.conv1,
            resnet1.bn,
            resnet1.relu,
            resnet1.maxPool,
            #resnet1.conv2,
            #resnet1.conv3
        )

        self.branch2_before_last = nn.Sequential(
            resnet2.conv1,
            resnet2.bn,
            resnet2.relu,
            resnet2.maxPool,
            #resnet2.conv2,
            #resnet2.conv3
        )

        for i in range(stages):
            m=i+2
            layer1="resnet1.conv"+str(m)
            layer2="resnet2.conv"+str(m)
            self.branch1_before_last.append(eval(layer1))
            self.branch2_before_last.append(eval(layer2))

        print(self.branch2_before_last)
        #self.branch1_last = resnet1.conv4
        #self.branch2_last = resnet2.conv4

        # Create new layers for after concatenation
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        # Projection layer
        self.projection = nn.Linear(128*(2**(stages-1)), d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda",
                    num_heads=4,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=256,
            num_blocks=block_num,
            embedding_dim=256,
            slstm_at=[1],

        )
        self.xlstm =xLSTMBlockStack(cfg)

        # Task-specific heads
        self.gesture_head = nn.Linear(d_model, num_gestures)
        self.person_head = nn.Linear(d_model, num_persons)

    def forward(self, x):
        x1 = x[:, 0:1, :, :]  # First channel
        x2 = x[:, 1:2, :, :]  # Second channel
        x1 = self.branch1_before_last(x1)
        x2 = self.branch2_before_last(x2)

        x = torch.cat((x1, x2), dim=1)  # Concatenate along channel dimension
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        features = self.projection(x)
        features = self.pos_encoder(features.unsqueeze(0)).squeeze(0)
        F.normalize(features)
        encoded_features= self.xlstm(features.unsqueeze(1))
        encoded_features=encoded_features.squeeze(1)
        F.normalize(encoded_features)
        #print(encoded_features.size())
        # Task-specific predictions
        if self.task=='gesture':
            output= self.gesture_head(encoded_features)
        elif self.task=='person':
            output = self.person_head(encoded_features)
        #gesture_output = self.gesture_head(encoded_features)
        #person_output = self.person_head(encoded_features) 
        return output


# In[8]:


def evaluate(model, data_loader, device, criterion, task):
    model.eval()
    preds = []
    labels = []
    loss_total = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, g_labels, p_labels in data_loader:
            inputs, g_labels, p_labels = inputs.to(device), g_labels.to(device), p_labels.to(device)
            outputs = model(inputs)

            # Calculate losses
            if task=='gesture': 
                loss = criterion(outputs, g_labels)
            elif task=='person': 
                loss = criterion(outputs, p_labels)

            # Accumulate losses
            loss_total += loss.item() * inputs.size(0)
            #person_loss_total += p_loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            # Collect predictions and labels
            preds.extend(outputs.argmax(dim=1).cpu().numpy())
            if task=='gesture': 
                labels.extend(g_labels.cpu().numpy())
            elif task=='person':
                labels.extend(p_labels.cpu().numpy())

    # Calculate average losses
    loss_avg = loss_total / total_samples

    # Calculate accuracy and F1 scores
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')

    return loss_avg, acc, f1,recall


# In[9]:


def train_model(layers,block_num,task,return_para=False):
    model = UniTModel(act_num, subject_num,layers=layers,block_num=block_num,task=task)

    # Define loss functions and optimizer
    criterion = nn.CrossEntropyLoss()
    num_epochs=200
    base_lr=0.0001

    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': base_lr}
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if return_para:
        getSummary(model,(1,2,224,224))
        return

    metrics_log = pd.DataFrame(columns=['Epoch', "Val Loss", "Val Acc", "Val F1","Val Recall"])

    for epoch in range(num_epochs):
        model.train()
        for inputs, gesture_labels, person_labels in train_dl:
            inputs = inputs.to(device)
            gesture_labels = gesture_labels.to(device)
            person_labels = person_labels.to(device)

            outputs = model(inputs)

            if task=='gesture': 
                loss = criterion(outputs, gesture_labels)
            elif task=='person': 
                loss = criterion(outputs, person_labels)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        # Evaluate on training set 
        train_loss_avg, train_acc,train_f1,train_recall= evaluate(model, train_dl, device,criterion,task)

        # Evaluate on validation set
        val_loss_avg, val_acc, val_f1,val_recall = evaluate(model, test_dl, device,criterion,task)

        metrics_log = metrics_log.append({
                'Epoch': epoch + 1,
                'Val Loss':val_loss_avg,
                "Val Acc":val_acc,
                "Val F1":val_f1,
                "Val Recall":val_recall,
            }, ignore_index=True)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss_avg:.4f}")
        print(f"Train Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss_avg:.4f}")
        print(f"Val Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Recall: {val_recall:.4f}")
        print("-----------------------------")

    print("Training completed!")
    log_name='res/MTxLSTM-Single'+'_'+str(layers)+'_'+str(block_num)+'_'+task+".csv"
    metrics_log.to_csv(log_name, index=False)


# In[ ]:


#layers=[[1,1],[1],[2,2],[2,2,2],[2,2,2,2],[3, 4, 6, 3]]
#block_nums=[2,3,5,7]
layers=[[2,2,2]]
block_nums=[5]
tasks=['gesture','person']
for layer in layers:
    for k in block_nums:
        for t in tasks:
            print(layer,k,t)
            train_model(layer,k,t,return_para=True)


# In[ ]:




