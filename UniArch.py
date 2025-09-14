#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import json
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torchmetrics
from senet import resnet18,getResNet
from sklearn.metrics import accuracy_score, f1_score,recall_score
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
from zeta.nn import MambaBlock
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


#x_train, x_test, subject_train, subject_test, act_train, act_test = train_test_split(spectros, subjects, activities,test_size=0.2)


# In[5]:


pf = "pickle.dat"
with open(pf, "rb") as f:
        x_train, x_test, subject_train, subject_test, act_train, act_test=pickle.load(f)
print(x_train.shape,x_test.shape)

def get_dataloader(x, y1, y2, batch_size=16):
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), 
                            torch.tensor(y1, dtype=torch.long), 
                            torch.tensor(y2, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_dl = get_dataloader(x_train, act_train, subject_train, batch_size=16)
test_dl = get_dataloader(x_test, act_test, subject_test, batch_size=16)


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


# In[31]:


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
    def __init__(self, num_gestures, num_persons, d_model=256,  nhead=8, num_layers=6,enc='xLSTM',dt='mciro'):
        super(UniTModel, self).__init__()

        # ResNet18 backbone
        #self.backbone = resnet18()
        #self.backbone.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.dt=dt

        resnet1 = getResNet([2,2,2,2],use_senet=False,in_ch=1)
        self.backbone= nn.Sequential(
            resnet1.conv1,
            resnet1.bn,
            resnet1.relu,
            resnet1.maxPool,
            resnet1.conv2,
            resnet1.conv3,
            resnet1.conv4
        )        

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Projection layer
        self.projection = nn.Linear(256, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder=None
        if enc=='xLSTM':
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
                num_blocks=5,
                embedding_dim=256,
                slstm_at=[1],

            )
            self.encoder =xLSTMBlockStack(cfg)
        elif enc=='Former':
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        elif enc=='Mamba':
            self.encoder = MambaBlock(dim=d_model, depth=1)

        # Task-specific heads
        self.gesture_head = nn.Linear(d_model, num_gestures)
        self.person_head = nn.Linear(d_model, num_persons)

    def forward(self, x):
        if self.dt=='micro':
            x=x[:,0:1,:,:]
            #print(x.size())
        elif self.dt=='range':
            x=x[:,1:2,:,:]
            #print(x.size())
        features = self.backbone(x)
        features = self.avgpool(features)
        features = features.view(features.size(0), -1)
        features = self.projection(features)

        # Add positional encoding
        features = self.pos_encoder(features.unsqueeze(0)).squeeze(0)
        #print(features.unsqueeze(1).size())
        # Pass through xLSTM
        F.normalize(features)
        encoded_features= self.encoder(features.unsqueeze(1))
        encoded_features=encoded_features.squeeze(1)
        F.normalize(encoded_features)
        gesture_output = self.gesture_head(encoded_features)
        person_output = self.person_head(encoded_features)

        return gesture_output, person_output


# In[32]:


def get_dataloader(x, y1, y2, batch_size=32):
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), 
                            torch.tensor(y1, dtype=torch.long), 
                            torch.tensor(y2, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Dynamic loss weighting
class DynamicWeightAverage:
    def __init__(self, num_tasks, momentum=0.9):
        self.num_tasks = num_tasks
        self.momentum = momentum
        self.weights = torch.ones(num_tasks) / num_tasks
        self.prev_losses = torch.ones(num_tasks)

    def update(self, losses):
        with torch.no_grad():
            w_hat = torch.div(losses, self.prev_losses)
            w_hat = torch.div(w_hat, torch.sum(w_hat))
            self.weights = self.momentum * self.weights + (1 - self.momentum) * w_hat
            self.prev_losses = losses
        return self.weights

def evaluate(model, data_loader, device, gesture_criterion, person_criterion):
    model.eval()
    gesture_preds, person_preds = [], []
    gesture_labels, person_labels = [], []
    gesture_loss_total = 0.0
    person_loss_total = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, g_labels, p_labels in data_loader:
            inputs, g_labels, p_labels = inputs.to(device), g_labels.to(device), p_labels.to(device)
            g_outputs, p_outputs = model(inputs)

            # Calculate losses
            g_loss = gesture_criterion(g_outputs, g_labels)
            p_loss = person_criterion(p_outputs, p_labels)

            # Accumulate losses
            gesture_loss_total += g_loss.item() * inputs.size(0)
            person_loss_total += p_loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            # Collect predictions and labels
            gesture_preds.extend(g_outputs.argmax(dim=1).cpu().numpy())
            person_preds.extend(p_outputs.argmax(dim=1).cpu().numpy())
            gesture_labels.extend(g_labels.cpu().numpy())
            person_labels.extend(p_labels.cpu().numpy())

    # Calculate average losses
    gesture_loss_avg = gesture_loss_total / total_samples
    person_loss_avg = person_loss_total / total_samples

    # Calculate accuracy and F1 scores
    gesture_acc = accuracy_score(gesture_labels, gesture_preds)
    gesture_f1 = f1_score(gesture_labels, gesture_preds, average='weighted')
    person_acc = accuracy_score(person_labels, person_preds)
    person_f1 = f1_score(person_labels, person_preds, average='weighted')
    gesture_recall = recall_score(gesture_labels, gesture_preds, average='weighted')
    person_recall = recall_score(person_labels, person_preds, average='weighted')

    return gesture_loss_avg, gesture_acc, gesture_f1,gesture_recall, person_loss_avg, person_acc, person_f1,person_recall


# In[33]:


def train_model(enc,dt,return_para=False):
    dwa = DynamicWeightAverage(num_tasks=2)
    train_dl = get_dataloader(x_train, act_train, subject_train, batch_size=16)
    test_dl = get_dataloader(x_test, act_test, subject_test, batch_size=32)
    # Create the model
    model = UniTModel(act_num, subject_num, enc=enc,dt=dt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if return_para:
        getSummary(model,(1,2,224,224))
        return

    gesture_criterion = nn.CrossEntropyLoss()
    person_criterion = nn.CrossEntropyLoss()
    num_epochs=200
    base_lr=0.0001
    shared_params = list(model.backbone.parameters()) +list(model.avgpool.parameters())+list(model.projection.parameters()) + list(model.encoder.parameters())
    gesture_params = list(model.gesture_head.parameters())
    person_params = list(model.person_head.parameters())

    optimizer = torch.optim.AdamW([
        {'params': shared_params, 'lr': base_lr},
        {'params': gesture_params, 'lr': base_lr * 0.8},  # Adjust these multipliers as needed
        {'params': person_params, 'lr': base_lr * 1.2}
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    metrics_log = pd.DataFrame(columns=['Epoch', "Val Gesture Loss","Val Person Loss", "Val Gesture Acc","Val Person Acc", "Val Gesture F1","Val Person F1",])

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, gesture_labels, person_labels in train_dl:
            inputs = inputs.to(device)
            gesture_labels = gesture_labels.to(device)
            person_labels = person_labels.to(device)

            gesture_outputs, person_outputs = model(inputs)

            gesture_loss = gesture_criterion(gesture_outputs, gesture_labels)
            person_loss = person_criterion(person_outputs, person_labels)

            # Dynamic loss weighting
#             loss_weights = dwa.update(torch.tensor([gesture_loss.item(), person_loss.item()]))
#             total_loss = loss_weights[0] * gesture_loss + loss_weights[1] * person_loss

            ##########
            total_loss = gesture_loss +  person_loss
            #############

            optimizer.zero_grad()
            total_loss.backward()

            # Gradient normalization
#             for name, param in model.named_parameters():
#                 if param.grad is not None:
#                     if 'gesture_head' in name:
#                         param.grad *= loss_weights[0]
#                     elif 'person_head' in name:
#                         param.grad *= loss_weights[1]

#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        # Evaluate on training set 
        train_gesture_loss_avg, train_gesture_acc,train_gesture_f1,train_gesture_recall, train_person_loss_avg, train_person_acc, train_person_f1,train_person_recall= evaluate(model, train_dl, device,
                                                                                                                                      gesture_criterion, person_criterion)

        # Evaluate on validation set
        val_gesture_loss_avg, val_gesture_acc, val_gesture_f1,val_gesture_recall, val_person_loss_avg, val_person_acc, val_person_f1,val_person_recall = evaluate(model, test_dl, device,
                                                                                                                             gesture_criterion, person_criterion)

        metrics_log = metrics_log.append({
                'Epoch': epoch + 1,
                'Val Gesture Loss':val_gesture_loss_avg,
                "Val Person Loss":val_person_loss_avg,
                "Val Gesture Acc":val_gesture_acc,
                "Val Person Acc":val_person_acc,
                "Val Gesture F1":val_gesture_f1,
                "Val Person F1":val_person_f1,
                "Val Gesture Recall":val_gesture_recall,
                "Val Person Recall":val_person_recall,

            }, ignore_index=True)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {total_loss/len(train_dl):.4f}")
        print(f"Train Gesture - Acc: {train_gesture_acc:.4f}, F1: {train_gesture_f1:.4f}")
        print(f"Train Person - Acc: {train_person_acc:.4f}, F1: {train_person_f1:.4f}")
        print(f"Val Gesture - Loss: {val_gesture_loss_avg:.4f}")
        print(f"Val Person - Loss: {val_person_loss_avg:.4f}")
        print(f"Val Gesture - Acc: {val_gesture_acc:.4f}, F1: {val_gesture_f1:.4f}, Recall: {val_gesture_recall:.4f}")
        print(f"Val Person - Acc: {val_person_acc:.4f}, F1: {val_person_f1:.4f}, Recall: {val_person_recall:.4f}")
        print("-----------------------------")
    log_name='res/Uni'+enc+"_"+dt+'_log.csv'
    print("Training completed!")
    metrics_log.to_csv(log_name, index=False)


# In[34]:


dts=['micro','range']
for dt in dts:
    train_model('xLSTM',dt)


# In[9]:


encs=['Mamba','Former','xLSTM']
#encs=['xLSTM']
for enc in encs:
    train_model(enc,True)


# In[ ]:




