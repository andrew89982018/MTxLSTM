#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import sys
import json
import pickle
import torch
from torch.nn import Sequential
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torchmetrics
from models.dvit import DeepViT
from models.dualvit import dualvit_s
from models.mambaout import mambaout_femto
from models.vit import ViT
from models.senet import resnet18
from models.resnet_cbam import resnet18_cbam
from models.van import van_b0,van_b1
from models.cait import CaiT
from models.densenet import densenet121
from models.swim import SwinTransformer
from models.wavevit import wavevit_s,WaveViT
from models.coatnet import coatnet_1
from models.maxvit import MaxViT
from vision_mamba import Vim
from sklearn.metrics import accuracy_score, f1_score,recall_score
from tqdm import tqdm  # For progress bar
import torch.nn.functional as F
import math
from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator
from collections.abc import Sequence


# In[6]:


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


# In[7]:


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


# In[8]:


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


# In[9]:


class MultiTaskModel(nn.Module):
    def __init__(self,backbone_name, channel_size,image_size, num_subjects, num_activities,args):
        super(MultiTaskModel, self).__init__()
        backbone=None
        if backbone_name=='DeepViT':
            backbone=DeepViT(image_size, args.patch_size, args.num_classes, args.dim, args.depth, args.heads, args.mlp_dim,channels=channel_size,dropout = 0.1,emb_dropout = 0.1)
        elif backbone_name=='ViT':
            backbone=ViT(image_size, args.patch_size, args.num_classes, args.dim, args.depth, args.heads, args.mlp_dim,channels=channel_size,dropout = 0.1,emb_dropout = 0.1)
        elif backbone_name=='Swim':
            backbone=SwinTransformer(num_classes = args.num_classes,in_chans=channel_size)
        elif backbone_name=='resnet':
            backbone=resnet18(num_classes=args.num_classes,use_senet=False,in_ch=channel_size)
        elif backbone_name=='densenet':
            backbone=densenet121(num_classes=args.num_classes,in_chans=channel_size)
        elif backbone_name=='vim':
            backbone = Vim(dim=128,dt_rank=16,dim_inner=128, d_state=64,num_classes=args.num_classes,image_size=image_size,patch_size=8,channels=channel_size, dropout=0.1,depth=1)
        elif backbone_name=='van':
            backbone=van_b1(in_chans=channel_size)
        elif backbone_name=='cait':
            backbone = CaiT(image_size = image_size,patch_size = args.patch_size,num_classes=args.num_classes,depth = 6,dim = 812,cls_depth = 2,heads = 12,mlp_dim = 2048,
                         in_chans=channel_size,dropout = 0.1,emb_dropout = 0.1,layer_dropout = 0.05)
        elif backbone_name=='wavevit':
            backbone=wavevit_s(num_classes=args.num_classes,in_chans=channel_size)
        elif backbone_name=='dualvit':
            backbone=dualvit_s(num_classes=args.num_classes,in_chans=channel_size)
        elif backbone_name=='mambaout':
            backbone=mambaout_femto(num_classes=args.num_classes,in_chans=channel_size)
        elif backbone_name=='coatnet':
            backbone=coatnet_1()
        elif backbone_name=='maxvit':
            backbone= MaxViT(num_classes = args.num_classes,dim_conv_stem = 32,  dim = 32, dim_head = 32, depth = (1,2,1),
                            window_size = 7, mbconv_expansion_rate = 32, mbconv_shrinkage_rate = 0.25, dropout = 0.1 )
        self.backbone = backbone
        if isinstance(self.backbone.head, Sequential):
            num_features = self.backbone.head[-1].in_features
        else:
            num_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()  # Remove the original classification layer

        # Define separate heads for each task
        self.gesture_head = nn.Linear(num_features, num_activities)
        self.person_head = nn.Linear(num_features, num_subjects)

    def forward(self, x):
        features = self.backbone(x)  # Extract features using the backbone
        gesture_output = self.gesture_head(features)  # Gesture recognition output
        person_output = self.person_head(features)  # Person identification output
        return gesture_output, person_output


# In[10]:


class Args:
    pass

args = Args()
args.patch_size = 16
args.num_classes = 2
args.dim = 1024
args.depth = 6
args.heads = 16
args.mlp_dim = 2048


# In[11]:


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


# In[12]:


def train_model(model_name,return_paras=True):

    # Create the model
    model = MultiTaskModel(model_name,2,224,subject_num, act_num,args)

    gesture_criterion = nn.CrossEntropyLoss()
    person_criterion = nn.CrossEntropyLoss()
    num_epochs=200
    base_lr=0.0001
    shared_params = list(model.backbone.parameters())
    gesture_params = list(model.gesture_head.parameters())
    person_params = list(model.person_head.parameters())

    optimizer = torch.optim.AdamW([
        {'params': shared_params, 'lr': base_lr},
        {'params': gesture_params, 'lr': base_lr * 0.8},  # Adjust these multipliers as needed
        {'params': person_params, 'lr': base_lr * 1.2}
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)



    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if return_paras:
        getSummary(model,(1,2,224,224))
        return


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
    log_name='res/SoTa_'+model_name+'_log.csv'


    print("Training completed!")
    metrics_log.to_csv(log_name, index=False)


# In[16]:


models=[ 'maxvit','coatnet','mambaout','wavevit','dualvit', "Swim", 'resnet', 'densenet','van' ,'cait','DeepViT','ViT'] # 'vim',# 
#models=['densenet']
for model in models:
    print(model,'############################################')
    train_model(model)


# In[ ]:




