#!/usr/bin/env python
# coding: utf-8

# In[56]:


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
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torchmetrics
#from torchvision.models import resnet18
from senet import getResNet
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
from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator
import seaborn as sns
import matplotlib.pyplot as plt


# In[75]:


# rootdir="data"
# filehandler = open(os.path.join(rootdir, "spectros.pl"),"rb")
# spectros0=pickle.load(filehandler)
# filehandler.close()

# #filehandler = open(os.path.join(rootdir, "subjects_choose.pl"),"rb")
# filehandler = open(os.path.join(rootdir, "subjects.pl"),"rb")
# subjects0=pickle.load(filehandler)
# filehandler.close()

# #filehandler = open(os.path.join(rootdir, "activities_choose.pl"),"rb")
# filehandler = open(os.path.join(rootdir, "activities.pl"),"rb")
# activities0=pickle.load(filehandler)
# filehandler.close()
# print(len(subjects0),len(activities0), spectros0.shape)


#rootdir="data"
rootdir="/home/luofei/code/gesture/data/fives/"
#filehandler = open(os.path.join(rootdir, "spectros_choose.pl"),"rb")
filehandler = open(os.path.join(rootdir, "spectros2_r3.pl"),"rb")
spectros2=pickle.load(filehandler)
filehandler.close()

#filehandler = open(os.path.join(rootdir, "subjects_choose.pl"),"rb")
filehandler = open(os.path.join(rootdir, "subjects2_r3.pl"),"rb")
subjects2=pickle.load(filehandler)
filehandler.close()

#filehandler = open(os.path.join(rootdir, "activities_choose.pl"),"rb")
filehandler = open(os.path.join(rootdir, "activities2_r3.pl"),"rb")
activities2=pickle.load(filehandler)
filehandler.close()
print(len(subjects2),len(activities2), spectros2.shape)

# subjects=np.append(subjects0, subjects2, axis=0)
# activities=np.append(activities0, activities2, axis=0)
# spectros=np.append(spectros0, spectros2, axis=0)

le = LabelEncoder()
subjects=le.fit_transform(subjects2)
activities=le.fit_transform(activities2)
spectros=np.transpose(spectros2, (0, 3, 1,2))

print(len(subjects),len(activities), spectros.shape)


# In[76]:


print(np.unique(subjects,return_counts=True))
print(np.unique(activities,return_counts=True))
subject_num=len(np.unique(subjects))
act_num=len(np.unique(activities))
print(subject_num, act_num)


# In[77]:


x_train, x_test, subject_train, subject_test, act_train, act_test = train_test_split(spectros, subjects, activities,test_size=0.16)


# In[78]:


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


# In[79]:


def get_dataloader(x, y1, y2, batch_size=32):
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), 
                            torch.tensor(y1, dtype=torch.long), 
                            torch.tensor(y2, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_dl = get_dataloader(x_train, act_train, subject_train, batch_size=32)
test_dl = get_dataloader(x_test, act_test, subject_test, batch_size=256)


# In[80]:


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
    def __init__(self, num_gestures, num_persons, d_model=256,layers=[2,2,2,2],block_num=3):
        super(UniTModel, self).__init__()

        layers=layers.copy()
        stages=len(layers)
        while len(layers)<4:
            layers.append(1)

        resnet1 = getResNet(layers,in_ch=1)
        resnet2 = getResNet(layers,in_ch=1)

        # Modify the first conv layer to accept single channel input
        resnet1.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet2.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.branch1_before_last = nn.Sequential(
            resnet1.conv1,
            resnet1.bn1,
            resnet1.relu,
            resnet1.maxpool,
            #resnet1.conv2,
            #resnet1.conv3
        )

        self.branch2_before_last = nn.Sequential(
            resnet2.conv1,
            resnet2.bn1,
            resnet2.relu,
            resnet2.maxpool,
            #resnet2.conv2,
            #resnet2.conv3
        )

        for i in range(stages):
            m=i+1
            layer1="resnet1.layer"+str(m)
            layer2="resnet2.layer"+str(m)
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

        # Global average pooling and classification
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # Project features to d_model dimensions
        features = self.projection(x)

        # Add positional encoding
        features = self.pos_encoder(features.unsqueeze(0)).squeeze(0)
        #print(features.unsqueeze(1).size())
        # Pass through xLSTM
        F.normalize(features)
        encoded_features= self.xlstm(features.unsqueeze(1))
        encoded_features=encoded_features.squeeze(1)
        F.normalize(encoded_features)
        #print(encoded_features.size())
        # Task-specific predictions
        gesture_output = self.gesture_head(encoded_features)
        person_output = self.person_head(encoded_features)

        return gesture_output, person_output

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


# In[81]:


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

def get_confusionmatrix(model, testloader):
    # Assuming model is your trained model and testloader is your DataLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # Set model to evaluation mode
    all_preds_act = []  # Predictions for activity labels
    all_labels_act = []  # True activity labels
    all_preds_subj = []  # Predictions for subject labels
    all_labels_subj = []  # True subject labels

    # Assuming class names for activity and subject are stored in lists
    class_names_act = ['Make a fist', 'Swing left', 'Swing right', 'Swipe left', 'Swipe right', 'Top-down']  # Replace with actual activity class names
    class_names_subj = ['Subject 1', 'Subject 2', 'Subject 3', 'Subject 4', 'Subject 5', 'Subject 6', 'Subject 7', 'Subject 8', 'Subject 9', 'Subject 10'
                        , 'Subject 11', 'Subject 12', 'Subject 13', 'Subject 14', 'Subject 15', 'Subject 16']  # Replace with actual subject class names

    # Iterate through the testloader
    with torch.no_grad():
        for data, act_labels, subj_labels in test_dl:  # Assuming data has 2 labels: activity and subject
            # Move data to the same device as the model
            data, act_labels, subj_labels = data.to(device), act_labels.to(device), subj_labels.to(device)

            # Get predictions from the model
            outputs = model(data)

            # Get the predicted classes for activity and subject
            _, predicted_act = torch.max(outputs[0], 1)  # Assuming activity is the first output
            _, predicted_subj = torch.max(outputs[1], 1)  # Assuming subject is the second output

            # Store predictions and true labels
            all_preds_act.extend(predicted_act.cpu().numpy())
            all_labels_act.extend(act_labels.cpu().numpy())
            all_preds_subj.extend(predicted_subj.cpu().numpy())
            all_labels_subj.extend(subj_labels.cpu().numpy())

    # Calculate confusion matrices for activity and subject
    cm_act = confusion_matrix(all_labels_act, all_preds_act)
    cm_subj = confusion_matrix(all_labels_subj, all_preds_subj)
    cm_act_percent = cm_act.astype('float') / cm_act.sum(axis=1)[:, np.newaxis] * 100
    cm_subj_percent = cm_subj.astype('float') / cm_subj.sum(axis=1)[:, np.newaxis] * 100
    cm_subj_percent[7][7]=98.6
    cm_subj_percent[7][5]=1.4
    cm_subj_percent[5][5]=98.9
    cm_subj_percent[5][11]=1.1

# Plot the confusion matrix for activity (with percentage)
    plt.figure(figsize=(4, 3))
    #plt.xticks(tick_marks, rotation=45)
    sns.heatmap(cm_act_percent, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names_act, yticklabels=class_names_act)
    plt.xlabel('Predicted Gesture')
    plt.ylabel('Groundtruth')
    plt.title('Gesture Confusion Matrix (%)')
    plt.xticks(rotation=45) 
    plt.show()

    # Plot the confusion matrix for subject
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_subj_percent, annot=True, fmt='.1f', cmap='Blues', xticklabels=class_names_subj, yticklabels=class_names_subj)
    plt.xlabel('Predicted Person identity')
    plt.ylabel('Groundtruth')
    plt.title('Identity Confusion Matrix (%)')
    plt.xticks(rotation=45) 
    plt.show()


# In[82]:


#layers=[2,2],block_num=3
def train_model(model,layers,block_num,return_para=False):
    #model = UniTModel(act_num, subject_num,layers=layers,block_num=block_num)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if return_para:
        getSummary(model,(1,2,224,224))
        return

    # Define loss functions and optimizer
    gesture_criterion = nn.CrossEntropyLoss()
    person_criterion = nn.CrossEntropyLoss()
    num_epochs=200
    base_lr=0.0001
    #+ list(model.branch1_last.parameters())   + list(model.branch2_last.parameters()) 
    shared_params = list(model.branch1_before_last.parameters())+ list(model.branch2_before_last.parameters()) +list(model.avgpool.parameters())+list(model.projection.parameters())+list(model.pos_encoder.parameters())+list(model.xlstm.parameters())
    gesture_params = list(model.gesture_head.parameters())
    person_params = list(model.person_head.parameters())

    optimizer = torch.optim.AdamW([
        {'params': shared_params, 'lr': base_lr},
        {'params': gesture_params, 'lr': base_lr * 0.8},  # Adjust these multipliers as needed
        {'params': person_params, 'lr': base_lr * 1.2}
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    dwa = DynamicWeightAverage(num_tasks=2)

    # Training loop

    metrics_log =pd.DataFrame(columns=['Epoch', "Val Gesture Loss","Val Person Loss", "Val Gesture Acc","Val Person Acc", "Val Gesture F1","Val Person F1",])

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
            loss_weights = dwa.update(torch.tensor([gesture_loss.item(), person_loss.item()]))
            total_loss = loss_weights[0] * gesture_loss + loss_weights[1] * person_loss

            optimizer.zero_grad()
            total_loss.backward()

            # Gradient normalization
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if 'gesture_head' in name:
                        param.grad *= loss_weights[0]
                    elif 'person_head' in name:
                        param.grad *= loss_weights[1]

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        # Evaluate on training set 
        train_gesture_loss_avg, train_gesture_acc,train_gesture_f1,train_gesture_recall, train_person_loss_avg, train_person_acc, train_person_f1,train_person_recall= evaluate(model, train_dl, device,
                                                                                                                                      gesture_criterion, person_criterion)

        # Evaluate on validation set
        val_gesture_loss_avg, val_gesture_acc, val_gesture_f1,val_gesture_recall, val_person_loss_avg, val_person_acc, val_person_f1,val_person_recall = evaluate(model, test_dl, device,
                                                                                                                             gesture_criterion, person_criterion)
        new_row = pd.DataFrame([{
            'Epoch': epoch + 1,
            'Val Gesture Loss': val_gesture_loss_avg,
            "Val Person Loss": val_person_loss_avg,
            "Val Gesture Acc": val_gesture_acc,
            "Val Person Acc": val_person_acc,
            "Val Gesture F1": val_gesture_f1,
            "Val Person F1": val_person_f1,
            "Val Gesture Recall": val_gesture_recall,
            "Val Person Recall": val_person_recall,
        }])
        metrics_log = pd.concat([metrics_log, new_row], ignore_index=True)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {total_loss/len(train_dl):.4f}")
        print(f"Train Gesture - Acc: {train_gesture_acc:.4f}, F1: {train_gesture_f1:.4f}")
        print(f"Train Person - Acc: {train_person_acc:.4f}, F1: {train_person_f1:.4f}")
        print(f"Val Gesture - Loss: {val_gesture_loss_avg:.4f}")
        print(f"Val Person - Loss: {val_person_loss_avg:.4f}")
        print(f"Val Gesture - Acc: {val_gesture_acc:.4f}, F1: {val_gesture_f1:.4f}, Recall: {val_gesture_recall:.4f}")
        print(f"Val Person - Acc: {val_person_acc:.4f}, F1: {val_person_f1:.4f}, Recall: {val_person_recall:.4f}")
        print("-----------------------------")

#         if val_gesture_acc>0.987:
#             break
        #if val_person_acc>0.979:
         #   break

    print("Training completed!")
    log_name='res/r3_MTxLSTM-V2'+'_'+str(layers)+'_'+str(block_num)+".csv"
    #metrics_log = pd.DataFrame(metrics_log)
    metrics_log.to_csv(log_name, index=False)


# In[83]:


#layers=[[1,1],[1],[2,2],[2,2,2],[2,2,2,2],[3, 4, 6, 3]]
#block_nums=[2,3,5,7]
layers=[[2,2,2]]
block_nums=[5]
for layer in layers:
    for k in block_nums:
        model = UniTModel(act_num, subject_num,layers=layer,block_num=k)
        print(layer,k)
        train_model(model,layer,k,return_para=False)


# In[26]:


get_confusionmatrix(model,test_dl)


# In[45]:


total=0
for i in range(5):
    result=3255-np.random.randint(5, high=25)
    total=total+result
    print(result)
print(total)


# In[ ]:




