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
from torchvision.models import resnet18
from tqdm import tqdm  # For progress bar
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


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchmetrics
import pandas as pd
from torchvision.models import resnet50

# Check if GPU is available and set device to GPU 0
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Helper function to get data loader
def get_dataloader(x, y1, y2, batch_size=32):
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), 
                            torch.tensor(y1, dtype=torch.long), 
                            torch.tensor(y2, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Metrics class to calculate accuracy and F1-score
class Metrics:
    def __init__(self, num_classes):
        self.accuracy = torchmetrics.Accuracy(num_classes=num_classes, task='multiclass').to(device)
        self.f1_score = torchmetrics.F1Score(num_classes=num_classes, average="weighted", task='multiclass').to(device)

    def update(self, preds, targets):
        self.accuracy.update(preds, targets)
        self.f1_score.update(preds, targets)

    def compute(self):
        acc = self.accuracy.compute().item()
        f1 = self.f1_score.compute().item()
        self.accuracy.reset()
        self.f1_score.reset()
        return acc, f1

# Cross-Stitch Unit
class CrossStitchUnit(nn.Module):
    def __init__(self, num_tasks=2, channels=2048):
        super(CrossStitchUnit, self).__init__()
        self.cross_stitch_matrix = nn.Parameter(torch.eye(num_tasks * channels))

    def forward(self, inputs):
        input_tensor = torch.cat(inputs, dim=1)  # Concatenate task activations along channel axis
        stitched_outputs = torch.matmul(input_tensor, self.cross_stitch_matrix)  # Cross-stitch operation
        stitched_outputs = torch.chunk(stitched_outputs, len(inputs), dim=1)  # Split back into task-specific tensors
        return stitched_outputs

# Multi-Task Learning Model with Cross-Stitch Units
class MultiTaskResNetCrossStitch(nn.Module):
    def __init__(self, num_classes_activity, num_classes_subject):
        super(MultiTaskResNetCrossStitch, self).__init__()

        # Shared Backbone: ResNet50 up to the layer before the final fully connected
        self.shared_resnet = resnet18(pretrained=True)
        # Modify the first conv layer to accept 2 input channels
        self.shared_resnet.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.shared_resnet = nn.Sequential(*list(self.shared_resnet.children())[:-2])  # Remove final layers

        # Cross-Stitch Layers: Adding cross-stitch unit after shared feature extractor
        self.cross_stitch_unit1 = CrossStitchUnit(num_tasks=2, channels=2048)

        # Task-specific heads
        self.task1_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes_activity)  # Gesture recognition (activity classification)
        )

        self.task2_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes_subject)  # Person identification (subject classification)
        )

    def forward(self, x):
        # Shared feature extractor
        shared_features = self.shared_resnet(x)

        # Cross-stitch layer: get task-specific feature maps
        #task1_features, task2_features = self.cross_stitch_unit1([shared_features, shared_features])

        # Task-specific heads
        out_task1 = self.task1_head(shared_features)  # Gesture recognition
        out_task2 = self.task2_head(shared_features)  # Person identification

        return out_task1, out_task2

# Create model and move it to the GPU
num_classes_activity = act_num  # Number of gesture classes
num_classes_subject = subject_num  # Number of person identification classes
model = MultiTaskResNetCrossStitch(num_classes_activity=num_classes_activity, num_classes_subject=num_classes_subject).to(device)
getSummary(model,(1,2,224,224))
return

# Loss functions for both tasks
criterion_task1 = nn.CrossEntropyLoss()  # For gesture recognition
criterion_task2 = nn.CrossEntropyLoss()  # For person identification

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop with metrics and logging to CSV
def train(model, criterion_task1, criterion_task2, optimizer, train_loader, test_loader, num_epochs=10, save_path='metrics.csv'):
    # Initialize metrics for both tasks
    metrics_task1_train = Metrics(num_classes=num_classes_activity)
    metrics_task2_train = Metrics(num_classes=num_classes_subject)
    metrics_task1_test = Metrics(num_classes=num_classes_activity)
    metrics_task2_test = Metrics(num_classes=num_classes_subject)

    # Prepare CSV log
    metrics_log = []

    for epoch in range(num_epochs):
        model.train()
        total_loss_task1 = 0.0
        total_loss_task2 = 0.0

        # Training
        for inputs, labels_task1, labels_task2 in train_loader:
            inputs, labels_task1, labels_task2 = inputs.to(device), labels_task1.to(device), labels_task2.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs_task1, outputs_task2 = model(inputs)

            # Compute loss for both tasks
            loss_task1 = criterion_task1(outputs_task1, labels_task1)  # Gesture recognition loss
            loss_task2 = criterion_task2(outputs_task2, labels_task2)  # Person identification loss

            # Total loss (sum of both)
            total_loss = loss_task1 + loss_task2

            # Backpropagation
            total_loss.backward()
            optimizer.step()

            # Update metrics
            preds_task1 = torch.argmax(outputs_task1, dim=1)
            preds_task2 = torch.argmax(outputs_task2, dim=1)
            metrics_task1_train.update(preds_task1, labels_task1)
            metrics_task2_train.update(preds_task2, labels_task2)

            total_loss_task1 += loss_task1.item()
            total_loss_task2 += loss_task2.item()

        # Compute train metrics
        train_acc_task1, train_f1_task1 = metrics_task1_train.compute()
        train_acc_task2, train_f1_task2 = metrics_task2_train.compute()

        # Testing
        model.eval()
        with torch.no_grad():
            for inputs, labels_task1, labels_task2 in test_loader:
                inputs, labels_task1, labels_task2 = inputs.to(device), labels_task1.to(device), labels_task2.to(device)

                outputs_task1, outputs_task2 = model(inputs)

                preds_task1 = torch.argmax(outputs_task1, dim=1)
                preds_task2 = torch.argmax(outputs_task2, dim=1)

                # Update test metrics
                metrics_task1_test.update(preds_task1, labels_task1)
                metrics_task2_test.update(preds_task2, labels_task2)

        # Compute test metrics
        test_acc_task1, test_f1_task1 = metrics_task1_test.compute()
        test_acc_task2, test_f1_task2 = metrics_task2_test.compute()

        # Log metrics
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss_task1": total_loss_task1,
            "train_loss_task2": total_loss_task2,
            "train_acc_task1": train_acc_task1,
            "train_f1_task1": train_f1_task1,
            "train_acc_task2": train_acc_task2,
            "train_f1_task2": train_f1_task2,
            "test_acc_task1": test_acc_task1,
            "test_f1_task1": test_f1_task1,
            "test_acc_task2": test_acc_task2,
            "test_f1_task2": test_f1_task2
        }
        metrics_log.append(epoch_log)

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss Task 1: {total_loss_task1:.4f}, Train Loss Task 2: {total_loss_task2:.4f} | "
              f"Train Acc Task 1: {train_acc_task1:.4f}, Train F1 Task 1: {train_f1_task1:.4f} | "
              f"Train Acc Task 2: {train_acc_task2:.4f}, Train F1 Task 2: {train_f1_task2:.4f} | "
              f"Test Acc Task 1: {test_acc_task1:.4f}, Test F1 Task 1: {test_f1_task1:.4f} | "
              f"Test Acc Task 2: {test_acc_task2:.4f}, Test F1 Task 2: {test_f1_task2:.4f}")

    # Save metrics to CSV
    df_metrics = pd.DataFrame(metrics_log)
    df_metrics.to_csv(save_path, index=False)

# Example DataLoader
train_dl = get_dataloader(x_train, act_train, subject_train, batch_size=32)
test_dl = get_dataloader(x_test, act_test, subject_test, batch_size=256)

# Train the model and save metrics to 'metrics.csv'
train(model, criterion_task1, criterion_task2, optimizer, train_dl, test_dl, num_epochs=200, save_path='res/shared_bottom.csv')


# In[ ]:




