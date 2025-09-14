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
from sklearn.metrics import accuracy_score, f1_score,recall_score
from tqdm import tqdm  # For progress bar
import torch.nn.functional as F


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
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator

class MultilinearRelationshipNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_tasks=2):
        super(MultilinearRelationshipNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_tasks = num_tasks

        # Task-specific layers
        self.task_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for _ in range(num_tasks)
        ])

        # Shared layer
        self.shared_layer = nn.Linear(hidden_dim, output_dim)

        # Multilinear relationship layers
        self.multilinear_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_tasks * (num_tasks - 1))
        ])

    def forward(self, x):
        task_features = [task_layer(x) for task_layer in self.task_layers]

        # Apply multilinear relationships
        combined_features = []
        for i in range(self.num_tasks):
            combined_feature = task_features[i]
            for j in range(self.num_tasks):
                if i != j:
                    idx = i * (self.num_tasks - 1) + j - (1 if j > i else 0)
                    combined_feature = combined_feature + self.multilinear_layers[idx](task_features[j])
            combined_features.append(combined_feature)

        # Apply shared layer
        outputs = [self.shared_layer(feat) for feat in combined_features]

        return outputs

class MultiTaskMRN(nn.Module):
    def __init__(self, num_gestures, num_persons):
        super(MultiTaskMRN, self).__init__()

        # ResNet18 backbone
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove the final fully connected layer

        # Multilinear Relationship Network
        self.mrn = MultilinearRelationshipNetwork(num_features, num_features, hidden_dim=256, num_tasks=2)

        # Task-specific layers
        self.gesture_fc = nn.Linear(num_features, num_gestures)
        self.person_fc = nn.Linear(num_features, num_persons)

    def forward(self, x):
        # Shared feature extraction
        features = self.backbone(x)

        # Apply MRN
        mrn_features = self.mrn(features)

        # Task-specific predictions
        gesture_output = self.gesture_fc(mrn_features[0])
        person_output = self.person_fc(mrn_features[1])

        return gesture_output, person_output

def get_dataloader(x, y1, y2, batch_size=32):
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), 
                            torch.tensor(y1, dtype=torch.long), 
                            torch.tensor(y2, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


train_dl = get_dataloader(x_train, act_train, subject_train, batch_size=32)
test_dl = get_dataloader(x_test, act_test, subject_test, batch_size=256)

# Create the model
model = MultiTaskMRN(act_num, subject_num)
metrics_log = pd.DataFrame(columns=['Epoch', "Val Gesture Loss","Val Person Loss", "Val Gesture Acc","Val Person Acc", "Val Gesture F1","Val Person F1",])
# Define loss functions and optimizer
gesture_criterion = nn.CrossEntropyLoss()
person_criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

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

# Training loop
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)
getSummary(model,(1,2,224,224))
return

num_epochs=200
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, gesture_labels, person_labels in train_dl:
        inputs = inputs.to(device)
        gesture_labels = gesture_labels.to(device)
        person_labels = person_labels.to(device)

        # Forward pass
        gesture_outputs, person_outputs = model(inputs)

        # Compute losses
        gesture_loss = gesture_criterion(gesture_outputs, gesture_labels)
        person_loss = person_criterion(person_outputs, person_labels)

        # Combine losses
        loss = gesture_loss + person_loss
        total_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate on training set
    #train_g_acc, train_g_f1, train_p_acc, train_p_f1 = evaluate(model, train_dl, device)

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
    #print(f"Train Loss: {total_loss/len(train_dl):.4f}")
    #print(f"Train Gesture - Acc: {train_gesture_acc:.4f}, F1: {train_gesture_f1:.4f}")
    #print(f"Train Person - Acc: {train_person_acc:.4f}, F1: {train_person_f1:.4f}")
    print(f"Val Gesture - Loss: {val_gesture_loss_avg:.4f}")
    print(f"Val Person - Loss: {val_person_loss_avg:.4f}")
    print(f"Val Gesture - Acc: {val_gesture_acc:.4f}, F1: {val_gesture_f1:.4f}, Recall: {val_gesture_recall:.4f}")
    print(f"Val Person - Acc: {val_person_acc:.4f}, F1: {val_person_f1:.4f}, Recall: {val_person_recall:.4f}")
    print("-----------------------------")

log_name='res/MLN_log.csv'
print("Training completed!")
metrics_log.to_csv(log_name, index=False)


# In[9]:


get_ipython().system('nvidia-smi')


# In[ ]:




