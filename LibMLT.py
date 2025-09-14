#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import torchmetrics
from sklearn.metrics import accuracy_score, f1_score, recall_score
from tqdm import tqdm  # For progress bar
import torch.nn.functional as F
import math
from LibMTL.weighting import EW  # or any other weighting strategy
from LibMTL.architecture import MTAN, MMoE  # choose either one
from LibMTL.config import LibMTL_args
from LibMTL import Trainer
import numpy as np
import pandas as pd
import os
import pickle
from LibMTL.model import resnet18
from LibMTL.utils import set_random_seed, set_device
from LibMTL.config import LibMTL_args, prepare_args
from LibMTL.loss import CELoss
from LibMTL.metrics import AccMetric
import pickle
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

print(np.unique(subjects,return_counts=True))
print(np.unique(activities,return_counts=True))
subject_num=len(np.unique(subjects))
act_num=len(np.unique(activities))
print(subject_num, act_num)


# In[3]:


#x_train, x_test, subject_train, subject_test, act_train, act_test = train_test_split(spectros, subjects, activities,test_size=0.2)


# In[4]:


# pf = "pickle.dat"
# with open(pf, "wb") as f:
#     pickle.dump((x_train, x_test, subject_train, subject_test, act_train, act_test), f)

pf = "pickle.dat"
with open(pf, "rb") as f:
        x_train, x_test, subject_train, subject_test, act_train, act_test=pickle.load(f)
print(x_train.shape,x_test.shape)


# In[5]:


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


# In[6]:


def get_dataset(x, y1):
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), 
                            torch.tensor(y1, dtype=torch.long))
    return dataset


def compute_metrics(y_pred, y_true):
    # Convert predictions to numpy arrays and get class predictions
    y_pred_np = y_pred.cpu().detach().numpy()
    y_true_np = y_true.cpu().detach().numpy()
    y_pred_classes = np.argmax(y_pred_np, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_true_np, y_pred_classes)
    f1 = f1_score(y_true_np, y_pred_classes, average='macro')

    return {
        'accuracy': accuracy,
        'f1_score': f1
    }


# In[15]:


class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetBackbone, self).__init__()
        hidden_dim = 512
        self.resnet_network = resnet18()
        self.resnet_network.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.hidden_layer_list = [nn.Linear(512, hidden_dim),
                                  nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.5)]
        self.hidden_layer = nn.Sequential(*self.hidden_layer_list)

        # initialization
        self.hidden_layer[0].weight.data.normal_(0, 0.005)
        self.hidden_layer[0].bias.data.fill_(0.1)

    def forward(self, inputs):
        out = self.resnet_network(inputs)
        if isinstance(out, list):
            #print(out)
            out=out[0]
        out = torch.flatten(self.avgpool(out), 1)
        out = self.hidden_layer(out)
        return out


# In[16]:


from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data.astype(np.float32)  # Ensure data is float32
        self.labels = labels.astype(np.int64)  # Ensure labels are int64

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert to PyTorch tensors
        data_tensor = torch.tensor(self.data[idx], dtype=torch.float32)  # Ensure data is FloatTensor
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.int64)  # Ensure labels are LongTensor
        return data_tensor, label_tensor

from torch.utils.data import DataLoader

def custom_dataloader(batchsize, x_train, act_train,subject_train, x_test, act_test,subject_test):
    # Define tasks based on your dataset
    tasks = ['gesture', 'person']
    data_loader = {}
    iter_data_loader = {}

    for k, d in enumerate(tasks):
        #print(k,d)
        data_loader[d] = {}
        iter_data_loader[d] = {}

        for mode in ['train', 'val']:
            shuffle = True if mode == 'train' else False
            drop_last = True if mode == 'train' else False
            txt_dataset=None
            # Create dataset based on the mode
            if mode == 'train':
                if d == 'gesture':
                    txt_dataset = CustomDataset(x_train, act_train)  # Assuming y_train contains gesture labels
                elif d == 'person':
                    txt_dataset = CustomDataset(x_train, subject_train)  # Assuming subject_train contains person labels
            else:  # mode == 'val'
                if d == 'gesture':
                    txt_dataset = CustomDataset(x_test, act_test)  # Assuming y_test contains gesture labels
                elif d == 'person':
                    txt_dataset = CustomDataset(x_test, subject_test)  # Assuming subject_test contains person labels

            # Create DataLoader
            data_loader[d][mode] = DataLoader(
                txt_dataset,
                num_workers=2,
                pin_memory=True,
                batch_size=batchsize,
                shuffle=shuffle,
                drop_last=drop_last
            )
            iter_data_loader[d][mode] = iter(data_loader[d][mode])

    return data_loader, iter_data_loader


# In[21]:


def train_mtl(arch,return_paras=True):

    # Initialize arguments
    args = LibMTL_args
    args.weighting = 'EW'
    args.arch = arch #'MTAN'
    args.gpu_id = 0
    args.epochs = 200
    args.batch_size = 32
    args.lr = 0.0001
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")    

    # Prepare data loaders
    # train_dataloaders = {
    #     'gesture': DataLoader(CustomDataset(x_train, act_train), batch_size=args.batch_size, shuffle=True),
    #     'person': DataLoader(CustomDataset(x_train, subject_train), batch_size=args.batch_size, shuffle=True)
    # }

    # val_dataloaders = {
    #     'gesture': DataLoader(get_dataset(x_test, act_test), batch_size=args.batch_size),
    #     'person': DataLoader(get_dataset(x_test, subject_test), batch_size=args.batch_size)
    # }
    task_name = ['gesture', 'person']
    data_loader, _ = custom_dataloader(32,x_train, act_train,subject_train, x_test, act_test,subject_test)
    train_dataloaders = {task: data_loader[task]['train'] for task in task_name}
    val_dataloaders = {task: data_loader[task]['val'] for task in task_name}
    test_dataloaders = {task: data_loader[task]['val'] for task in task_name}

    # Define loss functions
    criterions = {
        'gesture': CELoss(),
        'person': CELoss()
    }

    metrics = {
        'gesture': AccMetric(),
        'person': AccMetric()
    }

    # Initialize backbone and architecture
    backbone = ResNetBackbone()

    decoders = {
        'gesture': nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, act_num)
        ),
        'person': nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, subject_num)
        )
    }

    architecture = args.arch  # Pass the architecture type as a string


    optim_param = {
        'optim': 'adam',  # Specify the optimizer type (e.g., 'adam', 'sgd', etc.)
        'lr': args.lr,
        'weight_decay': 0.0001
    }

    scheduler_param = {
        'scheduler': 'step',  # Specify the type of scheduler (e.g., 'step', 'cos', etc.)
        'step_size': 50,
        'gamma': 0.1
    }
    kwargs = {'weight_args': {}, 'arch_args': {}}    

    # Prepare task_dict with weights and loss functions
    task_dict = {
        'gesture': {
            'dataloader': train_dataloaders['gesture'],
            'weight': 1.0,  # Set the weight for the gesture task
            'loss_fn': criterions['gesture'],  # Add the loss function for the gesture task
            'metrics': ['Acc'],
            'metrics_fn': metrics['gesture']
        },
        'person': {
            'dataloader': train_dataloaders['person'],
            'weight': 1.0,  # Set the weight for the person task
            'loss_fn': criterions['person'],  # Add the loss function for the person task
            'metrics': ['Acc'],
            'metrics_fn': metrics['person']
        }
    }


    # Initialize trainer
    trainer = Trainer(
        task_dict=task_dict,
        encoder_class=ResNetBackbone,  # Pass the encoder class
        decoders=decoders,
        rep_grad=False,
        multi_input=True,
        optim_param=optim_param,
        scheduler_param=scheduler_param,
        weighting='EW',
        architecture=architecture,
        criterions=criterions,
        metrics=metrics,
        train_dataloaders=train_dataloaders,
        val_dataloaders=val_dataloaders,
        args=args,**kwargs)


    # Train model
    trainer.train(train_dataloaders,val_dataloaders,1)
    if return_paras:
        getSummary(trainer.model,(1,2,224,224))
        return



# In[22]:


#archs=['MTAN','MMoE','LTB']
archs=['MMoE','MTAN']#
for arch in archs:
    train_mtl(arch)


# In[ ]:





# In[ ]:




