#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

torch.manual_seed(42)

torch.__version__


# In[2]:


OUTPUT_CLASSES = {
    'buildings' : 0,
    'forest' : 1,
    'glacier' : 2,
    'mountain' : 3,
    'sea' : 4,
    'street' : 5
}

train_data, test_data, train_targets, test_targets = [], [], [], []

# Loop through all the data
for class_type in OUTPUT_CLASSES:
    train_files = os.listdir(f"data//seg_train//{class_type}")
    test_files = os.listdir(f"data//seg_test//{class_type}")
    
    for file in train_files:
        image = cv2.imread(f"data//seg_train//{class_type}//{file}")
        class_value = OUTPUT_CLASSES[class_type]

        if (image is not None) and (len(image) == 150) and (len(image[0]) == 150) and (len(image[0][0]) == 3):
            train_data.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(int))
            train_targets.append(class_value)

    for file in test_files:
        image = cv2.imread(f"data//seg_test//{class_type}//{file}")
        class_tensor = torch.tensor(OUTPUT_CLASSES[class_type])

        if (image is not None) and (len(image) == 150) and (len(image[0]) == 150) and (len(image[0][0]) == 3):
            test_data.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(int))
            test_targets.append(class_value)

len(train_data), len(test_data), len(train_targets), len(test_targets)


# In[3]:


test_data[0]


# In[4]:


# Visualize 9 random test samples
import random

plt.figure(figsize=(9, 9))

NUM_ROWS = 3
NUM_COLS = 3
i = 1

GET_CLASS = {
    0 : 'Building',
    1 : 'Forest',
    2 : 'Glacier',
    3 : 'Mountain',
    4 : 'Sea',
    5 : 'Street'
}

for idx, sample in random.sample(list(enumerate(test_data)), k=9):
    plt.subplot(NUM_ROWS, NUM_COLS, i)
    plt.imshow(sample.tolist())
    plt.title(GET_CLASS[test_targets[idx]], fontsize=10)
    plt.axis(False);

    i += 1


# In[5]:


temp_tensor = torch.moveaxis(torch.tensor(test_data[i], dtype=torch.float32), -1, 0) / 255.0
print(f"temp_tensor dtype: {temp_tensor.dtype} | temp_tensor shape: {temp_tensor.shape}")
print(f"temp_tensor:\n{temp_tensor}")


# In[6]:


# Initialize DataLoaders to split data into batches
from torch.utils.data import DataLoader

BATCH_SIZE = 128

train_dataloader = DataLoader(
    [(torch.moveaxis(torch.tensor(train_data[i], dtype=torch.float32), -1, 0) / 255.0, train_targets[i]) for i in range(len(train_data))],
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_dataloader = DataLoader(
    [(torch.moveaxis(torch.tensor(test_data[i], dtype=torch.float32), -1, 0) / 255.0, test_targets[i]) for i in range(len(test_data))],
    batch_size=BATCH_SIZE,
    shuffle=True
)

print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")


# In[7]:


# Build CNN using TinyVGG architecture
class IntelCNN(nn.Module):
    def __init__(self, input_shape: int, output_shape: int, hidden_units: int) -> None:
        super().__init__()

        self.layer_stack_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=hidden_units),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            nn.Dropout(p=0.1)
        )

        self.layer_stack_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=hidden_units),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            nn.Dropout(p=0.15)
        )

        self.layer_stack_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=hidden_units),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            nn.Dropout(p=0.2)
        )

        self.classifier_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*18*18, out_features=hidden_units),
            nn.BatchNorm1d(num_features=hidden_units),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.BatchNorm1d(num_features=hidden_units),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_stack_1(x)
        x = self.layer_stack_2(x)
        x = self.layer_stack_3(x)
        x = self.classifier_layer(x)
        
        return x


# In[8]:


model = IntelCNN(
    input_shape=3,
    output_shape=len(OUTPUT_CLASSES),
    hidden_units=10
)

model


# In[9]:


# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=0.01)


# In[10]:


import requests
from pathlib import Path 

# Download helper functions from Learn PyTorch repo (if not already downloaded)
if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  # Note: you need the "raw" GitHub URL for this to work
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)

from helper_functions import accuracy_fn, print_train_time, plot_loss_curves


# In[11]:


# Create necessary functions to train, test, and evaluate model

def train(model: torch.nn.Module,
          data_loader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim,
          accuracy_fn) -> dict:

    train_loss, train_acc = 0.0, 0.0

    model.train()

    for X, y in data_loader:
        logits = model(X)
        preds = logits.argmax(dim=1)

        loss = loss_fn(logits, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y, y_pred=preds)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    # Divide by length of dataloader to get average loss and accuracy per batch
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    
    return {
        'loss' : train_loss,
        'acc' : train_acc
    }


def test(model: torch.nn.Module,
         data_loader: torch.utils.data.DataLoader,
         loss_fn: torch.nn.Module,
         optimizer: torch.optim,
         accuracy_fn) -> dict:

    test_loss, test_acc = 0.0, 0.0

    model.eval()

    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):
            logits = model(X)
            preds = logits.argmax(dim=1)

            loss = loss_fn(logits, y)
            test_loss += loss
            test_acc += accuracy_fn(y_true=y, y_pred=preds)

        test_loss /= len(data_loader)
        test_acc /= len(data_loader)

    return {
        'loss' : test_loss,
        'acc' : test_acc
    }


# In[ ]:


# Train model
from timeit import default_timer as timer 

START_TIME = timer()
EPOCHS = 10

for epoch in range(EPOCHS):
    train_step = train(
        model=model,
        data_loader=train_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn
    )

    test_step = test(
        model=model,
        data_loader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn
    )

    print(f"Epoch: {epoch} | Train Loss: {train_step['loss']} | Train Acc: {train_step['acc']:.2f}% | Test Loss: {test_step['loss']} | Test Acc: {test_step['acc']:.2f}%")

END_TIME = timer()

TOTAL_TRAIN_TIME = print_train_time(
    start=START_TIME,
    end=END_TIME,
    device='cpu'
)


# In[ ]:




