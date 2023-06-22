import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch import optim
import torch.nn as nn
import torch
from sklearn.metrics import roc_auc_score, plot_roc_curve, average_precision_score, precision_recall_curve


# Please modify the data_path by yourself
data_path = 

train_dataset = CustomDataset(data_path, 'train')
test_dataset = CustomDataset(data_path, 'test')
val_dataset = CustomDataset(data_path, 'val')
print('Train data num : ', len(train_dataset))
print('Val data num : ', len(val_dataset))
print('Test data num : ', len(test_dataset))
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=60, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=60, shuffle=False)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=60, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Network().cuda()
optimizer = optim.Adam(model.parameters(),lr=1e-5,weight_decay=1e-8)
criterion = nn.CrossEntropyLoss()
best_loss = float('inf')
best_acc = 0
for epoch in range(200):
        train_loss = 0
        val_loss = 0
        val_correct = 0
        test_correct = 0

        model.train()
        for image, label in train_loader:
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)
            pred = model(image)
            loss = criterion(pred, label)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        model.eval()
        for image, label in val_loader:
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)
            pred = model(image)
            loss = criterion(pred, label)
            val_loss += loss.item()
            val_correct += torch.sum(torch.argmax(pred, 1) == label).cpu()

        print('epoch : ', epoch, ', training loss : ',train_loss/len(train_dataset),', val acc: ', val_correct / len(val_dataset), ', val loss : ', val_loss / len(val_dataset))
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        prob = []
        true = []
        acc = 0
        for image, label in test_loader:
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)
            prediction = torch.argmax(model(image),1)
            acc+=torch.eq(prediction,label).sum().float().item()
            pred = model.forward(image).cpu().detach().T[1].numpy().tolist()
            for y_true in pred:
                prob.append(y_true)
            label = label.cpu().numpy().tolist()
            for y_label in label:
                true.append(y_label)
        auroc = roc_auc_score(true, prob)
        auprc = average_precision_score(true, prob)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    prob = []
    true = []
    acc = 0
    for image, label in test_loader:
        image = image.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.long)
        prediction = torch.argmax(model(image),1)
        acc+=torch.eq(prediction,label).sum().float().item()
        pred = model.forward(image).cpu().detach().T[1].numpy().tolist()
        for y_true in pred:
            prob.append(y_true)
        label = label.cpu().numpy().tolist()
        for y_label in label:
            true.append(y_label)
    auroc = roc_auc_score(true, prob)
    auprc = average_precision_score(true, prob)
    print('Training time : ', i, ' REsult is : ', acc / len(test_dataset))
