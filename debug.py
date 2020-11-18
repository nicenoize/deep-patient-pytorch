import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import time

class PatientVectorDataset(Dataset):
    """Patient vector dataset."""

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with vectors.
        """
        self.patientvecs = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.patientvecs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patient = self.patientvecs.iloc[idx, 1:]
        patient = np.array([patient])
        #patient = landmarks.astype('float').reshape(-1, 2)

        return patient

patientvecs_dataset = PatientVectorDataset(csv_file='/Users/nicenoize/Documents/DATEXIS/DeepPatient/multi_hot_encoded.csv')
#print(patientvecs_dataset[0,:])
dataloader = DataLoader(patientvecs_dataset, batch_size=4, shuffle=True, num_workers=4)
dataset = pd.read_csv('/Users/nicenoize/Documents/DATEXIS/DeepPatient/multi_hot_encoded.csv')
# X = Features
# y = diagnoses
X = dataset.drop(dataset.columns[0], axis=1)
y = dataset[dataset.columns[0]]
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

class CDAutoEncoder(nn.Module):
    """
    Convolutional denoising autoencoder layer for stacked autoencoders.
    This module is automatically trained when in model.training is True.

    Args:
        input_size: The number of features in the input
        output_size: The number of features to output
        stride: Stride of the convolutional layers.
    """
    def __init__(self, input_size, output_size, stride):
        super(CDAutoEncoder, self).__init__()

        self.forward_pass = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
        )
        self.backward_pass = nn.Sequential(
            nn.Linear(output_size, input_size), 
            nn.ReLU(),
        )

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

    def forward(self, x):
        # Train each autoencoder individually
        x = x.detach()
        #print(x.shape)
        # Add noise, but use the original lossless input as the target.
        x_noisy = x * (Variable(x.data.new(x.size()).normal_(0, 0.1)) > -.1).type_as(x)
        y = self.forward_pass(x_noisy)

        if self.training:
            x_reconstruct = self.backward_pass(y)
            loss = self.criterion(x_reconstruct, Variable(x.data, requires_grad=False))
            #loss = nn.CrossEntropyLoss()
            self.optimizer.zero_grad()
            #output = loss(x_reconstruct, x)
            loss.backward()
            self.optimizer.step()
            
        return y.detach()

    def reconstruct(self, x):
        return self.backward_pass(x)


class StackedAutoEncoder(nn.Module):
    """
    A stacked autoencoder made from the convolutional denoising autoencoders above.
    Each autoencoder is trained independently and at the same time.
    """

    def __init__(self):
        super(StackedAutoEncoder, self).__init__()

        #self.ae1 = CDAutoEncoder(500, 500, 500)
        #self.ae2 = CDAutoEncoder(500, 500, 500)
        #self.ae3 = CDAutoEncoder(500, 500, 500)

        #input_size, output_size, kernel_size=2, stride=stride, padding=0)
        self.ae1 = CDAutoEncoder(80, 80, 80)
        self.ae2 = CDAutoEncoder(80, 80, 80)
        self.ae3 = CDAutoEncoder(80, 80, 80)

        
    def forward(self, x):
        a1 = self.ae1(x)
        a2 = self.ae2(a1)
        a3 = self.ae3(a2)

        if self.training:
            return a3
            #return a1

        else:
            return a3, self.reconstruct(a3)
            #return a1, self.reconstruct(a1)


    def reconstruct(self, x):
            a2_reconstruct = self.ae3.reconstruct(x)
            a1_reconstruct = self.ae2.reconstruct(a2_reconstruct)
            x_reconstruct = self.ae1.reconstruct(a1_reconstruct)
            return x_reconstruct

model = StackedAutoEncoder() #.cuda()

num_epochs = 100
batch_size = 4

for epoch in range(num_epochs):
    print("Epoch:")
    print(epoch)
    print("\n")
    if epoch % 10 == 0:
        # Test the quality of our features with a randomly initialzed linear classifier.
        classifier = nn.Linear(80, 4) #.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    model.train()
    total_time = time.time()
    correct = 0

    for i, batch in enumerate(dataloader):
        #print(dataloader.dataset)
        #print(batch.shape)
        #patient, target = batch
        patient = np.random.rand(80)
        target = np.random.rand(80)
        #print(batch.reshape(4, 80))
        #just to test
        #prediction = batch[0].flatten()
        #batch = torch.arange(4.)
        #print(batch.shape)
        #target = batch.flatten()
        #print(target.shape)
        #patient, target, features = batch
        #batch = Variable(batch)#.cuda()
        #print(patient)
        #print(batch)
        features = model(torch.FloatTensor(np.random.rand(80,80)))#.detach()
        prediction = classifier(features.view(features.size(0), -1))
        loss = criterion(prediction, torch.tensor(target).type(torch.LongTensor))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = prediction.data.max(1, keepdim=True)[1]
        #correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    model.eval()
    #batch, _ = data
    batch = Variable(torch.tensor(patient).type(torch.LongTensor)) #.cuda()
    features, x_reconstructed = model(torch.tensor(batch).type(torch.FloatTensor))
    reconstruction_loss = torch.mean((x_reconstructed.data - batch.data)**2)


    print("Epoch {} complete\tTime: {:.4f}s\t\tLoss: {:.4f}".format(epoch, total_time, reconstruction_loss))
    print("Feature Statistics\tMean: {:.4f}\t\tMax: {:.4f}\t\tSparsity: {:.4f}%".format(
        torch.mean(features.data), torch.max(features.data), np.true_divide(torch.sum(features.data == 0.0)*100,features.data.numel()))
    )
    print("Linear classifier performance: {}/{} = {:.2f}%".format(correct, len(dataloader)*batch_size, 100*float(correct) / (len(dataloader)*batch_size)))
    print("="*80)

torch.save(model.state_dict(), './CDAE.pth')