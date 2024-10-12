import torch
from torchvision import datasets, transforms
import numpy as np
import config
from model import Model
from torchsummary import summary
import torch.nn as nn
from logger import Logger
from torch.utils.tensorboard import SummaryWriter

def main(batch_size=config.batch_size, epochs=config.epochs, learning_rate=config.learning_rate, shuffle=config.shuffle, drop_last=config.drop_last, log_interval=config.log_interval, device=config.device, run_name=config.run_name):
    
    # Download and load the training data
    train = datasets.MNIST('data/', download=True, train=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    # Download and load the test data
    valid = datasets.MNIST('data/', download=True, train=False, transform=transforms.ToTensor())
    validloader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    
    model = Model()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    writer = SummaryWriter(log_dir=run_name)
    
    trainLossLogger = Logger(writer, "train/Loss")
    validLossLogger = Logger(writer, "valid/Loss")
    
    trainAccuracyLogger = Logger(writer, "train/Accuracy")
    validAccuracyLogger = Logger(writer, "valid/Accuracy")  
    
    summary(model, (1, 28, 28))
    
    model = model.to(device)
    
    for epoch in range(epochs):
        for x, y in trainloader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if epoch % log_interval == 0 or epoch == epochs - 1:
                trainLossLogger.add(loss.item(), batch_size)
                trainAccuracyLogger.add(torch.sum(torch.argmax(y_pred, dim=1) == y).item(), batch_size)
        
        with torch.no_grad():
            for x, y in validloader:
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
            
            if epoch % log_interval == 0 or epoch == epochs - 1:
                validLossLogger.add(loss.item(), batch_size)
                validAccuracyLogger.add(torch.sum(torch.argmax(y_pred, dim=1) == y).item(), batch_size)
            
        if epoch % log_interval == 0 or epoch == epochs - 1:
            print(f"Epoch: {epoch} Train Loss: {trainLossLogger.get(epoch)} Train Accuracy: {trainAccuracyLogger.get(epoch)} Valid Loss: {validLossLogger.get(epoch)} Valid Accuracy: {validAccuracyLogger.get(epoch)}")

if __name__ == "__main__":
    main()
