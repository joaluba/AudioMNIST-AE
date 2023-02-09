import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, random_split
import torchdataset_prep as dsprep
import argparse

#  ------------------- MODEL DEFINITION --------------------
class AutoencoderConv(nn.Module):
    def __init__(self):
        super().__init__()      
        self.encoder=nn.Sequential(
            nn.Conv2d(1,16,3,2,1), 
            nn.ReLU(),
            nn.Conv2d(16,32,3,2,1),
            nn.ReLU(),
            nn.Conv2d(32,64,(129,11),1,0)
        )

        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(64,32,(129,11),1,0), 
            nn.ReLU(),
            nn.ConvTranspose2d(32,16,3,2,1,output_padding=(0,1)),
            nn.ReLU(),
            nn.ConvTranspose2d(16,1,3,2,1,output_padding=(0,1)),
            nn.Sigmoid() # because the values of the (normalized) input are either 0 or 1 
        )

    def forward(self,x):
        x_encoded=self.encoder(x)
        x_decoded=self.decoder(x_encoded)
        return x_decoded 

class AutoencoderLin(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Linear(513*44,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,12)
        )

        self.decoder=nn.Sequential(
            nn.Linear(12,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,513*44),
            nn.Sigmoid() # because the values of the (normalized) input are either 0 or 1 
        )

    def forward(self,x):
        x_encoded=self.encoder(x)
        x_decoded=self.decoder(x_encoded)
        return x_decoded


class SpectralConvergenceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_mag, y_mag):
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro") 

#  ------------------- TRAINING --------------------

def training(model, dataloader, num_epochs, device, store_outputs):
    # Loss Function, Optimizer and Scheduler
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    outputs=[]
    # Repeat for each epoch
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        # Repeat for each batch in the training set
        for i, data in tqdm(enumerate(dataloader)):
            # Get the input features and target labels, and put them on the GPU
            x_orig, labels = data[0].to(device), data[1].to(device)
            # Normalize the inputs
            inputs_m, inputs_s = x_orig.mean(), x_orig.std()
            x_orig = (x_orig - inputs_m) / inputs_s
            # empty gradient
            optimizer.zero_grad()
            # spectrogram reconstruction (forward pass)
            x_recons = model(x_orig.to(device))
            # reconstruction loss 
            loss = criterion(x_orig, x_recons)
            # compute gradients (differentiate loss function with respect to the weights)
            loss.backward()
            # update weights (add gradient to the previous weights)
            optimizer.step()
            # compute loss for the current batch
            running_loss += loss.item()

        # Print stats at the end of the epoch
        num_batches = len(dataloader)
        avg_loss = running_loss / num_batches
        print(f'Epoch: {epoch}, Loss: {avg_loss:.4f}')
        if store_outputs:
            outputs.append((epoch,x_orig,x_recons,labels))
        
    print('Finished Training')
    if store_outputs:
        return outputs

#  ------------------- TESTING --------------------

""" def test(model, dataloader,device):
    # Use data to
    model.eval()
    correct_prediction = 0
    total_prediction = 0

    for data in tqdm(dataloader):
        # get batch
        x_orig, labels = data[0].to(device), data[1].to(device)
        # inference (forward pass)
        x_recons = model(x_orig.to(device))
        # compute accuracy


    return acc """

if __name__ == "__main__":

    model=Autoencoder()
    citerion=nn.MSELoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)

    argParser = argparse.ArgumentParser()
    argParser.add_argument('snr',type=float,help="signal to (white) noise ratio of speech data")
    argParser.add_argument('nspeakers',type=int,help="number of speakers to take data from")
    args=argParser.parse_args()

    # choose computing device
    if torch.cuda.is_available():
        print("Using Cuda")
        device = torch.device("cuda")
    else:
        print("Using CPU")
        device = torch.device('cpu')

    # instantiate data set
    AUDIO_PATH = "/Users/joanna.luberadzka/Documents/AudioMNIST-data/data"
    SAMPLE_RATE = 22050
    SIG_LEN = 1
    SNR=args.snr
    N_SPK=args.nspeakers

    
    # Create dataset object
    dataset = dsprep.AudioMnist(AUDIO_PATH, SAMPLE_RATE, SIG_LEN,N_SPK,SNR)
    print("Number of data points:" + str(len(dataset.Name)))
    print("Dimensions of input data:" + str(dataset[20][0].shape))

    # split dataset into training set, test set and validation set
    N_train = round(len(dataset) * 0.8)
    N_rest = len(dataset) - N_train
    trainset, restset = random_split(dataset, [N_train, N_rest])
    N_test = round(len(restset) * 0.5)
    N_val = len(restset) - N_test
    testset, valset = random_split(restset, [N_test, N_val])

    # create dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)

    # instantiate a model
    model=Autoencoder().to(device)

    # training
    training(model, trainloader, 10, device)

    # test
    #test(model,testloader,device)

    torch.save(model.state_dict(), "/Users/joanna.luberadzka/Documents/VAE/models/curr_model_snr"+str(SNR)+".pth")