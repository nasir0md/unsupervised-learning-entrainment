import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

from trainer import fit
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datasets import TripletMNIST
from datasets import EntTripDataset
import h5py

cuda = torch.cuda.is_available()

# you can move these hard-coded paths to a config file
# Set up data loaders
datadir = '/home/nasir/workspace/acoustic/triplet/fisher/data'
model_path = '/home/nasir/workspace/acoustic/triplet/fisher/trained_models/'

fdset = EntTripDataset(datadir + '/'+  'train_Fisher_triplet_norm.h5')
fdset_val = EntTripDataset(datadir + '/' + 'val_Fisher_triplet_norm.h5')

# also add hyperparameters like this to a config file
batch_size = 256
kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

triplet_train_loader = torch.utils.data.DataLoader(fdset, batch_size=batch_size, shuffle=True, **kwargs)
triplet_test_loader = torch.utils.data.DataLoader(fdset_val, batch_size=batch_size, shuffle=False, **kwargs)


# Set up the network and training parameters
from networks import EmbeddingNet, TripletNet
from losses import TripletLoss
#these mean that embeddingnet and triplet net were used
margin = 1.
embedding_net = EmbeddingNet()
model = TripletNet(embedding_net)
if cuda:
    model.cuda()
loss_fn = TripletLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 50
log_interval = 500

torch.save(model, 	model_path + 'triplet_64d_50ep_fisher.pkl')


f= plt.figure()
train_losses, val_losses = fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)
plt.plot(list(range(len(val_losses))),train_losses, list(range(len(val_losses))), val_losses)
plt.plot(list(range(len(val_losses))),train_losses, list(range(len(val_losses))), val_losses)

f.savefig("losses.pdf", bbox_inches='tight')


