import h5py
import pdb
from networks import *
import numpy as np 
import csv
import random
import argparse
import glob
import os
from os.path import basename
import torch
import torch.utils.data
from torch.utils.data import Dataset
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt

work_dir = '/home/nasir/workspace/acoustic/NED_ecdc/'

model_path = '/home/nasir/workspace/acoustic/triplet/fisher/trained_models/'

model_name = model_path + 'triplet_64d_50ep_fisher.pkl'

SEED = 448
# --------------------------------------------------------------------------
# Load model

cuda = torch.cuda.is_available()
if cuda:
    torch.cuda.manual_seed(SEED)


embedding_net = EmbeddingNet()
model = TripletNet(embedding_net)

model.load_state_dict(torch.load(model_name))
# model = torch.load(model_name)

model.eval()

if cuda:
    model.cuda()

def lp_distance(x1, x2, p):
    dist = torch.dist(x1, x2,p)
    return dist
p=2
# --------------------------------------------------------------------------
# Load data
data_dir = '/home/nasir/data/suicide/feats_nonorm/'
dtset = 'suicide'
hff = h5py.File(work_dir+ 'data/test_' + dtset + '_nonorm_sep.h5', 'r')

highbond =['3030', '3008', '3031', '3042' , '3048']
lowbond =['3049','3014']
sessList= sorted(glob.glob(data_dir + '*.csv'))
random.seed(SEED)
random.shuffle(sessList)
count=0
outdata =[]
embeddings=[]

for sess_file in sessList:
    sess = basename(sess_file).split('.')[0]

    if 'Pre' not in sess and 'pre' not in sess:
        continue
    print(sess)
    subj_id = float(sess[0:4])
    if str(int(subj_id)) not in highbond:
        continue
    xx = np.array(hff[sess])
    count +=1
    a2b_embed = []
    
    for i in range(xx.shape[0]):

        x_data =  xx[i,:228]
        y_data = xx[i,228:-1]


        x_data = Variable(torch.from_numpy(x_data))
        y_data = Variable(torch.from_numpy(y_data))

        if cuda:
            x_data = x_data.cuda()
            y_data = y_data.cuda() 



        z_x = model.get_embedding(x_data)
        z_y = model.get_embedding(y_data)

        embed = torch.cat((z_x,z_y),1).data.cpu().numpy()
        embed = list(embed)[0]
        
        if xx[i,-1]==1:
            a2b_embed.append(list(embed))
        # else:
        #     b2a_loss.append(loss_pair)

    np.savetxt(work_dir + '../triplet/suicide/embedding/suicide_nonorm_l1_64dim_' + str(int(subj_id)) + '_.txt', np.array(a2b_embed), delimiter=',')



