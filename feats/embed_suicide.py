import h5py
import pdb
# from aeent import *
from ecdc import *
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



model_name = 'models/trained_VAE_030318_nonorm.pt'
data_dir = '~/Downloads/Fisher_corpus/feats_nonorm_nopre'
dtset = 'suicide'

SEED=448

#------------------------------------------------------------------
#Uncomment for parsing inputs

parser = argparse.ArgumentParser(description='entrainment testing')
parser.add_argument('--no-cuda', action='store_true', default=False,
	help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
	help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()



torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)



hff = h5py.File('data/test_' + dtset + '_nonorm_sep.h5', 'r')

# pdb.set_trace()


highbond =['3030', '3008', '3031', '3042' , '3048']
lowbond =['3049','3014']

sessList= sorted(glob.glob(data_dir + '*.csv'))
random.seed(SEED)
random.shuffle(sessList)

model = VAE().double()

if 'l1' in model_name:
    p=1
elif 'l2' in model_name:
    p=2
else:
    print("need better model name")
    p=1


model = torch.load(model_name)
model.eval()
if args.cuda:
    model.cuda()

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

    # pdb.set_trace()

    
    for i in range(xx.shape[0]):

        x_data =  xx[i,:228]
        y_data = xx[i,228:-1]


        x_data = Variable(torch.from_numpy(x_data))
        y_data = Variable(torch.from_numpy(y_data))

        if args.cuda:
            x_data = x_data.cuda()
            y_data = y_data.cuda() 



        z_x = model.embedding(x_data)
        z_y = model.embedding(y_data)

        embed = torch.cat((z_x,z_y),1).data.cpu().numpy()
        embed = list(embed)[0]
        
        if xx[i,-1]==1:
            a2b_embed.append(list(embed))
        # else:
        #     b2a_loss.append(loss_pair)

    np.savetxt('embedding/suicide_nonorm_l1_64dim_' + str(int(subj_id)) + '_.txt', np.array(a2b_embed), delimiter=',')

    

#     outdata.append()
# outdata = np.array(outdata)




# np.savetxt('similarity/suicide_nonorm_l1_30dim.txt', outdata, delimiter=',')



