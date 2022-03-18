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
data_dir = '/home/nasir/data/suicide/feats_nonorm/'
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

Loss=[]
Fake_loss = []
for sess_file in sessList:
    sess = basename(sess_file).split('.')[0]
    if 'Pre' not in sess and 'pre' not in sess:
        continue
    print(sess)
    subj_id = float(sess[0:4])
    xx = np.array(hff[sess])
    count +=1
    a2b_loss = []
    b2a_loss = []
    fwd_loss = []
    # pdb.set_trace()

    
    test_loss = 0
    fake_test_loss = 0
    for i in range(xx.shape[0]):

        idx_same_spk =list(np.where(xx[:,-1]==xx[i,-1]))[0]
        ll = random.choice(list(set(idx_same_spk) -set([i])))

        x_data =  xx[i,:228]
        y_data = xx[i,228:-1]
        y_fake_data = xx[ll,228:-1]

        x_data = Variable(torch.from_numpy(x_data))
        y_data = Variable(torch.from_numpy(y_data))
        y_fake_data = Variable(torch.from_numpy(y_fake_data))
        if args.cuda:
            x_data = x_data.cuda()
            y_data = y_data.cuda() 
            y_fake_data = y_fake_data.cuda()

        z_x = model.embedding(x_data)
        z_y = model.embedding(y_data)
        z_y_fake = model.embedding(y_fake_data)       



        y_pred = model.forward(x_data)
        loss_pair = lp_distance(z_x, z_y, p).data[0]    
        loss_direct = lp_distance(y_data, y_pred, p).data[0]
        loss_fake = lp_distance(z_x, z_y_fake, p).data[0]    

        fwd_loss.append(loss_direct)
        print() 

        if xx[i,-1]==1:
            a2b_loss.append(loss_pair)
        else:
            b2a_loss.append(loss_pair)
        test_loss += loss_pair
        fake_test_loss += loss_fake 

    Loss.append(test_loss)
    Fake_loss.append(fake_test_loss)


    outdata.append([subj_id, np.mean(np.array(a2b_loss)),  np.mean(np.array(b2a_loss)), np.mean(np.hstack((np.array(a2b_loss), 
        np.array(b2a_loss)))), np.mean(np.array(fwd_loss))])
outdata = np.array(outdata)




np.savetxt('similarity/suicide_nonorm_l1_30dim.txt', outdata, delimiter=',')




Loss=np.array(Loss)
Fake_loss=np.array(Fake_loss)

print(float(np.sum(Loss < Fake_loss))/Loss.shape[0])






    

# results = []



# # for k in range(10):

#     Loss=[]
#     Fake_loss = []
#     # for batch_idx, (x_data, y_data) in enumerate(test_loader):
#     N = int(X_test[-1,-1])

#     for spk_pair in range(1,N+1):

#         idx_same_spk =list(np.where(X_test[:,-1]==spk_pair))[0]


#         test_loss = 0
#         fake_test_loss = 0

#         for idx in idx_same_spk:

#             ll = random.choice(list(set(idx_same_spk) -set([idx])))
#             x_data = X_test[idx,:228]
#             y_data = X_test[idx,228:-1]
#             y_fake_data = X_test[ll,228:-1]

#             x_data = Variable(torch.from_numpy(x_data))
#             y_data = Variable(torch.from_numpy(y_data))
#             y_fake_data = Variable(torch.from_numpy(y_fake_data))

#             if args.cuda:
#                 x_data = x_data.cuda()
#                 y_data = y_data.cuda()
#                 y_fake_data = y_fake_data.cuda()

#             z_x = model.embedding(x_data)
#             z_y = model.embedding(y_data)
#             z_y_fake = model.embedding(y_fake_data)

#             loss_real = lp_distance(z_x, z_y, p).data[0]        
#             loss_fake = lp_distance(z_x, z_y_fake, p).data[0]
#             test_loss += loss_real
#             fake_test_loss += loss_fake 
#         # pdb.set_trace()
#         Loss.append(test_loss)
#         Fake_loss.append(fake_test_loss)
#         # print loss_real, loss_fake

#     Loss=np.array(Loss)
#     Fake_loss=np.array(Fake_loss)

#     total_test_loss = np.sum(Loss)/Loss.shape[0]

#     total_fake_test_loss = np.sum(Fake_loss)/Loss.shape[0]


#     print "Total Real Loss:"+str(total_test_loss) + "Total Fake Loss:" + str(total_fake_test_loss)

#     print float(np.sum(Loss < Fake_loss))/Loss.shape[0]

#     results.append(float(np.sum(Loss < Fake_loss))/Loss.shape[0])


# print np.mean(np.array(results))