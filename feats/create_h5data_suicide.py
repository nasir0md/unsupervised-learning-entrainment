import csv
import h5py
import numpy as np
import pandas as pd
import glob
import random
import pdb
import os
from os.path import basename

SEED=448
frac_train = 0
frac_val = 0

# Create h5 files

data_dir = '/home/nasir/data/suicide/feats_nonorm/'

sessList= sorted(glob.glob(data_dir + '*.csv'))
random.seed(SEED)
random.shuffle(sessList)

num_files_all = len(sessList)
num_files_train = int(np.ceil((frac_train*num_files_all)))
num_files_val = int(np.ceil((frac_val*num_files_all)))
num_files_test = num_files_all - num_files_train - num_files_val

sessTrain = sessList[:num_files_train]
sessVal = sessList[num_files_train:num_files_val+num_files_train]
sessTest = sessList[num_files_val+num_files_train:]
print len(sessTrain) + len(sessVal) + len(sessTest)

# all files separately
hfs = h5py.File('data/test_suicide_nonorm_sep.h5', 'w')  
# Create Test Data file
spk_base = 1
X_test =np.array([])
for sess_file in sessTest:
	# pdb.set_trace()
	df_i = pd.read_csv(sess_file)
	sess = basename(sess_file).split('.')[0]
	xx=np.array(df_i)
	N = xx.shape[0]
	if np.mod(N,2)==0:
		spk_label = np.tile([spk_base, spk_base+1], [1, N/2])
	else:
		spk_label = np.tile([spk_base, spk_base+1], [1, N/2])
		spk_label = np.append(spk_label, spk_base)
	xx = np.hstack((xx, spk_label.T.reshape([N,1])))
	xx = xx.astype('float64')
	hfs.create_dataset(sess, data=xx)
	# X_test=np.vstack([X_test, xx]) if X_test.size else xx
	# spk_base += 1


hfs.close()
# X_test = X_test.astype('float64')
# hf = h5py.File('data/test_suicide_nonorm_nopre.h5', 'w')
# hf.create_dataset('dataset', data=X_test)
# hf.close()



