# incorporates elements from https://github.com/marcovzla/discobert/blob/master/config.py

from argparse import Namespace
import os
from __future__ import division
import sys, os
import csv
from os.path import basename
from os.path import exists
import pandas as pd
import numpy as np
import argparse
import subprocess
import commands
from sklearn import preprocessing
# import matplotlib.pyplot as plt
import pdb

import h5py
import pdb
import numpy as np
import csv
import argparse
# from aeent import *
from ecdc import *
import torch
import torch.utils.data
from torch.utils.data import Dataset
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt

### ABSOLUTE FILEPATHS FOR INPUT, SOFTWARE#####
print sys.path
print '\n'
def_wav = '/Users/meghavarshinikrishnaswamy/Downloads/Fisher_corpus/fisher_eng_tr_sp_LDC2004S13_zip_2/fisher_eng_tr_sp_d1/audio/001/fe_03_00101.sph'
def_audio = '/Users/meghavarshinikrishnaswamy/Downloads/Fisher_corpus/fisher_eng_tr_sp_LDC2004S13_zip_2/fisher_eng_tr_sp_d1/audio/'
opensmile = '/Users/meghavarshinikrishnaswamy/github/tomcat-speech/external/opensmile-3.0/bin/SMILExtract'
opensmile_config = '/Users/meghavarshinikrishnaswamy/github/tomcat-speech/external/opensmile-3.0/config/emobase/emobase2010.conf'
sph2pipe = "/Users/meghavarshinikrishnaswamy/github/sph2pipe/sph2pipe"

transcript_dir='~/Downloads/Fisher_corpus/fe_03_p1_tran/data/trans/all_trans'
audio_dir_root = "~/Downloads/Fisher_corpus/fisher_eng_tr_sp_LDC2004S13_zip_2"
metaf = open('Fisher_meta.csv', 'rb')

###### OUTPUT FILES ###########
data_dir = '/Users/meghavarshinikrishnaswamy/Downloads/Fisher_corpus/feats_nonorm'
feat_dir = '/Users/meghavarshinikrishnaswamy/Downloads/Fisher_corpus/raw_feats'
out_dir = '/Users/meghavarshinikrishnaswamy/Downloads/Fisher_corpus/feats_nonorm_nopre'
# DEBUG = False # no saving of files; output in the terminal; first random seed from the list


##### ARGPARSE COMMANDS #######
parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--audio_file', type=str, required=False, default=def_wav,
					help='File path of the input audio file')
parser.add_argument('--openSMILE', type=str, required=False, default=opensmile,
					help='openSMILE path')
parser.add_argument('--openSMILE_config', type=str, required=False, default=opensmile_config,
					help='config file of openSMILE')
parser.add_argument('--output_path', type=str, required=False, default=out_dir,
					help='output folder path')
parser.add_argument('--norm', type=str, required=False, default=True,
					help='do session level normalization or not')
parser.add_argument('--window_size', required=False, type=float, default=None)
parser.add_argument('--shift_size', required=False, type=float, default=1)

args = parser.parse_args()

CONFIG_openSMILE = args.openSMILE_config
openSMILE		 =	args.openSMILE
INPUT_audio      = args.audio_file
OUTPUT_path      = args.output_path

window_size      = args.window_size
shift_size       = args.shift_size
norm             = args.norm


# # do you want to save dataset files?
# save_dataset = False
#
# # do you want to load pre-saved dataset files?
# load_dataset = True

# get this file's path to save a copy
CONFIG_FILE = os.path.abspath(__file__)

# num_feats = 130
# if feature_set.lower() == "is13":
#     num_feats = 130
# elif "combined_features" in feature_set.lower() or "custom" in feature_set.lower():
#     num_feats = 10

##### modify for model when needed ########
# model_params = Namespace(
#     # use gradnorm for loss normalization
#     use_gradnorm=False,
#     # decide whether to use early, intermediate, or late fusion
#     fusion_type="early",  # int, late, early
#     # consistency parameters
#     seed=88,  # 1007
#     # trying text only model or not
#     text_only=False,
#     audio_only=False,
#     # overall model parameters
#     model="Multitask_text_shared",
#     num_epochs=200,
#     batch_size=100,  # 128,  # 32
#     early_stopping_criterion=5,
#     num_gru_layers=2,  # 1,  # 3,  # 1,  # 4, 2,
#     bidirectional=False,
#     use_distilbert=True,
#     # set whether to have a single loss function
#     single_loss=False,
#     # input dimension parameters
#     text_dim=768,  # text vector length # 768 for bert/distilbert, 300 for glove
#     short_emb_dim=30,  # length of trainable embeddings vec
#     audio_dim=num_feats,  # audio vector length
#     # text NN
#     kernel_1_size=3,
#     kernel_2_size=4,
#     kernel_3_size=5,
#     out_channels=20,
#     text_cnn_hidden_dim=100,
#     # text_output_dim=30,   # 100,   # 50, 300,
#     text_gru_hidden_dim=300,  # 30,  # 50,  # 20
#     # acoustic NN
#     avgd_acoustic=False,  # set true to use avgd acoustic feat vectors without RNN
#     add_avging=True,  # set to true if you want to avg acoustic feature vecs upon input
#     acoustic_gru_hidden_dim=100,
#     # speaker embeddings
#     use_speaker=False,
#     num_speakers=13,  # check this number
#     speaker_emb_dim=3,
#     # gender embeddings
#     use_gender=False,
#     gender_emb_dim=4,
#     # outputs
#     output_dim=100,  # output dimensions from last layer of base model
#     output_0_dim=2,  # output vec for first task 2 7 5 7 2
#     output_1_dim=7,  # output vec for second task
#     output_2_dim=5,  # output vec for third task
#     output_3_dim=7,
#     output_4_dim=2,
#     # FC layer parameters
#     num_fc_layers=1,  # 1,  # 2,
#     fc_hidden_dim=100,  # 20,  must match output_dim if final fc layer removed from base model
#     final_hidden_dim=50,  # the out size of dset-specific fc1 and input of fc2
#     dropout=0.2,  # 0.2, 0.3
#     # optimizer parameters
#     lr=1e-4,
#     beta_1=0.9,
#     beta_2=0.999,
#     weight_decay=0.0001,
# )