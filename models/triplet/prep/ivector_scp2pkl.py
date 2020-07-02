import glob
import os
import csv
import pdb
import kaldi_io
import pprint, pickle
from sklearn.metrics.pairwise import euclidean_distances
from scipy import spatial
from sklearn.preprocessing import normalize
import random, math
import numpy as np
import pandas as pd
# -------------------------------------------------------- 
#only used once for reading kaldi ivector
create_pkl = True
if create_pkl:
      ivec_scp = "/home/nasir/data/Fisher/Fisher_ivector/ivectors_train/ivector.scp"
      ivec_norm_dict ={}
      for key,mat in kaldi_io.read_mat_scp(ivec_scp):
          key = '-'.join(key.split('-')[1:])
#          print(key)
          ivec_norm_dict[key] = normalize(mat)
      f = open('ivector_all_normalized.pkl', 'wb')
      pickle.dump(ivec_norm_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
## --------------------------------------------------------------------------

# key format     "SPKID-FILEID_Start-Stop"

