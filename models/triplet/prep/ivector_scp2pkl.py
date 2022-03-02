from entrainment_config import *
# -------------------------------------------------------- 
#only used once for reading kaldi ivector
create_pkl = True
if create_pkl:
      ivec_scp = ivec_scp
      ivec_norm_dict ={}
      for key,mat in kaldi_io.read_mat_scp(ivec_scp):
          key = '-'.join(key.split('-')[1:])
#          print(key)
          ivec_norm_dict[key] = normalize(mat)
      f = open('ivector_all_normalized.pkl', 'wb')
      pickle.dump(ivec_norm_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
## --------------------------------------------------------------------------

# key format     "SPKID-FILEID_Start-Stop"

