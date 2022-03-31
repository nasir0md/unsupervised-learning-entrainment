from entrainment_config import *

# -------------------------------------------------------- 
# only used once for reading kaldi ivector, change line 5 to 'True' and uncomment 5-12
# create_pkl = False
# if create_pkl:
# 	ivec_scp = "/Users/meghavarshinikrishnaswamy/Downloads/Fisher_corpus/Fisher_ivector/exp/ivectors_train/ivector.scp"
# 	ivec_norm_dict ={}
# 	for key,mat in kaldi_io.read_mat_scp(ivec_scp):
# 		ivec_norm_dict[key] = normalize(mat)
# 	f = open('ivector_normalized.pkl', 'w')
# 	pickle.dump(ivec_norm_dict, f)
# --------------------------------------------------------------------------

# key format     "SPKID-FILEID_Start-Stop"


def get_neighbor(sess_id, ivec_norm_dict, utt_id):
	turn2ivec = ivec_norm_dict[utt_id]
	turnlist = list(ivec_norm_dict.keys())
	candidates = random.sample(turnlist, 10000)
	candidates = list([x for x in candidates if sess_id not in x])
	lenpool = len(candidates)
	# print(lenpool)
	cosD = np.zeros(lenpool)
	for i in range(lenpool):
		target = ivec_norm_dict[candidates[i]]
		cosD[i] = spatial.distance.cosine(turn2ivec, target)

	chosen = candidates[np.argmin(cosD)]
	return chosen


def get_utt_id(line, metadata):
	start, stop, spk = line.split(':')[0].split(' ')
	if spk=="A":
		spk = metadata[sess_id_num][1]
	else:
		spk = metadata[sess_id_num][3]
	utt_id = spk+ '-' + sess_id +  '_' + str(int(1000*float(start))) 	+ '-' + str(int(1000*float(stop))) 
	return utt_id

# --------------------------------------------------------------------------------------------
transcript_dir= transcript_dir


turnfeatdir = feats_dir

metaf = open(fisher_meta, 'r')


reader = csv.reader(metaf)
metadata ={}
for row in reader:
	metadata[row[0]] = row[1:]

# ivec_dict = pickle.load( open( "ivector.pkl", "rb" ) )
ivec_norm_dict = pickle.load( open( "vectors/ivector_normalized.pkl", "rb" ) )
line_dict = pickle.load( open( "meta/file2line.pkl", "rb" ) )

# -----------------------------------------------------------------------------------------
SEED=448
frac_train = 0.8
frac_val = 0.1
sessList= sorted(glob.glob(turnfeatdir + '*.csv'))
random.seed(SEED)
random.shuffle(sessList)

num_files_all = len(sessList)
num_files_train = int(np.ceil((frac_train*num_files_all)))
num_files_val = int(np.ceil((frac_val*num_files_all)))
num_files_test = num_files_all - num_files_train - num_files_val

sessTrain = sessList[:num_files_train]
sessVal = sessList[num_files_train:num_files_val+num_files_train]
sessTest = sessList[num_files_val+num_files_train:]
print((len(sessTrain) + len(sessVal) + len(sessTest)))
# ------------------------------------------------------------
#  FOR DEBUG
sessList = [turnfeatdir + 'fe_03_03892_IPU_func_feat.csv']
# 
for sessfile in sessList:
	df_i = pd.read_csv(sessfile)
	allfeatsMat = np.array(df_i)
	totTurns = allfeatsMat.shape[0]
	sess_id = sessfile.split('_IPU')[0].split('/')[-1]
	tfile = sess_id + '.txt'
	sess_id_num = sess_id.split('_')[-1]
	transcript =  transcript_dir + tfile
	trans = open(transcript).readlines()
	turnno = 0
	for i, line in enumerate(trans):
		if line=='\n':
			continue
		if line[0] =='#':
			continue
		#  this is a turn

		turn1vec = allfeatsMat[turnno,:]
		turn2vec = allfeatsMat[turnno+1,:]

		utt_id = get_utt_id(trans[i+2], metadata)
		neighbor = get_neighbor(sess_id, ivec_norm_dict, utt_id)
		fakefeatfile = turnfeatdir + '_'.join(neighbor.split('-')[1].split('_')[0:3]) + '_IPU_func_feat.csv'
		df_j = pd.read_csv(fakefeatfile)
		fakefeatsMat = np.array(df_j)
		turn2fake = fakefeatsMat[line_dict[neighbor]-1,:]

		turnno +=1
		pdb.set_trace()
		if turnno==totTurns:
			break
