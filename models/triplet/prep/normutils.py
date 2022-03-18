import sys
import numpy as np
import random, math
from sklearn.metrics.pairwise import euclidean_distances
from scipy import spatial

def normalizefeats(feat_data, norm):
	
	##-----------------------------------------------------------------------
	## feature selection and normalization 
	##-----------------------------------------------------------------------
	# remove the mean for mfcc
	# normalize for pitch = log(f_0/u_0)
	# normalize for loudness 
	if norm:
		# do normalization
		# print("Do session level feature normalization... ", file=sys.stderr)
		# f0 normalization
		f0                            = np.copy(feat_data[:, 70])
		# replace 0 in f0 with nan
		f0[f0==0.]                     = np.nan
		f0_mean                       = np.nanmean(f0)
		f0[~np.isnan(f0)]             = np.log2(f0[~np.isnan(f0)]/f0_mean)
		f0                            = np.reshape(f0,(-1,1))
		
		# f0_de normalization
		f0_de                         = np.copy(feat_data[:, 74])
		f0_de[f0_de==0.]               = np.nan
		f0_de_mean                    = np.nanmean(f0_de)
		f0_de[~np.isnan(f0_de)]       = np.log2(f0_de[~np.isnan(f0_de)]/f0_de_mean)
		f0_de                         = np.reshape(f0_de,(-1,1))
		# intensity normalization
		intensity                     = np.copy(feat_data[:,2])
		int_mean                      = np.mean(intensity)
		intensity                     = intensity / int_mean
		intensity                     = np.reshape(intensity, (-1,1))
		
		# intensity_de normalization
		intensity_de                  = np.copy(feat_data[:,36])
		int_de_mean                   = np.mean(intensity_de)
		intensity_de                  = intensity_de / int_de_mean
		intensity_de                  = np.reshape(intensity_de, (-1,1))
		
		# all other features normalization, just 
		# feat_idx                      = range(3,34) + range(37, 68)   with spectral de
		feat_idx                      = range(3,34)
		mfcc_etc                      = np.copy(feat_data[:,feat_idx])
		
		mfcc_etc_mean                 = np.mean(mfcc_etc, axis=0)
		mfcc_etc_mean.reshape(-1,1)
		mfcc_etc_norm                 =  mfcc_etc - mfcc_etc_mean
		
		# jitter and shimmer normalization
		idx_jitter_shimmer            = [71,72,73]
		jitter_shimmer                = np.copy(feat_data[:,idx_jitter_shimmer])
		jitter_shimmer[jitter_shimmer==0.] = np.nan
		jitter_shimmer_mean           = np.nanmean(jitter_shimmer, axis=0)
		jitter_shimmer_mean.reshape(-1,1)
		jitter_shimmer_norm           = jitter_shimmer - jitter_shimmer_mean
	else:
		# did not do session level normalization
		print("Ignore session level feature normalization... ", sys.stderr)
		# f0 normalization
		f0                            = np.copy(feat_data[:, 70])
		# replace 0 in f0 with nan
		f0[f0==0.]                     = np.nan
		f0_mean                       = np.nanmean(f0)
		f0                            = np.reshape(f0,(-1,1))
		
		# f0_de normalization
		f0_de                         = np.copy(feat_data[:, 74])
		f0_de[f0_de==0.]               = np.nan
		f0_de                         = np.reshape(f0_de,(-1,1))
		
		# intensity normalization
		intensity                     = np.copy(feat_data[:,2])
		intensity                     = np.reshape(intensity, (-1,1))
		
		# intensity_de normalization
		intensity_de                  = np.copy(feat_data[:,36])
		intensity_de                  = np.reshape(intensity_de, (-1,1))
		
		# feat_idx                      = range(3,34) + range(37, 68)   with spectral de
		feat_idx                      = range(3,34)
		mfcc_etc                      = np.copy(feat_data[:,feat_idx])
		mfcc_etc_norm                 =  np.copy(mfcc_etc) 
		
		# jitter and shimmer normalization
		idx_jitter_shimmer            = [71,72,73]
		jitter_shimmer                = np.copy(feat_data[:,idx_jitter_shimmer])
		jitter_shimmer[jitter_shimmer==0.] = np.nan
		jitter_shimmer_norm           = jitter_shimmer 

	return np.hstack((f0, f0_de, intensity, intensity_de,  jitter_shimmer_norm, mfcc_etc_norm))



def final_feat_calculate(sample_index, all_raw_norm_feat, all_raw_feat_dim):
	whole_output_feat = np.array([], dtype=np.float32).reshape(0, all_raw_feat_dim*6)
	for idx_frame in sample_index:
		tmp_all_raw_norm_feat = np.copy(all_raw_norm_feat[idx_frame,:])
		funcs_per_frame = func_calculate(tmp_all_raw_norm_feat)
		whole_output_feat = np.concatenate((whole_output_feat, funcs_per_frame), axis=0)
	return whole_output_feat

def func_calculate(input_feat_matrix):
	'''
		Given a numpy array calculate its statistic functions 
		6 functions: mean, median, std, perc1, perc99, range99-1
	'''
	output_feat = np.array([], dtype=np.float32).reshape(1, -1)
	num_feat = input_feat_matrix.shape[1]
	for i in range(num_feat):
		#print i
		tmp              = input_feat_matrix[:,i]
		tmp_no_nan       = tmp[~np.isnan(tmp)]
		if tmp_no_nan.size == 0:

			mean_tmp         = 0
			std_tmp          = 0
			median_tmp       = 0
			perc1            = 0
			perc99           = 0
			range99_1        = 0
		else:
			mean_tmp         = np.nanmean(tmp)
			std_tmp          = np.nanstd(tmp)
			median_tmp       = np.median(tmp_no_nan)
			tmp_no_nan_sorted= np.sort(tmp_no_nan)
			total_len        = tmp_no_nan_sorted.shape[0]
			perc1_idx        = np.int_(np.ceil(total_len*0.01))
			if perc1_idx >= total_len:
				perc1_idx = 0 
			perc99_idx       = np.int_(np.floor(total_len*0.99))
			if perc99_idx < 0 or perc99_idx >= total_len:
				perc99_idx = total_len-1 
			perc1            = tmp_no_nan_sorted[perc1_idx]
			perc99           = tmp_no_nan_sorted[perc99_idx]
			range99_1        = perc99 - perc1
		# append for one 
		new_func = np.array([mean_tmp, median_tmp, std_tmp, perc1, perc99, range99_1])
		new_func = np.reshape(new_func, (1,6))
		output_feat = np.hstack((output_feat,new_func))

	return output_feat

def get_neighbor(sess_id, ivec_norm_dict, utt_id):
	turn2ivec = ivec_norm_dict[utt_id]
	turnlist = list(ivec_norm_dict.keys())
	candidates = random.sample(turnlist, 1000)
	candidates = list(filter(lambda x: sess_id not in x, candidates))
	lenpool = len(candidates)
	cosD = np.zeros(lenpool)
	for i in range(lenpool):
		target = ivec_norm_dict[candidates[i]]
		cosD[i] = spatial.distance.cosine(turn2ivec, target)
	chosen = candidates[np.argmin(cosD)]
	return chosen