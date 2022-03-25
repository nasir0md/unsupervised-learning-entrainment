#!/usr/bin/env bash
# cmddir=/home/nasir/inter_dynamics/scripts/NPC
cmddir=.
featdir=/home/nasir/data/Fisher/feats_triplets
raw_featdir=/home/nasir/data/Fisher/raw_feats
data_dir_triplets= '/Users/meghavarshinikrishnaswamy/Downloads/Fisher_corpus/feats_triplets'
raw_featdir='/Users/meghavarshinikrishnaswamy/Downloads/Fisher_corpus/raw_feats'

# audiodirroot=/data/Fisher/ldc2004s13

numParallelJobs=4
ctr=1
# for dir in $audiodir/*;
# do  
# 	cd $dir
# 	for file in *;
# 	do 
# 	python $cmddir/feat_extract.py --audio_file $dir/$file --openSMILE_config $cmddir/emobase2010_haoqi_revised.conf --output_path $featdir
# done
# 	cd ..
# done

for f in $raw_featdir/*fe_03_04{5,6,7,8,9}*.csv;
do
	# python $cmddir/feat_extract_nopre.py --audio_file $file --openSMILE_config $cmddir/emobase2010_haoqi_revised.conf --output_path $featdir
		echo $f;
	 (
	 	python $cmddir/feat_extract_triplet_ivec.py --audio_file $f --openSMILE_config $cmddir/emobase2010_haoqi_revised.conf --output_path $featdir
	 	) &
	if [ $(($ctr % $numParallelJobs)) -eq 0 ]
	then
		echo "Running $numParallelJobs jobs in parallel.."
		wait
	fi
	ctr=`expr $ctr + 1`
done




#   FOR AUDIO FILES


# for dir in $audiodirroot/f*;
# do
# 	for f in  $dir/audio/*/*.sph;
# 	do
# 		echo $f;
# 	 (
# 	 	python $cmddir/feat_extract_triplet_ivec.py --audio_file $f --openSMILE_config $cmddir/emobase2010_haoqi_revised.conf --output_path $featdir
# 	 	) &
# 	if [ $(($ctr % $numParallelJobs)) -eq 0 ]
# 	then
# #		echo "Running $numParallelJobs jobs in parallel.."
# 		wait
# 	fi
# 	ctr=`expr $ctr + 1`
	 	
# 	done;
# done
