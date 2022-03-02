#!/usr/bin/env bash
# cmddir=/home/nasir/inter_dynamics/scripts/NPC
cmddir=.
featdir=/Users/meghavarshinikrishnaswamy/Downloads/Fisher_corpus/feats
raw_featdir=/Users/meghavarshinikrishnaswamy/Downloads/Fisher_corpus/raw_feats
audiodirroot=/Users/meghavarshinikrishnaswamy/Downloads/Fisher_corpus/fisher_eng_tr_sp_LDC2004S13_zip_2

numParallelJobs=24
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

# for file in $audiodir/*.csv;
# do
# 	python $cmddir/feat_extract_nopre.py --audio_file $file --openSMILE_config $cmddir/emobase2010_haoqi_revised.conf --output_path $featdir
# done


for dir in $audiodirroot/f*;
do
	for f in  $dir/audio/*/*.sph;
	do
		echo $f;
	 (
	 	python $cmddir/feat_extract_nopre.py --audio_file $f --openSMILE_config $cmddir/emobase2010_revised.conf --output_path $featdir
	 	) &
	if [ $(($ctr % $numParallelJobs)) -eq 0 ]
	then
#		echo "Running $numParallelJobs jobs in parallel.."
		wait
	fi
	ctr=`expr $ctr + 1`
	 	
	done;
done
