import glob
import os
import csv
import pdb

transcript_dir='/home/nasir/data/Fisher/transcripts/'
audio_dir_root = "/media/Drobo/Data/Fisher/ldc2004s13/"
metaf = open('Fisher_meta.csv', 'rb')


reader = csv.reader(metaf)
metadata ={}
for row in reader:
	metadata[row[0]] = row[1:]

wavscpf = open('./wav.scp', 'w')
segf = open('./segments', 'w')
uttf = open('./utt2spk', 'w')


for dir in os.listdir(audio_dir_root):
	if "fe_03_p1" in dir:
		subdir = audio_dir_root + dir + "/audio" 
		for subsubdir in os.listdir(subdir):
			for audio in os.listdir(subdir + '/' + subsubdir):
				audio_path = subdir + '/' + subsubdir + '/'+ audio
				audio = audio.split(".")[0]
				sess_id = audio.split('_')[-1]
				wavscpf.write(audio + ' sph2pipe -f wav -p -c 1 ' + audio_path + ' |\n')
				transcript =  transcript_dir + audio + '.txt'
				trans = open(transcript).readlines()
				spk_list = []
				for line in trans:
					if line!='\n':
						if line[0] !='#':
							start, stop, spk = line.split(':')[0].split(' ')
							if spk=="A":
								spk = metadata[sess_id][1]
							else:
								spk = metadata[sess_id][3]
							spk_list.append([start, stop, spk])
							utt_id = spk+ '-' + audio +  '_' + str(int(1000*float(start))) 	+ '-' + str(int(1000*float(stop))) 
							segf.write(utt_id + ' ' + audio + ' ' + start + ' '+ stop + '\n')
							uttf.write(utt_id + ' ' +spk +'\n')
				# print audio, spk_list[0][2]




wavscpf.close()
segf.close()
uttf.close()
