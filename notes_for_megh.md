  1. hard-coding of file locations
     - we discussed this already; using config files to control all hard-coding in a single easy-to-alter location
  2. opensmile configs
     - easy: can you add the .conf files into the auto-installed code so that it gets copied into the same location as all other config files?
     - alternative: use relative paths to access these conf files in particular (~/feats/xxxxxx.conf) in the processing code
  3. set up pytorch and make sure it runs 
     - add it to a requirements.txt file or a setup.py file
     - create a venv ?
     - in pycharm, you can add this to your project resources (preferences > project: unsupervised-learning-entrainment > python interpreter, use the plus to add torch)
        - this didn't work for me (claimed i didn't have pip)
       - create a conda environment (this is what i did)
         - use python=2.7 for environment
         - pip install torch==1.4.0 torchvision
  				- torch will be downloaded for that environment
  				- still cannot download it in pycharm, but seems to work
  				- i had to use torch==1.4.0 because 1.5 doesn't work with torchvision
  4. you haven't included the links to the files here. skipping for now
  5. run_all_nopre.sh
  6. "figure out what these files do"
     - create_h5data.py: takes csv files from the fisher corpus, splits into train dev, test partitions, and converts each data file from a csv to an h5 data format (like a dict but annoying)
     - create_kaldi_files.py: seems to be creating files containing the segment/utterance information and lines preparing the audio data to be automatically converted from wav to pcm using: https://github.com/robd003/sph2pipe/blob/master/sph2pipe.c
     - create_line_file.py: creates a transcript of the lines in the data transcript--VERY similar to the above 
     - create_samples_siamese.py: creates train, dev, test samples from feature data (ivectors, so speaker-specific features?) for use in a siamese network. this one was less clear to me 
     - ivector_scp2pkl.py: uses this method: https://github.com/vesis84/kaldi-io-for-python/blob/master/kaldi_io/kaldi_io.py L346 to convert .scp ivector files to python pickle files 
     - normutils.py: contains functions to normalize data, calculate functionals (e.g. mean, stdev, range)
     - run_triplet_batch.sh: runs a file that doesn't seem to exist in the repo (feat_extract_triplet_ivec.py)
 
