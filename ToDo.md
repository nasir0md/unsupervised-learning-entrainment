# ToDo


1. ~~Create a config file with all absolute paths, so users can edit it to their convienience.~~ <br> This can be found [here](https://github.com/clulab/tomcat-speech/blob/master/tomcat_speech/models/parameters/multitask_config.py)
2. ~~Create a metadata file with the following fields from .tbl file~~
3. ~~Make the Kaldi file generator work, add the files to .gitignore~~
4. Test Torch
~~5. Figure out what `feat_extract_triplet_ivec.py` does~~
6. ~~Install:~~
    - ~~pip2~~
    - ~~torch 1.4~~
    - ~~torchvision~~
7. Research if the features are concatenated, averaged or a different calculation is employed
    - try train.py, (command+click for different variables/classes)
    - Checkout normutils.py, see if it helps
8. Find out how long these things take
    - creation of ivectors could be time-consuming
    - Kaldi can take time, comp. resources.
~~9. How are ivectors created?~~ 
   - ~~Kaldi files ? Check if these are pre-created, or if they are created.~~
10. Check out load: tkinter
11. What does `tmp.csv` in `NED/create_h5data.py` do?
    - Gets created in the python file, so look into the notes there.
    - Also, feats directory is empty, so work on running run_all_nopre.sh properly
    - Right now, it's not running opensmile and `feat_extract_nopre.py` correctly.
12. How is the `/media/nasir/xData/newdata/Fisher/ldc2004s13/fe_03_p1_sph1/feats/000/` being generated in analysis_trial.m? What does it do?
13. How is `/home/nasir/data/Fisher/feats_triplets_all/` in `triplet/prep/create_h5data.py` created? What does it do?
14. Fix `test/test_session.py` with the workpace directories
15. Work on a README
    1. External dependencies:
	    - LDC data
	    - Python dependencies
	    - Software
        - Creating a `trans` directory with all transcripts
        - Code for creating `fisher_meta.csv`
    2. Editing config file
    3. Main entry Point
16. Fix requirements
17. Upload Fisher corpus
18. Add master directory for all Fisher-related stuff
19. Add `chmod` info to Readme