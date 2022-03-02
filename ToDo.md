# ToDo


1. Create a config file with all absolute paths, so users can edit it to their convienience. <br> This can be found [here](https://github.com/clulab/tomcat-speech/blob/master/tomcat_speech/models/parameters/multitask_config.py)
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
    - creation of I-vectors could be time-consuming
    - Kaldi can take time, comp. resources.
9. How are i-vectors created? 
   - Kaldi files ? Check if these are pre-created, or if they are created.
10. Check out load: tkinter
11. What does `tmp.csv` in `create_h5data.py` do?
12. 
