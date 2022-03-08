# ADaptive-Structural-Fingerprint
implementation for Adaptive Structural Fingerprints for Graph Attention Networks 

most of the code are copy from the https://github.com/AvigdorZ/ADaptive-Structural-Fingerprint

However the code in that link can not be run successfully, so i make another implement which all the data and code are included.

    conda create -n ADSF python=3.9

    pip install -r requirements.txt 

    python train.py

the output is in `nohup.out`

`743.pkl`is the best model.

`interdata` has all the intermediate data, which are called in the `utils_nhop_neighbours.py`.

you can add a extra 'or True' after these branch to generate these files by yourself but it would cost a really long time(may be 3-6 hours)

if you have some great idea that can make the code has a better style, don't hesitate to take a PR.