# Automatic-cryoEM-preprocessing (in dev)
Tools to run user-free preprocessing of cryo-EM datasets: https://www.biorxiv.org/content/10.1101/2019.12.20.885541v1

**Don't want to read the instructions but still want to try it out?**

MicAssess and 2DAssess are incorporated into the freely available for academic research on COSMIC2 science gateway: https://cosmic2.sdsc.edu:8443/gateway/. Just upload your input files and you can run the jobs on the cloud!

**Installation:**

Both MicAssess and 2DAssess are python based and need anaconda installed to run. Anaconda can be downloaded and installed here: https://www.anaconda.com/distribution/

1. Create an anaconda environment
```
conda create -n cryoassess -c anaconda python=3.6 pyqt=5 cudnn=7.1.2 numpy=1.14.5 intel-openmp=2019.4
```
2. Activate this conda environment by
```
conda activate cryoassess
```
3. Install other required packages
```
pip install tensorflow==1.10.1 keras==2.2.5 Pillow==4.3.0 mrcfile==1.1.2 pandas==0.25.3 opencv-python==4.1.2.30 scikit-image==0.16.2
```
Alternatively, if using GPU:
```
pip install tensorflow-gpu==1.10.1 keras==2.2.5 Pillow==4.3.0 mrcfile==1.1.2 pandas==0.25.3 opencv-python==4.1.2.30 scikit-image==0.16.2
```

**Download .h5 model files:**

You will need the pre-trained model files to run MicAssess and 2DAssess. To download them, please go to https://cosmic-cryoem.org/software/cryo-assess/. You will need to fill in a short form, agree the terms and conditions, and we will email you the download link. These pre-trained neural networks are freely available for academic research.

**MicAssess:**

Note: MicAssess currently only works on K2 camera data.
You will need to activate the conda environment by ```conda activate cryoassess``` before using MicAssess.

To run MicAssess:
```
python micassess.py -i <a micrograph star file> -m <model file>
```
The input of MicAssess should be a .star file with a header similar to this:
```
data_
loop_
_rlnMicrographName
micrographs/xxxxxxx01.mrc
micrographs/xxxxxxx02.mrc
```
Note that the header must have the "\_rlnMicrographName". The star file must be in the correct relative path so that all the mrc files can be found.
MicAssess will output a "good_micrographs.star" file in the same directory of the input star file. It will also create a MicAssess directory with all the predictions (converted to .jpg files), in case you want to check the performance.

Note: if memory warning appears:
(W tensorflow/core/framework/allocator.cc:108] Allocation of 999571456 exceeds 10% of system memory.)
Reduce the batch size by adding ‘-b 16’, or even a smaller number (8 or 4). The default batch size is 32. You can also increase the batch size to a higher number like 64, if your memory allows. Higher batch size means faster.

**2DAssess:**

You will need to activate the conda environment by ```conda activate cryoassess``` before using 2DAssess.

To run 2DAssess:
```
python 2dassess.py -i <mrcs file outputted by RELION 2D classification> -m <model file>
```
The input of 2DAssess should be an .mrcs file outputted by RELION 2D classification with all the 2D class averages. The name is usually similar to "run_it025_classes.mrcs".
2DAssess will print the indices of the good class averages after the prediction. It will also output predicted 2D class averages into four different classess in the 2DAssess folder. All the class averages are already converted to .jpg files to ease the manual inspection.
