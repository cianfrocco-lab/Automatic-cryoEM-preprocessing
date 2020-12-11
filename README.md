# Automatic-cryoEM-preprocessing
Tools to run user-free preprocessing of cryo-EM datasets: https://www.biorxiv.org/content/10.1101/2019.12.20.885541v1

**Don't want to read the instructions but still want to try it out?**

MicAssess and 2DAssess are incorporated into the freely available for academic research on COSMIC2 science gateway: https://cosmic2.sdsc.edu:8443/gateway/. Just upload your input files and you can run the jobs on the cloud!

**Updates (12/1/2020, v0.2.0)**
1. MicAssess now supports Relion 3.1 star file as the input.
2. Fix requirements dependency issues.

**Note (5/8/2020)**
2DAssess gives syntax error for some users. We have fix the bug and it should be ok to run now.


**Updates (3/7/2020, v0.1.0)**
1. MicAssess now supports micrographs from K3 as well as K2.
2. pip install now enabled. (Credit to @pconesa)
3. MicAssess now can take a single mrc file or any valid glob wildcard as the input. (Credit to @pconesa)
4. MicAssess - now can specify which GPU(s) to use for prediction.

**Installation:**

Both MicAssess and 2DAssess are python based and need anaconda installed to run. Anaconda can be downloaded and installed here: https://www.anaconda.com/distribution/

1. Create an anaconda environment
```
conda create -n cryoassess -c anaconda python=3.6 pyqt=5 cudnn=7.1.2 intel-openmp=2019.4
```
2. Activate this conda environment by
```
conda activate cryoassess
```
3. Install cryoassess (this package) for cpu
```
pip install path-to-local-clone[cpu]
```
Alternatively, if using GPU:
```
pip install path-to-local-clone[gpu]
```

**Download .h5 model files:**

You will need the pre-trained model files to run MicAssess and 2DAssess. To download them, please go to https://cosmic-cryoem.org/software/cryo-assess/. You will need to fill in a short form, agree the terms and conditions, and we will email you the download link. These pre-trained neural networks are freely available for academic research.

**MicAssess:**

Note: MicAssess currently works on micrographs from both K2 and K3 camera.

Note: MicAssess currently does not support star file from Relion 3.1.

You will need to activate the conda environment by ```conda activate cryoassess``` before using MicAssess.

To run MicAssess:
```
micassess -i <a micrograph star file> -m <model file>
```

Optional arguments:

-d, --detector: Either "K2" or "K3". Default is "K2".

-o, --output: Name of the output star file. Default is good_micrographs.star.

-b, --batch_size: Batch size used in prediction. Default is 32. Increasing this number will result in faster prediction, if your GPU memory allows. If memory error/warning appears, you should lower this number.

-t, --threshold: Threshold for classification. Default is 0.1. Higher number will cause more good micrographs being classified as bad.

--threads: Number of threads for conversion. Default is None, using mp.cpu_count(). If get memory error, set it to a reasonable number (e.g. 10). This usually happens when you have super-resolution microgarphs from K3.

--gpus: Specify which GPU(s) to use, e.g. 0,1,2,3. Default is 0, which uses only the first GPU.

The input of MicAssess could be a .star file with a header similar to this:
```
data_
loop_
_rlnMicrographName
micrographs/xxxxxxx01.mrc
micrographs/xxxxxxx02.mrc
```
Note that the header must have the "\_rlnMicrographName". The star file must be in the correct relative path so that all the mrc files can be found.

Optionally, input could be a folder where micrographs are, or a pattern where wildcards are accepted. (See https://docs.python.org/3.6/library/glob.html for more details)

MicAssess will output a "good_micrographs.star" file in the same directory of the input star file. It will also create a MicAssess directory with all the predictions (converted to .jpg files), in case you want to check the performance.

Note: if memory warning appears:
(W tensorflow/core/framework/allocator.cc:108] Allocation of 999571456 exceeds 10% of system memory.)
Reduce the batch size by adding ‘-b 16’, or even a smaller number (8 or 4). The default batch size is 32. You can also increase the batch size to a higher number like 64, if your memory allows. Higher batch size means faster.

Note: We found in practice, the default threshold (0.1) will cause some empty images being misclassified to the "good" class. Increasing the threshold to 0.3 will help to solve this problem.

**2DAssess:**

You will need to activate the conda environment by ```conda activate cryoassess``` before using 2DAssess.

To run 2DAssess:
```
2dassess -i <mrcs file outputted by RELION 2D classification> -m <model file>
```
The input of 2DAssess should be an .mrcs file outputted by RELION 2D classification with all the 2D class averages. The name is usually similar to "run_it025_classes.mrcs".
2DAssess will print the indices of the good class averages after the prediction. It will also output predicted 2D class averages into four different classess in the 2DAssess folder. All the class averages are already converted to .jpg files to ease the manual inspection.
