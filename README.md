# QSMnet & QSMnet<sup>+</sup>
* The code is for reconstructing Quantitative Susceptibility Mapping by deep neural network (QSMnet) and QSMnet<sup>+</sup>. QSMnet<sup>+</sup> covers a wider range of susceptibility than QSMnet, using data augmentation approach. Data preprocessing, training and  inference code of QSMnet (.py) are availabe. 
* Last update : 2020.10.26

# References
* QSMnet </br>
_J. Yoon, E. Gong, I. Chatnuntawech, B. Bilgic, J. Lee, W. Jung, J. Ko, H. Jung, K. Setsompop, G. Zaharchuk, E.Y. Kim, J. Pauly, J. Lee.
Quantitative susceptibility mapping using deep neural network: QSMnet.
Neuroimage. 2018 Oct;179:199-206. https://www.sciencedirect.com/science/article/pii/S1053811918305378_
* QSMnet+ </br>
_W. Jung, J. Yoon, S. Ji, J. Choi, J. Kim, Y. Nam, E. Kim, J. Lee. Exploring linearity of deep neural network trained QSM: QSMnet+.
Neuroimage. 2020 May; 116619. https://www.sciencedirect.com/science/article/pii/S1053811920301063_</br>
* Review of deep learning QSM </br>
_W. Jung, S. Bollmann, J. Lee. Overview of quantitative susceptibility mapping using deep learning: Current status, challenges and opportunities.
NMR in Biomedicine. 2020 March; e4292. https://doi.org/10.1002/nbm.4292

# Overview
## (1) QSMnet
![Graphical_abstract](https://user-images.githubusercontent.com/29892433/62440733-5d4ad300-b78c-11e9-975d-ca56e77422aa.jpg)
## (2) QSMnet<sup>+</sup>
![Figure_QSMnetp](https://user-images.githubusercontent.com/29892433/66732154-a65a8a00-ee95-11e9-90aa-f23b0d6ee863.png)

## Requirements
* Python 3.7

* Tensorflow 1.14.0

* NVIDIA GPU (CUDA 10.0)

* MATLAB R2015b

## Data acquisition
* Training data of QSMnet was acquired at 3T MRI (SIEMENS).
* 3D single-echo GRE scan with following sequence parameters: FOV = 256 x 224 x 176 mm<sup>3</sup>, voxel size = 1 x 1 x 1 x mm<sup>3</sup>, TR = 33 ms, TE = 25 ms, bandwidth = 100 Hz/pixel, flip angle = 15°.

## Manual

### First Time Only
1. Clone this repository
```bash
git clone https://github.com/SNU-LIST/QSMnet.git
```
2. Download network </br>
In Checkpoints directory,
* For Linux User,
```bash
sh download_network.sh
```
* For Windows User, </br>
https://drive.google.com/drive/folders/1E7e9thvF5Zu68Sr9Mg3DBi-o4UdhWj-8 </br>
and unzip the files in 'Checkpoints/' </br>
* Last update : 2019.10.14

### Phase processing
* Requirements
  * FSL (_Ref: S.M. Smith. Fast robust automated brain extraction. Human brain mapping. 2002 Sep;17(3):143-155._)
  * STI Suite (_Ref: W. Li, A.V. Avram, B. Wu, X. Xiao, C. Liu. Integrated Laplacian‐based phase unwrapping and background phase removal for quantitative susceptibility mapping. NMR in Biomedicine. 2014 Dec;27(2):219-227._)
  
* Process flow
  * Extract magnitude and phase image from DICOMs
  * Brain extraction : BET (FSL)
  * Phase unwrapping : Laplaican phase unwrapping (STI Suite)
  * Background field removal : 3D V-SHARP (STI Suite)
  
* Usage:
  * If you acquired data with different resolution from 1 x 1 x 1 mm<sup>3</sup>,</br>
    you need to interpolate the data into 1 mm isotropic resolution before phase processing.</br>
    (e.g. zero-padding or truncating in Fourier domain)
    
  * After 3D V-SHARP in MATLAB, run 'save_input_data_for_QSMnet.m</br>'.
    'test_input{sub_num}.mat' and 'test_mask{sub_num}.mat' files will be saved in 'Data/Test/Input/'.
  ```bash
    save_input_data_for_QSMnet(TissuePhase, Mask, TE, B0, sub_num)
    % TissuePhase : Results of 3D V-SHARP
    % Mask : Results of 3D V-SHARP
    % TE (s)
    % B0 (T)
    % sub_num : subject number
    % Convert unit from Hz to ppm : field / (Sum(TE) * B0 * gyro) [ppm]
  ```
  * Save data with the same orientation and polarity as val_input.mat, val_mask.mat, and val_label.mat files in 'Data/Train/' folder.
   <img src="https://user-images.githubusercontent.com/29892433/64081330-5f2b9600-cd3a-11e9-9ff2-20e1e0ef2996.jpg" width="50%" height="50%">
  
### Training data
* The source data for training can be shared to academic institutions. Request should be sent to snu.list.software@gmail.com. For each request, individual approval from our institutional review board is required (i.e. takes time)

### Training process
* Requirements in python library
  * tensorflow, numpy, matplotlib, scipy.io, h5py, tqdm

* Usage
  * Before training, local field & susceptibility maps need to be dividied into 64 x 64 x 64 in Matlab
  
  ```bash
  python training_data_patch.py
  # PS : Patch size
  # net_name : Network name
  # sub_num : Number of subject to train
  # dir_num : Number of direction per subject
  # patch_num : Number of patches in [x, y, z]
  ```
  
  * Training process in python
  
  ```bash
  python train.py
  ```
  
### Inference
* Requirements in python library
  * tensorflow, scipy.io, matplotlib, numpy, niblabel

* Usage
```bash
python inference.py
```
  * 'subject#_<network_name>-epochs.mat' & 'subject#_<network_name>-epochs.nii' will be saved after QSMnet reconstruction.
