# QSMnet
* The code is for reconstructing Quantitative Susceptibility Mapping by deep neural network (QSMnet). Data preprocessing (.m) and the inference code of QSMnet (.py) are availabe. 

# References
* J. Yoon, E. Gong, I. Chatnuntawech, B. Bilgic, J. Lee, W. Jung, J. Ko, H. Jung, K. Setsompop, G. Zaharchuk, E.Y. Kim, J. Pauly, J. Lee
Quantitative susceptibility mapping using deep neural network: QSMnet
Neuroimage, 179 (2018), pp. 199-206.

# Overview
![Graphical_abstract](https://user-images.githubusercontent.com/29892433/62440733-5d4ad300-b78c-11e9-975d-ca56e77422aa.jpg)


## Requirements
* Python 2.7

* Tensorflow 1.8.0

* NVIDIA GPU (CUDA 9.0)

* MATLAB R2015b

## Usage

### Data acquisition
* Training data of QSMnet was acquired at 3T MRI (SIEMENS).
* 3D single-echo GRE scan was acuiqred using following sequence parameters: FOV = 256 x 224 x 176 mm<sup>3</sup>, voxel size = 1 x 1 x 1 x mm<sup>3</sup>, TR = 33 ms, TE = 25 ms, bandwidth = 100 Hz/pixel, flip angle = 15°.


### Phase processing
* Requirements
  * MEDI toolbox
  * STI Suite
  
* Process flow
  * Extract magnitude and phase image from DICOMs
  * Brain extraction : BET (MEDI toolbox)
  * Phase unwrapping : Laplaican phase unwrapping (STI Suite)
  * Background field removal : 3D V-SHARP (STI Suite)
  * Convert unit from Hz to ppm : field / (Sum(TE) * B0 * gyro) [ppm]
  
* Usage:
```bash
Phase_processing_for_QSMnet('DICOM_DIR','MATLAB_CODE_DIR','PREPROCESS_DIR')
% DICOM_DIR : Directory of DICOM folder
% MATLAB_CODE_DIR : Directory of Matlab_code
% PREPROCESS_DIR : Directory of phase processing result
```
  * 'field_data.mat' file will be saved after phase processing.
  * Due to 4 pooling layers in U-net structure, the data is cropped to **multiple of 16**.
  
### Inference
* Requirements in python library
  * tensorflow
  * h5py
  * os
  * argparse
  * scipy.io
  * numpy
  * niblabel

* Usage
```bash
python inference_QSMnet.py <PREPROCESS_DIR> <NETWORK_NAME>
```
  * 'deep_result_<network_name>.mat' & 'deep_result_<network_name>.nii' will be save after QSMnet reconstruction.
