# QSMnet
* The code is for reconstructing Quantitative Susceptibility Mapping by deep neural network (QSMnet). Data preprocessing (.m) and the inference code of QSMnet (.py) are availabe. 

# References
* QSMnet </br>
_J. Yoon, E. Gong, I. Chatnuntawech, B. Bilgic, J. Lee, W. Jung, J. Ko, H. Jung, K. Setsompop, G. Zaharchuk, E.Y. Kim, J. Pauly, J. Lee.
Quantitative susceptibility mapping using deep neural network: QSMnet.
Neuroimage. 2018 Oct;179:199-206._
* QSMnet+ </br>
_W. Jung, J. Yoon, J. Choi, E. Kim, J. Lee. On the linearity of deep neural network trained QSM.
ISMRM 27th annual meeting & exhibition. 2019 May:0317._

# Overview
![Graphical_abstract](https://user-images.githubusercontent.com/29892433/62440733-5d4ad300-b78c-11e9-975d-ca56e77422aa.jpg)


## Requirements
* Python 2.7

* Tensorflow 1.8.0

* NVIDIA GPU (CUDA 9.0)

* MATLAB R2015b

## Manual

### Data acquisition
* Training data of QSMnet was acquired at 3T MRI (SIEMENS).
* 3D single-echo GRE scan with following sequence parameters: FOV = 256 x 224 x 176 mm<sup>3</sup>, voxel size = 1 x 1 x 1 x mm<sup>3</sup>, TR = 33 ms, TE = 25 ms, bandwidth = 100 Hz/pixel, flip angle = 15°.


### Phase processing
* Requirements
  * MEDI toolbox (_Ref: T. Liu, J. Liu, L.D. Rochefort, P. Spincemaille, I. Khalidov, J.R. Ledoux, Y. Wang. Morphology enabled dipole inversion (MEDI) from a single‐angle acquisition: comparison with COSMOS in human brain imaging. Magnetic resonance in medicine. 2011 Apr;66(3):777-783._)
  * STI Suite (_Ref: W. Li, A.V. Avram, B. Wu, X. Xiao, C. Liu. Integrated Laplacian‐based phase unwrapping and background phase removal for quantitative susceptibility mapping. NMR in Biomedicine. 2014 Dec;27(2):219-227._)
  
* Process flow
  * Extract magnitude and phase image from DICOMs
  * Brain extraction : BET (MEDI toolbox)
  * Phase unwrapping : Laplaican phase unwrapping (STI Suite)
  * Background field removal : 3D V-SHARP (STI Suite)
  
* Usage:
```bash
save_input_data_for_QSMnet(TissuePhase, Mask, TE, B0)
% TissuePhase : Results of 3D V-SHARP
% Mask : Results of 3D V-SHARP
% TE (ms)
% B0 (T)
% Convert unit from Hz to ppm : field / (Sum(TE) * B0 * gyro) [ppm]
```
  * 'inf_data.mat' file will be saved after phase processing.
  
### Inference
* Requirements in python library
  * tensorflow
  * os
  * argparse
  * scipy.io
  * numpy
  * niblabel

* Usage
First Time Only
1. Clone this repository

```bash
git clone https://github.com/SNU-LIST/QSMnet.git
```
2. Download network 
```bash
python inference_QSMnet.py <PREPROCESS_DIR> <NETWORK_NAME>
```
  * 'deep_result_<network_name>.mat' & 'deep_result_<network_name>.nii' will be save after QSMnet reconstruction.
