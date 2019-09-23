function save_input_data_for_QSMnet(TissuePhase, Mask, TE, B0, sub_num)
%
% Description:
%    Convert unit from Hz to ppm : field / (Sum(TE) * B0 * gyro)   
%    Save input data to the appropriate format for QSMnet
%   
% Inputs:
%   TissuePhase : Result of 3D V-SHARP
%   Mask : Result of 3D V-SHARP
%   TE : unit ms
%   B0 : unit T
% Outputs:
%   test_input{sub_num}.mat file will be saved for network input
%
% Copyright @ Woojin Jung
% Laboratory for Imaging Science and Technology
% Seoul National University
% email: wjjung93@snu.ac.kr
%

gyro = 2*pi*42.58;
phs_tissue = TissuePhase / (sum(TE)*B0*gyro);
mask = Mask;
save(sprintf('Data/Test/Input/test_input%d.mat',sub_num), 'phs_tissue', 'mask');
