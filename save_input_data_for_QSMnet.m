function save_input_data_for_QSMnet(TissuePhase, Mask, TE, B0, out_dir)
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
%   out_dir : directory of patient data
% Ouputs:
%   inf_data.mat file will be saved for network input
%
% Copyright @ Woojin Jung
% Laboratory for Imaging Science and Technology
% Seoul National University
% email: wjjung93@snu.ac.kr
%

gyro = 2*pi*42.58;
field = TissuePhase / (sum(TE)*B0*gyro);
if ~exist(out_dir, 'dir')
       mkdir(out_dir)
end
cd(out_dir)
    
save field_data.mat field Mask
