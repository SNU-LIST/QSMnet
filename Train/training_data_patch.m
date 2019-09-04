function training_data_patch(training_bf_patch_dir, mask_dir, aug_process, sub_num, aug_num, symmetry_mode)

%
% Description:
%    Patching training data into 64 x 64 x 64 matrices
%   
% Inputs:
%   training_bf_patch_dir : directory of training data before patch
%                           inputs  - local field data (multiphs)
%                           outputs - cosmos susceptibility data (multicos)
%   mask_dir : director of mask data
%   aug_process : data augmentation type - 1 for QSMnet, 2 for QSMnet+ 
%   sub_num : subject number (max 7)
%   aug_num : augmentation number (max 5)
%   symmetry_mode : data augmentation by inverting the sign of trainig datset
%                   if symmetry_mode == 1, training data size become double
%
% Outputs:
%   save training_data_patch_64.mat 
%
% Copyright @ Woojin Jung
% Laboratory for Imaging Science and Technology
% Seoul National University
% email: wjjung93@snu.ac.kr
%


load(training_bf_patch_dir);
load(mask_dir);

scaling_factor = 4;
patch_stride = [7, 8, 6];
matrix_cut = [172, 176, 159];

PS = 64; %patch size 64 - 172, 176, 159 
str_x = (matrix_cut(1)-PS)/(patch_stride(1)-1);
str_y = (matrix_cut(2)-PS)/(patch_stride(2)-1);
str_z = (matrix_cut(3)-PS)/(patch_stride(3)-1);


[yy, xx] = meshgrid(1:patch_stride(2),1:patch_stride(1));
xx = repmat(xx,[1,1,patch_stride(3)]);
yy = repmat(yy,[1,1,patch_stride(3)]);
for kk=1:patch_stride(3)
    zz(:,:,kk) = ones(patch_stride(1),patch_stride(2))*kk;
end
xx=xx(:);
yy=yy(:);
zz=zz(:);
tt=length(xx);

data_input = single(zeros(PS,PS,PS,sub_num,tt*(5+aug_num)));
data_label = single(zeros(PS,PS,PS,sub_num,tt*(5+aug_num)));
mask_input = single(zeros(PS,PS,PS,sub_num,tt*(5+aug_num)));

matrix_size=[176, 176, 160];
voxel_size=[1,1,1];
B0_dir = [0, 0, 1];
D = single(dipole_kernel(matrix_size,voxel_size,B0_dir));
input_voxel = [];
output_voxel = [];

for ii=1:sub_num
    eval(sprintf('data1 = multicos%d;',ii));
    eval(sprintf('data2 = multiphs%d;',ii));
    eval(sprintf('maskt = mask%d;',ii));
    % Original training dataset
    for aug = 1:5
        data_L = data1(3:174,:,1:159,aug);
        data_I = data2(3:174,:,1:159,aug);
        mask_L = maskt(3:174,:,1:159,aug);
        
        % Process for data normalization
        cc = data_L;
        pp = data_I;
        mm = mask_L;
        cc(mm==0)=0;
        cc=cc(:);
        cc(cc==0)=[];
        pp(mm==0)=0;
        pp=pp(:);
        pp(pp==0)=[];
        input_voxel = cat(1,input_voxel,pp);
        output_voxel = cat(1,output_voxel,cc);

        for jj=1:tt;
            data_input(1,:,:,:,ii,tt*(aug-1)+jj) = data_I(str_x*xx(jj)-(str_x-1):str_x*xx(jj)+(PS-str_x),...
                str_y*yy(jj)-(str_y-1):str_y*yy(jj)+(PS-str_y),str_z*zz(jj)-(str_z-1):str_z*zz(jj)+(PS-str_z));
            data_label(1,:,:,:,ii,tt*(aug-1)+jj) = data_L(str_x*xx(jj)-(str_x-1):str_x*xx(jj)+(PS-str_x),...
                str_y*yy(jj)-(str_y-1):str_y*yy(jj)+(PS-str_y),str_z*zz(jj)-(str_z-1):str_z*zz(jj)+(PS-str_z));
            mask_input(1,:,:,:,ii,tt*(aug-1)+jj) = mask_L(str_x*xx(jj)-(str_x-1):str_x*xx(jj)+(PS-str_x),...
                str_y*yy(jj)-(str_y-1):str_y*yy(jj)+(PS-str_y),str_z*zz(jj)-(str_z-1):str_z*zz(jj)+(PS-str_z));
        end
    end
    % Data augmentation by dipole convolution
    if aug_num < 1 
        break;
    else
        for aug = 6:aug_num+5
            data_L = data1(:,:,:,aug);
            if aug_process == 1 % QSMnet (no scaling factor applied)
                data_L = data1(3:174,:,1:159,aug);
                net_name = 'QSMnet';
            elseif aug_process == 2 % QSMnet+
                data_L(data_L>0) = data_L(data_L>0)*scaling_factor;
                net_name = 'QSMnet+';
            end
            data_I = ifftn(fftn(data_L).*D);
            
            data_L = data_L(3:174,:,1:159,aug);
            data_I = data_I(3:174,:,1:159,aug);
            mask_L = maskt(3:174,:,1:159,aug);
            
            % Process for data normalization
            cc = data_L;
            pp = data_I;
            mm = mask_L;
            cc(mm==0)=0;
            cc=cc(:);
            cc(cc==0)=[];
            pp(mm==0)=0;
            pp=pp(:);
            pp(pp==0)=[];
            input_voxel = cat(1,input_voxel,pp);
            output_voxel = cat(1,output_voxel,cc); 
            
            for jj=1:tt;
                data_input(1,:,:,:,ii,tt*(aug-6)+jj) = data_I(str_x*xx(jj)-(str_x-1):str_x*xx(jj)+(PS-str_x),...
                    str_y*yy(jj)-(str_y-1):str_y*yy(jj)+(PS-str_y),str_z*zz(jj)-(str_z-1):str_z*zz(jj)+(PS-str_z));
                data_label(1,:,:,:,ii,tt*(aug-6)+jj) = data_L(str_x*xx(jj)-(str_x-1):str_x*xx(jj)+(PS-str_x),...
                    str_y*yy(jj)-(str_y-1):str_y*yy(jj)+(PS-str_y),str_z*zz(jj)-(str_z-1):str_z*zz(jj)+(PS-str_z));
                mask_input(1,:,:,:,ii,tt*(aug-6)+jj) = mask_L(str_x*xx(jj)-(str_x-1):str_x*xx(jj)+(PS-str_x),...
                    str_y*yy(jj)-(str_y-1):str_y*yy(jj)+(PS-str_y),str_z*zz(jj)-(str_z-1):str_z*zz(jj)+(PS-str_z));
            end
        end
    end
end

data_input = data_input(1,:,:,:,:);
data_label = data_label(1,:,:,:,:);
mask_input = mask_input(1,:,:,:,:);

data_input = single(permute(data_input,[1 4 3 2 5])); 
data_label = single(permute(data_label,[1 4 3 2 5]));
mask_input = single(permute(mask_input,[1 4 3 2 5]));

if symmetry_mode == 1
    sym_name = 'sym';
    data_input = cat(4, data_input, -data_input);
    data_label = cat(4, data_label, -data_label);
    mask_input = cat(4, mask_input, mask_input);
    input_voxel = cat(1,input_voxel, -input_voxel);
    output_voxel = cat(1,output_voxel, -output_voxel);
else
    sym_name = 'asym';
end
save_data_name = ['training_data_patch_64_',net_name,'_',sym_name,'.mat'];
save_norm_name = ['norm_factor_',net_name,'_',sym_name,'.mat'];

input_mean = mean(input_voxel);
input_std = std(input_voxel);
label_mean = mean(output_voxel);
label_std = std(output_voxel);

eval(sprintf('save %s data_input data_label mask_input -v7.3',save_data_name));
eval(sprintf('save %s input_mean input_std label_mean label_std -v7.3',save_norm_name));
