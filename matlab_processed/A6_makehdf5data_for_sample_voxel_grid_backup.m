clc
clear all
close all

  L_pitch_yaw_angle_list_total = []
  R_pitch_yaw_angle_list_total =[]
for i = 34:34
    i
    for j = 5:5
        j
        load eyetracker2world_parameters  %加载眼动仪坐标系到世界坐标系的坐标转换关系
        %求解转换矩阵
        load  D:\remote_dataset\calibration\CheckerboardPoints\tform_total.mat
        
        path_folder = ['D:\remote_dataset\user_',num2str(i),'\exp',num2str(j)];
        
        [flir_began] = textread(fullfile(path_folder, '\convert2eventspace\','timestamp_win.txt'), '%f');   % event 的第一个时间戳
        flir_began = flir_began(2);
        
        [gazepoint_began ] = textread(fullfile(path_folder, '\gazepoint\','time_win.txt'), '%f');   % event 的第一个时间戳
        gazepoint_began = gazepoint_began(2);
        
        timedifference = (flir_began - gazepoint_began);
        
        totaldata  = csvread(fullfile(path_folder,'\gazepoint\gazepoint.csv'));
        %计算世界坐标系下的3d gaze，从眼动仪坐标系转换到世界坐标系，世界坐标系的原点在屏幕左上角
        
        [Gazetimestamp, LGAZE, RGAZE] = wcs_3d_gaze(totaldata,R,t);
        
        
        % 转到棋盘格的世界坐标系 x 33.7cm   y9.25cm    60.55cm 全体反向就行
        LGAZE = - LGAZE;
        RGAZE = - RGAZE;
        
        Gazetimestamp = Gazetimestamp - timedifference; % 可能最好是减掉
        
        [frame_align_event_time] = textread(fullfile(path_folder,'\convert2eventspace\frame_align_event_timestamp.txt'), '%f');   % event 的第一个时间戳
        
        voxel_grid_timestamp = textread(['C:\Users\Aiot_Server\Desktop\remote_dataset\voxel_grid_representation\user_',num2str(i),'\exp',num2str(j),'\voxel_grid_saved\inter_timestamp.txt'], '%f');
        
        left_flir_folder = dir(['C:\Users\Aiot_Server\Desktop\remote_dataset\voxel_grid_representation\user_',num2str(i),'\exp',num2str(j),'\voxel_grid_saved\left_eye\','*.jpg*']);
        left_names_file = sort_nat({left_flir_folder.name}); %每一个aps 图片的名字
        
        left_npz_folder = dir(['C:\Users\Aiot_Server\Desktop\remote_dataset\voxel_grid_representation\user_',num2str(i),'\exp',num2str(j),'\voxel_grid_saved\left_eye\','*.npz']);
        left_names_npz_file = sort_nat({left_npz_folder.name}); %每一个aps 图片的名字
        
        right_flir_folder = dir(['C:\Users\Aiot_Server\Desktop\remote_dataset\voxel_grid_representation\user_',num2str(i),'\exp',num2str(j),'\voxel_grid_saved\right_eye\','*.jpg*']);
        right_names_file = sort_nat({right_flir_folder.name}); %每一个aps 图片的名字
        
        right_npz_folder = dir(['C:\Users\Aiot_Server\Desktop\remote_dataset\voxel_grid_representation\user_',num2str(i),'\exp',num2str(j),'\voxel_grid_saved\right_eye\','*.npz']);
        right_names_npz_file = sort_nat({right_npz_folder.name}); %每一个aps 图片的名字
        
        
        time_frame = (voxel_grid_timestamp - frame_align_event_time(1))/(1e+6);
        sort_label_value_list = [];
        sort_label_ind_list = [];
        klist = [];
        L_pitch_yaw_angle_list = [];
        R_pitch_yaw_angle_list = [];
        
        
           
        hr_mat = ['D:\remote_dataset\processed_data\user_',num2str(i),'\exp',num2str(j),'\convert_frame_normalized\left_eye\left_hr_norm_list.mat']
        load (hr_mat);
        scale_mat = ['D:\remote_dataset\processed_data\user_',num2str(i),'\exp',num2str(j),'\convert_frame_normalized\left_eye\left_scale_vector_list.mat']
        load (scale_mat);
        rotate_mat = ['D:\remote_dataset\processed_data\user_',num2str(i),'\exp',num2str(j),'\convert_frame_normalized\left_eye\left_rv_gazenorm_list.mat']
        load (rotate_mat);
        
        
        
        hr_mat = ['D:\remote_dataset\processed_data\user_',num2str(i),'\exp',num2str(j),'\convert_frame_normalized\right_eye\right_hr_norm_list.mat']
        load (hr_mat);
        scale_mat = ['D:\remote_dataset\processed_data\user_',num2str(i),'\exp',num2str(j),'\convert_frame_normalized\right_eye\right_scale_vector_list.mat']
        load (scale_mat);
        rotate_mat = ['D:\remote_dataset\processed_data\user_',num2str(i),'\exp',num2str(j),'\convert_frame_normalized\right_eye\right_rv_gazenorm_list.mat']
        load (rotate_mat);
        
        
%       right_scale_vector_list = round(right_scale_vector_list, 8);
        
        
        for k =1:length(left_names_file)
%             rounded_number = floor((k-1)/20)+1;
            rounded_number = floor((k-1)/19)+1;
            
            [sort_label_value, sort_label_ind] = min((abs(Gazetimestamp(:) - (time_frame(k))) * 1000)); %查找距离当前帧最近的tobii gt
            sort_label_value_list (k) = sort_label_value;
            sort_label_ind_list (k) = sort_label_ind;
            % movefile(['D:\remote_dataset\user_',num2str(i),'\exp',num2str(j),'\prophesee\y.txt'],['D:\remote_dataset\raw_filr\user_',num2str(i),'\exp',num2str(j),'\prophesee\y.txt']);
            
%             left_W = squeeze(left_scale_vector_list(rounded_number,:,:))*squeeze(left_rv_gazenorm_list(rounded_number,:,:));
            
%             right_W = squeeze(right_scale_vector_list(rounded_number,:,:))*squeeze(right_rv_gazenorm_list(rounded_number,:,:));
             left_W = squeeze(left_rv_gazenorm_list(rounded_number,:,:));
           
             right_W =squeeze(right_rv_gazenorm_list(rounded_number,:,:));
             
             
            T_headpose_r_left =   rotation_vector2matrix(  left_hr_norm_list(rounded_number,:));  % 计算归一化头部姿态;  % Rn = MRr,
            T_headpose_r_right =  rotation_vector2matrix(  right_hr_norm_list(rounded_number,:));
            
            T_headpose_r_left_vec =  T_headpose_r_left(:, 3);
            T_headpose_r_right_vec  = T_headpose_r_right(:, 3);
            
            T_headpose_r_left_pitch_yaw = HeadTo2d( T_headpose_r_left_vec); %  hn can be represented as a two-dimensiona rotation vector (horizontal and vertical orientations)
            T_headpose_r_right_pitch_yaw = HeadTo2d( T_headpose_r_right_vec); 
            
            
            headpose_left_pitch_yaw_list(k, :) = T_headpose_r_left_pitch_yaw;
            headpose_right_pitch_yaw_list(k, :)  = T_headpose_r_right_pitch_yaw;
            
            
            
            left_gazenorm_sample = left_W* LGAZE(:,sort_label_ind);
            right_gazenorm_sample = right_W* RGAZE(:,sort_label_ind);
           %  left_gazenorm_sample =  LGAZE(:,sort_label_ind);
            % right_gazenorm_sample =  RGAZE(:,sort_label_ind);
            
            
            
            
            R_pitch_yaw = vector_to_pitchyaw(right_gazenorm_sample');
            R_pitch_yaw_angle  =  R_pitch_yaw;  %R_pitch_yaw_angle  = 180* R_pitch_yaw/pi;
            L_pitch_yaw = vector_to_pitchyaw(left_gazenorm_sample');
            L_pitch_yaw_angle  =  L_pitch_yaw;
            L_pitch_yaw_angle_list(k, :) = L_pitch_yaw_angle;
            R_pitch_yaw_angle_list(k, :) = R_pitch_yaw_angle;
            
            klist(k) =  k;
            
        end
%         sort_label_value_list = sort_label_value_list(10:20:end);
%         klist = klist (10:20:end);
%         L_pitch_yaw_angle_list = L_pitch_yaw_angle_list(10:20:end,:);
%         R_pitch_yaw_angle_list =  R_pitch_yaw_angle_list(10:20:end,:);

        del_index = find( sort_label_value_list>=1);
        klist(del_index) = [];
        klist = klist';
        %         plot( sort_label_ind_list)
        %         plot( sort_label_value_list)
        %         plot(klist)
        
        L_pitch_yaw_angle_list(del_index,:) = [];
        R_pitch_yaw_angle_list(del_index,:) = [];
        headpose_left_pitch_yaw_list(del_index,:) = [];
        headpose_right_pitch_yaw_list(del_index,:) = [];
        
%         hdf5write (['C:\Users\Aiot_Server\Desktop\remote_dataset\data_for_voxel_grid_smooth\','L_pitch_yaw_user_',num2str(i),'_exp',num2str(j),'.h5'],'/data', L_pitch_yaw_angle_list);
%         hdf5write (['C:\Users\Aiot_Server\Desktop\remote_dataset\data_for_voxel_grid_smooth\','R_pitch_yaw_user_',num2str(i),'_exp',num2str(j),'.h5'],'/data', R_pitch_yaw_angle_list);
   

%         hdf5write (['C:\Users\Aiot_Server\Desktop\remote_dataset\data_for_voxel_grid_smooth\','no_norm_L_pitch_yaw_user_',num2str(i),'_exp',num2str(j),'.h5'],'/data', L_pitch_yaw_angle_list);
%         hdf5write (['C:\Users\Aiot_Server\Desktop\remote_dataset\data_for_voxel_grid_smooth\','no_norm_R_pitch_yaw_user_',num2str(i),'_exp',num2str(j),'.h5'],'/data', R_pitch_yaw_angle_list);
   

        L_pitch_yaw_angle_list_total =  vertcat(L_pitch_yaw_angle_list_total, L_pitch_yaw_angle_list);
        R_pitch_yaw_angle_list_total =  vertcat(R_pitch_yaw_angle_list_total, R_pitch_yaw_angle_list);

%         hdf5write (['C:\Users\Aiot_Server\Desktop\remote_dataset\data_for_voxel_grid_smooth\','headpose_L_pitch_yaw_user_',num2str(i),'_exp',num2str(j),'.h5'],'/data', headpose_left_pitch_yaw_list);
%         hdf5write (['C:\Users\Aiot_Server\Desktop\remote_dataset\data_for_voxel_grid_smooth\','headpose_R_pitch_yaw_user_',num2str(i),'_exp',num2str(j),'.h5'],'/data', headpose_right_pitch_yaw_list);
%         
        
%         
%         parfor u = 1:length(klist)
% %             copyfile(['C:\Users\Aiot_Server\Desktop\remote_dataset\voxel_grid_representation\user_',num2str(i),'\exp',num2str(j),'\voxel_grid_saved\left_eye\', left_names_file{klist(u)}],['C:\Users\Aiot_Server\Desktop\remote_dataset\voxel_grid_representation\user_',num2str(i),'\exp',num2str(j),'\voxel_grid_for_eye_tracker\left_eye\', left_names_file{klist(u)}])
%             copyfile(['C:\Users\Aiot_Server\Desktop\remote_dataset\voxel_grid_representation_smooth\user_',num2str(i),'\exp',num2str(j),'\voxel_grid_saved\left_eye\', left_names_npz_file{klist(u)}],['C:\Users\Aiot_Server\Desktop\remote_dataset\voxel_grid_representation_smooth\user_',num2str(i),'\exp',num2str(j),'\voxel_grid_for_eye_tracker\left_eye\', left_names_npz_file{klist(u)}])
% %             copyfile(['C:\Users\Aiot_Server\Desktop\remote_dataset\voxel_grid_representation\user_',num2str(i),'\exp',num2str(j),'\voxel_grid_saved\right_eye\', right_names_file{klist(u)}],['C:\Users\Aiot_Server\Desktop\remote_dataset\voxel_grid_representation\user_',num2str(i),'\exp',num2str(j),'\voxel_grid_for_eye_tracker\right_eye\', right_names_file{klist(u)}])
%             copyfile(['C:\Users\Aiot_Server\Desktop\remote_dataset\voxel_grid_representation_smooth\user_',num2str(i),'\exp',num2str(j),'\voxel_grid_saved\right_eye\', right_names_npz_file{klist(u)}],['C:\Users\Aiot_Server\Desktop\remote_dataset\voxel_grid_representation_smooth\user_',num2str(i),'\exp',num2str(j),'\voxel_grid_for_eye_tracker\right_eye\', right_names_npz_file{klist(u)}])
%         end
      
 
%        parfor v = 1:length(klist)
%         right_eye_normalized = imread(['C:\Users\Aiot_Server\Desktop\remote_dataset\voxel_grid_representation_smooth\user_',num2str(i),'\exp',num2str(j),'\voxel_grid_saved\right_eye\', right_names_file{klist(v)}]);
%         right_eye_normalized  = imresize(right_eye_normalized , [64, 96]);
% %         imshow( right_eye_normalized )
% %         right_eye_normalized  =  rgb2gray(right_eye_normalized);  % 
%         right_eye_normalized_list(v, :, :) = right_eye_normalized;
%         
%         left_eye_normalized = imread(['C:\Users\Aiot_Server\Desktop\remote_dataset\voxel_grid_representation_smooth\user_',num2str(i),'\exp',num2str(j),'\voxel_grid_saved\left_eye\', left_names_file{klist(v)}]);
%         left_eye_normalized  = imresize(left_eye_normalized , [64, 96]);
% %         left_eye_normalized  =  rgb2gray(left_eye_normalized);  % 
%         left_eye_normalized_list(v, :, :) = left_eye_normalized;
%         
%        end
        
       
       
       
%         hdf5write (['C:\Users\Aiot_Server\Desktop\remote_dataset\data_for_voxel_grid_smooth\','64_96_right_eye_normalized_user_',num2str(i),'_exp',num2str(j),'.h5'],'/data', right_eye_normalized_list);
%         hdf5write (['C:\Users\Aiot_Server\Desktop\remote_dataset\data_for_voxel_grid_smooth\','64_96_left_eye_normalized_user_',num2str(i),'_exp',num2str(j),'.h5'],'/data', left_eye_normalized_list);

        
        
        clearvars -except i j   L_pitch_yaw_angle_list_total    R_pitch_yaw_angle_list_total 
    end
    
    
end