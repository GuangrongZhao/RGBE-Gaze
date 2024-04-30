clc
clear all
close all


for i = 1:66
    
    for j =1:6
        load eyetracker2world_parameters
      
        path_folder = ['G:\remote_apperance_gaze_dataset\raw_data\user_',num2str(i),'\exp',num2str(j)];
        
        left_gazenorm = load (fullfile(['G:\remote_apperance_gaze_dataset\processed_data\random_data_for_frame_method_eva\user_',num2str(i),'\exp',num2str(j), '\convert_frame_normalized\left_eye\left_rv_gazenorm_list.mat'])).left_rv_gazenorm_list;
        right_gazenorm = load (fullfile(['G:\remote_apperance_gaze_dataset\processed_data\random_data_for_frame_method_eva\user_',num2str(i),'\exp',num2str(j)', '\convert_frame_normalized\right_eye\right_rv_gazenorm_list.mat'])).right_rv_gazenorm_list;
        
        left_hr_vector = load (fullfile(['G:\remote_apperance_gaze_dataset\processed_data\random_data_for_frame_method_eva\user_',num2str(i),'\exp',num2str(j), '\convert_frame_normalized\left_eye\left_hr_norm_list.mat'])).left_hr_norm_list;
        right_hr_vector = load (fullfile(['G:\remote_apperance_gaze_dataset\processed_data\random_data_for_frame_method_eva\user_',num2str(i),'\exp',num2str(j), '\convert_frame_normalized\right_eye\right_hr_norm_list.mat'])).right_hr_norm_list;
        
        [flir_began] = textread(fullfile(path_folder, '\convert2eventspace\','timestamp_win.txt'), '%f');   % event 的第一个时间戳
        flir_began = flir_began(2);
        
        [gazepoint_began ] = textread(fullfile(path_folder, '\gazepoint\','time_win.txt'), '%f');   % event 的第一个时间戳
        gazepoint_began = gazepoint_began(2);
        
        timedifference = (flir_began - gazepoint_began);
        
        totaldata  = csvread(fullfile(path_folder,'\gazepoint\gazepoint.csv'));
        
        [Gazetimestamp, LGAZE, RGAZE] = wcs_3d_gaze(totaldata,R,t);
        
        LGAZE = - LGAZE;
        RGAZE = - RGAZE;
        
        Gazetimestamp = Gazetimestamp - timedifference;
        
        processed_path_folder = ['G:\remote_apperance_gaze_dataset\processed_data\random_data_for_frame_method_eva\user_',num2str(i),'\exp',num2str(j)];
        %%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%% read frame&event %%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%
        flir_folder = dir(fullfile(processed_path_folder,'\convert_frame_normalized\face\','*.jpg*'));
        names_file = sort_nat({flir_folder.name}); %每一个aps 图片的名字
        [time_frame] = textread(fullfile(path_folder, '\convert2eventspace\','timestamp.txt'), '%f');   % event 的第一个时间戳
        time_frame = time_frame/(1e+9);
        time_frame = time_frame - time_frame(1) ;
        sort_label_value_list = [];
        
        indexArray = textread(['G:\remote_apperance_gaze_dataset\frame_eva_random_index\user_',num2str(i),'_exp_',num2str(j),'.txt'], '%f');   % event 的第一个时间戳
        
        parfor sk =1:length(indexArray)
            k =  indexArray (sk)+1;
            left_eye_normalized = (imread(cell2mat(fullfile(['G:\remote_apperance_gaze_dataset\processed_data\random_data_for_frame_method_eva\user_',num2str(i),'\exp',num2str(j),'\convert_frame_normalized\left_eye\', names_file(sk)]))));
            right_eye_normalized = (imread(cell2mat(fullfile(['G:\remote_apperance_gaze_dataset\processed_data\random_data_for_frame_method_eva\user_',num2str(i),'\exp',num2str(j),'\convert_frame_normalized\right_eye\', names_file(sk)]))));
            %
            left_eye_normalized  = imresize(left_eye_normalized , [64,96]);
            right_eye_normalized  = imresize(right_eye_normalized , [64,96]);
%             left_eye_normalized  = imresize(left_eye_normalized , [36,60]);
%             right_eye_normalized  = imresize(right_eye_normalized , [36,60]);
            left_eye_normalized  =  rgb2gray(left_eye_normalized);
            right_eye_normalized =  rgb2gray(right_eye_normalized);
            
            [sort_label_value, sort_label_ind] = min((abs(Gazetimestamp(:) - (time_frame(k))) * 1000)); %查找距离当前帧最近的tobii gt
            sort_label_value_list (sk) = sort_label_value;
            
            T_left_gazenorm = squeeze(left_gazenorm(sk,:,:));
            T_right_gazenorm = squeeze(right_gazenorm(sk,:,:));
            
            T_headpose_r_left =   rotation_vector2matrix(left_hr_vector(sk,:));  % 计算归一化头部姿态;  % Rn = MRr,
            T_headpose_r_right =  rotation_vector2matrix(right_hr_vector(sk,:));
            
            T_headpose_r_left_vec =  T_headpose_r_left(:, 3);
            T_headpose_r_right_vec  = T_headpose_r_right(:, 3);
            
            T_headpose_r_left_pitch_yaw = HeadTo2d( T_headpose_r_left_vec); %  hn can be represented as a two-dimensiona rotation vector (horizontal and vertical orientations)
            T_headpose_r_right_pitch_yaw = HeadTo2d( T_headpose_r_right_vec);
            
            
            headpose_left_pitch_yaw_list(sk, :) = T_headpose_r_left_pitch_yaw;
            headpose_right_pitch_yaw_list(sk, :)  = T_headpose_r_right_pitch_yaw;
            
            
            left_gazenorm_sample = T_left_gazenorm* LGAZE(:,sort_label_ind); %进行gazenorm，使用w2e先将gaze 从世界坐标系转到相机坐标系
            right_gazenorm_sample = T_right_gazenorm*  RGAZE(:,sort_label_ind);
            
            
            R_pitch_yaw = vector_to_pitchyaw(right_gazenorm_sample');
            R_pitch_yaw_angle  =  R_pitch_yaw;
            L_pitch_yaw = vector_to_pitchyaw(left_gazenorm_sample');
            L_pitch_yaw_angle  =  L_pitch_yaw;
            
            
            L_pitch_yaw_angle_list(sk, :) = L_pitch_yaw_angle;
            R_pitch_yaw_angle_list(sk, :) = R_pitch_yaw_angle;
            
            left_eye_normalized_list(sk, :, :) = left_eye_normalized ;
            right_eye_normalized_list(sk, :, :) = right_eye_normalized ;
            
            
        end
        del_index = find( sort_label_value_list  >=4);
        L_pitch_yaw_angle_list(del_index , :) = [];
        R_pitch_yaw_angle_list(del_index , :) = [];
        
        left_eye_normalized_list(del_index , :, :) = [];
        right_eye_normalized_list(del_index , :, :) = [];
        
        headpose_left_pitch_yaw_list(del_index , :, :) = [];
        headpose_right_pitch_yaw_list(del_index , :, :) = [];
        
        hdf5write (['G:\remote_apperance_gaze_dataset\processed_data\data_network_training_for_frame_method_eva\','64_96_L_eye_normalized_user_',num2str(i),'_exp',num2str(j),'.h5'],'/data', left_eye_normalized_list);
        hdf5write (['G:\remote_apperance_gaze_dataset\processed_data\data_network_training_for_frame_method_eva\','64_96_R_eye_normalized_user_',num2str(i),'_exp',num2str(j),'.h5'],'/data', right_eye_normalized_list);
        hdf5write (['G:\remote_apperance_gaze_dataset\processed_data\data_network_training_for_frame_method_eva\','L_headpose_user_',num2str(i),'_exp',num2str(j),'.h5'],'/data', headpose_left_pitch_yaw_list);
        hdf5write (['G:\remote_apperance_gaze_dataset\processed_data\data_network_training_for_frame_method_eva\','R_headpose_user_',num2str(i),'_exp',num2str(j),'.h5'],'/data', headpose_right_pitch_yaw_list);
        hdf5write (['G:\remote_apperance_gaze_dataset\processed_data\data_network_training_for_frame_method_eva\','L_pitch_yaw_user_',num2str(i),'_exp',num2str(j),'.h5'],'/data', L_pitch_yaw_angle_list);
        hdf5write (['G:\remote_apperance_gaze_dataset\processed_data\data_network_training_for_frame_method_eva\','R_pitch_yaw_user_',num2str(i),'_exp',num2str(j),'.h5'],'/data', R_pitch_yaw_angle_list);
        
        clearvars -except i j
    end
end