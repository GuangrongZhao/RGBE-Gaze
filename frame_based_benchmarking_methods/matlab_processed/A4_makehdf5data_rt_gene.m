clc
clear all
close all

for i = 1:66
    for j =1:6
        load eyetracker2world_parameters
       
        path_folder = ['G:\remote_apperance_gaze_dataset\raw_data\user_',num2str(i),'\exp',num2str(j)];
        
        left_gazenorm = load (fullfile(['G:\remote_apperance_gaze_dataset\processed_data\random_data_for_frame_method_eva\user_',num2str(i),'\exp',num2str(j), '\convert_frame_normalized\left_eye\left_rv_gazenorm_list.mat'])).left_rv_gazenorm_list;
        right_gazenorm = load (fullfile(['G:\remote_apperance_gaze_dataset\processed_data\random_data_for_frame_method_eva\user_',num2str(i),'\exp',num2str(j)', '\convert_frame_normalized\right_eye\right_rv_gazenorm_list.mat'])).right_rv_gazenorm_list;
   
        face_gazenorm = load (fullfile(['G:\remote_apperance_gaze_dataset\processed_data\random_data_for_frame_method_eva\user_',num2str(i),'\exp',num2str(j)', '\convert_frame_normalized\face\rv_gazenorm_list.mat'])).rv_gazenorm_list;
        hr_gazenorm = load (fullfile(['G:\remote_apperance_gaze_dataset\processed_data\random_data_for_frame_method_eva\user_',num2str(i),'\exp',num2str(j)', '\convert_frame_normalized\face\hr_norm_list.mat'])).hr_norm_list;
        landmark_gazenorm = load (fullfile(['G:\remote_apperance_gaze_dataset\processed_data\random_data_for_frame_method_eva\user_',num2str(i),'\exp',num2str(j)', '\convert_frame_normalized\face\total_landmarks_normalized_list.mat'])).total_landmarks_normalized_list;
        [flir_began] = textread(fullfile(path_folder, '\convert2eventspace\','timestamp_win.txt'), '%f');   % event 的第一个时间戳
        flir_began = flir_began(2);
        
        [gazepoint_began ] = textread(fullfile(path_folder, '\gazepoint\','time_win.txt'), '%f');   % event 的第一个时间戳
        gazepoint_began = gazepoint_began(2);
        
        timedifference = (flir_began - gazepoint_began);

        totaldata  = csvread(fullfile(path_folder,'\gazepoint\gazepoint.csv'));
        
        [Gazetimestamp, Combine_left_right_GAZE] = combine_wcs_3d_gaze(totaldata,R,t);
        Combine_left_right_GAZE = - Combine_left_right_GAZE;
        Gazetimestamp = Gazetimestamp - timedifference; % 可能最好是减
        
        %%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%% read frame&event %%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%
        processed_path_folder = ['G:\remote_apperance_gaze_dataset\processed_data\random_data_for_frame_method_eva\user_',num2str(i),'\exp',num2str(j)];
        %%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%% read frame&event %%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%
        flir_folder = dir(fullfile(processed_path_folder,'\convert_frame_normalized\face\','*.jpg*'));
        names_file = sort_nat({flir_folder.name}); %每一个aps 图片的名字
        

        [time_frame] = textread(fullfile(path_folder, '\convert2eventspace\','timestamp.txt'), '%f');   % event 的第一个时间戳
        time_frame = time_frame/(1e+9);
        time_frame = time_frame - time_frame(1) ;
        sort_label_value_list = [];
        
        indexArray = textread(['G:\remote_apperance_gaze_dataset\frame_eva_random_index\user_',num2str(i),'_exp_',num2str(j),'.txt'], '%f');   % event 的第一个时间戳
        
       for sk =1:length(indexArray )
            k =  indexArray (sk)+1;
            
            face_normalized = (imread(cell2mat(fullfile(['G:\remote_apperance_gaze_dataset\processed_data\random_data_for_frame_method_eva\user_',num2str(i),'\exp',num2str(j),'\convert_frame_normalized','\face\',  names_file(sk)]))));
            
            face_normalized =  rgb2gray(face_normalized);
            
            [sort_label_value, sort_label_ind] = min((abs(Gazetimestamp(:) - (time_frame(k))) * 1000)); %查找距离当前帧最近的tobii gt
            sort_label_value_list (sk) = sort_label_value;
            
            Combine_gazenorm = (face_gazenorm(sk,:,:));

            Combine_gazenorm_sample = squeeze(Combine_gazenorm)*Combine_left_right_GAZE(:,sort_label_ind);
            
            Combine_pitch_yaw = vector_to_pitchyaw( Combine_gazenorm_sample');
            Combine_pitch_yaw_angle  =   Combine_pitch_yaw;
            
            Combine_pitch_yaw_angle_list(sk, :) = Combine_pitch_yaw_angle ;
              
            landmark_gazenorm_temp = landmark_gazenorm(sk,:,:);
            
            left_eye_normalized =  face_normalized((((landmark_gazenorm_temp(:,37,2)+landmark_gazenorm_temp(:,40,2))/2)-35):(((landmark_gazenorm_temp(:,37,2)+landmark_gazenorm_temp(:,40,2))/2)+19),(1:80));
            right_eye_normalized =  face_normalized((((landmark_gazenorm_temp(:,43,2)+landmark_gazenorm_temp(:,46,2))/2)-35):(((landmark_gazenorm_temp(:,43,2)+landmark_gazenorm_temp(:,46,2))/2)+19),(224-79:224));
            left_eye_normalized  = imresize(left_eye_normalized , [36, 60]);
            right_eye_normalized  = imresize(right_eye_normalized , [36, 60]);
            
            left_eye_normalized_list(sk, :, :) = left_eye_normalized ;
            right_eye_normalized_list(sk, :, :) = right_eye_normalized;
            
            tess =  hr_gazenorm (sk,:);
            headmatrix = rotation_vector2matrix(tess);
            vec =  headmatrix(:, 3);
            twodhead_list (sk,:) = HeadTo2d(vec);
            
            
        end
        del_index = find( sort_label_value_list  >=4);
        Combine_pitch_yaw_angle_list(del_index , :) = [];
        left_eye_normalized_list(del_index , :, :) = [];
        right_eye_normalized_list(del_index , :, :) = [];
        twodhead_list(del_index , :) = [];
        
        
        hdf5write (['G:\remote_apperance_gaze_dataset\processed_data\data_network_training_for_frame_method_eva\','two_branch_left_eye_normalized_user_',num2str(i),'_exp',num2str(j),'.h5'],'/data', left_eye_normalized_list);
        hdf5write (['G:\remote_apperance_gaze_dataset\processed_data\data_network_training_for_frame_method_eva\','two_branch_right_eye_normalized_user_',num2str(i),'_exp',num2str(j),'.h5'],'/data', right_eye_normalized_list);
        hdf5write (['G:\remote_apperance_gaze_dataset\processed_data\data_network_training_for_frame_method_eva\','face_norm_headpose_user_',num2str(i),'_exp',num2str(j),'.h5'],'/data', twodhead_list);
        hdf5write (['G:\remote_apperance_gaze_dataset\processed_data\data_network_training_for_frame_method_eva\','combine_pitch_yaw_user_',num2str(i),'_exp',num2str(j),'.h5'],'/data',  Combine_pitch_yaw_angle_list);
        
        clearvars -except i j
    end
    
end