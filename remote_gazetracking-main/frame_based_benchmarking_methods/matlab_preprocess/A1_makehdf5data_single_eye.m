clc
clear all
close all

base_path = 'G:\remote_apperance_gaze_dataset\';

for i = 1:66
    for j = 1:6
        load eyetracker2world_parameters
      
        path_folder = fullfile(base_path, ['raw_data\user_', num2str(i), '\exp', num2str(j)]);
        
        left_gaze_norm_path = fullfile(base_path, ['processed_data\random_data_for_event_method_eva\frame\user_', num2str(i), '\exp', num2str(j), '\convert_frame_normalized\left_eye\left_rv_gazenorm_list.mat']);
        right_gaze_norm_path = fullfile(base_path, ['processed_data\random_data_for_event_method_eva\frame\user_', num2str(i), '\exp', num2str(j), '\convert_frame_normalized\right_eye\right_rv_gazenorm_list.mat']);
        left_hr_norm_path = fullfile(base_path, ['processed_data\random_data_for_event_method_eva\frame\user_', num2str(i), '\exp', num2str(j), '\convert_frame_normalized\left_eye\left_hr_norm_list.mat']);
        right_hr_norm_path = fullfile(base_path, ['processed_data\random_data_for_event_method_eva\frame\user_', num2str(i), '\exp', num2str(j), '\convert_frame_normalized\right_eye\right_hr_norm_list.mat']);
        
        left_gazenorm = load(left_gaze_norm_path).left_rv_gazenorm_list;
        right_gazenorm = load(right_gaze_norm_path).right_rv_gazenorm_list;
        left_hr_vector = load(left_hr_norm_path).left_hr_norm_list;
        right_hr_vector = load(right_hr_norm_path).right_hr_norm_list;
        
        flir_began_path = fullfile(path_folder, 'convert2eventspace', 'timestamp_win.txt');
        gazepoint_began_path = fullfile(path_folder, 'gazepoint', 'time_win.txt');
        
        [flir_began] = textread(flir_began_path, '%f');
        flir_began = flir_began(2);
        
        [gazepoint_began] = textread(gazepoint_began_path, '%f');
        gazepoint_began = gazepoint_began(2);
        
        timedifference = flir_began - gazepoint_began;
        
        totaldata = csvread(fullfile(path_folder, 'gazepoint', 'gazepoint.csv'));
        
        [Gazetimestamp, LGAZE, RGAZE] = wcs_3d_gaze(totaldata, R, t);
        
        LGAZE = -LGAZE;
        RGAZE = -RGAZE;
        
        Gazetimestamp = Gazetimestamp - timedifference;
        
        processed_path_folder = fullfile(base_path, ['processed_data\random_data_for_event_method_eva\frame\user_', num2str(i), '\exp', num2str(j)]);
        
        % Read frame & event files
        flir_folder = dir(fullfile(processed_path_folder, '\convert_frame_normalized\face\*.jpg*'));
        names_file = sort_nat({flir_folder.name});
        [time_frame] = textread(fullfile(path_folder, 'convert2eventspace', 'timestamp.txt'), '%f');
        time_frame = time_frame / (1e+9);
        time_frame = time_frame - time_frame(1);
        sort_label_value_list = [];
        
        indexArray = textread(fullfile(base_path, ['event_eva_random_index\user_', num2str(i), '_exp_', num2str(j), '.txt']), '%f');
        
        parfor sk = 1:length(indexArray)
            k = indexArray(sk) + 1;
            left_eye_normalized = imread(fullfile(processed_path_folder, 'convert_frame_normalized\left_eye', names_file{sk}));
            right_eye_normalized = imread(fullfile(processed_path_folder, 'convert_frame_normalized\right_eye', names_file{sk}));
            
            left_eye_normalized = imresize(left_eye_normalized, [64, 96]);
            right_eye_normalized = imresize(right_eye_normalized, [64, 96]);
%             left_eye_normalized = imresize(left_eye_normalized, [36, 60]);
%             right_eye_normalized = imresize(right_eye_normalized, [36, 60]);
            left_eye_normalized = rgb2gray(left_eye_normalized);
            right_eye_normalized = rgb2gray(right_eye_normalized);
            
            [sort_label_value, sort_label_ind] = min(abs(Gazetimestamp(:) - time_frame(k)) * 1000);
            sort_label_value_list(sk) = sort_label_value;
            
            T_left_gazenorm = squeeze(left_gazenorm(sk, :, :));
            T_right_gazenorm = squeeze(right_gazenorm(sk, :, :));
            
            T_headpose_r_left = rotation_vector2matrix(left_hr_vector(sk, :));
            T_headpose_r_right = rotation_vector2matrix(right_hr_vector(sk, :));
            
            T_headpose_r_left_vec = T_headpose_r_left(:, 3);
            T_headpose_r_right_vec = T_headpose_r_right(:, 3);
            
            T_headpose_r_left_pitch_yaw = HeadTo2d(T_headpose_r_left_vec);
            T_headpose_r_right_pitch_yaw = HeadTo2d(T_headpose_r_right_vec);
            
            headpose_left_pitch_yaw_list(sk, :) = T_headpose_r_left_pitch_yaw;
            headpose_right_pitch_yaw_list(sk, :) = T_headpose_r_right_pitch_yaw;
            
            left_gazenorm_sample = T_left_gazenorm * LGAZE(:, sort_label_ind);
            right_gazenorm_sample = T_right_gazenorm * RGAZE(:, sort_label_ind);
            
            R_pitch_yaw = vector_to_pitchyaw(right_gazenorm_sample');
            R_pitch_yaw_angle = R_pitch_yaw;
            L_pitch_yaw = vector_to_pitchyaw(left_gazenorm_sample');
            L_pitch_yaw_angle = L_pitch_yaw;
            
            L_pitch_yaw_angle_list(sk, :) = L_pitch_yaw_angle;
            R_pitch_yaw_angle_list(sk, :) = R_pitch_yaw_angle;
            
            left_eye_normalized_list(sk, :, :) = left_eye_normalized;
            right_eye_normalized_list(sk, :, :) = right_eye_normalized;
        end
        
        del_index = find(sort_label_value_list >= 6);
        L_pitch_yaw_angle_list(del_index, :) = [];
        R_pitch_yaw_angle_list(del_index, :) = [];
        left_eye_normalized_list(del_index, :, :) = [];
        right_eye_normalized_list(del_index, :, :) = [];
        headpose_left_pitch_yaw_list(del_index, :, :) = [];
        headpose_right_pitch_yaw_list(del_index, :, :) = [];
        
        % Save HDF5 files
        hdf5write(fullfile(base_path, ['processed_data\data_network_training_for_frame_method_eva\64_96_L_eye_normalized_user_', num2str(i), '_exp', num2str(j), '.h5']), '/data', left_eye_normalized_list);
        hdf5write(fullfile(base_path, ['processed_data\data_network_training_for_frame_method_eva\64_96_R_eye_normalized_user_', num2str(i), '_exp', num2str(j), '.h5']), '/data', right_eye_normalized_list);
%          hdf5write(fullfile(base_path, ['processed_data\data_network_training_for_frame_method_eva\36_60_L_eye_normalized_user_', num2str(i), '_exp', num2str(j), '.h5']), '/data', left_eye_normalized_list);
%         hdf5write(fullfile(base_path, ['processed_data\data_network_training_for_frame_method_eva\36_60_R_eye_normalized_user_', num2str(i), '_exp', num2str(j), '.h5']), '/data', right_eye_normalized_list);
        hdf5write(fullfile(base_path, ['processed_data\data_network_training_for_frame_method_eva\L_headpose_user_', num2str(i), '_exp', num2str(j), '.h5']), '/data', headpose_left_pitch_yaw_list);
        hdf5write(fullfile(base_path, ['processed_data\data_network_training_for_frame_method_eva\R_headpose_user_', num2str(i), '_exp', num2str(j), '.h5']), '/data', headpose_right_pitch_yaw_list);
        hdf5write(fullfile(base_path, ['processed_data\data_network_training_for_frame_method_eva\L_pitch_yaw_user_', num2str(i), '_exp', num2str(j), '.h5']), '/data', L_pitch_yaw_angle_list);
        hdf5write(fullfile(base_path, ['processed_data\data_network_training_for_frame_method_eva\R_pitch_yaw_user_', num2str(i), '_exp', num2str(j), '.h5']), '/data', R_pitch_yaw_angle_list);
        
        clearvars -except base_path i j
    end
end
