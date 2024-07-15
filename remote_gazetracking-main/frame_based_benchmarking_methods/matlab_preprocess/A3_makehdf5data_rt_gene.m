clc
clear all
close all

base_path = 'G:\remote_apperance_gaze_dataset\';

for i = 1:66
    for j = 1:6
        load eyetracker2world_parameters
        
        % Define paths
        path_folder = fullfile(base_path, 'raw_data', ['user_', num2str(i)], ['exp', num2str(j)]);
        processed_path_folder = fullfile(base_path, 'processed_data', 'random_data_for_event_method_eva', 'frame', ['user_', num2str(i)], ['exp', num2str(j)]);
        event_index_file = fullfile(base_path, 'event_eva_random_index', ['user_', num2str(i), '_exp_', num2str(j), '.txt']);
        
        % Load necessary data
        left_gazenorm = load(fullfile(processed_path_folder, 'convert_frame_normalized', 'left_eye', 'left_rv_gazenorm_list.mat')).left_rv_gazenorm_list;
        right_gazenorm = load(fullfile(processed_path_folder, 'convert_frame_normalized', 'right_eye', 'right_rv_gazenorm_list.mat')).right_rv_gazenorm_list;
        face_gazenorm = load(fullfile(processed_path_folder, 'convert_frame_normalized', 'face', 'rv_gazenorm_list.mat')).rv_gazenorm_list;
        hr_gazenorm = load(fullfile(processed_path_folder, 'convert_frame_normalized', 'face', 'hr_norm_list.mat')).hr_norm_list;
        landmark_gazenorm = load(fullfile(processed_path_folder, 'convert_frame_normalized', 'face', 'total_landmarks_normalized_list.mat')).total_landmarks_normalized_list;
        
        % Read timestamps
        [flir_began] = textread(fullfile(path_folder, 'convert2eventspace', 'timestamp_win.txt'), '%f');
        flir_began = flir_began(2);
        
        [gazepoint_began ] = textread(fullfile(path_folder, 'gazepoint', 'time_win.txt'), '%f');
        gazepoint_began = gazepoint_began(2);
        
        timedifference = flir_began - gazepoint_began;

        totaldata = csvread(fullfile(path_folder, 'gazepoint', 'gazepoint.csv'));
        
        % Combine left and right gaze vectors
        [Gazetimestamp, Combine_left_right_GAZE] = combine_wcs_3d_gaze(totaldata, R, t);
        Combine_left_right_GAZE = -Combine_left_right_GAZE;
        Gazetimestamp = Gazetimestamp - timedifference;
        
        % Read frame timestamps and names
        [time_frame] = textread(fullfile(path_folder, 'convert2eventspace', 'timestamp.txt'), '%f');
        time_frame = time_frame / (1e+9);
        time_frame = time_frame - time_frame(1);
        
        flir_folder = dir(fullfile(processed_path_folder, 'convert_frame_normalized', 'face', '*.jpg*'));
        names_file = sort_nat({flir_folder.name});
        
        % Read index array
        indexArray = textread(event_index_file, '%f');
        
        % Initialize arrays
        sort_label_value_list = zeros(length(indexArray), 1);
        left_eye_normalized_list = zeros(length(indexArray), 36, 60);
        right_eye_normalized_list = zeros(length(indexArray), 36, 60);
        twodhead_list = zeros(length(indexArray), 2);
        Combine_pitch_yaw_angle_list = zeros(length(indexArray), 2);
        
        % Process images
        parfor sk = 1:length(indexArray)
            k = indexArray(sk) + 1;
            
            % Read and preprocess face image
            face_normalized = imread(fullfile(processed_path_folder, 'convert_frame_normalized', 'face', names_file{sk}));
            face_normalized = rgb2gray(face_normalized);
            
            % Find closest gaze timestamp
            [sort_label_value, sort_label_ind] = min(abs(Gazetimestamp(:) - time_frame(k)) * 1000);
            sort_label_value_list(sk) = sort_label_value;
            
            % Extract left and right eye regions
            landmark_gazenorm_temp = landmark_gazenorm(sk,:,:);
            left_eye_normalized = face_normalized((((landmark_gazenorm_temp(:,37,2)+landmark_gazenorm_temp(:,40,2))/2)-35):(((landmark_gazenorm_temp(:,37,2)+landmark_gazenorm_temp(:,40,2))/2)+19),(1:80));
            right_eye_normalized = face_normalized((((landmark_gazenorm_temp(:,43,2)+landmark_gazenorm_temp(:,46,2))/2)-35):(((landmark_gazenorm_temp(:,43,2)+landmark_gazenorm_temp(:,46,2))/2)+19),(224-79:224));
            left_eye_normalized = imresize(left_eye_normalized, [36, 60]);
            right_eye_normalized = imresize(right_eye_normalized, [36, 60]);
            
            % Calculate combined gaze direction
            Combine_gazenorm = face_gazenorm(sk,:,:);
            Combine_gazenorm_sample = squeeze(Combine_gazenorm) * Combine_left_right_GAZE(:,sort_label_ind);
            Combine_pitch_yaw = vector_to_pitchyaw(Combine_gazenorm_sample');
            Combine_pitch_yaw_angle_list(sk, :) = Combine_pitch_yaw;
            
            % Calculate head pose
            tess = hr_gazenorm(sk,:);
            headmatrix = rotation_vector2matrix(tess);
            vec = headmatrix(:, 3);
            twodhead_list(sk,:) = HeadTo2d(vec);
            
            % Store results
            left_eye_normalized_list(sk, :, :) = left_eye_normalized;
            right_eye_normalized_list(sk, :, :) = right_eye_normalized;
        end
        
        % Remove entries with sort_label_value >= 6
        del_index = find(sort_label_value_list >= 6);
        left_eye_normalized_list(del_index, :, :) = [];
        right_eye_normalized_list(del_index, :, :) = [];
        twodhead_list(del_index, :) = [];
        Combine_pitch_yaw_angle_list(del_index, :) = [];
        
        % Save HDF5 files
        hdf5write(fullfile(base_path, 'processed_data', 'data_network_training_for_frame_method_eva', ['two_branch_left_eye_normalized_user_', num2str(i), '_exp', num2str(j), '.h5']), '/data', left_eye_normalized_list);
        hdf5write(fullfile(base_path, 'processed_data', 'data_network_training_for_frame_method_eva', ['two_branch_right_eye_normalized_user_', num2str(i), '_exp', num2str(j), '.h5']), '/data', right_eye_normalized_list);
        hdf5write(fullfile(base_path, 'processed_data', 'data_network_training_for_frame_method_eva', ['face_norm_headpose_user_', num2str(i), '_exp', num2str(j), '.h5']), '/data', twodhead_list);
        hdf5write(fullfile(base_path, 'processed_data', 'data_network_training_for_frame_method_eva', ['combine_pitch_yaw_user_', num2str(i), '_exp', num2str(j), '.h5']), '/data', Combine_pitch_yaw_angle_list);
    end
end
