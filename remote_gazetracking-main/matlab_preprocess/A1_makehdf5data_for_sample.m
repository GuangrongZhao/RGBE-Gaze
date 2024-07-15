clc
clear all
close all

for i = 1:1
    % Define the paths outside the inner loop
    base_raw_data_path = 'G:\remote_apperance_gaze_dataset\raw_data\user_';
    base_processed_data_path = 'G:\remote_apperance_gaze_dataset\processed_data\random_data_for_event_method_eva\';
    base_network_training_path = 'G:\remote_apperance_gaze_dataset\processed_data\data_network_training_for_event_method_eva\';

    for j = 1:1
        % Load the eye tracker to world coordinate transformation relationship
        load eyetracker2world_parameters  

        % Define the folder paths
        path_folder = [base_raw_data_path, num2str(i), '\exp', num2str(j)];
        frame_align_event_path = [base_processed_data_path, 'frame_align_event_timestamp\user_', num2str(i), '_exp_', num2str(j), '.txt'];
        voxel_grid_path = [base_processed_data_path, 'voxel_grid_representation_2.5k\user_', num2str(i), '\exp', num2str(j), '\voxel_grid_saved\'];
        voxel_grid_path_eye_tracker = [base_processed_data_path, 'voxel_grid_representation_2.5k\user_', num2str(i), '\exp', num2str(j), '\voxel_grid_for_eye_tracker\'];
        % Load timestamps
        [flir_began] = textread(fullfile(path_folder, '\convert2eventspace\', 'timestamp_win.txt'), '%f');
        flir_began = flir_began(2);
        [gazepoint_began] = textread(fullfile(path_folder, '\gazepoint\', 'time_win.txt'), '%f');
        gazepoint_began = gazepoint_began(2);
        timedifference = (flir_began - gazepoint_began);

        % Read gaze data
        totaldata = csvread(fullfile(path_folder, '\gazepoint\gazepoint.csv'));
        [Gazetimestamp, LGAZE, RGAZE] = wcs_3d_gaze(totaldata, R, t);

        % Convert to the checkerboard world coordinate system
        LGAZE = -LGAZE;
        RGAZE = -RGAZE;
        Gazetimestamp = Gazetimestamp - timedifference;

        % Load frame alignment event timestamp
        [frame_align_event_time] = textread(frame_align_event_path, '%f');
        voxel_grid_timestamp = textread([voxel_grid_path, 'inter_timestamp.txt'], '%f');

        % List of left eye images and npz files
        left_flir_folder = dir([voxel_grid_path, 'left_eye\', '*.jpg*']);
        left_names_file = sort_nat({left_flir_folder.name});
        left_npz_folder = dir([voxel_grid_path, 'left_eye\', '*.npz']);
        left_names_npz_file = sort_nat({left_npz_folder.name});

        % List of right eye images and npz files
        right_flir_folder = dir([voxel_grid_path, 'right_eye\', '*.jpg*']);
        right_names_file = sort_nat({right_flir_folder.name});
        right_npz_folder = dir([voxel_grid_path, 'right_eye\', '*.npz']);
        right_names_npz_file = sort_nat({right_npz_folder.name});

        % Compute the time frame
        time_frame = (voxel_grid_timestamp - frame_align_event_time(1)) / (1e+6);

        % Initialize lists
        sort_label_value_list = [];
        sort_label_ind_list = [];
        klist = [];
        L_pitch_yaw_angle_list = [];
        R_pitch_yaw_angle_list = [];

        % Load head rotation and gaze normalization data for left eye
        hr_mat_left = [base_processed_data_path, 'frame\user_', num2str(i), '\exp', num2str(j), '\convert_frame_normalized\left_eye\left_hr_norm_list.mat'];
        load(hr_mat_left);
        rotate_mat_left = [base_processed_data_path, 'frame\user_', num2str(i), '\exp', num2str(j), '\convert_frame_normalized\left_eye\left_rv_gazenorm_list.mat'];
        load(rotate_mat_left);

        % Load head rotation and gaze normalization data for right eye
        hr_mat_right = [base_processed_data_path, 'frame\user_', num2str(i), '\exp', num2str(j), '\convert_frame_normalized\right_eye\right_hr_norm_list.mat'];
        load(hr_mat_right);
        rotate_mat_right = [base_processed_data_path, 'frame\user_', num2str(i), '\exp', num2str(j), '\convert_frame_normalized\right_eye\right_rv_gazenorm_list.mat'];
        load(rotate_mat_right);

        for k = 1:length(left_names_file)
            rounded_number = floor((k-1)/50) + 1;
            [sort_label_value, sort_label_ind] = min(abs(Gazetimestamp(:) - time_frame(k)) * 1000);
            sort_label_value_list(k) = sort_label_value;
            sort_label_ind_list(k) = sort_label_ind;

            left_W = squeeze(left_rv_gazenorm_list(rounded_number, :, :));
            right_W = squeeze(right_rv_gazenorm_list(rounded_number, :, :));

            T_headpose_r_left = rotation_vector2matrix(left_hr_norm_list(rounded_number, :));
            T_headpose_r_right = rotation_vector2matrix(right_hr_norm_list(rounded_number, :));

            T_headpose_r_left_vec = T_headpose_r_left(:, 3);
            T_headpose_r_right_vec = T_headpose_r_right(:, 3);

            T_headpose_r_left_pitch_yaw = HeadTo2d(T_headpose_r_left_vec);
            T_headpose_r_right_pitch_yaw = HeadTo2d(T_headpose_r_right_vec);

            headpose_left_pitch_yaw_list(k, :) = T_headpose_r_left_pitch_yaw;
            headpose_right_pitch_yaw_list(k, :) = T_headpose_r_right_pitch_yaw;

            left_gazenorm_sample = left_W * LGAZE(:, sort_label_ind);
            right_gazenorm_sample = right_W * RGAZE(:, sort_label_ind);

            R_pitch_yaw = vector_to_pitchyaw(right_gazenorm_sample');
            R_pitch_yaw_angle = R_pitch_yaw;
            L_pitch_yaw = vector_to_pitchyaw(left_gazenorm_sample');
            L_pitch_yaw_angle = L_pitch_yaw;
            L_pitch_yaw_angle_list(k, :) = L_pitch_yaw_angle;
            R_pitch_yaw_angle_list(k, :) = R_pitch_yaw_angle;

            klist(k) = k;
        end

        % Remove indices with large errors
        del_index = find(sort_label_value_list >= 0.4);
        klist(del_index) = [];
        klist = klist';

        L_pitch_yaw_angle_list(del_index, :) = [];
        R_pitch_yaw_angle_list(del_index, :) = [];
        headpose_left_pitch_yaw_list(del_index, :) = [];
        headpose_right_pitch_yaw_list(del_index, :) = [];

        % Save data
        hdf5write([base_network_training_path, 'L_pitch_yaw_user_', num2str(i), '_exp', num2str(j), '.h5'], '/data', L_pitch_yaw_angle_list);
        hdf5write([base_network_training_path, 'R_pitch_yaw_user_', num2str(i), '_exp', num2str(j), '.h5'], '/data', R_pitch_yaw_angle_list);
        hdf5write([base_network_training_path, 'headpose_L_pitch_yaw_user_', num2str(i), '_exp', num2str(j), '.h5'], '/data', headpose_left_pitch_yaw_list);
        hdf5write([base_network_training_path, 'headpose_R_pitch_yaw_user_', num2str(i), '_exp', num2str(j), '.h5'], '/data', headpose_right_pitch_yaw_list);

        % Copy voxel grid files
        parfor u = 1:length(klist)
            copyfile([voxel_grid_path, 'left_eye\', left_names_npz_file{klist(u)}], [  voxel_grid_path_eye_tracker, 'left_eye\', left_names_npz_file{klist(u)}]);
            copyfile([voxel_grid_path, 'right_eye\', right_names_npz_file{klist(u)}], [  voxel_grid_path_eye_tracker, 'right_eye\', right_names_npz_file{klist(u)}]);
        end

        % Normalize and save frame images
        parfor v = 1:length(klist)
            right_eye_normalized = imread([voxel_grid_path, 'right_eye\', right_names_file{klist(v)}]);
            right_eye_normalized = imresize(right_eye_normalized, [64, 96]);
            right_eye_normalized_list(v, :, :) = right_eye_normalized;

            left_eye_normalized = imread([voxel_grid_path, 'left_eye\', left_names_file{klist(v)}]);
            left_eye_normalized = imresize(left_eye_normalized, [64, 96]);
            left_eye_normalized_list(v, :, :) = left_eye_normalized;
        end

        hdf5write([base_network_training_path, '64_96_right_eye_normalized_user_', num2str(i), '_exp', num2str(j), '.h5'], '/data', right_eye_normalized_list);
        hdf5write([base_network_training_path, '64_96_left_eye_normalized_user_', num2str(i), '_exp', num2str(j), '.h5'], '/data', left_eye_normalized_list);

        clearvars -except i j base_raw_data_path base_processed_data_path base_network_training_path
    end
end
