clc
clear all
close all

base_path = 'G:\remote_apperance_gaze_dataset\';

for i = 1:66
    for j = 1:6
        load eyetracker2world_parameters
        
        % Define paths
        raw_data_folder = fullfile(base_path, 'raw_data', ['user_', num2str(i)], ['exp', num2str(j)]);
        processed_data_folder = fullfile(base_path, 'processed_data', 'random_data_for_event_method_eva', 'frame', ['user_', num2str(i)], ['exp', num2str(j)]);
        event_index_file = fullfile(base_path, 'event_eva_random_index', ['user_', num2str(i), '_exp_', num2str(j), '.txt']);
        
        % Read FLIR and gaze point timestamps
        [flir_began] = textread(fullfile(raw_data_folder, 'convert2eventspace', 'timestamp_win.txt'), '%f');
        flir_began = flir_began(2);
        
        [gazepoint_began] = textread(fullfile(raw_data_folder, 'gazepoint', 'time_win.txt'), '%f');
        gazepoint_began = gazepoint_began(2);
        
        timedifference = flir_began - gazepoint_began;
        
        % Read total data and compute gaze timestamps
        totaldata = csvread(fullfile(raw_data_folder, 'gazepoint', 'gazepoint.csv'));
        [Gazetimestamp, LGAZE, RGAZE] = wcs_3d_gaze(totaldata, R, t);
        Gazetimestamp = Gazetimestamp - timedifference;
        
        % Read frame timestamps and names
        [time_frame] = textread(fullfile(raw_data_folder, 'convert2eventspace', 'timestamp.txt'), '%f');
        time_frame = time_frame / (1e+9);
        time_frame = time_frame - time_frame(1);
        
        flir_folder = dir(fullfile(processed_data_folder, 'convert_frame_normalized', 'face', '*.jpg*'));
        names_file = sort_nat({flir_folder.name});
        
        % Read index array
        indexArray = textread(event_index_file, '%f');
        
        % Initialize arrays
        sort_label_value_list = zeros(length(indexArray), 1);
        face_normalized_list = zeros(length(indexArray), 448, 448);
        
        % Process images
        parfor sk = 1:length(indexArray)
            k = indexArray(sk) + 1;
            face_normalized = imread(fullfile(processed_data_folder, 'convert_frame_normalized', 'face', names_file{sk}));
            face_normalized = imresize(face_normalized, [448, 448]);
            face_normalized = rgb2gray(face_normalized);
            
            [sort_label_value, ~] = min(abs(Gazetimestamp(:) - time_frame(k)) * 1000);
            sort_label_value_list(sk) = sort_label_value;
            face_normalized_list(sk, :, :) = face_normalized;
        end
        
        % Remove entries with sort_label_value >= 6
        del_index = find(sort_label_value_list >= 6);
        face_normalized_list(del_index, :, :) = [];
        
        % Save HDF5 file
        hdf5write(fullfile(base_path, 'processed_data', 'data_network_training_for_frame_method_eva', ['448_448_face_normalized_user_', num2str(i), '_exp', num2str(j), '.h5']), '/data', face_normalized_list);
    end
end
