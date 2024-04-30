clc
clear all
close all


for i = 1:66
    for j =1:6
        load eyetracker2world_parameters
   
        path_folder = ['G:\remote_apperance_gaze_dataset\raw_data\user_',num2str(i),'\exp',num2str(j)];
        [flir_began] = textread(fullfile(path_folder, '\convert2eventspace\','timestamp_win.txt'), '%f');   % event 的第一个时间戳
        flir_began = flir_began(2);
        [gazepoint_began ] = textread(fullfile(path_folder, '\gazepoint\','time_win.txt'), '%f');   % event 的第一个时间戳
        gazepoint_began = gazepoint_began(2);
        timedifference = (flir_began - gazepoint_began);
        
        totaldata  = csvread(fullfile(path_folder,'\gazepoint\gazepoint.csv'));
        
        [Gazetimestamp, LGAZE, RGAZE] = wcs_3d_gaze(totaldata,R,t);

        Gazetimestamp = Gazetimestamp - timedifference; % 可能最好是减
        
        processed_path_folder = ['G:\remote_apperance_gaze_dataset\processed_data\random_data_for_frame_method_eva\user_',num2str(i),'\exp',num2str(j)];
        %%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%% read frame&event %%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%
        flir_folder = dir(fullfile(processed_path_folder,'\convert_frame_normalized\face\','*.jpg*'));
        names_file = sort_nat({flir_folder.name}); %每一个aps 图片的名字
        
        
        [time_frame] = textread(fullfile(path_folder, '\convert2eventspace\','timestamp.txt'), '%f'); 
        time_frame = time_frame/(1e+9);
        time_frame = time_frame - time_frame(1) ;
        sort_label_value_list = [];
        
        indexArray = textread(['G:\remote_apperance_gaze_dataset\frame_eva_random_index\user_',num2str(i),'_exp_',num2str(j),'.txt'], '%f'); 
        
        parfor sk =1:length(indexArray)
            k =  indexArray (sk)+1;
            face_normalized = (imread(cell2mat(fullfile(['G:\remote_apperance_gaze_dataset\processed_data\random_data_for_frame_method_eva\user_',num2str(i),'\exp',num2str(j),'\convert_frame_normalized\face\', names_file(sk)]))));

            face_normalized  = imresize(face_normalized, [448, 448]);
            face_normalized =  rgb2gray(face_normalized);
            
            [sort_label_value, sort_label_ind] = min((abs(Gazetimestamp(:) - (time_frame(k))) * 1000)); 
            sort_label_value_list (sk) = sort_label_value;
            face_normalized_list(sk, :, :) = face_normalized;
            
        end
        del_index = find( sort_label_value_list  >=4);
        
        face_normalized_list(del_index , :, :) = [];
        
        hdf5write (['G:\remote_apperance_gaze_dataset\processed_data\data_network_training_for_frame_method_eva\','448_448_face_normalized_user_',num2str(i),'_exp',num2str(j),'.h5'],'/data', face_normalized_list);
        
        
        clearvars -except i j
    end
    
end