for i = 1:66
    for j = 1:6
        j
        
        % delete([['D:/remote_dataset/user_',num2str(i),'/exp',num2str(j),'convert_frame_normalized']]);
        % delete([['D:/remote_dataset/user_',num2str(i),'/exp',num2str(j),'convert_frame_normalized.h5']]);
        
%           mkdir(['C:/Users/Aiot_Server/Desktop/remote_dataset/voxel_grid_representation/user_',num2str(i),'/exp',num2str(j),'/frame_1ms_voxel_grid_for_eye_tracker/left_eye']);
%           mkdir(['C:/Users/Aiot_Server/Desktop/remote_dataset/voxel_grid_representation/user_',num2str(i),'/exp',num2str(j),'/frame_1ms_voxel_grid_for_eye_tracker/right_eye']);
        %     mkdir(['C:/Users/Aiot_Server/Desktop/remote_dataset/user_',num2str(i),'/exp',num2str(j),'/simple_crop/left_eye']);
        %      mkdir(['C:/Users/Aiot_Server/Desktop/remote_dataset/user_',num2str(i),'/exp',num2str(j),'/simple_crop/right_eye']);
        %     mkdir(['C:/Users/Aiot_Server/Desktop/remote_dataset/user_',num2str(i),'/exp',num2str(j),'/flir']);
        %       mkdir(['C:/Users/Aiot_Server/Desktop/remote_dataset/user_',num2str(i),'/exp',num2str(j),'/gazepoint']);
        %           mkdir(['D:/remote_dataset/raw_prophesee/user_',num2str(i),'/exp',num2str(j),'/prophesee']);
        
   
        
         mkdir(['G:\remote_apperance_gaze_dataset\processed_data\data_for_visualisation\voxel_grid_representation_2.5k/user_',num2str(i),'/exp',num2str(j),'/voxel_grid_saved/left_eye']);

         mkdir(['G:\remote_apperance_gaze_dataset\processed_data\data_for_visualisation\voxel_grid_representation_2.5k/user_',num2str(i),'/exp',num2str(j),'/voxel_grid_for_eye_tracker/left_eye']);
         
% %          movefile(['G:\remote_apperance_gaze_dataset\processed_data\random_data_for_event_method_eva\voxel_grid_representation/user_',num2str(i),'/exp',num2str(j)]);
%          
% %          movefile(['G:\remote_apperance_gaze_dataset\processed_data\random_data_for_event_method_eva\voxel_grid_representation_2.5k/user_',num2str(i),'/exp',num2str(j),'/voxel_grid_for_eye_tracker_0.5ms']);
%          
         mkdir(['G:\remote_apperance_gaze_dataset\processed_data\data_for_visualisation\voxel_grid_representation_2.5k/user_',num2str(i),'/exp',num2str(j),'/voxel_grid_saved/right_eye']);

         mkdir(['G:\remote_apperance_gaze_dataset\processed_data\data_for_visualisation\voxel_grid_representation_2.5k/user_',num2str(i),'/exp',num2str(j),'/voxel_grid_for_eye_tracker/right_eye']);
        

%          
                  mkdir(['G:\remote_apperance_gaze_dataset\processed_data\data_for_visualisation\frame/user_',num2str(i),'/exp',num2str(j),'/convert_frame_normalized/']);
                  mkdir(['G:\remote_apperance_gaze_dataset\processed_data\data_for_visualisation\frame/user_',num2str(i),'/exp',num2str(j),'/convert_frame_normalized/face']);
                  mkdir(['G:\remote_apperance_gaze_dataset\processed_data\data_for_visualisation\frame/user_',num2str(i),'/exp',num2str(j),'/convert_frame_normalized/left_eye']);
                  mkdir(['G:\remote_apperance_gaze_dataset\processed_data\data_for_visualisation\frame/user_',num2str(i),'/exp',num2str(j),'/convert_frame_normalized/right_eye']);


                  

        %           mkdir(['D:/remote_dataset/old2/user_',num2str(i),'/exp',num2str(j),'/convert_frame_normalized/'];
        %           mkdir(['D:/remote_dataset/old2/user_',num2str(i),'/exp',num2str(j),'/convert_frame_normalized/face']);
        %           mkdir(['D:/remote_dataset/old2/user_',num2str(i),'/exp',num2str(j),'/convert_frame_normalized/left_eye']);
        %           mkdir(['D:/remote_dataset/old2/user_',num2str(i),'/exp',num2str(j),'/convert_frame_normalized/right_eye']);
        % %
        
%         if exist([' D:/remote_dataset/processed_data/user_',num2str(i),'/exp',num2str(j),'/convert_frame_normalized'], 'dir')
%             % 如果目录不存在，则创建目录
%                    rmdir([' D:/remote_dataset/processed_data/user_',num2str(i),'/exp',num2str(j),'/convert_frame_normalized'],'s');
%       
% 
%             
%             disp('目录创建成功！');
%         end
%        
%         
%           rmdir([' D:/remote_dataset/processed_data/user_',num2str(i),'/exp',num2str(j),'/convert_frame_normalized/left_eye/'],'s');
%           
          
          
        % mkdir(['C:/Users/Aiot_Server/Desktop/remote_dataset/user_',num2str(i),'/exp',num2str(j),'/kinect']);
        % rmdir(['C:/Users/Aiot_Server/Desktop/remote_dataset/user_',num2str(i),'/exp',num2str(j),'/kinect'] ,'s');
        % rmdir(['C:/Users/Aiot_Server/Desktop/remote_dataset/user_',num2str(i),'/exp',num2str(j),'/face'] ,'s');
        % rmdir([['C:/Users/Aiot_Server/Desktop/remote_dataset/user_',num2str(i),'/exp',num2str(j),'/convert_frame_normalized_']],'s');
        
%         rmdir([' D:/remote_dataset/processed_data/user_',num2str(i),'/exp',num2str(j),'/convert_frame_normalized/right_eye/'],'s');
%         rmdir([' D:/remote_dataset/processed_data/user_',num2str(i),'/exp',num2str(j),'/convert_frame_normalized/left_eye/'],'s');
%         D:\remote_dataset\processed_data\user_50\exp5\convert_frame_normalized\right_eye
        %  file_id = fopen(['C:/Users/Aiot_Server/Desktop/remote_dataset/user_',num2str(i),'/exp',num2str(j),'/gazepoint/time_win.txt'], 'w');
        % % 关闭文件
        % fclose(file_id);
        
        
        %           mkdir(['C:/Users/Aiot_Server/Desktop/remote_dataset/user_',num2str(i),'/exp',num2str(j),'/convert_frame_normalized/face']);
        %           mkdir(['C:/Users/Aiot_Server/Desktop/remote_dataset/user_',num2str(i),'/exp',num2str(j),'/convert_frame_normalized/left_eye']);
        %           mkdir(['C:/Users/Aiot_Server/Desktop/remote_dataset/user_',num2str(i),'/exp',num2str(j),'/convert_frame_normalized/right_eye']);
        
        %          sourcepath =  ['D:\remote_dataset\user_',num2str(i),'\exp',num2str(j),'\flir\no_detected_faces_index.txt'];
        %          despath =  ['D:\remote_dataset\user_',num2str(i),'\exp',num2str(j),'\convert2eventspace\no_detected_faces_index.txt'];
        % movefile(sourcepath,despath);
        %    sourcepath =  ['D:\remote_dataset\user_',num2str(i),'\exp',num2str(j),'\flir\timestamp.txt'];
        %          despath =  ['D:\remote_dataset\user_',num2str(i),'\exp',num2str(j),'\convert2eventspace\timestamp.txt'];
        % movefile(sourcepath,despath);
        % movefile(['D:\remote_dataset\user_',num2str(i),'\exp',num2str(j),'\flir\timestamp_cpu.txt'],['D:\remote_dataset\user_',num2str(i),'\exp',num2str(j),'\convert2eventspace\timestamp_cpu.txt']);
        
        % movefile(['D:\remote_dataset\raw_filr\user_',num2str(i),'\exp',num2str(j),'\prophesee'],['D:\remote_dataset\raw_prophesee\user_',num2str(i),'\exp',num2str(j),'\']);
        % movefile(['D:\remote_dataset\user_',num2str(i),'\exp',num2str(j),'\prophesee\x.txt'],['D:\remote_dataset\raw_filr\user_',num2str(i),'\exp',num2str(j),'\prophesee\x.txt']);
%         movefile(['D:\remote_dataset\user_',num2str(i),'\exp',num2str(j),'\prophesee\y.txt'],['D:\remote_dataset\raw_filr\user_',num2str(i),'\exp',num2str(j),'\prophesee\y.txt']);
%          movefile(['D:/remote_dataset/processed_data_voxel_grid/user_',num2str(i),'/exp',num2str(j),'/image_list.mat'],['D:/remote_dataset/user_',num2str(i),'/exp',num2str(j),'/image_list.mat']);
        
        
%           save(['D:/remote_dataset/processed_data_voxel_grid/user_',num2str(i),'/exp',num2str(j),'/image_list.mat'], 'names_file_list');
%    
% if exist(['G:\remote_apperance_gaze_dataset\processed_data\random_data_for_event_method_eva\voxel_grid_representation_2.5k\user_',num2str(i),'\exp',num2str(j)','\voxel_grid_for_eye_tracker_2.5k'], 'dir')
%    i
%         rmdir(['G:\remote_apperance_gaze_dataset\processed_data\random_data_for_event_method_eva\voxel_grid_representation_2.5k\user_',num2str(i),'\exp',num2str(j)','\voxel_grid_for_eye_tracker_2.5k'],'s');
%         
% end
% %         
% if exist(['G:\remote_apperance_gaze_dataset\processed_data\random_data_for_event_method_eva\voxel_grid_representation\user_',num2str(i),'\exp',num2str(j)'], 'dir')
%         rmdir(['G:\remote_apperance_gaze_dataset\processed_data\random_data_for_event_method_eva\voxel_grid_representation\user_',num2str(i),'\exp',num2str(j)'],'s');
%         
% end
        % rmdir(['C:/Users/Aiot_Server/Desktop/remote_dataset/voxel_grid_representation/user_',num2str(i),'/exp',num2str(j),'/voxel_grid_for_eye_tracker'],'s');

        % movefile(['C:/Users/Aiot_Server/Desktop/remote_dataset/user_',num2str(i),'/exp',num2str(j),'/flir/no_detected_faces_index.txt'],['C:/Users/Aiot_Server/Desktop/remote_dataset/user_',num2str(i),'/exp',num2str(j),'/convert_frame_normalized/no_detected_faces_index.txt']);
        
        
    end
end