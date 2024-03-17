clc
close all
clear all
projectpath = input('Enter project location(such as /home/remote_apperance_gaze_dataset): ', 's');
load ([projectpath, '/align_frame_event_pixel/tform_total.mat'])
for user_num =1:1
    for exp = 1:1
        folder_path1 = [projectpath, '/raw_data/user_',num2str(user_num),'/exp',num2str(exp),'/flir/'];
        files1 = dir(fullfile(folder_path1, '*.jpg'));
        names_file1 = sort_nat({files1.name});
        output_path =  [projectpath, '/raw_data/user_',num2str(user_num),'/exp',num2str(exp),'/convert2eventspace/'];
        if ~exist(output_path, 'dir')
            mkdir(output_path);
            disp('The directory was created successfully!');
        end
        
        for i = 1:length(files1)
            file_path1 = cell2mat(fullfile(folder_path1, names_file1(i)));
            I1 = imread(file_path1);
            
            I1= fliplr(I1);
            I1 = imresize(I1, [ 720 1280], 'Method', 'bicubic');
            %figure(2)
            %imshow(I1)
            
            outputView = imref2d(size(I1));
            J = imwarp(I1, tform, 'OutputView', outputView);
            
            %figure(3);
            %imshow(J);
            %title('Transformed Image');
            output_path_each = cell2mat(fullfile(output_path, names_file1(i)));
            imwrite(J,output_path_each );
            
            
        end
        
    end
end
