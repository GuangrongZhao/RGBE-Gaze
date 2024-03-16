clc
close all
clear all

mse_list = [];
load  D:\remote_dataset\calibration\CheckerboardPoints\tform_total.mat
for user_num =79:79
    user_num
    for exp =2:2
        exp
        folder_path1 = ['C:\Users\Aiot_Server\Desktop\remote_dataset\user_',num2str(user_num),'\exp',num2str(exp),'\flir\'];
        files1 = dir(fullfile(folder_path1, '*.png'));
        names_file1 = sort_nat({files1.name}); %每一个aps图片的名字
        
        output_path =  ['C:\Users\Aiot_Server\Desktop\remote_dataset\user_',num2str(user_num),'\exp',num2str(exp),'\convert2eventspace\'];
        if ~exist(output_path, 'dir')
            % 如果目录不存在，则创建目录
            mkdir(output_path);
            disp('目录创建成功！');
        end

        
        parfor i = 1:length(files1)
%             i
            file_path1 = cell2mat(fullfile(folder_path1, names_file1(i)));
            I1 = imread(file_path1);
%             I1 = rgb2gray(I1);
            I1= fliplr(I1);
            I1 = imresize(I1, [ 720 1280], 'Method', 'bicubic');
            % figure(2)
            % imshow(I1) % 翻转加变形后的rgb 图片
            
            outputView = imref2d(size(I1));
            J = imwarp(I1, tform, 'OutputView', outputView); % 映射后的rgb 图像
            
            
            %             figure(3);
            %             imshow(J);
            %             title('Transformed Image');
            output_path_each = cell2mat(fullfile(output_path, names_file1(i)));
            imwrite(J,output_path_each );
            
            
        end
        
    end
end
