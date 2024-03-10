clc 
close all
clear all

mse_list = [];
folder_path1 = 'C:\Users\Aiot_Server\Desktop\remote_dataset\calibration\rgb\';
files1 = dir(fullfile(folder_path1, '*.png'));
names_file1 = sort_nat({files1.name}); %每一个aps图片的名字

folder_path2 = 'C:\Users\Aiot_Server\Desktop\remote_dataset\calibration\events\';
files2 = dir(fullfile(folder_path2, '*.png'));
names_file2 = sort_nat({files2.name}); %每一个aps图片的名字

% load initial_imagePoints
imagePoints1_list = [];
imagePoints2_list = [];

for i = 1:1
    
file_path1 = cell2mat(fullfile(folder_path1, names_file1(i)));
I1 = imread(file_path1);

% figure(1)
% imshow(I1) % 原始rgb 图片
I1 = rgb2gray(I1);
I1= fliplr(I1);
I1 = imresize(I1, [ 720 1280], 'Method', 'bicubic');
% figure(2)
% imshow(I1) % 翻转加变形后的rgb 图片

file_path2 = cell2mat(fullfile(folder_path2, names_file2(i)));
I2 = imread(file_path2);
I2 = rgb2gray(I2);
% figure(3)
% imshow(I2)
img1 = I1;
img2 = I2;

% 
% [imagePoints1] = detectCheckerboardPoints(img1);
% [imagePoints2] = detectCheckerboardPoints(img2);
% [imagePoints2, boardSizeFound2] = detectCheckerboardPoints(img2, boardSize);

% imagePoints1_list= [imagePoints1_list;imagePoints1];
% imagePoints2_list= [imagePoints2_list;imagePoints2];

% figure(1);
% imshow(img1);
% hold on;
% plot(imagePoints1(:,1), imagePoints1(:,2), 'ro', 'MarkerSize', 10, 'LineWidth', 3);
% title('Detected Checkerboard Points');
% hold off;

% figure(2);
% imshow(img2);
% hold on;
% plot(imagePoints2(:,1), imagePoints2(:,2), 'ro', 'MarkerSize', 10, 'LineWidth', 3);
% title('Detected Checkerboard Points');
% hold off;


outputView = imref2d(size(img1)); 
% load C:\Users\Aiot_Server\Desktop\remote_dataset\calibration\CheckerboardPoints\totalcheckerboardpoints.mat %选择使用所有的checkpoints 来计算转换关系
% tform = fitgeotrans(imagePoints1_list, imagePoints2_list, 'projective');

load  C:\Users\Aiot_Server\Desktop\remote_dataset\calibration\CheckerboardPoints\tform_total.mat

% tform = fitgeotrans(imagePoints1, imagePoints2, 'projective'); %计算转换关系

% save tform2 tform
% load tform2

J = imwarp(img1, tform, 'OutputView', outputView); % 映射后的rgb 图像


% figure(3);
% imshow(J);
% title('Transformed Image');

% figure(4)
% imshowpair(img2,J, 'montage');

[imagePoints1] = detectCheckerboardPoints(J);
[imagePoints2] = detectCheckerboardPoints(img2);


% 将两幅图像叠加
% merged_image = imfuse(img2,J,'falsecolor');
% figure(5)
% % 显示叠加后的图像
% imshow(merged_image);
% hold on
% plot(imagePoints2(:,1), imagePoints2(:,2), 'ro', 'MarkerSize', 10, 'LineWidth', 3);
% hold off

% mse = sum(sum((imagePoints1-imagePoints2).^2)) / numel(imagePoints1);


mse= mean(sqrt(sum((imagePoints1 - imagePoints2).^2, 2))); % 对每个点计算欧氏距离

fprintf('MSE %d\n' ,mse)
mse_list (end+1) = mse;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%图像左右拼接显示%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 获取两幅图像的大小
[height1, width1, ~] = size(img2);
[height2, width2, ~] = size(J);  
A= img2;
B = J;

w = size(A,2); % 获取图像宽度
h = size(A,1); % 获取图像高度

half_w = round(w / 2); % 计算宽度的一半
crop_A = imcrop(A, [half_w 1 half_w h]); % 对图像1进行裁剪
crop_B = imcrop(B, [1 1 half_w h]); % 对图像2进行裁剪
C = [crop_B crop_A]; % 进行拼接
imshow(C)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%图像左右拼接显示%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

mean(mse_list)
%     0.0305