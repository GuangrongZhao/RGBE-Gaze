clc 
clear all
close all


datalist = [];



for s =34:66
file_path = ['G:\remote_apperance_gaze_dataset\processed_data\pre_trained_models_for_frame_method_eva\ui_rtgene\user_', num2str(s)];
file_contents = fileread(file_path);
pattern = 'distance:(\d+\.\d+)';
distances = regexp(file_contents, pattern, 'tokens');
distances = str2double([distances{:}]);
datalist(end+1) =distances;
end

datalist_rtgene = datalist ;








% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
datalist = [];



for s =34:66
file_path = ['G:\remote_apperance_gaze_dataset\processed_data\pre_trained_models_for_frame_method_eva\ui_fullface\user_', num2str(s)];
file_contents = fileread(file_path);
pattern = 'distance:(\d+\.\d+)';
distances = regexp(file_contents, pattern, 'tokens');
distances = str2double([distances{:}]);
datalist(end+1) =distances;
end

datalist_full_face = datalist ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

datalist_vgg = [];
datalist_R = [];
datalist_L = [];

for s =34:66
file_path = ['G:\remote_apperance_gaze_dataset\processed_data\pre_trained_models_for_frame_method_eva\ui_gazenet\user_', num2str(s),'R'];
file_contents = fileread(file_path);
pattern = 'distance:(\d+\.\d+)';
distances = regexp(file_contents, pattern, 'tokens');
distances = str2double([distances{:}]);
datalist_R(end+1) =distances;
end


for s =34:66
file_path = ['G:\remote_apperance_gaze_dataset\processed_data\pre_trained_models_for_frame_method_eva\ui_gazenet\user_', num2str(s),'L'];
file_contents = fileread(file_path);
pattern = 'distance:(\d+\.\d+)';
distances = regexp(file_contents, pattern, 'tokens');
distances = str2double([distances{:}]);
datalist_L(end+1) =distances;
end


datalist_R = datalist_R';
datalist_L = datalist_L';

datalist = (datalist_R +datalist_L )/2;
datalist_vgg = datalist ;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






datalist_minist = [];
datalist_R = [];
datalist_L = [];

for s =34:66
file_path = ['G:\remote_apperance_gaze_dataset\processed_data\pre_trained_models_for_frame_method_eva\ui_minist\user_', num2str(s),'R'];
file_contents = fileread(file_path);
pattern = 'distance:(\d+\.\d+)';
distances = regexp(file_contents, pattern, 'tokens');
distances = str2double([distances{:}]);
datalist_R(end+1) =distances;
end


for s =34:66
file_path = ['G:\remote_apperance_gaze_dataset\processed_data\pre_trained_models_for_frame_method_eva\ui_minist\user_', num2str(s),'L'];
file_contents = fileread(file_path);
pattern = 'distance:(\d+\.\d+)';
distances = regexp(file_contents, pattern, 'tokens');
distances = str2double([distances{:}]);
datalist_L(end+1) =distances;
end


datalist_R = datalist_R';
datalist_L = datalist_L';

datalist = (datalist_R +datalist_L )/2;
datalist_minist = datalist ;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

datalist_voxel = [];
datalist_R = [];
datalist_L = [];

for s =34:66
file_path = ['G:\remote_apperance_gaze_dataset\processed_data\pre_trained_models_for_event_method_eva\ui_voxel_grid\user_', num2str(s),'R'];
file_contents = fileread(file_path);
pattern = 'distance:(\d+\.\d+)';
distances = regexp(file_contents, pattern, 'tokens');
distances = str2double([distances{:}]);
datalist_R(end+1) =distances;
end


for s =34:66
file_path = ['G:\remote_apperance_gaze_dataset\processed_data\pre_trained_models_for_event_method_eva\ui_voxel_grid\user_', num2str(s),'L'];
file_contents = fileread(file_path);
pattern = 'distance:(\d+\.\d+)';
distances = regexp(file_contents, pattern, 'tokens');
distances = str2double([distances{:}]);
datalist_L(end+1) =distances;
end


datalist_R = datalist_R';
datalist_L = datalist_L';

datalist = (datalist_R +datalist_L )/2;
datalist_voxel = datalist ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


datalist_frame = [];
datalist_R = [];
datalist_L = [];

for s =34:66
file_path = ['G:\remote_apperance_gaze_dataset\processed_data\pre_trained_models_for_frame_method_eva\ui_dilated\user_', num2str(s),'R'];
file_contents = fileread(file_path);
pattern = 'distance:(\d+\.\d+)';
distances = regexp(file_contents, pattern, 'tokens');
distances = str2double([distances{:}]);
datalist_R(end+1) =distances;
end


for s =34:66

file_path = ['G:\remote_apperance_gaze_dataset\processed_data\pre_trained_models_for_frame_method_eva\ui_dilated\user_', num2str(s),'L'];
file_contents = fileread(file_path);
pattern = 'distance:(\d+\.\d+)';
distances = regexp(file_contents, pattern, 'tokens');
distances = str2double([distances{:}]);
datalist_L(end+1) =distances;
end


datalist_R = datalist_R';
datalist_L = datalist_L';

datalist = (datalist_R +datalist_L )/2;
datalist_frame = datalist ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





datalist_voxel_only_frame = [];
datalist_R = [];
datalist_L = [];

for s =34:66
% s
file_path = ['G:\remote_apperance_gaze_dataset\processed_data\pre_trained_models_for_event_method_eva\ui_frame_branch\user_', num2str(s),'R'];
file_contents = fileread(file_path);
pattern = 'distance:(\d+\.\d+)';
distances = regexp(file_contents, pattern, 'tokens');
distances = str2double([distances{:}]);
datalist_R(end+1) =distances;
end


for s =34:66
% s
file_path = ['G:\remote_apperance_gaze_dataset\processed_data\pre_trained_models_for_event_method_eva\ui_frame_branch\user_', num2str(s),'L'];
file_contents = fileread(file_path);
pattern = 'distance:(\d+\.\d+)';
distances = regexp(file_contents, pattern, 'tokens');
distances = str2double([distances{:}]);
datalist_L(end+1) =distances;
end


datalist_R = datalist_R';
datalist_L = datalist_L';

datalist = (datalist_R +datalist_L )/2;

datalist_voxel_only_frame = datalist ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%







fprintf( 'Mnist : %.2f \n',mean (datalist_minist));
fprintf( 'Vgg :%.2f \n',mean (datalist_vgg));
fprintf( 'Full_face:%.2f \n', mean (datalist_full_face));
fprintf( 'Dilated-Net:%.2f \n', mean (datalist_frame));
fprintf( 'RT-Gene:%.2f \n', mean (datalist_rtgene));
fprintf( 'RGBE-Gaze:%.2f \n', mean (datalist_voxel));
fprintf( 'Dilated-Net only frame  2:%.2f \n', mean (datalist_voxel_only_frame));
figure(1)
x = 34:1:66;

% y = [datalist_voxel,datalist_voxel_only_frame ,datalist_dilatednet,datalist_rtgene,datalist_mnist ,datalist_full_face , datalist_vgg];

% y = [datalist_minist,datalist_voxel,datalist_frame,datalist_full_face' ,datalist_voxel_only_frame,datalist_rtgene'];
y = [datalist_voxel, datalist_voxel_only_frame, datalist_frame,  datalist_rtgene',datalist_minist,datalist_full_face' ,datalist_vgg];

b=bar(x,y,0.9);



set(b(1),'FaceColor',[246 83 20]/255) % 颜色调节

set(b(2),'FaceColor',[124 187 0]/255)% 颜色调节

set(b(3),'FaceColor',[0 161 241]/255) % 颜色调节

set(b(4),'FaceColor',[255 187 0]/255) % 颜色调节

set(b(5),'FaceColor',[0.49 0.18 0.56]) % 颜色调节


% set(b(6),'FaceColor',[219 249 244]/255) % 颜色调节
% 
% set(b(7),'FaceColor',[0,0.78,0.55]) % 颜色调节

grid on;

% set(gca,'YLim',[0.60 1]);
% set(gca,'ytick',0.60:0.1:1)
% set(gca,'XLim',[0.5 20.5]);
set(gca,'Xtick',34:1:66)
% set(gca,'XTickLabel',33:1:66)
% set(gca,'XTickLabel',{'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'})
xlabel('Subject id');

ylabel('Angular error [degree]');

set(gcf, 'Color', 'white');
% legend('RGBE-Gaze(2%) 4.71 ','Dilated-Net(2%) 4.94 ','Dilated-Net(10%) 4.68 ','RT-Gene(10%) 4.77 ','MnistNet(10%) 4.98','FullFace(10%) 5.66 ','GazeNet(10%) 5.70 ');

legend('RGBE-Gaze 4.57 ','RGBE-Gaze (w/o EB) 4.86 ','Dilated-Net 4.96','RT-Gene 4.99','MnistNet 5.09', 'FullFace 5.55 ', 'GazeNet 5.80');

set(gca,'linewidth',2)

set(gca,'FontSize',20,'FontWeight','bold')


