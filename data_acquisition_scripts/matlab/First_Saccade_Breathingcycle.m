% 创建一个图形窗口

clc;clear;close all
%生成新的figure
hold on
disp('first')
axis([ -10 110 -10 100])
set(gca,'YLim',[-10 110]);%X轴的数据显示范围
set(gca,'XLim',[-10 110]);%X轴的数据显示范围
set(gcf,'MenuBar','none');
% axis off;
box on;
set(gcf,'color','black');
colordef black;
% set(gca,'LooseInset',get(gca,'TightInset'))
% set(gca, 'LooseInset', [0,0,0,0]);
set(gca, 'LooseInset', [500,500,500,500]);
pause(3)
pause(2)

x_list = [0:10:100]';
y_list = [0:10:100]';

for j = 1:1:11
    for k = 1:1:11
     
xy_mat((j-1)*11+k,:) = [x_list(j);y_list(k)];

    end
end


% D = randperm(xy_mat,121);

% randIndex = randperm(121);
% save randIndex randIndex

load randIndex 

xy_mat_new = xy_mat(randIndex(1:60) ,:);




% 设置绘图区域的坐标范围
% axis([-1, 1, -1, 1]);

% 设置绘图窗口不可改变大小
% axis equal;
% axis off;

% 初始半径和最终半径
initial_radius = 2.5;
final_radius = 5;

% 创建一个红色圆形
theta = linspace(0, 2*pi, 100); % 角度范围
x = initial_radius * cos(theta);
y = initial_radius * sin(theta);
h = fill(x, y, 'r'); % 填充红色

num_frames = 40; % 动画帧数
animation_time = 1.5; % 动画总时间（秒）

% 计算每一帧的半径增量
radius_increment = (final_radius - initial_radius) / num_frames;


num_frames_decrease= 40; % 动画帧数
animation_time_decrease = 0.8; % 动画总时间（秒）

% 计算每一帧的半径增量
radius_increment_decrease = (final_radius - initial_radius) / num_frames_decrease;


% num_rows = 6;
% num_columns = 10;
% 
% % 创建一个空的二维数组
% matrix = zeros(num_rows, num_columns);

% 定义 x 和 y 的取值范围
% x_values = 10:10:100; % 从 10 到 100，每隔 10 一个值
% y_values = 10:10:60; % 从 10 到 100，每隔 10 一个值
% 
% % 创建一个空的二维数组，用于存储坐标点
% coordinates = [];
% 
% % 填充数组
% for i = 1:length(y_values)
%     y = y_values(i);
%     % 逐渐增大 x（S 形）
%     if mod(i, 2) == 1
%         for x = x_values
%             coordinates = [coordinates; x, y];
%         end
%     % 逐渐减小 x（S 形）
%     else
%         for x = fliplr(x_values)
%             coordinates = [coordinates; x, y];
%         end
%     end
% end

center_positions = xy_mat_new ; % 自定义不同位置的圆心坐标



num_positions = size(center_positions, 1);




% 设置绘图区域的坐标范围，以保持宽高比一致




for position_index = 1:num_positions

tic

center_x = center_positions(position_index, 1);
center_y = center_positions(position_index, 2);
% countdown_text = text(center_x, center_y, '3', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 20, 'Color', 'white');


% 添加一个黑点到圆心
plot(center_x, center_y, 'k.', 'MarkerSize', 45);

% 设置动画参数

% 创建呼吸动画效果
for frame = 1:num_frames
    % 计算当前半径
%         current_countdown = ceil(3 - frame * animation_time);
%     current_countdown = ceil(animation_time - frame * (animation_time / num_frames));
    
    current_radius = initial_radius + frame * radius_increment;
    
%     set(countdown_text, 'String', num2str(current_countdown));
    % 更新圆的尺寸
    scaled_x = current_radius/1.73 * cos(theta)+ center_x;
    scaled_y = current_radius * sin(theta)+ center_y;
    
    % 更新圆的位置
    set(h, 'XData', scaled_x, 'YData', scaled_y);
    
    % 暂停一段时间以形成动画效果
    pause(animation_time / num_frames);
end

% countdown_text = text(center_x, center_y, num2str(random_sequence(position_index)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 20, 'Color', 'white');
% if random_sequence(position_index)==0
countdown_text = text(center_x, center_y, 'click!', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 20, 'Color', 'white','FontWeight','bold');
% else
% countdown_text = text(center_x, center_y, 'M', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 20, 'Color', 'white','FontWeight','bold');
% end

pause(0.5);

delete(countdown_text);

for frame = 1:num_frames_decrease
    % 计算当前半径
%         current_countdown = ceil(3 - frame * animation_time);
%     current_countdown = ceil( (animation_time - frame * (animation_time / num_frames)));
    
    current_radius = final_radius - frame  * radius_increment_decrease;
    
%     set(countdown_text, 'String', num2str(current_countdown));
    % 更新圆的尺寸
    scaled_x = current_radius/1.73 * cos(theta)+ center_x;
    scaled_y = current_radius * sin(theta)+ center_y;
    
    % 更新圆的位置
    set(h, 'XData', scaled_x, 'YData', scaled_y);
    
    % 暂停一段时间以形成动画效果
    pause(animation_time_decrease / num_frames_decrease);
end
toc

set(h, 'XData', 0, 'YData', 0);

% pause(1);

end



plot(50, 50, 'r.', 'MarkerSize', 200);
countdown_text = text(50, 50, '请看这》》》》请稍等》》》》', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 20, 'Color', 'white');
pause(40);