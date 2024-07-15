% Create a graphics window

clc;clear;close all
% Generate new figure
disp('second')
hold on
axis([ -10 110 -10 100])
set(gca,'YLim',[-10 110]); % Y-axis data display range
set(gca,'XLim',[-10 110]); % X-axis data display range
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

xy_mat_new = xy_mat(randIndex(61:121),:);

% Set the coordinate range of the drawing area
% axis([-1, 1, -1, 1]);

% Set the drawing window to be non-resizable
% axis equal;
% axis off;

% Initial radius and final radius
initial_radius = 2.5;
final_radius = 5;

% Create a red circle
theta = linspace(0, 2*pi, 100); % Angle range
x = initial_radius * cos(theta);
y = initial_radius * sin(theta);
h = fill(x, y, 'r'); % Fill with red

num_frames = 40; % Number of animation frames
animation_time = 1.5; % Total animation time (seconds)

% Calculate the radius increment for each frame
radius_increment = (final_radius - initial_radius) / num_frames;

num_frames_decrease = 40; % Number of animation frames
animation_time_decrease = 0.8; % Total animation time (seconds)

% Calculate the radius decrement for each frame
radius_increment_decrease = (final_radius - initial_radius) / num_frames_decrease;

% Define the range of x and y values
% x_values = 10:10:100; % From 10 to 100, with a value every 10
% y_values = 10:10:60; % From 10 to 100, with a value every 10

% Create an empty 2D array to store coordinates
% coordinates = [];

% Fill the array
% for i = 1:length(y_values)
%     y = y_values(i);
%     % Gradually increase x (S-shape)
%     if mod(i, 2) == 1
%         for x = x_values
%             coordinates = [coordinates; x, y];
%         end
%     % Gradually decrease x (S-shape)
%     else
%         for x = fliplr(x_values)
%             coordinates = [coordinates; x, y];
%         end
%     end
% end

center_positions = xy_mat_new ; % Custom coordinates of different center positions

num_positions = size(center_positions, 1);

% Set the coordinate range of the drawing area to maintain aspect ratio

tic

for position_index = 1:num_positions
    
    center_x = center_positions(position_index, 1);
    center_y = center_positions(position_index, 2);
    % countdown_text = text(center_x, center_y, '3', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 20, 'Color', 'white');
    
    % Add a black dot to the center
    plot(center_x, center_y, 'k.', 'MarkerSize', 45);
    
    % Set animation parameters
    
    % Create breathing animation effect
    for frame = 1:num_frames
        % Calculate the current radius
        % current_countdown = ceil(3 - frame * animation_time);
        % current_countdown = ceil(animation_time - frame * (animation_time / num_frames));
        
        current_radius = initial_radius + frame * radius_increment;
        
        % set(countdown_text, 'String', num2str(current_countdown));
        % Update the size of the circle
        scaled_x = current_radius/1.73 * cos(theta) + center_x;
        scaled_y = current_radius * sin(theta) + center_y;
        
        % Update the position of the circle
        set(h, 'XData', scaled_x, 'YData', scaled_y);
        
        % Pause for a while to create an animation effect
        pause(animation_time / num_frames);
    end
    
    % countdown_text = text(center_x, center_y, num2str(random_sequence(position_index)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 20, 'Color', 'white');
    % if random_sequence(position_index) == 0
    countdown_text = text(center_x, center_y, 'click!', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 20, 'Color', 'white', 'FontWeight', 'bold');
    % else
    % countdown_text = text(center_x, center_y, 'M', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 20, 'Color', 'white', 'FontWeight', 'bold');
    % end
    pause(0.5);
    delete(countdown_text);
    
    for frame = 1:num_frames_decrease
        % Calculate the current radius
        % current_countdown = ceil(3 - frame * animation_time);
        % current_countdown = ceil((animation_time - frame * (animation_time / num_frames)));
        
        current_radius = final_radius - frame * radius_increment_decrease;
        
        % set(countdown_text, 'String', num2str(current_countdown));
        % Update the size of the circle
        scaled_x = current_radius/1.73 * cos(theta) + center_x;
        scaled_y = current_radius * sin(theta) + center_y;
        
        % Update the position of the circle
        set(h, 'XData', scaled_x, 'YData', scaled_y);
        
        % Pause for a while to create an animation effect
        pause(animation_time_decrease / num_frames_decrease);
    end
    
    set(h, 'XData', 0, 'YData', 0);
    
    % pause(1);
    toc
end

plot(50, 50, 'r.', 'MarkerSize', 200);
countdown_text = text(50, 50, 'Please watch this >>> Please wait >>>', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 20, 'Color', 'white');
pause(20);
