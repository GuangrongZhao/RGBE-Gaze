clc;
clear all;
close all;

% Loop through users from user_1R to user_22R
for userIdx = 50:55
    user = sprintf('user_%dR', userIdx);
    
    % Predict Data
    folder = 'G:\remote_apperance_gaze_dataset\processed_data\data_for_visualisation\result';
    filePattern = fullfile(folder, sprintf('batch_*predict_%s.mat', user));
    matFiles = dir(filePattern);
    combinedData_predict = [];

    for k = 1:length(matFiles)
        fullPath = fullfile(folder, matFiles(k).name);
        loadedData = load(fullPath);
        if k == 1
            combinedData_predict = loadedData.savepredict_final;
        else
            combinedData_predict = cat(1, combinedData_predict, loadedData.savepredict_final);
        end
    end

    % Reference Data
    filePattern = fullfile(folder, sprintf('batch_*reference_%s.mat', user));
    matFiles = dir(filePattern);
    combinedData_reference = [];

    for k = 1:length(matFiles)
        fullPath = fullfile(folder, matFiles(k).name);
        loadedData = load(fullPath);
        if k == 1
            combinedData_reference = loadedData.savereference_final;
        else
            combinedData_reference = cat(1, combinedData_reference, loadedData.savereference_final);
        end
    end

    % Angle Error Data
    filePattern = fullfile(folder, sprintf('batch_*angle_error_%s.mat', user));
    matFiles = dir(filePattern);
    combinedData_angle_error = [];

    for k = 1:length(matFiles)
        fullPath = fullfile(folder, matFiles(k).name);
        loadedData = load(fullPath);
        if k == 1
            combinedData_angle_error = loadedData.saveouputs;
        else
            combinedData_angle_error = cat(1, combinedData_angle_error, loadedData.saveouputs);
        end
    end

    % Only Frame Predict Data
    filePattern = fullfile(folder, sprintf('only_frame_batch_*predict_%s.mat', user));
    matFiles = dir(filePattern);
    only_frame_combinedData_predict = [];

    for k = 1:length(matFiles)
        fullPath = fullfile(folder, matFiles(k).name);
        loadedData = load(fullPath);
        if k == 1
            only_frame_combinedData_predict = loadedData.savepredict_final;
        else
            only_frame_combinedData_predict = cat(1, only_frame_combinedData_predict, loadedData.savepredict_final);
        end
    end

    % Only Frame Reference Data
    filePattern = fullfile(folder, sprintf('only_frame_batch_*reference_%s.mat', user));
    matFiles = dir(filePattern);
    only_frame_combinedData_reference = [];

    for k = 1:length(matFiles)
        fullPath = fullfile(folder, matFiles(k).name);
        loadedData = load(fullPath);
        if k == 1
            only_frame_combinedData_reference = loadedData.savereference_final;
        else
            only_frame_combinedData_reference = cat(1, only_frame_combinedData_reference, loadedData.savereference_final);
        end
    end

    % Only Frame Angle Error Data
    filePattern = fullfile(folder, sprintf('only_frame_batch_*angle_error_%s.mat', user));
    matFiles = dir(filePattern);
    only_frame_combinedData_angle_error = [];

    for k = 1:length(matFiles)
        fullPath = fullfile(folder, matFiles(k).name);
        loadedData = load(fullPath);
        if k == 1
            only_frame_combinedData_angle_error = loadedData.saveouputs;
        else
            only_frame_combinedData_angle_error = cat(1, only_frame_combinedData_angle_error, loadedData.saveouputs);
        end
    end

    % Plotting and Saving Figures
    figure;
    plot(only_frame_combinedData_predict(:,1))
    hold on;
    plot(only_frame_combinedData_reference(:,1))
    plot(only_frame_combinedData_angle_error)
    plot(combinedData_predict(:,1))
    plot(combinedData_angle_error)
    legend('frame predict', 'frame reference', 'only frame angle error', 'combinedData predict', 'combinedData angle error')
    saveas(gcf, fullfile(folder, sprintf('png_1_%s.png', user)))

    figure;
    plot(only_frame_combinedData_predict(:,2))
    hold on;
    plot(only_frame_combinedData_reference(:,2))
    plot(only_frame_combinedData_angle_error)
    plot(combinedData_predict(:,2))
    plot(combinedData_angle_error)
    legend('frame predict', 'frame reference', 'only frame angle error', 'combinedData predict', 'combinedData angle error')
    saveas(gcf, fullfile(folder, sprintf('png_2_%s.png', user)))

    figure;
    scatter(1:length(combinedData_predict(:,2)), combinedData_predict(:,2), 'o', 'filled');
    hold on;
    scatter(1:length(only_frame_combinedData_predict(:,2)), only_frame_combinedData_predict(:,2), 'o', 'filled');
    scatter(1:length(only_frame_combinedData_reference(:,2)), only_frame_combinedData_reference(:,2), 'o', 'filled');
    legend('event frame predict', 'frame predict', 'reference')
    saveas(gcf, fullfile(folder, sprintf('png_3_%s.png', user)))

    close all;
end