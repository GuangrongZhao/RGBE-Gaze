function [rotation_matrix] = rotation_vector2matrix(rotation_vector)
% 旋转向量

% 将旋转向量转换为旋转矩阵
theta = norm(rotation_vector);
k = rotation_vector / theta;
K = [0 -k(3) k(2); k(3) 0 -k(1); -k(2) k(1) 0];
rotation_matrix = eye(3) + sin(theta) * K + (1 - cos(theta)) * K^2;

% disp("旋转向量：");
% disp(rotation_vector);
% disp("旋转矩阵：");
% disp(rotation_matrix);

