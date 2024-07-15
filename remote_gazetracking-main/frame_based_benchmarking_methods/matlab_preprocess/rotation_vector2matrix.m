function [rotation_matrix] = rotation_vector2matrix(rotation_vector)
% Rotation vectors

% Convert rotation vector to rotation matrix
theta = norm(rotation_vector);
k = rotation_vector / theta;
K = [0 -k(3) k(2); k(3) 0 -k(1); -k(2) k(1) 0];
rotation_matrix = eye(3) + sin(theta) * K + (1 - cos(theta)) * K^2;
% disp("rotation_vector:");
% disp(rotation_vector);
% disp("rotation_matrix:");
% disp(rotation_matrix);
