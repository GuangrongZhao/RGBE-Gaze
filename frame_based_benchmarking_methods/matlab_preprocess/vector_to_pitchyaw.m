% function out = vector_to_pitchyaw(vectors)
%     n = size(vectors, 1);
%     out = zeros(n, 2);
%     vectors = vectors ./ sqrt(sum(vectors.^2, 2));
%     out(:, 1) = asin(vectors(:, 2));  % theta
%     out(:, 2) = atan2(vectors(:, 1), vectors(:, 3));  % phi
% end

function pitchyaw = vector_to_pitchyaw(gaze_vectors)
 
    norms = vecnorm(gaze_vectors, 2, 2);
    normalized_vectors = gaze_vectors ./ norms;
   
%     pitch = asin(-normalized_vectors(:, 2));
%     yaw = atan2(normalized_vectors(:, 1), -normalized_vectors(:, 3));
%     
    pitch = asin(normalized_vectors(:, 2));
    yaw = atan2(normalized_vectors(:, 1), normalized_vectors(:, 3));
%     
    pitchyaw = [pitch, yaw];
end