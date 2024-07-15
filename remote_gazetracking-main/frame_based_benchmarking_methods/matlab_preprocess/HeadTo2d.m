function result = HeadTo2d(head)
%     assert numel(head) == 3, ['The length of head must be 3, which is ' num2str(numel(head)) ' currently'];
%     R = Rodrigues(head);
% %     vec = R ;
%     vec = R ./ norm(R);
%     pitch = asin(-vec(2));
%     yaw = atan2(vec(1),-vec(3));
%     
    pitch = asin(head(2));
    yaw = atan2(head(1),head(3));
%     yaws;
    result = [pitch,yaw];
end

