function [LPD,RPD,Gazetimestamp,LGAZE,RGAZE,depth]= pupil_depth_wcs_3d_gaze(data,R,t)
LPD_id = 42;
RPD_id = 44;
LPV_id = 43;
RPV_id = 45;
BPOGV_id = 18;
del_3d_left = find(data(:,33)==0);
data(del_3d_left,:) = [];

del_3d_right = find(data(:,38)==0);
data(del_3d_right,:) =[];

del_2d_left = find(data(:,12)==0);
data(del_2d_left,:) = [];

del_2d_right = find(data(:,15)==0);
data(del_2d_right,:) =[];


% del_pupil_left = find(data(:,LPV_id)==0);
% 
% data(del_pupil_left,:) = [];
% 
% del_pupil_right = find(data(:,RPV_id)==0);
% data(del_pupil_right,:) =[];
% 
% 
% 
% del_bpog = find(data(:,BPOGV_id)==0);
% data(del_bpog,:) =[];
% 
% disp('del_pupil_left:');
% disp(length(del_pupil_left));
% disp('del_pupil_right:');
% disp(length(del_pupil_right));


 LPD = data(:,LPD_id);
 RPD = data(:,RPD_id);
    





Gazetimestamp = data(:,2);
Gazetimestamp = Gazetimestamp - Gazetimestamp(1);
% plot(diff(Gazetimestamp));
% plot(Gazetimestamp);

LPOGX_LIST = (data(:,10))*52.8;  %要乘以屏幕宽度cm
LPOGY_LIST  = (data(:,11))*29.8; %要乘以屏幕高度cm
LPOGV_LIST = (data(:,12)); %按屏幕坐标系来转换

LPOGZ_LIST  = zeros(length(LPOGY_LIST),1); %要乘以屏幕高度cm

LEYEX_LIST  = (100*data(:,29));
LEYEY_LIST  = (100*data(:,30));
LEYEZ_LIST  = (100*data(:,31));


% figure(1)
% plot(LEYEX_LIST)
% figure(2)
% plot(LEYEY_LIST)
% figure(3)
% plot(LEYEZ_LIST)
LPUPILV_LIST = data(:,33);


LEYE = [LEYEX_LIST, LEYEY_LIST, LEYEZ_LIST]';
LEYE_world = R*LEYE+t; % 右眼转换至世界坐标系
LPOG_world = [LPOGX_LIST,LPOGY_LIST,LPOGZ_LIST]';

LGAZE =  LPOG_world - LEYE_world;

RPOGX_LIST = (data(:,13))*52.8;
RPOGY_LIST  = (data(:,14))*29.8;
RPOGV_LIST = (data(:,15));
RPOGZ_LIST  = zeros(length(RPOGY_LIST),1);% 要乘以屏幕高度cm


REYEX_LIST  = (100*data(:,34));
REYEY_LIST  = (100*data(:,35));
REYEZ_LIST  = (100*data(:,36));
RPUPILV_LIST = data(:,38);

REYE = [ REYEX_LIST, REYEY_LIST, REYEZ_LIST]';
REYE_world = R*REYE+t;   %重算的结果
RPOG_world = [RPOGX_LIST,RPOGY_LIST,RPOGZ_LIST]';

RGAZE =  RPOG_world - REYE_world;

BPOGX_LIST = (data(:,16))*52.8;  %要乘以屏幕宽度cm
BPOGY_LIST  = (data(:,17))*29.8; %要乘以屏幕高度cm
BPOGV_LIST = (data(:,18)); %按屏幕坐标系来转换
BPOGZ_LIST  = zeros(length(BPOGY_LIST),1); %要乘以屏幕高度cm

BEYE_world = (LEYE_world + REYE_world) / 2;   %重算的结果
BPOG_world = [BPOGX_LIST,BPOGY_LIST,BPOGZ_LIST]';

BGAZE =  BPOG_world - BEYE_world;
% disp('size(BGAZE)')
% disp(size(BGAZE))
depth = BGAZE(3,:); 


LPD = LPD';

RPD = RPD';



% MEANGAZELEFT =  [mean(LPOGX_LIST),mean(LPOGY_LIST),mean(LPOGZ_LIST)] - mean(LEYE_world');
% MEANGAZERIGHT = [mean(RPOGX_LIST),mean(RPOGY_LIST),mean(RPOGZ_LIST)] - mean(REYE_world');


% RLPOGX = mean((RPOGX_LIST + LPOGX_LIST))/2;
% RLPOGY = mean((RPOGY_LIST + LPOGY_LIST))/2;
% RLPOGZ = mean((RPOGZ_LIST + LPOGZ_LIST))/2;
% RLCENTER =  mean(REYE_world' + LEYE_world')/2;
% MEANXYZ = [RLPOGX,RLPOGY,RLPOGZ];
% MEANGAZE = [RLPOGX,RLPOGY,RLPOGZ] - RLCENTER;


end