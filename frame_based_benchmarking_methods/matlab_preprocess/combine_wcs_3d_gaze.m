function [Gazetimestamp,Combine_left_right_GAZE ]= combine_wcs_3d_gaze(data,R,t)


del_3d_left = find(data(:,33)==0);
data(del_3d_left,:) = [];

del_3d_right = find(data(:,38)==0);
data(del_3d_right,:) =[];

del_2d_left = find(data(:,12)==0);
data(del_2d_left,:) = [];

del_2d_right = find(data(:,15)==0);
data(del_2d_right,:) =[];

Gazetimestamp = data(:,2);
Gazetimestamp = Gazetimestamp - Gazetimestamp(1);
% plot(diff(Gazetimestamp));
% plot(Gazetimestamp);

LPOGX_LIST = (data(:,10))*52.8;  %Multiply by screen width cm
LPOGY_LIST  = (data(:,11))*29.8; %multiply the screen height cm
LPOGV_LIST = (data(:,12)); %Conversion by screen coordinate system

LPOGZ_LIST  = zeros(length(LPOGY_LIST),1); %Multiply by screen height cm

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
LEYE_world = R*LEYE+t; % Right eye converted to world coordinate system
LPOG_world = [LPOGX_LIST,LPOGY_LIST,LPOGZ_LIST]';

LGAZE =  LPOG_world - LEYE_world;

RPOGX_LIST = (data(:,13))*52.8;
RPOGY_LIST  = (data(:,14))*29.8;
RPOGV_LIST = (data(:,15));
RPOGZ_LIST  = zeros(length(RPOGY_LIST),1);% Multiply by screen height cm


REYEX_LIST  = (100*data(:,34));
REYEY_LIST  = (100*data(:,35));
REYEZ_LIST  = (100*data(:,36));
RPUPILV_LIST = data(:,38);

REYE = [ REYEX_LIST, REYEY_LIST, REYEZ_LIST]';
REYE_world = R*REYE+t;  
RPOG_world = [RPOGX_LIST,RPOGY_LIST,RPOGZ_LIST]';

RGAZE =  RPOG_world - REYE_world;


Combine_left_right_GAZE =  (LPOG_world+ RPOG_world)/2 - (LEYE_world+REYE_world)/2;


% MEANGAZELEFT =  [mean(LPOGX_LIST),mean(LPOGY_LIST),mean(LPOGZ_LIST)] - mean(LEYE_world');
% MEANGAZERIGHT = [mean(RPOGX_LIST),mean(RPOGY_LIST),mean(RPOGZ_LIST)] - mean(REYE_world');


% RLPOGX = mean((RPOGX_LIST + LPOGX_LIST))/2;
% RLPOGY = mean((RPOGY_LIST + LPOGY_LIST))/2;
% RLPOGZ = mean((RPOGZ_LIST + LPOGZ_LIST))/2;
% RLCENTER =  mean(REYE_world' + LEYE_world')/2;
% MEANXYZ = [RLPOGX,RLPOGY,RLPOGZ];
% MEANGAZE = [RLPOGX,RLPOGY,RLPOGZ] - RLCENTER;


end