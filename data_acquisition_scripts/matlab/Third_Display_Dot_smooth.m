clc;clear;close all
%生成新的figure
disp('third')
hold on
axis([ 0 100 0 100])
set(gca,'YLim',[0 100]);%X轴的数据显示范围
set(gca,'XLim',[0 100]);%X轴的数据显示范围
set(gcf,'MenuBar','none');
% axis off;
box on;
set(gcf,'color','black');
colordef black;
% set(gca,'LooseInset',get(gca,'TightInset'))
set(gca, 'LooseInset', [0,0,0,0]);

x_list = [0:5:100]';
y_list = [0:5:100]';

for j = 1:1:21
    if (mod(j,2)==0)
        k_list = 1:1:21;
    else
        k_list = 21:-1:1;
    end
        i=1;
    for k = k_list
        
xy_mat((j-1)*21+k,:) = [x_list(j);y_list(i)];
i = i+1;
    end
end


% plot(xy_mat(:,1), xy_mat(:,2))
% plot(xy_mat(:,2), xy_mat(:,1))


% 
% Calibration_point = [50,50]
% Calibration_point = [0,0;
%                      0,100;    
%                      100,0
%                      100,100
%                      50,50];

% xy_mat_new = [Calibration_point;xy_mat];
xy_mat_new = [xy_mat];
x = 50;
y = 50;
h= plot( x,y , 'Marker','o', 'MarkerEdgeColor','r','LineWidth', 60);
m= plot( x,y , 'Marker','o', 'MarkerEdgeColor','k','LineWidth', 40);
s= plot( x,y , 'Marker','o', 'MarkerEdgeColor','r','LineWidth', 20);
% kk = text(x,y,'准备开始','color','w','FontSize',50)



% for n=1:5
% set(h,'xdata',xy_mat_new(n,1),'ydata',xy_mat_new(n,2));
% set(m,'xdata',xy_mat_new(n,1),'ydata',xy_mat_new(n,2));
% set(s,'xdata',xy_mat_new(n,1),'ydata',xy_mat_new(n,2));
% pause(1.5)
% end

n=1
set(h,'xdata',xy_mat_new(n,2),'ydata',xy_mat_new(n,1));
set(m,'xdata',xy_mat_new(n,2),'ydata',xy_mat_new(n,1));
set(s,'xdata',xy_mat_new(n,2),'ydata',xy_mat_new(n,1));
pause(5)

tic
for n=1:441
set(h,'xdata',xy_mat_new(n,2),'ydata',xy_mat_new(n,1));
set(m,'xdata',xy_mat_new(n,2),'ydata',xy_mat_new(n,1));
set(s,'xdata',xy_mat_new(n,2),'ydata',xy_mat_new(n,1));
pause(0.1)
end


for n=1:441
set(h,'xdata',xy_mat_new(n,1),'ydata',xy_mat_new(n,2));
set(m,'xdata',xy_mat_new(n,1),'ydata',xy_mat_new(n,2));
set(s,'xdata',xy_mat_new(n,1),'ydata',xy_mat_new(n,2));
pause(0.1)
end
toc


% set(h,'xdata',xy_mat_new(n,1),'ydata',xy_mat_new(n,2));
% set(m,'xdata',xy_mat_new(n,1),'ydata',xy_mat_new(n,2));
% set(s,'xdata',xy_mat_new(n,1),'ydata',xy_mat_new(n,2));
