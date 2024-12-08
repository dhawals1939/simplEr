clc, clear, close all
% rng(1)
addpath('helperFunctions');

%% Test cases
% 1. symmetric matrix, coefficients should be symmetric
% 2. For a random matrix, reconstruct a row and a column at high res and
%    make sure the nodes are exact. Also make sure the numerical
%    derivatives of high res match the interpolation gradients
% 3. Extrapolate and reconstruct only at nodes and check that the image is
%    mirror symmetric

%% Test 1
% d1 = rand(15, 1)*10;
% 
% x = d1*d1';
% 
% 
% xmin  = [sqrt(13) sqrt(13)];
% xmax  = [sqrt(803) sqrt(803)];
% xres  = (xmax-xmin)./(length(x)-1);
% 
% S = Spline(2, xmin, xmax, size(x));
% S = S.build(x);
% S.coeff

%% Test 2 Part-1 (with rows)
% d1 = rand(15, 1)*10;
% d2 = rand(15, 1)*10;
% 
% x = d1*d2';
% 
% 
% xmin  = [sqrt(13) sqrt(13)];
% xmax  = [sqrt(803) sqrt(803)];
% xres  = (xmax-xmin)./(length(x)-1);
% 
% S = Spline(2, xmin, xmax, size(x));
% S = S.build(x);
% 
% % Pick a row and a column
% r = randi(size(x, 1), 1);
% 
% step  = .001;
% T   = -30:step:size(x,2)+30;
% y   = zeros(size(T));
% yad = zeros(size(T));
% yad2= zeros(size(T));
% index = 1;
% convertedr = (r - 1)/(size(x,1) - 1) * (xmax(1) - xmin(1)) + xmin(1);
% for t=T
%     y(index)    = S.value([convertedr, t], [0 0]);    
%     yad(index)  = S.value([convertedr, t], [0 1]);
%     yad2(index) = S.value([convertedr, t], [0 2]);
%     index = index + 1;
% end
% 
% yd  = gradient(y , step/xres(2));
% yd2 = gradient(yd, step/xres(2));
% 
% xd  = gradient(x(r, :),  1);
% xd2 = gradient(xd, 1);
% 
% xAxis = linspace(xmin(2), xmax(2), size(x,2));
% figure, 
% subplot(1, 3, 1);
% plot(T, y); hold on;
% plot(xAxis, x(r, :), 'ro'); 
% 
% subplot(1, 3, 2);
% plot(T, yad); hold on;
% plot(T, yd); hold on;
% plot(xAxis, xd, 'ro');
% 
% subplot(1, 3, 3);
% plot(T, yad2); hold on;
% plot(T, yd2); hold on;
% plot(xAxis, xd2, 'ro');


%% Test 2 Part-2 (with columns)
d1 = rand(15, 1)*10;
d2 = rand(20, 1)*10;

x = d1*d2';


xmin  = [sqrt(13) sqrt(803)];
xmax  = [sqrt(803) sqrt(2050)];
xres  = (xmax-xmin)./(length(x)-1);

S = Spline(2, xmin, xmax, size(x));
S = S.build(x);

% Pick a column
c = randi(size(x, 2), 1);

step  = .01;
T   = -30:step:size(x,2)+30;
y   = zeros(size(T));
yad = zeros(size(T));
yad2= zeros(size(T));
index = 1;
convertedc = (c - 1)/(size(x,2) - 1) * (xmax(2) - xmin(2)) + xmin(2);
for t=T
    y(index)    = S.value([t, convertedc], [0 0]);    
    yad(index)  = S.value([t, convertedc], [1 0]);
    yad2(index) = S.value([t, convertedc], [2 0]);
    index = index + 1;
end

yd  = gradient(y , step/xres(1));
yd2 = gradient(yd, step/xres(1));



xd  = gradient(x(:, c),  1);
xd2 = gradient(xd, 1);

xAxis = linspace(xmin(1), xmax(1), size(x,1));
figure, 
subplot(1, 3, 1);
plot(T, y); hold on;
plot(xAxis, x(:, c), 'ro'); 

subplot(1, 3, 2);
plot(T, yad); hold on;
plot(T, yd); hold on;
plot(xAxis, xd, 'ro');

subplot(1, 3, 3);
plot(T, yad2); hold on;
plot(T, yd2); hold on;
plot(xAxis, xd2, 'ro');


%% Test 3 (Only at knots)
% x = double(imread('cameraman.tif'));
% 
% xmin  = [1 1];
% xmax  = [size(x, 1), size(x, 2)];
% xres  = (xmax-xmin)./(length(x)-1);
% 
% S = Spline(2, xmin, xmax, size(x));
% S = S.build(x);
% 
% disp('built splines');
% 
% x = [x(:, end-1:-1:2) x x(:, end-1:-1:2)];
% x = [x(end-1:-1:2, :);x;x(end-1:-1:2, :)];
% 
% y = zeros(size(x));
% 
% % for i=1:256
% %     for j=1:256
% %         y(i,j) = S.value([i, j],[0, 0]);
% %     end
% % end
% for i=1-254:256+254
%     for j=1-254:256+254
%         y(i+254,j+254) = S.value([i, j],[0, 0]);
%     end
% end
% 
% figure, imshow(x,[]);
% figure, imshow(y,[]);
% sum(abs(x - y), 'all')

%% Test 4: Analytical derivatives

% t = -1:.02:1;
% u = -1:.02:1;
% 
% [U, T] = meshgrid(u, t);
% 
% x   = 3*cos(U).*sin(T);
% 
% xd  = 3*cos(U).*cos(T);
% xd2 =-3*cos(U).*sin(T);
% yd  =-3*sin(U).*sin(T);
% yd2 =-3*cos(U).*sin(T);
% xdyd=-3*sin(U).*cos(T);
% 
% xmin  = [t(1)   u(1)  ];
% xmax  = [t(end) u(end)];
% xres  = (xmax-xmin)./([length(t) length(u)]-1);
% 
% S = Spline(2, xmin, xmax, size(x));
% S = S.build(x);
% 
% sp = zeros(size(x));
% 
% sp_xd   = zeros(size(x));
% sp_xd2  = zeros(size(x));
% sp_yd   = zeros(size(x));
% sp_yd2  = zeros(size(x));
% sp_xdyd = zeros(size(x));
% 
% for i=1:length(t)
%     for j=1:length(u)
%             sp(i, j)     = S.value([t(i), u(j)],[0, 0]);
%             
%             sp_xd(i, j)  = S.value([t(i), u(j)],[1, 0]);
%             sp_xd2(i, j) = S.value([t(i), u(j)],[2, 0]);
%             sp_yd(i, j)  = S.value([t(i), u(j)],[0, 1]);
%             sp_yd2(i, j) = S.value([t(i), u(j)],[0, 2]);
%             sp_xdyd(i, j)= S.value([t(i), u(j)],[1, 1]);
%     end
% end
% 
% mean(abs(x(3:end-2, 3:end-2)  - sp(3:end-2, 3:end-2)), 'all')
% mean(abs((xd(3:end-2, 3:end-2)  - sp_xd(3:end-2, 3:end-2))./xd(3:end-2, 3:end-2)), 'all')
