clc, clear, close all

addpath('helperFunctions');

% Test 1: Random values
x = [1 2 4 2 3 1 1 2 3 1 -2 1 2 3];
% x = rand(1, 15);
xmin  = sqrt(13);
xmax  = sqrt(803);
xres  = (xmax-xmin)/(length(x)-1);

S = Spline(1, xmin, xmax, length(x));
S = S.build(x);

step  = .001;
T   = -30:step:length(x)+30;
y   = zeros(size(T));
yad = zeros(size(T));
yad2= zeros(size(T));
index = 1;
for t=T
%     y(index)    = S.value(t, @S.kernel);
%     yad(index)  = S.value(t, @S.deriv_kernel);
%     yad2(index) = S.value(t, @S.deriv2_kernel);
    y(index)    = S.value(t, 0);
    yad(index)  = S.value(t, 1);
    yad2(index) = S.value(t, 2);
    index = index + 1;
end

yd  = gradient(y , step/xres);
yd2 = gradient(yd, step/xres);

xd  = gradient(x,  1);
xd2 = gradient(xd, 1);

xAxis = linspace(xmin, xmax, length(x));
figure, 
subplot(1, 3, 1);
plot(T, y); hold on;
plot(xAxis, x, 'ro'); 

subplot(1, 3, 2);
plot(T, yad); hold on;
plot(T, yd); hold on;
plot(xAxis, xd, 'ro');

subplot(1, 3, 3);
plot(T, yad2); hold on;
plot(T, yd2); hold on;
plot(xAxis, xd2, 'ro');

%% Test 2: analytical function and derivatives

% t = -3:.1:3;
% x = 10*sin(2*t).^2 + 20*sin(3*t).*cos(t);
% xd  = 40*sin(2*t).*cos(2*t) + 60*cos(3*t).*cos(t) - 20*sin(3*t).*sin(t);
% xd2 = 80*cos(2*t).*cos(2*t) - 80*sin(2*t).*sin(2*t) - 180*sin(3*t).*cos(t) - 60 * cos(3*t).* sin(t) - 60*cos(3*t).*sin(t) - 20*sin(3*t).*cos(t);
% xmin  = t(1);
% xmax  = t(end);
% xres  = (xmax-xmin)/(length(x)-1);
% 
% S = Spline(1, xmin, xmax, length(x));
% S = S.build(x);
% 
% step  = .01;
% T   = -10:step:10;
% y   = zeros(size(T));
% yad = zeros(size(T));
% yad2= zeros(size(T));
% index = 1;
% for t=T
% %     y(index)    = S.value(t, @S.kernel);
% %     yad(index)  = S.value(t, @S.deriv_kernel);
% %     yad2(index) = S.value(t, @S.deriv2_kernel);
%     y(index)    = S.value(t, 0);
%     yad(index)  = S.value(t, 1);
%     yad2(index) = S.value(t, 2);
%     index = index + 1;
% end
% 
% yd  = gradient(y , step);
% yd2 = gradient(yd, step);
% 
% % xd  = gradient(x,  1);
% % xd2 = gradient(xd, 1);
% 
% 
% xAxis = linspace(xmin, xmax, length(x));
% figure, 
% subplot(1, 3, 1);
% plot(T, y); hold on;
% plot(xAxis, x, 'ro'); 
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
% mean(abs( (yad(3:end-2)  - yd(3:end-2))./(yad(3:end-2)+1e-3)))
% mean(abs( (yad2(3:end-2) - yd2(3:end-2))./(yad(3:end-2)+1e-3)))

% Test 3: Just a single spline at a high resolution
% 
% step = 1e-6;
% t = -4:step:4;
% 
% S = Spline(1, t(1), t(end), length(t));
% 
% y    = S.kernel(t);
% yad  = S.deriv_kernel(t);
% yad2 = S.deriv2_kernel(t);
% 
% yd   = gradient(y, step);
% yd2  = gradient(yd, step);
% 
% figure, 
% subplot(1, 3, 1);
% plot(t, y); hold on;
% 
% subplot(1, 3, 2);
% plot(t, yad); hold on;
% plot(t, yd); hold off;
% 
% subplot(1, 3, 3);
% plot(t, yad2); hold on;
% plot(t, yd2); hold off;
% 
% sum(abs(yad(3:end-2)   - yd(3:end-2) ))
% sum(abs(yad2(3:end-2)  - yd2(3:end-2)))
% mean(abs(yad(3:end-2)  - yd(3:end-2) ))
% mean(abs(yad2(3:end-2) - yd2(3:end-2)))
% 
% 
