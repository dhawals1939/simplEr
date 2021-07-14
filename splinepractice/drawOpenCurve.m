clc, clear, close all

addpath('helperFunctions');

figure, 
xlim([-10, 10]);
ylim([-10, 10]);

index = 1;
x = 0;
y = 0;

% x = (2*rand(10,1) - 1) * 10;
% y = (2*rand(10,1) - 1) * 10;

while(true)
    [x(index),y(index),button] = ginput(1);
    index = index + 1;
    plot(x(index-1), y(index-1), 'r+'); hold on;
    xlim([-10, 10]);
    ylim([-10, 10]);
    if(button == 2 || button == 3)
        break;
    end 
end

cx = build1dspline(x);
cy = build1dspline(y);

step = 1e-3;
u = zeros((length(x)+2)/step, 1);
v = zeros((length(x)+2)/step, 1);

index = 1;
for t = 0:step:(length(x) + 1)
    u(index) = value1dspline(cx, t);
    v(index) = value1dspline(cy, t);
    index = index + 1;
end
u(index:end, :) = [];
v(index:end, :) = [];

plot(u, v, 'b-');hold off