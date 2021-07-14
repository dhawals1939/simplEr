clc, clear, close all
rng(1)
addpath('helperFunctions');

%% Test cases
% 1. Extrapolate and reconstruct only at nodes and check that the image is
%    mirror symmetric
% 
x = rand(10, 10, 10);

xmin  = [1 1 1];
xmax  = [size(x, 1), size(x, 2), size(x, 3)];
xres  = (xmax-xmin)./(length(x)-1);

S = Spline_inefficient(3, xmin, xmax, size(x));
S = S.build(x);

disp('built splines');

% y = zeros(size(x));
% for i=1:size(x, 1)
%     for j=1:size(x, 2)
%         for k=1:size(x, 3)
%             y(i, j, k) = S.value([i, j, k],[0, 0, 0]);
%         end
%     end
% end
% sum(abs(x - y), 'all')


x = [x(:, end-1:-1:2, :) x x(:, end-1:-1:2, :)];
x = [x(end-1:-1:2, :, :);x;x(end-1:-1:2, :, :)];
x = cat(3, x(:, :, end-1:-1:2),x,x(:, :, end-1:-1:2));
y = zeros(size(x));
for i=1-size(x, 1):2*size(x, 1)-1
    for j=1-size(x, 2):2*size(x, 2)-1
        for k=1-size(x, 3):2*size(x, 3)-1
            y(i+254,j+254, k+254) = S.value([i, j, k],[0, 0, 0]);
        end
    end
end

sum(abs(x - y), 'all')

%% Test 2: analytical function and derivatives

% t = -1:.02:1;
% u = -1:.02:1;
% v = -1:.02:1;
% 
% [U, T, V] = meshgrid(u, t, v);
% 
% x   = 10*T.*sin(2*V) + 20*sin(V).*cos(U) + 5 * sin(T);
% 
% xd  = 10*sin(2*V) + 5 * cos(T);
% xd2 = -5*sin(T);
% yd  = -20*sin(V).*sin(U);
% yd2 = -20*sin(V).*cos(U);
% zd  = 20*T.*cos(2*V) + 20*cos(V).*cos(U);
% zd2 =-40*T.*sin(2*V) - 20*sin(V).*cos(U); 
% xdyd= zeros(size(T)); 
% ydzd= -20*cos(V).*sin(U);
% xdzd= 20*cos(2*V);
% 
% xmin  = [t(1)   u(1)   v(1)];
% xmax  = [t(end) u(end) v(end)];
% xres  = (xmax-xmin)./([length(t) length(u) length(v)]-1);
% 
% S = Spline(3, xmin, xmax, size(x));
% S = S.build(x);
% 
% sp = zeros(size(x));
% 
% sp_xd   = zeros(size(x));
% sp_xd2  = zeros(size(x));
% sp_yd   = zeros(size(x));
% sp_yd2  = zeros(size(x));
% sp_zd   = zeros(size(x));
% sp_zd2  = zeros(size(x));
% 
% sp_xdyd = zeros(size(x));
% sp_ydzd = zeros(size(x));
% sp_xdzd = zeros(size(x));
% 
% for i=1:length(t)
%     for j=1:length(u)
%         for k=1:length(v)
%             sp(i, j, k)     = S.value([t(i), u(j), v(k)],[0, 0, 0]);
%             
%             sp_xd(i, j, k)  = S.value([t(i), u(j), v(k)],[1, 0, 0]);
%             sp_xd2(i, j, k) = S.value([t(i), u(j), v(k)],[2, 0, 0]);
%             sp_yd(i, j, k)  = S.value([t(i), u(j), v(k)],[0, 1, 0]);
%             sp_yd2(i, j, k) = S.value([t(i), u(j), v(k)],[0, 2, 0]);
%             sp_zd(i, j, k)  = S.value([t(i), u(j), v(k)],[0, 0, 1]);
%             sp_zd2(i, j, k) = S.value([t(i), u(j), v(k)],[0, 0, 2]);
%             
%             sp_xdyd(i, j, k) = S.value([t(i), u(j), v(k)],[1, 1, 0]);
%             sp_ydzd(i, j, k) = S.value([t(i), u(j), v(k)],[0, 1, 1]);
%             sp_xdzd(i, j, k) = S.value([t(i), u(j), v(k)],[1, 0, 1]);
%         end
%     end
% end
% mean(mean(mean(abs(x(3:end-2, 3:end-2, 3:end-2)  - sp(3:end-2, 3:end-2, 3:end-2)),     'all')))
% 
% mean(mean(mean(abs(xd(3:end-2, 3:end-2, 3:end-2) - sp_xd(3:end-2, 3:end-2, 3:end-2)),  'all')))
% mean(mean(mean(abs(xd2(3:end-2, 3:end-2, 3:end-2)- sp_xd2(3:end-2, 3:end-2, 3:end-2)), 'all')))
% mean(mean(mean(abs(yd(3:end-2, 3:end-2, 3:end-2) - sp_yd(3:end-2, 3:end-2, 3:end-2)),  'all')))
% mean(mean(mean(abs(yd2(3:end-2, 3:end-2, 3:end-2)- sp_yd2(3:end-2, 3:end-2, 3:end-2)), 'all')))
% mean(mean(mean(abs(zd(3:end-2, 3:end-2, 3:end-2) - sp_zd(3:end-2, 3:end-2, 3:end-2)),  'all')))
% 
% mean(mean(mean(abs(xdyd(3:end-2, 3:end-2, 3:end-2) - sp_xdyd(3:end-2, 3:end-2, 3:end-2)), 'all')))
% mean(mean(mean(abs(ydzd(3:end-2, 3:end-2, 3:end-2) - sp_ydzd(3:end-2, 3:end-2, 3:end-2)), 'all')))
% mean(mean(mean(abs(xdzd(3:end-2, 3:end-2, 3:end-2) - sp_xdzd(3:end-2, 3:end-2, 3:end-2)), 'all')))
