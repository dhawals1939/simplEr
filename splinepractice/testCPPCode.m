clc, clear, close all

rng(1)
%% Test-1: Kernels
% t = -3:.01:3;
% x = 10*sin(2*t).^2 + 20*sin(3*t).*cos(t);
% xd  = 40*sin(2*t).*cos(2*t) + 60*cos(3*t).*cos(t) - 20*sin(3*t).*sin(t);
% xd2 = 80*cos(2*t).*cos(2*t) - 80*sin(2*t).*sin(2*t) - 180*sin(3*t).*cos(t) - 60 * cos(3*t).* sin(t) - 60*cos(3*t).*sin(t) - 20*sin(3*t).*cos(t);
% xmin  = t(1);
% xmax  = t(end);
% xres  = (xmax-xmin)/(length(x)-1);
% 
% S = Spline(1, xmin, xmax, length(x));
% S = S.build(x);
% figure, 
% I = csvread('kernel.csv');
% 
% y = S.kernel(t);
% sum(abs(y' - I(1:601,2)), 'all')
% subplot(1, 3, 1);
% plot(t, y);hold on;
% plot(t, I(1:601,2)); hold on;
% plot(t, (y' - I(1:601,2)));
% 
% yd = S.deriv_kernel(t);
% sum(abs(yd' - I(602:1202,2)), 'all')
% subplot(1, 3, 2);
% plot(t, yd);hold on;
% plot(t, I(602:1202,2)); hold on;
% plot(t, (yd' - I(602:1202,2)));
% 
% yd2= S.deriv2_kernel(t);
% sum(abs(yd2' - I(1203:1803,2)), 'all')
% subplot(1, 3, 3);
% plot(t, yd2);hold on;
% plot(t, I(1203:1803,2)); hold on;
% plot(t, (yd2' - I(1203:1803,2)));

%% Test-2: Interpolation

x = [1 2 4 2 3 1 1 2 3 1 -2 1 2 3];
xmin  = sqrt(13);
xmax  = sqrt(803);
xres  = (xmax-xmin)/(length(x)-1);

S = Spline(1, xmin, xmax, length(x));
S = S.build(x);

step  = .01;
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

I   = csvread('interpolate1d.csv');
v   = I(1:length(I)/3);
vd  = I(length(I)/3+1:2*length(I)/3);
vd2 = I(2*length(I)/3+1:length(I));

figure, 
subplot(1, 3, 1); plot(T, v);   hold on;plot(T, y);   hold off;
subplot(1, 3, 2); plot(T, vd);  hold on;plot(T, yad); hold off;
subplot(1, 3, 3); plot(T, vd2); hold on;plot(T, yad2);hold off;

%% Test-3: 2D-Interpolation
% d1 = 1:20;
% d2 = -7:7;
% 
% x = d1'*d2;
% N = size(x);
% x = x(:);
% 
% xmin  = [1 1];
% xmax  = [41 41];
% xres  = (xmax-xmin)./(length(x)-1);
% 
% S = Spline(2, xmin, xmax, [length(d1) length(d2)]);
% S = S.build(x);
% 
% tx    = -30:2.5:30;
% ty    = -30:2.5:30;
% Y     = zeros(length(tx), length(ty));
% Yxd   = zeros(size(Y));
% Yxd2  = zeros(size(Y));
% Yyd   = zeros(size(Y));
% Yyd2  = zeros(size(Y));
% Yxdyd = zeros(size(Y));
% 
% for index_x=1:length(tx)
%     for index_y=1:length(ty)
%         P     = [tx(index_x) ty(index_y)];
%         Y(index_x, index_y)      = S.value(P, [0 0]); 
% 
%         Yxd(index_x, index_y)    = S.value(P, [1 0]); 
%         Yxd2(index_x, index_y)   = S.value(P, [2 0]); 
%         Yyd(index_x, index_y)    = S.value(P, [0 1]); 
%         Yyd2(index_x, index_y)   = S.value(P, [0 2]); 
%         Yxdyd(index_x, index_y)  = S.value(P, [1 1]); 
%     end
% end
% 
% 
% I   = csvread('interpolate2d.csv');
% I(:, end) = [];
% v   = I(1:length(tx), :);
% I(1:length(tx), :) = [];
% vxd = I(1:length(tx), :);
% I(1:length(tx), :) = [];
% vxd2= I(1:length(tx), :);
% I(1:length(tx), :) = [];
% vyd = I(1:length(tx), :);
% I(1:length(tx), :) = [];
% vyd2= I(1:length(tx), :);
% I(1:length(tx), :) = [];
% vxdyd= I(1:length(tx), :);
% 
% sum(abs(v     - Y),    'all')
% sum(abs(vxd   - Yxd),  'all')
% sum(abs(vxd2  - Yxd2), 'all')
% sum(abs(vyd   - Yyd), 'all')
% sum(abs(vyd2  - Yyd2), 'all')
% sum(abs(vxdyd - Yxdyd), 'all')


%% Test-3: 3D-Interpolation
% x = rand(10, 10, 10);
% xmin  = [1 1 1];
% xmax  = [size(x, 1), size(x, 2), size(x, 3)];
% xres  = (xmax-xmin)./(length(x)-1);
% 
% S = Spline(3, xmin, xmax, size(x));
% S = S.build(x);
% 
% c = zeros(size(x,1)*size(x,3), size(x,2));
% for k=1:size(x, 3)
%     c((k-1)*size(x, 1)+1:k*size(x, 1), :) = S.coeff(:, :, k);
% end
% 
% % I   = csvread('coefficients3d.csv');
% % I(:,end)=[];
% % sum(abs(I-c), 'all')
% 
% tx =  1:1.5:12;
% ty =  1:1.5:12;
% tz = -1:1.5:12;
% 
% Y     = zeros(length(tx), length(ty), length(tz));
% 
% Yxd   = zeros(size(Y));
% Yxd2  = zeros(size(Y));
% Yyd2  = zeros(size(Y));
% Yzd   = zeros(size(Y));
% Yzdyd = zeros(size(Y));
% Yzdxd = zeros(size(Y));
% 
% 
% for index_x=1:length(tx)
%     for index_y=1:length(ty)
%         for index_z=1:length(tz)
%             P     = [tx(index_x) ty(index_y) tz(index_z)];
%             Y(index_x, index_y, index_z)      = S.value(P, [0 0 0]); 
% 
%             Yxd(index_x, index_y, index_z)    = S.value(P, [1 0 0]); 
%             Yxd2(index_x, index_y, index_z)   = S.value(P, [2 0 0]); 
%             Yyd2(index_x, index_y, index_z)   = S.value(P, [0 2 0]); 
%             Yzd(index_x, index_y, index_z)    = S.value(P, [0 0 1]); 
%             Yzdyd(index_x, index_y, index_z)  = S.value(P, [0 1 1]); 
%             Yzdxd(index_x, index_y, index_z)  = S.value(P, [1 0 1]); 
%         end
%     end
% end
% 
% Y = Y(:);
% 
% Yxd = Yxd(:);
% Yxd2 = Yxd2(:);
% Yyd2 = Yyd2(:);
% Yzd = Yzd(:);
% Yzdyd = Yzdyd(:);
% Yzdxd = Yzdxd(:);
% 
% I   = csvread('interpolate3d.csv');
% 
% v   = I(1:7:end);
% 
% 
% vxd = I(2:7:end);
% 
% vxd2= I(3:7:end);
% 
% vyd2= I(4:7:end);
% 
% vzd = I(5:7:end);
% 
% vzdyd= I(6:7:end);
% 
% vzdxd= I(7:7:end);
% 
% sum(abs(Y- v), 'all')
% 
% sum(abs(Yxd- vxd), 'all')
% sum(abs(Yxd2- vxd2), 'all')
% sum(abs(Yyd2- vyd2), 'all')
% sum(abs(Yzd- vzd), 'all')
% sum(abs(Yzdyd- vzdyd), 'all')
% sum(abs(Yzdxd- vzdxd), 'all')
% 
