classdef Spline
    properties
        xmin
        xmax
        xres
        N
        coeff % Internal and allocated at the time of built
        dim   % dimension: Currently coding only for 1D
        built % boolean
        z1 = -2 + sqrt(3);

    end
%     methods (Access = private)
    methods
        function y = convertToX(obj, x)
            if(size(obj.xmin) ~= size(x))
                x = x';
            end
            if(size(obj.xmin) ~= size(x))
                disp('wrong dimensions of x');
            end

            y = (x - obj.xmin)./(obj.xmax - obj.xmin) .* (obj.N - 1) + 1;
        end
        function y = kernel(obj, x)
            y = +  ((x+2)>0).*(x+2).^3  ... 
                -4*((x+1)>0).*(x+1).^3  ... 
                +6*((x)>0)  .*(x).^3  ... 
                -4*((x-1)>0).*(x-1).^3  ... 
                +  ((x-2)>0).*(x-2).^3;
            y = y/6;
        end
        function y = deriv_kernel(obj, x)
            y = +  ((x+2)>0).*(x+2).^2  ... 
                -4*((x+1)>0).*(x+1).^2  ... 
                +6*((x)>0)  .*(x).^2  ... 
                -4*((x-1)>0).*(x-1).^2  ... 
                +  ((x-2)>0).*(x-2).^2;
            y = y/2;
        end
        function y = deriv2_kernel(obj, x)
            y = +  ((x+2)>0).*(x+2)  ... 
                -4*((x+1)>0).*(x+1)  ... 
                +6*((x)>0)  .*(x)  ... 
                -4*((x-1)>0).*(x-1)  ... 
                +  ((x-2)>0).*(x-2);
        end
        
        function y = optkernel(obj, x)
            x = abs(x);
            y = (x < 2 & x > 1).* (1/6).*(2-x).^3 +  ...
                (x <=1 ).* (2/3 + (.5.*x - 1).*x.*x);
        end
        function y = optderiv_kernel(obj, x)
            s = sign(x);
            x = abs(x);
            y = (x < 2 & x > 1).*-(1/2).*(2-x).^2 +  ...
                (x <=1 ).* ((1.5.*x - 2).*x);
            y = s.*y;
        end
        function y = optderiv2_kernel(obj, x)
            x = abs(x);
            y = (x < 2 & x > 1).*(2-x) +  ...
                (x <=1 ).* (3.*x - 2);
        end
        
        
        function c = build1D(obj, s, offset, stride, N)            
            cp = zeros(N, 1);
            cn = zeros(N, 1);

            cp(1) = 0;

            for i=1:N
                cp(1) = cp(1) + s(offset + (i-1)*stride) * obj.z1^(i-1);
            end

            for i=N-1:-1:2
                cp(1) = cp(1) + s(offset + (i-1)*stride) * obj.z1^(2*N - 1 - i);
            end

            cp(1) = cp(1)/(1-obj.z1^(2*N - 2));

            for i=2:N
                cp(i) = s(offset + (i-1)*stride) + obj.z1 * cp(i-1);
            end

            cn(N) = obj.z1/(obj.z1*obj.z1-1) * (cp(N) + obj.z1 * cp(N-1));

            for i=N-1:-1:1
                cn(i) = obj.z1 * (cn(i+1) - cp(i));
            end

            c = 6*cn;
        end
        
        function c = build2D(obj, s, N)  
            
            c = zeros(size(s));
            % Build along rows
            for i=1:N(1)
                c(i:N(1):i+N(1)*(N(2)-1)) = obj.build1D(s, i, N(1), N(2));
            end
            % Build along columns
            for j=1:N(2)
                c((j-1)*N(1)+1:1:j*N(1)) = obj.build1D(c, (j-1)*N(1)+1, 1, N(1));
            end
        end
        
        function c = build3D(obj, s, N)   
            c = zeros(size(s));
            for k=1:N(3)
                for i=1:N(1)
                    c(i, :, k) = obj.build1D(s, (k-1)*N(1)*N(2) + i, N(1), N(2));
                end
            end
            for k=1:N(3)
                for j=1:N(2)
                    c(:, j, k) = obj.build1D(c, (k-1)*N(1)*N(2) + (j-1)*N(1) + 1, 1, N(1));
                end
            end
            for i=1:N(1)
                for j=1:N(2)
                    c(i, j, :) = obj.build1D(c, (j-1)*N(1) + i, N(1)*N(2), N(3));
                end
            end 
        end
        function v = value1d(obj, x, deriv_axes)
            x = obj.convertToX(x);
            v = 0;
            if(~obj.built)
                return;
            end
            for index=ceil(x-2):floor(x+2)
%                 index
                wrap_index = mod(index-1, 2*obj.N - 2) + 1;
%                 wrap_index
                if(wrap_index > obj.N)
                    wrap_index  = 2*obj.N - wrap_index;
                end
%                 wrap_index
                K = 1;
                if(deriv_axes(1) == 0)
                    K = K*obj.kernel(x - index);
                elseif(deriv_axes(1) == 1)
                    K = K*obj.deriv_kernel(x - index)*obj.xres;
                elseif(deriv_axes(1) == 2)
                    K = K*obj.deriv2_kernel(x - index)*obj.xres*obj.xres;
                end         
                v = v + obj.coeff(wrap_index) * K;
            end
        end
        
        % deriv_axes encoded which kernel to use for each dimension
        % [0 0] -> normal kernel in each dimension
        % [0 1] -> normal kernel in x dimension and derivative kernel in y
        % [0 2] -> normal kernel in x dimension and double-derivative kernel in y
        % [1 1] -> derivative kernel in both dimensions
        
        function v = value2d(obj, x, deriv_axes) 
            x = obj.convertToX(x);
            v = 0;
            if(~obj.built)
                return;
            end
            for index1=ceil(x(1)-2):floor(x(1)+2)
                wrap_index1= mod(index1-1, 2*obj.N(1) - 2) + 1;
                if(wrap_index1 > obj.N(1))
                    wrap_index1  = 2*obj.N(1) - wrap_index1;
                end
                for index2=ceil(x(2)-2):floor(x(2)+2)
                    wrap_index2= mod(index2-1, 2*obj.N(2) - 2) + 1;
                    if(wrap_index2 > obj.N(2))
                        wrap_index2  = 2*obj.N(2) - wrap_index2;
                    end
                    K = 1;
                    if(deriv_axes(1) == 0)
                        K = K*obj.kernel(x(1) - index1);
                    elseif(deriv_axes(1) == 1)
                        K = K*obj.deriv_kernel(x(1) - index1)*obj.xres(1);
                    elseif(deriv_axes(1) == 2)
                        K = K*obj.deriv2_kernel(x(1) - index1)*obj.xres(1)*obj.xres(1);
                    end                 
                    if(deriv_axes(2) == 0)
                        K = K*obj.kernel(x(2) - index2);
                    elseif(deriv_axes(2) == 1)
                        K = K*obj.deriv_kernel(x(2) - index2)*obj.xres(2);
                    elseif(deriv_axes(2) == 2)
                        K = K*obj.deriv2_kernel(x(2) - index2)*obj.xres(2)*obj.xres(2);
                    end                        
                    v = v + obj.coeff(wrap_index1+(wrap_index2-1)*obj.N(1)) * K;
                end
            end
        end
        
                % deriv_axes encoded which kernel to use for each dimension
        % [0 0] -> normal kernel in each dimension
        % [0 1] -> normal kernel in x dimension and derivative kernel in y
        % [0 2] -> normal kernel in x dimension and double-derivative kernel in y
        % [1 1] -> derivative kernel in both dimensions
        
        function v = value3d(obj, x, deriv_axes) 
            x = obj.convertToX(x);
            v = 0;
            if(~obj.built)
                return;
            end
            for index1=ceil(x(1)-2):floor(x(1)+2)
                wrap_index1= mod(index1-1, 2*obj.N(1) - 2) + 1;
                if(wrap_index1 > obj.N(1))
                    wrap_index1  = 2*obj.N(1) - wrap_index1;
                end
                for index2=ceil(x(2)-2):floor(x(2)+2)
                    wrap_index2= mod(index2-1, 2*obj.N(2) - 2) + 1;
                    if(wrap_index2 > obj.N(2))
                        wrap_index2  = 2*obj.N(2) - wrap_index2;
                    end                    
                    for index3=ceil(x(3)-2):floor(x(3)+2)
                        wrap_index3= mod(index3-1, 2*obj.N(3) - 2) + 1;
                        if(wrap_index3 > obj.N(3))
                            wrap_index3  = 2*obj.N(3) - wrap_index3;
                        end
                        
                        K = 1;
                        if(deriv_axes(1) == 0)
                            K = K*obj.kernel(x(1) - index1);
                        elseif(deriv_axes(1) == 1)
                            K = K*obj.deriv_kernel(x(1) - index1)*obj.xres(1);
                        elseif(deriv_axes(1) == 2)
                            K = K*obj.deriv2_kernel(x(1) - index1)*obj.xres(1)*obj.xres(1);
                        end                 
                        if(deriv_axes(2) == 0)
                            K = K*obj.kernel(x(2) - index2);
                        elseif(deriv_axes(2) == 1)
                            K = K*obj.deriv_kernel(x(2) - index2)*obj.xres(2);
                        elseif(deriv_axes(2) == 2)
                            K = K*obj.deriv2_kernel(x(2) - index2)*obj.xres(2)*obj.xres(2);
                        end                        
                        if(deriv_axes(3) == 0)
                            K = K*obj.kernel(x(3) - index3);
                        elseif(deriv_axes(3) == 1)
                            K = K*obj.deriv_kernel(x(3) - index3)*obj.xres(3);
                        elseif(deriv_axes(3) == 2)
                            K = K*obj.deriv2_kernel(x(3) - index3)*obj.xres(3)*obj.xres(3);
                        end                        
                        v = v + obj.coeff(wrap_index1+(wrap_index2-1)*obj.N(1)+(wrap_index3-1)*obj.N(1)*obj.N(2)) * K;
                    end
                end
            end
        end
%% using function pointer makes the code compact but slow by > 10X :( 
%         function v = value(obj, x, func)
%             x = obj.convertToX(x);
%             v = 0;
%             if(~obj.built)
%                 return;
%             end
%             for index=ceil(x-2):floor(x+2)
%                 wrap_index = mod(index-1, 2*obj.N - 2) + 1;
%                 if(wrap_index > obj.N)
%                     wrap_index  = 2*obj.N - wrap_index;
%                 end
% %                 v = v + obj.coeff(wrap_index) * feval(str2func(func), (x - index));
%                 v = v + obj.coeff(wrap_index) * func(x - index);
%             end
%         end
    end
    methods
        function obj = Spline(dim, xmin, xmax, N)
            obj.z1 = -2 + sqrt(3);
            obj.built = false;
            obj.dim = dim;
            obj.xmin = xmin;
            obj.xmax = xmax;
            obj.N = N;
            %Make sure dimensions of xmin, xmax, xres, N match: Make them
            %all xmin dimensions
            if(size(obj.xmax) ~= size(obj.xmin))
                obj.xmax = obj.xmax';
            end
            if(size(obj.N) ~= size(obj.xmin))
                obj.N = obj.N';
            end
            if(min(size(obj.xmax) ~= size(obj.xmin)) || min(size(obj.N) ~= size(obj.xmin)))
                disp('spline construction is not valid, wrong input dimensions');
            end
            obj.xres = (obj.N-1)./(obj.xmax - obj.xmin); 
            
        end
        function v = value(obj, x, deriv_axes)
            if(obj.dim == 1)
                v = obj.value1d(x, deriv_axes);
            elseif(obj.dim == 2)
                v = obj.value2d(x, deriv_axes);
            elseif(obj.dim == 3)
                v = obj.value3d(x, deriv_axes);
            else
                disp('Error: Current implementation of spline interpolation works only upto 3 dimensions');
            end
        end
        
        function obj = build(obj, s)
            if(min(obj.N ~= size(s)') & prod(obj.N) ~= length(s))
                disp('Error: Mismatch in dimensions of the signal being interpolated and values given at the initialization');
                return;
            end
            sdim = ndims(s);
            if(sdim == 2)
                if(size(s,1) == 1 || size(s,2) == 1)
                    sdim = 1;
                end
            end
            
            if(obj.dim == 1)
                obj.coeff = obj.build1D(s, 1, 1 , obj.N);
            elseif(obj.dim == 2)
                obj.coeff = obj.build2D(s, obj.N);
            elseif(obj.dim == 3)
                obj.coeff = obj.build3D(s, obj.N);
            else
                disp('Error: Current implementation of spline interpolation works only upto 3 dimensions');
            end
            obj.built = true;
        end
    end
    % DON'T FORGET DESTRUCTOR IN CPP
end