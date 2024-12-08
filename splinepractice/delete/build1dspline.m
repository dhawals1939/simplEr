function c = build1dspline(s)
%From Michael Unser: Splines, A perfect fit
    z1 = -2 + sqrt(3);
    
    N = length(s);

    cp = zeros(N, 1);
    cn = zeros(N, 1);

    cp(1) = 0;

    for i=1:N
        cp(1) = cp(1) + s(i) * z1^(i-1);
    end

    for i=N-1:-1:2
        cp(1) = cp(1) + s(i) * z1^(2*N - 1 - i);
    end

    cp(1) = cp(1)/(1-z1^(2*N - 2));
    
%     cp(1)=1/(1-z1^N)*sum(s.*(z1.^[0 N-1:-1:1])); %For periodic case
    for i=2:N
        cp(i) = s(i) + z1 * cp(i-1);
    end

    cn(N) = z1/(z1*z1-1) * (cp(N) + z1 * cp(N-1));
%     cn(N)=-z1*(cp(N)+z1/(1-z1^N)*(sum(cp.*(z1.^(0:N-1)')))); %For periodic case

    for i=N-1:-1:1
        cn(i) = z1 * (cn(i+1) - cp(i));
    end

    c = 6*cn;
end