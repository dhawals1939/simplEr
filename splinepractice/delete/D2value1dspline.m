function v = value1dspline(c, x)

v = 0;
% if x < 1 || x > length(c)
%     return;
% end

for i=ceil(x-2):floor(x+2)
    if(i<1)
        v = v + c(2-i) * splineD2(x - i);
    elseif(i>length(c))
        v = v + c(2*length(c) - i) * splineD2(x - i);
    else
        v = v + c(i) * splineD2(x - i);
    end
    
end
end