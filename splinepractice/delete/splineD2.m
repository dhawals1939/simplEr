function y = splineD2(x)

y = +  ((x+2)>0).*(x+2)  ... 
    -4*((x+1)>0).*(x+1)  ... 
    +6*((x)>0)  .*(x)  ... 
    -4*((x-1)>0).*(x-1)  ... 
    +  ((x-2)>0).*(x-2);
y = y;

end