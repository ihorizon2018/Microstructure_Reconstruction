function O = Reflection(image,r)
% Reflection boundary condition

o=double(image);
[ysize xsize]=size(o);
LU=fliplr(flipud(o));
MU=flipud(o);
RU=fliplr(flipud(o));
ML=fliplr(o);
MR=fliplr(o);
BL=flipud(fliplr(o));
BM=flipud(o);
BR=flipud(fliplr(o));

A=[LU MU RU;ML o MR;BL BM BR];
O=A(ysize+1-r:2*ysize+r,xsize+1-r:2*ysize+r);
end

