function O = Periodical(image,r)
% Periodical boundary condition
o=double(image);
[ysize xsize] = size(o);
A=repmat(o,[3,3]);
O=A(ysize+1-r:2*ysize+r,xsize+1-r:2*ysize+r);
end

