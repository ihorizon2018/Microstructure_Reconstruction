function [X,Y] = Data(image,r)
% Detailed explanation of this function goes here:
% Input: 'image' is a binary image, and 'r' is the radius of a data template with one central pixel.
% Ouput: Training data (data event). 
% 'Y' is the central pixel, and 'X' is the surrounding neibourhoods.

[ysize xsize] = size(image);
C=double(image);
L=r*(2*r+1)+r;
T=(ysize-2*r)*(xsize-2*r);
X=zeros(T,L);
Y=zeros(T,1);
t=1;

for i=r+1:ysize
    for j=r+1:xsize-r
        WD=C(i-r:i-1,j-r:j+r);
        X1=reshape(WD,1,(2*r+1)*r);
        X2=C(i,j-r:j-1);
        XX=cat(2,X1,X2);
        YY=C(i,j);
        
        X(t,:)=XX;
        Y(t,:)=YY;
        t=t+1;
    end
end
