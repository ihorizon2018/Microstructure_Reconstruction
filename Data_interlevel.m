function [X,Y] = Data_interlevel(I1,I2,r)
% This function is to collect training data (data events) from images at two different levels.
% Detailed explanation of this function goes here:
% 'I1' is the training image at high level.
% 'I2' is the training image at low level.
% 'r' is the radius of the data template with 4 central pixels.
% 'Y' is the central pixel, and 'X' is the surrounding neibourhoods.

[x1,y1]=size(I1);
T=(x1-2*r-2)/2*(y1-2*r-2)/2;
t=1;
rr=round(r/2);% The radius of data template for I2
L=((2*r+1)^2-1)/2+r+(2*rr+1)^2;
X=zeros(T,L);
Y1=zeros(T,1);
Y=zeros(T,16);

for i=r+1:2:x1-r-2
    for j=r+1:2:y1-r-2
        %%% Image I1 %%%
        XB11=I1(i-r:i-1,j-r:j+r);
        XB1=reshape(XB11,[1,r*(2*r+1)]);
        XB22=I1(i:i+1,j-r:j-1);
        XB2=reshape(XB22,[1,r*2]);
        XB=cat(2,XB1, XB2);
        YY1=I1(i:i+1,j:j+1);
        internal=reshape(YY1,1,4);
        YY2=num2str(internal);
        YY=bin2dec(YY2);
        
        %%% Image I2 %%%
        ii=round((i+1)/2);
        jj=round((j+1)/2);
        XS1=I2(ii-rr:ii+rr,jj-rr:jj+rr);
        XS=reshape(XS1,[1,(2*rr+1)^2]);
        
        XX=cat(2,XB, XS);
        X(t,:)=XX;
        Y1(t,1)=YY;
        t=t+1;
    end
end
for m=1:T
    tt=Y1(m)+1;
    Y(m,tt)=1;
end
end
