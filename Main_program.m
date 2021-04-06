%%% The main program of 2D microstructure characterization and reconstruction via deep neural networks%%%
%%% The three-level approch %%%
%%% Coder: Jinlong Fu %%%
%%% Institute: Swansea University %%%
%%% Email: 887538@swansea.ac.uk %%%
%%% MATLAB 2016a %%%

clc;
clear all;
close all;

load original.mat   % Loading the original image (binary segmentation)

%% Image Pyramid %%
I1=logical(original);% The training image at the 1st level
I2 = impyramid(I1, 'reduce');% The training image at the 2nd level
I3 = impyramid(I2, 'reduce');% The training image at the 3rd level

%% Microstructure charaterization and reconstruction at the 3rd (coarse) level %%
%%% Step 1: Collecting training data (data events)
r3=4;  % The radius of data template with 1 central pixel
% Training data: 'X3' is neighboring pixels, and 'Y3' is the central pixels
[X3,Y3] = Data(I3,r3);

%%% Step 2: Stochatic microstructure characterization
%%% Train the 1st DNN model: 'SSAE+Softmax' classifier
X3=X3'; % Response varables
Y3=Y3'; % Predictor varables

L1=12; % L1 is the number of hidden unit for layer 1.
DNN3 = Training_DNN3(X3,Y3,L1);% The trainned DNN model.
close all;

%%% Stochastic microstructure reconstruction
[ysize3 xsize3] = size(I3); % The size of the training image
Rc=round(rand(xsize3+4*r3,ysize3+4*r3));% The initial guess of reconstruction
for k3=1:3
    for i=r3+1:xsize3+3*r3
        for j=r3+1:ysize3+3*r3
            WD=Rc(i-r3:i-1,j-r3:j+r3);
            XR1=reshape(WD,1,(2*r3+1)*r3);
            XR2=Rc(i,j-r3:j-1);
            XR=cat(2,XR1,XR2);
            XR=XR';
            P1=DNN3(XR);% Class probability
            Rc(i,j)=binornd(1,P1);% Probability sampling and pixel update
        end
    end
    rc3=Rc;
    rc=Rc(r3+1:ysize3+3*r3,r3+1:xsize3+3*r3);% Remove the border
    Rc=Periodical(rc,r3); % Periodical boundary condition
    % Rc=Reflection(rc,r3);% Reflected boundary condition
end
rc3=rc3;% The reconstruction result at the coarse level

%% Microstructure charaterization and reconstruction at the 2nd (middel) level %%
%%% Step 1: Collecting training data (data events)
% The radius of data template with 4 central pixels to collect data from 'I2'
r2=8;
% The radius of data template with 1 central pixel to collect data from 'I3'
rr2=round(r2/2);% radius for I2
[X2,Y2] = Data_interlevel(I2,I3,r2); % Training data at the 2nd level.

%%% Step 2: Stochatic microstructure characterization
%%% Train the 2nd DNN model: 'SSAE+Softmax' classifier
X2=X2'; % Response varables
Y2=Y2'; % Predictor varables

L1=100;% L1 is the number of hidden unit for layer 1.
L2=50;% L2 is the number of hidden unit for layer 2.
DNN2 = Training_DNN2(X2,Y2,L1,L2);% The trainned DNN model.
close all;

%%% Step 3: Stochastic microstructure reconstruction
[ysize3 xsize3] = size(rc3);
Rc=round(rand(2*xsize3,2*ysize3)); % Initial guess of reconstruction
[ysize2 xsize2] = size(Rc);

for k2=1:2
    for i=r2+1:2:xsize2-r2-2
        for j=r2+1:2:ysize2-r2-2
            XB11=Rc(i-r2:i-1,j-r2:j+r2);
            XB1=reshape(XB11,[1,r2*(2*r2+1)]);
            XB22=Rc(i:i+1,j-r2:j-1);
            XB2=reshape(XB22,[1,r2*2]);
            XB=cat(2, XB1, XB2);
            
            ii=round((i+1)/2);
            jj=round((j+1)/2);
            
            XS1=rc3(ii-rr2:ii+rr2,jj-rr2:jj+rr2);
            XS=reshape(XS1,[1,(2*rr2+1)^2]);
            XR=cat(2,XB, XS);
            XR=XR';
            P=DNN2(XR);% Probabilty vector
            
            % Probability sampling
            Rs=randsample(0:15,1,true,[P(1),P(2),P(3),P(4),P(5),P(6),...
                P(7),P(8),P(9),P(10),P(11),P(12),P(13),P(14),P(15),P(16)]);
            
            % Translate the decimal number to a binary number.
            Re=dec2bin(Rs,4);
            
            % Translate the numbers from char format to double format,
            % and update the pixels.
            Rc(i,j)=strread(Re(1), '%d');
            Rc(i+1,j)=strread(Re(2), '%d');
            Rc(i,j+1)=strread(Re(3), '%d');
            Rc(i+1,j+1)=strread(Re(4), '%d');
        end
    end
    rc=Rc(r2+1:ysize2-r2-2,r2+1:xsize2-r2-2);
    % Rc=Reflection(rc,r2+1); % Reflected boundary condition
    Rc=Periodical(rc,r2+1); % Periodical boundary condition
end
rc2=rc; % The reconstruction result at the middle level

%% Microstructure charaterization and reconstruction at the 1st (fine) level %%
%%% Step 1: Collecting training data (data events)
% The radius of data template with 4 central pixels to collect data from 'I1'
r1=14;
% The radius of data template with 1 central pixel to collect data from 'I2'
rr1=round(r1/2);
[X1,Y1] = Data_interlevel(I1,I2,r1); % Training data at the 1st level.

%%% Step 2: Stochastic microstructure characterization
%%% Train the 3rd DNN model: 'SSAE+Softmax' classifier
X1=X1';% Response varables
Y1=Y1';% Predictor varables

L1=100;% L1 is the number of hidden unit for layer 1.
L2=50;% L2 is the number of hidden unit for layer 2.
DNN1 = Training_DNN1(X1,Y1,L1,L2);% The trainned DNN model.

%%% Step 3: Stochastic microstructure reconstruction
[ysize2 xsize2] = size(rc2);
Rc=round(rand(2*ysize2,2*xsize2));% Initial guess of reconstruction
[ysize1 xsize1] = size(Rc);

for k1=1:2
    for i=r1+1:2:xsize1-r1-2
        for j=r1+1:2:ysize1-r1-2
            XB11=Rc(i-r1:i-1,j-r1:j+r1);
            XB1=reshape(XB11,[1,r1*(2*r1+1)]);
            XB22=Rc(i:i+1,j-r1:j-1);
            XB2=reshape(XB22,[1,r1*2]);
            XB=cat(2, XB1, XB2);
            
            ii=round((i+1)/2);
            jj=round((j+1)/2);
            
            XS1=rc2(ii-rr1:ii+rr1,jj-rr1:jj+rr1);
            XS=reshape(XS1,[1,(2*rr1+1)^2]);
            XR=cat(2,XB, XS);
            XR=XR';
            P=DNN1(XR);% Probabilty vector
            
            % Probability sampling
            Rs=randsample(0:15,1,true,[P(1),P(2),P(3),P(4),P(5),P(6),...
                P(7),P(8),P(9),P(10),P(11),P(12),P(13),P(14),P(15),P(16)]);
            
            % Translate the decimal number to a binary number.
            Re=dec2bin(Rs,4);
            
            % Translate the numbers from char format to double format,
            % and update the pixels.
            Rc(i,j)=strread(Re(1), '%d');
            Rc(i+1,j)=strread(Re(2), '%d');
            Rc(i,j+1)=strread(Re(3), '%d');
            Rc(i+1,j+1)=strread(Re(4), '%d');
        end
    end
    rc=Rc(r1+1:ysize1-r1-2,r1+1:xsize1-r1-2);% Remove the border
    Rc=Periodical(rc,r1+1); % Periodical boundary condition
    % Rc=Reflection(rc,r1+1); % Reflected boundary condition
end
rc1=rc; % Reconstruction result at the 1st level

%% Plot images %%
figure;
subplot(2,3,1);
imshow(I3); title('I3');
subplot(2,3,2);
imshow(I2); title('I2');
subplot(2,3,3);
imshow(I1); title('I1');

subplot(2,3,4);
imshow(rc3); title('rc3');
subplot(2,3,5);
imshow(rc2); title('rc2');
subplot(2,3,6);
imshow(rc1); title('rc1');
