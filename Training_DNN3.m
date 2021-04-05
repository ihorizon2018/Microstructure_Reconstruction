function deepnet = Training_DNN3(X,Y,L1)
% Training the DNN model: 'SSAE+Softmax' classifier
% 'deepnet' is the trained DNN model.
% 'X' is the response varables.
% 'Y' is the predictor varables.
% 'L1' is the number of hidden unit for layer 1.

hiddenSize1 = L1;
% Training the first autoencoder
autoenc1 = trainAutoencoder(X,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

% Generating low dimensional features from the trained autoencoder
feat1 = encode(autoenc1,X);

% Training the softmax layer
softnet = trainSoftmaxLayer(feat1,Y,'MaxEpochs',1000);

% Stacking the encoders from the autoencoders together with
% the softmax layer to form a stacked network for classification.
deepnet1 = stack(autoenc1,softnet);

% Perform fine tuning to improve predictive performance for new dataset.
deepnet = train(deepnet1,X,Y);
end