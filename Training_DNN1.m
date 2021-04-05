function deepnet = Training_DNN3( X,Y,L1,L2 )
% Training the DNN model, and deepnet is the trained model.
% X is the response varables.
% Y is the predictor varables.
% L1 is the number of hidden unit for layer 1.
% L2 is the number of hidden unit for layer 2.

hiddenSize1 = L1;
% Training the 1st autoencoder
autoenc1 = trainAutoencoder(X,hiddenSize1, ...
    'MaxEpochs',250, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);
% Generating low dimensional features from the trained autoencoder
feat1 = encode(autoenc1,X);

hiddenSize2 = L2;
% Training the 2nd autoencoder
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',300, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);
% Generating low dimensional features from the trained autoencoder
feat2 = encode(autoenc2,feat1);

% Training the softmax layer
softnet = trainSoftmaxLayer(feat2,Y,'MaxEpochs',300);

% Stacking the encoders from the autoencoders together with
% the softmax layer to form a stacked network for classification
deepnet1 = stack(autoenc1,autoenc2,softnet);

% Perform fine tuning to improve predictive performance for new dataset.
deepnet = train(deepnet1,X,Y);
end
