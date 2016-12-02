clearvars -except newXtest newXtrain newTargetXtest newTargetXtrain output
close all

% Reset the pseudo random number generator
rng('default')

% Layer size of the first encoder
tic
hiddenSize1 = 100;

% Train an autoencoder using the Xtrain as input
autoenc1man = trainAutoencoder(newXtrain,hiddenSize1, ...
    'MaxEpochs',1000, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

% Code that manually returns the features/encoded data from the first hidden layer
autoenc1Weights = autoenc1man.EncoderWeights;
autoenc1Biases = autoenc1man.EncoderBiases;

feat1man = logsig(autoenc1Weights*newXtrain+autoenc1Biases);

% Layer size of the second encoder
hiddenSize2 = 25;

% Train the second autoencoder using the feat1 as input
autoenc2man = trainAutoencoder(feat1man,hiddenSize2, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

autoenc2Weights = autoenc2man.EncoderWeights;
autoenc2Biases = autoenc2man.EncoderBiases;

% Extract the features/encoded data manually from the second hidden layer
feat2man = logsig(autoenc2Weights*feat1man+autoenc2Biases);

% Train the last layer in the neural network using the reshaped
% targetXtrain and feat2 as input
softnet = trainSoftmaxLayer(feat2man,newTargetXtrain,'MaxEpochs',400);
%softnet.layers{1}.transferFcn = 'purelin';
softnetWeights = softnet.IW{1};
softnetBiases = softnet.b{1};

feat3man = logsig(softnetWeights * feat2man + softnetBiases);

% Put all layers together using the stack method. 
deepnetMan = stack(autoenc1man,autoenc2man,softnet);

% Use some test data on the deepnet
deepnetMan = train(deepnetMan, newXtrain, newTargetXtrain); 
netout = deepnetMan(newXtrain);

MSEAutoenc = immse(newTargetXtrain,netout);
toc
figure, plotconfusion(newTargetXtrain,output,'PCA approach', newTargetXtrain,netout,'Autoencoder approach');

