rng('default')

% Layer size of the first encoder
hiddenSize1 = 25;

% Train an autoencoder using the Xtrain as input
autoenc1 = trainAutoencoder(newXtrain,hiddenSize1, ...
    'MaxEpochs',1000, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

% Returns the encoded data for the input Xtrain using the autoenc1 encoder
feat1 = encode(autoenc1,newXtrain);

% Layer size of the second encoder
hiddenSize2 = 10;

% Train the second autoencoder using the feat1 as input
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

% Returns the encoded data for the input feat1 using the autoenc2 encoder
feat2 = encode(autoenc2,feat1);

% Train the last layer in the neural network using the reshaped
% targetXtrain and feat2 as input

softnet = trainSoftmaxLayer(feat2,newTargetXtrain,'MaxEpochs',400);

% Put all layers together using the stack method. 
deepnetAuto = stack(autoenc1,autoenc2,softnet);
y1 = deepnetAuto(newXtest);

% Calculate the probability of false alarm and detection of the pretrained network (estimate performance)
[C, CM, PER] = confusion(newTargetXtest, y1);
% True negatives
TN1 = CM(1,1);
% True positives
TP1 = CM(2,2);
% False positive
FP1 = CM(1,2);
% False negative
FN1 = CM(2,1);

probDetection1 = TP/(TP+FN);
probFalseAlarm1 = FP/(TN + FP);

%--------Visualize the neurons response from the pre trained 2nd hidden layer--------%
layer1Weights = deepnetAuto.IW{1};
layer1Biases = deepnetAuto.b{1};
layer1Response = logsig(layer1Weights * newXtest + layer1Biases);

layer2Weights = deepnetAuto.LW{2,1};
layer2Biases = deepnetAuto.b{2};
layer2Response = logsig(layer2Weights * layer1Response + layer2Biases);

figure 
plot(layer2Response','o');
title('Pre trained network neurons responses')
xlabel('Image number')
ylabel('Neuron response value')

% Fine tune the deepnet
deepnetAuto = train(deepnetAuto,newXtrain,newTargetXtrain);

%--------Visualize the neurons response from the fine tuned 2nd hidden layer--------%
layer1Weights = deepnetAuto.IW{1};
layer1Biases = deepnetAuto.b{1};
layer1Response = logsig(layer1Weights * newXtest + layer1Biases);

layer2Weights = deepnetAuto.LW{2,1};
layer2Biases = deepnetAuto.b{2};
layer2Response = logsig(layer2Weights * layer1Response + layer2Biases);

figure 
plot(layer2Response','o');
title('Fine tuned network neurons responses')
xlabel('Image number')
ylabel('Neuron response value')

y2 = deepnetAuto(newXtest);
% Calculate the probability of false alarm and detection of the fine tuned network (estimate performance)
[C, CM, PER] = confusion(newTargetXtest, y2);
% True negatives
TN2 = CM(1,1);
% True positives
TP2 = CM(2,2);
% False positive
FP2 = CM(1,2);
% False negative
FN2 = CM(2,1);

probDetection2 = TP2/(TP2+FN2);
probFalseAlarm2 = FP2/(TN2 + FP2);

% Use the network on some test data
y2 = deepnetAuto(newXtest);

% Comparison of the two outputs
figure, plotconfusion(newTargetXtest,y1,'Pre trained network', newTargetXtest,y2,'Fine tuned network');

% Find and plot the images that produce the highest neuron response
result_index=[];
result_value=[];
figure
for i=1:10
    [M, I] = max(layer2Response(i,:));
    result_index = [result_index I];
    result_value =[result_value M];
    subplot(4,5,i);
    imshow(Xtest{result_index(i)});
end