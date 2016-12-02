% Reset the pseudo random number generator
rng('default')

% Layer size of the first encoder
hiddenSize1 = 25;

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
hiddenSize2 = 10;

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

softnetWeights = softnet.IW{1};
softnetBiases = softnet.b{1};

feat3man = logsig(softnetWeights * feat2man + softnetBiases);

% Put all layers together using the stack method. 
deepnetMan = stack(autoenc1man,autoenc2man,softnet);

% Use some test data on the deepnet
pretrainedResult = deepnetMan(newXtest);

%------------Use weights and biases from the pretrained network on the test data------------%
feat1Test = logsig(autoenc1Weights * newXtest + autoenc1Biases);
feat2Test = logsig(autoenc2Weights * feat1Test + autoenc2Biases);
feat3Test = logsig(softnetWeights * feat2Test + softnetBiases);

%------------Visualize the neurons response from the pretrained 2nd layer------------%
preTrainedLayer1Response = logsig(deepnetMan.IW{1} * newXtest + deepnetMan.b{1});
preTrainedLayer2Response = logsig(deepnetMan.LW{2,1} * preTrainedLayer1Response + deepnetMan.b{2});
figure, plot(preTrainedLayer2Response', 'o');
legend('Neuron 1','Neuron 2','Neuron 3','Neuron 4','Neuron 5', ...
    'Neuron 6','Neuron 7','Neuron 8','Neuron 9','Neuron 10');
title('Pretrained network neurons responses')
xlabel('Image number')
ylabel('Neuron response value')

% Calculate the probability of false alarm and detection of the pretrained network(estimate performance)
[C, CM, PER] = confusion(newTargetXtest, pretrainedResult);
% True negatives
TN1= CM(1,1);
% True positives
TP1 = CM(2,2);
% False positive
FP1 = CM(1,2);
% False negative
FN1 = CM(2,1);

probDetection1 = TP1/(TP1+FN1);
probFalseAlarm1 = FP1/(TN1 + FP1);

%------------Fine tune the network------------%
deepnetMan = train(deepnetMan, newXtrain, newTargetXtrain);
netOutput = deepnetMan(newXtest);

% Confusion matrix of the test data used on the pretrained NN and the fine
% tuned NN.

% Comparison of the "manually" calculated network output and the
% automatically calculated network output.
figure, plotconfusion(newTargetXtest,feat3Test,'Pretrained network manually', newTargetXtest,pretrainedResult,'Pretrained network automatically');

% Calculate the probability of false alarm and detection of the fine tuned network(estimate performance)
[C, CM, PER] = confusion(newTargetXtest, netOutput);
% True negatives
TN2= CM(1,1);
% True positives
TP2 = CM(2,2);
% False positive
FP2 = CM(1,2);
% False negative
FN2 = CM(2,1);

probDetection2 = TP2/(TP2+FN2);
probFalseAlarm2 = FP2/(TN2 + FP2);

%--------Visualize the neurons response from the fine tuned 2nd hidden layer--------%
layer1Weights = deepnetMan.IW{1};
layer1Biases = deepnetMan.b{1};
layer1Response = logsig(layer1Weights * newXtest + layer1Biases);

layer2Weights = deepnetMan.LW{2,1};
layer2Biases = deepnetMan.b{2};
layer2Response = logsig(layer2Weights * layer1Response + layer2Biases);

figure 
plot(layer2Response','o');
title('Fine tuned network neurons responses')
xlabel('Image number')
ylabel('Neuron response value')

% Find and plot the images that produce the highest neuron response
result_index=[];
result_value=[];
figure
title('Second')
for i=1:10
    [M, I] = max(layer2Response(i,:));
    result_index = [result_index I];
    result_value =[result_value M];
    subplot(4,5,i);
    imshow(Xtest{result_index(i)});
end
figure, plot(netOutput','o');
title('Final network output')
xlabel('Image number')
ylabel('Network response')