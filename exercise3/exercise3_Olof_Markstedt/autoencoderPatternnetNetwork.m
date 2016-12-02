rng('default')

hiddenSize1 = 25;
hiddenSize2 = 10;
hiddenSize3 = 7;

autoenc1 = trainAutoencoder(newXtrain,hiddenSize1, ...
    'MaxEpochs',1000, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

feat1 = logsig(autoenc1.EncoderWeights * newXtrain + autoenc1.EncoderBiases);

autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

feat2 = logsig(autoenc2.EncoderWeights * feat1 + autoenc2.EncoderBiases);

net = patternnet(hiddenSize3);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'logsig';
[net,tr] = train(net,feat2,newTargetXtrain);

%--------Calculate the features from the patternnet--------%
w3 = net.IW{1};
b3 = net.b{1};
feat3 = logsig(w3*feat2+b3);

%--------Create the deepnet--------%
deepnet = stack(autoenc1, autoenc2, net);

%--------Use the test data on the pretrained network--------%
netOutput1 = deepnet(newXtest);

%--------Manually calculate the output of the network using some test data--------%
feat1Test = logsig(autoenc1.EncoderWeights * newXtest + autoenc1.EncoderBiases);
feat2Test = logsig(autoenc2.EncoderWeights * feat1Test + autoenc2.EncoderBiases);
feat3Test = logsig(w3 * feat2Test + b3);
feat4Test = logsig(net.lw{2,1} * feat3Test + net.b{2});

%--------Plot the classification of manually calculated network output and automatically--------%
figure, plotconfusion(newTargetXtest, feat4Test,'Pretrained network output manually calculated',...
    newTargetXtest, netOutput1, 'Pretrained network output automatically calculated');

%--------Visualize the response of the neurons from layer 2 in the pretrained deepnet------%
layer1WeightsPre = deepnet.IW{1};
layer1BiasesPre = deepnet.b{1};
layer1ResponsePre = logsig(layer1WeightsPre * newXtest + layer1BiasesPre);

layer2WeightsPre = deepnet.LW{2,1};
layer2BiasesPre = deepnet.b{2};
layer2ResponsePre = logsig(layer2WeightsPre * layer1ResponsePre + layer2BiasesPre);

figure, plot(layer2ResponsePre','o');
title('Layer 2 neurons responses from pretrained network')
xlabel('Image number')
ylabel('Neuron response value')
legend('Neuron 1','Neuron 2','Neuron 3','Neuron 4','Neuron 5', ...
    'Neuron 6','Neuron 7','Neuron 8','Neuron 9','Neuron 10');
% Calculate the probability of false alarm and detection of the pretrained network(estimate performance)
[C, CM, PER] = confusion(newTargetXtest, netOutput1);
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

%--------Fine tune the deepnet--------%
deepnet = train(deepnet, newXtrain, newTargetXtrain);

%--------Use the test data on the fine tuned network--------%
netOutput2 = deepnet(newXtest);

%--------Fine tuned output confusion matrix-------%
figure, plotconfusion(newTargetXtest, netOutput2);

%--------Visualize the response from layer 2 in the fine tuned network--------%
layer1Weights = deepnet.IW{1};
layer1Biases = deepnet.b{1};
layer1Response = logsig(layer1Weights * newXtest + layer1Biases);

layer2Weights = deepnet.LW{2,1};
layer2Biases = deepnet.b{2};
layer2Response = logsig(layer2Weights * layer1Response + layer2Biases);

figure, plot(layer2Response','o');
title('Layer 2 neuron responses from fine tuned network')
xlabel('Image number')
ylabel('Neuron response value')
legend('Neuron 1','Neuron 2','Neuron 3','Neuron 4','Neuron 5', ...
    'Neuron 6','Neuron 7','Neuron 8','Neuron 9','Neuron 10');
%-------Calculate probability of detection on the fine tuned network--------%
[C, CM, PER] = confusion(newTargetXtest, netOutput2);
% True negatives
TN2 = CM(1,1);
% True positives
TP2 = CM(2,2);
% False positive
FP2 = CM(1,2);
% False negative
FN2 = CM(2,1);

probDetection2 = TP2/(TP2+FN2);
falseAlarm2 = FP2/(TN2 + FP2);

%-------Plot the images that gives the maximum response-------%
result_index = [];
result_value = [];
for i=1:10
    [M, I] = max(layer2Response(i,:));
    result_index = [result_index I];
    result_value =[result_value M];
    subplot(4,5,i);
    imshow(Xtest{result_index(i)});
end