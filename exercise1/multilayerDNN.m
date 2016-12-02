clear
% Run assignemnt1 so that the necessary variables are set
assignment1

% E)
% The neural network architecture. 3 corresponding to 100-3-100
% [20,3,20] corresponding to 100-20-3-20-100.
networkArchitecture = 3;
%netnetworkArchitecture = [20,3,20];

% Training function, trainbfg and trainlm are the most efficent for "smaller"
% networks
trainingFunction = 'trainbfg';

% Create the network with the given architecture and training function.
% feedforwardnet has built in early stopping method to avoid overfitting.
net = feedforwardnet(networkArchitecture, trainingFunction);

% Set hidden layers transfer function to purelin/logsig. Comment out for
% tansig, tansig is default.

% net.layers{1}.transferFcn = 'purelin';
% net.layers{2}.transferFcn = 'purelin';
% net.layers{3}.transferFcn = 'purelin';

mat = mat';
inputs = mat;
targets = mat;

% Divide the data into train, validation and testing sets.
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Train the Network
[net,tr] = train(net,inputs,targets);

outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs);

view(net);

% Retrieval of the hidden layer output
hiddenOutput = radbas(netprod(dist(net.IW{1,1},mat),net.b{1}));

figure, plot3(hiddenOutput(1,:),hiddenOutput(2,:),hiddenOutput(3,:),'.')
title('Hidden output plot 20-3-20')
xlabel('Hidden output vector 1')
ylabel('Hidden output vector 2')
zlabel('Hidden output vector 3')