% These variables are set and created in the setVariables.m script
targets = newTargetXtrain;
inputs = newXtrain;

% Define the hidden layer(s)
networkArchitecture = [25,10];

% Default optimization method for patternet is traincsg
optimizationMethod = 'trainscg';

% Create the network with the given architecture and training function.
net = patternnet(networkArchitecture, optimizationMethod);


net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Change the settings of the net
%net.performFcn = 'mse';

% The default min_grad is 1e-06, if I set it to 1.0e-14 then the training
% time of the network is increased. This might increase the precision of
% the network.

%net.trainParam.min_grad = 1.0e-17;
%net.trainParam.max_fail = 1000;
%net.performFcn = 'mse';

% Train the Network
[net,tr] = train(net,inputs,targets);

outputs1 = net(inputs);

% Use the trained network on the test data
outputs2 = net(newXtest);
errors = gsubtract(outputs1,targets);

performanceTrain = perform(net,targets,outputs1);

% Create the confusion matrix 
plotconfusion(newTargetXtest,outputs2)

% view(net);