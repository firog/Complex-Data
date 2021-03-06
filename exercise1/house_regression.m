clear
[inputs, targets] = house_dataset;

% inputs = cancerInputs;
% targets = cancerTargets;
 
% Creates a Fitting Network
hiddenLayerSize = 100;
net = feedforwardnet(hiddenLayerSize,'trainlm');

% Set up Division of Data for Training, Validation, Testing
% 70% of the data is used for training
% 15% is used for validation and the last 15% is used for testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
 
% Train the Network
[net,tr] = train(net,inputs,targets);
 
% Test the Network
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs);

 
% Plots
% Uncomment these lines to enable various plots.
% figure, plotperform(tr)
% figure, plottrainstate(tr)
% figure, plotfit(targets,outputs)
figure, plotregression(targets,outputs)
%figure, plotconfusion(targets,outputs)