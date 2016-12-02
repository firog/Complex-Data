clearvars -except newXtest newXtrain newTargetXtest newTargetXtrain B MSETaskA
close all

inputE = B';
M = 50;
H = 25;
trainOpt = 'trainscg';
net = feedforwardnet([M H M], trainOpt);

net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
net.layers{3}.transferFcn = 'purelin';

net = train(net, inputE, inputE);
output = net(inputE);

% m = mean(inputE);
% tn1 = output*(newXtrain-m');
% yn1 = m' + B*tn1;

MSETaskE = immse(output,inputE);