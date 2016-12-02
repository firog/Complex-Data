clearvars -except newXtest newXtrain newTargetXtest newTargetXtrain
close all
tic
inputC = newXtrain';

H = 50;
%------Layer 1-------%
%------Extract the eigenvectors-------%
% Covariance matrix
covMatrix = cov(inputC);
% V are the eigenvectors and D the eigenvalues
[V, D] = eig(covMatrix);

%------Append the eigenvectors corresponding to the largest eigenvalues------%
B1 = zeros(784,100);
for i=0:99
    B1(:,end-i) = V(:,end-i);
end

m1 = mean(inputC);

%-------Calculate D1(i,i)-------%
D1 = zeros(1,100);
sIN = 1; 
for i=0:99
    D1(:,end-i) = sIN/(sqrt(max(D(end,end))));
end
D1 = diag(D1);
% Set the weights and biases
w1 = D1*B1';
b1 = -w1*m1';

% Linear representation of layer output
v1n = w1*inputC'+b1;

v1nCov = cov(v1n');
% Calculate standard deviation
STD1 = sqrt(diag(v1nCov));

% Layer output
z1 = tanh(v1n);

% Reconstruct the data
yn1 = m1' + B1*v1n;
% Calculate the mean squared error of the reconstructed data
MSE1 = immse(inputC,yn1');

%--------Layer 2---------%
% Input for layer 2
z1 = z1';
covMatrix = cov(z1);
% V are the eigenvectors and D the eigenvalues
[V, D] = eig(covMatrix);

B2 = zeros(100,H);
for i=0:H-1
    B2(:,end-i) = V(:,end-i);
end

m2 = mean(z1);

%-------Calculate D2(i,i)-------%
D2 = zeros(1,H);

for i=0:H-1
    D2(:,end-i) = sIN/(sqrt(max(D(end,end))));
end
D2 = diag(D2);

w2 = D2*B2';
b2 = -w2*m2';

v2n = w2*z1'+b2;
v2nCov = cov(v2n');

STD2 = sqrt(diag(v2nCov));

z2 = tanh(w2*z1'+b2);

yn2 = m2' + B2*v2n;
MSE2 = immse(z1',yn2);
%-------Reconstruct the data--------%
w3 = D2\B2';
b3 = m2;

w4 = D1\B1';
b4 = m1;

z3 = tanh(w3'*z2+b3');
z4 = w4'*z3+b4';
z4 = z4';

MSE4 = immse(z4',newXtrain);
toc
%------Compare the reconstructed data with the input data------%
randomIndeces = rand(25,1)*length(inputC)-1;
randomIndeces = int16(randomIndeces);
figure
for i=1:25
    subplot(5,5,i)
    imshow(reshape(z4(randomIndeces(i),:),[28,28]))
end

figure
for i=1:25
    subplot(5,5,i)
    imshow(reshape(inputC(randomIndeces(i),:),[28,28]))
end

%figure,plot([MSE1,MSE2,MSE4]);

%-------    (vi)   -------%
tic
trainOpt = 'trainscg';
net = feedforwardnet([100 H], trainOpt);
inputC = inputC';

net = configure(net, inputC, newTargetXtrain);

net.IW{1} = w1;
net.b{1} = b1;
net.LW{2,1} = w2;
net.b{2} = b2;

net = train(net, newXtrain, newTargetXtrain);

output = net(newXtrain);
figure,plotconfusion(newTargetXtrain,output);
MSEFeedForward = immse(newTargetXtrain,output);
toc