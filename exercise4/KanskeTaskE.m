clearvars -except newXtest newXtrain newTargetXtest newTargetXtrain
close all

input1 = newXtrain';
%------Layer 1-------%
%------Extract the eigenvectors-------%
% Covariance matrix
covMatrix = cov(input1);
% V are the eigenvectors and D the eigenvalues
[V, D] = eig(covMatrix);

hiddenSize1 = 25;
%------Append the eigenvectors corresponding to the largest eigenvalues------%
B1 = zeros(784,hiddenSize1);
for i=0:hiddenSize1-1
    B1(:,end-i) = V(:,end-i);
end

m1 = mean(input1);

%-------Calculate D1(i,i)-------%
D1 = zeros(1,hiddenSize1);
sIN = 1; 
for i=0:hiddenSize1-1
    D1(:,end-i) = sIN/(sqrt(max(D(end,end))));
end
D1 = diag(D1);
% Set the weights and biases
w1 = D1*B1';
b1 = -w1*m1';

% Linear representation of layer output
v1n = w1*input1'+b1;

v1nCov = cov(v1n');
% Calculate standard deviation
STD1 = sqrt(diag(v1nCov));

% Layer output
z1 = tanh(v1n);

% Reconstruct the data
yn1 = m1' + B1*v1n;
MSE1 = immse(yn1,newXtrain);
%------Layer 2-------%
%------Extract the eigenvectors-------%
% Covariance matrix
z1 = z1';
covMatrix = cov(z1);
% V are the eigenvectors and D the eigenvalues
[V, D] = eig(covMatrix);

hiddenSize2 = 25;
%------Append the eigenvectors corresponding to the largest eigenvalues------%
B2 = zeros(25,hiddenSize2);
for i=0:hiddenSize2-1
    B2(:,end-i) = V(:,end-i);
end
m2 = mean(z1);

%-------Calculate D2(i,i)-------%
D2 = zeros(1,hiddenSize2);
sIN = 1; 
for i=0:hiddenSize2-1
    D2(:,end-i) = sIN/(sqrt(max(D(end,end))));
end
D2 = diag(D2);
% Set the weights and biases
w2 = D2*B2';
b2 = -w2*m2';

% Linear representation of layer output
v2n = w2*z1'+b2;

v2nCov = cov(v2n');
% Calculate standard deviation
STD2 = sqrt(diag(v2nCov));

% Layer output
z2 = tanh(v2n);

% Reconstruct the data
yn2 = m2' + B2*v2n;

%------Layer 3-------%
%------Extract the eigenvectors-------%
% Covariance matrix
z2 = z2';
covMatrix = cov(z2);
% V are the eigenvectors and D the eigenvalues
[V, D] = eig(covMatrix);

hiddenSize3 = 3; % H 
%------Append the eigenvectors corresponding to the largest eigenvalues------%
B3 = zeros(25,hiddenSize3);
for i=0:hiddenSize3-1
    B3(:,end-i) = V(:,end-i);
end

%-------Calculate D2(i,i)-------%
D3 = zeros(1,hiddenSize3);
sIN = 1; 
for i=0:hiddenSize3-1
    D3(:,end-i) = sIN/(sqrt(max(D(end,end))));
end
D3 = diag(D3);
% Set the weights and biases
m3 = mean(z2);
w3 = D3*B3';
b3 = -w3*m3';

% Linear representation of layer output
v3n = w3*z2'+b3;

v3nCov = cov(v3n');
% Calculate standard deviation
STD3 = sqrt(diag(v3nCov));

% Layer output
z3 = tanh(v3n);

% Reconstruct the data
yn3 = m3' + B3*v3n;

MSE3 = immse(yn3',z2);
%-------- Layer 4 and 5---------%

w4 = D3\B3';
b4 = m3;

z4 = tanh(w4'*z3+b4');

w5 = D2\B2;
b5 = m2;
z5 = tanh(w5'*z4+b5');

yn5 = B1*z5 + m1';

MSE5 = immse(newXtrain,yn5);
yn5 = yn5';
%----- show some figures ------%
randomIndeces = rand(25,1)*length(input1)-1;
randomIndeces = int16(randomIndeces);
figure
for i=1:25
    subplot(5,5,i)
    imshow(reshape(yn5(randomIndeces(i),:),[28,28]))
end