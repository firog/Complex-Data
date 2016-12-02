clearvars -except newXtest newXtrain newTargetXtest newTargetXtrain
close all

inputA = newXtrain';
%------Extract the eigenvectors-------%
% Covariance matrix
covMatrix = cov(inputA);
% V are the eigenvectors and D the eigenvalues
[V, D] = eig(covMatrix);

m = mean(inputA);

% Choose layer size
H = 50;
M = zeros(784,H);
for i=0:H-1
    M(:,end-i) = V(:,end-i);
end

B = M;
tn = B'*(inputA'-m');
% Reconstruct the data using the H biggest eigenvectors
yn = m' + B*tn;

MSETaskA = immse(inputA,yn');

randomIndeces = rand(25,1)*length(inputA)-1;
randomIndeces = int16(randomIndeces);
yn = yn';
figure
for i=1:25
    subplot(5,5,i)
    imshow(reshape(yn(randomIndeces(i),:),[28,28]))
end

%input1 = input1';
figure
for i=1:25
    subplot(5,5,i)
    imshow(reshape(inputA(randomIndeces(i),:),[28,28]))
end
