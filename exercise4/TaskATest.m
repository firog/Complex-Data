clearvars -except newXtest newXtrain newTargetXtest newTargetXtrain

input2 = newXtest;
%------Perform PCA-------%
[coeff, score, latent] = pca(input2);

%------Extract the eigenvectors-------%
% Covariance matrix
covMatrix = cov(input2);
% V are the eigenvectors and D the eigenvalues
[V, D] = eig(covMatrix);

m = mean(input2);

M = [];
for i=1:25
    M = [M, V(:,end-i)];
end

B = zeros(length(input2),length(input2)-length(M));
B = [B, M];

tn = B'*(input2'-m');
yn = m' + B*tn;

meanSquaredError = immse(input2,yn');

randomIndeces = rand(25,1)*length(input2)-1;
randomIndeces = int16(randomIndeces);

figure
for i=1:25
    subplot(5,5,i)
    imshow(reshape(yn(randomIndeces(i),:),[28,28]))
end
