clear
%B)
% Preallocation of memory to increase performance
mat = zeros(5000,100);
%tv = zeros(1,5000);
% Three randomly created 100 by 1 vectors
b1 = randn(100,1);
b2 = randn(100,1);
b3 = randn(100,1);
for n = 1:5000
    % Randomly chosen value of t from a normal distribution
    t = normrnd(0,1);
    % A vector that is a multiple of b1, b2 and b3
    x = b1*t+b2*t^2+b3*t^3;
    % Appends all the generated vectors to a matrix
    mat(n,:) = x;
end
 
% Plot the 5000 vectors
figure, plot(mat')
title('Plot of the 5000 vectors')
xlabel('Vector index')
ylabel('Index value')

% C)
% ev contains the eigenvalues. The largest variance between the data points
% are in 3 dimension thus there are only 3 eigenvalues larger than 0. 
[coeff, score, ev] = pca(mat);


% 3D-plot of the score values from the pca
figure, plot3(score(:,1),score(:,2),score(:,3),'.')
title('3D-plot of the score values from the pca')
xlabel('Score vector 1')
ylabel('Score vector 2')
zlabel('Score vector 3')