rng('default')
tic
% Layer size of the first encoder
hiddenSize1 = 25;

% Train an autoencoder using the Xtrain as input
autoenc1 = trainAutoencoder(Xtrain,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

% Returns the encoded data for the input Xtrain using the autoenc1 encoder
feat1 = encode(autoenc1,Xtrain);

% Layer size of the second encoder
hiddenSize2 =10;

% Train the second autoencoder using the feat1 as input
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);

% Returns the encoded data for the input feat1 using the autoenc2 encoder
feat2 = encode(autoenc2,feat1);

% Train the last layer in the neural network using the reshaped
% targetXtrain and feat2 as input
softnet = trainSoftmaxLayer(feat2,newTargetXtrain,'MaxEpochs',400);

% view(autoenc1)
% view(autoenc2)
% view(softnet)

% Put all layers together using the stack method. 
deepnet = stack(autoenc1,autoenc2,softnet);

%view(deepnet)
 toc
% Turn test images into vectors and put them in a matrix
newXtest = zeros(784,numel(Xtest));
for i = 1:numel(Xtest)
    newXtest(:,i) = Xtest{i}(:);
end

% Turn the training images into vectors and put them in a matrix
newXtrain = zeros(784,numel(Xtrain));
for i = 1:numel(Xtrain)
    newXtrain(:,i) = Xtrain{i}(:);
end

% Fine tune the deepnet
deepnet = train(deepnet,newXtrain,newTargetXtrain);

% Use the network on some test data
y = deepnet(newXtest);
plotconfusion(newTargetXtest,y);

% Plot the weights in layer 1 
W = autoenc1.EncoderWeights;
figure
W=W'; %Now the dimension is 784x25
for i=1:25
    WeightAsImage{i}=reshape(W(:,i), 28,28);
    subplot(5,5,i);
    imagesc(WeightAsImage{i})
    colormap(1-gray)
end

% D iii)

x1=autoenc1.EncoderWeights*newXtrain+autoenc1.EncoderBiases;
y_hiddenlayer1 = 1./(1+exp(-x1));
y_hiddenlayer1 = autoenc2.EncoderWeights*y_hiddenlayer1 + autoenc2.EncoderBiases;
y_hiddenlayer2 = 1./(1+exp(-y_hiddenlayer1));

% Max values
result_index=[];
result_value=[];
for i=1:10
    [M, I] = max(y_hiddenlayer2(i,:));
    result_index = [result_index I];
    result_value =[result_value M];
end
