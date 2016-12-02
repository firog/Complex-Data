clear

X = mat

hiddenSize = 100;
autoenc1 = trainAutoencoder(X,hiddenSize,...
    'L2WeightRegularization',0.001,...
    'SparsityRegularization',4,...
    'SparsityProportion',0.05,...
    'DecoderTransferFunction','purelin');

features1 = encode(autoenc1,X);

hiddenSize = 3;
autoenc2 = trainAutoencoder(features1,hiddenSize,...
    'L2WeightRegularization',0.001,...
    'SparsityRegularization',4,...
    'SparsityProportion',0.05,...
    'DecoderTransferFunction','purelin',...
    'ScaleData',false);

features2 = encode(autoenc2,features1);

softnet = trainSoftmaxLayer(features2,T,'LossFunction','crossentropy');

deepnet = stack(autoenc1,autoenc2,softnet);

deepnet = train(deepnet,X,T);

wine_type = deepnet(X);

plotconfusion(T,wine_type);