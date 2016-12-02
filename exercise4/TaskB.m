clearvars -except newXtrain newXtest newTargetXtest newTargetXtrain
close all

inputB = newXtrain;

H = 50;
trainOpt = 'trainscg';

net = feedforwardnet(H, trainOpt);

net.layers{1}.transferFcn = 'purelin';
net.layers{2}.transferFcn = 'purelin';
net.trainParam.min_grad = 1e-3;
net = train(net,inputB,inputB);

output = net(newXtrain);

MSETaskB = immse(newXtrain, output)

randomIndeces = rand(25,1)*length(inputB)-1;
randomIndeces = int16(randomIndeces);
output = output';
figure
for i=1:25
    subplot(5,5,i)
    imshow(reshape(output(randomIndeces(i),:),[28,28]))
end

inputB = inputB';

figure
for i=1:25
    subplot(5,5,i)
    imshow(reshape(inputB(randomIndeces(i),:),[28,28]))
end

