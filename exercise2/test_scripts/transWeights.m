W = autoenc1.EncoderWeights;
figure
W=W'; %Now the dimension is 784x25
for i=1:25
    WeightAsImage{i}=reshape(W(:,i), 28,28);
    subplot(5,5,i);
    imagesc(WeightAsImage{i})
    colormap(1-gray)
end