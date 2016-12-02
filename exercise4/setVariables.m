clear
% Load the data
[Xtrain,targetsXTrain] = digitTrainCellArrayData;
[Xtest,targetsXTest] = digitTestCellArrayData;

% Reshape the cells in Xtrain into a vector of size 784x1
newXtrain = [];
for i = 1:5003
       newXtrain = [newXtrain, reshape(Xtrain{i},[784,1])];
end

% Reshape the cells in Xtest into a vector of size 784x1
newXtest = [];
for i = 1:4997
       newXtest = [newXtest, reshape(Xtest{i},[784,1])];
end

% Create the target train matrix. If the value is 1 in the third row of each column
% in the TargetXtrain matrix then that picture depicts the number 3 which is the target. Otherwise
% it's a 0. The output matrix will be a 1x5003 and we want to classify/divide up all
% the figures in two class, one containing all the threes and one containing all the rest.
newTargetXtrain = [];
for i = 1:5003
    if targetsXTrain(3,i) == 1
        newTargetXtrain = [newTargetXtrain, [1]];
    else
        newTargetXtrain = [newTargetXtrain, [0]];
    end
end

% Create the target test matrix.
newTargetXtest = [];
for i = 1:4997
    if targetsXTest(3,i) == 1
        newTargetXtest = [newTargetXtest, [1]];
    else
        newTargetXtest = [newTargetXtest, [0]];
    end
end