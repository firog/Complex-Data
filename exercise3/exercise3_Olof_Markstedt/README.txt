USAGE: Run the script setVariables.m to transform the raw data into suitable data for the remaining scripts in this folder.

The script autoencoderNetworkManually.m calculates the features "manually" by extracting the weights and biases, multiplying these with the input and applies the sigmoid function to each element in the resulting matrix. This script also does the "stack" manually and in addition it uses the built-in function "stack". The features are called feat#man (# = 1/2/3). These features were compared visually with the features extracted using the "encode" function.

The script autoencoderNetworkAutomatic.m does everything that the previous script does but using built-in MATLAB functions such as "encode".

The script autoencoderNetworkPatternnet.m uses the patternnet as the final hidden layer instead of the softmax layer used in the previous network scripts.
