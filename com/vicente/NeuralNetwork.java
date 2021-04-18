package com.serafiroh;

import com.serafiroh.utils.FileUtils;

import java.util.Arrays;

public class NeuralNetwork {

    private double bias;
    private int inputNeurons;
    private int hiddenNeurons;
    private int outputNeurons;
    private int rows;
    private int columns;

    private final double learningRate;
    private final double lmse;
    private final double momentum;
    private final double maxEpochs;

    /* Weights */
    private double[][] firstLayerWeights;
    private double[][] secondLayerWeights;

    /* Used for storing partial values */
    private double[][] hiddenLayer;
    private double[][] outputLayer;


    /* Arrays for data from files */
    private double[][] firstFileData = new double[rows][columns];
    private double[][] secondFileData = new double[rows][columns];

    protected NeuralNetwork(double learningRate, double lmse, double momentum, int maxEpochs, int inputNeurons, int hiddenNeurons, int outputNeurons, int rows, int columns, double bias) {
        this.learningRate = learningRate;
        this.lmse = lmse;
        this.momentum = momentum;
        this.maxEpochs = maxEpochs;
        this.inputNeurons = inputNeurons;
        this.hiddenNeurons = hiddenNeurons;
        this.outputNeurons = outputNeurons;
        this.rows = rows;
        this.columns = columns;
        this.bias = bias;
        firstLayerWeights = new double[inputNeurons][hiddenNeurons];
        secondLayerWeights = new double[hiddenNeurons][outputNeurons];
        hiddenLayer = new double[rows][hiddenNeurons];
        outputLayer = new double[rows][outputNeurons];
    }

    public void layerTransition(double[][] startingLayer, double[][] nextLayer, double[][] weights){
        //We don't need to return to reassign anything as arrays work by reference
        double neuronValue;

        // For each neuron in the firstLayer
        for(int startingNeuron = 0; startingNeuron < startingLayer.length; startingNeuron++) {
            // For each neuron in the next layer
            for(int nextLayerNeuron = 0; nextLayerNeuron < nextLayer[startingNeuron].length; nextLayerNeuron++) {
                neuronValue = 0;
                for(int startingLayerNeuron = 0; startingLayerNeuron < startingLayer[startingNeuron].length - 1; startingLayerNeuron++){
                    neuronValue += startingLayer[startingNeuron][startingLayerNeuron] * weights[startingLayerNeuron][nextLayerNeuron];
                }
                nextLayer[startingNeuron][nextLayerNeuron] = sigmoid(neuronValue + bias);;
            }
        }
    }

    //We read from persistent storage into memory
    public void storeData(String firstFile, String secondFile, String hiddenWeightsFile, String outputWeightsFile) {
        firstFileData = FileUtils.fileRead(firstFile, rows, columns);
        secondFileData = FileUtils.fileRead(secondFile, rows, columns);
        firstLayerWeights = FileUtils.fileRead(hiddenWeightsFile, rows, columns);
        secondLayerWeights = FileUtils.fileRead(outputWeightsFile, rows, columns);
    }

    public void transitionLayers(double[][] inputLayer) {
        //Input to hidden
        layerTransition(inputLayer, hiddenLayer, firstLayerWeights);

        //Hidden to output.
        layerTransition(hiddenLayer, outputLayer, secondLayerWeights);
    }

    /*Modify this */
    public void trainNeuralNetwork(double[][] input) {
        double mse = 0.0;
        int epochCount = 1;

        // Total error variable
        double error = 0.0;

        // Lowest error threshold variable
        double errorThreshold = 0.0001;

        // Target variable (last element of input array)
        double target;

        // Target position in array
        int targetPosition = columns - 1;

        // Hidden layer delta
        double[][] hdelta = new double[rows][hiddenNeurons];

        // Output layer delta
        double[][] odelta = new double[rows][outputNeurons];

        // Temporary weights for training
        double[][] tempHiddenWeights = Arrays.copyOf(firstLayerWeights, firstLayerWeights.length);
        double[][] tempOutputWeights = Arrays.copyOf(secondLayerWeights, secondLayerWeights.length);

        // Previous weights for training
        double[][] previousHiddenWeights = Arrays.copyOf(firstLayerWeights, firstLayerWeights.length);
        double[][] previousOutputWeights = Arrays.copyOf(secondLayerWeights, secondLayerWeights.length);

        // Loop until errorThreshold is reached
        while(Math.abs(mse - lmse) > errorThreshold) {
            // For each epoch reset the mean square error
            mse = 0.0;

            // Loop through all the inputs
            for(int fileArray = 0; fileArray < input.length; fileArray++) {

                // Calculate dot products from input to hidden and from hidden to output
                transitionLayers(input);

                // Set the target which is the last element of input array
                target = input[fileArray][targetPosition];

                // Backpropagation from output layer
                for(int outputNeuron = 0; outputNeuron < outputLayer[fileArray].length; outputNeuron++) {
                    // Calculate delta and error if output neuron IS NOT the target
                    if(outputNeuron != target) {
                        odelta[fileArray][outputNeuron] = (0.0 - outputLayer[fileArray][outputNeuron]) * sigmoidDerivative(outputLayer[fileArray][outputNeuron]);
                        error += (0.0 - outputLayer[fileArray][outputNeuron]) * (0.0 - outputLayer[fileArray][outputNeuron]);
                    }
                    // Calculate delta and error if output neuron IS the target
                    else {
                        odelta[fileArray][outputNeuron] = (1.0 - outputLayer[fileArray][outputNeuron]) * sigmoidDerivative(outputLayer[fileArray][outputNeuron]);
                        error += (1.0 - outputLayer[fileArray][outputNeuron]) * (1.0 - outputLayer[fileArray][outputNeuron]);
                    }
                }

                /* Backpropagation from hidden layer */
                for(int hiddenNeuron = 0; hiddenNeuron < hiddenLayer[fileArray].length; hiddenNeuron++) {
                    // Zero the values from the previous iteration
                    hdelta[fileArray][hiddenNeuron] = 0.0;

                    // Add to the delta for each output neuron
                    for(int outputNeuron = 0; outputNeuron < outputLayer[fileArray].length; outputNeuron++) {
                        hdelta[fileArray][outputNeuron] += odelta[fileArray][outputNeuron] * firstLayerWeights[hiddenNeuron][outputNeuron] ;
                    }

                    // Use sigmoid derivative for later weight adjustments
                    hdelta[fileArray][hiddenNeuron] *= sigmoidDerivative(hiddenLayer[fileArray][hiddenNeuron]);
                }

                tempHiddenWeights = Arrays.copyOf(firstLayerWeights, firstLayerWeights.length);
                tempOutputWeights = Arrays.copyOf(secondLayerWeights, secondLayerWeights.length);

                /* Input to hidden weights */
                for(int inputNeuron = 0; inputNeuron < input[fileArray].length - 1; inputNeuron++) {
                    for(int hiddenNeuron = 0; hiddenNeuron < hiddenLayer[fileArray].length; hiddenNeuron++) {
                        firstLayerWeights[inputNeuron][hiddenNeuron] +=
                                (momentum * (firstLayerWeights[inputNeuron][hiddenNeuron]
                                        - previousHiddenWeights[inputNeuron][hiddenNeuron]))
                                        + (learningRate * hdelta[fileArray][hiddenNeuron] * input[fileArray][inputNeuron]);
                    }
                }

                for(int outputNeuron = 0; outputNeuron < outputLayer[fileArray].length; outputNeuron++) {
                    for(int hiddenNeuron = 0; hiddenNeuron < hiddenLayer[fileArray].length; hiddenNeuron++) {
                        secondLayerWeights[hiddenNeuron][outputNeuron] +=
                                (momentum * (secondLayerWeights[hiddenNeuron][outputNeuron]
                                        - previousOutputWeights[hiddenNeuron][outputNeuron]))
                                        + (learningRate * odelta[fileArray][outputNeuron] * hiddenLayer[fileArray][hiddenNeuron]);
                    }
                }

                // Save modified weights as previous for each loop
                previousHiddenWeights = Arrays.copyOf(tempHiddenWeights, tempHiddenWeights.length);
                previousOutputWeights = Arrays.copyOf(tempOutputWeights, tempOutputWeights.length);

                // Get total mean squared error for epoch
                mse += error / (outputNeurons + 1);

                // Reset error for next loop
                error = 0.0;
            }

            // Print the process
            System.out.println("Epoch: " + epochCount + " | Error value = " + mse);

            // Check for the epoch count and add one to the counter.
            if(epochCount++ >= this.maxEpochs)
                break;

            // Save weights into a file after each epoch
            FileUtils.fileWrite("newWeightsHidden.txt", previousHiddenWeights);
            FileUtils.fileWrite("newWeightsHidden.txt", previousOutputWeights);
        }
    }
    private static double sigmoidDerivative(double value) {
        return value * (1 - value);
    }

    private static double sigmoid(double value) {
        return 1 / (1 + Math.exp(-value));
    }

    public void getAccuracy(double[][] data) {
        double biggest;
        int correctAnswerPos;
        double correctPredictionCounter = 0;
        double accuracy;

        // Loop through the output layer
        for(int outputNeuronIndex = 0; outputNeuronIndex < outputLayer.length; outputNeuronIndex++){
            // Reset values for each loop
            correctAnswerPos = 0;
            biggest = 0;
            // Loop through the output elements and check for the winner
            for(int layerElement = 0; layerElement < outputLayer[outputNeuronIndex].length; layerElement++){
                // Find the biggest number (winner, which should be the prediction).
                if(outputLayer[outputNeuronIndex][layerElement] > biggest){
                    biggest = outputLayer[outputNeuronIndex][layerElement];
                    correctAnswerPos = layerElement;
                }
            }
            // Count correct predictions
            if(correctAnswerPos == data[outputNeuronIndex][64])
                correctPredictionCounter++;
        }
        // Count the accuracy
        accuracy = (correctPredictionCounter / data.length) * 100;

        // Output
        System.out.println("==============================");
        System.out.println("Input count: " + data.length);
        System.out.println("Correct Count: " + (int) correctPredictionCounter);
        System.out.println("Accuracy: " + roundToTwoDecimals(accuracy) + "%");
        System.out.println("==============================");
    }
    public static double roundToTwoDecimals(double value) {
        return Math.round(value * 100) /100;
    }

    public double[][] getFirstFileData() {
        return firstFileData;
    }

    public double[][] getSecondFileData() {
        return secondFileData;
    }
}

class NeuralNetworkBuilder {

    public NeuralNetworkBuilder setBIAS(double BIAS) {
        this.BIAS = BIAS;
        return this;
    }

    public NeuralNetworkBuilder setInputNeurons(int inputNeurons) {
        this.inputNeurons = inputNeurons;
        return this;
    }

    public NeuralNetworkBuilder setHiddenNeurons(int hiddenNeurons) {
        this.hiddenNeurons = hiddenNeurons;
        return this;
    }

    public NeuralNetworkBuilder setOutputNeurons(int outputNeurons) {
        this.outputNeurons = outputNeurons;
        return this;
    }

    public NeuralNetworkBuilder setRows(int rows) {
        this.rows = rows;
        return this;
    }

    public NeuralNetworkBuilder setColumns(int columns) {
        this.columns = columns;
        return this;
    }

    public NeuralNetworkBuilder setLearningRate(double learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    public NeuralNetworkBuilder setLmse(double lmse) {
        this.lmse = lmse;
        return this;
    }

    public NeuralNetworkBuilder setMomentum(double momentum) {
        this.momentum = momentum;
        return this;
    }

    public NeuralNetworkBuilder setMaxEpochs(int maxEpochs) {
        this.maxEpochs = maxEpochs;
        return this;
    }

    public NeuralNetwork build(){
        return new NeuralNetwork(learningRate, lmse, momentum, maxEpochs, inputNeurons, hiddenNeurons, outputNeurons, rows, columns, BIAS);
    }

    private double BIAS;
    private int inputNeurons;
    private int hiddenNeurons;
    private int outputNeurons;
    private int rows;
    private int columns;

    private double learningRate;
    private double lmse;
    private double momentum;
    private int maxEpochs;


}
