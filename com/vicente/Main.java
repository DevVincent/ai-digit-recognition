package com.serafiroh;

public class Main {
    /* NN CONSTANTS */
    private static final int INPUT_NEURONS = 64;
    private static final int HIDDEN_NEURONS = 20;
    private static final int OUTPUT_NEURONS = 10;
    private static final int ROWS = 2810;
    private static final int COLUMNS = 65;
    private static final double BIAS = 1;

    /* Input files path constants*/
    private static final String FIRST_FILE = "src/resources/cw2DataSet1.csv";
    private static final String SECOND_FILE = "src/resources/cw2DataSet2.csv";

    /* Weights files constants*/
    private static final String HIDDEN_WEIGHTS_FILE = "src/resources/hiddenWeights.txt";
    private static final String OUTPUT_WEIGHTS_FILE = "src/resources/outputWeights.txt";

    /* Training constants */
    private static final double learningRate = 0.04;
    private static final double lmse = 0.01;
    private static final double momentum = 0.4;
    private static final int maxEpochs = 1000;

    public static void main(String[] args) {
        NeuralNetwork neuralNetwork = new NeuralNetworkBuilder()
                .setBIAS(BIAS)
                .setColumns(COLUMNS)
                .setRows(ROWS)
                .setOutputNeurons(OUTPUT_NEURONS)
                .setHiddenNeurons(HIDDEN_NEURONS)
                .setInputNeurons(INPUT_NEURONS)
                .setLearningRate(learningRate)
                .setLmse(lmse)
                .setMomentum(momentum)
                .setMaxEpochs(maxEpochs)
                .build();


        boolean train = true; //set to true for training, false for getting acc score.
        initialise(neuralNetwork, train);
    }

    public static void printResults(NeuralNetwork neuralNetwork) {
        System.out.println("First file: \n");
        neuralNetwork.transitionLayers(neuralNetwork.getFirstFileData());
        neuralNetwork.getAccuracy(neuralNetwork.getSecondFileData());

        System.out.println("Second file:\n");
        neuralNetwork.transitionLayers(neuralNetwork.getSecondFileData());
        neuralNetwork.getAccuracy(neuralNetwork.getSecondFileData());
    }

    public static void initialise(NeuralNetwork neuralNetwork, boolean train) {
        neuralNetwork.storeData(FIRST_FILE, SECOND_FILE, HIDDEN_WEIGHTS_FILE, OUTPUT_WEIGHTS_FILE);

        if(train) {
            neuralNetwork.trainNeuralNetwork(neuralNetwork.getFirstFileData());
        } else {
            neuralNetwork.getAccuracy(neuralNetwork.getFirstFileData());
            printResults(neuralNetwork);
        }
    }
}
