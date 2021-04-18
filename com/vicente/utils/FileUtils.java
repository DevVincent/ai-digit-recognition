package com.serafiroh.utils;

import java.io.*;

public class FileUtils {

    private static final String COMMA = ",";

    public static double[][] fileRead(String filePath, int rowCount, int columnCount){
        double[][] values = new double[rowCount][columnCount];
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            int lineNumber = 0;
            while ((line = br.readLine()) != null) {
                String[] split = line.split(COMMA);
                int columnNumber = 0;
                for (String s : split) {
                    double v = Double.parseDouble(s);
                    values[lineNumber][columnNumber++] = v;
                }
                lineNumber++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return values;
    }

    public static void fileWrite(String fileName, double[][] data) {
        // Create PrintStream class object
        PrintStream printStreamObj;
        double arrayElement;
        String path = "src/resources/";

        try {
            printStreamObj = new PrintStream(new FileOutputStream(path + fileName));

            for(int row = 0; row < data.length; row++){
                // Dont make a new line on the first iteration
                if(row != 0)
                    // Next line
                    printStreamObj.println();
                for(int column = 0; column < data[row].length; column++){
                    arrayElement = data[row][column];
                    // Separate by comma;
                    printStreamObj.print(arrayElement + ",");
                }
            }
            // Close the writing into file
            printStreamObj.close();
        } catch (FileNotFoundException errorName) {
            System.out.println("File was not created. ERROR: " + errorName.getMessage());
        }
    }
}
