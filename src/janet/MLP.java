/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * ad open the template in the editornd open the template in the editor.
 */

package janet;

import static janet.np.print;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 *
 * @author Deus
 */

public class MLP {

    private static Map<String, double[][]> nextBatch(List<double[][]> x, double[][] y, int end) {
        double[][] X = new double[100][];
        double[][] Y = new double[100][];

        int j = 0;
        for (int row = end - 100; row < end; row++) {
            X[j] =  Arrays.stream(x.get(row))
                    .flatMapToDouble(Arrays::stream)
                    .toArray();
            j++;
        }

        int k = 0;
        for (int row = end - 100; row < end; row++) {
            Y[k] = y[row];
            k++;
        }

        Map<String, double[][]> batch = new HashMap<>();
        batch.put("Y", Y);
        batch.put("X", X);
        return batch;
    }

    public static void main(String[] args) {

        double[][] y = MnistReader.getLabels("mnist/train-labels.idx1-ubyte");
        List<double[][]> x = MnistReader.getImages("mnist/train-images.idx3-ubyte");

        int m = 100;
        int nodes = 128;

        double[][] W1 = np.random(nodes, 784);
        double[][] b1 = new double[nodes][m];

        double[][] W2 = np.random(1, nodes);
        double[][] b2 = new double[1][m];

        for (int batches = 1; batches < 7; batches++) {
            Map<String, double[][]> batch = nextBatch(x, y, batches * 100);

            double[][] X = batch.get("X");
            double[][] Y = batch.get("Y");

//            print("X.shape = " + np.shape(X));
//            print("Y.shape = " + np.shape(Y));
//
//            print("X = " + Arrays.toString(X[0]));
//            print("Y = " + Arrays.deepToString(Y));
            X = np.T(X);
            Y = np.T(Y);

            for (int i = 0; i < 1000; i++) {
                // Foward Prop
                print(".");
                // LAYER 1
                double[][] Z1 = np.add(np.dot(W1, X), b1);
                double[][] A1 = np.maximum(0, Z1);

                //LAYER 2
                double[][] Z2 = np.add(np.dot(W2, A1), b2);
                double[][] A2 = np.softmax(Z2);

                //double cost = np.cross_entropy(m, Y, A2);
                // Back Prop
                // compute the gradient on scores
                double[][] dscores = A2;
                for (int r = 0; r < dscores.length; r++) {
                    dscores[r][(int) Y[0][r]] -= 1.0;
                }
                dscores = np.divide(dscores, m);

                //LAYER 2
                double[][] dW2 = np.divide(np.dot(dscores, np.T(A1)), m);
                double[][] db2 = np.divide(dscores, m);

                // next backprop into hidden layer
                double[][] dZ1 = np.dot(np.T(W2), dscores);
                for (int r = 0; r < A1.length; r++) {
                    for (int c = 0; c < A1[1].length; i++) {
                        if (A1[r][c] < 0) {
                            dZ1[r][c] = 0;
                        }
                    }
                }

                //LAYER 1
                double[][] dW1 = np.divide(np.dot(dZ1, np.T(X)), m);
                double[][] db1 = np.divide(dZ1, m);

                // G.D
                W1 = np.subtract(W1, np.multiply(0.01, dW1));
                b1 = np.subtract(b1, np.multiply(0.01, db1));

                W2 = np.subtract(W2, np.multiply(0.01, dW2));
                b2 = np.subtract(b2, np.multiply(0.01, db2));

                if (i % 1 == 0) {
                    print("==============");
                    //print("Cost = " + cost);
                    print("True = " + Arrays.deepToString(Y));
                    print("Prediction = " + Arrays.deepToString(A2));
                }
            }
        }

    }

}
