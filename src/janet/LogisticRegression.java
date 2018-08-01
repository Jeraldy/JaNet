/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package janet;

import static janet.np.print;
import java.util.Arrays;

/**
 *
 * @author Deus
 */
public class LogisticRegression {

    public static void main(String[] args) {

        double[][] X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[][] Y = {{0}, {0}, {0}, {1}};

        int m = 4;

        X = np.T(X);
        Y = np.T(Y);

        double[][] W = np.random(1, 2);
        double[][] b = np.random(1, m);

        for (int i = 0; i < 2000; i++) {
            // Foward Prop
            double[][] Z = np.add(np.dot(W, X), b);
            double[][] A = np.sigmoid(Z);
            double cost = np.cross_entropy(m, Y, A);

            // Back Prop
            double[][] dZ = np.subtract(A, Y);
            double[][] dW = np.divide(np.dot(dZ, np.T(X)), m);
            double[][] db = np.divide(dZ, m);

            // G.D
            W = np.subtract(W, np.multiply(0.01, dW));
            b = np.subtract(b, np.multiply(0.01, db));

            if (i % 200 == 0) {
                print("==============");
                print("Cost = " + cost);
                print("A = " + Arrays.deepToString(A));
            }
        }

    }
}
