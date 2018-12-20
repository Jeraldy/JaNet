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
public class tf {

    public double[][][] conv2d(double[][][] X, double[][][] W, double[] strides, String padding) {
       
        return null;
    }

    public double[][][] max_pool(double[][][] A, double[] ksize, double[] strides, String padding) {
        return null;
    }

    public double[][][] relu(double[][][] Z) {
        return null;
    }

    public double[] flatten() {
        return null;
    }

    public double[][][] fully_connected() {
        return null;
    }

    public static void main(String[] args){
            int[][][] x = {
            {{1, 2}, {4, 5}, {7, 8}},
            {{10, 11}, {13, 14}, {16, 17}},
            {{19, 20}, {22, 23}, {25, 26}}
        };
        //print(""+Arrays.deepToString(x));
        print("" + x[0].length);
        print("" + x[0][0].length);
        print("" + x.length);
        int pad = 2;
        double[][][] padded = new double[x.length][x[0].length + 2 * pad][x[0][0].length + 2 * pad];

        print("rows = " + padded[0].length);
        print("cols = " + padded[0][0].length);
        print("chan = " + padded.length);

        for (int ch = 1; ch <= 3; ch++) {
            int i = 0;
            for (int r = (pad + 1); r < x[0].length; r++) {
                for (int c = (2 + 6 * pad) + i; c < x[0][0].length + (2 + 6 * pad); c++) {
                    i = x[0].length + (2 * pad + 1);
                    padded[ch][r][c] = x[ch][r - (pad + 1)][c - (2 + 6 * pad) - 1];
                }
            }

        }
        print("padded = " + Arrays.deepToString(padded));
//        double[][][] x2 = new double[2][2][3];
//        print(""+Arrays.deepToString(x2[0]));

    }
}
