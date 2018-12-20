/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package janet;

import static janet.np.print;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 *
 * @author Deus
 */
public class RNN {

    public static Map<String, double[][]> initParams(int n_a, int n_x, int n_y, int m) {
        Map<String, double[][]> parameters = new HashMap<>();
        parameters.put("Wax", np.random(n_a, n_x));
        parameters.put("Waa", np.random(n_a, n_a));
        parameters.put("Wya", np.random(n_y, n_a));
        parameters.put("ba", np.random(n_a, m));
        parameters.put("by", np.random(n_y, m));
        return parameters;
    }

    public static Map<String, double[][]> initGradients(int n_a, int n_x, int n_y, int m) {
        Map<String, double[][]> gradients = new HashMap<>();
        gradients.put("dWax", new double[n_a][n_x]);
        gradients.put("dWaa", new double[n_a][n_a]);
        gradients.put("dWya", new double[n_y][n_a]);
        gradients.put("dba", new double[n_a][m]);
        gradients.put("dby", new double[n_y][m]);
        return gradients;
    }

    public static ArrayList<Map> rnnCellFoward(Map<String, double[][]> parameters, double[][] xt, double[][] a_prev) {
        Map<String, double[][]> cache = new HashMap<>();
        ArrayList<Map> returnVals = new ArrayList<>();

        double[][] Waa = parameters.get("Waa");
        double[][] Wax = parameters.get("Wax");
        double[][] Wya = parameters.get("Wya");
        double[][] ba = parameters.get("ba");
        double[][] by = parameters.get("by");

        double[][] a_next = np.tanh(np.add(np.add(np.dot(Waa, a_prev), np.dot(Wax, xt)), ba));
        double[][] yt_pred = np.softmax(np.add(np.dot(Wya, a_next), by));

        cache.put("yt_pred", yt_pred);
        cache.put("a_next", a_next);
        cache.put("a_prev", a_prev);
        cache.put("xt", xt);

        returnVals.add(cache);
        returnVals.add(parameters);
        return returnVals;
    }

    public static Map<String, double[][]> rnnCellBackward(
            Map<String, double[][]> gradients,
            Map<String, double[][]> parameters,
            double[][] dy,
            double[][] x,
            double[][] a,
            double[][] a_prev
    ) {
        double[][] Waa = parameters.get("Waa");
        double[][] Wya = parameters.get("Wya");
        double[][] dWaa = gradients.get("Waa");
        double[][] dWax = gradients.get("Wax");
        double[][] dWya = gradients.get("Wya");
        double[][] dba = gradients.get("ba");
        double[][] dby = gradients.get("by");
        double[][] da_next = gradients.get("da_next");

        gradients.put("dWya", np.add(dWya, np.dot(dy, np.T(a))));
        gradients.put("dby", np.add(dby, dy));

        double[][] da = np.add(np.dot(np.T(Wya), dy), da_next);
        double[][] dtanh = np.multiply(np.subtract(1, np.power(a, 2)), da);

        gradients.put("dba", np.add(dba, dtanh));
        gradients.put("dWax", np.add(dWax, np.dot(dtanh, np.T(x))));
        gradients.put("dWaa", np.add(dWaa, np.dot(dtanh, np.T(a_prev))));
        gradients.put("da_next", np.dot(np.T(Waa), dtanh));

        return gradients;
    }

    public static Map<String, double[][]> updateParams(Map<String, double[][]> parameters, double lr) {
        parameters.put("Wax", np.subtract(1, np.multiply(lr, parameters.get("Wax"))));
        parameters.put("Waa", np.subtract(1, np.multiply(lr, parameters.get("Waa"))));
        parameters.put("Wya", np.subtract(1, np.multiply(lr, parameters.get("Wya"))));
        parameters.put("ba", np.subtract(1, np.multiply(lr, parameters.get("ba"))));
        parameters.put("by", np.subtract(1, np.multiply(lr, parameters.get("by"))));
        return parameters;
    }

    public static Map<String, ArrayList> rnnFoward(ArrayList<double[][]> x, double[][] a0,
            Map<String, double[][]> parameters) {

        Map<String, ArrayList> returnVals = new HashMap<>();

        ArrayList<Map<String, double[][]>> caches = new ArrayList<>();
        ArrayList returnParams = new ArrayList<>();

        ArrayList<double[][]> a = new ArrayList<>();
        ArrayList<double[][]> y_pred = new ArrayList<>();

        double[][] a_next = a0;

        for (int t = 0; t < x.size(); t++) {
            ArrayList<Map> vals = rnnCellFoward(parameters, x.get(t), a_next);
            a_next = (double[][]) vals.get(0).get("a_next");

            a.add((double[][]) vals.get(0).get("a_next"));
            y_pred.add((double[][]) vals.get(0).get("y_pred"));
            caches.add(vals.get(1));
        }

        returnParams.add(x);
        returnParams.add(y_pred);
        returnParams.add(a);
        returnVals.put("caches", caches);
        returnVals.put("returnParams", returnParams);

        return returnVals;
    }

    public static ArrayList rnnBackward(ArrayList<double[][]> X, ArrayList<double[][]> Y,
            ArrayList<ArrayList> caches, Map<String, double[][]> parameters) {
        Map<String, double[][]> gradients = initGradients(5, 10, 3, 5);
        ArrayList returnVals = new ArrayList<>();
        
        ArrayList a = (ArrayList) caches.get(2);

        for (int t = X.size(); t >= 0; t--) {
            gradients = rnnCellBackward(
                    gradients,
                    parameters,
                    Y.get(t),
                    X.get(t),
                    (double[][]) a.get(t),
                    (double[][]) a.get(t - 1)
            );

        }
        returnVals.add(a);
        returnVals.add(gradients);
        return returnVals;
    }

    public static void main(String args[]) {

        double[][] xt = np.random(3, 10);
        ArrayList<double[][]> x = new ArrayList<>();
        x.add(xt);
        x.add(xt);
        x.add(xt);
        x.add(xt);

        double[][] a_prev = np.random(5, 10);
        double[][] Waa = np.random(5, 5);
        double[][] Wax = np.random(5, 3);
        double[][] Wya = np.random(2, 5);
        double[][] ba = np.random(5, 10);
        double[][] by = np.random(2, 10);

        Map<String, double[][]> parameters = new HashMap<>();

        parameters.put("Waa", Waa);
        parameters.put("Wax", Wax);
        parameters.put("Wya", Wya);
        parameters.put("ba", ba);
        parameters.put("by", by);

        double[][] da_next = np.random(5, 10);
        ArrayList<Map> vals = rnnCellFoward(parameters, xt, a_prev);
//        Map<String, double[][]> gradients = rnnCellBackward(da_next, vals);
//        print("" + Arrays.deepToString(gradients.get("da_prev")));
        //rnnFoward(x, a_prev, parameters);
        //ArrayList var = (ArrayList) rnnFoward(x, a_prev, parameters).get("returnParams").get(2);
        //print("" + Arrays.deepToString((double[][]) var.get(1)));
        //print(""+Arrays.deepToString((double[][])rnnCellFoward(parameters, xt, a_prev).get(0).get("a")));
    }
}
