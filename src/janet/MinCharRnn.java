/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package janet;

import static janet.np.print;
import static janet.utils.charToIx;
import static janet.utils.ixToChar;
import static janet.utils.listUniqueChars;
import static janet.utils.open;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 *
 * @author Deus
 */
public class MinCharRnn {

    static final String DATA = open("dinos.txt", Charset.defaultCharset());
    static final String[] TOKENS = DATA.toLowerCase().trim().replaceAll("\\s+", ",").split("(?!^)");
    static final char[] CHARS = listUniqueChars(DATA);
    static final int VOCAB_SIZE = CHARS.length;
    static Map<Character, Integer> char_to_ix = charToIx(CHARS);
    static Map<Integer, Character> ix_to_char = ixToChar(CHARS);

    // hyperparameters
    static final int HIDEN_SIZE = 100;
    static double learning_rate = 1e-1;

    public static Map<String, double[][]> initParams() {
        Map<String, double[][]> params = new HashMap<>();
        params.put("Wxh", np.random(HIDEN_SIZE, VOCAB_SIZE));
        params.put("Whh", np.random(HIDEN_SIZE, HIDEN_SIZE));
        params.put("Why", np.random(VOCAB_SIZE, HIDEN_SIZE));
        params.put("bh", new double[HIDEN_SIZE][1]);
        params.put("by", new double[VOCAB_SIZE][1]);
        return params;
    }

    public static Map<String, double[][]> initGrads() {
        Map<String, double[][]> grads = new HashMap<>();
        grads.put("dWxh", np.random(HIDEN_SIZE, VOCAB_SIZE));
        grads.put("dWhh", np.random(HIDEN_SIZE, HIDEN_SIZE));
        grads.put("dWhy", np.random(VOCAB_SIZE, HIDEN_SIZE));
        grads.put("dbh", new double[HIDEN_SIZE][1]);
        grads.put("dby", new double[VOCAB_SIZE][1]);
        grads.put("dhnext", new double[HIDEN_SIZE][1]);
        return grads;
    }

    public static Map<String, double[][]> updateParams(Map<String, double[][]> grads, Map<String, double[][]> params) {
        params.put("Wxh", np.subtract(params.get("Wxh"), np.multiply(learning_rate, grads.get("dWxh"))));
        params.put("Whh", np.subtract(params.get("Whh"), np.multiply(learning_rate, grads.get("dWhh"))));
        params.put("Why", np.subtract(params.get("Why"), np.multiply(learning_rate, grads.get("dWhy"))));
        params.put("bh", np.subtract(params.get("bh"), np.multiply(learning_rate, grads.get("dbh"))));
        params.put("by", np.subtract(params.get("by"), np.multiply(learning_rate, grads.get("dby"))));
        return params;
    }

    public static ArrayList<Map<String, double[][]>> lossFun(ArrayList<Integer> inputs,
            ArrayList<Integer> targets, double[][] hprev,
            Map<String, double[][]> grads, Map<String, double[][]> params) {

        Map<Integer, double[][]> xs = new HashMap<>();
        Map<Integer, double[][]> hs = new HashMap<>();
        Map<Integer, double[][]> ys = new HashMap<>();
        Map<Integer, double[][]> ps = new HashMap<>();

        double[][] Wxh = params.get("Wxh");
        double[][] Whh = params.get("Whh");
        double[][] Why = params.get("Why");
        double[][] bh = params.get("bh");
        double[][] by = params.get("by");

        double[][] dWxh = grads.get("dWxh");
        double[][] dWhh = grads.get("dWhh");
        double[][] dWhy = grads.get("dWhy");
        double[][] dbh = grads.get("dbh");
        double[][] dby = grads.get("dby");
        double[][] dhnext = grads.get("dhnext");

        hs.put(-1, hprev);
        double loss = 0;
        double smooth_loss = -Math.log(1.0 / VOCAB_SIZE) * inputs.size();

        // FOWARD PROP
        for (int t = 0; t < inputs.size(); t++) {
            xs.put(t, new double[VOCAB_SIZE][1]);
            xs.get(t)[inputs.get(t)][0] = 1;
            hs.put(t, np.tanh(np.add(np.add(np.dot(Wxh, xs.get(t)), np.dot(Whh, hs.get(t - 1))), bh)));
            ys.put(t, np.add(np.dot(Why, hs.get(t)), by));
            ps.put(t, np.softmax(ys.get(t)));
            loss += -Math.log(ps.get(t)[inputs.get(t)][0]);
        }
        smooth_loss = smooth_loss * 0.999 + loss * 0.001;

        print("Loss = " + smooth_loss);
        print("Wxh = " + Arrays.deepToString(Wxh));
        print("Why = " + Arrays.deepToString(Why));
        print("dWhy = " + Arrays.deepToString(dWhy));
        print("dby = " + Arrays.deepToString(dby));

        // BACK PROP
        for (int t = inputs.size() - 1; t >= 0; t--) {
            double[][] dy = ps.get(t);
            dy[targets.get(t)][0] -= 1;
            dWhy = np.add(dWhy, np.dot(dy, np.T(hs.get(t))));
            dby = np.add(dby, dy);
            double[][] dh = np.add(np.dot(np.T(Why), dy), dhnext);
            double[][] dhraw = np.multiply(np.subtract(1, np.power(hs.get(t), 2)), dh);
            dbh = np.add(dbh, dhraw);
            dWxh = np.add(dWxh, np.dot(dhraw, np.T(xs.get(t))));
            dWhh = np.add(dWhh, np.dot(dhraw, np.T(hs.get(t - 1))));
            dhnext = np.dot(np.T(Whh), dhraw);
        }

        // GRAD CLIP
        grads.put("dWxh", np.clip(dWxh, 5));
        grads.put("dWhh", np.clip(dWhh, 5));
        grads.put("dWhy", np.clip(dWhy, 5));
        grads.put("dbh", np.clip(dbh, 5));
        grads.put("dby", np.clip(dby, 5));
        grads.put("dhnext", hs.get(inputs.size() - 1));

        params.put("Wxh", Wxh);
        params.put("Whh", Whh);
        params.put("Why", Why);
        params.put("bh", bh);
        params.put("by", by);
        params.put("hprev", hprev);

        ArrayList<Map<String, double[][]>> prm = new ArrayList<>();
        prm.add(grads);
        prm.add(params);

        return prm;
    }

    public static ArrayList sample(double[][] h, int seed_ix, int n, Map<String, double[][]> params) {

        double[][] Wxh = params.get("Wxh");
        double[][] Whh = params.get("Whh");
        double[][] Why = params.get("Why");
        double[][] bh = params.get("bh");
        double[][] by = params.get("by");

        double[][] x = new double[VOCAB_SIZE][1];
        x[seed_ix][0] = 1;
        ArrayList ixes = new ArrayList<>();

        for (int t = 0; t < n; t++) {
            h = np.tanh(np.add(np.add(np.dot(Wxh, x), np.dot(Whh, h)), bh));
            double[][] y = np.add(np.dot(Why, h), by);
            double[][] p = np.softmax(y);
            int ix = np.choice(p);
            x = new double[VOCAB_SIZE][1];
            x[ix][0] = 1;
            ixes.add(ix);
        }

        return ixes;
    }

    public static void main(String args[]) {
        print(Arrays.toString(TOKENS));
        Map<String, double[][]> params = initParams();
        Map<String, double[][]> grads = initGrads();

        ArrayList<Integer> inputs = new ArrayList<>();
        ArrayList<Integer> targets = new ArrayList<>();

        for (int i = 0; i < TOKENS.length; i++) {
            inputs.add(char_to_ix.get(TOKENS[i].charAt(0)));
            if (i == TOKENS.length - 1) {
                targets.add(char_to_ix.get(TOKENS[0].charAt(0)));
            } else {
                targets.add(char_to_ix.get(TOKENS[i + 1].charAt(0)));
            }
            //print(Arrays.toString(inputs.toArray()));
        }

        double[][] h = new double[HIDEN_SIZE][1];

        for (int i = 0; i < 1000; i++) {
            print(" ===================== Iteration - " + i + " =================== ");
            // compute loss
            ArrayList<Map<String, double[][]>> prm = lossFun(inputs, targets, h, grads, params);
            // update grad
            grads = prm.get(0);
            params = prm.get(1);
            h = params.get("hprev");

            params = updateParams(grads, params);
            // sample
            if (i % 1 == 0) {
                ArrayList pred = sample(h, inputs.get(0), 13, params);
                pred.stream().forEach((Object pred1) -> {
                    System.out.print(ix_to_char.get(pred1));
                });
                print("");
            }
        }
    }
}
