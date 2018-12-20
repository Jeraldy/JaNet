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
import java.util.HashMap;
import java.util.Map;

/**
 *
 * @author Deus
 */
public class SimpleRNN {

    static final String DATA = open("story.txt", Charset.defaultCharset());
    static final String[] TOKENS = DATA.toLowerCase().split("(?!^)");
    static final char[] CHARS = listUniqueChars(DATA);
    static final int VOCAB_SIZE = CHARS.length;
    static Map<Character, Integer> char_to_ix = charToIx(CHARS);
    static Map<Integer, Character> ix_to_char = ixToChar(CHARS);

    // Hypeparameters
    static final int HIDDEN_SIZE = 100; //number of units(neurons)
    static double learning_rate = 0.01;
    static int seq_length = 20;

    // Parameters
    static double[][] Whh = np.random(HIDDEN_SIZE, HIDDEN_SIZE);
    static double[][] Whx = np.random(HIDDEN_SIZE, VOCAB_SIZE);
    static double[][] bh = new double[HIDDEN_SIZE][1];
    static double[][] Wyh = np.random(VOCAB_SIZE, HIDDEN_SIZE);
    static double[][] by = new double[VOCAB_SIZE][1];

    // Gradients
    static double[][] dWhh = new double[HIDDEN_SIZE][HIDDEN_SIZE];
    static double[][] dWhx = new double[HIDDEN_SIZE][VOCAB_SIZE];
    static double[][] dbh = new double[HIDDEN_SIZE][1];
    static double[][] dWyh = new double[VOCAB_SIZE][HIDDEN_SIZE];
    static double[][] dby = new double[VOCAB_SIZE][1];

    public static Map<String, double[][]> train(int[] x, int[] y, double[][] hprev) {
        Map<Integer, double[][]> xs = new HashMap<>();
        Map<Integer, double[][]> hs = new HashMap<>();
        Map<Integer, double[][]> ys = new HashMap<>();
        Map<Integer, double[][]> ps = new HashMap<>();

        hs.put(-1, hprev);
        double loss = 0;
        double[][] dhnext = new double[HIDDEN_SIZE][1];
        // Foward prop
        for (int t = 0; t < x.length; t++) {
            xs.put(t, new double[VOCAB_SIZE][1]);
            xs.get(t)[x[t]][0] = 1;
            hs.put(t, np.tanh(np.add(np.add(np.dot(Whx, xs.get(t)), np.dot(Whh, hs.get(t - 1))), bh)));
            ys.put(t, np.add(np.dot(Wyh, hs.get(t)), by));
            ps.put(t, np.softmax(ys.get(t)));
            loss += -Math.log(ps.get(t)[y[t]][0]);
        }

        // Back prop
        for (int t = y.length-1; t >= 0; t--) {
            double[][] dy = ps.get(t);
            dy[y[t]][0] -= 1;
            dWyh = np.add(dWyh, np.dot(dy, np.T(hs.get(t))));
            dby = np.add(dby, dy);
            double[][] dh = np.add(np.dot(np.T(Wyh), dy), dhnext);
            double[][] dhraw = np.multiply(np.subtract(1, np.power(hs.get(t), 2)), dh);
            dbh = np.add(dbh, dhraw);
            dWhx = np.add(dWhx, np.dot(dhraw, np.T(xs.get(t))));
            dWhh = np.add(dWhh, np.dot(dhraw, np.T(hs.get(t - 1))));
            dhnext = np.dot(np.T(Whh), dhraw);
        }

        // Grad clip
        dWhx = np.clip(dWhx, 5);
        dWhh = np.clip(dWhh, 5);
        dWyh = np.clip(dWyh, 5);
        dby = np.clip(dby, 5);
        dbh = np.clip(dbh, 5);

        Map<String, double[][]> params = new HashMap<>();
        double[][] _loss = {{loss}};
        params.put("dWhx", dWhx);
        params.put("dWhh", dWhh);
        params.put("dWyh", dWyh);
        params.put("dbh", dbh);
        params.put("dby", dby);
        params.put("dby", dby);
        params.put("loss", _loss);
        params.put("hs", hs.get(x.length - 1));

        return params;
    }

    public static ArrayList sample(double[][] h, int seed_ix, int n) {
        double[][] x = new double[VOCAB_SIZE][1];
        x[seed_ix][0] = 1;
        ArrayList ixes = new ArrayList<>();

        for (int t = 0; t < n; t++) {
            h = np.tanh(np.add(np.add(np.dot(Whx, x), np.dot(Whh, h)), bh));
            double[][] y = np.add(np.dot(Wyh, h), by);
            double[][] p = np.softmax(y);
            int ix = np.choice(p);
            x = new double[VOCAB_SIZE][1];
            x[ix][0] = 1;
            ixes.add(ix);
        }

        return ixes;
    }

    public static void gradCheck(int[] x, int[] y, double[][] hprev) {
        int num_checks = 10;
        double delta = 1e-5;
        
        Map<String, double[][]> params = train(x, y, hprev);
        assert np.shape(bh).equals(np.shape(params.get("dbhy")));
        assert np.shape(by).equals(np.shape(params.get("dby")));
        assert np.shape(Whh).equals(np.shape(params.get("dWhh")));
        assert np.shape(Whx).equals(np.shape(params.get("dWhx")));
        assert np.shape(Wyh).equals(np.shape(params.get("dWyh")));
        
        
    }

    public static void main(String[] args) {
        int n = 0, p = 0;
        int[] x = new int[seq_length];
        int[] y = new int[seq_length];

        // Memory variable for Adagrad
        double[][] mWhh = new double[HIDDEN_SIZE][HIDDEN_SIZE];
        double[][] mWhx = new double[HIDDEN_SIZE][VOCAB_SIZE];
        double[][] mbh = new double[HIDDEN_SIZE][1];
        double[][] mWyh = new double[VOCAB_SIZE][HIDDEN_SIZE];
        double[][] mby = new double[VOCAB_SIZE][1];

        double smooth_loss = -Math.log(1.0 / VOCAB_SIZE) * seq_length;
        double[][] hprev = new double[HIDDEN_SIZE][1];

        while (true) {
            if (p + seq_length + 1 >= TOKENS.length || n == 0) {
                hprev = new double[HIDDEN_SIZE][1];
                p = 0;
            }
            int counter1 = 0, counter2 = 0;
            for (int i = p; i < p + seq_length; i++) {
                int index = char_to_ix.get(TOKENS[i].charAt(0));
                x[counter1] = index;
                counter1++;
            }

            for (int i = p + 1; i < p + seq_length + 1; i++) {
                int index = char_to_ix.get(TOKENS[i].charAt(0));
                y[counter2] = index;
                counter2++;
            }

            if (n % 100 == 0) {
                ArrayList sample_ixes = sample(hprev, x[0], 80);
                ArrayList text = new ArrayList<>();
                sample_ixes.forEach(val -> {
                    text.add(ix_to_char.get(val));
                });
                text.forEach(u -> {
                    System.out.print(u);
                });
            }

            Map<String, double[][]> params = train(x, y, hprev);
            hprev = params.get("hs");
            
            gradCheck(x, y, hprev);
            
            smooth_loss = smooth_loss * 0.999 + params.get("loss")[0][0] * 0.001;

            if (n % 100 == 0) {
                print("");
                print("--------------------------------");
                print(" Iteration: " + n + " , Loss: " + smooth_loss);
            }

            //Perform parameter update with Adagrad
            mWhx = np.add(mWhx, np.multiply(params.get("dWhx"), params.get("dWhx")));
            Whx = np.add(Whx, np.divide(np.multiply(-learning_rate, params.get("dWhx")), np.sqrt(np.add(1e-8, mWhx))));

            mWhh = np.add(mWhh, np.multiply(params.get("dWhh"), params.get("dWhh")));
            Whh = np.add(Whh, np.divide(np.multiply(-learning_rate, params.get("dWhh")), np.sqrt(np.add(1e-8, mWhh))));

            mWyh = np.add(mWyh, np.multiply(params.get("dWyh"), params.get("dWyh")));
            Wyh = np.add(Wyh, np.divide(np.multiply(-learning_rate, params.get("dWyh")), np.sqrt(np.add(1e-8, mWyh))));

            mbh = np.add(mbh, np.multiply(params.get("dbh"), params.get("dbh")));
            bh = np.add(bh, np.divide(np.multiply(-learning_rate, params.get("dbh")), np.sqrt(np.add(1e-8, mbh))));

            mby = np.add(mby, np.multiply(params.get("dby"), params.get("dby")));
            by = np.add(by, np.divide(np.multiply(-learning_rate, params.get("dby")), np.sqrt(np.add(1e-8, mby))));

            p += seq_length;
            n += 1;
        }
    }
}
