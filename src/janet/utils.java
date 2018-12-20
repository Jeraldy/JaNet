/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package janet;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

/**
 *
 * @author Deus
 */
public class utils {

    public static String open(String path, Charset encoding) {
        byte[] encoded = null;
        try {
            encoded = Files.readAllBytes(Paths.get(path));
        } catch (IOException ex) {
            Logger.getLogger(utils.class.getName()).log(Level.SEVERE, null, ex);
        }
        return new String(encoded, encoding);
    }

    public static char[] listUniqueChars(String data) {
        String a = data.trim();
        String s = a.toLowerCase().chars().distinct().mapToObj(c -> String.valueOf((char) c)).collect(Collectors.joining());
        char[] chars = new char[s.length() + 1];
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            chars[i] = c;
        }
//        for (int i = 0; i < chars.length; i++) {
//            System.out.println(i + " = " + chars[i]);
//        }

        return chars;
    }

    public static Map<Character, Integer> charToIx(char[] chars) {
        Map<Character, Integer> dict = new HashMap<>();
        for (int i = 0; i < chars.length; i++) {
            dict.put(chars[i], i);
           // System.out.println("c = "+chars[i]+":i = "+i);
        }
        return dict;
    }

    public static Map<Integer, Character> ixToChar(char[] chars) {
        Map<Integer, Character> dict = new HashMap<>();
        for (int i = 0; i < chars.length; i++) {
            dict.put(i, chars[i]);
        }
        return dict;
    }
}
