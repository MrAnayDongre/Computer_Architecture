package Assembler;
import java.util.HashMap;
import java.util.Map;

class Code {

    public static String dest(String mnemonic) {
        if (mnemonic != null) {
            int bits = 0b000;
            if (mnemonic.contains("M")) {
                bits |= 0b001;
            }
            if (mnemonic.contains("D")) {
                bits |= 0b010;
            }
            if (mnemonic.contains("A")) {
                bits |= 0b100;
            }
            String binaryString = Integer.toBinaryString(bits);
            // Pad with leading zeros if necessary to ensure 3 bits
            while (binaryString.length() < 3) {
                binaryString = "0" + binaryString;
            }
            return binaryString;
        } else {
            return "000";
        }
    }

    public static String comp(String mnemonic) {
        if (mnemonic != null) {
            Map<String, String> compDict = new HashMap<>();
            compDict.put("0", "101010");
            compDict.put("1", "111111");
            compDict.put("-1", "111010");
            compDict.put("D", "001100");
            compDict.put("A", "110000");
            compDict.put("M", "110000"); // Added M here as it was in original python code and in problem description
            compDict.put("!D", "001101");
            compDict.put("!A", "110001");
            compDict.put("!M", "110001"); // Added !M
            compDict.put("-D", "001111");
            compDict.put("-A", "110011");
            compDict.put("-M", "110011"); // Added -M
            compDict.put("D+1", "011111");
            compDict.put("A+1", "110111");
            compDict.put("M+1", "110111"); // Added M+1
            compDict.put("D-1", "001110");
            compDict.put("A-1", "110010");
            compDict.put("M-1", "110010"); // Added M-1
            compDict.put("D+A", "000010");
            compDict.put("D+M", "000010"); // Added D+M
            compDict.put("D-A", "010011");
            compDict.put("D-M", "010011"); // Added D-M
            compDict.put("A-D", "000111");
            compDict.put("M-D", "000111"); // Added M-D
            compDict.put("D&A", "000000");
            compDict.put("D&M", "000000"); // Added D&M
            compDict.put("D|A", "010101");
            compDict.put("D|M", "010101"); // Added D|M

            return compDict.get(mnemonic);
        } else {
            return "000"; // Or handle null case as needed
        }
    }

    public static String jump(String mnemonic) {
        if (mnemonic != null) {
            Map<String, String> jumpDict = new HashMap<>();
            jumpDict.put("JGT", "001");
            jumpDict.put("JEQ", "010");
            jumpDict.put("JGE", "011");
            jumpDict.put("JLT", "100");
            jumpDict.put("JNE", "101");
            jumpDict.put("JLE", "110");
            jumpDict.put("JMP", "111");
            return jumpDict.get(mnemonic);
        } else {
            return "000";
        }
    }
}