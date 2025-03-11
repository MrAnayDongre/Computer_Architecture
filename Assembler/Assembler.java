import java.io.*;

public class Assembler {

    public static void main(String[] args) throws IOException {
        if (args.length != 1) {
            System.out.println("Usage: java Assembler <assembly_file.asm>");
            return;
        }

        File inputFile = new File(args[0]);
        String baseName = inputFile.getName().replaceFirst("[.][^.]+$", ""); // Remove extension
        String outputFilePath = baseName + ".hack";
        File outputFile = new File(outputFilePath);

        SymbolTable symbolTable = new SymbolTable();
        Parser parser = new Parser(inputFile);

        int ROMaddress = 0;
        while (parser.hasMoreCommands()) {
            parser.advance();
            if (parser.commandType().equals("A_COMMAND")) {
                ROMaddress++;
            } else if (parser.commandType().equals("C_COMMAND")) {
                ROMaddress++;
            } else if (parser.commandType().equals("L_COMMAND")) {
                symbolTable.addEntry(parser.symbol(), ROMaddress);
            }
        }
        parser.close();

        Parser parser2 = new Parser(inputFile); // Re-initialize parser for second pass
        PrintWriter writer = new PrintWriter(outputFile);
        int RAMaddress = 16;

        parser2.setCurrentCommandCounter(0); // Reset command counter for second pass

        while (parser2.hasMoreCommands()) {
            parser2.advance();
            String commandType = parser2.commandType();

            if (commandType.equals("A_COMMAND")) {
                String symbol = parser2.symbol();
                int address;
                try {
                    address = Integer.parseInt(symbol);
                } catch (NumberFormatException e) {
                    if (symbolTable.contains(symbol)) {
                        address = symbolTable.getAddress(symbol);
                    } else {
                        symbolTable.addEntry(symbol, RAMaddress);
                        address = RAMaddress;
                        RAMaddress++;
                    }
                }
                String binaryAddress = Integer.toBinaryString(address);
                writer.write("0" + String.format("%15s", binaryAddress).replace(' ', '0') + "\n");

            } else if (commandType.equals("C_COMMAND")) {
                String destMnemonic = parser2.dest();
                String compMnemonic = parser2.comp();
                String jumpMnemonic = parser2.jump();

                String aBit = "0";
                if (compMnemonic != null && compMnemonic.contains("M")) {
                    aBit = "1";
                    compMnemonic = compMnemonic.replace("M", "A"); // Use 'A' version for comp encoding
                }

                writer.write("111" + aBit + Code.comp(compMnemonic) + Code.dest(destMnemonic) + Code.jump(jumpMnemonic) + "\n");
            }
        }

        parser2.close();
        writer.close();
        System.out.println("Successfully assembled " + inputFile.getName() + " to " + outputFilePath);
    }
}