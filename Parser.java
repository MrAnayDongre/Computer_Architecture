
import java.io.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

class Parser {
    private BufferedReader reader;
    private String currentCommand;
    private int currentCommandCounter;

    public Parser(File inputFile) throws FileNotFoundException {
        reader = new BufferedReader(new FileReader(inputFile));
        currentCommand = null;
        currentCommandCounter = 0;
    }

    public boolean hasMoreCommands() throws IOException {
        return reader.ready();
    }

    public void advance() throws IOException {
        currentCommand = reader.readLine();
        currentCommandCounter++;
        if (currentCommand != null) {
            currentCommand = currentCommand.trim().split("//")[0].replace(" ", "");
        }
    }

    public String commandType() {
        if (currentCommand == null || currentCommand.isEmpty()) {
            return ""; // Handle empty lines
        } else if (currentCommand.startsWith("@")) {
            return "A_COMMAND";
        } else if (currentCommand.startsWith("(")) {
            return "L_COMMAND";
        } else {
            return "C_COMMAND";
        }
    }

    public String symbol() {
        if (currentCommand.startsWith("@")) {
            return currentCommand.substring(1);
        } else if (currentCommand.startsWith("(")) {
            Pattern pattern = Pattern.compile("\\((.*?)\\)");
            Matcher matcher = pattern.matcher(currentCommand);
            if (matcher.find()) {
                return matcher.group(1);
            }
        }
        return null; // Or handle if no symbol found as needed
    }

    public String dest() {
        if (currentCommand.contains("=")) {
            return currentCommand.split("=")[0];
        }
        return null;
    }

    public String comp() {
        if (currentCommand.contains("=")) {
            if (currentCommand.contains(";")) {
                return currentCommand.split("=")[1].split(";")[0];
            } else {
                return currentCommand.split("=")[1];
            }
        } else if (currentCommand.contains(";")) {
            return currentCommand.split(";")[0];
        }
        return null;
    }

    public String jump() {
        if (currentCommand.contains(";")) {
            return currentCommand.split(";")[1];
        }
        return null;
    }

    public int getCurrentCommandCounter() {
        return currentCommandCounter;
    }

    public void setCurrentCommandCounter(int counter) {
        this.currentCommandCounter = counter;
    }

    public void close() throws IOException {
        reader.close();
    }
}