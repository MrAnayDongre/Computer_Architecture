# Quantum Peek-a-Boo: Grover's Algorithm Demonstration

This project demonstrates the principles of **Grover's Algorithm**, a quantum search algorithm, by comparing it with a classical search approach. The goal is to find a "marked item" in a dataset using both classical and quantum methods, highlighting the quantum advantage in scaling.

---

## Features

- **Classical Search**: A brute-force search through all possible items.
- **Quantum Search**: Implements Grover's Algorithm to find the marked item with fewer queries.
- **Visualization**:
  - Probability distributions of quantum states during Grover's iterations.
  - Final measurement histogram of the quantum circuit.
  - Full quantum circuit diagram.
- **Comparison**: Highlights the scaling advantage of quantum search over classical search.

---

## Requirements

Ensure the following Python libraries are installed:

- `qiskit`
- `qiskit-aer`
- `matplotlib`
- `pylatexenc` (optional, for better circuit diagrams)

Install them using:

```bash
pip install qiskit qiskit-aer matplotlib pylatexenc