# Variational Quantum Classifier (VQC) with PennyLane

This repository contains a from-scratch implementation of a Variational Quantum Classifier built using PennyLane and Python. It demonstrates a hybrid classical-quantum machine learning loop capable of binary classification.

## Project Architecture
* **Frameworks:** PennyLane, NumPy, scikit-learn, Matplotlib.
* **Feature Encoding:** Angle Embedding (maps classical data into quantum states via Y-axis rotations).
* **Ansatz:** Strongly Entangling Layers (acts as the trainable quantum weights).
* **Optimizer:** Classical Gradient Descent using the parameter-shift rule.
* **Measurement:** Pauli-Z expectation value mapped to continuous values [-1, 1].

## Results
The model successfully maps classical data into a quantum space and uses parameterized entanglement to classify the dataset. Because the VQC utilizes the spherical geometry of qubits, it naturally carves out complex, non-linear decision boundaries.
