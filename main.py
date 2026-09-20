import pennylane as qml
from pennylane import numpy as np

# Boot up a simulator device with 2 qubits
dev = qml.device("default.qubit", wires=2)

print(f"Quantum device initialized with {len(dev.wires)} qubits!")


def feature_map(x):
     qml.AngleEmbedding(features=x, wires=range(2) ,rotation='Y')
print("Feature map created successfully! ")
def ansatz(weights):
    qml.stronglyEntanglingLayers(weights,wires=range(2))
print("Ansatz created successfully! ")

@qml.qnode(dev)
def circuit(weights,x):
    feature_map(x)
    ansatz(weights)
    return qml.expval(qml.PauliZ(0))
print("Quantum Circuit wired together successfully! ")

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import GradientDescentOptimizer

dev = qml.device("default.qubit", wires=2)


def feature_map(x):
    qml.AngleEmbedding(features=x, wires=range(2), rotation='Y')


def ansatz(weights):
    qml.StronglyEntanglingLayers(weights, wires=range(2))


@qml.qnode(dev)
def circuit(weights, x):
    feature_map(x)
    ansatz(weights)
    return qml.expval(qml.PauliZ(0))


def variational_classifier(weights, bias, x):
    return circuit(weights, x) + bias


def cost(weights, bias, X, Y):
    predictions = np.array([variational_classifier(weights, bias, x) for x in X])
    return np.mean((Y - predictions) ** 2)


from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


print("Loading real-world Iris data...")
iris = load_iris()

X_raw = iris.data[:, :2]
Y_raw = iris.target

X_filtered = X_raw[Y_raw != 2]
Y_filtered = Y_raw[Y_raw != 2]

Y_train = np.array([1.0 if y == 1 else -1.0 for y in Y_filtered], requires_grad=False)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_filtered)

print(f"Successfully loaded and scaled {len(X_train)} real data points!")

weights = np.random.randn(3, 2, 3, requires_grad=True)
bias = np.array(0.0, requires_grad=True)
opt = GradientDescentOptimizer(stepsize=0.1)

print("Starting training...")
for i in range(15):
    (weights, bias), cost_val = opt.step_and_cost(lambda w, b: cost(w, b, X_train, Y_train), weights, bias)

    val = cost_val._value if hasattr(cost_val, '_value') else cost_val
    print(f"Epoch {i + 1} | cost: {float(val):.4f}")

def accuracy(labels,predictions):
    correct = sum(1 for l , p in zip(labels,predictions) if np.sign(p)==l)
    return correct / len(labels)
final_predictions = [variational_classifier(weights, bias, x) for x in X_train]
acc =accuracy(Y_train,final_predictions)
print(f"final Accuracy: {acc * 100:.2f}%")

import matplotlib.pyplot as plt
print("Generating quantum decision boundry graph...")
xx,yy =np.meshgrid(np.linspace(-3,3,30),np.linspace(-3,3,30))
grid_points=np.column_stack((xx.ravel(),yy.ravel()))

predictions = [float(variational_classifier(weights, bias, x)) for x in grid_points]
z= np.sign(predictions).reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx,yy,z,alpha=0.4,cmap="RdBu")
plt.scatter(X_train[:,0], X_train[:,1], c=Y_train,cmap="RdBu", edgecolors="k")
plt.title("Quantum VQC Decision Boundry Graph")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
