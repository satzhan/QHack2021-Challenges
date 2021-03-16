#! /usr/bin/python3
import sys
import pennylane as qml
from pennylane import numpy as np

# DO NOT MODIFY any of these parameters
a = 0.7
b = -0.3
dev = qml.device("default.qubit", wires=3)


def natural_gradient(params):
    

    natural_grad = np.zeros(6)

    # QHACK #
    def parameter_shift_term(qnode, params, i):
        shifted = params.copy()
        s = np.pi/2
        shifted[i] += s
        forward = qnode(shifted)
        shifted[i] -= 2 * s
        backward = qnode(shifted)
        return 0.5 * (forward - backward) / np.sin(s)
    
    def parameter_shift_func(qnode, params):
        gradients = np.zeros((len(params)))
        for i in range(len(params)):
            gradients[i] = parameter_shift_term(qnode, params, i)
        return gradients
    
    gradient = parameter_shift_func(qnode, params)
    
    def parameter_shift_terms(qnode, params, i, j):
        shifted = params.copy()
        #shifted = np.zeros(len(params))
        s = np.pi / 2
#         shifted[i] += s + params[i]
#         shifted[j] += s + params[j]
        shifted[i] += s
        shifted[j] += s
        first = qnode(params, shifted)[0]
        shifted[j] -= s * 2
        second = qnode(params, shifted)[0]
        shifted[i] -= s * 2
        forth = qnode(params, shifted)[0]
        shifted[j] += s * 2
        third = qnode(params, shifted)[0]
        return 0.125 * (-first + second + third - forth)
    
    def full_metric(qnode, params):
        F = np.zeros((len(params), len(params)))
        for i in range(len(params)):
            for j in range(len(params)):
                F[i][j] = parameter_shift_terms(qnode, params, i, j)
        return F
    
    
    F = full_metric(qnode_probs, params)
    
#     print(gradient)
#     print("FFFFFFF_FUNC")
#     print(np.diag(
#         np.round(
#             qml.metric_tensor(qnode)(params),
#             3
#         )
#     ))
#     print("FFFFFFFFFFF")
#     print(np.round(np.diag(F),3))
#     natural_grad = np.linalg.inv(
#         qml.metric_tensor(qnode)(params)
#     ).dot(gradient)
    
    natural_grad = np.linalg.inv(
        F
    ).dot(gradient)
    
#     print(natural_grad.shape)
    
    # QHACK #

    return natural_grad


def non_parametrized_layer():
    """A layer of fixed quantum gates.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    qml.RX(a, wires=0)
    qml.RX(b, wires=1)
    qml.RX(a, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.RZ(a, wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(b, wires=1)
    qml.Hadamard(wires=0)


def variational_circuit(params):
    """A layered variational circuit composed of two parametrized layers of single qubit rotations
    interleaved with non-parameterized layers of fixed quantum gates specified by
    ``non_parametrized_layer``.

    The first parametrized layer uses the first three parameters of ``params``, while the second
    layer uses the final three parameters.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    non_parametrized_layer()
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)
    non_parametrized_layer()
    qml.RX(params[3], wires=0)
    qml.RY(params[4], wires=1)
    qml.RZ(params[5], wires=2)

@qml.template
def variational_circuit_inv(params):

    non_parametrized_layer()
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)
    non_parametrized_layer()
    qml.RX(params[3], wires=0)
    qml.RY(params[4], wires=1)
    qml.RZ(params[5], wires=2)


@qml.qnode(dev)
def qnode(params):
    """A PennyLane QNode that pairs the variational_circuit with an expectation value
    measurement.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    variational_circuit(params)
    return qml.expval(qml.PauliX(1))

@qml.qnode(dev)
def qnode_probs(params,shifted_params):
    variational_circuit(params)
    qml.inv(variational_circuit_inv(shifted_params))
    return qml.probs(wires=[0,1,2])


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process inputs
    params = sys.stdin.read()
    params = params.split(",")
    params = np.array(params, float)

    updated_params = natural_gradient(params)

    print(*updated_params, sep=",")
