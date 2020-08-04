import itertools
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import (
    RXGate, RYGate, RZGate, CRXGate, CRYGate, CRZGate
)


def gradient_lookup(gate):
    """Returns a circuit implementing the gradient of the input gate."""

    param = gate.params[0]
    if isinstance(gate, RXGate):
        derivative = QuantumCircuit(gate.num_qubits)
        derivative.rx(param, 0)
        derivative.x(0)
        return [[0.5j, derivative]]
    if isinstance(gate, RYGate):
        derivative = QuantumCircuit(gate.num_qubits)
        derivative.ry(param, 0)
        derivative.y(0)
        return [[0.5j, derivative]]
    if isinstance(gate, RZGate):
        derivative = QuantumCircuit(gate.num_qubits)
        derivative.rz(param, 0)
        derivative.z(0)
        return [[0.5j, derivative]]
    if isinstance(gate, CRXGate):
        proj1 = QuantumCircuit(gate.num_qubits)
        proj1.rx(param, 1)
        proj1.x(1)

        proj2 = QuantumCircuit(gate.num_qubits)
        proj2.z(0)
        proj2.rx(param, 1)
        proj2.x(1)

        return [[0.25j, proj1], [-0.25j, proj2]]
    if isinstance(gate, CRYGate):
        proj1 = QuantumCircuit(gate.num_qubits)
        proj1.ry(param, 1)
        proj1.y(1)

        proj2 = QuantumCircuit(gate.num_qubits)
        proj2.z(0)
        proj2.ry(param, 1)
        proj2.y(1)

        return [[0.25j, proj1], [-0.25j, proj2]]
    if isinstance(gate, CRZGate):
        proj1 = QuantumCircuit(gate.num_qubits)
        proj1.rz(param, 1)
        proj1.z(1)

        proj2 = QuantumCircuit(gate.num_qubits)
        proj2.z(0)
        proj2.rz(param, 1)
        proj2.z(1)

        return [[0.25j, proj1], [-0.25j, proj2]]
    raise NotImplementedError('Cannot implement for', gate)


def analytic_gradient(circuit, parameter=None):
    """Return the analytic gradient of the input circuit."""

    if parameter is not None:
        if parameter not in circuit.parameters:
            raise ValueError('Parameter not in this circuit.')

        if len(circuit._parameter_table[parameter]) > 1:
            raise NotImplementedError('No product rule support yet, params must be unique.')

    summands, op_context = [], []
    for i, op in enumerate(circuit.data):
        gate = op[0]
        op_context += [op[1:]]
        if (parameter is None and len(gate.params) > 0) or parameter in gate.params:
            summands += [gradient_lookup(gate)]
        else:
            summands += [[[1, gate]]]

    gradient = []
    for product_rule_term in itertools.product(*summands):
        summand_circuit = QuantumCircuit(*circuit.qregs)
        coeff = 1
        for i, a in enumerate(product_rule_term):
            coeff *= a[0]
            summand_circuit.data.append([a[1], *op_context[i]])
        gradient += [[coeff, summand_circuit.copy()]]

    return gradient
