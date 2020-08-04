from unittest import TestCase, main
import numpy as np
from ddt import ddt, data
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import Statevector

from circuit_gradients import grad, itgrad


@ddt
class TestGradients(TestCase):
    """Test the computation of the gradients."""

    @data(itgrad, grad)
    def test_rx(self, gradient_function):
        # the input state
        init = Statevector.from_label('1')

        # we compute the gradient w.r.t. to the state ansatz|init>
        ansatz = QuantumCircuit(1)
        value = 0.681
        ansatz.rx(value, 0)

        # the operator for computing the expectation value
        op = QuantumCircuit(1)
        op.h(0)

        grads = gradient_function(ansatz, op, init)

        # reference value
        ref = [-0.4451734157850243]

        np.testing.assert_array_almost_equal(grads, ref)

    @data(itgrad, grad)
    def test_rxry(self, gradient_function):
        p = [0.8, 0.2]

        ansatz = QuantumCircuit(1)
        ansatz.rx(p[0], 0)
        ansatz.ry(p[1], 0)

        op = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        init = Statevector.from_int(1, dims=(2,))
        grads = gradient_function(ansatz, op, init)

        # reference value
        ref = [-0.5979106735501365, 0.3849522583908403]

        np.testing.assert_array_almost_equal(grads, ref)

    def test_partial_gradient(self):
        p = ParameterVector('p', 2)
        values = [0.8, 0.2]

        ansatz = QuantumCircuit(1)
        ansatz.rx(p[0], 0)
        ansatz.ry(p[1], 0)

        op = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        init = Statevector.from_int(1, dims=(2,))
        grad = itgrad(ansatz, op, init, [p[1]], dict(zip(p, values)))[0]
        # reference value
        ref = 0.3849522583908403

        self.assertAlmostEqual(grad, ref)


if __name__ == '__main__':
    main()
