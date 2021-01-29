from unittest import TestCase, main
import numpy as np
from ddt import ddt, data
from qiskit.opflow import I, X, Y, Z, Gradient, StateFn
from qiskit.circuit import QuantumCircuit, ParameterVector, Parameter
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap
from qiskit.quantum_info import Statevector

# from gradients import grad, itgrad
from gradients import StateGradient


@ddt
class TestGradients(TestCase):
    """Test the computation of the gradients."""

    @data('reference_gradients', 'iterative_gradients')
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

        # set up gradient object and compute gradient
        grad = StateGradient(op, ansatz, init)
        grads = getattr(grad, gradient_function)()

        # reference value
        ref = [-0.4451734157850243]

        np.testing.assert_array_almost_equal(grads, ref)

    @data('reference_gradients', 'iterative_gradients')
    def test_rxry(self, gradient_function):
        p = [0.8, 0.2]

        ansatz = QuantumCircuit(1)
        ansatz.rx(p[0], 0)
        ansatz.ry(p[1], 0)

        op = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        init = Statevector.from_int(1, dims=(2,))

        # set up gradient object and compute gradient
        grad = StateGradient(op, ansatz, init)
        grads = getattr(grad, gradient_function)()

        # reference value
        ref = [-0.5979106735501365, 0.3849522583908403]

        np.testing.assert_array_almost_equal(grads, ref)

    @data('reference_gradients', 'iterative_gradients')
    def test_larger_circuit(self, gradient_function):
        op = (Y ^ Z) + 3 * (X ^ X) + (Z ^ I) + (I ^ Z) + (I ^ X)
        op = op.to_matrix_op().primitive

        theta = [0.275932, 0.814824, 0.670661, 0.627729, 0.596198]

        ansatz = QuantumCircuit(2)
        ansatz.h([0, 1])
        ansatz.ry(theta[0], 0)
        ansatz.ry(theta[1], 1)
        ansatz.rz(theta[2], 0)
        ansatz.rz(theta[3], 1)
        ansatz.cx(0, 1)
        ansatz.crx(theta[4], 1, 0)

        init = Statevector.from_label('00')

        # set up gradient object and compute gradient
        grad = StateGradient(op, ansatz, init)
        grads = getattr(grad, gradient_function)()

        ref = [1.2990890015773053, 0.47864124516756174, 1.9895319019377231,
               0.09137636702470253, 0.40256649191876637]

        np.testing.assert_array_almost_equal(grads, ref)

    def test_partial_gradient(self):
        p = ParameterVector('p', 2)
        values = [0.8, 0.2]

        ansatz = QuantumCircuit(1)
        ansatz.rx(p[0], 0)
        ansatz.ry(p[1], 0)

        op = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        init = Statevector.from_int(1, dims=(2,))

        # set up gradient object and compute gradient
        grad = StateGradient(op, ansatz, init, [p[1]])
        grads = grad.iterative_gradients(dict(zip(p, values)))[0]

        # reference value
        ref = 0.3849522583908403

        self.assertAlmostEqual(grads, ref)

    @data('reference_gradients', 'iterative_gradients')
    def test_partial_large_circuit(self, method):
        np.random.seed(21)

        featuremap = ZFeatureMap(2, reps=1)
        featuremap.assign_parameters(np.random.random(
            featuremap.num_parameters), inplace=True)

        ansatz = RealAmplitudes(2, reps=1)
        params = ansatz.ordered_parameters[:]
        values = np.random.random(ansatz.num_parameters)
        init = Statevector.from_int(1, dims=(2, 2))

        circuit = featuremap.compose(ansatz)

        grad = StateGradient(X ^ X, circuit, init, params)
        grads = getattr(grad, method)(dict(zip(params, values)))
        ref = [-0.7700884147948044, 0.011116605029003569, -0.6889501710944109, 
               -0.07972088641561373]

        np.testing.assert_array_almost_equal(grads, ref)

    @data('reference_gradients', 'iterative_gradients')
    def test_product_rule(self, method):
        x = Parameter('x')
        circuit = QuantumCircuit(1)
        circuit.rx(x, 0)
        circuit.ry(x, 0)
        circuit.rx(x, 0)

        state_in = Statevector.from_int(1, dims=(2,))

        grad = StateGradient(Z, circuit, state_in, [x])
        grads = getattr(grad, method)({x: 1})

        print(Gradient().convert(~StateFn(Z) @ StateFn(circuit), params=[x]).bind_parameters({x: 1}).eval())



if __name__ == '__main__':
    main()
