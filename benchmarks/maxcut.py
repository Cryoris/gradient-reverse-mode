"""Maxcut example."""

import numpy as np
from qiskit.algorithms.minimum_eigen_solvers.qaoa.var_form import QAOAVarForm
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.parametertable import ParameterTable
from qiskit.opflow import Z, I, H

from .benchmark import Benchmark

class QAOAAnsatz(QuantumCircuit):
    """QAOA ansatz as a quantum circuit."""

    def __init__(self, operator, reps=1):
        self._reps = reps
        self._operator = operator
        super().__init__(operator.num_qubits)
        self._build()

    @property
    def reps(self):
        """Get the number of repetitions of the circuit."""
        return self._reps

    @reps.setter
    def reps(self, value):
        """Set the number of repetitions. Rebuilds the circuit."""
        self._reps = value
        self._build()  # rebuild

    def assign_parameters(self, params, inplace=False):
        """Assign parameters."""
        if isinstance(params, (list, np.ndarray)):
            params = dict(zip(self._params[:], params))

        return super().assign_parameters(params, inplace=inplace)

    @property
    def ordered_parameters(self):
        return self._params[:]

    def _build(self):
        # wipe current state
        self._data = []
        self._parameter_table = ParameterTable()

        # get QAOA circuit
        qaoa = QAOAVarForm(self._operator, self._reps)
        params = ParameterVector('th', qaoa.num_parameters)
        circuit = qaoa.construct_circuit(params)

        # store the parameters in a list for assigning them
        self._params = params


        # combine the circuit
        self.compose(circuit, inplace=True)


def run_maxcut():
    operator = (I ^ I ^ Z ^ Z) + (I ^ Z ^ I ^ Z) + (Z ^ I ^ I ^ Z) + (I ^ Z ^ Z ^ I)
    circuit = QAOAAnsatz(operator)

    benchmark = Benchmark(2 ** np.arange(2, 8), H, 24)
    benchmark.run_benchmark(circuit, 'free')
    benchmark.plot(show=True)
