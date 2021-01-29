"""A classification example where we we only have asymptotically 1/4th of parameterized gates."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.circuit.parametertable import ParameterTable
from qiskit.opflow import H

from .benchmark import Benchmark

class Classification(QuantumCircuit):
    """UCCSD with the same API as Qiskit's circuit library."""

    def __init__(self, num_qubits, reps=3):
        self._reps = reps
        super().__init__(num_qubits)
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
        if not isinstance(params, (list, np.ndarray)):
            params = dict(zip(self._params[:], params))

        return super().assign_parameters(params, inplace=inplace)

    def _build(self):
        # wipe current state
        self._data = []
        self._parameter_table = ParameterTable()

        # get UCCSD circuit
        featmap = ZFeatureMap(self.num_qubits, reps=self.reps)
        ansatz = RealAmplitudes(self.num_qubits, reps=self.reps, entanglement='circular')

        # store the parameters in a list for assigning them
        self._params = ansatz.ordered_parameters

        # set the data circuit with some input data
        featmap.assign_parameters(np.random.random(featmap.num_parameters), inplace=True)

        # combine the circuit
        self.compose(featmap, inplace=True)
        self.compose(ansatz, inplace=True)

        print(self.draw())


def run_featuremap():
    circuit = Classification(4)

    benchmark = Benchmark(2 ** np.arange(2, 5), H, 10)
    benchmark.run_benchmark(circuit)
    benchmark.plot(show=True)
