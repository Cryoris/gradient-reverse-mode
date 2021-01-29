"""Run a UCCSD example, which has a lot more unparameterized gates than parameterized ones."""

import numpy as np
from qiskit import transpile
from qiskit.circuit import QuantumCircuit, ParameterVector, QuantumRegister
from qiskit.circuit.parametertable import ParameterTable
from qiskit.chemistry.components.variational_forms import UCCSD as UCCSDVarForm
from qiskit.opflow import H
from .benchmark import Benchmark

class UCCSD(QuantumCircuit):
    """UCCSD with the same API as Qiskit's circuit library."""

    def __init__(self, num_orbitals, num_particles, reps=3):
        self._reps = reps
        self._num_particles = num_particles
        self._num_orbitals = num_orbitals
        self._params = None

        super().__init__()
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
        self.qregs = []
        self._qubits = []
        self._parameter_table = ParameterTable()
        print(self.parameters)

        # get UCCSD circuit
        uccsd = UCCSDVarForm(self._num_orbitals, self._num_particles, self.reps)
        params = ParameterVector('th', uccsd.num_parameters)
        circuit = uccsd.construct_circuit(params)

        # transpile to a basis gate set we can handle in the gradient calculation
        transpiled = transpile(circuit, basis_gates=['sx', 'rz', 'cx'])

        # add to self and store the parametervector for assigning
        self._params = params
        qreg = QuantumRegister(uccsd.num_qubits)
        self.add_register(qreg)
        self.compose(transpiled, inplace=True)

def run_uccsd():
    circuit = UCCSD(4, 2)  # 1/5th of the gates are parameterized
    benchmark = Benchmark(2 ** np.arange(2, 5), H, 10)
    benchmark.run_benchmark(circuit, 'free')
    benchmark.plot(show=True)
