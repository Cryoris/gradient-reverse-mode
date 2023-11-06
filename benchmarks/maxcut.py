"""Maxcut example."""

import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import QAOAAnsatz
from qiskit.circuit.parametertable import ParameterTable
from qiskit.opflow import Z, I, H

from .benchmark import Benchmark

def run_maxcut():
    operator = (I ^ I ^ Z ^ Z) + (I ^ Z ^ I ^ Z) + (Z ^ I ^ I ^ Z) + (I ^ Z ^ Z ^ I)
    # operator = I + Z
    circuit = QAOAAnsatz(operator)

    benchmark = Benchmark(2 ** np.arange(0, 7), H, 24)
    # benchmark = Benchmark(2 ** np.arange(0, 12), H, 24)
    benchmark.run_benchmark(circuit, 'free')
    benchmark.plot(show=True)
