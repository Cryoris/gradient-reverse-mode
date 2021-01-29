"""Run a benchmark for Qiskit's efficient SU(2) circuit."""

import numpy as np
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import H
from .benchmark import Benchmark

def run_efficientsu2():
    # define the circuit to run the benchmark on
    circuit = EfficientSU2(4)

    # define the stats for the benchmark
    benchmark = Benchmark(2 ** np.arange(2, 6), H, 12)

    # run 
    benchmark.run_benchmark(circuit)

    # and plot
    benchmark.plot(show=True)
