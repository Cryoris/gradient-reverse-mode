import time
import tracemalloc
import numpy as np
from gradients import grad, itgrad
from qiskit.quantum_info import Statevector


class Benchmark:
    def __init__(self, nums_qubits, single_qubit_op, nreps):
        self.nums_qubits = nums_qubits
        self.single_qubit_op = single_qubit_op
        self.nreps = nreps
        self.verbose = False

    def run_benchmark(self, library_circuit, filename=None):
        # store the runtime and memory as dicts, such that
        # runtime = {'grad':
        #             {'avg': {num_qubits: avg_runtime, ...}
        #              'std': {num_qubits, std_runtime, ...}
        #             },
        #            'itgrad': ...}
        runtime, memory = 2 * [{'grad': {'avg': {}, 'std': {}},
                                'itgrad': {'avg': {}, 'std': {}}}]
        for num_qubits in self.nums_qubits:
            if self.verbose:
                print()
                print('num_qubits:', num_qubits)
                print('runs:')
            # get operator and input state of proper size
            op = (self.single_qubit_op ^ num_qubits).to_matrix_op().primitive
            state_in = Statevector.from_label('0' * num_qubits)

            # resize and parameterize library circuit
            library_circuit.num_qubits = num_qubits
            ansatz = library_circuit.assign_parameters(
                np.random.random(library_circuit.num_parameters)
            )

            # compute the average over nreps repetitions
            grad_runtimes, grad_memory_peaks = [], []
            itgrad_runtimes, itgrad_memory_peaks = [], []
            for i in range(self.nreps):
                if self.verbose:
                    print(i, end=' ')
                times, peaks = self.single_run(ansatz, op, state_in)
                grad_runtimes.append(times[0])
                grad_memory_peaks.append(peaks[0])
                itgrad_runtimes.append(times[1])
                itgrad_memory_peaks.append(peaks[1])

            runtime['grad']['avg'][num_qubits] = np.mean(grad_runtimes)
            runtime['grad']['std'][num_qubits] = np.std(grad_runtimes)
            runtime['itgrad']['avg'][num_qubits] = np.mean(itgrad_runtimes)
            runtime['itgrad']['std'][num_qubits] = np.std(itgrad_runtimes)

            memory['grad']['avg'][num_qubits] = np.mean(grad_memory_peaks)
            memory['grad']['std'][num_qubits] = np.std(grad_memory_peaks)
            memory['itgrad']['avg'][num_qubits] = np.mean(itgrad_memory_peaks)
            memory['itgrad']['std'][num_qubits] = np.std(itgrad_memory_peaks)

        self.last_run = {'runtime': runtime, 'memory': memory}

        if filename is None:
            filename = f'n{self.nums_qubits[0]}_{self.nums_qubits[-1]}_c{library_circuit.name}'

        print(self.last_run)

        self.store_benchmark(filename)

    def store_benchmark(self, filename):
        np.save(filename, self.last_run)

    def load_benchmark(self, filename):
        return np.load(filename).item()

    @staticmethod
    def single_run(ansatz, op, state_in):
        times, peaks = [], []
        for method in [grad, itgrad]:
            tracemalloc.start()
            t0 = time.time()
            _ = method(ansatz, op, state_in)
            te = time.time()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            times.append(te - t0)
            peaks.append(peak)

        return times, peaks


def efficient_su2():
    from qiskit.circuit.library import EfficientSU2
    from qiskit.aqua.operators import H

    nums_qubits = [2, 4, 6, 8, 10]
    nreps = 10

    b = Benchmark(nums_qubits, H, nreps)
    b.verbose = True
    b.run_benchmark(EfficientSU2())


efficient_su2()
