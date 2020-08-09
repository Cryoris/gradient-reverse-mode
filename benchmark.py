import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
from gradients import StateGradient
from qiskit.quantum_info import Statevector


class Benchmark:
    def __init__(self, reps, single_qubit_op, nreps):
        self.num_reps = reps
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
        for reps in self.num_reps:
            # resize and parameterize library circuit
            library_circuit.reps = reps
            ansatz = library_circuit.assign_parameters(
                np.random.random(library_circuit.num_parameters)
            )

            # get operator and input state of proper size
            num_qubits = library_circuit.num_qubits
            op = (self.single_qubit_op ^ num_qubits).to_matrix_op().primitive
            state_in = Statevector.from_label('0' * num_qubits)

            if self.verbose:
                print()
                print('num_qubits:', num_qubits)
                print('reps:', reps)
                print('runs:')

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

            num_parameters = library_circuit.num_parameters
            runtime['grad']['avg'][num_parameters] = np.mean(grad_runtimes)
            runtime['grad']['std'][num_parameters] = np.std(grad_runtimes)
            runtime['itgrad']['avg'][num_parameters] = np.mean(itgrad_runtimes)
            runtime['itgrad']['std'][num_parameters] = np.std(itgrad_runtimes)

            memory['grad']['avg'][num_parameters] = np.mean(grad_memory_peaks)
            memory['grad']['std'][num_parameters] = np.std(grad_memory_peaks)
            memory['itgrad']['avg'][num_parameters] = np.mean(itgrad_memory_peaks)
            memory['itgrad']['std'][num_parameters] = np.std(itgrad_memory_peaks)

        self.last_run = {'runtime': runtime, 'memory': memory}

        if filename is None:
            filename = f'n{self.num_reps[0]}_{self.num_reps[-1]}_c{library_circuit.name}'

        print(self.last_run)

        self.store_benchmark(filename)

    def store_benchmark(self, filename):
        np.save(filename, self.last_run)

    def load_benchmark(self, filename):
        return np.load(filename, allow_pickle=True).item()

    def plot(self, key='runtime', filename=None):
        if filename is None:
            try:
                benchmark = self.last_run
            except NameError:
                raise RuntimeError('Run a benchmark or pass a filename.')
        else:
            benchmark = self.load_benchmark(filename)

        data = benchmark[key]
        nums_parameters = list(data['grad']['avg'].keys())

        colors = ['tab:blue', 'tab:orange']
        markers = ['o', '^']
        methods = ['grad', 'itgrad']

        plt.figure()
        plt.loglog()  # shortcut for getting log scaling on x and y axis
        for method, color, marker in zip(methods, colors, markers):
            avg = list(data[method]['avg'].values())
            std = list(data[method]['std'].values())
            plt.errorbar(nums_parameters, avg, yerr=std, color=color, marker=marker, label=method)

        plt.title(key)
        plt.legend(loc='best')

        if filename is None:
            filename = f'n{self.num_reps[0]}_{self.num_reps[-1]}_{key}.pdf'

        plt.savefig('img/' + filename)
        plt.show()

    @staticmethod
    def single_run(ansatz, op, state_in):
        times, peaks = [], []
        grad = StateGradient(op, ansatz, state_in)
        for method in ['reference_gradients', 'iterative_gradients']:
            tracemalloc.start()
            t0 = time.time()
            _ = getattr(grad, method)()  # run gradient computation
            te = time.time()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            times.append(te - t0)
            peaks.append(peak)

        return times, peaks


def efficient_su2():
    from qiskit.circuit.library import EfficientSU2
    from qiskit.aqua.operators import H

    reps = [2, 4, 6, 8]
    nreps = 10

    b = Benchmark(reps, H, nreps)
    b.verbose = True
    b.run_benchmark(EfficientSU2(5, entanglement='linear'))
    b.plot('runtime')
    b.plot('memory')


if __name__ == '__main__':
    efficient_su2()
