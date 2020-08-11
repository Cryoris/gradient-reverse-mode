import time
import tracemalloc
from multiprocessing import Pool
from memory_profiler import memory_usage
import numpy as np
import matplotlib.pyplot as plt
from gradients import StateGradient
from scipy.optimize import curve_fit
from qiskit.quantum_info import Statevector

NUM_PROCESSES = 3


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
        runtime = {'num_parameters': [],
                   'grad': {'avg': [], 'std': []},
                   'itgrad': {'avg': [], 'std': []}}
        memory = {'num_parameters': [],
                  'grad': {'avg': [], 'std': []},
                  'itgrad': {'avg': [], 'std': []}}

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

            # run the number of runs in parallel
            args = self.nreps * [(ansatz, op, state_in)]
            with Pool(processes=NUM_PROCESSES) as pool:
                all_results = pool.map(single_run, args)

            # extract the results
            grad_runtimes, grad_memory_peaks = [], []
            itgrad_runtimes, itgrad_memory_peaks = [], []
            for (times, peaks) in all_results:
                grad_runtimes.append(times[0])
                grad_memory_peaks.append(peaks[0])
                itgrad_runtimes.append(times[1])
                itgrad_memory_peaks.append(peaks[1])

            num_parameters = library_circuit.num_parameters
            runtime['num_parameters'].append(num_parameters)
            runtime['grad']['avg'].append(np.mean(grad_runtimes))
            runtime['grad']['std'].append(np.std(grad_runtimes))
            runtime['itgrad']['avg'].append(np.mean(itgrad_runtimes))
            runtime['itgrad']['std'].append(np.std(itgrad_runtimes))

            memory['num_parameters'].append(num_parameters)
            memory['grad']['avg'].append(np.mean(grad_memory_peaks))
            memory['grad']['std'].append(np.std(grad_memory_peaks))
            memory['itgrad']['avg'].append(np.mean(itgrad_memory_peaks))
            memory['itgrad']['std'].append(np.std(itgrad_memory_peaks))

        self.last_run = {'runtime': runtime, 'memory': memory}

        if filename is None:
            filename = f'ep_r{self.num_reps[0]}_{self.num_reps[-1]}_c{library_circuit.name}'

        print(self.last_run)

        self.store_benchmark(filename)

    def store_benchmark(self, filename):
        np.save(filename, self.last_run)

    def load_benchmark(self, filename):
        return np.load(filename, allow_pickle=True).item()

    def plot(self, key='runtime', filename=None, saveas=None, show=False, cutoffs=None):
        if filename is None:
            try:
                benchmark = self.last_run
            except NameError:
                raise RuntimeError('Run a benchmark or pass a filename.')
        else:
            benchmark = self.load_benchmark(filename)

        data = benchmark[key]

        colors = ['tab:blue', 'tab:orange']
        markers = ['o', '^']
        linestyles = ['--', ':']
        methods = ['grad', 'itgrad']
        labels = ['reference', 'reverse mode']
        if cutoffs is None:
            cutoffs = [0, 0]

        plt.figure()
        plt.loglog()  # shortcut for getting log scaling on x and y axis
        for (
            method, color, marker, line, label, cutoff
        ) in zip(
            methods, colors, markers, linestyles, labels, cutoffs
        ):
            avg = data[method]['avg']
            std = data[method]['std']
            nums_parameters = data['num_parameters']
            plt.errorbar(nums_parameters, avg, yerr=std, color=color, marker=marker, label=label)

            avg = avg[cutoff:]
            std = std[cutoff:]
            nums_parameters = nums_parameters[cutoff:]
            (a, b), _ = curve_fit(lambda x, a, b: a * x + b, np.log(nums_parameters), np.log(avg))
            plt.plot(nums_parameters, nums_parameters ** a * np.exp(b), 'k' + line,
                     label=r'$O^{' + f'{np.round(a, 2)}' + r'}$')

        plt.title('Gradient ' + key + ' comparison')
        plt.xlabel('number of ansatz parameters, $P$')
        plt.ylabel('time [$s$]')

        handles, labels = plt.gca().get_legend_handles_labels()
        order = [2, 3, 0, 1]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='best',
                   ncol=2)
        # plt.legend(loc='best', ncol=2)
        plt.xticks([50, 100, 500], [r'$0.5 \cdot 10^2$', r'$10^2$', r'$0.5 \cdot 10^3$'])
        if saveas is None:
            saveas = f'ep_r{self.num_reps[0]}_{self.num_reps[-1]}_{key}.pdf'

        plt.grid()
        plt.savefig('img/' + saveas)
        if show:
            plt.show()


def single_run(arg):
    ansatz, op, state_in = arg
    times, peaks = [], []
    grad = StateGradient(op, ansatz, state_in)
    for method in ['reference_gradients', 'iterative_gradients']:
        # tracemalloc.start()
        t0 = time.time()
        _ = getattr(grad, method)()  # run gradient computation
        # peak = np.sum(memory_usage(getattr(grad, method),
        #                            include_children=True,
        #                            multiprocess=True))
        te = time.time()
        peak = 0
        # peak, _ = tracemalloc.get_traced_memory()
        # tracemalloc.stop()
        times.append(te - t0)
        peaks.append(peak)

    return times, peaks


if __name__ == '__main__':
    from qiskit.circuit.library import EfficientSU2
    from qiskit.aqua.operators import H

    reps = 2 ** np.arange(3, 8)
    nreps = 24

    b = Benchmark(reps, H, nreps)
    b.verbose = True
    b.run_benchmark(EfficientSU2(5, entanglement='linear'))
    b.plot('runtime')
    # b.plot('memory')
