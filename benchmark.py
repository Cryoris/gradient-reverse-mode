import time
import tracemalloc
from multiprocessing import Pool
from memory_profiler import memory_usage
import numpy as np
import matplotlib.pyplot as plt
from gradients import StateGradient
from scipy.optimize import curve_fit
from qiskit.quantum_info import Statevector

NUM_PROCESSES = 2


class Benchmark:
    def __init__(self, reps, single_qubit_op, nreps):
        self.num_reps = reps
        self.single_qubit_op = single_qubit_op
        self.nreps = nreps
        self.verbose = False

    def run_benchmark(self, library_circuit, filename=None):
        # store the runtime as dict in the following format
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

            # compute the average over nreps repetitions

            # run the number of runs in parallel
            args = self.nreps * [(ansatz, op, state_in)]
            with Pool(processes=NUM_PROCESSES) as pool:
                all_results = pool.map(single_run, args)

            # extract the results
            grad_runtimes, itgrad_runtimes = [], []
            for (grad_time, itgrad_time) in all_results:
                grad_runtimes.append(grad_time)
                itgrad_runtimes.append(itgrad_time)

            num_parameters = library_circuit.num_parameters
            runtime['num_parameters'].append(num_parameters)
            runtime['grad']['avg'].append(np.mean(grad_runtimes))
            runtime['grad']['std'].append(np.std(grad_runtimes))
            runtime['itgrad']['avg'].append(np.mean(itgrad_runtimes))
            runtime['itgrad']['std'].append(np.std(itgrad_runtimes))

        self.last_run = runtime

        if filename is None:
            filename = f'{library_circuit.name}_' + \
                f'q{num_qubits}_r{self.num_reps[0]}_{self.num_reps[-1]}'

        print(self.last_run)

        self.store_benchmark(filename)

    def store_benchmark(self, filename):
        np.save(filename, self.last_run)

    def load_benchmark(self, filename):
        return np.load(filename, allow_pickle=True).item()

    def plot(self, filename=None, saveas=None, show=False, cutoffs=None):
        if filename is None:
            try:
                data = self.last_run
            except NameError:
                raise RuntimeError('Run a benchmark or pass a filename.')
        else:
            data = self.load_benchmark(filename)

        colors = ['tab:blue', 'tab:orange']
        markers = ['o', '^']
        linestyles = ['--', ':']
        methods = ['grad', 'itgrad']
        labels = ['reference', 'reverse mode']
        if cutoffs is None:
            cutoffs = [0, 0]

        plt.figure(figsize=(4, 3))
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

        # plt.title('Gradient run comparison')
        plt.xlabel('number of ansatz parameters, $P$')
        plt.ylabel('time [$s$]')
        ax = plt.axes()
        # plt.text(-0.1, 1, '$(a)$', transform=ax.transAxes, usetex=True, fontsize=12)

        handles, labels = plt.gca().get_legend_handles_labels()
        order = [2, 3, 0, 1]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='best',
                   ncol=2)
        # plt.legend(loc='best', ncol=2)
        plt.xticks([50, 100, 500], [r'$0.5 \cdot 10^2$', r'$10^2$', r'$0.5 \cdot 10^3$'])
        if saveas is None:
            saveas = f'ep_r{self.num_reps[0]}_{self.num_reps[-1]}_{key}.pdf'

        plt.grid()
        plt.savefig('img/' + saveas, bbox_inches='tight')
        if show:
            plt.show()


def single_run(arg):
    ansatz, op, state_in = arg
    runtimes = []
    grad = StateGradient(op, ansatz, state_in)
    for method in ['reference_gradients', 'iterative_gradients']:
        t0 = time.time()
        _ = getattr(grad, method)()  # run gradient computation
        te = time.time()
        peak = 0
        runtimes.append(te - t0)

    return runtimes
