"""A class to run benchmarks of gradient calculation runtimes."""

import time
from multiprocessing import Pool
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from qiskit.quantum_info import Statevector

from gradients import StateGradient

NUM_PROCESSES = 2


class Benchmark:
    """A class to run benchmarks of gradient calculation runtimes."""

    def __init__(self, reps, single_qubit_op, nreps):
        self.num_reps = reps
        self.single_qubit_op = single_qubit_op
        self.nreps = nreps
        self.verbose = False
        self.last_run = None

    def run_benchmark(self, library_circuit, target_parameters='all', filename=None):
        """Run a single benchmark.

        Args:
            library_circuit (QuantumCircuit): A circuit that allows to assign parameters via
                an array of values.
            target_parameters (str): 'all' for all parameters, 'free' for all free parameters.
            filename (str, optional): A file to store the benchmark results. If None is given,
                a default filename is used.
        """
        # store the runtime as dict in the following format
        # runtime = {'grad':
        #             {'avg': {num_qubits: avg_runtime, ...}
        #              'std': {num_qubits, std_runtime, ...}
        #             },
        #            'itgrad': ...}
        runtime = {'num_parameters': [],
                   'grad': {'avg': [], 'std': []},
                   'itgrad': {'avg': [], 'std': []}}

        for reps in self.num_reps:
            # resize and parameterize library circuit
            library_circuit.reps = reps

            # pylint: disable=no-member
            parameters = np.random.random(library_circuit.num_parameters)
            if target_parameters == 'all':
                ansatz = library_circuit.assign_parameters(parameters)
            else:
                ansatz = library_circuit

            # get operator and input state of proper size
            num_qubits = library_circuit.num_qubits
            operator = (self.single_qubit_op ^ num_qubits).to_matrix_op().primitive
            state_in = Statevector.from_label('0' * num_qubits)

            if self.verbose:
                print()
                print('num_qubits:', num_qubits)
                print('reps:', reps)

            # compute the average over nreps repetitions
            # run the number of runs in parallel
            if target_parameters == 'all':
                args = self.nreps * [(ansatz, operator, state_in, None, None)]
            else:
                free_parameters = list(ansatz._parameter_table.keys())
                parameter_binds = dict(zip(free_parameters, parameters))
                args = self.nreps * [(ansatz, operator, state_in, free_parameters,
                                      parameter_binds)]

            # all_results = []
            # for i, arg in enumerate(args):
            #     all_results.append(single_run(arg))
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
        """Store the last benchmark."""
        np.save(filename, self.last_run)

    def load_benchmark(self, filename):
        """Load a benchmark from ``filename``."""
        return np.load(filename, allow_pickle=True).item()

    def plot(self, filename=None, saveas=None, show=False, cutoffs=None):
        """Plot the a set of benchmarks.

        Args:
            filename (str): The filename to load the data from. If None, the last run is used. If
                no last run exists, an error is thrown.
            saveas (str): The filename for the plots. If None, a default name is generated.
            show (bool): If True, shows the plots after saving them using ``pyplot.show()``.
            cutoffs (list): A list with two integers specifying how many runtimes to skip
                for calculating the fit. Can be useful since the asympyotic runtimes can be
                misrepresented for small system sizes.
        """
        if filename is None:
            if self.last_run is None:
                raise RuntimeError('Run a benchmark or pass a filename.')
            data = self.last_run
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
            plt.errorbar(nums_parameters, avg, yerr=std,
                         color=color, marker=marker, label=label)

            avg = avg[cutoff:]
            std = std[cutoff:]
            nums_parameters = nums_parameters[cutoff:]
            (a, b), _ = curve_fit(lambda x, a, b: a *
                                  x + b, np.log(nums_parameters), np.log(avg))
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
        plt.xticks([50, 100, 500], [r'$0.5 \cdot 10^2$',
                                    r'$10^2$', r'$0.5 \cdot 10^3$'])
        if saveas is None:
            saveas = f'ep_r{self.num_reps[0]}_{self.num_reps[-1]}.pdf'

        plt.grid()
        plt.savefig('img/' + saveas, bbox_inches='tight')
        if show:
            plt.show()


def single_run(arg):
    """Execute a single run of the gradient calculation.

    This is in a separate function for multiprocessing.
    """
    ansatz, operator, state_in, target_parameters, parameter_binds = arg
    runtimes = []
    grad = StateGradient(operator, ansatz, state_in, target_parameters)
    for method in ['reference_gradients', 'iterative_gradients']:
        start = time.time()
        _ = getattr(grad, method)(parameter_binds)  # run gradient computation
        runtimes.append(time.time() - start)

    return runtimes
