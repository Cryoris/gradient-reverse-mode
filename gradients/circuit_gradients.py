from functools import reduce
from qiskit.circuit import QuantumCircuit
from .split_circuit import split
from .gradient_lookup import analytic_gradient


class StateGradient:

    def __init__(self, operator, ansatz, state_in):
        self.operator = operator
        self.ansatz = ansatz
        self.state_in = state_in

        if isinstance(ansatz, QuantumCircuit):
            self.unitaries = split(ansatz)
        elif isinstance(ansatz, list):
            self.unitaries = ansatz
        else:
            raise NotImplementedError('Unsupported type of ansatz.')

    def reference_gradients(self):
        ulist = self.unitaries
        lam = reduce(lambda x, y: x.evolve(y), ulist, self.state_in).evolve(self.operator)
        num_parameters = len(ulist)
        grads = []
        for j in range(num_parameters):
            grad = 0
            for coeff, gate in analytic_gradient(ulist[j]):
                dj_ulist = ulist[:max(0, j)] + [gate] + ulist[min(num_parameters, j + 1):]
                phi = reduce(lambda x, y: x.evolve(y), dj_ulist, self.state_in)
                grad += coeff * lam.conjugate().data.dot(phi.data)
            grads += [2 * grad.real]
        return grads

    def iterative_gradients(self):
        ulist = self.unitaries
        phi = reduce(lambda x, y: x.evolve(y), ulist, self.state_in)
        # phi = init.evolve(ansatz)
        lam = phi.evolve(self.operator)

        num_parameters = len(ulist)
        grads = []
        for j in reversed(range(num_parameters)):
            uj = ulist[j]
            deriv = analytic_gradient(uj)
            uj_dagger = uj.inverse()

            phi = phi.evolve(uj_dagger)
            # TODO use projection
            grad = 2 * sum(coeff * lam.conjugate().data.dot(phi.evolve(gate).data)
                           for coeff, gate in deriv).real
            grads += [grad]

            if j > 0:
                lam = lam.evolve(uj_dagger)

        return list(reversed(grads))

    def iterative_gradients_selective(self, parameters, parameter_binds=None):
        op, ansatz, init = self.operator, self.ansatz, self.state_in
        if isinstance(ansatz, QuantumCircuit):
            ulist, paramlist = split(ansatz, parameters, return_parameters=True)
        else:
            raise NotImplementedError('Unsupported type of ansatz.')

        if parameter_binds is not None:
            ansatz = ansatz.assign_parameters({p: parameter_binds[p] for p in ansatz.parameters})

        phi = init.evolve(ansatz)
        lam = phi.evolve(op)
        num_parameters = len(ulist)
        grads = []
        for j in reversed(range(num_parameters)):
            uj = ulist[j]
            deriv = analytic_gradient(uj, paramlist[j][0] if parameters != 'all' else None)
            for coeff, gate in deriv:
                gate.assign_parameters({p: parameter_binds[p]
                                        for p in gate.parameters}, inplace=True)

            # TODO: need to decompose here, Parameter bug
            uj_dagger = uj.assign_parameters(
                {p: parameter_binds[p] for p in uj.decompose().parameters}
            ).inverse()

            phi = phi.evolve(uj_dagger)
            # TODO use projection
            grad = 2 * sum(coeff * lam.conjugate().data.dot(phi.evolve(gate).data)
                           for coeff, gate in deriv).real
            grads += [grad]

            if j > 0:
                lam = lam.evolve(uj_dagger)

        return list(reversed(grads))
