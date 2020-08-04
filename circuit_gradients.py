from functools import reduce
from qiskit.circuit import QuantumCircuit
from split_circuit import split
from gradient_lookup import analytic_gradient


def grad(ansatz, op, init):
    if isinstance(ansatz, QuantumCircuit):
        ulist = split(ansatz)
    elif isinstance(ansatz, list):
        ulist = ansatz
    else:
        raise NotImplementedError('Unsupported type of ansatz.')

    lam = reduce(lambda x, y: x.evolve(y), ulist, init).evolve(op)
    num_parameters = len(ulist)
    grads = []
    for j in range(num_parameters):
        grad = 0
        for coeff, gate in analytic_gradient(ulist[j]):
            dj_ulist = ulist[:max(0, j)] + [gate] + ulist[min(num_parameters, j + 1):]
            phi = reduce(lambda x, y: x.evolve(y), dj_ulist, init)
            grad += coeff * lam.conjugate().data.dot(phi.data)
        grads += [2 * grad.real]
    return grads


def itgrad(ansatz, op, init, parameters='all', parameter_binds=None):
    # safeguard, otherwise states might be changed
    ansatz = ansatz.copy()
    op = op.copy()
    init = init.copy()

    if isinstance(ansatz, QuantumCircuit):
        ulist, paramlist = split(ansatz, parameters, return_parameters=True)
    elif isinstance(ansatz, list):
        ulist = ansatz
        empty = QuantumCircuit(ulist[0].num_qubits)
        ansatz = reduce(lambda x, y: x.compose(y), ulist, empty)
    else:
        raise NotImplementedError('Unsupported type of ansatz.')

    phi = init.evolve(ansatz.bind_parameters({p: parameter_binds[p] for p in ansatz.parameters}))

    lam = phi.evolve(op)
    num_parameters = len(ulist)
    grads = []
    for j in reversed(range(num_parameters)):
        uj = ulist[j]
        deriv = analytic_gradient(uj, paramlist[j][0] if parameters != 'all' else None)
        for coeff, gate in deriv:
            gate.assign_parameters({p: parameter_binds[p] for p in gate.parameters}, inplace=True)

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
