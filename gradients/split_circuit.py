from qiskit.circuit import QuantumCircuit, ParameterExpression, Parameter


def split(circuit, parameters='all',
          separate_parameterized_gates=False,
          return_parameters=False):
    """Split the circuit at ParameterExpressions.

    Args:
        circuit: The circuit to split.
        separate_paramterized_gates: If True, parameterized gates are in an individual block,
            otherwise they contain also preceding non-parameterized gates.

    Returns:
        A list of the split circuits.
    """
    circuits = []
    corresponding_parameters = []

    sub = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    for op in circuit.data:
        # check if new split must be created
        if parameters == 'all':
            params = op[0].params
        elif parameters == 'free':
            params = [param for param in op[0].params if isinstance(param, ParameterExpression)]
        elif isinstance(parameters, Parameter):
            if op[0].definition is not None:
                free_op_params = op[0].definition.parameters
            else:
                free_op_params = {}
            params = [parameters] if parameters in free_op_params else []
        elif isinstance(parameters, list):
            if op[0].definition is not None:
                free_op_params = op[0].definition.parameters
            else:
                free_op_params = {}
            params = [p for p in parameters if p in free_op_params]
        else:
            raise NotImplementedError('Unsupported type of parameters:', parameters)

        new_split = bool(len(params) > 0)

        if new_split:
            if separate_parameterized_gates and len(sub.data) > 0:
                corresponding_parameters.append([])
                circuits.append(sub)
                sub = QuantumCircuit(*circuit.qregs, *circuit.cregs)

            sub.data += [op]
            circuits.append(sub)
            corresponding_parameters.append(params)
            sub = QuantumCircuit(*circuit.qregs, *circuit.cregs)
        else:
            sub.data += [op]

    if len(sub.data) > 0:  # handle leftover gates
        if separate_parameterized_gates or len(circuits) == 0:
            corresponding_parameters.append(params)
            circuits.append(sub)  # create new if parameterized gates should be separated
        else:
            circuits[-1].compose(sub, inplace=True)  # or attach to last

    if return_parameters:
        return circuits, corresponding_parameters
    return circuits
