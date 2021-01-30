from qiskit.circuit.library import EfficientSU2

print(EfficientSU2(5, entanglement='linear', reps=1).draw(output='latex_source', initial_state=True))
