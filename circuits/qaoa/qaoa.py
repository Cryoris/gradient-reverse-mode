from qiskit.opflow import I, Z
from benchmarks.maxcut import QAOAAnsatz 

operator = (I ^ I ^ Z ^ Z) + (I ^ Z ^ I ^ Z) + (Z ^ I ^ I ^ Z) + (I ^ Z ^ Z ^ I)
circuit = QAOAAnsatz(operator)

print(circuit.draw(output='latex_source', initial_state=True))
