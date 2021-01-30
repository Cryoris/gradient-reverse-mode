from benchmarks.featuremap import Classification

circuit = Classification(4, reps=1)
print(circuit.draw(output='latex_source'))
