"""Script to run the benchmarks."""

# pylint: disable=invalid-name

import sys
from benchmarks.benchmark import Benchmark
from benchmarks.efficient_su2 import run_efficientsu2
from benchmarks.featuremap import run_featuremap
from benchmarks.maxcut import run_maxcut

if len(sys.argv) < 2:
    print('Please specify a benchmark to run. Available: efficientsu2 featuremap maxcut')
    sys.exit()

task = sys.argv[1]

if task == 'plot':
    if len(sys.argv) < 3:
        print('To plot please specify the filename as second argument.')
        sys.exit()
    fname = sys.argv[2]

    if len(sys.argv) > 3:
        saveas = sys.argv[3]
    else:
        saveas = None

if task == 'efficientsu2':
    run_efficientsu2()
elif task == 'featuremap':
    run_featuremap()
elif task == 'maxcut':
    run_maxcut()
elif task == 'plot':
    benchmark = Benchmark([0], None, None)
    benchmark.plot(fname, saveas=saveas, show=True)
else:
    raise ValueError(f'Invalid argument: {task}')
