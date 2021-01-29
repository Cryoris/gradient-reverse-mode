"""Script to run the benchmarks."""

import sys
from benchmarks.efficient_su2 import run_efficientsu2
from benchmarks.featuremap import run_featuremap

if len(sys.argv) != 2:
    print('Please specify a benchmark to run. Available: efficientsu2 uccsd')

if sys.argv[1] == 'efficientsu2':
    run_efficientsu2()
elif sys.argv[1] == 'featuremap':
    run_featuremap()
else:
    raise ValueError(f'Invalid benchmark name: {sys.argv[1]}')
