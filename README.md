1 · Overview
This repository contains a compact C++-17 solver for the Electric-Vehicle Routing Problem (EVRP) based on the benchmark set from the 2020 EVRP competition
https://mavrovouniotis.github.io/EVRPcompetition2020/.

Core Ideas

-Simulated Annealing (SA) provides global exploration with probabilistic acceptance of uphill moves.

-2-Opt is the main local-search operator that refines every tour.

-Optional neighbourhoods (swap, relocate, insert/remove charging station)
can be toggled via pre-processor flags in heuristic.cpp.

-A 20-run driver prints best cost, mean and standard deviation, and writes results to simple text files for later plotting.

-Builds with one make / mingw32-make command

2 · Build

-Windows (MinGW-w64)
mingw32-make # produces main.exe

-Linux / macOS (g++ ≥ 10)
make # produces ./main

3. Running a benchmark
   ./main ../evrp-benchmark-set/E-n22-k4.evrp

Example console output (20 runs, seeds 1 - 20)
Run: 3 with random seed 3
start 802.725
End of run 3 with best solution quality 278.437 total evaluations: 750000

Files produced per instance:
| File name | Description |
| ---------------------- | ---------------------------------------------------- |
| `stats.E-n22-k4.txt` | one cost per run + mean / stdev |
| `best_tour_run<N>.txt` | space-separated node IDs of the best tour in run _N_ |

5 · Visualising a Tour

# any Python 3.x environment

python -m pip install numpy matplotlib # run once

python plot_route.py \
 ../evrp-benchmark-set/E-n22-k4.evrp \
 best_tour_run1.txt

Legend - green ▲: customers · blue ■: charging/depot · orange: driven edges.

6 · Tuning Knobs

| Constant | Role                               | Default |
| -------- | ---------------------------------- | :-----: |
| `T0`     | starting temperature               | 1000.0  |
| `Tmin`   | stop when `T` falls below `Tmin`   |  0.05   |
| `alpha`  | cooling factor `T ← alpha · T`     |  0.95   |
| `L`      | SA iterations per temperature step |   50    |
