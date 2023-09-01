# README

**Run the experiment inside this subfolder.**

Install [Julia](https://julialang.org) and make sure that the `julia` command can be ran from your command line.

Install packages used in the scripts, they are all quite standard, and can be installed with the `Pkg.add` command.
To do that, type `julia` in your command line (you should be inside julia now with your terminal starting with `julia>`) and then run:

```julia
using Pkg
Pkg.add(["MAT", "Plots", "GLM", "DataFrames", "Distributions", "StatsBase", "HypothesisTests", "StatsPlots", "Statistics", "NPZ", "JLD2"])
```

Exit the julia terminal with `Ctrl+D`, then you are good to run in your terminal:

```bash
julia plots.jl
```

to produce:

![](figures/subjective_value_barchart_resized.png)

![](figures/neuron_time_359.png)

![](figures/neuron_histogram_359_1.png)

![](figures/neuron_histogram_359_2.png)

![](figures/neuron_histogram_359_3.png)

![](figures/neuron_histogram_359_4.png)

![](figures/neuron_histogram_359_5.png)

![](figures/neuron_time_368.png)

![](figures/neuron_histogram_368_1.png)

![](figures/neuron_histogram_368_2.png)

![](figures/neuron_histogram_368_3.png)

![](figures/neuron_histogram_368_4.png)

![](figures/neuron_histogram_368_5.png)

![](figures/neuron_time_360.png)

![](figures/neuron_histogram_360_1.png)

![](figures/neuron_histogram_360_2.png)

![](figures/neuron_histogram_360_3.png)

![](figures/neuron_histogram_360_4.png)

![](figures/neuron_histogram_360_5.png)
