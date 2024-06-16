# plotting functions used in the neural data analysis for the reward basis paper
using MAT
using Plots
using GLM
using DataFrames
using Distributions
using StatsBase
using HypothesisTests
using StatsPlots
using Statistics
using NPZ
using Plots.PlotMeasures
include("utils.jl")

ALL_SITUATIONS = [1, 2, 3, 4, 5] #, 25]
neuronlist = [359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377]

default(size=(750, 450),
        xtickfontsize=18,
        ytickfontsize=18,
        legendfontsize=16,
        guidefontsize=18,
        titlefontsize=17,
        framestyle=:box,     # Ensure there are axis lines
        gridlinewidth=2,   # Thickness of the grid lines
        linewidth=4,# Thickness of the plot lines
        margin=20mm,         # Add margin around the plot
        titlelocation=:top   # Move the title higher
)         

function plot_neuron(neuron_id, window_size=200)
    bucket_start = -500
    bucket_end = 1500
    half_window_size = window_size / 2
    print("STARTING")
    xs = collect(bucket_start:50:bucket_end-100)
    neuron_means, neuron_stds = count_plot_by_neuron(neuron_id, bucket_start, bucket_end, window_size, true, true)

    window_size_seconds = window_size / 1000
    neuron_means = neuron_means ./ window_size_seconds


    plot(xs .+ half_window_size, neuron_means[5][1, :], label="0.9ml juice", color=:blue)
    plot!(xs .+ half_window_size, neuron_means[4][1, :], label="0.5ml juice", color=:blue, linestyle=:dashdot)
    plot!(xs .+ half_window_size, neuron_means[3][1, :], label="0.2ml juice", color=:blue, linestyle=:dot)
    plot!(xs .+ half_window_size, neuron_means[1][1, :], label="1.5g banana", color=:orange, linestyle=:solid)
    plot!(xs .+ half_window_size, neuron_means[2][1, :], label="0.3g banana", color=:orange, linestyle=:dashdot)

    vline!([0], linestyle=:dash, alpha=0.6, color=:gray, label="")

    xlabel!("Time (ms) after cue")
    ylabel!("Firing rate within window (Hz)")
    title!("Neuron $(neuron_id)")
    savefig("figures/neuron_time_$(neuron_id).png")
    savefig("figures/neuron_time_$(neuron_id).pdf")
end


function subjective_value_barchart(USE_TITLE=false)
    subjective_values = [0.05, 0.1, 0.5, 0.7, 1]
    situation_descriptions = ["0.3g Banana", "0.2ml Juice", "0.5ml Juice", "1.5g Banana", "0.9ml Juice"]
    bar(situation_descriptions, subjective_values, legend=:none, color=[:orange, :blue, :blue, :orange, :blue], alpha=0.5)

    xlabel!("Experimental condition")
    ylabel!("Subjective value")
    plot!(size=(750, 400))
    if USE_TITLE
        title!("Subjective value by condition")
    end
    savefig("figures/subjective_value_barchart_resized.png")
    savefig("figures/subjective_value_barchart_resized.pdf")
end





subjective_value_barchart()
plot_neuron(359)
# plot_neuron(368)
# plot_neuron(360)

# # [359, 372, 361, 362, 376, 360, 375, 363, 374, 367, 369, 377, 364, 366, 373, 365, 371, 370, 368]

# plot_neuron(372)
# plot_neuron(361)
# plot_neuron(362)
# plot_neuron(376)
# plot_neuron(375)
# plot_neuron(363)
# plot_neuron(374)
# plot_neuron(367)
# plot_neuron(369)
# plot_neuron(377)
# plot_neuron(364)
# plot_neuron(366)
# plot_neuron(373)
# plot_neuron(365)
# plot_neuron(371)
# plot_neuron(370)
