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

ALL_SITUATIONS = [1, 2, 3, 4, 5, 25]
neuronlist = [359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377]

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
    title!("$(neuron_id) responsive neuron")
    savefig("figures/neuron_time_$(neuron_id).png")
    savefig("figures/neuron_time_$(neuron_id).pdf")

    window_start = 150
    window_end = 500

    window_size = window_end - window_start
    divisor = 1 # This was Beren's original line # divisor = window_size / 1000 # msec to sec

    spiketimes, situations, stim_onsets = load_data(joinpath(ENV["RB_CODE_DIR"], "Data/CleanData/w065-0$(neuron_id)" * ".jld2"))
    situation_1 = counts_per_trial_window_by_situation(spiketimes, stim_onsets, situations, [1], window_start, window_end, false) / divisor
    situation_2 = counts_per_trial_window_by_situation(spiketimes, stim_onsets, situations, [2], window_start, window_end, false) / divisor
    situation_3 = counts_per_trial_window_by_situation(spiketimes, stim_onsets, situations, [3], window_start, window_end, false) / divisor
    situation_4 = counts_per_trial_window_by_situation(spiketimes, stim_onsets, situations, [4], window_start, window_end, false) / divisor
    situation_5 = counts_per_trial_window_by_situation(spiketimes, stim_onsets, situations, [5], window_start, window_end, false) / divisor
    # this assumes a window size of 300 and is fragile to that
    xs = collect(0:1:10) # This was Beren's original line # xs = collect(0:3:30)

    bar_width = (1 / divisor) - 0.3
    xticks = (xs[1:end-1] .+ (0.5 / divisor), string.(xs))
    histogram_list = [situation_1, situation_2, situation_3, situation_4, situation_5]
    label_list = ["1.5g Banana" "0.3g Banana" "0.2ml Juice" "0.5ml Juice" "0.9ml Juice"]
    color_list = [:orange :orange :blue :blue :blue]

    N = length(histogram_list)
    for i in 1:N
        hist_plot = histogram(xs, histogram_list[i], label=label_list[i], color=color_list[i], ylims=(0, 6), bins=xs, xticks=xticks, bar_width=bar_width, size=(500, 125), yticks=[2, 4, 6])
        savefig("figures/neuron_histogram_$(neuron_id)_$i" * ".png")
        savefig("figures/neuron_histogram_$(neuron_id)_$i" * ".pdf")
    end
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
plot_neuron(368)
plot_neuron(360)
plot_neuron(366)
plot_neuron(375)
plot_neuron(374)
plot_neuron(369)
plot_neuron(364)
plot_neuron(373)
plot_neuron(365)
