# utility functions for plotting for the data analysis
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
using JLD2

normalize(vec) = (vec .- mean(vec) ./ std(vec))

function bucket_count_spikes(spikes, window, bucket_start=-2000, bucket_end=3500, use_overlapping_window=false)
    println("In bucket count spikes $use_overlapping_window")
    if use_overlapping_window == true

        return overlapping_window_count_spikes(spikes, window)
    end
    bucket_edges = collect(bucket_start:window:bucket_end)
    counts = zeros(length(bucket_edges))
    for i in 1:length(spikes)
        stimes = spikes[i]
        for time in stimes
            if time <= bucket_edges[1]
                counts[1] += 1
            end
            for b in 1:length(bucket_edges)-2
                if time >= bucket_edges[b] && time <= bucket_edges[b+1]
                    counts[b+1] += 1
                end
            end
            if time >= bucket_edges[length(bucket_edges)]
                counts[length(bucket_edges)] += 1
            end
        end
    end
    return counts
end

function count_spikes_within_window(spikes, tstart, tend)
    count = 0
    for stimes in spikes    # loop over all spiketrains
        for t in stimes
            if t >= tstart && t <= tend
                count += 1
            end
        end
    end
    return count
end

function overlapping_window_count_spikes(spikes, window_size, bucket_start=-500, bucket_end=1500, step_size=50, mean_normalize_across_trials=true)
    buckets = []
    bucket_edge = bucket_start
    println("in overlapping window count spikes $bucket_start")
    bucket_starts = collect(bucket_start:step_size:bucket_end)
    counts = zeros(length(bucket_starts))
    for i in 1:length(spikes)
        stimes = spikes[i]
        for time in stimes
            if time <= bucket_starts[1]
                counts[1] += 1
            end
            if time >= bucket_starts[end]
                counts[end] += 1
            end
            for i in 1:length(bucket_starts)
                #if time >= bucket_starts[i] -  && time <= bucket_starts[i] + window_size
                if time >= (bucket_starts[i] - (window_size / 2)) && time <= (bucket_starts[i] + (window_size / 2))
                    counts[i] += 1
                end
            end
        end
    end
    if mean_normalize_across_trials
        L = length(spikes)
        println("mean normalizing across spikes $L")
        counts = counts ./ length(spikes)
    end
    return counts
end


function get_relative_times(spikes, event_times)
    relative_times = []
    for i in 1:length(spikes)
        if typeof(event_times[i]) == Int16
            #println("INDEX: $i")
            push!(relative_times, spikes[i] .- event_times[i])
        else
            e = event_times[i]
            #println("error: $e")
        end
    end
    return relative_times
end

function filter_spikes_events_by_situation(spikes, events, situations, situation_code_list)
    # figure out filter by situatios
    spike_list_situations = []
    event_list_situations = []
    for i in 1:length(spikes)
        if situations[i][2] in situation_code_list
            push!(spike_list_situations, spikes[i])
            push!(event_list_situations, events[i])
        end
    end
    return spike_list_situations, event_list_situations
end


function bucket_counts_by_situation(spiketimes, events, situations, situation_list, window_size, bucket_start, bucket_end, use_overlapping_window=false)
    spikes_filt, events_filt = filter_spikes_events_by_situation(spiketimes, events, situations, situation_list)
    relative_spikes = get_relative_times(spikes_filt, events)
    counts = bucket_count_spikes(relative_spikes, window_size, bucket_start, bucket_end, use_overlapping_window)
    return counts
end


function load_data(sname="CleanData/w064-0122.jld2")

    loaded_spiketimes, loaded_stim_onsets, loaded_situations = jldopen(sname, "r") do file
        read(file, "spiketimes"), read(file, "stim_onsets"), read(file, "situations")
    end

    return loaded_spiketimes, loaded_situations, loaded_stim_onsets
end


function bucket_count_by_situation_neuronlist(neuronlist, situation_list, window_size, bucket_start, bucket_end, crop_end=true, use_overlapping_window=false)
    all_situation_list = []
    for situation in ALL_SITUATIONS
        all_counts = []
        for neuron in neuronlist
            spiketimes, situations, stim_onsets = load_data("CleanData/w065-0$neuron" * ".jld2")
            try
                counts = bucket_counts_by_situation(spiketimes, stim_onsets, situations, [situation], window_size, bucket_start, bucket_end, use_overlapping_window)
                if crop_end
                    push!(all_counts, counts[2:end-1]) # crop end collapsed buckets
                else
                    push!(all_counts, counts)
                end
            catch e
                println("ERROR ON NEURON $neuron")
                println("Exception $e")
            end
        end
        push!(all_situation_list, all_counts)
    end
    return all_situation_list
end

function count_plot_by_neuron(neuronlist, bucket_start, bucket_end, window_size, normalize_by_prev_window=false, use_overlapping_window=false)
    println("In count plot by neuron $use_overlapping_window")
    all_bucket_counts = bucket_count_by_situation_neuronlist(neuronlist, ALL_SITUATIONS, window_size, bucket_start, bucket_end, true, use_overlapping_window)
    if normalize_by_prev_window == true
        print("not implemented") # I don't think this actually makes sense in this case!
    end
    mean_all_counts = []
    std_all_counts = []
    for i in 1:length(ALL_SITUATIONS)
        counts_matrix = hcat(vcat.(all_bucket_counts[i]...)...)
        mean_counts = mean(counts_matrix, dims=1)
        std_counts = std(counts_matrix, dims=1)
        push!(mean_all_counts, mean_counts)
        push!(std_all_counts, std_counts)
    end
    return mean_all_counts, std_all_counts
end

function counts_per_trial_window_by_situation(spiketimes, events, situations, situation_list, window_start, window_end, control_normalization=false, per_trial_normalization=false)
    spikes_filt, events_filt = filter_spikes_events_by_situation(spiketimes, events, situations, situation_list)
    relative_spikes = get_relative_times(spikes_filt, events_filt)
    # count spikes for each trial
    counts_per_trial = []
    total_control_counts = 0
    Tdiff = window_end - window_start
    for stimes in relative_spikes
        count = 0
        for t in stimes
            if t >= window_start && t <= window_end
                count += 1
            end
            if control_normalization == true
                if t >= window_start - 100 - Tdiff && t <= window_start - 100
                    total_control_counts += 1
                end
            end
        end
        push!(counts_per_trial, count)
    end
    mean_control_counts = total_control_counts / length(relative_spikes)
    if control_normalization == true
        # normalize
        counts_per_trial = counts_per_trial ./ mean_control_counts
    end
    return counts_per_trial
end