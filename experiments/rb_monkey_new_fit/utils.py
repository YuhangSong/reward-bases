import pandas as pd

import analysis_utils as au

import sys
sys.path.insert(0, "/Users/yuhang/working_dir/reward-base/")

from monkey.get_clean_data import get_clean_data

def train(config):

    spiketimes_list, stim_onsets_list, situations_list = get_clean_data(path="/Users/yuhang/working_dir/reward-base/monkey/CleanData/w065-0359.jld2")

    print()
    print(f"len(spiketimes_list) = {len(spiketimes_list)}")
    print(f"len(stim_onsets_list) = {len(stim_onsets_list)}")
    print(f"len(situations_list) = {len(situations_list)}")

    print()
    print(f"spiketimes_list[0] = {spiketimes_list[0]}")
    print(f"len(spiketimes_list[0]) = {len(spiketimes_list[0])}")
    print(f"stim_onsets_list[0] = {stim_onsets_list[0]}")
    print(f"situations_list[0] = {situations_list[0]}")

    print()
    print(f"spiketimes_list[1] = {spiketimes_list[1]}")
    print(f"len(spiketimes_list[1]) = {len(spiketimes_list[1])}")
    print(f"stim_onsets_list[1] = {stim_onsets_list[1]}")
    print(f"situations_list[1] = {situations_list[1]}")

    results = {}

    return results