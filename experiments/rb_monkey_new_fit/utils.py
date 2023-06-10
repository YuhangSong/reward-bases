import pandas as pd

import analysis_utils as au

import sys
sys.path.insert(0, "/Users/yuhang/working_dir/reward-base/")

from monkey.get_clean_data import get_clean_data

def train(config):

    spiketimes_list, stim_onsets_list, situations_list = get_clean_data(path="/Users/yuhang/working_dir/reward-base/monkey/CleanData/w065-0359.jld2")

    data = {
        'trial_i': [],
        'onset': [],
        'situation': [],
        'identity': [],
        'value': [],
        'dopamine': [],
    }
    for trial_i in range(len(spiketimes_list)):
        
        data['trial_i'].append(trial_i)
        
        onset = stim_onsets_list[trial_i]
        data['onset'].append(onset)
        
        # 1 = 1.5g banana, 2 = 0.3g banana, 3 = 0.2ml juice, 4 = 0.5ml juice, 5 = 0.9ml juice, 25 = a trial without a stimulus (empty)
        situation = {
            1: "1.5g banana",
            2: "0.3g banana",
            3: "0.2ml juice",
            4: "0.5ml juice",
            5: "0.9ml juice",
            25: "empty",
        }[situations_list[trial_i]]
        data['situation'].append(situation)

        identity = {
            "1.5g banana": +1,
            "0.3g banana": +1,
            "0.2ml juice": -1,
            "0.5ml juice": -1,
            "0.9ml juice": -1,
            "empty": 0,
        }[situation]
        data['identity'].append(identity)
        
        value = {
            "1.5g banana": 1.5,
            "0.3g banana": 0.3,
            "0.2ml juice": 0.2,
            "0.5ml juice": 0.5,
            "0.9ml juice": 0.9,
            "empty": 0,
        }[situation]
        data['value'].append(value)

        spiketimes = spiketimes_list[trial_i]
        # count how many spikes are in interval 150 to 500
        dopamine = len([spiketime for spiketime in spiketimes if 150 <= spiketime <= 500])
        data['dopamine'].append(dopamine)
    
    df = pd.DataFrame.from_dict(data)

    # Set-up Python libraries - you need to run this but you don't need to change it
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    import pandas
    import seaborn as sns
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    # Fit a regression model: Y = dopamine, x1 = food, x2 = drink.

    # first we run this line to tell statsmodels where to find the data and the explanatory variables
    reg_formula = sm.regression.linear_model.OLS.from_formula(data = df, formula = 'dopamine ~ value + identity')

    # then we run this line to fit the regression (work out the values of intercept and slope)
    # the output is a structure which we will call reg_results
    reg_results = reg_formula.fit()

    # let's view a summary of the regression results
    reg_results.summary()

    results = {}

    return results