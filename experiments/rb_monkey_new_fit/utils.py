import pandas as pd

import analysis_utils as au

import sys
sys.path.insert(0, "/Users/yuhang/working_dir/reward-base/")

import numpy as np
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

from monkey.get_clean_data import get_clean_data

def train(config):

    spiketimes_list, stim_onsets_list, situations_list = get_clean_data(
        path=f"/Users/yuhang/working_dir/reward-base/monkey/CleanData/w065-{config['neuron']}.jld2"
    )

    date = {
        "0358": "11/09/2013",
        "0359": "01/10/2013",
        "0360": "01/10/2013",
        "0361": "01/10/2013",
        "0362": "02/10/2013",
        "0363": "02/10/2013",
        "0364": "03/10/2013",
        "0365": "03/10/2013",
        "0366": "03/10/2013",
        "0367": "03/10/2013",
        "0368": "04/10/2013",
        "0369": "04/10/2013",
        "0370": "04/10/2013",
        "0371": "04/10/2013",
        "0372": "08/10/2013",
        "0373": "09/10/2013",
        "0374": "09/10/2013",
        "0375": "09/10/2013",
        "0376": "11/10/2013",
        "0377": "11/10/2013",
        "0378": "20/02/2014",
    }[config['neuron']]

    data = {
        'date': [],
        'trial_i': [],
        'onset': [],
        'situation': [],
        'identity': [],
        'value': [],
        'dopamine': [],
    }
    for trial_i in range(len(spiketimes_list)):
        
        # 1 = 1.5g banana, 2 = 0.3g banana, 3 = 0.2ml juice, 4 = 0.5ml juice, 5 = 0.9ml juice, 25 = a trial without a stimulus (empty)
        situation = {
            1: "1.5g banana",
            2: "0.3g banana",
            3: "0.2ml juice",
            4: "0.5ml juice",
            5: "0.9ml juice",
            25: "empty",
        }[situations_list[trial_i]]
        onset = stim_onsets_list[trial_i]
        spiketimes = spiketimes_list[trial_i]
        if situation == "empty" or onset == [] or (not isinstance(spiketimes, list)):
            continue

        data['date'].append(date)
        
        data['trial_i'].append(trial_i)

        data['situation'].append(situation)
        
        data['onset'].append(onset)

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
            "1.5g banana": 0.7,
            "0.3g banana": 0.05,
            "0.2ml juice": 0.1,
            "0.5ml juice": 0.5,
            "0.9ml juice": 1,
            "empty": 0,
        }[situation]
        data['value'].append(value)

        # count how many spikes are in interval 150 to 500
        dopamine = len([spiketime for spiketime in spiketimes if ((onset+150) <= spiketime <= (onset+500))])
        data['dopamine'].append(dopamine)
    
    df = pd.DataFrame.from_dict(data)

    # first we run this line to tell statsmodels where to find the data and the explanatory variables
    reg_formula = sm.regression.linear_model.OLS.from_formula(data = df, formula = f'dopamine ~ {config["formula"]}')

    # then we run this line to fit the regression (work out the values of intercept and slope)
    # the output is a structure which we will call reg_results
    reg_results = reg_formula.fit()

    # let's view a summary of the regression results
    # reg_results.summary()

    results = {}

    results['rsquared'] = reg_results.rsquared

    # get bic score
    results['bic'] = reg_results.bic

    return results

def plot(df, y, x):

    df = df.rename(columns={f"df['{y}'].iloc[-1]": y})

    g=sns.catplot(hue="formula", x=x, y=y, data=df, kind="bar")

    # rotate x-axis labels for 90 degrees
    g.set_xticklabels(rotation=90)

