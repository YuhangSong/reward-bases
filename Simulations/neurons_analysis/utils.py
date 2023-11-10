import analysis_utils as au
import utils as u
import random
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy import stats

import os

RB_CODE_DIR = os.environ["RB_CODE_DIR"]

# for import neurons_visualisation, need improvement
import sys

sys.path.insert(0, os.path.join(RB_CODE_DIR, "Data"))

neurons = [
    "0359",
    "0360",
    "0361",
    "0362",
    "0363",
    "0364",
    "0365",
    "0366",
    "0367",
    "0368",
    "0369",
    "0370",
    "0371",
    "0372",
    "0373",
    "0374",
    "0375",
    "0376",
    "0377",
]


import os


def get_df(
    neuron,
    response_window_start=150,
    response_window_end=500,
    baseline_window_start=-500,
    baseline_window_end=0,
    is_shuffle_identity=False,
    num_trial_blocks=1,
    trial_block_idxes=[0],
):
    from get_clean_data import get_clean_data

    spiketimes_list, stim_onsets_list, situations_list = get_clean_data(
        path=f"w065-{neuron}.jld2",
    )

    # date each neuron is recorded
    # these are obtained from ./file list and date.PNG
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
    }[neuron]

    # data used to construct the dataframe, where each row is a trial
    data = {
        "date": [],
        "trial_i": [],
        "onset": [],
        "situation": [],
        "identity": [],
        "value": [],
        "subjective_value_banana": [],
        "subjective_value_juice": [],
        "dopamine": [],
        "relative_firing_rate": [],
    }

    trial_is = list(range(len(spiketimes_list)))
    trial_blocks = u.split_list_specific(trial_is, n=num_trial_blocks)

    trial_is = []
    if isinstance(trial_block_idxes, int):
        trial_block_idxes = [trial_block_idxes]
    for idx in trial_block_idxes:
        trial_is.extend(trial_blocks[idx])

    for trial_i in trial_is:
        # situation is the stimulus presented in this trial
        situation = {
            1: "1.5g banana",
            2: "0.3g banana",
            3: "0.2ml juice",
            4: "0.5ml juice",
            5: "0.9ml juice",
            25: "empty",
        }[situations_list[trial_i]]
        # onset is the time when the stimulus is presented
        onset = stim_onsets_list[trial_i]
        # spiketimes is the list of spike times in this trial
        spiketimes = spiketimes_list[trial_i]
        if situation == "empty" or onset == [] or (not isinstance(spiketimes, list)):
            # in these cases, we don't have enough data to construct a row of a trial
            continue

        # add date, trial_i, situation, onset to the row of this trial
        data["date"].append(date)
        data["trial_i"].append(trial_i)
        data["situation"].append(situation)
        data["onset"].append(onset)

        # number of spikes in the response window (in contrast to in baseline window)
        num_spikes_in_response_window = len(
            [
                spiketime
                for spiketime in spiketimes
                if (
                    (onset + response_window_start)
                    <= spiketime
                    <= (onset + response_window_end)
                )
            ]
        )
        # size of the response window
        response_window_size = (response_window_end - response_window_start) / 1000.0
        # firing rate in the response window
        firing_rate_in_response_window = (
            num_spikes_in_response_window / response_window_size
        )

        # dopamine is the number of spikes in the response window
        dopamine = num_spikes_in_response_window
        data["dopamine"].append(dopamine)

        # number of spikes in the baseline window (in contrast to in response window)
        num_spikes_in_baseline_window = len(
            [
                spiketime
                for spiketime in spiketimes
                if (
                    (onset + baseline_window_start)
                    <= spiketime
                    <= (onset + baseline_window_end)
                )
            ]
        )
        # size of the baseline window
        baseline_window_size = (baseline_window_end - baseline_window_start) / 1000.0
        # firing rate in the baseline window
        firing_rate_in_baseline_window = (
            num_spikes_in_baseline_window / baseline_window_size
        )

        # relative_firing_rate is the difference between firing rate in the response window and in the baseline window
        relative_firing_rate = (
            firing_rate_in_response_window - firing_rate_in_baseline_window
        )
        data["relative_firing_rate"].append(relative_firing_rate)

        # identity is the identity of the stimulus presented in this trial
        # it is 1 if the stimulus is a banana, -1 if the stimulus is a juice, and 0 if the stimulus is empty
        identity = {
            "1.5g banana": -1,
            "0.3g banana": -1,
            "0.2ml juice": +1,
            "0.5ml juice": +1,
            "0.9ml juice": +1,
            "empty": 0,
        }[situation]
        data["identity"].append(identity)

        # value is the value of the stimulus presented in this trial, normalized across different stimuli
        value = {
            "1.5g banana": 0.7,
            "0.3g banana": 0.05,
            "0.2ml juice": 0.1,
            "0.5ml juice": 0.5,
            "0.9ml juice": 1,
            "empty": 0,
        }[situation]
        data["value"].append(value)

        # subjective_value is the similar as value
        data["subjective_value_banana"].append(
            value if situation.endswith("banana") else 0
        )
        data["subjective_value_juice"].append(
            value if situation.endswith("juice") else 0
        )

    if is_shuffle_identity:
        random.shuffle(data["identity"])

    df = pd.DataFrame.from_dict(data)

    return df


def do_regression(df, formula, fit_to="dopamine"):
    # first we run this line to tell statsmodels where to find the data and the explanatory variables
    reg_formula = sm.regression.linear_model.OLS.from_formula(
        data=df, formula=f"{fit_to} ~ {formula}"
    )

    # then we run this line to fit the regression (work out the values of intercept and slope)
    # the output is a structure which we will call reg_results
    reg_results = reg_formula.fit()

    # let's view a summary of the regression results
    # reg_results.summary()

    return reg_results


def train(config):
    df = get_df(
        neuron=config["neuron"],
        **config.get("get_df_kwargs", {}),
    )

    reg_results = do_regression(
        df=df,
        formula=config["formula"],
    )

    results = {}

    # get r2 score
    results["rsquared"] = reg_results.rsquared

    # get bic score
    results["bic"] = reg_results.bic

    # get aic score
    results["aic"] = reg_results.aic

    # get coeff
    if "coeff_id" in config:
        # get coeff
        results["coeff"] = reg_results.params[config["coeff_id"]]

        # get pvalue
        results["pvalue"] = reg_results.pvalues[config["coeff_id"]]

    return results


def train_formula_block(config):
    results = {}

    for trial_block_idx in [0, 1]:
        df = get_df(
            neuron=config["neuron"],
            num_trial_blocks=2,
            trial_block_idxes=trial_block_idx,
        )

        reg_results = do_regression(
            df=df,
            formula="value + identity : value",
        )

        results[f"coeff: trial_block_idx={trial_block_idx}"] = reg_results.params[
            "identity:value"
        ]

    return results


def model_recovery(config):
    results = {}

    # set numpy seed
    np.random.seed(config["seed"])

    df = get_df(
        neuron=config["neuron"],
    )

    reg_results = do_regression(
        df=df,
        fit_to="dopamine",
        formula=config["generate_with_formula"],
    )

    if config["generate_with_formula"] == "value + situation":
        # Extract the parameters from the regression results
        params = reg_results.params

        # Identify the keys for the parameters you want to shuffle
        keys_to_shuffle = [
            "situation[T.0.3g banana]",
            "situation[T.0.5ml juice]",
            "situation[T.0.9ml juice]",
            "situation[T.1.5g banana]",
        ]

        # Extract the values of these parameters
        values_to_shuffle = [params[key] for key in keys_to_shuffle]

        # Shuffle the values
        np.random.shuffle(values_to_shuffle)

        # Assign the shuffled values back to the params
        for i, key in enumerate(keys_to_shuffle):
            params[key] = values_to_shuffle[i]

        # Now, params contains the shuffled values for the specified parameters

    # Calculate the residuals from the fitted model
    residuals = reg_results.resid

    # Get the standard deviation of the residuals
    std_dev = residuals.std()

    # When making predictions, add a random term sampled from a normal distribution
    # with mean zero and the standard deviation of the residuals
    random_error = np.random.normal(0, std_dev, size=df.shape[0])

    # Adding the random error to the deterministic predictions to introduce stochasticity
    df["generated_dopamine"] = reg_results.predict(df) + random_error

    reg_results = do_regression(
        df=df,
        fit_to="generated_dopamine",
        formula=config["fit_generated_data_with_formula"],
    )

    # get bic score
    results["bic"] = reg_results.bic

    # get aic score
    results["aic"] = reg_results.aic

    return results


def get_num_significant_coeffs(config):
    num_significant_coeffs = 0

    for neuron in neurons:
        df = get_df(
            neuron=neuron,
            is_shuffle_identity=config["is_shuffle_identity"],
        )

        reg_results = do_regression(
            df=df,
            formula="value + identity : value",
        )

        p_value = reg_results.pvalues["identity:value"]

        if p_value < 0.05:
            num_significant_coeffs += 1

    return {
        "num_significant_coeffs": num_significant_coeffs,
    }


def get_two_regressor_coeffs(neuron):
    df = get_df(
        neuron=neuron,
    )

    # fit to formula
    reg_formula = sm.regression.linear_model.OLS.from_formula(
        data=df, formula=f"dopamine ~ subjective_value_banana + subjective_value_juice"
    )
    reg_results = reg_formula.fit()

    # get coeff
    coeff_banana = reg_results.params[f"subjective_value_banana"]
    coeff_juice = reg_results.params[f"subjective_value_juice"]

    return df, coeff_banana, coeff_juice


def train_two_regressor(config):
    results = {}

    df, coeff_banana, coeff_juice = get_two_regressor_coeffs(
        neuron=config["neuron"],
    )

    results[f"coeff_banana"] = coeff_banana
    results[f"coeff_juice"] = coeff_juice

    # compare coeff of two formula to confirm the code is consistent with theorectical derivation
    if "compare_coeff" in config:
        # compare against another model

        # fit to formula
        reg_formula = sm.regression.linear_model.OLS.from_formula(
            data=df, formula=f"dopamine ~ value + identity : value"
        )
        reg_results = reg_formula.fit()
        # get coeff
        coeff_value_fitted = reg_results.params[f"value"]
        coeff_identity_value_fitted = reg_results.params[f"identity:value"]

        coeff_value_inferred = (coeff_banana + coeff_juice) / 2
        coeff_identity_value_inferred = (coeff_juice - coeff_banana) / 2

        if config["compare_coeff"] == "inferred":
            results[f"coeff_value"] = coeff_value_inferred
            results[f"coeff_identity_value"] = coeff_identity_value_inferred
        elif config["compare_coeff"] == "fitted":
            results[f"coeff_value"] = coeff_value_fitted
            results[f"coeff_identity_value"] = coeff_identity_value_fitted
        else:
            raise NotImplementedError

    return results


def get_neuron_responses(neuron, is_shuffle_situation=False):
    df = get_df(
        neuron=neuron,
    )

    # get df with only rows is situation='1.5g banana' or '0.9ml juice'
    df = df[df["situation"].isin(["1.5g banana", "0.9ml juice"])]

    if is_shuffle_situation:
        df["situation"] = df["situation"].sample(frac=1).values

    neuron_responses = {
        "banana": {
            "mean": None,
            "sem_half": None,
        },
        "juice": {
            "mean": None,
            "sem_half": None,
        },
    }

    # relative firing rate of the biggest banana stimulus
    biggest_banana_relative_firing_rate = df[df["situation"] == "1.5g banana"][
        "relative_firing_rate"
    ]
    # mean of it
    neuron_responses["banana"]["mean"] = biggest_banana_relative_firing_rate.mean()
    # half of the standard error of it
    neuron_responses["banana"]["sem_half"] = (
        biggest_banana_relative_firing_rate.sem() / 2
    )

    # relative firing rate of the biggest juice stimulus
    biggest_juice_relative_firing_rate = df[df["situation"] == "0.9ml juice"][
        "relative_firing_rate"
    ]
    # mean of it
    neuron_responses["juice"]["mean"] = biggest_juice_relative_firing_rate.mean()
    # half of the standard error of it
    neuron_responses["juice"]["sem_half"] = biggest_juice_relative_firing_rate.sem() / 2

    return neuron_responses


def get_neuron_responses_correlation(config):
    seed = config["seed"]
    # in numpy (pandas uses numpy for random number generation)
    np.random.seed(seed)
    # in random
    random.seed(seed)

    banana = []
    juice = []

    for neuron in neurons:
        neuron_responses = get_neuron_responses(
            neuron=neuron,
            is_shuffle_situation=config["is_shuffle_situation"],
        )
        banana.append(neuron_responses["banana"]["mean"])
        juice.append(neuron_responses["juice"]["mean"])

    corr, _ = stats.pearsonr(banana, juice)

    return {
        "corr": corr,
    }


def train_neuron_response(config):
    neuron_responses = get_neuron_responses(
        neuron=config["neuron"],
    )

    results = {}

    # put into results
    results["biggest_banana_relative_firing_rate_mean"] = neuron_responses["banana"][
        "mean"
    ]
    results["biggest_juice_relative_firing_rate_mean"] = neuron_responses["juice"][
        "mean"
    ]

    results["biggest_banana_relative_firing_rate_sem_half"] = neuron_responses[
        "banana"
    ]["sem_half"]
    results["biggest_juice_relative_firing_rate_sem_half"] = neuron_responses["juice"][
        "sem_half"
    ]

    return results


def proc_df(df, log_id):
    # rename the col for easier access

    if isinstance(log_id, str):
        log_id = [log_id]
    assert isinstance(log_id, list)

    for v in log_id:
        df = df.rename(columns={f"df['{v}'].iloc[-1]": v})

    return df


def sort_by_id_coeff(
    df, filter_df_fn=lambda df: df[df["coeff_id"] == "identity:value"]
):
    # filter the DataFrame
    df_filtered = filter_df_fn(df)

    # sort the filtered DataFrame and get the sorted 'neuron' values
    sorted_neurons = df_filtered.sort_values("coeff", ascending=False)["neuron"]

    # convert sorted_neurons to a categorical type with its categories being the sorted neurons
    df["neuron"] = pd.Categorical(
        df["neuron"],
        categories=sorted_neurons,
        ordered=True,
    )

    return df


def plot_neuron_response(df):
    # convert a pd.Series to a np.array
    def convert(x):
        return np.squeeze(x.to_numpy())

    # plot error bars in both x and y directions
    plt.errorbar(
        convert(df[["biggest_juice_relative_firing_rate_mean"]]),
        convert(df[["biggest_banana_relative_firing_rate_mean"]]),
        convert(df[["biggest_juice_relative_firing_rate_sem_half"]]),
        convert(df[["biggest_banana_relative_firing_rate_sem_half"]]),
        "none",
        ecolor="gray",
        elinewidth=1.5,
        capsize=1.7,
        capthick=1.5,
        zorder=1,
    )

    # scatter plot
    ax = sns.scatterplot(
        data=df,
        x="biggest_juice_relative_firing_rate_mean",
        y="biggest_banana_relative_firing_rate_mean",
    )

    ax.set_aspect("equal", adjustable="box")

    ax.axhline(0, color="red", linestyle="--")  # Adds a horizontal line at y=0
    ax.axvline(0, color="red", linestyle="--")  # Adds a vertical line at x=0

    # Get the range of the data
    xmin, xmax = np.floor(df["biggest_juice_relative_firing_rate_mean"].min()), np.ceil(
        df["biggest_juice_relative_firing_rate_mean"].max()
    )
    ymin, ymax = np.floor(
        df["biggest_banana_relative_firing_rate_mean"].min()
    ), np.ceil(df["biggest_banana_relative_firing_rate_mean"].max())

    # Combine the ranges
    combined_min = min(xmin, ymin)
    combined_max = max(xmax, ymax)

    # Make a list of ticks
    # plus 1 to include the maximum
    ticks = np.arange(combined_min, combined_max + 1)

    # Set the ticks
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)


def train_data_model(config):
    seed = config["seed"]
    # in numpy (pandas uses numpy for random number generation)
    np.random.seed(seed)
    # in random
    random.seed(seed)

    # r

    r = {
        "banana": [0.7, 0.05, 0.0, 0.0, 0.0],
        "juice": [0.0, 0.0, 0.1, 0.5, 1.0],
    }

    def get_r(i, x):
        return r[i][x]

    # V

    V = {
        "banana": [0.0, 0.0, 0.0, 0.0, 0.0],
        "juice": [0.0, 0.0, 0.0, 0.0, 0.0],
    }

    def get_V(i, x):
        return V[i][x]

    def set_V(i, x, v):
        V[i][x] = v

    # beta

    coeffs = {}
    for neuron in neurons:
        _, coeff_banana, coeff_juice = get_two_regressor_coeffs(
            neuron=neuron,
        )
        coeffs[neuron] = {
            "banana": coeff_banana,
            "juice": coeff_juice,
        }

    def get_beta(i, k):
        return coeffs[k][i]

    # delta

    delta = {}
    for neuron in neurons:
        delta[neuron] = [0.0, 0.0, 0.0, 0.0, 0.0]

    def get_delta(k, x):
        return delta[k][x]

    def set_delta(k, x, v):
        delta[k][x] = v

    # total set of indexes
    ks = neurons
    xs = [0, 1, 2, 3, 4]
    is_ = ["banana", "juice"]

    epoch_history = []
    V_history = []

    for epoch in range(config["epochs"]):
        x = int(np.random.uniform(low=0, high=5))

        for k in ks:
            vs = []
            for i in is_:
                vs.append(get_beta(i=i, k=k) * (get_r(i=i, x=x) - get_V(i=i, x=x)))
            v = sum(vs)

            set_delta(
                k=k,
                x=x,
                v=v,
            )

        for i in is_:
            delta_Vs = []
            for k in ks:
                delta_Vs.append(get_beta(i=i, k=k) * get_delta(k=k, x=x))
            delta_V = sum(delta_Vs) * config["alpha"]
            set_V(
                i=i,
                x=x,
                v=get_V(i=i, x=x) + delta_V,
            )

        epoch_history.append(epoch)
        V_history.append(get_V(i=config["V_history_i"], x=config["V_history_x"]))

    return {
        "epoch_history": epoch_history,
        "V_history": V_history,
    }


def eval_extract_lists(df, cols):
    def eval_col(df, col_eval, col_new):
        def eval_with_nan_inf(v):
            nan = float("nan")
            inf = float("inf")
            return eval(v)

        df = au.new_col(df, col_new, lambda row: eval_with_nan_inf(row[col_eval]))
        df = df.drop(col_eval, axis=1)
        return df

    for col in cols:
        df = eval_col(df, f"df['{col}'].iloc[0]", col)

    df = au.extract_lists(df, cols)

    return df


def plot_data_model(df):
    df = eval_extract_lists(df, ["epoch_history", "V_history"])

    sns.relplot(
        data=df,
        kind="line",
        x="epoch_history",
        y="V_history",
        hue="V_history_x",
        col="V_history_i",
    )


def plot_confusion_matrix(df):
    seeds = df["seed"].unique()
    generate_with_formulas = sorted(list(df["generate_with_formula"].unique()))
    fit_generated_data_with_formulas = sorted(
        list(df["fit_generated_data_with_formula"].unique())
    )
    assert len(generate_with_formulas) == len(fit_generated_data_with_formulas)
    assert generate_with_formulas[0] == fit_generated_data_with_formulas[0]
    assert generate_with_formulas[1] == fit_generated_data_with_formulas[1]

    confusion_matrix = pd.DataFrame(
        {
            "generate_with_formula": [
                "value + identity : value",
                "value + identity : value",
                "value + situation",
                "value + situation",
            ],
            "fit_generated_data_with_formula": [
                "value + identity : value",
                "value + situation",
                "value + identity : value",
                "value + situation",
            ],
            "count": [0, 0, 0, 0],
        }
    )

    for generate_with_formula in generate_with_formulas:
        for seed in seeds:
            two_rows = df[
                (df["seed"] == seed)
                & (df["generate_with_formula"] == generate_with_formula)
            ]
            if (
                two_rows[
                    df["fit_generated_data_with_formula"] == "value + identity : value"
                ]["sum_aic"].item()
                < two_rows[
                    df["fit_generated_data_with_formula"] == "value + situation"
                ]["sum_aic"].item()
            ):
                confusion_matrix.loc[
                    (confusion_matrix["generate_with_formula"] == generate_with_formula)
                    & (
                        confusion_matrix["fit_generated_data_with_formula"]
                        == "value + identity : value"
                    ),
                    "count",
                ] += 1
            else:
                confusion_matrix.loc[
                    (confusion_matrix["generate_with_formula"] == generate_with_formula)
                    & (
                        confusion_matrix["fit_generated_data_with_formula"]
                        == "value + situation"
                    ),
                    "count",
                ] += 1

    confusion_matrix["percentage"] = confusion_matrix["count"] / len(seeds) * 100

    confusion_matrix.rename(
        columns={"generate_with_formula": "Data generating formula"}, inplace=True
    )

    ax = sns.heatmap(
        confusion_matrix.pivot(
            index="fit_generated_data_with_formula",
            columns="Data generating formula",
            values="percentage",
        ),
        annot=True,
        fmt=".1f",
        cmap="Blues",
    )

    for t in ax.texts:
        t.set_text(t.get_text() + " %")

    # Remove the y-axis label
    ax.set_ylabel("")
