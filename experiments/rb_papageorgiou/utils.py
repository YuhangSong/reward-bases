import random
import torch
import numpy as np
from scipy.special import softmax

import analysis_utils as au
import fit_data as fd

no_food_water = 0
food_water = 1
more_food_water = 4

# probability of transfering from s1 to p1, p2, p3
s1_food = 0.8
s1_water = 0.15
s1_more_food = 0.05

# probability of transfering from s2 to p1, p2, p3
s2_food = 0.15
s2_water = 0.8
s2_more_water = 0.05


def train_agent_papageorgiou_exp(N_steps=500, lr=0.1):
    # value of food value function, each corresponds to one lever
    V_food = np.zeros(2)
    # value of water value function, each corresponds to one lever
    V_water = np.zeros(2)

    # coefficients of the food value function
    a_food = 1
    # coefficients of the water value function
    a_water = 1

    for n in range(N_steps):

        # compute total V and normalize
        V = (a_food * V_food) + (a_water * V_water)
        V = softmax(V)

        a = np.random.choice([0, 1], p=V)

        if a == 0:  # state 1
            r_idx = int(np.random.choice([0, 1, 2], p=np.array(
                [s1_food, s1_water, s1_more_food])))
            if r_idx == 0:
                V_food[0] += lr * (food_water - V_food[0])
                V_water[0] += lr * (no_food_water - V_water[0])
            if r_idx == 1:
                V_food[0] += lr * (no_food_water - V_food[0])
                V_water[0] += lr * (food_water - V_water[0])
            if r_idx == 2:
                V_food[0] += lr * (more_food_water - V_food[0])
                V_water[0] += lr * (no_food_water - V_water[0])
        if a == 1:
            r_idx = int(np.random.choice([0, 1, 2], p=np.array(
                [s2_food, s2_water, s2_more_water])))
            if r_idx == 0:
                V_food[1] += lr * (food_water - V_food[1])
                V_water[1] += lr * (no_food_water - V_water[1])
            if r_idx == 1:
                V_food[1] += lr * (no_food_water - V_food[1])
                V_water[1] += lr * (food_water - V_water[1])
            if r_idx == 2:
                V_food[1] += lr * (no_food_water - V_food[1])
                V_water[1] += lr * (more_food_water - V_water[1])

    return V_food, V_water


valued_s = 0
devalued_s = 1


def panel_b(config):

    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    V_food, V_water = train_agent_papageorgiou_exp()

    value = [
        config['theta_f'] * V_food[valued_s] +
        config['theta_w'] * V_water[valued_s],
        config['theta_f'] * V_food[devalued_s] +
        config['theta_w'] * V_water[devalued_s],
    ]

    return {
        'value-along-index': torch.stack([
            torch.Tensor(value),
            torch.Tensor(list(range(len(value)))),
        ]).t().tolist(),
        'done': True,
    }


def panel_c(config):

    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    V_food, V_water = train_agent_papageorgiou_exp()

    value = [
        config['theta_f'] * (food_water-V_food[valued_s]) +
        config['theta_w'] * (no_food_water-V_water[valued_s]),
        config['theta_f'] * (no_food_water-V_food[devalued_s]) +
        config['theta_w'] * (food_water-V_water[devalued_s]),
        config['theta_f'] * (more_food_water-V_food[valued_s]) +
        config['theta_w'] * (no_food_water-V_water[valued_s]),
        config['theta_f'] * (no_food_water-V_food[devalued_s]) +
        config['theta_w'] * (more_food_water-V_water[devalued_s]),
        config['theta_f'] * (food_water-V_food[devalued_s]) +
        config['theta_w'] * (no_food_water-V_water[devalued_s]),
        config['theta_f'] * (no_food_water-V_food[valued_s]) +
        config['theta_w'] * (food_water-V_water[valued_s]),
    ]

    return {
        'value-along-index': torch.stack([
            torch.Tensor(value),
            torch.Tensor(list(range(len(value)))),
        ]).t().tolist(),
        'done': True,
    }


def panel_bc(config):

    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    V_food, V_water = train_agent_papageorgiou_exp()

    value = [
        config['theta_f'] * V_food[valued_s] +
        config['theta_w'] * V_water[valued_s],
        config['theta_f'] * V_food[devalued_s] +
        config['theta_w'] * V_water[devalued_s],
        config['theta_f'] * (food_water-V_food[valued_s]) +
        config['theta_w'] * (no_food_water-V_water[valued_s]),
        config['theta_f'] * (no_food_water-V_food[devalued_s]) +
        config['theta_w'] * (food_water-V_water[devalued_s]),
        config['theta_f'] * (more_food_water-V_food[valued_s]) +
        config['theta_w'] * (no_food_water-V_water[valued_s]),
        config['theta_f'] * (no_food_water-V_food[devalued_s]) +
        config['theta_w'] * (more_food_water-V_water[devalued_s]),
        config['theta_f'] * (food_water-V_food[devalued_s]) +
        config['theta_w'] * (no_food_water-V_water[devalued_s]),
        config['theta_f'] * (no_food_water-V_food[valued_s]) +
        config['theta_w'] * (food_water-V_water[valued_s]),
    ]

    return {
        'value-along-index': torch.stack([
            torch.Tensor(value),
            torch.Tensor(list(range(len(value)))),
        ]).t().tolist(),
        'done': True,
    }


def panel_b_fit_data_and_plot(df):

    config_columns = [
        'theta_f',
        'theta_w',
    ]

    mean_columns = [
        'seed',
    ]

    metric_column = 'value'

    method_column = 'method'

    plot_column = 'index'

    df = au.extract_plot(df, metric_column, plot_column)

    df, s = fd.fit_data(
        df=df,
        config_columns=config_columns,
        mean_columns=mean_columns,
        metric_column=metric_column,
        method_column=method_column,
        plot_column=plot_column,
        raw_data=[
            # valued
            [0, 0.30621110563277043],
            # devalued
            [1, 0.006223098756712675],
        ],
        process_plot_column_fn_in_raw_data=lambda plot: np.round(plot),
        fit_with='k',
    )

    plot_kwargs = {
        'data': df,
        'x': plot_column,
        'y': f'{metric_column}: fitted',
        'col': method_column,
        # 'palette': 'rocket',
        # 'aspect': 0.84,
        # 'sharey': True,
    }

    g = au.nature_catplot(
        kind='bar',
        capsize=0.2,
        errwidth=3,
        errorbar=('se'),
        **plot_kwargs,
    )


def panel_c_fit_data_and_plot(df):

    config_columns = [
        'theta_f',
        'theta_w',
    ]

    mean_columns = [
        'seed',
    ]

    metric_column = 'value'

    method_column = 'method'

    plot_column = 'index'

    df = au.extract_plot(df, metric_column, plot_column)

    df, s = fd.fit_data(
        df=df,
        config_columns=config_columns,
        mean_columns=mean_columns,
        metric_column=metric_column,
        method_column=method_column,
        plot_column=plot_column,
        raw_data=[
            # expected valued
            [0, (-0.09460814540014345-0.1070539419087137)/2],
            # expected devalued
            [1, (-0.13230665035834505-0.1867219917012451)/2],
            # more valued
            [2, (0.7907926739139199+0.4820744702293768)/2],
            # more devalued
            [3, (0.16999019067711207-0.010904473319985675)/2],
            # switch to valued
            [4, (0.12589283006661478-0.015767634854771784)/2],
            # switch to devalued
            [5, (-0.1347660396049613-0.16514522821576755)/2],
        ],
        process_plot_column_fn_in_raw_data=lambda plot: np.round(plot),
        fit_with='k',
    )

    plot_kwargs = {
        'data': df,
        'x': plot_column,
        'y': f'{metric_column}: fitted',
        'col': method_column,
        # 'palette': 'rocket',
        # 'aspect': 0.84,
        # 'sharey': True,
    }

    g = au.nature_catplot(
        kind='bar',
        capsize=0.2,
        errwidth=3,
        errorbar=('se'),
        **plot_kwargs,
    )


def panel_bc_fit_data_and_plot(df):

    config_columns = [
        'theta_f',
        'theta_w',
    ]

    mean_columns = [
        'seed',
    ]

    metric_column = 'value'

    method_column = 'method'

    plot_column = 'index'

    df = au.extract_plot(df, metric_column, plot_column)

    df, s = fd.fit_data(
        df=df,
        config_columns=config_columns,
        mean_columns=mean_columns,
        metric_column=metric_column,
        method_column=method_column,
        plot_column=plot_column,
        raw_data=[
            # valued
            [0, 0.30621110563277043],
            # devalued
            [1, 0.006223098756712675],
            # expected valued
            [2, (-0.09460814540014345-0.1070539419087137)/2],
            # expected devalued
            [3, (-0.13230665035834505-0.1867219917012451)/2],
            # more valued
            [4, (0.7907926739139199+0.4820744702293768)/2],
            # more devalued
            [5, (0.16999019067711207-0.010904473319985675)/2],
            # switch to valued
            [6, (0.12589283006661478-0.015767634854771784)/2],
            # switch to devalued
            [7, (-0.1347660396049613-0.16514522821576755)/2],
        ],
        process_plot_column_fn_in_raw_data=lambda plot: np.round(plot),
        fit_with='k',
    )

    plot_kwargs = {
        'data': df,
        'x': plot_column,
        'y': f'{metric_column}: fitted',
        'hue': method_column,
        # 'palette': 'rocket',
        # 'aspect': 0.84,
        # 'sharey': True,
    }

    g = au.nature_catplot(
        kind='bar',
        capsize=0.2,
        errwidth=3,
        errorbar=('se'),
        **plot_kwargs,
    )
