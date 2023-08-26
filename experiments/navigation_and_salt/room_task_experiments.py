import numpy as np
from envs import *
from learners import *
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import os

np.random.seed(123)


sns.set_theme('talk', font_scale=1.2)
# plt.rcParams['figure.figsize'] = (np.array([6.4, 4.8])/3).tolist()
cmap = sns.color_palette('Blues', as_cmap=True)

def save_figure(sname):
    plt.savefig(sname+'.png', format='png')
    plt.savefig(sname+'.pdf', format='pdf')

def room_r1(env, position, print_det=False, simulated=False, neg_reward=-0.1, pos_reward=5):
    xpos, ypos = position
    if print_det:
        print("in r1: " + str(position) + "  " +
              str(env.state[xpos, ypos]) + " " + str(env.p1_rewarded))
    if env.state[xpos, ypos] == 2 and not env.p1_rewarded:
        if not simulated:
            env.p1_rewarded = True
        return pos_reward
    elif env.state[xpos, ypos] == 3:
        return neg_reward
    elif env.state[xpos, ypos] == 4:
        return neg_reward
    else:
        return -0.1


def room_r2(env, position, print_det=False, simulated=False, neg_reward=-0.1, pos_reward=5):
    xpos, ypos = position
    if print_det:
        print("in r2: " + str(position) + "  " + str(env.state[xpos, ypos]))
    if env.state[xpos, ypos] == 2:
        return neg_reward
    elif env.state[xpos, ypos] == 3 and not env.p2_rewarded:
        if not simulated:
            env.p2_rewarded = True
        return pos_reward
    elif env.state[xpos, ypos] == 4:
        return neg_reward
    else:
        return -0.1


def room_r3(env, position, print_det=False, simulated=False, neg_reward=-0.1, pos_reward=5):
    if print_det:
        print("in r3: " + str(position) + "  " + str(env.state[xpos, ypos]))
    xpos, ypos = position
    if env.state[xpos, ypos] == 2:
        return neg_reward
    elif env.state[xpos, ypos] == 3:
        return neg_reward
    elif env.state[xpos, ypos] == 4 and not env.p3_rewarded:
        if not simulated:
            env.p3_rewarded = True
        return pos_reward
    else:
        return -0.1


def room_all(env, position, simulated=False, print_det=False):
    xpos, ypos = position
    if env.state[xpos, ypos] == 2 and not env.p1_touched:
        if not simulated:
            env.p1_rewarded = True
        return 10
    elif env.state[xpos, ypos] == 3 and not env.p2_touched:
        if not simulated:
            env.p2_rewarded = True
        return 10
    elif env.state[xpos, ypos] == 4 and not env.p3_touched:
        if not simulated:
            env.p3_rewarded = True
        return 10
    else:
        return -0.1


def room_two_vb(env, position, simulated=False, print_det=False):
    return (0 * room_r1(env, position, simulated=simulated)) + (1 * room_r2(env, position, simulated=simulated)) + (1 * room_r3(env, position, simulated=simulated))


def room_combined(env, position, simulated=False, print_det=False, coeff=1):
    return (coeff * room_r1(env, position, simulated=simulated)) + (coeff * room_r2(env, position, simulated=simulated)) + (coeff * room_r3(env, position, simulated=simulated))


def room_triple_reversal_protocol(agent, steps_per_reversal, r1, r2, r3, RB_learner=False, homeostatic_learner=False):
    if RB_learner:
        print("IN RB learner")
        agent.alphas = [1, 0, 0]
    elif homeostatic_learner:
        agent.reward_function = r1
        agent.set_kappa_reward_function()
        print("INIT HOMEOSTATIC")
    else:
        agent.reward_function = r1
    # run first interaction
    rs1, V1s = agent.interact(steps_per_reversal)
    # reversal
    if RB_learner:
        agent.alphas = [0, 1, 0]
        # ensure that the value function is updated immediately
        agent.V = agent.compute_total_v(agent.alphas)
        #agent.env.termination_condition = agent.env.all_points_termination_condition
    elif homeostatic_learner:
        agent.reward_function = r2
        agent.set_kappa_reward_function()

    else:
        agent.reward_function = r2
        #agent.env.termination_condition = agent.env.all_points_termination_condition
    rs2, V2s = agent.interact(steps_per_reversal)
    # reversal
    if RB_learner:
        agent.alphas = [0, 0, 1]
        # ensure that the value function is updated immediately
        agent.V = agent.compute_total_v(agent.alphas)
        #agent.env.termination_condition = agent.env.all_points_termination_condition
    elif homeostatic_learner:
        agent.reward_function = r3
        agent.set_kappa_reward_function()
    else:
        agent.reward_function = r3
        #agent.env.termination_condition = agent.env.all_points_termination_condition
    rs3, V3s = agent.interact(steps_per_reversal)

    # reversal
    if RB_learner:
        agent.alphas = [1, 0, 0]
        # ensure that the value function is updated immediately
        agent.V = agent.compute_total_v(agent.alphas)
        #agent.env.termination_condition = agent.env.all_points_termination_condition
    elif homeostatic_learner:
        agent.reward_function = r1
        agent.set_kappa_reward_function()
    else:
        agent.reward_function = r1
        #agent.env.termination_condition = agent.env.all_points_termination_condition
    rs4, V4s = agent.interact(steps_per_reversal)

    return np.concatenate((rs1, rs2, rs3, rs4), axis=0), np.concatenate((V1s, V2s, V3s, V4s), axis=0)


def room_all_reversal_protocol(agent, steps_per_reversal, r1, r_all, RB_learner=False):
    print("USING ROOM ALL REVRESAL PROTOCOL")
    if RB_learner:
        agent.alphas = [1, 0, 0]
    else:
        agent.reward_function = r1
    rs1, V1s = agent.interact(steps_per_reversal)
    if RB_learner:
        #print("RB learner")
        # sns.heatmap(cmap=cmap,data=agent.V.reshape(6,6))
        #
        agent.alphas = [0, 1, 1]
        agent.V = agent.compute_total_v(agent.alphas)
        # sns.heatmap(cmap=cmap,data=agent.V.reshape(6,6))
        #
        # for v in agent.Vs:
        #  sns.heatmap(cmap=cmap,data=v.reshape(6,6))
        #
    else:
        print("NOT RB LEARNER")
        agent.reward_function = r_all
    rs2, V2s = agent.interact(steps_per_reversal)
    return np.concatenate((rs1, rs2), axis=0), np.concatenate((V1s, V2s), axis=0)


def plot_reward_figure(rs, rs_stds=None, sname=None, title="Reward During Reversal for Temporal Difference learner", N_reversals=3):
    fig = plt.figure()
    xs = np.arange(0, len(rs))
    plt.plot(xs, rs, label="Choice", linewidth="2")
    if rs_stds is not None:
        plt.fill_between(xs, rs - rs_stds, rs + rs_stds, alpha=0.5)
    # sns.despine(left=False, top=True, right=True, bottom=False)
    plt.ylabel("Total reward per episode")
    plt.xlabel("Timestep")
    plt.title(title)
    for i in range(N_reversals):
        plt.axvline(steps_per_reversal * (i+1), color="green",
                    linestyle="--", linewidth="1.5", label="Motivational Switch")
    plt.legend()
    plt.xticks()
    plt.yticks()
    if sname is not None:
        save_figure(sname)


def plot_combined_figure(td_means, td_stds, rb_means, rb_stds, sname=None, title="Average Reward before and after reversal", ylabel_label="Mean Reward at each time-step", label_yaxis=False):
    fig = plt.figure()
    xs = np.arange(0, len(td_means))
    plt.plot(xs, td_means, label="TD Learner", linewidth="2")
    plt.fill_between(xs, td_means - td_stds, td_means + td_stds, alpha=0.5)
    plt.plot(xs, rb_means, label="RB Learner", linewidth="2")
    plt.fill_between(xs, rb_means - rb_stds, rb_means + rb_stds, alpha=0.5)
    # sns.despine(left=False, top=True, right=True, bottom=False)
    plt.xlabel("Episode")
    plt.ylabel(str(ylabel_label))
    plt.title(title)
    for i in range(3):
        plot_label = "Motivational Switch" if i == 0 else None
        plt.axvline(steps_per_reversal * (i+1), color="green",
                    linestyle="--", linewidth="1.5", label=plot_label)
    plt.yticks()
    plt.xticks()
    plt.legend()
    if sname is not None:
        save_figure(sname)


def plot_triple_combined_figure(td_means, td_stds, rb_means, rb_stds, sr_means, sr_stds, steps_per_reversal, sname="None", title="Average Reward before and after reversal", ylabel_label="Mean Reward at each time-step", label_yaxis=False, N_reversals=3, final_label="SR"):
    fig = plt.figure()
    xs = np.arange(0, len(td_means))
    # sns.set_theme(context='talk', font='sans-serif', font_scale=1.0)
    plt.plot(xs, td_means, label="TD", linewidth="2")
    plt.fill_between(xs, td_means - td_stds, td_means + td_stds, alpha=0.5)
    plt.plot(xs, rb_means, label="RB", linewidth="2")
    plt.fill_between(xs, rb_means - rb_stds, rb_means + rb_stds, alpha=0.5)
    plt.plot(xs, sr_means, label=final_label, linewidth="2")
    plt.fill_between(xs, sr_means - sr_stds, sr_means + sr_stds, alpha=0.5)
    # sns.despine(left=False, top=True, right=True, bottom=False)
    plt.xlabel("Episode")
    plt.ylabel(str(ylabel_label))
    plt.title(title)
    for i in range(N_reversals):
        plot_label = "Motivational Switch" if i == 0 else None
        plt.axvline(steps_per_reversal * (i+1), color="green",
                    linestyle="--", linewidth="1.5", label=plot_label)
    plt.yticks()
    plt.xticks()
    plt.legend()
    if sname is not None:
        save_figure(sname)


def plot_quadruple_combined_figure(td_means, td_stds, rb_means, rb_stds, sr_means, sr_stds, h_means, h_stds, steps_per_reversal, sname="None", title="Average Reward before and after reversal", ylabel_label="Mean Reward at each time-step", label_yaxis=False, N_reversals=3, final_label="SR"):
    fig = plt.figure()
    xs = np.arange(0, len(td_means))
    print("TD stds: ", td_stds)
    print("rb_stds, ", rb_stds)
    print("sr_stds, ", sr_stds)
    print("H stds: ", h_stds)
    # sns.set_theme(context='talk', font='sans-serif', font_scale=1.0)
    plt.plot(xs, rb_means, label="RB", linewidth="2")
    plt.fill_between(xs, rb_means - rb_stds, rb_means + rb_stds, alpha=0.5)
    plt.plot(xs, sr_means, label="SR", linewidth="2")
    plt.fill_between(xs, sr_means - sr_stds, sr_means + sr_stds, alpha=0.5)
    plt.plot(xs, td_means, label="TD", linewidth="2")
    plt.fill_between(xs, td_means - td_stds, td_means + td_stds, alpha=0.5)
    plt.plot(xs, h_means, label="Homeostatic", linewidth="2", color="purple")
    plt.fill_between(xs, h_means - h_stds, h_means +
                     h_stds, alpha=0.5, color="purple")
    # sns.despine(left=False, top=True, right=True, bottom=False)
    plt.xlabel("Episode")
    plt.ylabel(str(ylabel_label))
    plt.title(title)
    for i in range(N_reversals):
        plot_label = "Motivational Switch" if i == 0 else None
        plt.axvline(steps_per_reversal * (i+1), color="green",
                    linestyle="--", linewidth="1.5", label=plot_label)
    plt.yticks()
    plt.xticks()
    plt.legend()
    if sname is not None:
        save_figure(sname)


def plot_results(learning_rate, beta, gamma, steps_per_reversal, plot_results=False, use_successor_agent=False, plot_successor_matrix=False, preset_env=None, use_homeostatic_agent=False):
    if preset_env is not None:
        print("USING PRESPECIFIED ENVIRONMENT")
        env = preset_env
    else:
        env = RoomEnv()
    TD_agent = TD_Learner(gamma, room_r1, deepcopy(env), learning_rate, beta)
    RB_agent = Reward_Basis_Learner(gamma, [room_r1, room_r2, room_r3], deepcopy(
        env), learning_rate, beta, [1, 0, 0])
    if use_successor_agent:
        print("INITING successor agent")
        SR_agent = SuccessorRepresentationLearner(
            gamma, room_r2, deepcopy(env), learning_rate, beta)
    if use_homeostatic_agent:
        H_agent = Homeostatic_TD_Learner(
            gamma, room_r2, deepcopy(env), learning_rate, beta)

    rs_rb, vs_rb = room_triple_reversal_protocol(
        RB_agent, steps_per_reversal, room_r1, room_r2, room_r3, RB_learner=True)
    rs_td, vs_td = room_triple_reversal_protocol(
        TD_agent, steps_per_reversal, room_r1, room_r2, room_r3, RB_learner=False)
    if use_successor_agent:
        rs_sr, vs_sr = room_triple_reversal_protocol(
            SR_agent, steps_per_reversal, room_r1, room_r2, room_r3, RB_learner=False)
        if plot_successor_matrix:
            M = SR_agent.M
            V = SR_agent.V
            print(V.shape)
            print(V.reshape(6, 6))
            sns.heatmap(cmap=cmap, data=V.reshape(6, 6))

            print("M:", M.shape)
            sns.heatmap(cmap=cmap, data=M)

    if use_homeostatic_agent:
        rs_h, vs_h = room_triple_reversal_protocol(
            H_agent, steps_per_reversal, room_r1, room_r2, room_r3, RB_learner=False, homeostatic_learner=True)
    if plot_results:
        plot_reward_figure(
            rs_rb, title="Reward During Reversal for Reward Basis Learner")
        plot_reward_figure(
            rs_td, title="Reward During Reversal for Temporal Difference Learner")
        if use_successor_agent:
            plot_reward_figure(
                rs_sr, title="Reward During Reversal for Successor Agent")

    if use_successor_agent and use_homeostatic_agent:
        return rs_td, vs_td, rs_rb, vs_rb, rs_sr, vs_sr, rs_h, vs_h
    if use_successor_agent:
        return rs_td, vs_td, rs_rb, vs_rb, rs_sr, vs_sr
    if use_homeostatic_agent:
        return rs_td, vs_td, rs_rb, vs_rb, rs_h, vs_h
    else:
        return rs_td, vs_td, rs_rb, vs_rb, None, None


def plot_combined_experiment(learning_rate, beta, gamma, steps_per_reversal, plot_results=True, use_successor_agent=False, plot_successor_matrix=False, preset_env=None):
    print("IN PLOT COMBINED EXPERIMENT")
    if preset_env is not None:
        print("USING PRESPECIFIED ENVIRONMENT")
        env = preset_env
    else:
        env = RoomEnv()
    TD_agent = TD_Learner(gamma, room_r1, env, learning_rate, beta)
    RB_agent = Reward_Basis_Learner(
        gamma, [room_r1, room_r2, room_r3], env, learning_rate, beta, [1, 0, 0])
    if use_successor_agent:
        print("INITING successor agent")
        SR_agent = SuccessorRepresentationLearner(
            gamma, room_r1, env, learning_rate, beta)

    rs_rb, vs_rb = room_all_reversal_protocol(
        RB_agent, steps_per_reversal, room_r1, room_combined, RB_learner=True)
    rs_td, vs_td = room_all_reversal_protocol(
        TD_agent, steps_per_reversal, room_r1, room_combined, RB_learner=False)
    if use_successor_agent:
        rs_sr, vs_sr = room_all_reversal_protocol(
            SR_agent, steps_per_reversal, room_r1, room_combined, RB_learner=False)
        if plot_successor_matrix:
            M = SR_agent.M
            V = SR_agent.V
            print(V.shape)
            print(V.reshape(6, 6))
            sns.heatmap(cmap=cmap, data=V.reshape(6, 6))

            print("M:", M.shape)
            sns.heatmap(cmap=cmap, data=M)

    if plot_results:
        plot_reward_figure(
            rs_rb, title="Reward During Reversal for Reward Basis Learner", N_reversals=1)
        plot_reward_figure(
            rs_td, title="Reward During Reversal for Temporal Difference Learner", N_reversals=1)
        if use_successor_agent:
            plot_reward_figure(
                rs_sr, title="Reward During Reversal for Successor Agent", N_reversals=1)

    if use_successor_agent:
        return rs_td, vs_td, rs_rb, vs_rb, rs_sr, vs_sr
    else:
        return rs_td, vs_td, rs_rb, vs_rb, None, None


def plot_N_results(N_results, learning_rate, beta, gamma, steps_per_reversal, results_fn=plot_results, combined_figure_flag=True, use_successor_agent=False, plot_results=True, preset_env=None, use_homeostatic_agent=False, save_data=False, data_sname="", use_standard_error=True, run_afresh=True):

    division_factor = np.sqrt(N_results) if use_standard_error else 1
    if run_afresh:
        rs_tds = []
        rs_rbs = []
        rs_ks = []
        rs_srs = []
        rs_hs = []

        for i in range(N_results):
            print("iteration: ", i)
            if use_successor_agent and use_homeostatic_agent:
                rs_td, V_TD, rs_rb, V_RB, rs_sr, V_SR, rs_h, V_h = results_fn(
                    learning_rate, beta, gamma, steps_per_reversal, use_successor_agent=use_successor_agent, preset_env=preset_env, use_homeostatic_agent=use_homeostatic_agent)
            else:
                rs_td, V_TD, rs_rb, V_RB, rs_sr, V_SR = results_fn(
                    learning_rate, beta, gamma, steps_per_reversal, use_successor_agent=use_successor_agent, preset_env=preset_env, use_homeostatic_agent=use_homeostatic_agent)
            rs_tds.append(rs_td)
            rs_rbs.append(rs_rb)
            rs_srs.append(rs_sr)
            if use_successor_agent and use_homeostatic_agent:
                rs_hs.append(rs_h)

        rs_tds = np.array(rs_tds)
        print("rs tds: ", rs_tds.shape)
        rs_rbs = np.array(rs_rbs)
        rs_tds_mean = np.mean(rs_tds, axis=0)
        rs_rbs_mean = np.mean(rs_rbs, axis=0)
        rs_tds_std = np.std(rs_tds, axis=0) / division_factor
        rs_rbs_std = np.std(rs_rbs, axis=0) / division_factor
        if use_successor_agent or use_homeostatic_agent:
            rs_srs = np.array(rs_srs)
            print("RS:SR:", rs_sr.shape)
            rs_srs_mean = np.mean(rs_srs, axis=0)
            print(rs_srs_mean)
            rs_srs_std = np.std(rs_srs, axis=0) / division_factor
        if use_homeostatic_agent and use_successor_agent:
            rs_hs = np.array(rs_hs)
            print("RS H", rs_h.shape)
            rs_h_mean = np.mean(rs_hs, axis=0)
            rs_h_std = np.std(rs_hs, axis=0) / (division_factor * 100)

        if save_data:
            np.save("data/" + str(data_sname) + "rs_tds.npy", rs_tds)
            np.save("data/" + str(data_sname) + "rs_rbs.npy", rs_rbs)
            np.save("data/" + str(data_sname) + "rs_srs.npy", rs_srs)
            if use_homeostatic_agent and use_successor_agent:
                np.save("data/" + str(data_sname) + "rs_h.npy", rs_hs)
    else:
        rs_tds = np.load("data/" + str(data_sname) + "rs_tds.npy")
        rs_rbs = np.load("data/" + str(data_sname) + "rs_rbs.npy")
        rs_srs = np.load("data/" + str(data_sname) + "rs_srs.npy")
        if use_homeostatic_agent and use_successor_agent:
            rs_hs = np.load("data/" + str(data_sname) + "rs_h.npy")
        rs_tds = np.array(rs_tds)
        print("rs tds: ", rs_tds.shape)
        rs_rbs = np.array(rs_rbs)
        rs_tds_mean = np.mean(rs_tds, axis=0)
        rs_rbs_mean = np.mean(rs_rbs, axis=0)
        rs_tds_std = np.std(rs_tds, axis=0) / division_factor
        rs_rbs_std = np.std(rs_rbs, axis=0) / division_factor
        if use_successor_agent or use_homeostatic_agent:
            rs_srs = np.array(rs_srs)
            rs_srs_mean = np.mean(rs_srs, axis=0)
            print(rs_srs_mean)
            rs_srs_std = np.std(rs_srs, axis=0) / division_factor
        if use_homeostatic_agent and use_successor_agent:
            rs_hs = np.array(rs_hs)
            rs_h_mean = np.mean(rs_hs, axis=0)
            rs_h_std = np.std(rs_hs, axis=0) / (division_factor * 100)

    if plot_results:
        if combined_figure_flag:
            if use_successor_agent and use_homeostatic_agent:
                plot_quadruple_combined_figure(rs_tds_mean, rs_tds_std, rs_rbs_mean, rs_rbs_std, rs_srs_mean, rs_srs_std, rs_h_mean, rs_h_std, steps_per_reversal,
                                               title="Reward for RB, TD, SR, and Homeostatic", sname="figures/room_task_quadruple_"+str(learning_rate), N_reversals=3)
            elif use_successor_agent:
                plot_triple_combined_figure(rs_tds_mean, rs_tds_std, rs_rbs_mean, rs_rbs_std, rs_srs_mean, rs_srs_std, steps_per_reversal,
                                            title="Reward during reversal for RB, TD, and SR", sname="figures/combined_room_task_reward_3_" + str(learning_rate), N_reversals=3)
            elif use_homeostatic_agent:
                plot_triple_combined_figure(rs_tds_mean, rs_tds_std, rs_rbs_mean, rs_rbs_std, rs_srs_mean, rs_srs_std, steps_per_reversal, title="Reward during reversal for RB, TD, and Homeostatic",
                                            sname="figures/homeostatic_combined_room_task_reward_3_" + str(learning_rate), N_reversals=3, final_label="Homeostatic")
            else:
                plot_combined_figure(rs_tds_mean, rs_tds_std, rs_rbs_mean, rs_rbs_std, title="Reward During Reversal for TD and RB Learner",
                                     sname="figures/combined_room_task_reward_2_" + str(learning_rate))

        else:
            plot_reward_figure(rs_tds_mean, rs_tds_std, title="Reward During Reversal for Temporal Difference Learner",
                               sname="figures/room_task_td_reward")
            plot_reward_figure(rs_rbs_mean, rs_rbs_std, title="Reward During Reversal for Reward Basis Learner",
                               sname="figures/room_task_rb_reward")
            if use_successor_agent:
                plot_reward_figure(
                    rs_srs_mean, rs_srs_std, title="Reward During Reversal for Successor Representation Learner", sname="figures/room_task_sr_reward")
        #plot_reward_figure(rs_ks_mean, rs_ks_std,title="Reward During Reversal for Homeostatic Temporal Difference Learner")
    if use_homeostatic_agent and use_successor_agent:
        return rs_tds, rs_rbs, rs_srs, rs_hs
    else:
        return rs_tds, rs_rbs, rs_srs


def plot_successor_matrix_evolution(learning_rate, beta, gamma, N_iterations, plot_N):

    env = RoomEnv()
    RB_agent = Reward_Basis_Learner(
        gamma, [room_r1, room_r2, room_r3], env, learning_rate, beta, [1, 0, 0], random_policy=True)
    SR_agent = SuccessorRepresentationLearner(
        gamma, room_r2, env, learning_rate, beta, random_policy=True)
    rs, Vs, Vss = RB_agent.interact(N_iterations, return_rb_vlist=True)
    _, _, Ms = SR_agent.interact(N_iterations, return_mlist=True)

    step_size = N_iterations // plot_N
    columns = 5
    rows = 4
    fig, ax_array = plt.subplots(
        rows, columns, squeeze=False, )
    for i, ax_row in enumerate(ax_array):
        for j, axes in enumerate(ax_row):
            idx = int((i * rows) + j) * step_size
            M = Ms[idx, :, :]
            axes.imshow(M)
            axes.set_title('{}'.format(idx))
            axes.set_yticklabels([])
            axes.set_xticklabels([])
            axes.set_xticks([])
            axes.set_yticks([])
    fig.suptitle("Evolution of the Learned Successor Matrix")
    # plt.tight_layout()
    save_figure("figures/successor_evolution_2")

    columns = 5
    rows = 4
    fig, ax_array = plt.subplots(
        rows, columns, squeeze=False, )
    for i, ax_row in enumerate(ax_array):
        for j, axes in enumerate(ax_row):
            idx = int((i * rows) + j) * step_size
            V = Vss[idx, 0, :].reshape(6, 6)
            axes.imshow(V)
            axes.set_title('{}'.format(idx))
            axes.set_yticklabels([])
            axes.set_xticklabels([])
            axes.set_xticks([])
            axes.set_yticks([])
    fig.suptitle("Evolution of the Learned Value Function")
    # plt.tight_layout()
    save_figure("figures/VF_evolution_2")


def convergence_times(learning_rate, beta, gamma, N_iterations, N_runs):
    Vss_diffss = []
    Ms_diffss = []
    for n in range(N_runs):
        env = RoomEnv()
        RB_agent = Reward_Basis_Learner(
            gamma, [room_r1, room_r2, room_r3], env, learning_rate, beta, [1, 0, 0], random_policy=True)
        SR_agent = SuccessorRepresentationLearner(
            gamma, room_r2, env, learning_rate, beta, random_policy=True)
        rs, Vs, Vss = RB_agent.interact(N_iterations, return_rb_vlist=True)
        _, _, Ms = SR_agent.interact(N_iterations, return_mlist=True)
        print("VSS: ", Vss.shape)
        print("MSS: ", Ms.shape)
        Vss = np.array([Vss[i, :, :] / np.sum(np.abs(Vss[i, :, :]))
                       for i in range(len(Vss))])
        Ms = np.array([Ms[i, :, :] / np.sum(np.abs(Ms[i, :, :]))
                      for i in range(len(Ms))])
        final_Vss = Vss[-1, :, :]
        final_Ms = Ms[-1, :, :]

        Vss_diffs = np.array([np.abs(Vss[i, :, :] - final_Vss)
                             for i in range(len(Vss))])
        Vss_mean_diffs = np.sum(Vss_diffs, axis=(1, 2))
        #Vss_std_diffs = np.std(Vss_diffs, axis=[1,2])

        Ms_diffs = np.array([np.abs(Ms[i, :, :] - final_Ms)
                            for i in range(len(Ms))])
        Ms_mean_diffs = np.sum(Ms_diffs, axis=(1, 2))
        Vss_diffss.append(Vss_mean_diffs)
        Ms_diffss.append(Ms_mean_diffs)
        #Ms_std_diffs = np.std(Ms_diffs, axis=[1,2])
    Vss_diffss = np.array(Vss_diffss)
    Ms_diffss = np.array(Ms_diffss)
    print(Ms_diffss.shape)
    Vss_mean_diffs = np.mean(Vss_diffss, axis=0)
    Ms_mean_diffs = np.mean(Ms_diffss, axis=0)
    Vss_std_diffs = np.std(Vss_diffss, axis=0)
    Ms_std_diffs = np.std(Ms_diffss, axis=0)
    xs = np.arange(0, len(Vss_mean_diffs))

    # figure 1
    fig = plt.figure()
    # sns.set_theme(context='talk', font='sans-serif', font_scale=1.0)

    plt.plot(xs, Vss_mean_diffs, label="RB")
    plt.fill_between(xs, Vss_mean_diffs - Vss_std_diffs,
                     Vss_mean_diffs + Vss_std_diffs, alpha=0.5)
    plt.plot(xs, Ms_mean_diffs, label="SR")
    plt.fill_between(xs, Ms_mean_diffs - Ms_std_diffs,
                     Ms_mean_diffs + Ms_std_diffs, alpha=0.5)
    # plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Mean absolute difference")
    plt.title("Convergence of SR and RB")
    # sns.despine(left=False, top=True, right=True, bottom=False)
    # plt.xticks()
    # plt.yticks()
    save_figure("figures/SR_RB_convergence")


def learning_rate_sweep(beta, gamma, steps_per_reversal):
    #lrs = [0.01,0.05,0.1,0.2,0.3]
    lrs = [0.5]
    for lr in lrs:
        plot_N_results(10, lr, beta, gamma, steps_per_reversal,
                       combined_figure_flag=True, use_successor_agent=True)


def learning_rate_sweep_combined_graph(lrs, beta, gamma, steps_per_reversal, N_runs=10, run_afresh=False, line_plot=False, fignum=0, intermediate_plots=False, use_homeostatic=True):
    if run_afresh:
        rb_rs = []
        td_rs = []
        sr_rs = []
        rb_std = []
        td_std = []
        sr_std = []
        h_rs = []
        for lr in lrs:
            if use_homeostatic:
                rs_tds, rs_rbs, rs_srs, rs_h = plot_N_results(
                    N_runs, lr, beta, gamma, steps_per_reversal, use_successor_agent=True, plot_results=False, use_homeostatic_agent=use_homeostatic)
            else:
                rs_tds, rs_rbs, rs_srs = plot_N_results(
                    N_runs, lr, beta, gamma, steps_per_reversal, use_successor_agent=True, plot_results=False, use_homeostatic_agent=use_homeostatic)
            rb_rs.append(np.mean(rs_rbs, axis=1))
            td_rs.append(np.mean(rs_tds, axis=1))
            sr_rs.append(np.mean(rs_srs, axis=1))
            if use_homeostatic:
                h_rs.append(np.mean(rs_h, axis=1))
            if intermediate_plots:
                fig = plt.figure()
                plt.plot(np.mean(rs_rbs, axis=0), label="RB")
                plt.plot(np.mean(rs_srs, axis=0), label="SR")
                plt.legend()

        rb_rs = np.array(rb_rs)
        print("rb rs: ", rb_rs.shape)
        print(rb_rs)
        td_rs = np.array(td_rs)
        print("td rs: ", td_rs.shape)
        print(td_rs)
        sr_rs = np.array(sr_rs)
        print("sr rs: ", sr_rs.shape)
        print(sr_rs)
        rb_std = np.std(rb_rs, axis=1)
        td_std = np.std(td_rs, axis=1)
        sr_std = np.std(sr_rs, axis=1)
        rb_rs = np.mean(rb_rs, axis=1)
        td_rs = np.mean(td_rs, axis=1)
        sr_rs = np.mean(sr_rs, axis=1)
        if use_homeostatic:
            h_rs = np.array(h_rs)
            h_std = np.std(h_rs, axis=1)
            h_rs = np.mean(h_rs, axis=1)

        np.save("data/lr_combined_rb_rs.npy", rb_rs)
        np.save("data/lr_combined_td_rs.npy", td_rs)
        np.save("data/lr_combined_sr_rs.npy", sr_rs)
        np.save("data/lr_combined_rb_std.npy", rb_std)
        np.save("data/lr_combined_td_std.npy", td_std)
        np.save("data/lr_combined_sr_std.npy", sr_std)
        if use_homeostatic:
            np.save("data/lr_combined_h_rs.npy", h_rs)
            np.save("data/lr_combined_h_std.npy", h_std)
        print("DONE!")
    else:
        rb_rs = np.load("data/lr_combined_rb_rs.npy")  # [2:7]
        td_rs = np.load("data/lr_combined_td_rs.npy")  # [2:7]
        sr_rs = np.load("data/lr_combined_sr_rs.npy")  # [2:7]
        rb_std = np.load("data/lr_combined_rb_std.npy")  # [2:7]
        td_std = np.load("data/lr_combined_td_std.npy")  # [2:7]
        sr_std = np.load("data/lr_combined_sr_std.npy")  # [2:7]
        if use_homeostatic:
            h_rs = np.load("data/lr_combined_h_rs.npy")
            h_std = np.load("data/lr_combined_h_std.npy")

    # figure
    if line_plot:
        xs = [i for i in range(len(lrs))]
        print(len(xs))
        print(len(rb_rs))
        lrs_labels = [str(lr) for lr in lrs]
        fig = plt.figure()
        # sns.set_theme(context='talk', font='sans-serif', font_scale=1.0)
        plt.errorbar(xs, rb_rs, yerr=rb_std /
                     np.sqrt(N_runs), label="RB", capsize=4)
        #plt.fill_between(xs, rb_rs - rb_std,rb_rs + rb_std, alpha=0.5)
        plt.errorbar(xs, sr_rs, yerr=sr_std /
                     np.sqrt(N_runs), label="SR", capsize=4)
        #plt.fill_between(xs, sr_rs - sr_std, sr_rs + sr_std, alpha=0.5)
        plt.errorbar(xs, td_rs, yerr=td_std /
                     np.sqrt(N_runs), label="TD", capsize=4)
        #plt.fill_between(xs, td_rs - td_std, td_rs + td_std, alpha=0.5)
        if use_homeostatic:
            plt.errorbar(xs, h_rs, yerr=h_std /
                         np.sqrt(N_runs), label="H", capsize=4)
        # plt.legend()
        plt.xlabel("Learning Rate")
        plt.xticks(ticks=xs, labels=lrs_labels)
        plt.ylabel("Total Reward")
        plt.title("Reward Obtained against Learning Rate")
        # sns.despine(left=False, top=True, right=True, bottom=False)
        # plt.xticks()
        # plt.yticks()
        save_figure("figures/learning_rate_maze_combined_" +
                    str(fignum))

    else:
        # bar chart
        labels = [str(lr) for lr in lrs]
        xs = np.arange(len(labels))
        # sns.set_theme(context='talk', font='sans-serif', font_scale=1.0)
        fig, ax = plt.subplots()
        width = 0.2
        bar_rb = ax.bar(xs - (width + 0.02), rb_rs, width,
                        label="RB", alpha=0.9, yerr=rb_std)
        #bar_td = ax.bar(xs + (width + 0.02), td_rs, width, label="TD",alpha=0.9,yerr=td_std)
        bar_sr = ax.bar(xs, sr_rs, width, label="SR", alpha=0.9, yerr=sr_std)
        ax.set_ylabel("Total Reward")
        ax.set_xlabel("Learning Rate")
        ax.set_title("Mean Reward obtained against learning rate")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.tick_params(axis='x', which='major')
        ax.tick_params(axis='x', which='minor')
        ax.tick_params(axis='y', which='major')
        ax.tick_params(axis='y', which='minor')
        # ax.legend()
        # sns.despine(left=False, top=True, right=True, bottom=False)

        save_figure(
            "figures/learning_rate_maze_combined_bar_3")


def steps_per_reversal_sweep_combined_graph(lr, beta, gamma, steps_per_reversal, N_runs=10, run_afresh=False, line_plot=False, fignum=0, intermediate_plots=False, use_homeostatic=True, plot_intermediate_results=False):
    if run_afresh:
        rb_rs = []
        td_rs = []
        sr_rs = []
        rb_std = []
        td_std = []
        sr_std = []
        h_rs = []
        for step in steps_per_reversal:
            if use_homeostatic:
                rs_tds, rs_rbs, rs_srs, rs_hs = plot_N_results(
                    N_runs, lr, beta, gamma, step, use_successor_agent=True, plot_results=plot_intermediate_results, use_homeostatic_agent=use_homeostatic)
            else:
                rs_tds, rs_rbs, rs_srs = plot_N_results(
                    N_runs, lr, beta, gamma, step, use_successor_agent=True, plot_results=plot_intermediate_results, use_homeostatic_agent=use_homeostatic)
            rb_rs.append(np.mean(rs_rbs, axis=1))
            td_rs.append(np.mean(rs_tds, axis=1))
            sr_rs.append(np.mean(rs_srs, axis=1))
            if use_homeostatic:
                h_rs.append(np.mean(rs_hs, axis=1))
            if intermediate_plots:
                fig = plt.figure()
                plt.plot(np.mean(rs_rbs, axis=0), label="RB")
                plt.plot(np.mean(rs_srs, axis=0), label="SR")
                plt.legend()

        for el in rb_rs:
            print("EL: ", el.shape)
        print(rb_rs[0].shape)
        rb_rs = np.array(rb_rs)
        print("rb rs: ", rb_rs.shape)
        print(rb_rs)
        td_rs = np.array(td_rs)
        print("td rs: ", td_rs.shape)
        print(td_rs)
        sr_rs = np.array(sr_rs)
        print("sr rs: ", sr_rs.shape)
        print(sr_rs)

        rb_std = np.std(rb_rs, axis=1)
        td_std = np.std(td_rs, axis=1)
        sr_std = np.std(sr_rs, axis=1)
        rb_rs = np.mean(rb_rs, axis=1)
        td_rs = np.mean(td_rs, axis=1)
        sr_rs = np.mean(sr_rs, axis=1)
        if use_homeostatic:
            h_rs = np.array(h_rs)
            h_std = np.std(h_rs, axis=1)
            h_rs = np.mean(h_rs, axis=1)
        np.save("data/steps_combined_rb_rs_3.npy", rb_rs)
        np.save("data/steps_combined_td_rs_3.npy", td_rs)
        np.save("data/steps_combined_sr_rs_3.npy", sr_rs)
        np.save("data/steps_combined_rb_std_3.npy", rb_std)
        np.save("data/steps_combined_td_std_3.npy", td_std)
        np.save("data/steps_combined_sr_std_3.npy", sr_std)
        if use_homeostatic:
            np.save("data/steps_combined_h_rs_3.npy", h_rs)
            np.save("data/steps_combined_h_std_3.npy", h_std)
        print("DONE!")
    else:
        rb_rs = np.load("data/steps_combined_rb_rs_3.npy")  # [2:7]
        td_rs = np.load("data/steps_combined_td_rs_3.npy")  # [2:7]
        sr_rs = np.load("data/steps_combined_sr_rs_3.npy")  # [2:7]
        rb_std = np.load("data/steps_combined_rb_std_3.npy")  # [2:7]
        td_std = np.load("data/steps_combined_td_std_3.npy")  # [2:7]
        sr_std = np.load("data/steps_combined_sr_std_3.npy")  # [2:7]
        if use_homeostatic:
            h_std = np.load("data/steps_combined_h_std_3.npy")
            h_rs = np.load("data/steps_combined_h_rs_3.npy")

    if line_plot:
        xs = [i for i in range(len(steps_per_reversal))]
        print(len(xs))
        print(len(rb_rs))
        lrs_labels = [str(lr) for lr in steps_per_reversal]
        fig = plt.figure()
        # sns.set_theme(context='talk', font='sans-serif', font_scale=1.0)
        plt.errorbar(xs, rb_rs, yerr=rb_std /
                     np.sqrt(N_runs), label="RB", capsize=4)
        #plt.fill_between(xs, rb_rs - rb_std,rb_rs + rb_std, alpha=0.5)
        plt.errorbar(xs, sr_rs, yerr=sr_std /
                     np.sqrt(N_runs), label="SR", capsize=4)
        #plt.fill_between(xs, sr_rs - sr_std, sr_rs + sr_std, alpha=0.5)
        plt.errorbar(xs, td_rs, yerr=td_std /
                     np.sqrt(N_runs), label="TD", capsize=4)
        #plt.fill_between(xs, td_rs - td_std, td_rs + td_std, alpha=0.5)
        if use_homeostatic:
            plt.errorbar(xs, h_rs, yerr=h_std / np.sqrt(N_runs),
                         label="H", capsize=4, color="purple")
        plt.legend()
        plt.xlabel("Steps Per Reversal")
        plt.xticks(ticks=xs, labels=lrs_labels)
        plt.ylabel("Total Reward")
        plt.title("Reward Obtained against Steps Per Reversal")
        # sns.despine(left=False, top=True, right=True, bottom=False)
        plt.xticks()
        plt.yticks()
        save_figure("figures/interval_steps_maze_combined_" +
                    str(fignum))

    else:
        # bar chart
        labels = [str(lr) for lr in steps_per_reversal]
        xs = np.arange(len(labels))
        # sns.set_theme(context='talk', font='sans-serif', font_scale=1.0)
        fig, ax = plt.subplots()
        width = 0.2
        bar_rb = ax.bar(xs - (width + 0.02), rb_rs, width,
                        label="RB", alpha=0.9, yerr=rb_std)
        #bar_td = ax.bar(xs + (width + 0.02), td_rs, width, label="TD",alpha=0.9,yerr=td_std)
        bar_sr = ax.bar(xs, sr_rs, width, label="SR", alpha=0.9, yerr=sr_std)
        ax.set_ylabel("Total Reward")
        ax.set_xlabel("Steps per Reversal")
        ax.set_title(
            "Mean Reward obtained against steps per reversal")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.tick_params(axis='x', which='major')
        ax.tick_params(axis='x', which='minor')
        ax.tick_params(axis='y', which='major')
        ax.tick_params(axis='y', which='minor')
        ax.legend()
        # sns.despine(left=False, top=True, right=True, bottom=False)

        save_figure(
            "figures/interval_steps_maze_combined_bar_3")


def plot_value_function(V, sname, title):
    fig = plt.figure()
    sns.heatmap(cmap=cmap, data=V.reshape(6, 6), vmin=-0.1, vmax=5)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    save_figure(sname)


def plot_reward_function(rfun, sname, title=""):
    env = RoomEnv()
    mat = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            r = rfun(env, [i, j], simulated=True)
            mat[i, j] = r

    fig = plt.figure()
    sns.heatmap(cmap=cmap, data=mat, vmin=-0.1, vmax=5)
    plt.xticks([])
    plt.yticks([])
    if title != "":
        plt.title(title)
    save_figure(sname)


def convergence_rb_td(beta, gamma, learning_rate, N_iterations, N_runs, coeff=0.33):
    all_Vs = []
    all_Vss = []
    for n in range(N_runs):
        env = RoomEnv()
        TD_agent = TD_Learner(gamma, room_combined, env,
                              learning_rate, beta, random_policy=True)
        RB_agent = Reward_Basis_Learner(
            gamma, [room_r1, room_r2, room_r3], env, learning_rate, beta, [1, 0, 0], random_policy=True)
        rb_rs, rb_Vs, rb_Vss = RB_agent.interact(
            N_iterations, return_rb_vlist=True)
        rs, Vs = TD_agent.interact(N_iterations)
        all_Vs.append(Vs)
        all_Vss.append(np.array(rb_Vss))

    all_Vs = np.array(all_Vs)
    all_Vss = np.array(all_Vss)
    print("SHAPES")
    print(all_Vs.shape)
    print(all_Vss.shape)
    # compute diffs
    diffs = np.zeros((N_runs, N_iterations))
    for i in range(len(all_Vs)):
        for n in range(N_iterations):
            td_v = all_Vs[i, n, :]
            vss = all_Vss[i, n, :, :]
            print(td_v.shape)
            print(vss.shape)
            comb_v = (coeff * vss[0, :]) + \
                (coeff * vss[1, :]) + (coeff * vss[2, :])
            print(comb_v.shape)
            diffs[i, n] = np.sum(np.square(td_v - comb_v))

    mean_diffs = np.mean(diffs, axis=0)
    std_diffs = np.std(diffs, axis=0)
    xs = np.arange(0, len(mean_diffs))
    fig = plt.figure()
    # sns.set_theme(context='talk', font='sans-serif', font_scale=1.0)
    plt.plot(xs, mean_diffs, label="Diffs")
    plt.fill_between(xs, mean_diffs - std_diffs,
                     mean_diffs + std_diffs, alpha=0.5)
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Total Squared Differences")
    plt.title("Convergence of TD and RB estimated Value Functions")
    # sns.despine(left=False, top=True, right=True, bottom=False)
    plt.xticks()
    plt.yticks()
    save_figure("figures/combined_TD_RB_convergence")


def plot_combined_value_functions(beta, gamma, learning_rate, N_iterations, no_title=True, coeff_list=[1, 1, 1], sname_add=""):
    env = RoomEnv()
    TD_agent = TD_Learner(gamma, room_combined, env,
                          learning_rate, beta, random_policy=True)
    RB_agent = Reward_Basis_Learner(
        gamma, [room_r1, room_r2, room_r3], env, learning_rate, beta, [1, 0, 0], random_policy=True)
    rb_rs, rb_Vs, rb_Vss = RB_agent.interact(
        N_iterations, return_rb_vlist=True)
    rs, Vs = TD_agent.interact(N_iterations)
    print(rb_Vss.shape)
    print(Vs.shape)
    vss = rb_Vss[-1, :, :]
    TD_V = Vs[-1, :]
    if no_title:
        plot_value_function(
            TD_V, sname="figures/TD_room_vf" + sname_add, title="")
        plot_value_function(
            vss[0, :], sname="figures/RB_room_1vf" + sname_add, title="")
        plot_value_function(
            vss[1, :], sname="figures/RB_room_2vf" + sname_add, title="")
        plot_value_function(
            vss[2, :], sname="figures/RB_room_3vf" + sname_add, title="")
        comb_v = (coeff_list[0] * vss[0, :]) + (coeff_list[1]
                                                * vss[1, :]) + (coeff_list[2] * vss[2, :])
        plot_value_function(
            comb_v, sname="figures/RB_room_comb_v" + sname_add, title="")
    else:
        plot_value_function(TD_V, sname="figures/TD_room_vf" +
                            sname_add, title="TD combined value function")
        plot_value_function(vss[0, :], sname="figures/RB_room_1vf" +
                            sname_add, title="RB Value Basis 1")
        plot_value_function(vss[1, :], sname="figures/RB_room_2vf" +
                            sname_add, title="RB Value Basis 2")
        plot_value_function(vss[2, :], sname="figures/RB_room_3vf" +
                            sname_add, title="RB Value Basis 3")
        comb_v = (coeff_list[0] * vss[0, :]) + (coeff_list[1]
                                                * vss[1, :]) + (coeff_list[2] * vss[2, :])
        plot_value_function(comb_v, sname="figures/RB_room_comb_v" +
                            sname_add, title="RB Combined Value Function")

    print("TD : ", TD_V.reshape(6, 6))
    print("RB: ", comb_v.reshape(6, 6))


def room_sizes_sweep_graph(lr, beta, gamma, room_sizes, steps_per_reversal, N_runs=10, run_afresh=False, line_plot=False, fignum=0, intermediate_plots=False):
    if run_afresh:
        rb_rs = []
        td_rs = []
        sr_rs = []
        rb_std = []
        td_std = []
        sr_std = []
        for room_size in room_sizes:
            env = RoomEnv(room_size=room_size, random_goal_positions=True)
            rs_tds, rs_rbs, rs_srs = plot_N_results(
                N_runs, lr, beta, gamma, steps_per_reversal, use_successor_agent=True, plot_results=True, preset_env=env)
            rb_rs.append(np.mean(rs_rbs, axis=1))
            td_rs.append(np.mean(rs_tds, axis=1))
            sr_rs.append(np.mean(rs_srs, axis=1))
            if intermediate_plots:
                fig = plt.figure()
                plt.plot(np.mean(rs_rbs, axis=0), label="RB")
                plt.plot(np.mean(rs_srs, axis=0), label="SR")
                plt.legend()

        for el in rb_rs:
            print("EL: ", el.shape)
        print(rb_rs[0].shape)
        rb_rs = np.array(rb_rs)
        print("rb rs: ", rb_rs.shape)
        print(rb_rs)
        td_rs = np.array(td_rs)
        print("td rs: ", td_rs.shape)
        print(td_rs)
        sr_rs = np.array(sr_rs)
        print("sr rs: ", sr_rs.shape)
        print(sr_rs)
        rb_std = np.std(rb_rs, axis=1)
        td_std = np.std(td_rs, axis=1)
        sr_std = np.std(sr_rs, axis=1)
        rb_rs = np.mean(rb_rs, axis=1)
        td_rs = np.mean(td_rs, axis=1)
        sr_rs = np.mean(sr_rs, axis=1)
        np.save("data/room_sizes_rb_rs.npy_3", rb_rs)
        np.save("data/room_sizes_td_rs.npy_3", td_rs)
        np.save("data/room_sizes_sr_rs.npy_3", sr_rs)
        np.save("data/room_sizes_rb_std.npy_3", rb_std)
        np.save("data/room_sizes_td_std.npy_3", td_std)
        np.save("data/room_sizes_sr_std.npy_3", sr_std)
        print("DONE!")
    else:
        rb_rs = np.load("data/room_sizes_rb_rs.npy")  # [2:7]
        td_rs = np.load("data/room_sizes_td_rs.npy")  # [2:7]
        sr_rs = np.load("data/room_sizes_sr_rs.npy")  # [2:7]
        rb_std = np.load("data/room_sizes_rb_std.npy")  # [2:7]
        td_std = np.load("data/room_sizes_td_std.npy")  # [2:7]
        sr_std = np.load("data/room_sizes_sr_std.npy")  # [2:7]

    # figure
    if line_plot:
        xs = [i for i in range(len(room_sizes))]
        print(len(xs))
        print(len(rb_rs))
        lrs_labels = [str(room_size) for room_size in room_sizes]
        fig = plt.figure()
        # sns.set_theme(context='talk', font='sans-serif', font_scale=1.0)
        plt.errorbar(xs, rb_rs, yerr=rb_std /
                     np.sqrt(N_runs), label="RB", capsize=4)
        #plt.fill_between(xs, rb_rs - rb_std,rb_rs + rb_std, alpha=0.5)
        plt.errorbar(xs, sr_rs, yerr=sr_std /
                     np.sqrt(N_runs), label="SR", capsize=4)
        #plt.fill_between(xs, sr_rs - sr_std, sr_rs + sr_std, alpha=0.5)
        plt.errorbar(xs, td_rs, yerr=td_std /
                     np.sqrt(N_runs), label="TD", capsize=4)
        #plt.fill_between(xs, td_rs - td_std, td_rs + td_std, alpha=0.5)
        plt.legend()
        plt.xlabel("Room Size")
        plt.xticks(ticks=xs, labels=lrs_labels)
        plt.ylabel("Total Reward")
        plt.title("Reward Obtained Against Room Size")
        # sns.despine(left=False, top=True, right=True, bottom=False)
        plt.xticks()
        plt.yticks()
        save_figure("figures/romm_size_maze_combined_" +
                    str(fignum))

    else:
        # bar chart
        labels = [str(room_size) for room_size in room_sizes]
        xs = np.arange(len(labels))
        # sns.set_theme(context='talk', font='sans-serif', font_scale=1.0)
        fig, ax = plt.subplots()
        width = 0.2
        bar_rb = ax.bar(xs - (width + 0.02), rb_rs, width,
                        label="RB", alpha=0.9, yerr=rb_std)
        #bar_td = ax.bar(xs + (width + 0.02), td_rs, width, label="TD",alpha=0.9,yerr=td_std)
        bar_sr = ax.bar(xs, sr_rs, width, label="SR", alpha=0.9, yerr=sr_std)
        ax.set_ylabel("Total Reward")
        ax.set_xlabel("Room Size")
        ax.set_title("Mean Reward obtained Against Room Size")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.tick_params(axis='x', which='major')
        ax.tick_params(axis='x', which='minor')
        ax.tick_params(axis='y', which='major')
        ax.tick_params(axis='y', which='minor')
        ax.legend()
        # sns.despine(left=False, top=True, right=True, bottom=False)

        save_figure("figures/room_sizes_maze_combined_bar_3")


def verify_kappa_homeostatic():
    env = RoomEnv()
    homeostatic_agent = Homeostatic_TD_Learner(
        gamma, room_r1, env, learning_rate, beta)
    kappa = homeostatic_agent.set_kappa_reward_function()
    sns.heatmap(cmap=cmap, data=kappa.reshape(6, 6))


if __name__ == '__main__':
    if not os.path.exists("figures/"):
        os.makedirs("figures/")
    learning_rate = 0.01
    beta = 1
    gamma = 0.99
    steps_per_reversal = 200
    plot_N_results(20, learning_rate, beta, gamma, steps_per_reversal, results_fn=plot_results, combined_figure_flag=True, use_successor_agent=True,
                   use_homeostatic_agent=True, save_data=True, data_sname="quadruple_combined_recolored_2_", use_standard_error=True, run_afresh=False)
    #convergence_times(learning_rate, beta, gamma, 1000,10)
    #plot_successor_matrix_evolution(learning_rate, beta, gamma, 1000, 20)
    #learning_rate_sweep(beta, gamma, steps_per_reversal)
    plot_combined_value_functions(
        beta, gamma, learning_rate, 5000, no_title=True)
    plot_combined_value_functions(
        beta, gamma, learning_rate, 5000, no_title=True,
        coeff_list=[1, 0.5, 0],
        sname_add="_other_example",
    )
    convergence_rb_td(beta, gamma, learning_rate, 5000, 10)
    plot_reward_function(room_r1, "figures/room_r1")
    plot_reward_function(room_r2, "figures/room_r2")
    plot_reward_function(room_r3, "figures/room_r3")
    plot_reward_function(room_combined, "figures/room_rcombined")

    #lrs = [0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.8]
    lrs = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
    N_steps = 20
    learning_rate_sweep_combined_graph(
        lrs, beta, gamma, N_steps, N_runs=20, run_afresh=False, line_plot=True, fignum=6, use_homeostatic=False)
    lr = 0.1
    #steps = [50,100,150,200]
    steps = [2, 5, 10, 20, 30, 50, 100, 150, 200, 250, 300]
    steps_per_reversal_sweep_combined_graph(
        lr, beta, gamma, steps, N_runs=20, run_afresh=False, line_plot=True, fignum=1, use_homeostatic=False)
    #room_sizes = [6,10,15,20,30,40,50,75,100]
    #room_sizes_sweep_graph(learning_rate, beta,gamma,room_sizes,steps_per_reversal,N_runs=1,run_afresh=True,line_plot=True,fignum=1)

    # verify_kappa_homeostatic()
