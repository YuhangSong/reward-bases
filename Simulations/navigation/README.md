# README

**Run the experiment inside this subfolder.**

Install Python and make sure that the `python` command can be ran from your command line.
Install required python packages with:

```bash
pip install seaborn pandas matplotlib numpy scipy
```

Then run

```bash
python room_task_Simulations.py
```

to produce:

| Column 1                                  | Column 2                                   | Column 3                                   | Column 4                                   | Column 5                                      |
| ----------------------------------------- | ------------------------------------------ | ------------------------------------------ | ------------------------------------------ | --------------------------------------------- |
| ![](figures/room_rcombined.png)           | ![](figures/room_r1.png)                   | ![](figures/room_r2.png)                   | ![](figures/room_r3.png)                   |                                               |
| ![](figures/TD_room_vf.png)               | ![](figures/RB_room_1vf.png)               | ![](figures/RB_room_2vf.png)               | ![](figures/RB_room_3vf.png)               | ![](figures/RB_room_comb_v.png)               |
| ![](figures/TD_room_vf_other_example.png) | ![](figures/RB_room_1vf_other_example.png) | ![](figures/RB_room_2vf_other_example.png) | ![](figures/RB_room_3vf_other_example.png) | ![](figures/RB_room_comb_v_other_example.png) |

| Column 1                                        | Column 2                                       | Column 3                                 |
| ----------------------------------------------- | ---------------------------------------------- | ---------------------------------------- |
| ![](figures/interval_steps_maze_combined_1.png) | ![](figures/learning_rate_maze_combined_6.png) | ![](figures/room_task_quadruple_0.1.png) |

## Further information

The file `envs.py` contains the environment classes for the various Simulations and the `learners.py` file contains the classes for the reward basis, temporal difference, and successor representation agent.
