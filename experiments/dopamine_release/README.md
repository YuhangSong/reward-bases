- [panel\_b](#panel_b)
- [panel\_c](#panel_c)
- [panel\_bc](#panel_bc)

Data extrained from [here](https://drive.google.com/drive/folders/1J4yQ3XjXebkNY2GNYrFttIfAT5ODuVJb?usp=sharing) with [this software](https://apps.automeris.io/wpd/).

# panel_b

```bash
python main.py -c rb_papageorgiou/panel_b
```

```bash
python analysis_v1.py \
-t "panel_b" \
-l $RESULTS_DIR/rb_papageorgiou/ \
-m "eval(df['value-along-index'].iloc[-1])" \
-f "./experiments/rb_papageorgiou/panel_b.yaml" \
-v \
"import experiments.rb_papageorgiou.utils as eu" \
"eu.panel_b_fit_data_and_plot(df)"
```

![](panel_b-.png)

# panel_c

```bash
python main.py -c rb_papageorgiou/panel_c
```

```bash
python analysis_v1.py \
-t "panel_c" \
-l $RESULTS_DIR/rb_papageorgiou/ \
-m "eval(df['value-along-index'].iloc[-1])" \
-f "./experiments/rb_papageorgiou/panel_c.yaml" \
-v \
"import experiments.rb_papageorgiou.utils as eu" \
"eu.panel_c_fit_data_and_plot(df)"
```

![](panel_c-.png)

# panel_bc

Try to use the same set of parameters for fitting both panels.

```bash
python main.py -c rb_papageorgiou/panel_bc
```

```bash
python analysis_v1.py \
-t "panel_bc" \
-l $RESULTS_DIR/rb_papageorgiou/ \
-m "eval(df['value-along-index'].iloc[-1])" \
-f "./experiments/rb_papageorgiou/panel_bc.yaml" \
-v \
"import experiments.rb_papageorgiou.utils as eu" \
"eu.panel_bc_fit_data_and_plot(df)"
```

![](panel_bc-.png)
