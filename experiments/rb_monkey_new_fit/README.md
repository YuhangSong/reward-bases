
# base

```bash
rm -r $RESULTS_DIR/rb_monkey_new_fit/ ; \
python main.py -c rb_monkey_new_fit/base
```

```bash
python analysis_v1.py \
-t "base-bic-all_neuron" \
--p "sns.set_theme()" \
-l $RESULTS_DIR/rb_monkey_new_fit/ \
-m "df['bic'].iloc[-1]" \
-f "./experiments/rb_monkey_new_fit/base.yaml" \
-v \
"import experiments.rb_monkey_new_fit.utils as eu" \
"df=eu.proc_df(df, 'bic')" \
"df=au.reduce(df, ['formula'], lambda df: {'sum_bic': df['bic'].sum()})" \
"g=sns.catplot(data=df, kind='bar', y='sum_bic', x='formula')" \
"g.set(ylim=(3380, 3480))" \
"g.set_xticklabels(rotation=90)"
```

![](base-bic-all_neuron-.png)

# base-formula

```bash
rm -r $RESULTS_DIR/rb_monkey_new_fit/ ; \
python main.py -c rb_monkey_new_fit/base-formula
```

```bash
python analysis_v1.py \
-t "base-formula-coeff" \
--p "sns.set_theme()" \
-l $RESULTS_DIR/rb_monkey_new_fit/ \
-m "df['coeff'].iloc[-1]" "df['pvalue'].iloc[-1]" \
-f "./experiments/rb_monkey_new_fit/base-formula.yaml" \
-v \
"import experiments.rb_monkey_new_fit.utils as eu" \
"df=eu.proc_df(df, ['coeff', 'pvalue'])" \
"df=eu.sort_by_id_coeff(df)" \
"g=sns.catplot(data=df, kind='bar', y='coeff', x='neuron', hue='coeff_id')" \
"g.set_xticklabels(rotation=90)"
```

![](base-formula-coeff-.png)

```bash
python analysis_v1.py \
-t "base-formula-pvalue" \
--p "sns.set_theme()" \
-l $RESULTS_DIR/rb_monkey_new_fit/ \
-m "df['coeff'].iloc[-1]" "df['pvalue'].iloc[-1]" \
-f "./experiments/rb_monkey_new_fit/base-formula.yaml" \
-v \
"import experiments.rb_monkey_new_fit.utils as eu" \
"df=eu.proc_df(df, ['coeff', 'pvalue'])" \
"df=eu.sort_by_id_coeff(df)" \
"g=sns.catplot(data=df, kind='bar', y='pvalue', x='neuron', hue='coeff_id')" \
"g.set(ylim=(0, 0.05))" \
"g.set_xticklabels(rotation=90)"
```

![](base-formula-pvalue-.png)

# base-two-regressor

```bash
rm -r $RESULTS_DIR/rb_monkey_new_fit/ ; \
python main.py -c rb_monkey_new_fit/base-two-regressor
```

```bash
python analysis_v1.py \
-t "base-two-regressor" \
--p "sns.set_theme()" \
-l $RESULTS_DIR/rb_monkey_new_fit/ \
-m "df['coeff_banana'].iloc[-1]" "df['coeff_juice'].iloc[-1]" \
-f "./experiments/rb_monkey_new_fit/base-two-regressor.yaml" \
-v \
"import experiments.rb_monkey_new_fit.utils as eu" \
"df=eu.proc_df(df, ['coeff_banana', 'coeff_juice'])" \
"g=sns.relplot(data=df, kind='scatter', y='coeff_banana', x='coeff_juice')" \
"ax = plt.gca()" \
"ax.set_aspect('equal', adjustable='box')"
```

![](base-two-regressor-.png)

# base-two-regressor-compare_coeff

```bash
rm -r $RESULTS_DIR/rb_monkey_new_fit/ ; \
python main.py -c rb_monkey_new_fit/base-two-regressor-compare_coeff
```

```bash
python analysis_v1.py \
-t "base-two-regressor-compare_coeff" \
--p "sns.set_theme()" \
-l $RESULTS_DIR/rb_monkey_new_fit/ \
-m "df['coeff_value'].iloc[-1]" "df['coeff_identity_value'].iloc[-1]" \
-f "./experiments/rb_monkey_new_fit/base-two-regressor-compare_coeff.yaml" \
-v \
"import experiments.rb_monkey_new_fit.utils as eu" \
"df=eu.proc_df(df, ['coeff_value', 'coeff_identity_value'])" \
"g=sns.relplot(data=df, kind='scatter', y='coeff_value', x='coeff_identity_value', style='compare_coeff')"
```

![](base-two-regressor-compare_coeff-.png)

# base-neuron-response

```bash
rm -r $RESULTS_DIR/rb_monkey_new_fit/ ; \
python main.py -c rb_monkey_new_fit/base-neuron-response && \
python analysis_v1.py \
-t "base-neuron-response" \
--p "sns.set_theme()" \
-l $RESULTS_DIR/rb_monkey_new_fit/ \
-m "df['biggest_banana_relative_firing_rate_mean'].iloc[-1]" "df['biggest_juice_relative_firing_rate_mean'].iloc[-1]" "df['biggest_banana_relative_firing_rate_sem_half'].iloc[-1]" "df['biggest_juice_relative_firing_rate_sem_half'].iloc[-1]" \
-f "./experiments/rb_monkey_new_fit/base-neuron-response.yaml" \
-v \
"import experiments.rb_monkey_new_fit.utils as eu" \
"df=eu.proc_df(df, ['biggest_banana_relative_firing_rate_mean', 'biggest_juice_relative_firing_rate_mean', 'biggest_banana_relative_firing_rate_sem_half', 'biggest_juice_relative_firing_rate_sem_half'])" \
"eu.plot_neuron_response(df)"
```

![](base-neuron-response-.png)