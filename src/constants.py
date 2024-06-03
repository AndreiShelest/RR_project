date_index_label = 'Date'
signal_label = 'Signal'

buy_hold_system = 'b&h'
without_pca_system = 'without_pca'
with_pca_system = 'with_pca'
with_pca_and_dwt_system = 'with_pca_and_dwt'
with_pca_dwt_mooga_system = 'with_pca_dwt_mooga'
system_titles = {
    buy_hold_system: 'Buy & Hold',
    without_pca_system: 'Basic',
    with_pca_system: 'PCA',
    with_pca_and_dwt_system: 'PCA & DWT',
    with_pca_dwt_mooga_system: 'PCA & DWT & MOOGA'
}

equity_label = 'Equity'
equity_curve_label = '_equity_curve'

return_label = "Return"
return_pa_label = "Return (p.a.)"
vol_pa_label = "Vol (p.a.)"
bh_return_label = "B&H Return"
sharpe_label = "Sharpe Ratio"
mdd_label = "Max Drawdown"
accuracy_label = "Accuracy"
transactions_label = "Positions Taken"

strategy_metrics = [
    ("Return [%]", return_label),
    ("Return (Ann.) [%]", return_pa_label),
    ("Volatility (Ann.) [%]", vol_pa_label),
    ("Buy & Hold Return [%]", bh_return_label),
    ("Sharpe Ratio", sharpe_label),
    ("Max. Drawdown [%]", mdd_label),
    ("Win Rate [%]", accuracy_label),
    ("# Trades", transactions_label),
]
strategy_metrics_labels = [label_pair[1] for label_pair in strategy_metrics]
