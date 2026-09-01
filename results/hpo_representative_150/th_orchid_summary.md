# HPO comparison: th/orchid

Budget 150 evals/method (30 rounds x 5), 10-fold CV, objective F17+Trie(w=0.0005).

| Method | F_1/7 | Trie nodes | Params (bad1..4, thr) | Score |
|---|---|---|---|---|
| Hand-tuned (cshyphen) | 0.96561 | 28587 | - | - |
| Hand-tuned (wortliste) | 0.95239 | 24010 | - | - |
| GP (ours) | 0.97301 | 32666 | 30 7 29 1 1 | 0.99765 |
| Random Search | 0.97297 | 30091 | 22 7 5 1 1 | 0.99705 |
| TPE | 0.97277 | 32465 | 29 5 25 1 1 | 0.99762 |

## LaTeX

```latex
\begin{table}[tb]
  \centering
  \caption{Budget-matched HPO comparison on th/orchid: hand-tuned baseline vs.\ Random Search, TPE, and Gaussian Process, each given the same budget of 150 patgen evaluations. F$_{1/7}$ and trie nodes are 10-fold cross-validation means of each method's winning profile.}
  \label{tab:hpo-baselines-th_orchid}
  \begin{tabular}{l c c}
    \toprule
    Method & F$_{1/7}$ & Trie nodes \\
    \midrule
    Hand-tuned (cshyphen) & 0.9656 & 28587 \\
    Hand-tuned (wortliste) & 0.9524 & \textbf{24010} \\
    GP (ours) & \textbf{0.9730} & 32666 \\
    Random Search & 0.9730 & 30091 \\
    TPE & 0.9728 & 32465 \\
    \bottomrule
  \end{tabular}
\end{table}
```
