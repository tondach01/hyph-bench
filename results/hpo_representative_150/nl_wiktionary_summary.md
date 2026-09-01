# HPO comparison: nl/wiktionary

Budget 150 evals/method (30 rounds x 5), 10-fold CV, objective F17+Trie(w=0.0005).

| Method | F_1/7 | Trie nodes | Params (bad1..4, thr) | Score |
|---|---|---|---|---|
| Hand-tuned (cshyphen) | 0.98111 | 28744 | - | - |
| Hand-tuned (wortliste) | 0.91657 | 26930 | - | - |
| GP (ours) | 0.98378 | 36627 | 18 1 30 1 1 | 0.99984 |
| Random Search | 0.98183 | 30216 | 12 19 20 2 1 | 0.99977 |
| TPE | 0.98182 | 31307 | 13 3 14 1 1 | 0.99980 |

## LaTeX

```latex
\begin{table}[tb]
  \centering
  \caption{Budget-matched HPO comparison on nl/wiktionary: hand-tuned baseline vs.\ Random Search, TPE, and Gaussian Process, each given the same budget of 150 patgen evaluations. F$_{1/7}$ and trie nodes are 10-fold cross-validation means of each method's winning profile.}
  \label{tab:hpo-baselines-nl_wiktionary}
  \begin{tabular}{l c c}
    \toprule
    Method & F$_{1/7}$ & Trie nodes \\
    \midrule
    Hand-tuned (cshyphen) & 0.9811 & 28744 \\
    Hand-tuned (wortliste) & 0.9166 & \textbf{26930} \\
    GP (ours) & \textbf{0.9838} & 36627 \\
    Random Search & 0.9818 & 30216 \\
    TPE & 0.9818 & 31307 \\
    \bottomrule
  \end{tabular}
\end{table}
```
