# HPO comparison: ms/wiktionary

Budget 150 evals/method (30 rounds x 5), 10-fold CV, objective F17+Trie(w=0.0005).

| Method | F_1/7 | Trie nodes | Params (bad1..4, thr) | Score |
|---|---|---|---|---|
| Hand-tuned (cshyphen) | 0.79416 | 616 | - | - |
| Hand-tuned (wortliste) | 0.74970 | 582 | - | - |
| GP (ours) | 0.75677 | 598 | 4 2 27 30 1 | 0.99968 |
| Random Search | 0.78656 | 611 | 9 2 19 16 1 | 0.99968 |
| TPE | 0.75677 | 598 | 4 2 24 17 1 | 0.99968 |

## LaTeX

```latex
\begin{table}[tb]
  \centering
  \caption{Budget-matched HPO comparison on ms/wiktionary: hand-tuned baseline vs.\ Random Search, TPE, and Gaussian Process, each given the same budget of 150 patgen evaluations. F$_{1/7}$ and trie nodes are 10-fold cross-validation means of each method's winning profile.}
  \label{tab:hpo-baselines-ms_wiktionary}
  \begin{tabular}{l c c}
    \toprule
    Method & F$_{1/7}$ & Trie nodes \\
    \midrule
    Hand-tuned (cshyphen) & \textbf{0.7942} & 616 \\
    Hand-tuned (wortliste) & 0.7497 & \textbf{582} \\
    GP (ours) & 0.7568 & 598 \\
    Random Search & 0.7866 & 611 \\
    TPE & 0.7568 & 598 \\
    \bottomrule
  \end{tabular}
\end{table}
```
