# HPO comparison: ru/wiktionary

Budget 150 evals/method (30 rounds x 5), 10-fold CV, objective F17+Trie(w=0.0005).

| Method | F_1/7 | Trie nodes | Params (bad1..4, thr) | Score |
|---|---|---|---|---|
| Hand-tuned (cshyphen) | 0.89249 | 50959 | - | - |
| Hand-tuned (wortliste) | 0.84108 | 42052 | - | - |
| GP (ours) | 0.91726 | 66321 | 17 1 30 1 1 | 0.99551 |
| Random Search | 0.89099 | 46926 | 10 27 12 1 1 | 0.99442 |
| TPE | 0.91863 | 61949 | 29 5 22 1 1 | 0.99529 |

## LaTeX

```latex
\begin{table}[tb]
  \centering
  \caption{Budget-matched HPO comparison on ru/wiktionary: hand-tuned baseline vs.\ Random Search, TPE, and Gaussian Process, each given the same budget of 150 patgen evaluations. F$_{1/7}$ and trie nodes are 10-fold cross-validation means of each method's winning profile.}
  \label{tab:hpo-baselines-ru_wiktionary}
  \begin{tabular}{l c c}
    \toprule
    Method & F$_{1/7}$ & Trie nodes \\
    \midrule
    Hand-tuned (cshyphen) & 0.8925 & 50959 \\
    Hand-tuned (wortliste) & 0.8411 & \textbf{42052} \\
    GP (ours) & 0.9173 & 66321 \\
    Random Search & 0.8910 & 46926 \\
    TPE & \textbf{0.9186} & 61949 \\
    \bottomrule
  \end{tabular}
\end{table}
```
