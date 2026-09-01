# HPO comparison: de/wortliste

Budget 150 evals/method (30 rounds x 5), 10-fold CV, objective F17+Trie(w=0.0005).

| Method | F_1/7 | Trie nodes | Params (bad1..4, thr) | Score |
|---|---|---|---|---|
| Hand-tuned (cshyphen) | 0.97752 | 47548 | - | - |
| Hand-tuned (wortliste) | 0.89847 | 43219 | - | - |
| GP (ours) | 0.98038 | 57996 | 19 2 28 2 1 | 0.99978 |
| Random Search | 0.97947 | 54168 | 24 7 19 3 1 | 0.99969 |
| TPE | 0.98115 | 60810 | 28 1 23 2 1 | 0.99979 |

## LaTeX

```latex
\begin{table}[tb]
  \centering
  \caption{Budget-matched HPO comparison on de/wortliste: hand-tuned baseline vs.\ Random Search, TPE, and Gaussian Process, each given the same budget of 150 patgen evaluations. F$_{1/7}$ and trie nodes are 10-fold cross-validation means of each method's winning profile.}
  \label{tab:hpo-baselines-de_wortliste}
  \begin{tabular}{l c c}
    \toprule
    Method & F$_{1/7}$ & Trie nodes \\
    \midrule
    Hand-tuned (cshyphen) & 0.9775 & 47548 \\
    Hand-tuned (wortliste) & 0.8985 & \textbf{43219} \\
    GP (ours) & 0.9804 & 57996 \\
    Random Search & 0.9795 & 54168 \\
    TPE & \textbf{0.9811} & 60810 \\
    \bottomrule
  \end{tabular}
\end{table}
```
