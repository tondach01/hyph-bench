# HPO comparison: cssk/cshyphen

Budget 150 evals/method (30 rounds x 5), 10-fold CV, objective F17+Trie(w=0.0005).

| Method | F_1/7 | Trie nodes | Params (bad1..4, thr) | Score |
|---|---|---|---|---|
| Hand-tuned (cshyphen) | 0.92924 | 11896 | - | - |
| Hand-tuned (wortliste) | 0.90887 | 16636 | - | - |
| GP (ours) | 0.93163 | 14295 | 18 2 24 30 1 | 0.99999 |
| Random Search | 0.93160 | 12991 | 18 3 30 30 1 | 0.99999 |
| TPE | 0.93153 | 12513 | 18 4 28 25 1 | 0.99999 |

## LaTeX

```latex
\begin{table}[tb]
  \centering
  \caption{Budget-matched HPO comparison on cssk/cshyphen: hand-tuned baseline vs.\ Random Search, TPE, and Gaussian Process, each given the same budget of 150 patgen evaluations. F$_{1/7}$ and trie nodes are 10-fold cross-validation means of each method's winning profile.}
  \label{tab:hpo-baselines-cssk_cshyphen}
  \begin{tabular}{l c c}
    \toprule
    Method & F$_{1/7}$ & Trie nodes \\
    \midrule
    Hand-tuned (cshyphen) & 0.9292 & \textbf{11896} \\
    Hand-tuned (wortliste) & 0.9089 & 16636 \\
    GP (ours) & \textbf{0.9316} & 14295 \\
    Random Search & 0.9316 & 12991 \\
    TPE & 0.9315 & 12513 \\
    \bottomrule
  \end{tabular}
\end{table}
```
