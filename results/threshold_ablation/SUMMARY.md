# Threshold ablation (GPoptval4 protocol, budget 153 evals per arm)

Test F_{1/7} on the held-out split; deltas vs the fixed1 paper arm. `thr` = selected threshold(s), one per level. Missing arms shown as `-`.

## GP arms

| dataset | fixed1 | shared_gp | perlayer_gp |
|---|---|---|---|
| `cssk/cshyphen` | 0.9649 | 0.9650 (+0.0001) | 0.9652 (+0.0004) |
| `thr` selected | 1,1,1,1 | 4 | 5,1,5,1 |
| `es/wiktionary` | 0.9834 | 0.9835 (+0.0000) | 0.9835 (+0.0000) |
| `thr` selected | 1,1,1,1 | 1 | 5,1,4,1 |
| `de/wiktionary` | 0.9895 | 0.9891 (-0.0004) | 0.9899 (+0.0004) |
| `thr` selected | 1,1,1,1 | 1 | 4,1,4,1 |
| `cs/cshyphen_cstenten` | 0.9594 | 0.9593 (-0.0000) | 0.9596 (+0.0002) |
| `thr` selected | 1,1,1,1 | 4 | 4,4,5,1 |
| `de/wortliste` | 0.9815 | 0.9804 (-0.0012) | 0.9819 (+0.0004) |
| `thr` selected | 1,1,1,1 | 5 | 4,4,5,1 |
| `nl/wiktionary` | 0.9838 | 0.9843 (+0.0006) | 0.9847 (+0.0010) |
| `thr` selected | 1,1,1,1 | 3 | 4,5,4,1 |
| `is/hyphenation-is` | 0.9422 | 0.9501 (+0.0079) | 0.9516 (+0.0095) |
| `thr` selected | 1,1,1,1 | 4 | 5,5,5,1 |
| `ru/wiktionary` | 0.9315 | 0.9303 (-0.0012) | 0.9363 (+0.0048) |
| `thr` selected | 1,1,1,1 | 1 | 5,5,4,2 |
| `pl/wiktionary` | 0.9686 | 0.9696 (+0.0010) | 0.9707 (+0.0021) |
| `thr` selected | 1,1,1,1 | 5 | 4,4,4,1 |
| `cs/cshyphen_ujc` | 0.9713 | 0.9719 (+0.0006) | 0.9765 (+0.0051) |
| `thr` selected | 1,1,1,1 | 5 | 5,1,5,1 |
| `it/wiktionary` | 0.9960 | 0.9966 (+0.0006) | 0.9968 (+0.0008) |
| `thr` selected | 1,1,1,1 | 4 | 4,1,5,2 |
| `cs/wiktionary` | 0.9516 | 0.9582 (+0.0065) | 0.9599 (+0.0083) |
| `thr` selected | 1,1,1,1 | 4 | 5,1,5,1 |
| `pt/wiktionary` | 0.9894 | 0.9894 (+0.0000) | 0.9898 (+0.0004) |
| `thr` selected | 1,1,1,1 | 1 | 5,1,5,2 |
| `th/orchid` | 0.9757 | 0.9760 (+0.0003) | 0.9780 (+0.0023) |
| `thr` selected | 1,1,1,1 | 4 | 5,4,4,1 |
| `el/wiktionary` | 0.9110 | 0.9118 (+0.0008) | 0.9130 (+0.0020) |
| `thr` selected | 1,1,1,1 | 5 | 5,1,4,1 |
| `uk/wiktionary` | 0.9455 | 0.9588 (+0.0133) | 0.9635 (+0.0180) |
| `thr` selected | 1,1,1,1 | 4 | 5,1,5,3 |
| `tr/wiktionary` | 0.9904 | 0.9897 (-0.0007) | 0.9907 (+0.0003) |
| `thr` selected | 1,1,1,1 | 1 | 4,2,4,5 |
| `ms/wiktionary` | 0.8012 | 0.8326 (+0.0313) | 0.8291 (+0.0279) |
| `thr` selected | 1,1,1,1 | 5 | 5,3,5,4 |

- shared: mean delta vs fixed1 = +0.0033 over 18 datasets; better on 13
- perlayer: mean delta vs fixed1 = +0.0047 over 18 datasets; better on 18

## TPE arms

| dataset | fixed1 | shared_tpe | perlayer_tpe |
|---|---|---|---|
| `cssk/cshyphen` | 0.9649 | 0.9648 (-0.0000) | 0.9649 (+0.0001) |
| `thr` selected | 1,1,1,1 | 4 | 4,2,1,3 |
| `es/wiktionary` | 0.9834 | 0.9834 (-0.0000) | 0.9834 (-0.0000) |
| `thr` selected | 1,1,1,1 | 3 | 4,4,2,3 |
| `de/wiktionary` | 0.9895 | 0.9896 (+0.0000) | 0.9897 (+0.0002) |
| `thr` selected | 1,1,1,1 | 2 | 5,5,4,2 |
| `cs/cshyphen_cstenten` | 0.9594 | 0.9591 (-0.0003) | 0.9593 (-0.0001) |
| `thr` selected | 1,1,1,1 | 2 | 3,2,3,1 |
| `de/wortliste` | 0.9815 | 0.9808 (-0.0007) | 0.9819 (+0.0004) |
| `thr` selected | 1,1,1,1 | 4 | 2,4,4,2 |
| `nl/wiktionary` | 0.9838 | 0.9835 (-0.0003) | 0.9843 (+0.0005) |
| `thr` selected | 1,1,1,1 | 2 | 5,5,4,1 |
| `is/hyphenation-is` | 0.9422 | 0.9485 (+0.0064) | 0.9517 (+0.0095) |
| `thr` selected | 1,1,1,1 | 4 | 5,5,4,1 |
| `ru/wiktionary` | 0.9315 | 0.9276 (-0.0039) | 0.9323 (+0.0007) |
| `thr` selected | 1,1,1,1 | 4 | 5,3,5,3 |
| `pl/wiktionary` | 0.9686 | 0.9696 (+0.0010) | 0.9698 (+0.0013) |
| `thr` selected | 1,1,1,1 | 5 | 5,5,4,4 |
| `cs/cshyphen_ujc` | 0.9713 | 0.9710 (-0.0003) | 0.9707 (-0.0007) |
| `thr` selected | 1,1,1,1 | 2 | 2,3,3,3 |
| `it/wiktionary` | 0.9960 | 0.9972 (+0.0012) | 0.9969 (+0.0009) |
| `thr` selected | 1,1,1,1 | 1 | 5,2,4,1 |
| `cs/wiktionary` | 0.9516 | 0.9578 (+0.0061) | 0.9604 (+0.0088) |
| `thr` selected | 1,1,1,1 | 5 | 5,2,4,1 |
| `pt/wiktionary` | 0.9894 | 0.9894 (+0.0000) | 0.9895 (+0.0001) |
| `thr` selected | 1,1,1,1 | 3 | 1,3,5,1 |
| `th/orchid` | 0.9757 | 0.9762 (+0.0005) | 0.9777 (+0.0020) |
| `thr` selected | 1,1,1,1 | 4 | 5,4,5,1 |
| `el/wiktionary` | 0.9110 | 0.9116 (+0.0006) | 0.9127 (+0.0017) |
| `thr` selected | 1,1,1,1 | 4 | 5,3,5,1 |
| `uk/wiktionary` | 0.9455 | 0.9557 (+0.0101) | 0.9573 (+0.0118) |
| `thr` selected | 1,1,1,1 | 4 | 5,5,4,1 |
| `tr/wiktionary` | 0.9904 | 0.9897 (-0.0007) | 0.9909 (+0.0005) |
| `thr` selected | 1,1,1,1 | 1 | 4,5,4,2 |
| `ms/wiktionary` | 0.8012 | 0.8326 (+0.0313) | 0.8291 (+0.0279) |
| `thr` selected | 1,1,1,1 | 4 | 4,4,4,3 |

- shared: mean delta vs fixed1 = +0.0028 over 18 datasets; better on 10
- perlayer: mean delta vs fixed1 = +0.0036 over 18 datasets; better on 15

## Random arms

| dataset | fixed1 | shared_random | perlayer_random |
|---|---|---|---|
| `cssk/cshyphen` | 0.9649 | 0.9650 (+0.0001) | 0.9652 (+0.0003) |
| `thr` selected | 1,1,1,1 | 5 | 4,1,5,2 |
| `es/wiktionary` | 0.9834 | 0.9830 (-0.0004) | 0.9834 (-0.0000) |
| `thr` selected | 1,1,1,1 | 3 | 5,4,5,2 |
| `de/wiktionary` | 0.9895 | 0.9891 (-0.0004) | 0.9895 (-0.0001) |
| `thr` selected | 1,1,1,1 | 1 | 4,1,5,2 |
| `cs/cshyphen_cstenten` | 0.9594 | 0.9593 (-0.0001) | 0.9593 (-0.0001) |
| `thr` selected | 1,1,1,1 | 3 | 4,1,5,2 |
| `de/wortliste` | 0.9815 | 0.9800 (-0.0015) | 0.9821 (+0.0006) |
| `thr` selected | 1,1,1,1 | 2 | 2,5,5,1 |
| `nl/wiktionary` | 0.9838 | 0.9836 (-0.0002) | 0.9841 (+0.0004) |
| `thr` selected | 1,1,1,1 | 3 | 5,4,5,2 |
| `is/hyphenation-is` | 0.9422 | 0.9491 (+0.0070) | 0.9514 (+0.0092) |
| `thr` selected | 1,1,1,1 | 4 | 4,1,5,2 |
| `ru/wiktionary` | 0.9315 | 0.9274 (-0.0041) | 0.9316 (+0.0000) |
| `thr` selected | 1,1,1,1 | 4 | 2,5,5,1 |
| `pl/wiktionary` | 0.9686 | 0.9688 (+0.0002) | 0.9705 (+0.0019) |
| `thr` selected | 1,1,1,1 | 4 | 2,5,5,1 |
| `cs/cshyphen_ujc` | 0.9713 | 0.9702 (-0.0011) | 0.9764 (+0.0051) |
| `thr` selected | 1,1,1,1 | 4 | 4,1,5,2 |
| `it/wiktionary` | 0.9960 | 0.9966 (+0.0006) | 0.9968 (+0.0007) |
| `thr` selected | 1,1,1,1 | 4 | 4,1,5,2 |
| `cs/wiktionary` | 0.9516 | 0.9577 (+0.0061) | 0.9597 (+0.0081) |
| `thr` selected | 1,1,1,1 | 5 | 5,3,4,3 |
| `pt/wiktionary` | 0.9894 | 0.9892 (-0.0002) | 0.9896 (+0.0002) |
| `thr` selected | 1,1,1,1 | 3 | 4,1,5,2 |
| `th/orchid` | 0.9757 | 0.9744 (-0.0013) | 0.9775 (+0.0018) |
| `thr` selected | 1,1,1,1 | 4 | 5,4,5,2 |
| `el/wiktionary` | 0.9110 | 0.9115 (+0.0005) | 0.9128 (+0.0018) |
| `thr` selected | 1,1,1,1 | 5 | 5,4,5,2 |
| `uk/wiktionary` | 0.9455 | 0.9547 (+0.0092) | 0.9572 (+0.0117) |
| `thr` selected | 1,1,1,1 | 5 | 5,3,4,3 |
| `tr/wiktionary` | 0.9904 | 0.9895 (-0.0010) | 0.9909 (+0.0005) |
| `thr` selected | 1,1,1,1 | 3 | 5,4,5,2 |
| `ms/wiktionary` | 0.8012 | 0.8326 (+0.0313) | 0.8291 (+0.0279) |
| `thr` selected | 1,1,1,1 | 5 | 5,3,4,3 |

- shared: mean delta vs fixed1 = +0.0025 over 18 datasets; better on 8
- perlayer: mean delta vs fixed1 = +0.0039 over 18 datasets; better on 15

