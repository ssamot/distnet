# distnet
Distance Networks, an memory network like learning scheme
Results Comparison Thus far

Task Number/Name                LSTM Baseline    PE LS RN JOINT (3-hops, fb)  BoW (1-hop,fb)        Weak Supervision 1K Memory
----------------------------  ---------------  ----------------                ------------                        ----------------------------
QA1 - Single Supporting Fact            0.5             1               0.992                         1
QA2 - Two Supporting Facts              0.2             0.886           0.38                          0.847
QA3 - Three Supporting Facts            0.2             0.781           0.231                         0.625
QA4 - Two Arg. Relations                0.61            0.866           0.772                         0.877
QA5 - Three Arg. Relations              0.7             0.856           0.89                          0.825
QA6 - Yes/No Questions                  0.48            0.972           0.928                         0.982
QA7 - Counting                          0.49            0.817           0.841                         0.877
QA8 - Lists/Sets                        0.45            0.907           0.868                         0.949
QA9 - Simple Negation                   0.64            0.981           0.949                         0.997
QA10 - Indefinite Knowledge             0.44            0.935           0.894                         0.982
QA11 - Basic Coreference                0.62            0.997           0.916                         0.919
QA12 - Conjunction                      0.74            0.999           0.996                         1
QA13 - Compound Coreference             0.94            0.998           0.937                         0.943
QA14 - Time Reasoning                   0.27            0.931           0.631                         0.964
QA15 - Basic Deduction                  0.21            1               0.536                         1
QA16 - Basic Induction                  0.23            0.973           0.526                         0.459
QA17 - Positional Reasoning             0.51            0.596           0.556                         0.534
QA18 - Size Reasoning                   0.52            0.906           0.904                         0.889
QA19 - Path Finding                     0.08            0.12            0.093                         0.131
QA20 - Agent's Motivations              0.91            1               1                             1
Overall Mean                            0.487           0.87605         0.742                         0.84

