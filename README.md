# Extending and Evaluating a Rank Similarity Measure<br>(CSE3000 Research Project, EEMCS, TU Delft)

## Purpose of the Study
This repository contains the codebase and the aggregated results for the Bachelor thesis of Konstantin-Asen Yordanov, supervised by Matteo Corsi and Julián Urbano. The title of this work is **Redefining the Single RBO Score to Achieve a More Accurate Similarity Estimate _(Alternatives to the Assumption of Constant Agreement)_**. The study aims to reformulate the RBO point estimate (RBO<sub>EXT</sub>) under more relaxed assumptions in an effort to increase the accuracy of the extrapolated similarity score. The closeness of the proposed RBO<sub>EXT</sub> redefinitions to the real RBO score is measured and compared against that of the original implementation. This way, the work seeks to determine whether redefining the RBO point estimate merits further research.

Central to this work is the interpretation of agreement as the probability that an element selected at random appears in both rankings **_S_** and **_L_**. Starting at depth **_(s + 1)_** and continuing up to infinity, this **membership degree (or probability)** is estimated for items ranked in the unseen parts of the two top-weighted lists, and it is used to compute the **assumed agreement** at those depths. For more information regarding RBO and its properties, please refer to the works of Webber et al. [1], Moffat and Zobel [2], or Corsi and Urbano [3].

Three approaches to redefining RBO<sub>EXT</sub> have been proposed:
- **Previous-Value Approach**: the estimated membership probability at depth **_d_** is taken as the assumed agreement at the previous depth;
- **Logistic-Regression Approach**: the estimated membership probability at depth **_d_** is taken as the output of a logistic-regression model based on a linear combination of the depth;
- **Logistic-GAM Approach**: the estimated membership probability at depth **_d_** is taken as the output of the more flexible generalized additive model (GAM) that uses the penalized cubic spline as a smoothing function before applying the sigmoid on the computed value (more details on GAMs and their applications can be found in the work of Hastie and Tibshirani [4]).

## Structure of the Codebase
Listed below are the main files and directories relevant for assessing and reproducing the results of the study:
- `data` includes two TXT files containing 5000 newline-separated pairs of simulated rankings each (in one file, the length of a full ranking is 1000 elements, and in the other, it is 2000). The results discussed in the research paper are produced from the evaluation of the rankings in the `data_5000pairs_2000length.txt` file. All pairs of rankings were generated using [the simulation code](https://github.com/julian-urbano/sigir2024-rbo) written by the authors of [3].
- `src/rbo_redefinition.py` contains the Python code for all formulations of RBO<sub>EXT</sub> (original, previous-value, logistic-regression, and logistic-GAM).
- `src/run.py` includes the full pipeline for running testing configurations, plotting figures, and storing the results. It includes several tunable hyperparameters (described in the next section of the README), such as the values of **_p_** to be tested for and the seed for the pseudorandom-number generation of **_s_** and **_l_**. All results are stored as JSON files, and the generated figures are persisted in PNG format.
- `results` contains the JSON files with the aggregated accuracy-scores of all four RBO<sub>EXT</sub> implementations for each of the specified values of **_p_**. The contents of this folder are the findings discussed in the research paper, using the simulated pairs of rankings from the `data_5000pairs_2000length.txt` file and three values for **_p_** (0.8, 0.9, 0.95). The JSON files also include the best- and worst-performing instances for each RBO<sub>EXT</sub> formulation in terms of RBO-accuracy and agreement-accuracy. For improved readability and a more fine-grained evaluation, the results are split into four categories: overall (averaged across all 5000 pairs of rankings), small **_s_** (averaged over those instances where the randomly-generated value for **_s_** was less than or equal to 15), medium **_s_** (in the range 15 to 45), and large **_s_** (greater than 45).
- `figures` contains PNG files of (1) agreement (actual and assumed) plotted against depth for all four RBO<sub>EXT</sub> formulations, as well as (2) the training performance (observed vs. fitted agreement for depths 1 through **_s_**) of the two regression-based approaches (logistic-regression and logistic-GAM). These figures were generated from the best- and worst-performing scenarios in terms of agreement-accuracy, which are persisted as JSON objects in the `results` directory.
- `results_5000pairs_1000length` is similar to `results` except for the data that were used (the `data_5000pairs_1000length.txt` file).
- `figures_5000pairs_1000length` is similar to `figures` except for the data that were used (the `data_5000pairs_1000length.txt` file).

## Running the Code
Running the testing configurations to generate results and figures is done via the `main` method of the `src/run.py` file. The code was written in Python 3.11, and there are several required modules that can be installed by running the following command:

```
pip install -r requirements.txt
```

Listed below are all of the tunable hyperparameters in the codebase:
- `random.seed` as the seed for the pseudorandom-number generation of **_s_** and **_l_** (default of 42)
- `p_values` as the list of values to be tested for (default of 0.8, 0.9, and 0.95)
- `data_file` as the path to the file containing the pairs of simulated rankings (default of `../data/data_5000pairs_2000length.txt`)
- `l_upper_threshold` as the upper bound for the random-number generation of **_l_** (default of 100)
- `s_medium_threshold` as the value beyond which **_s_** is considered medium, the third category in the JSON files containing the results (default of 15)
- `s_large_threshold` as the value beyond which **_s_** is considered large, the fourth category in the JSON files containing the results (default of 45)
- `plotting_depth` as the value at which to truncate the x-axis when plotting the actual and assumed agreement-values against depth (default of 140)

## References
- [1] William Webber, Alistair Moffat, and Justin Zobel. A Similarity Measure for Indefinite Rankings. _ACM Trans. Inf. Syst._, 28(4), 11 2010.
- [2] Alistair Moffat and Justin Zobel. Rank-Biased Precision for Measurement of Retrieval Effectiveness. _ACM Trans. Inf. Syst._, 27(1), 12 2008.
- [3] Matteo Corsi and Julián Urbano. The Treatment of Ties in Rank-Biased Overlap. In _International ACM SIGIR Conference on Research and Development in Information Retrieval_, 2024.
- [4] Trevor Hastie and Robert Tibshirani. Generalized Additive Models. _Statistical Science_, 1(3):297-318, 1986.