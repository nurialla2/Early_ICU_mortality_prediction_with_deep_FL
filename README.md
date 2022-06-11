# Early_ICU_mortality_prediction_with_deep_FL

This repository houses the code used in our thesis, "Early prediction of the risk of ICU mortality with Deep Federated Learning". All code is our own apart from the notebook *mimic_iii_preprocessing.ipynb*, which we took and adapted from Mondrejevski et al. (2022) and is used to preprocess the raw data.

- The *explotaratory_analysis.ipynb* file provides an overview of the data. An exploration of the variables and their distribution, class distribution, and length of stay distribution is provided.
- The *mimic_iii_labeling.ipynb* file provides both window selection and labeling steps.
- The *model.ipynb* file provides the RNN model architecture.
- The *helpers.py* file provides all the classes and functions needed for training and evaluating the model.
- The *results_analysis.ipynb* file provides the processing of the results.

## Scores

The scores subfolder houses the validation scores, test scores, and predictions for the tests we present in the thesis. Each pickle file in there relates to a single test run performed with five-fold cross-validation. The files are named according to the following convention:

  scores\_[*labeling*]\_[*number of clients*]\_[*history length*].pickle

## Dataset

All the tests have been performed using the MIMIC-III (v1.4) dataset (Johnson et al., 2016). Access to this dataset can be gained through [PhysioNet](https://physionet.org/content/mimiciii/1.4/). This repository does not contain any data from the dataset.

## Disclaimer

*This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.*

*This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.*

*You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.*

*All rights to the files *[flicu_icustay_detail.sql](https://github.com/nurialla2/Early_ICU_mortality_prediction_with_deep_FL/blob/main/flicu_icustay_detail.sql)*, *[flicu_pivoted_lab.sql](https://github.com/nurialla2/Early_ICU_mortality_prediction_with_deep_FL/blob/main/flicu_pivoted_lab.sql)*, *[pivoted_lab.sql](https://github.com/nurialla2/Early_ICU_mortality_prediction_with_deep_FL/blob/main/pivoted_lab.sql)*, *[pivoted_vital.sql](https://github.com/nurialla2/Early_ICU_mortality_prediction_with_deep_FL/blob/main/flicu_pivoted_vital.sql)*, and *[postgres-functions.sql](https://github.com/nurialla2/Early_ICU_mortality_prediction_with_deep_FL/blob/main/postgres-functions.sql)* belong to their author Lena Mondrejevsky, who has graciously allowed us to use and publish them. The file *[mimic_iii_preprocessing.ipynb](https://github.com/nurialla2/Early_ICU_mortality_prediction_with_deep_FL/blob/main/mimic_iii_preprocessing.ipynb)* was also authored by Lena Mondrejevsky but further altered by us.*

_____________________________________________________________________________________________
[Mondrejevski, L., Miliou, I., Montanino, A., Pitts, D., Hollmén, J. & Papapetrou, P. (2022),
‘FLICU: A federated learning workflow for intensive care unit mortality prediction’.](https://arxiv.org/abs/2205.15104 "FLICU: A federated learning workflow for intensive care unit mortality prediction")

[Johnson, A. E. W., Pollard, T. J. & Mark, R. G. (2016), ‘Mimic-iii clinical database (version 1.4)’,
PhysioNet.](https://doi.org/10.13026/C2XW26 "Mimic-iii clinical database (version 1.4)")
