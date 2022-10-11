# GBMatch_CNN
Predicting TS &amp; risk from glioblastoma whole slide images

# Reference
Upcoming paper: stay tuned...

# Dependencies
randaugment by Khrystyna Faryna: https://github.com/DIAGNijmegen/pathology-he-auto-augment

tensorflow 2.1.0

scikit-survival 0.13.1

pandas 1.0.3

lifelines 0.25.0

scipy 1.4.1

WSIHandler (https://github.com/tovaroe/WSIHandler) for usage of gbm_predictor.py

# Description
The pipeline implemented here predicts transcriptional subtypes and survival of glioblastoma patients based on H&E stained whole slide scans. Sample data is provided in this repository. To test the basic functionality with 5-fold-CV simply run train_model_OS.py (for survival) or train_model_TS.py (for transcriptional subtypes). Please note that this will not reproduce the results from the manuscript, as only a small fraction of the image data can be provided in this repository due to size constraints.
If you wish to adopt this pipeline for your own use, please be sure to set the correct parameters in config.py.

Moreover, we provide a fully trained model in gbm_predictor.py for predicting new samples.
