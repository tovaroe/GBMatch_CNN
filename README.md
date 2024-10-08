# GBMatch_CNN - predicting transcriptional subtypes and risk from glioblastoma whole slide images

# Description
The pipeline implemented here predicts transcriptional subtypes and survival of glioblastoma patients based on H&E stained whole slide scans. Sample data is provided in this repository. To test the basic functionality with 5-fold-CV simply run train_model_OS.py (for survival) or train_model_TS.py (for transcriptional subtypes). Please note that this will not reproduce the results from the manuscript, as only a small fraction of the image data can be provided in this repository due to size constraints. In order to reproduce the results from the manuscript, please refer to the step by step guide below. The whole dataset can be accessed at https://www.medical-epigenomics.org/papers/GBMatch/.
If you wish to adopt this pipeline for your own use, please be sure to set the correct parameters in config.py.

Moreover, we provide a fully trained model in gbm_predictor.py for predicting new samples (supported WSI formats are ndpi and svs). To use GBMPredictor, simply initialize by calling 
`gbm_predictor = GBMPredictor()`
and predict your sample by calling
`(predicted_TS, risk_group, median_riskscore) = gbm_predictor.predict(*path_to_slidescan*)`
Heatmaps and detailed results will be automatically saved in a subfolder in your sample path.

# Dependencies
python 3.7.7

randaugment by Khrystyna Faryna: https://github.com/tovaroe/pathology-he-auto-augment

tensorflow 2.1.0

scikit-survival 0.13.1

pandas 1.0.3

lifelines 0.25.0

# Reproducing the manuscript results - step by step guide

## Training the CNN model
1. Clone this repository and install the dependencies in your environment. Make sure that the path for randaugment is correctly set in the config.py (should be correct by default).
2. Download all included image tiles at https://doi.org/10.5281/zenodo.8358673 and replace the data/training/image_tiles folder with the image_tiles folder from zenodo.
3. Run train_model_OS.py and/or train_model_TS.py to reproduce the training with 5-fold cross validation. Models and results will be saved in the data/models folder.
4. Run train_final_model_OS.py and/or train_final_model_TS.py to train the final model on the whole training dataset.

## Validate the CNN model on TCGA data
1. Download scans and clinical data of the TCGA glioblastoma cohort from https://www.cbioportal.org/ and/or https://portal.gdc.cancer.gov/
2. Copy tumor segmentations from GBMatch_CNN/data/validation/segmentation into the same folder as the TCGA slide scans
3. Predict TCGA samples with gbm_predictor (see above).
(You can also find all prediction results in GBMatch_CNN/data/validation/TCGA_annotation_prediction.csv.)

## Evaluation of the tumor microenvironment
1. Install qupath 0.3.0 (newer versions should also work): https://qupath.github.io/.
2. Download immunohistochemical slides from https://www.medical-epigenomics.org/papers/GBMatch/.
3. Download annotation (IHC_geojsons) from https://doi.org/10.5281/zenodo.8358673.
4. Create a new project and import all immunohistochemical slides & annotations.
5. Copy the CD34 and HLA-DR thresholder from GBMatch_CNN/qupath into your project.
6. Run GBMatch_CNN/qupath/IHC_eval.groovy for all slides - immunohistochemistry results will be saved to a IHC_results-folder.
7. Create a new project and import all HE image tiles.
8. Run GBMatch_CNN/qupath/cellularity.groovy for all slides - cellularity results will be saved to a HE-results-folder.

# Reference
Please consider citing our paper if you find any of the ressources helpful:

Thomas Roetzer-Pejrimovsky, Karl-Heinz Nenning, Barbara Kiesel, Johanna Klughammer, Martin Rajchl, Bernhard Baumann, Georg Langs, Adelheid Woehrer, Deep learning links localized digital pathology phenotypes with transcriptional subtype and patient outcome in glioblastoma, GigaScience, Volume 13, 2024, giae057

https://doi.org/10.1093/gigascience/giae057
