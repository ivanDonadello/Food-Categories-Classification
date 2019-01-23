# Food Categories Classification

This repository contains the dataset and the source code for the classification of food categories from meal images.

## The Food and Food Categories (FFoCat) Dataset

[Here](http://bit.do/eGcW5) you can download the `FFoCat.zip` file, unzip it in your local machine. The dataset is already divided into the `train` and `test` folder. The file `label.tsv` contains the food labels, the file `food_food_category_map.tsv` contains the food labels with the corresponding food category labels.

## Using the Source Code

- The `models` folder will contain the multiclass and multilabel models after the training;
- The `history` folder will contain the accuracies and losses of the multiclass and multilabel models after the training;
- The `results` folder will contain the precision-recall curves results for the trained models after the evaluation.

**Requirements**

We train and test the models with the following software configuration. However, more recent versions of the libraries could also work:

- Ubuntu 14.04;
- Python 2.7.6;
- Keras 2.1.3;
- TensorFlow 1.4.0;
- Numpy 1.13.1;
- Scikit-learn 0.18.1;
- Matplotlib 1.5.1;

**Training a model**

Before training a model set in `food_category_classification.py` the following variables:

- `TYPE_CLASSIFIER` if you want to train the multiclass or multilabel model. It takes the values `multiclass` or `multilabel`;
- `DATA_DIR` that is your local path to the `FFoCat` folder.

To run a train use the following command
```
python food_category_classification.py
```
Models will be saved in the `models` folder.

**Evaluating a model**

Before evaluating a model set in `evaluation_classifier.py` the following variables:

- `TYPE_CLASSIFIER` if you want to test the multiclass or multilabel model. It takes the values `multiclass` or `multilabel`;
- `DATA_DIR` as above;
- `USE_PREDICTION_SCORE` useful only for the multiclass classification. If you want to use the classification score for the inferred food category labels set it to `True`, `False` otherwise.

To run the evaluation use the following command
```
python evaluation_classifier.py
```
Results will be saved in the `results` folder.
