# Food Categories Classification

This repository contains the dataset and the source code for the classification of food categories from meal images.

## The FFoCat Dataset

Click [here](http://bit.do/eGcW5), download the `FFoCat.zip` file and just unzip it.

## Using the Source Code

- The `models` folder will contain the multiclass and multilabel models after the training;
- The `history` folder will contain the accuracies and losses of the multiclass and multilabel models after the training;
- The `results` folder will contain the precision-recall curves results for the trained models after the evaluation.

**Requirements**

We train and test the models with the following software configuration but more recent versions of the libraries work:

- Ubuntu 14.04;
- Python 2.7.6;
- Keras 2.1.3;
- TensorFlow 1.4.0;
- Numpy 1.13.1;
- Sklearn 0.18.1;
- Matplotlib 1.5.1;

**Train a model**

```
python food_category_classification.py
```
Models will be saved in the `models` folder.

**Evaluate a model**

```
python evaluation_classifier.py
```
Results will be saved in the `results` folder.
