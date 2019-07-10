# Food Categories Classification

This repository contains the dataset and the source code for the classification of food categories from meal images.

## The Food and Food Categories (FFoCat) Dataset

[Here](https://drive.google.com/drive/folders/1ZWuq5x9qPbzMBXPTPaKl3P9aSNLL3q0d?usp=sharing) you can download the `FFoCat.zip` file, unzip it in your local machine. The dataset is already divided into the `train` and `test` folder. The file `label.tsv` contains the food labels, the file `food_food_category_map.tsv` contains the food labels with the corresponding food category labels. The following table compares the FFoCat dataset with previous datasets for food recognition. The Type column refers to the number of labels in each image. Single (S) means only one label per image, Multi (M) means many labels per image.

| Name           | Year | #Images  | #Food Classes | Type |          Reference         |
|----------------|:----:|:--------:|:-------------:|:----:|:--------------------------:|
| Food50         | 2009 |   5,000  |       50      |   S  |    1    |
| PFID           | 2009 |   1,098  |       61      |   S  |      2     |
| TADA           | 2009 |  50/256  |       -       |  S/M | 3 |
| Food85         | 2010 |   8,500  |       85      |   S  |    4    |
| Chen           | 2012 |   5,000  |       50      |   S  |  5  |
| UEC FOOD-100   | 2012 |   9,060  |      100      |  S/M |  6 |
| Food-101       | 2014 |  101,000 |      101      |   S  |   7  |
| UEC FOOD-256   | 2014 |  31,397  |      256      |  S/M |   8   |
| UNICT-FD889    | 2014 |   3,583  |      889      |   S  |   9  |
| Diabetes       | 2014 |   4,868  |       11      |   S  |    10    |
| VIREO Food-172 | 2016 |  110,241 |      172      |  S/M |    11    |
| UNIMIB2016     | 2016 |   1,027  |       73      |   M  |     12    |
| Food524DB      | 2017 |  247,636 |      524      |   S  |   13  |
| FFoCat         | 2018 |  58,962  |      156      |  S/M |              -             |

1. Taichi Joutou, Keiji Yanai: A food image recognition system with Multiple Kernel Learning. ICIP 2009: 285-288;
2. Mei Chen, Kapil Dhingra, Wen Wu, Lei Yang, Rahul Sukthankar, Jie Yang: PFID: Pittsburgh fast-food image dataset. ICIP 2009: 289-292;
3. Anand Mariappan, Marc Bosch, Fengqing Zhu, Carol J. Boushey, Deborah A. Kerr, David S. Ebert, Edward J. Delp: Personal dietary assessment using mobile devices. Computational Imaging 2009: 72460;
4. Hajime Hoashi, Taichi Joutou, Keiji Yanai: Image Recognition of 85 Food Categories by Feature Fusion. ISM 2010: 296-30;
5. Chen Mei-Yun, Yung-Hsiang Yang, Chia-Ju Ho, Shih-Han Wang, Shane-Ming Liu, Eugene Chang, Che-Hua Yeh, Ming Ouhyoung: Automatic chinese food identification and quantity estimation. SIGGRAPH Asia 2012: 29;
6. Yoshiyuki Kawano, Keiji Yanai: Real-Time Mobile Food Recognition System. CVPR Workshops 2013: 1-7
;
7. Lukas Bossard, Matthieu Guillaumin, Luc J. Van Gool: Food-101 - Mining Discriminative Components with Random Forests. ECCV (6) 2014: 446-461;
8. Yoshiyuki Kawano, Keiji Yanai: FoodCam-256: A Large-scale Real-time Mobile Food RecognitionSystem employing High-Dimensional Features and Compression of Classifier Weights. ACM Multimedia 2014: 761-762;
9. Giovanni Maria Farinella, Dario Allegra, Filippo Stanco: A Benchmark Dataset to Study the Representation of Food Images. ECCV Workshops (3) 2014: 584-599;
10. Joachim Dehais, Marios Anthimopoulos, Sergey Shevchik, Stavroula G. Mougiakakou: Two-View 3D Reconstruction for Food Volume Estimation. IEEE Trans. Multimedia 19(5): 1090-1099 (2017);
11. Jingjing Chen, Chong-Wah Ngo: Deep-based Ingredient Recognition for Cooking Recipe Retrieval. ACM Multimedia 2016: 32-41;
12. Gianluigi Ciocca, Paolo Napoletano, Raimondo Schettini: Food Recognition: A New Dataset, Experiments, and Results. IEEE J. Biomedical and Health Informatics 21(3): 588-598 (2017);
13. Gianluigi Ciocca, Paolo Napoletano, Raimondo Schettini: Learning CNN-based Features for Retrieval of Food Images. ICIAP Workshops 2017: 426-434;

Every food label has (on average) almost 389 example pictures and every food categories has (on average) approximately 4915 examples. The huge amount of data for the multilabel classification makes the problem easier with respect to a food recognition (and food categories inference) setting. However, this is counterbalanced with the distribution of the labels in the dataset Fig.~\ref{fig:label_distr}. For the food recognition task the dataset is quite balanced (Fig.~\ref{fig:food_distr}), whereas the food category recognition the dataset is unbalanced and presents the so-called \emph{long-tail problem}: many labels with few examples
![alt text](https://drive.google.com/open?id=18lX_pzEN1GbH91KfR0PbUVxVyit7jTto)
![alt text](https://drive.google.com/file/d/1JXvCpZFFtCeYGFQ0U1X6z0k58Cc_hnxk/view?usp=sharing)


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
