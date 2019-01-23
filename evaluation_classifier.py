from keras import metrics
import tensorflow as tf
import os
import time
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras import backend as K
from sets import Set
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
import pdb
import functools
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt


def multilabel_flow_from_directory(flow_from_directory_gen, multilabel_getter):
    for x, y in flow_from_directory_gen:
        yield x, multilabel_getter(y)

def multilabel_getter(y):
    classes = np.argmax(y, axis=1)
    labels = []
    for cl in classes:
        labels.append(recipe_food_dict[label_map[cl].split("_")[0]])

    labels = np.array(labels)
    mlb = MultiLabelBinarizer(labels_list)
    labels = mlb.fit_transform(labels)
    return labels

def multilabel_scores(y, use_prediction_score):
    classes = np.argmax(y, axis=1)
    labels = []
    for cl in classes:
        labels.append(recipe_food_dict[label_map[cl].split("_")[0]])

    labels = np.array(labels)
    mlb = MultiLabelBinarizer(labels_list)
    labels = mlb.fit_transform(labels)
    if use_prediction_score:
        scores = np.max(y, axis=1)
        labels = labels*scores[:, None]
    return labels 

def build_labels_dict(dataset_path, recipe_food_map_path):
    print("[INFO] loading labels ...")
    recipe_food_map = np.genfromtxt(recipe_food_map_path, delimiter="\t", dtype=str)
    recipe_label = np.genfromtxt(os.path.join(dataset_path, 'label.tsv'), delimiter="_", dtype=str)
    recipe_ids = recipe_label[:, 0].tolist()
    recipe_food_dict = {}
    labels_list = Set([])

    for recipe_food in recipe_food_map:
        if recipe_food[0] in recipe_food_dict:
            if recipe_food[0] in recipe_ids:
                recipe_food_dict[recipe_food[0]].append(recipe_food[2])
                labels_list.add(recipe_food[2])
        else:
            if recipe_food[0] in recipe_ids:
                recipe_food_dict[recipe_food[0]] = [recipe_food[2]]
                labels_list.add(recipe_food[2])

    labels_list = list(labels_list)
    labels_list.sort()
    return recipe_food_dict, labels_list

if __name__ == "__main__":

    MODELS_IMG_DIR = 'models'
    USE_PREDICTION_SCORE = True
    RESULTS_DIR = 'results'
    TYPE_CLASSIFIER = 'multiclass' # accepted values only: ['multiclass', 'multilabel'] 
    DATA_DIR = '/your/local/folder/FFoCat'
    RECIPE_FOOD_MAP = os.path.join(DATA_DIR, 'food_food_category_map.tsv')
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VALID_DIR = os.path.join(DATA_DIR, 'valid')
    IMG_WIDTH, IMG_HEIGHT = 299, 299
    BATCH_SIZE = 512

    if K.image_data_format() == 'channels_first':
        input_shape = (3, IMG_WIDTH, IMG_HEIGHT)
    else:
        input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)

    num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])
    num_valid_steps = num_valid_samples // BATCH_SIZE + 1

    recipe_food_dict, labels_list = build_labels_dict(DATA_DIR, RECIPE_FOOD_MAP)

    # construct the image generator for data augmentation
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(IMG_WIDTH, IMG_HEIGHT), batch_size=BATCH_SIZE, class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory(VALID_DIR, target_size=(IMG_WIDTH, IMG_HEIGHT), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

    label_map = train_generator.class_indices
    label_map = dict((v, k) for k, v in label_map.items())
    multilabel_validation_generator = multilabel_flow_from_directory(validation_generator, multilabel_getter)

    # Evaluate the network
    print("[INFO] loading models ...")
    model_img_class = load_model(os.path.join(MODELS_IMG_DIR, 'inceptionv3_' + TYPE_CLASSIFIER + '_best.h5'))

    y_true_stack = np.empty((0, len(labels_list)))
    y_pred_multi_class_stack = np.empty((0, len(labels_list)))
    cnt = 0
    print("[INFO] evaluating network ...")
    while cnt < num_valid_steps:
        start = time.time()

        #for batch_test in multilabel_validation_generator:
        batch_test = next(multilabel_validation_generator)

        cnt += 1
        y_true = batch_test[1]
        x_true = batch_test[0]
        y_pred_img_class = model_img_class.predict(x_true)
        
        if TYPE_CLASSIFIER is "multiclass":
            y_pred_multi_class = multilabel_scores(y_pred_img_class, USE_PREDICTION_SCORE)
        else:
            y_pred_multi_class = y_pred_img_class

        y_true_stack = np.vstack((y_true_stack, y_true))
        y_pred_multi_class_stack = np.vstack((y_pred_multi_class_stack, y_pred_multi_class))
        end = time.time()
        print "Time for batch {}/{}: {:.2f} secs".format(cnt, num_valid_steps, end - start)

    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(labels_list)):
        precision[i], recall[i], _ = precision_recall_curve(y_true_stack[:, i], y_pred_multi_class_stack[:, i])
        average_precision[i] = average_precision_score(y_true_stack[:, i], y_pred_multi_class_stack[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_stack.ravel(), y_pred_multi_class_stack.ravel())
    average_precision["micro"] = average_precision_score(y_true_stack, y_pred_multi_class_stack, average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(100*average_precision["micro"]))

    plt.figure()
    np.savetxt('recall_' + TYPE_CLASSIFIER + '.csv', recall['micro'])
    np.savetxt('precision_' + TYPE_CLASSIFIER + '.csv', precision['micro'])
    plt.plot(recall['micro'], precision['micro'], color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(100*average_precision["micro"]))
    plt.savefig(os.path.join(RESULTS_DIR, 'AP_' + TYPE_CLASSIFIER + '.png'))
