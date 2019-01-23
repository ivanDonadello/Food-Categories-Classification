import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import applications
from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from keras.utils import multi_gpu_model
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
import pdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sets import Set
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import keras.backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.49
set_session(tf.Session(config=config))


def show_acc_history(history):
    plt.clf()
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    pdb.set_trace()
    if TYPE_CLASSIFIER is 'multiclass':
        plt.plot(history.history['categorical_accuracy'])
        plt.plot(history.history['val_categorical_accuracy'])
    else:
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
    plt.legend(['train_accuracy', 'test_accuracy'], loc='best')
    plt.savefig(os.path.join(HISTORY_DIR, 'acc_inceptionv3_' + TYPE_CLASSIFIER + '.png'))


def show_loss_history(history):
    plt.clf()
    pdb.set_trace()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'test_loss'], loc='best')
    plt.savefig(os.path.join(HISTORY_DIR, 'loss_inceptionv3_' + TYPE_CLASSIFIER + '.png'))


def multilabel_flow_from_directory(flow_from_directory_gen):

    while True: #keras needs infinite generators
        x, y = next(flow_from_directory_gen)
        classes = np.argmax(y, axis=1)
        labels = []
        for cl in classes:
            labels.append(recipe_food_dict[label_map[cl].split("_")[0]])

        labels = np.array(labels)
        mlb = MultiLabelBinarizer(labels_list)
        labels = mlb.fit_transform(labels)
        yield x, labels


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


if __name__=="__main__":

    HISTORY_DIR = 'history'
    MODELS_DIR = 'models'
    DATA_DIR = '/your/local/folder/FFoCat'
    RECIPE_FOOD_MAP = os.path.join(DATA_DIR, 'food_food_category_map.tsv')
    TYPE_CLASSIFIER = 'multilabel' # accepted values only: ['multiclass', 'multilabel'] 
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VALID_DIR = os.path.join(DATA_DIR, 'valid')
    BATCH_SIZE = 16
    EPOCHS = 100
    INIT_LR = 1e-6
    IMG_WIDTH, IMG_HEIGHT = 299, 299  # dimensions of our images

    if K.image_data_format() == 'channels_first':
        input_shape = (3, IMG_WIDTH, IMG_HEIGHT)
    else:
        input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)

    num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
    num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])

    num_train_steps = num_train_samples // BATCH_SIZE + 1
    num_valid_steps = num_valid_samples // BATCH_SIZE + 1

    recipe_food_dict, labels_list = build_labels_dict(DATA_DIR, RECIPE_FOOD_MAP)
    print "Number of labels {}".format(len(labels_list))

    # construct the image generator for data augmentation
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=25,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode="nearest")
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(IMG_WIDTH, IMG_HEIGHT), batch_size=BATCH_SIZE, class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory(VALID_DIR, target_size=(IMG_WIDTH, IMG_HEIGHT), batch_size=BATCH_SIZE, class_mode='categorical')

    label_map = (train_generator.class_indices)
    label_map = dict((v, k) for k, v in label_map.items())
    if TYPE_CLASSIFIER is 'multilabel':
        multilabel_train_generator = multilabel_flow_from_directory(train_generator)
        multilabel_validation_generator = multilabel_flow_from_directory(validation_generator)

    # create the base pre-trained model
    base_model = applications.inception_v3.InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # let's add a fully-connected layer
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)

    # and a logistic layer
    if TYPE_CLASSIFIER is 'multiclass':
        predictions = Dense(train_generator.num_classes, activation='softmax')(x)
    else:
        predictions = Dense(len(labels_list), activation='sigmoid')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # compile the model using binary cross-entropy rather than categorical cross-entropy -- this may seem counterintuitive for
    # multi-label classification, but keep in mind that the goal here is to treat each output label as an independent Bernoulli distribution
    if TYPE_CLASSIFIER is 'multiclass':
        model.compile(optimizer=optimizers.Adam(lr=INIT_LR), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    else:
        model.compile(optimizer=optimizers.Adam(lr=INIT_LR), loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(patience=15)

    checkpointer = ModelCheckpoint(os.path.join(MODELS_DIR, 'inceptionv3_' + TYPE_CLASSIFIER + '_best.h5'), verbose=1, save_best_only=True)

    # train the network
    print("[INFO] training network...")

    if TYPE_CLASSIFIER is 'multiclass':
        history = model.fit_generator(train_generator, steps_per_epoch=num_train_steps, epochs=EPOCHS, verbose=1, callbacks=[early_stopping, checkpointer], validation_data=validation_generator, validation_steps=num_valid_steps, workers=12, use_multiprocessing=True)
    else:
        history = model.fit_generator(multilabel_train_generator, steps_per_epoch=num_train_steps, epochs=EPOCHS, verbose=1, callbacks=[early_stopping, checkpointer], validation_data=multilabel_validation_generator, validation_steps=num_valid_steps, workers=12, use_multiprocessing=True)
    model.save(os.path.join(MODELS_DIR, 'inceptionv3_' + TYPE_CLASSIFIER + '_final.h5'))
    show_acc_history(history)
    show_loss_history(history)
