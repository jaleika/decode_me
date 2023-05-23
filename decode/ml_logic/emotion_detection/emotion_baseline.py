from tensorflow.keras.utils import image_dataset_from_directory
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from joblib import dump, load
from decode.params import *
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import models, Sequential, layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# assign batch size, used for reading in the dataset
BATCH_SIZE = 32
DIR_TRAIN = f"{LOCAL_MODELS_DATA_PATH}/raw_data/fer_2013/train"
DIR_TEST = f"{LOCAL_MODELS_DATA_PATH}/raw_data/fer_2013/test"
DIR_MODELS = f"{LOCAL_MODELS_DATA_PATH}/models"


# load the data into a dataset
def load_data_local(
    dir, batch_size=BATCH_SIZE, image_size=(95, 95), color_mode="grayscale"
):
    ds = image_dataset_from_directory(
        dir,
        shuffle=True,
        seed=42,
        image_size=image_size,
        color_mode=color_mode,
        batch_size=batch_size,
    )
    return ds
    # train_ds, val_ds = image_dataset_from_directory(, shuffle=True, seed = 42, subset = "both", image_size=(95,95), color_mode = "greyscale", validation_split = 0.3, batch_size = BATCH_SIZE)


# get the first num_batches from the dataset ds, returns image batches and labels
def get_first_batches(ds, num_batches):
    counter = 0
    for images, labels in ds:
        if counter < 1:
            concat_batches = images
            concat_labels = labels
        elif counter < num_batches:
            concat_batches = layers.Concatenate(axis=0)([concat_batches, images])
            concat_labels = layers.Concatenate(axis=0)([concat_labels, labels])
        counter += 1
    return concat_batches, concat_labels


# get the number of images in dataset
def get_num_images(ds, batch_size=BATCH_SIZE):
    num_images = ds.cardinality() * batch_size
    classes = -1 * np.ones(num_images)
    for image, labels in ds:
        classes[
            idx * batch_size : idx * batch_size + len(labels.numpy())
        ] = labels.numpy()
        idx += 1
    classes = classes[classes != -1]
    return len(classes)



def get_class_distribution(classes):
    """Get the class distribution: Returns the frequency of classes.

    Parameters
    ----------
    classes : np.array

    Returns
    -------
    np.array
        Frequency per class
    """
    return np.unique(classes, return_counts=True)[1] / len(classes)


def get_baseline_model(
    train_ds, test_ds, num_batches_train=40, num_batches_test=20, dummy=True
):
    """Get a baseline model

    Parameters
    ----------
    train_ds : _type_
        _description_
    test_ds : _type_
        _description_
    num_batches_train : int, optional
        _description_, by default 40
    num_batches_test : int, optional
        _description_, by default 20
    dummy : bool, optional
        _description_, by default True

    Returns
    -------
    _type_
        _description_
    """    """"""

    X_train, y_train = get_first_batches(train_ds, 40)
    y_cat_train = to_categorical(y_train, num_classes=7)
    X_test, y_test = get_first_batches(test_ds, 20)
    y_cat_test = to_categorical(y_test, num_classes=7)
    if dummy:
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X_train, y_train)
        dummy_score_train = model.score(X_train, y_train)
        dummy_score_test = model.score(model.predict(X_test), y_test)
        print(
            f"The dummy model has an accuracy of {round(dummy_score_train*100,1)}% on the train set and {round(dummy_score_test*100,1)}% on the test set."
        )
        return model
    return None



def pipeline_baseline(dummy=True):
    """"""
    """Call this function to load the train and test dataset and return a baseline model.
    If you want to call the model for predicting a single image make sure your image is in a 4-d tensor. If not call
    model.predict(np.expand_dims(bs[0],0))

    Args:
        dummy (bool, optional): If True, returns a dummy model predicting the most probable class in train set.

    Returns:
        _type_: sklearn Model
    """
    train_ds = load_data_local(DIR_TRAIN)
    test_ds = load_data_local(DIR_TEST)
    assert train_ds.class_names == test_ds.class_names
    model = get_baseline_model(train_ds, test_ds, dummy=dummy)
    return model


def create_and_save_dummy_model(filename, dummy=True):
    model = pipeline_baseline(dummy)
    dump(model, f"{filename}.joblib")
    print(f"Dummy model saved to {filename}.joblib")
    return model

def load_dummy_model(filename):
    model = load(f"{filename}.joblib")
    print(f"Dummy model loaded from {filename}.joblib.")
    return model

def get_dummy_model(filename):
    try:
        model = load_dummy_model(f"{DIR_MODELS}/dummy_emotion")
        print("try block ends")
    except:
        print("except begins")
        model = create_and_save_dummy_model(f"{DIR_MODELS}/dummy_emotion", dummy=True)
    return model

if __name__ == "__main__":
    """Return a model, try first if there is one saved on disk
    """

    get_dummy_model("dummy_emotion")
