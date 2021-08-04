# Import libraries

from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import RMSprop, SGD, Adam, Adadelta, Adagrad, Adamax, Nadam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, Callback
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow import keras
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, make_scorer
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import ImageOps

from cnn_models.vgg16 import create_vgg16
from data_visualisation.csv_report import generate_csv_report, generate_csv_metadata
from data_visualisation.roc_curves import plot_roc_curve_binary, plot_roc_curve_multiclass
from data_visualisation.plots import plot_confusion_matrix, plot_comparison_chart, plot_training_results

from utils import create_label_encoder

import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time


cbis_path = 'CBIS_DDSM'

def evaluate_model(model, x:list, y_true: list, runtime) -> None:
    label_encoder = create_label_encoder()
    prediction = model.predict(x=x)
    # Inverse transform y_true and y_pred from one-hot-encoding to original label.
    if label_encoder.classes_.size == 2:
        y_true_inv = y_true
        y_pred_inv = np.round_(prediction, 0)
    else:
        y_true_inv = label_encoder.inverse_transform(
            np.argmax(y_true, axis=1))
        y_pred_inv = label_encoder.inverse_transform(
            np.argmax(prediction, axis=1))

    # Calculate accuracy.
    accuracy = float('{:.4f}'.format(
        accuracy_score(y_true_inv, y_pred_inv)))
    print("Accuracy = {}\n".format(accuracy))

    # Generate CSV report.
    generate_csv_report(y_true_inv, y_pred_inv, label_encoder, accuracy)
    generate_csv_metadata(runtime)

    # Plot confusion matrix and normalised confusion matrix.
    # Calculate CM with original label of classes
    cm = confusion_matrix(y_true_inv, y_pred_inv)
    plot_confusion_matrix(cm, 'd', label_encoder, False)
    # Calculate normalized confusion matrix with original label of classes.
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized[np.isnan(cm_normalized)] = 0
    plot_confusion_matrix(cm_normalized, '.2f', label_encoder, True)

    # Plot ROC curve.
    plot_roc_curve_binary(y_true, model.prediction)

def preprocess_image(image_path: str) -> np.ndarray:
    image = load_img(image_path, color_mode="grayscale", target_size=(150,150))
    image = ImageOps.autocontrast(image, cutoff=3)
    image = img_to_array(image)
    #image /= 255.0
    return image

def encode_labels(labels_list: np.ndarray) -> np.ndarray:
    label_encoder = create_label_encoder()
    labels = label_encoder.fit_transform(labels_list)
    return labels

def import_cbisddsm_dataset():
    print("Importing CBIS-DDSM training set")   
    trainingPath = "all_mask_local_png.csv"
    df = pd.read_csv(trainingPath)
    list_IDs = df['img_path'].values
    labels = encode_labels(df['label'].values)
    images = list()

    for image_path in list_IDs:
        images.append(preprocess_image(image_path))
    #images = np.array(images, dtype="float32")  
    images = np.array(images)  

    return images, labels

def load_training():
    """
    Load the training set (excluding baseline patches)
    """
    images = np.load(os.path.join(
        cbis_path, 'numpy data', 'train_tensor.npy'))[1::2]
    labels = np.load(os.path.join(
        cbis_path, 'numpy data', 'train_labels.npy'))[1::2]
    return images, labels

def load_testing():
    """
    Load the test set (abnormalities patches and labels, no baseline)
    """
    images = np.load(os.path.join(cbis_path, 'numpy data',
                                    'public_test_tensor.npy'))[1::2]
    labels = np.load(os.path.join(cbis_path, 'numpy data',
                                    'public_test_labels.npy'))[1::2]
    return images, labels

def remap_label(l):
    """
    Remap the labels to 0->mass 1->calcification
    """
    if l == 1 or l == 2:
        return 0
    elif l == 3 or l == 4:
        return 1
    else:
        print("[WARN] Unrecognized label (%d)" % l)
        return None

testingImages = True

if testingImages : 
    images, labels = import_cbisddsm_dataset()
    X_train, y_train, X_test, y_test = train_test_split(images,
    labels, test_size=0.20, random_state=0, shuffle=True)
    train_images, train_labels, test_images, test_labels = X_train, X_test, y_train, y_test
else: 
    train_images, train_labels = load_training()
    test_images, test_labels = load_testing()

# Number of images
n_train_img = train_images.shape[0]
n_test_img = test_images.shape[0]
print("Train size: %d \t Test size: %d" % (n_train_img, n_test_img))

# Compute width and height of images
img_w = train_images.shape[1]
img_h = train_images.shape[2]
print("Image size: %dx%d" % (img_w, img_h))

# Remap labels
if not(testingImages):
    train_labels = np.array([remap_label(l) for l in train_labels])
    test_labels = np.array([remap_label(l) for l in test_labels])
    # Create a new dimension for color in the images arrays
    train_images = train_images.reshape((n_train_img, img_w, img_h, 1))
    test_images = test_images.reshape((n_test_img, img_w, img_h, 1))

# Convert from 16-bit (0-65535) to to 8-bit (0-255)
train_images = train_images.astype('uint16') / 256
test_images = test_images.astype('uint16') / 256

# Replicate the only color channel (gray) 3 times, for VGGNet compatibility
train_images = np.repeat(train_images, 3, axis=3)
test_images = np.repeat(test_images, 3, axis=3)

# Shuffle the training set (originally sorted by label)
perm = np.random.permutation(n_train_img)
train_images = train_images[perm]
train_labels = train_labels[perm]

# Create a generator for training images
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=15,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='reflect'
)

# Fit the generator with some images
train_datagen.fit(train_images)

# Split train images into actual training and validation
train_generator = train_datagen.flow(
    train_images, train_labels, batch_size=128, subset='training')
validation_generator = train_datagen.flow(
    train_images, train_labels, batch_size=128, subset='validation')

# Preprocess the test images as well
preprocess_input(test_images)

# Instantiate a VGG16 network with custom final layer
vgg16_fe = create_vgg16(dropout=0.5, fc_size=128)

# Early stopping (stop training after the validation loss reaches the minimum)
earlystopping = EarlyStopping(
    monitor='val_loss', mode='min', patience=30, verbose=1)

# Callback for checkpointing
checkpoint = ModelCheckpoint('vgg16_fe_2cl_best.h5',
                                monitor='val_loss', mode='min', verbose=1,
                                save_best_only=True, save_freq='epoch'
                                )

# Compile the model
vgg16_fe.compile(optimizer='rmsprop',
                    loss='binary_crossentropy', metrics=['accuracy'])

start_time = time.time()
# Train
history_vgg16_fe = vgg16_fe.fit_generator(
    train_generator,
    steps_per_epoch=n_train_img // 128,
    epochs=200,
    validation_data=validation_generator,
    callbacks=[checkpoint, earlystopping],
    shuffle=True,
    verbose=1,
    initial_epoch=0
)

# Save
models.save_model(vgg16_fe, 'vgg16_fe_2cl_end.h5')

fineTuning = False
if fineTuning:
    vgg16_fe = models.load_model('vgg16_fe_2cl_end.h5')

    # Fine tuning: unfreeze the last convolutional layer
    for layer in vgg16_fe.get_layer('vgg16').layers:
        if layer.name.startswith('block5_conv3'):
            layer.trainable = True
        else:
            layer.trainable = False

    # Recompile the model ()
    vgg16_fe.compile(optimizer=RMSprop(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    # Callback for early-stopping
    early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=30, verbose=1)

    # Callback for checkpointing
    checkpoint = ModelCheckpoint(
            'vgg16_fe_2cl_best.h5',
            monitor='val_loss',
            mode='min',
            verbose=1, 
            save_best_only=True, 
            save_freq='epoch')

    # Train
    history_vgg16_fe = vgg16_fe.fit_generator(
            train_generator,
            steps_per_epoch=n_train_img // 128,
            epochs=200,
            validation_data=validation_generator,
            callbacks=[early_stop, checkpoint],
            shuffle=True,
            verbose=1,
            initial_epoch=0)

    # Save
    models.save_model(vgg16_fe, 'vgg16_fe_2cl_end.h5')

# Save training runtime.
runtime = round(time.time() - start_time, 2)

if testingImages:
    evaluate_model(vgg16_fe, test_images, test_labels, runtime)

# History of accuracy and loss
tra_loss_fe = history_vgg16_fe.history['loss']
tra_acc_fe = history_vgg16_fe.history['acc']
val_loss_fe = history_vgg16_fe.history['val_loss']
val_acc_fe = history_vgg16_fe.history['val_acc']

# Total number of epochs training
epochs_fe = range(1, len(tra_acc_fe)+1)
end_epoch_fe = len(tra_acc_fe)

# Epoch when reached the validation loss minimum
opt_epoch_fe = val_loss_fe.index(min(val_loss_fe)) + 1

# Loss and accuracy on the validation set
end_val_loss_fe = val_loss_fe[-1]
end_val_acc_fe = val_acc_fe[-1]
opt_val_loss_fe = val_loss_fe[opt_epoch_fe-1]
opt_val_acc_fe = val_acc_fe[opt_epoch_fe-1]

# Loss and accuracy on the test set
opt_vgg16_fe = models.load_model('vgg16_fe_2cl_best.h5')
test_loss_fe, test_acc_fe = vgg16_fe.evaluate(test_images, test_labels, verbose=False)
opt_test_loss_fe, opt_test_acc_fe = opt_vgg16_fe.evaluate(test_images, test_labels, verbose=False)

print("Epoch [end]: %d" % end_epoch_fe)
print("Epoch [opt]: %d" % opt_epoch_fe)
print("Valid accuracy [end]: %.4f" % end_val_acc_fe)
print("Valid accuracy [opt]: %.4f" % opt_val_acc_fe)
print("Test accuracy [end]:  %.4f" % test_acc_fe)
print("Test accuracy [opt]:  %.4f" % opt_test_acc_fe)
print("Valid loss [end]: %.4f" % end_val_loss_fe)
print("Valid loss [opt]: %.4f" % opt_val_loss_fe)
print("Test loss [end]:  %.4f" % test_loss_fe)
print("Test loss [opt]:  %.4f" % opt_test_loss_fe)

if testingImages: 
    # Model accuracy
    plt.figure(figsize=(7, 7), dpi=80, facecolor='w', edgecolor='k')
    plt.title('VGG16 FE accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.plot(epochs_fe, tra_acc_fe, 'r', label='Training set')
    plt.plot(epochs_fe, val_acc_fe, 'g', label='Validation set')
    plt.plot(opt_epoch_fe, val_acc_fe[opt_epoch_fe-1], 'go')
    plt.vlines(opt_epoch_fe, min(val_acc_fe), opt_val_acc_fe, linestyle="dashed", color='g', linewidth=1)
    plt.hlines(opt_val_acc_fe, 1, opt_epoch_fe, linestyle="dashed", color='g', linewidth=1)
    plt.savefig("ModelAccuracy.png")
    plt.legend(loc='lower right')

    # Model loss
    plt.figure(figsize=(7, 7), dpi=80, facecolor='w', edgecolor='k')
    plt.title('VGG16 FE loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.ylim(0.1,0.8)
    plt.plot(epochs_fe, tra_loss_fe, 'r', label='Training set')
    plt.plot(epochs_fe, val_loss_fe, 'g', label='Validation set')
    plt.plot(opt_epoch_fe, val_loss_fe[opt_epoch_fe-1], 'go')
    plt.vlines(opt_epoch_fe, min(val_loss_fe), opt_val_loss_fe, linestyle="dashed", color='g', linewidth=1)
    plt.hlines(opt_val_loss_fe, 1, opt_epoch_fe, linestyle="dashed", color='g', linewidth=1)
    plt.savefig("ModelLoss.png")
    plt.legend();
