# import packages
import argparse, os
import numpy as np

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-q", "-m", "pip", "install", package])

install('matplotlib')

import matplotlib.pyplot as plt

import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


if __name__ == '__main__':
    # Hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
#     parser.add_argument('--batch-size', type=int, default=1)
    
    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--output', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args, _ = parser.parse_known_args()
    
    epochs = args.epochs
#     batch_size = args.batch_size
    
    gpu_count = args.gpu_count
    model_dir = args.model_dir
    training_dir = args.training
    test_dir = args.test
    output_dir = args.output
    
    X_train = np.load(os.path.join(training_dir, 'training.npz'))['image']
    y_train = np.load(os.path.join(training_dir, 'training.npz'))['label']
    X_test = np.load(os.path.join(test_dir, 'test.npz'))['image']
    y_test = np.load(os.path.join(test_dir, 'test.npz'))['label']
    
    print(f'X_train shape: {X_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    print(f'y_test shape: {y_test.shape}')
    
    def vis_training(hlist, start=1):
        """
            This function will help to visualize the loss, val_loss, accuracy etc.
        """
        # getting history of all kpi for each epochs
        loss = np.concatenate([hlist.history['loss']])
        val_loss = np.concatenate([hlist.history['val_loss']])
        acc = np.concatenate([hlist.history['accuracy']])
        val_acc = np.concatenate([hlist.history['val_accuracy']])
        epoch_range = range(1,len(loss)+1)

        # Block for training vs validation loss
        plt.figure(figsize=[12,6])
        plt.subplot(1,2,1)
        plt.plot(epoch_range[start-1:], loss[start-1:], label='Training Loss')
        plt.plot(epoch_range[start-1:], val_loss[start-1:], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.legend()
        # Block for training vs validation accuracy
        plt.subplot(1,2,2)
        plt.plot(epoch_range[start-1:], acc[start-1:], label='Training Accuracy')
        plt.plot(epoch_range[start-1:], val_acc[start-1:], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'train_log.png'))
    
    
    # Since this is a small dataset, we need to generate some data to increase accuracy
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1, 
        height_shift_range=0.1,
        horizontal_flip=False, 
        vertical_flip=False
    )
    
    # Converting labels in int format as TF accepts only int in targets class
    y_train[y_train == 'Covid'] = 0
    y_train[y_train == 'Normal'] = 1
    
    y_test[y_test == 'Covid'] = 0
    y_test[y_test == 'Normal'] = 1
    
    # converting to int format
    num_classes = 2
    y_train = to_categorical(y_train.astype('int'), num_classes)
    y_test = to_categorical(y_test.astype('int'), num_classes)
    
    # adding extra data from datagen
    train_gen = datagen.flow(X_train, y_train)
    test_gen = datagen.flow(X_test, y_test)
    
    # define model structure
    model = tf.keras.Sequential()
    ## 1st convolution layer
    ### convolutional layer with 64 filters
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)))
    ## downsample using a max pooling layer, which feeds into the next set of convolutional layers
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    ## 2nd convolultion layer
    ## convolutional layer with 32 filters
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    # flatten and classify
    ## flattern spacial information into a vector, and learn the final probability distribution for each class
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(2, activation='sigmoid')) # number of neurons at the output layer = number of classes of the target (2)
    # Take a look at the model summary
    model.summary()
    
    if gpu_count > 1:
        model = multi_gpu_model(model, gpus=gpu_count)
        
    # compile the model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    
    # early stopping
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1)
    
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())

    train_log = model.fit(train_gen,
                          validation_data = test_gen,
                          #batch_size=8,
                          epochs=epochs,
                          callbacks=[es]
                        )

    # evaluate on training data
    score_train = model.evaluate(train_gen, verbose=0)
    print(f'Training set loss is: {score_train[0]}')
    print(f'Training set accuracy is: {score_train[1]}')
    print(f'Training set precision is: {score_train[2]}')
    print(f'Training set recall is: {score_train[3]}')        
    
    # evaluate on test data
    score = model.evaluate(test_gen, verbose=0)
    print(f'Test set loss is: {score[0]}')
    print(f'Test set accuracy is: {score[1]}')
    print(f'Test set precision is: {score[2]}')
    print(f'Test set recall is: {score[3]}')

    # Visuals of loss and accuracy
    vis_training(train_log, start=1)
    
    # save Keras model for Tensorflow Serving
    model.save(os.path.join(model_dir, '1'))    
    
