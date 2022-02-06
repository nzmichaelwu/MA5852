import argparse, os
import numpy as np

import tensorflow as tf


if __name__ == '__main__':
    
    # Hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=60)
    
    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    
    args, _ = parser.parse_known_args()
    
    epochs = args.epochs
    batch_size = args.batch_size
    
    gpu_count = args.gpu_count
    model_dir = args.model_dir
    training_dir = args.training
    test_dir = args.test
    
    X_train = np.load(os.path.join(training_dir, 'training.npz'))['feature']
    y_train = np.load(os.path.join(training_dir, 'training.npz'))['target']
    X_test = np.load(os.path.join(test_dir, 'test.npz'))['feature']
    y_test = np.load(os.path.join(test_dir, 'test.npz'))['target']
    
    '''
        define model structure with dropout and early stopping
    '''
    model_mod = tf.keras.models.Sequential()
    model_mod.add(tf.keras.layers.Dense(58, activation='relu')) # 2/3 of the number of inputs (83) + number of outputs (3, as there are 3 classes) ~ 58
    model_mod.add(tf.keras.layers.Dropout(0.5))
    model_mod.add(tf.keras.layers.Dense(29, activation='relu')) # half number of neurons as the first hidden layer
    model_mod.add(tf.keras.layers.Dropout(0.5))
    model_mod.add(tf.keras.layers.Dense(3, activation='softmax')) # number of neurons at the output layer = number of classes of the target (3)

    if gpu_count > 1:
        model_mod = multi_gpu_model(model_mod, gpus=gpu_count)
    
    # compile the model
    model_mod.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    
    # early stopping
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1)
    
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())

    model_mod.fit(X_train, y_train, batch_size=batch_size,
                  validation_split=0.1,
                  epochs=epochs,
                  callbacks=[es]
                 )
    
    print(model_mod.summary())
    
    score = model_mod.evaluate(X_test, y_test, verbose=0)
    print('Test loss    :', score[0])
    print('Test accuracy:', score[1])
    print('Test precision:', score[2])
    print('Test recall:', score[3])
    
    # save Keras model for Tensorflow Serving
    model_mod.save(os.path.join(model_dir, '1'))