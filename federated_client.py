import flwr as fl
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten
from utils import INPUT_SHAPE, batch_generator
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os

# Adapted code for loading data and building the model as functions
def load_data(data_dir, test_size):
    # Adaptation based on the provided code, e.g., for driving_log_split_1.csv
    data_df = pd.read_csv(data_dir, names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=0)
    return X_train, X_valid, y_train, y_valid

def build_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()
    return model

def train_model(model, data_dir, X_train, X_valid, y_train, y_valid, batch_size, nb_epoch, samples_per_epoch, learning_rate):
    checkpoint = ModelCheckpoint('federated_split_1-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=False,
                                 mode='auto')
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate))
    model.fit(x=batch_generator(data_dir, X_train, y_train, batch_size, True),
              steps_per_epoch=samples_per_epoch // batch_size,
              epochs=nb_epoch,
              validation_data=batch_generator(data_dir, X_valid, y_valid, batch_size, False),
              validation_steps=len(X_valid) // batch_size,
              callbacks=[checkpoint],
              verbose=1)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, data_dir, batch_size, nb_epoch, samples_per_epoch, learning_rate, test_size, keep_prob):
        self.model = model
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.samples_per_epoch = samples_per_epoch
        self.learning_rate = learning_rate
        self.test_size = test_size
        self.keep_prob = keep_prob

    def get_parameters(self, config):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        X_train, X_valid, y_train, y_valid = load_data(self.data_dir, self.test_size)
        train_model(self.model, self.data_dir, X_train, X_valid, y_train, y_valid, self.batch_size, self.nb_epoch, self.samples_per_epoch, self.learning_rate)
        return self.get_parameters(config), len(X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        X_train, X_valid, y_train, y_valid = load_data(self.data_dir, self.test_size)
        results = self.model.evaluate(batch_generator(self.data_dir, X_valid, y_valid, self.batch_size, False), steps=len(X_valid) // self.batch_size)

        # If your model is compiled with one metric (accuracy)
        if isinstance(results, list) and len(results) == 2:
            loss, accuracy = results
        else:
            loss = results
            accuracy = None  # Or handle this case as you see fit
        return loss, len(X_valid), {"accuracy": accuracy} if accuracy is not None else {}

def main():
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    args = parser.parse_args()

    # Initialize and configure your model and Flower client here
    model = build_model()  # Build the actual model using the `build_model` function
    client = FlowerClient(model, args.data_dir, batch_size=40, nb_epoch=1, samples_per_epoch=21000, learning_rate=1e-4, test_size=0.2, keep_prob=0.5)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client.to_client())

if __name__ == "__main__":
    main()
