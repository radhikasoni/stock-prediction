import os
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dropout, Dense, LSTM


class LongShortTermMemory:
    def __init__(self, project_folder):
        self.project_folder = project_folder

    def get_defined_metrics(self):
        defined_metrics = [
            tf.keras.metrics.MeanSquaredError(name='MSE')
        ]
        return defined_metrics

    def get_callback(self):
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)
        return callback

    def create_model(self, x_train):
        # model = Sequential()
        # # 1st layer with Dropout regularisation
        # # * units = add 100 neurons is the dimensionality of the output space
        # # * return_sequences = True to stack LSTM layers so the next LSTM layer has a three-dimensional sequence input
        # # * input_shape => Shape of the training dataset
        
        # # model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))

        # model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        # # 20% of the layers will be dropped
        # model.add(Dropout(0.2))
        # # 2nd LSTM layer
        # # * units = add 50 neurons is the dimensionality of the output space
        # # * return_sequences = True to stack LSTM layers so the next LSTM layer has a three-dimensional sequence input
        # model.add(LSTM(units=50, return_sequences=True))
        # # 20% of the layers will be dropped
        # model.add(Dropout(0.2))
        # # 3rd LSTM layer
        # # * units = add 50 neurons is the dimensionality of the output space
        # # * return_sequences = True to stack LSTM layers so the next LSTM layer has a three-dimensional sequence input
        # model.add(LSTM(units=50, return_sequences=True))
        # # 50% of the layers will be dropped
        # model.add(Dropout(0.5))
        # # 4th LSTM layer
        # # * units = add 50 neurons is the dimensionality of the output space
        # model.add(LSTM(units=50))
        # # 50% of the layers will be dropped
        # model.add(Dropout(0.5))
        # # Dense layer that specifies an output of one unit
        # model.add(Dense(units=1))
        # model.summary()
        # #tf.keras.utils.plot_model(model, to_file=os.path.join(self.project_folder, 'model_lstm.png'), show_shapes=True,
        # #                          show_layer_names=True)

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)  # Predict Close Price
        ])
        return model
