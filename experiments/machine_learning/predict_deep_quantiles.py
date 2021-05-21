import random
from datetime import datetime

import numpy as np

from tensorflow.keras.layers import LSTM, Input, BatchNormalization, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras import losses, optimizers
from tensorflow.keras.models import Model

from sklearn import preprocessing

import matplotlib.pyplot as plt
import tqdm

import pykoda


def timestmp_to_time_of_day(timestamp):
    time_of_day = []
    for ts in timestamp:
        t = datetime.fromtimestamp(ts)
        time_of_day.append(t.hour / 24 + t.minute / (24 * 60) + t.second / (24 * 3600))

    return time_of_day


def get_features(company, date, start_hour, end_hour, selected_features, n_select=None):
    df = pykoda.datautils.get_data_range(feed='TripUpdates', company=company, start_date=date,
                                         start_hour=start_hour, end_hour=end_hour, merge_static=True)

    # There are some non-public-facing trips that should be removed.
    # One of them is indicated as Servicelinje, so we create a mask
    trips_to_exclude = df.query('stop_headsign=="Servicelinje"').trip_id.unique()
    to_remove = df.trip_id.map(set(trips_to_exclude).issubset)
    vehicle_positions = df[~to_remove]

    # Take the trips that have at least 25 stops
    grouped_trips = df.groupby('trip_id')
    candidate_trips = set(grouped_trips.filter(lambda x: len(x) > 25).trip_id.unique())

    if n_select is None:
        candidate_trips = list(candidate_trips)
    else:
        candidate_trips = list(candidate_trips)[:n_select]
    delay_list = []
    feature_list = []
    for tripid in tqdm.tqdm(candidate_trips, desc='Loading trips'):
        this_data = grouped_trips.get_group(tripid).sort_values(by='timestamp')

        # Extract and compute some of the possible features
        latitude = this_data.stop_lat
        longitude = this_data.stop_lon
        time_observed = this_data.observed_arrival_time

        _sched_arrival = this_data.scheduled_arrival_time.values.astype(np.int64) // 10 ** 9  # Convert to timestamp
        travel_time_scheduled = (_sched_arrival - _sched_arrival.min()) / 60
        delay = this_data.arrival_delay
        stop_length = (this_data.scheduled_departure_time - this_data.scheduled_departure_time).astype(np.int64) / 60e9
        travelled_distance = this_data.shape_dist_traveled / 1000.
        this_delay = np.array(delay, dtype=np.float32) / 60 / 60
        time_of_day = timestmp_to_time_of_day(_sched_arrival)

        # Stack the ones that we selected
        locs = locals()
        features = np.stack([locs[f] for f in selected_features], axis=1).astype(np.float32)

        is_nan = np.logical_or(np.isnan(features).any(axis=1), np.isnan(this_delay))

        if (~is_nan).sum() > 2:
            delay_list.append(this_delay[~is_nan, np.newaxis])
            feature_list.append(features[~is_nan, :])
    return feature_list, delay_list


def get_deep_model(depth=2, width=16, n_features=2, l2_reg=1e-6, n_quantiles=5):
    if l2_reg > 0:
        reg = l2(l2_reg)
    else:
        reg = None

    dense_options = dict(activation='relu', kernel_initializer='he_normal', kernel_regularizer=reg)
    lstm_options = dict(return_sequences=True, return_state=False, kernel_regularizer=reg)

    input_ = Input(shape=(None, n_features))
    x = BatchNormalization()(input_)
    x = Dense(width, **dense_options)(x)
    x = Dropout(0.2)(x)

    for _ in range(depth):
        x = BatchNormalization()(x)
        x = LSTM(width, **lstm_options)(x)
    x = BatchNormalization()(x)
    x = Dense(width * 2, **dense_options)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(width * 2, **dense_options)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='linear')(x)

    model = Model(inputs=input_, outputs=x)
    model.compile(loss=losses.Huber(delta=0.1), optimizer=optimizers.Adam(clipnorm=1), metrics=['mae', 'mse'])
    model.summary()
    return model


def main():
    START_HOUR = 7
    END_HOUR = 15
    COMPANY_NAME = 'otraf'
    DATE_TRAIN = '2020-08-21'
    DATE_TEST = '2020-08-22'
    SAMPLES = 200 * 3
    EPOCHS = 100
    BATCH_SIZE = 2
    WIDTH = 128
    N_QUANTILES = 5
    FEATURES = ['travelled_distance', 'travel_time_scheduled', 'stop_length', 'latitude', 'longitude', 'time_of_day']

    train_X, train_y_raw = get_features(COMPANY_NAME, DATE_TRAIN, START_HOUR, END_HOUR, FEATURES, n_select=SAMPLES)
    test_X, test_y_raw = get_features(COMPANY_NAME, DATE_TEST, START_HOUR, END_HOUR, FEATURES, n_select=SAMPLES)

    quantiler = preprocessing.QuantileTransformer(n_quantiles=N_QUANTILES)
    quantiler.fit(np.concatenate(train_y_raw))
    train_y_scaled = [quantiler.transform(y) for y in train_y_raw]
    test_y_scaled = [quantiler.transform(y) for y in test_y_raw]

    def _data_generator(x, y):
        while True:
            # Shuffle the data
            temp = list(zip(x, y))
            random.shuffle(temp)
            x, y = zip(*temp)

            for a, b in zip(x, y):
                yield a[np.newaxis, ...], b[np.newaxis, ...]

    def data_generator(x, y, batch_size):
        generator = _data_generator(x, y)
        while True:
            X = []
            Y = []

            # Get the batch
            for _ in range(batch_size):
                x_, y_ = next(generator)
                X.append(x_)
                Y.append(y_)

            # Pad them to the same size
            max_len = max(y.shape[1] for y in Y)
            X_ = []
            Y_ = []
            weights = []
            for x_, y_ in zip(X, Y):
                pad_ = max_len - y_.shape[1]

                X_.append(np.pad(x_, ((0, 0), (0, pad_), (0, 0)), mode='constant', constant_values=0))
                Y_.append(np.pad(y_, ((0, 0), (0, pad_), (0, 0)), mode='constant', constant_values=0))

                # Define the sample weights of the padded area to 0.
                w = np.ones(max_len)
                w[max_len:] = 0
                weights.append(w)

            yield np.concatenate(X_, axis=0), np.concatenate(Y_, axis=0), np.array(weights)

    model = get_deep_model(width=WIDTH, depth=1, n_features=train_X[0].shape[1], n_quantiles=N_QUANTILES)

    print()
    print('Ready to train on', sum(len(x) for x in train_y_raw), 'datapoints from', len(train_y_raw), 'trips')
    print('and test on', sum(len(x) for x in test_y_raw), 'samples from', len(test_y_raw), 'trips')
    print()

    history = model.fit(data_generator(train_X, train_y_scaled, BATCH_SIZE),
                        validation_data=data_generator(test_X, test_y_scaled, BATCH_SIZE),
                        validation_steps=len(test_y_scaled) // BATCH_SIZE,
                        steps_per_epoch=len(train_y_scaled) // BATCH_SIZE, epochs=EPOCHS)

    fig, axes = plt.subplots(3, 1, sharex=True, constrained_layout=True)
    plt.sca(axes[0])
    plt.plot(history.history['loss'], color='b', ls='--', label='Train')
    plt.plot(history.history['val_loss'], color='b', label='Test')
    plt.ylabel('Loss')
    plt.semilogy()
    plt.legend(loc='upper right')

    plt.sca(axes[1])
    plt.plot(history.history['mae'], color='g', ls='--')
    plt.plot(history.history['val_mae'], color='g')
    plt.ylabel('MAE')
    plt.semilogy()
    plt.sca(axes[2])
    plt.plot(history.history['mse'], color='r', ls='--')
    plt.plot(history.history['val_mse'], color='r')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.semilogy()
    plt.savefig('history.png')

    predicted_test_scaled = np.concatenate([model(x[:, np.newaxis], training=False).numpy().squeeze() for x in test_X])
    predicted_test_raw = quantiler.inverse_transform(predicted_test_scaled[:, None]).squeeze()
    truth_test_scaled = np.concatenate(test_y_scaled)
    truth_test_raw = quantiler.inverse_transform(truth_test_scaled[:, None]).squeeze()

    predicted_train_scaled = np.concatenate(
        [model(x[:, np.newaxis], training=False).numpy().squeeze() for x in train_X])
    predicted_train_raw = quantiler.inverse_transform(predicted_train_scaled[:, None]).squeeze()
    truth_train_scaled = np.concatenate(train_y_scaled)
    truth_train_raw = quantiler.inverse_transform(truth_train_scaled[:, None]).squeeze()

    plt.figure()
    plt.scatter(truth_test_scaled, predicted_test_scaled, alpha=0.5, s=3, label='Test set')
    plt.scatter(truth_train_scaled, predicted_train_scaled, alpha=0.5, s=3, label='Training set')

    plt.axvline(0, color='k', alpha=0.3)
    plt.axhline(0, color='k', alpha=0.3)

    _range = min(truth_test_scaled), max(truth_test_scaled)
    plt.plot(_range, _range, color='k', ls='-.')
    plt.xlabel('Actual delay [quantile]')
    plt.ylabel('Predicted delay [quantile]')
    plt.savefig('plot1')

    plt.figure()
    plt.scatter(truth_test_raw, predicted_test_raw, alpha=0.5, s=3, label='Test set')
    plt.scatter(truth_train_raw, predicted_train_raw, alpha=0.5, s=3, label='Training set')

    plt.axvline(0, color='k', alpha=0.3)
    plt.axhline(0, color='k', alpha=0.3)

    _range = min(truth_test_raw), max(truth_test_raw)
    plt.plot(_range, _range, color='k', ls='-.')
    plt.xlabel('Actual delay [h]')
    plt.ylabel('Predicted delay [h]')
    plt.savefig('plot2')

    plt.show()


if __name__ == '__main__':
    main()
