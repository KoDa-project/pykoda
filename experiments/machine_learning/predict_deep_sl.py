import random
from datetime import datetime

import numpy as np

from tensorflow.keras.layers import LSTM, Input, BatchNormalization, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras import losses, optimizers, callbacks
from tensorflow.keras.models import Model

import matplotlib.pyplot as plt
import tqdm

import pykoda


def timestmp_to_time_of_day(timestamp):
    time_of_day = []
    for ts in timestamp:
        t = datetime.fromtimestamp(ts)
        time_of_day.append(t.hour / 24 + t.minute / (24 * 60) + t.second / (24 * 3600))

    return time_of_day


def get_features(company, date_start, date_end, start_hour, end_hour, selected_features, n_select=None):
    # Load all the data for the selected route.
    df = pykoda.datautils.get_data_range(feed='TripUpdates', company=company, start_date=date_start,
                                         end_date=date_end, start_hour=start_hour, end_hour=end_hour,
                                         merge_static=True,
                                         query='route_short_name == "3"')  # 'route_desc == "blåbuss"'

    # Find out the corresponding trips (each individual instance of the route)
    grouped_trips = df.groupby('trip_id')
    candidate_trips = list(df.trip_id.unique())
    if n_select is not None:
        candidate_trips = candidate_trips[:n_select]
    delay_list = []
    feature_list = []

    for tripid in tqdm.tqdm(candidate_trips, desc='Loading trips'):
        this_data = grouped_trips.get_group(tripid).sort_values(by='stop_sequence')

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
        direction_id = this_data.direction_id.values

        # Stack the ones that we selected
        locs = locals()
        features = np.stack([locs[f] for f in selected_features], axis=1).astype(np.float32)

        is_nan = np.logical_or(np.isnan(features).any(axis=1), np.isnan(this_delay))

        if (~is_nan).sum() > 2:
            delay_list.append(this_delay[~is_nan, np.newaxis])
            feature_list.append(features[~is_nan, :])
    return feature_list, delay_list


def get_deep_model(depth: int = 2, width=16, n_features=2, l2_reg=1e-6):
    """
    Define a simple LSTM-based deep learning model.

    :param depth: number of stacked LSTM layers
    :param width: number of neurons per LSTM cell
    :param n_features: number of feature channels
    :param l2_reg: L^2 regularisation parameters
    :return:
    """
    if l2_reg > 0:
        reg = l2(l2_reg)
    else:
        reg = None

    dense_options = dict(activation='relu', kernel_initializer='he_normal', kernel_regularizer=reg)
    lstm_options = dict(return_sequences=True, return_state=False, kernel_regularizer=reg)

    input_ = Input(shape=(None, n_features))
    x = BatchNormalization()(input_)
    x = Dense(width, **dense_options)(x)  # Project the inputs to a space of dimension `width`
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

    # We use a Huber loss, that is parabolic for small errors, but linear for large ones.
    # This prevents us from penalising outliers too much
    loss = losses.Huber(delta=1 / 6.)  # The cross-over point is ten minutes

    # We use the RMSProp optimizer with a gradient clipping. This is important in recurrent neural networks to help
    # stability and convergence.
    opt = optimizers.RMSprop(clipnorm=1.)
    model.compile(loss=loss, optimizer=opt, metrics=['mae', 'mse'])
    model.summary()
    return model


def main():
    # Training and testing data parameters
    start_hour = 0
    end_hour = 23
    company_name = 'sl'
    date_train_start = '2020_09_15'
    date_train_end = '2020_09_17'
    date_test_start = '2020_09_22'
    date_test_end = '2020_09_24'

    # Training features
    features = ['travelled_distance', 'travel_time_scheduled', 'stop_length', 'latitude', 'longitude', 'time_of_day']

    # Training
    epochs = 20
    batch_size = 4
    width = 16

    train_X, train_y = get_features(company_name, date_train_start, date_train_end, start_hour, end_hour, features)
    test_X, test_y = get_features(company_name, date_test_start, date_test_end, start_hour, end_hour, features)

    model = get_deep_model(width=width, depth=1, n_features=train_X[0].shape[1], l2_reg=1e-15)

    def data_generator(x, y, batch_size):
        """This iterator stacks data in batches of constant length.
        """

        def _data_generator(x, y):
            """Yield all pairs of training and test, on different orders every time."""
            while True:
                # Shuffle the data
                temp = list(zip(x, y))
                random.shuffle(temp)
                x, y = zip(*temp)

                for a, b in zip(x, y):
                    yield a[np.newaxis, ...], b[np.newaxis, ...]

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

    print()
    print('Ready to train on', sum(len(x) for x in train_y), 'datapoints from', len(train_y), 'trips')
    print('and test on', sum(len(x) for x in test_y), 'samples from', len(test_y), 'trips')
    print()

    epoch_steps = 7
    decay_rate = 0.8
    initial_lr = 0.001
    lr_scheduler = callbacks.LearningRateScheduler(lambda epoch: initial_lr * decay_rate ** (epoch // epoch_steps))

    history = model.fit(data_generator(train_X, train_y, batch_size),
                        validation_data=data_generator(test_X, test_y, batch_size),
                        validation_steps=len(test_y) // batch_size,
                        steps_per_epoch=len(train_y) // batch_size, epochs=epochs,
                        callbacks=[lr_scheduler])

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

    predicted = np.concatenate([model(x[:, np.newaxis], training=False).numpy().squeeze() for x in test_X]).squeeze()
    truth = np.concatenate(test_y).squeeze()

    predicted_train = np.concatenate(
        [model(x[:, np.newaxis], training=False).numpy().squeeze() for x in train_X]).squeeze()
    truth_train = np.concatenate(train_y).squeeze()

    print('MSE train:', np.mean(np.square(truth_train - predicted_train)))
    print('MSE test:', np.mean(np.square(truth - predicted)))

    print('MAE train:', np.mean(np.abs(truth_train - predicted_train)))
    print('MAE test:', np.mean(np.abs(truth - predicted)))

    plt.figure()
    plt.scatter(truth, predicted, alpha=0.5, s=3, label='Test set')
    plt.scatter(truth_train, predicted_train, alpha=0.5, s=3, label='Training set')

    plt.axvline(0, color='k', alpha=0.3)
    plt.axhline(0, color='k', alpha=0.3)

    _range = min(truth), max(truth)
    plt.plot(_range, _range, color='k', ls='-.')
    plt.xlabel('Actual delay [h]')
    plt.ylabel('Predicted delay [h]')
    plt.legend(loc='best')
    plt.savefig('plot')

    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, constrained_layout=True)
    station_index = np.concatenate([np.arange(len(y)) for y in test_y])
    station_index_train = np.concatenate([np.arange(len(y)) for y in train_y])
    plt.sca(ax[0])
    plt.title('Test set')
    plt.scatter(truth, predicted, alpha=0.5, s=3, c=station_index)
    plt.colorbar(label='Station index')
    plt.axvline(0, color='k', alpha=0.3)
    plt.axhline(0, color='k', alpha=0.3)
    _range = min(truth), max(truth)
    plt.plot(_range, _range, color='k', ls='-.')

    plt.sca(ax[1])
    plt.title('Training set')
    plt.scatter(truth_train, predicted_train, alpha=0.5, s=3, label='Training set', c=station_index_train)
    plt.colorbar(label='Station index')
    plt.axvline(0, color='k', alpha=0.3)
    plt.axhline(0, color='k', alpha=0.3)
    _range = min(truth), max(truth)
    plt.plot(_range, _range, color='k', ls='-.')

    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, constrained_layout=True)
    plt.sca(ax[0])
    plt.title('Test set')
    indexes_ts = list(range(len(test_y)))
    random.shuffle(indexes_ts)
    for ix in indexes_ts[:20]:
        plt.plot(test_y[ix], color='b', alpha=0.1)
    plt.axhline(0, color='k', alpha=0.3)

    plt.sca(ax[1])
    plt.title('Training set')
    indexes_tr = list(range(len(train_y)))
    random.shuffle(indexes_tr)
    for ix in indexes_tr[:20]:
        plt.plot(train_y[ix], color='b', alpha=0.1)
    plt.axhline(0, color='k', alpha=0.3)

    plt.show()


if __name__ == '__main__':
    main()
