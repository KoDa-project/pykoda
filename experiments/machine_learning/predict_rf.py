import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import sklearn.metrics as metrics
from scipy import stats

import matplotlib.pyplot as plt

import pykoda


def main():
    START_HOUR = 9
    END_HOUR = 15
    COMPANY = 'otraf'
    DATE = '2020-08-21'

    df = pykoda.datautils.get_data_range(feed='TripUpdates', company=COMPANY, start_date=DATE,
                                         start_hour=START_HOUR, end_hour=END_HOUR, merge_static=True)

    # Take the trips that have at least 25 stops
    candidate_trips = set(df.groupby('trip_id').filter(lambda x: len(x) > 25).trip_id.unique())
    # Take the first few
    candidate_trips = list(candidate_trips)[0:50]

    delay_list = []
    distance_list = []

    grouped = df.groupby('trip_id')
    for tripid in candidate_trips:
        this_trip_data = grouped.get_group(tripid)

        delay_list.extend(this_trip_data.arrival_delay)
        distance_list.extend(this_trip_data.shape_dist_traveled)

    def evaluate_regression_model(y_realtime, y_predicted):
        mae = metrics.mean_absolute_error(y_realtime, y_predicted)
        mse = metrics.mean_squared_error(y_realtime, y_predicted)

        print('MAE: ', round(mae, 4))
        print('MSE: ', round(mse, 4))
        print('RMSE: ', round(np.sqrt(mse), 4))
        print('R:', round(stats.pearsonr(y_realtime, y_predicted)[0], 2))

    # Build a model
    X = (np.array(distance_list) / 1000.)[:, np.newaxis]  # Distance in km
    y = np.array(delay_list)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

    for model in (LinearRegression(), RandomForestRegressor()):
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        model_name = type(model).__name__
        print('==', model_name, '==')
        evaluate_regression_model(y_test, y_pred)

        plt.scatter(y_test, y_pred, label=model_name, alpha=0.5)

    plt.axvline(0, color='k', alpha=0.3)
    plt.axhline(0, color='k', alpha=0.3)

    _range = y_test.min(), y_test.max()
    plt.plot(_range, _range, color='k', ls='-.')
    plt.xlabel('Delay time [s]')
    plt.ylabel('Predicted delay [s]')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
