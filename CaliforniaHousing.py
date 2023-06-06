import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# Import the dataset
ca_housing = pd.read_csv('../../Datasets/housing.csv')

# Impute missing values
ca_housing['total_bedrooms'] = ca_housing['total_bedrooms'].interpolate(method='linear',
                                                                        limit_direction='forward', axis=0)

# Get dummy values for categorical variables
ca_housing = pd.get_dummies(ca_housing, dtype=int)
print(ca_housing.head())

# Print the shape of the data
print('The shape of our data is:', ca_housing.shape)

# Define labels and features
labels = np.array(ca_housing['median_house_value'])
features = ca_housing.drop('median_house_value', axis=1)

feature_list = list(features.columns)

# Convert pandas dataframe to np array
features = np.array(features)

# Create test and train sets
train_features, test_features, train_labels, test_labels = \
    train_test_split(features, labels, test_size=0.25, random_state=42)

# Print the dimensions of train and test sets
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# Fit the model with random regression
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(train_features, train_labels)

# Get predictions and print the MAE
predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)
print('Mean Absolute Error:', np.mean(errors), 'degrees.')

# Get accuracy
mape = 100 * (errors / test_labels)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
# Result: Accuracy is around 82 percent.

# Introduce parameters to see if accuracy can be improved
param_grid = {'n_estimators': [5, 20, 50, 75, 100], 'max_features': ['sqrt'],
              'max_depth':  [int(x) for x in np.linspace(10, 120, num=12)],
              'min_samples_split': [2, 6, 10],
              'min_samples_leaf': [1, 3, 4],
              'bootstrap': [True, False]}

rf_hyper = RandomizedSearchCV(estimator=rf, param_distributions=param_grid,
                              n_iter=72, cv=5, verbose=2, random_state=35, n_jobs=-1)

rf_hyper.fit(train_features, train_labels)

# Get the best parameters
print(rf_hyper.best_params_)

# Fit the model with new parameters
rf_new = RandomForestRegressor(n_estimators=75, min_samples_split=6, min_samples_leaf=1,
                               max_features='sqrt', max_depth=50, bootstrap=False)
rf_new.fit(train_features, train_labels)

# Get MAE
new_predictions = rf_new.predict(test_features)
new_errors = abs(new_predictions - test_labels)
print('Mean Absolute Error:', np.mean(new_errors), 'degrees.')

# Get accuracy
new_mape = 100 * (new_errors / test_labels)
new_accuracy = 100 - np.mean(new_mape)
print('Accuracy:', round(new_accuracy, 2), '%.')
# Result: Accuracy seems to be dropped from 82 percent to 81 percent. Accuracy can be improved with better tuning
# parameters.

# Get the important variables
importance_list = list(rf_new.feature_importances_)
feature_importance = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importance_list)]

feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importance]

# Map the median house values on a map
figure = px.scatter_mapbox(ca_housing, lat='latitude', lon='longitude',
                           color='median_house_value',
                           size='median_house_value',
                           color_continuous_scale='viridis',
                           mapbox_style='open-street-map', size_max=10, zoom=10)
figure.show()


