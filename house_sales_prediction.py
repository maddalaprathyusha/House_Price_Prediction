import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Loading the training and testing  datasets
train_df = pd.read_csv('housing_train.csv')
test_df = pd.read_csv('housing_test.csv')

# Preprocess the data
X_train = train_df.drop(columns=['houseID', 'price'])
y_train = train_df['price']
X_test = test_df.drop(columns=['houseID'])

# Encoding  the categorical variables
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Scale the numerical variables
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Selecting  the most relevant features using RFE
model = LinearRegression()
rfe = RFE(model, n_features_to_select=10)
rfe = rfe.fit(X_train, y_train)
X_train = rfe.transform(X_train)
X_test = rfe.transform(X_test)

# Training  the model
model.fit(X_train, y_train)

# Make predictions on the test dataset
predictions = model.predict(X_test)

# Saving in file
predictions_df = pd.DataFrame({'houseID': test_df['houseID'], 'predicted': predictions})

predictions_df.to_csv('predictions.csv', index=False)












