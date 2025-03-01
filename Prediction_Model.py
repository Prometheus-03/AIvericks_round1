import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

data_path = "/kaggle/input/llcp2023xpt/LLCP2023.XPT"
df = pd.read_sas(data_path, format='xport', encoding="utf-8")
print("dataframe shape:", df.shape)


'''
Independent variables:
_AGE80: age
_SEX: gender
_BMI5: body mass index
EXEROFT1: number of physical activities within past month (integer)

Dependent variables:
DIABETE4: whether person has diabetes
_MICHD: whether person has heart disease
CHCKDNY2: whether person has kidney disease
'''
selected_cols = ['_AGE80', '_SEX', '_BMI5', 'EXEROFT1', 'DIABETE4', '_MICHD', 'CHCKDNY2']

df = df[selected_cols].copy()

rename_dict = {
    '_AGE80': 'age',
    '_SEX': 'gender',
    '_BMI5': 'bmi',
    'EXEROFT1': 'physical_activity',
    
    'DIABETE4': 'diabetes',
    '_MICHD': 'heart_disease',
    'CHCKDNY2': 'kidney_disease'
}
df.rename(columns=rename_dict, inplace=True)
print("renamed columns:", df.columns.tolist())

df.dropna(inplace=True)
print("dataframe shape:", df.shape)


# recode physical_activity to number of times activity done in a month
df['physical_activity'] = df['physical_activity'].apply(lambda x: 0 if x <= 100 or x >= 300 else x)
df['physical_activity'] = df['physical_activity'].apply(lambda x: (x - 100) * 4 if x > 100 and x < 200 else (x - 200 if x > 200 and x < 300 else x))

# drop unusually large values
df = df[df['physical_activity'] <= 90]

# show range
print(df['physical_activity'].describe())


# recode physical_activity to a categorical variable
'''
1 - sedentary
2 - lightly active
3 - moderately active
4 - very active
'''
def active_numerical(x):
    if x < 4: return 1
    elif x < 12: return 2
    elif x < 24: return 3
    else: return 4

df['physical_activity'] = df['physical_activity'].apply(lambda x: active_numerical(x))

# show range
print(df['physical_activity'].describe())


# recode diabetes variable from survey results: 1 indicates having diabetes, otherwise 0
df['diabetes'] = df['diabetes'].apply(lambda x: 0 if x == 3 or x == 4 else x)

# drop columns with other values
df = df[df['diabetes'] <= 1]

print("dataframe shape:", df.shape)


# recode heart_disease variable from survey results: 1 indicates having heart disease, otherwise 0
df['heart_disease'] = df['heart_disease'].apply(lambda x: 0 if x == 2 else x)

print("dataframe shape:", df.shape)


# drop other values
df = df[df['kidney_disease'] <= 2]

# recode kidney_disease variable from survey results: 1 indicates having kidney disease, otherwise 0
df['kidney_disease'] = df['kidney_disease'].apply(lambda x: 0 if x == 2 else x)

print("dataframe shape:", df.shape)

# shows statistics of those with either disease
print(df[df['diabetes'] + df['heart_disease'] + df['kidney_disease'] > 0].describe())


# normalize age and bmi
scaler = MinMaxScaler()
df[['age', 'bmi']] = scaler.fit_transform(df[['age', 'bmi']])
print(df.head())

# set independent and dependent variables
X = df[['age', 'bmi', 'gender', 'physical_activity']]
y1 = df['diabetes']
y2 = df['heart_disease']
y3 = df['kidney_disease']

# split into training and testing
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y1, test_size=0.2, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y2, test_size=0.2, random_state=42)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y3, test_size=0.2, random_state=42)
print("train1 shape:", X_train1.shape, "test1 shape:", X_test1.shape)
print("train2 shape:", X_train2.shape, "test2 shape:", X_test2.shape)
print("train3 shape:", X_train3.shape, "test3 shape:", X_test3.shape)


# create regression model1 for diabetes
model1 = Sequential([
    Dense(64, activation='relu', input_dim=X_train1.shape[1]),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model1.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
model1.summary()

# train model1
history1 = model1.fit(X_train1, y_train1, epochs=50, batch_size=32, validation_split=0.1)

# evaluate model1
loss1, mse1 = model1.evaluate(X_test1, y_test1)
print(f"Test1 Mean Squared Error: {mse1:.4f}")


# create regression model2 for heart disease
model2 = Sequential([
    Dense(64, activation='relu', input_dim=X_train2.shape[1]),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model2.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
model2.summary()

# train model2
history2 = model2.fit(X_train2, y_train2, epochs=50, batch_size=32, validation_split=0.1)

# evaluate model2
loss2, mse2 = model2.evaluate(X_test2, y_test2)
print(f"Test2 Mean Squared Error: {mse2:.4f}")


# create regression model3 for kidney disease
model3 = Sequential([
    Dense(64, activation='relu', input_dim=X_train3.shape[1]),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model3.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
model3.summary()

# train model
history3 = model3.fit(X_train3, y_train3, epochs=50, batch_size=32, validation_split=0.1)

# evaluate model
loss3, mse3 = model3.evaluate(X_test3, y_test3)
print(f"Test3 Mean Squared Error: {mse3:.4f}")


plt.figure(figsize=(10, 4))
plt.plot(history1.history['mse'], label='Train1 Mean Squared Error')
plt.plot(history1.history['val_mse'], label='Validation1 Mean Squared Error')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Training and Validation1 Mean Squared Error')
plt.legend()
plt.show()

# make predictions
predictions1 = model1.predict(X_test1)
print(X_test1[:10])
print(predictions1[:10])


plt.figure(figsize=(10, 4))
plt.plot(history2.history['mse'], label='Train2 Mean Squared Error')
plt.plot(history2.history['val_mse'], label='Validation2 Mean Squared Error')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Training and Validation2 Mean Squared Error')
plt.legend()
plt.show()

# make predictions
predictions2 = model2.predict(X_test2)
print(X_test2[:10])
print(predictions2[:10])


plt.figure(figsize=(10, 4))
plt.plot(history3.history['mse'], label='Train3 Mean Squared Error')
plt.plot(history3.history['val_mse'], label='Validation3 Mean Squared Error')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Training and Validation3 Mean Squared Error')
plt.legend()
plt.show()

# make predictions
predictions3 = model3.predict(X_test3)
print(X_test3[:10])
print(predictions3[:10])


input_data = [[0.50000, 0.40000, 2.0, 0.0],
          [1.000000, 0.176855, 2.0, 8.0],
          [0.822581, 0.271705, 1.0, 16.0]]

input_test = pd.DataFrame(input_data, columns = ['age', 'bmi', 'gender', 'physical_activity'])
print(input_test)

# Make predictions on the test set
input_prediction1 = model1.predict(input_test)
input_prediction2 = model2.predict(input_test)
input_prediction3 = model3.predict(input_test)
print(input_prediction1)
print(input_prediction2)
print(input_prediction3)


# Format new test data from file
data_path = "/kaggle/input/biodata/diabetic_input_set.xlsx"
bd = pd.read_excel(data_path, sheet_name="detailed_meals_macros_")
print("dataframe shape:", bd.shape)


selected_cols = ['Ages', 'Gender', 'BMI', 'Activity Level']

bd = bd[selected_cols].copy()

rename_bd_dict = {
    'Ages': 'age',
    'Gender': 'gender',
    'BMI': 'bmi',
    'Activity Level': 'physical_activity'
}

bd.rename(columns=rename_bd_dict, inplace=True)
print("renamed columns:", bd.columns.tolist())

bd.dropna(inplace=True)
print("dataframe shape:", bd.shape)


# format bd to match implied 2 dp
bd['bmi'] = bd['bmi'] * 100

# use scaler from training data
bd[['age', 'bmi']] = scaler.transform(bd[['age', 'bmi']])
print(bd.head())


print(df['bmi'].describe())
print(bd['bmi'].describe())


def gender(x):
    if x == "Male": return 1
    else: return 2

def activity_categorical(x):
    if x == "Sedentary": return 1
    elif x == "Lightly Active": return 2
    elif x == "Moderately Active": return 3
    else: return 4

bd['gender'] = bd['gender'].apply(lambda x: gender(x))
bd['physical_activity'] = bd['physical_activity'].apply(lambda x: activity_categorical(x))
print(bd.head())


normalized_test = bd
print(normalized_test)

# Make predictions on the test set
bd_prediction1 = model1.predict(normalized_test)
bd_prediction2 = model2.predict(normalized_test)
bd_prediction3 = model3.predict(normalized_test)

print(bd_prediction1)
print(bd_prediction2)
print(bd_prediction3)


output_df1 = pd.DataFrame(bd_prediction1)
output_df2 = pd.DataFrame(bd_prediction2)
output_df3 = pd.DataFrame(bd_prediction3)

print(output_df1.describe())
print(output_df2.describe())
print(output_df3.describe())

output_df1.to_excel('diabetes_risk.xlsx', index=False, header='diabetes_risk')
output_df2.to_excel('heart_disease_risk.xlsx', index=False, header='heart_disease_risk')
output_df3.to_excel('kidney_disease_risk.xlsx', index=False, header='kidney_disease_risk')


model1.save('model1.h5')
model2.save('model2.h5')
model3.save('model3.h5')