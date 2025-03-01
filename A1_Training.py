import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import math

# Load the Dataset
data = pd.read_csv("detailed_meals_macros_three_disease.csv")


# Prepare Inputs and Outputs
X = data[['Heart Disease Risk', 'Diabetes Risk', 'Kidney Disease Risk']].values
y = data[['Protein', 'Carbohydrates', 'Fat']].values
N_outputs = y.shape[1]

# Standardize the outputs
scaler = StandardScaler()
y_scaled = scaler.fit_transform(y)

# Cosine Annealing Learning Rate Scheduler
def cosine_annealing(epoch, lr):
    min_lr = 1e-5
    max_lr = 1e-3
    return min_lr + (max_lr - min_lr) * (1 + math.cos(math.pi * epoch / 50)) / 2

# Mean Absolute Percentage Error (MAPE) Function
def mean_absolute_percentage_error(y_true, y_pred):
    epsilon = 1e-5  # Avoid division by zero
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

# 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []
fold = 1

for train_index, test_index in kf.split(X):
    print(f"Training Fold {fold}...")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_scaled[train_index], y_scaled[test_index]
    
    # Define the Model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(3,)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(N_outputs, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = LearningRateScheduler(cosine_annealing)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=1000,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stop, lr_scheduler],
        verbose=1
    )
    
    # Evaluate the Model
    y_test_pred_scaled = model.predict(X_test)
    y_test_pred = scaler.inverse_transform(y_test_pred_scaled)
    y_test_true = scaler.inverse_transform(y_test)
    
    mse = mean_squared_error(y_test_true, y_test_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_true, y_test_pred)
    mape = mean_absolute_percentage_error(y_test_true, y_test_pred)
    r2 = r2_score(y_test_true, y_test_pred)
    fold_results.append((mse, rmse, mae, mape, r2))
    
    print(f"Fold {fold} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%, R² Score: {r2:.4f}")
    fold += 1

# Compute Mean Results Across Folds
avg_mse = np.mean([res[0] for res in fold_results])
avg_rmse = np.mean([res[1] for res in fold_results])
avg_mae = np.mean([res[2] for res in fold_results])
avg_mape = np.mean([res[3] for res in fold_results])
avg_r2 = np.mean([res[4] for res in fold_results])
print(f"Average Results - MSE: {avg_mse:.4f}, RMSE: {avg_rmse:.4f}, MAE: {avg_mae:.4f}, MAPE: {avg_mape:.2f}%, R² Score: {avg_r2:.4f}")

# Train and Validation Loss Plots
plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.show()

# Save the final model
