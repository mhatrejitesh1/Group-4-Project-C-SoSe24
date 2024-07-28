import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import shap
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/Users/vincentkellerer/Desktop/PythonSeminar/Python Project/cleaned_imputed_data.csv'
df = pd.read_csv(file_path)

# Assuming the input columns are from column index 2 to 17 (modify as needed)
input_columns = df.columns[2:17]
output_columns = df.columns[17:]

# Convert the dataframe to numpy arrays
X = df[input_columns].values
y = df[output_columns].values

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Prepare the dataset and dataloader
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class FinalPyrolysisNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, n_layers, dropout_rate):
        super(FinalPyrolysisNN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_size, output_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Example best hyperparameters from Optuna study (replace with actual best params)
best_params = {
    'hidden_size': 256,
    'n_layers': 3,
    'dropout_rate': 0.3,
    'learning_rate': 0.001
}

input_dim = X.shape[1]
output_dim = y.shape[1]
model = FinalPyrolysisNN(input_dim, output_dim, 
                         best_params['hidden_size'], 
                         best_params['n_layers'], 
                         best_params['dropout_rate'])

optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
criterion = nn.MSELoss()

# Train the model
num_epochs = 200

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Validate the model
model.eval()
val_loss = 0.0
with torch.no_grad():
    y_true, y_pred = [], []
    for X_batch, y_batch in val_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        val_loss += loss.item() * X_batch.size(0)
        y_true.append(y_batch.numpy())
        y_pred.append(outputs.numpy())
    
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    val_loss /= len(val_loader.dataset)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"Validation Loss: {val_loss:.4f}")
print(f"R-Squared (R^2): {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Perform SHAP analysis
explainer = shap.DeepExplainer(model, X_tensor)
shap_values = explainer.shap_values(X_tensor)

# Plot the SHAP summary plot
plt.figure()
shap.summary_plot(shap_values, X_tensor.numpy(), feature_names=input_columns)
plt.savefig('shap_summary_plot.png')

# Plot the SHAP dependence plot for the first feature
plt.figure()
shap.dependence_plot(0, shap_values, X_tensor.numpy(), feature_names=input_columns)
plt.savefig('shap_dependence_plot_feature_0.png')

# Force plot for the first prediction
plt.figure()
shap.force_plot(explainer.expected_value[0], shap_values[0][0], X_tensor.numpy()[0], feature_names=input_columns, matplotlib=True)
plt.savefig('shap_force_plot_feature_0.png')

# Display the plots
plt.show()
