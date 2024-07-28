
# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import optuna
import pandas as pd

# %%
file_path = '/Users/vincentkellerer/Desktop/PythonSeminar/Python Project/cleaned_imputed_data.csv'

# Load the dataset
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

# Define an improved neural network model
class ImprovedPyrolysisNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ImprovedPyrolysisNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.layer2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.layer3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.layer4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.layer5 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Adding dropout to prevent overfitting
        
    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.layer3(x)))
        x = self.dropout(x)
        x = self.relu(self.bn4(self.layer4(x)))
        x = self.layer5(x)
        return x

input_dim = X.shape[1]
output_dim = y.shape[1]
model = ImprovedPyrolysisNN(input_dim, output_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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



# %%

#optimisation via optuna

def objective(trial):
    # Suggest hyperparameters
    n_layers = trial.suggest_int('n_layers', 2, 5)
    hidden_size = trial.suggest_int('hidden_size', 64, 512)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3)

    class OptunaPyrolysisNN(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_size, n_layers, dropout_rate):
            super(OptunaPyrolysisNN, self).__init__()
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

    model = OptunaPyrolysisNN(input_dim, output_dim, hidden_size, n_layers, dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Training loop
    num_epochs = 50
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
    
    # Validation loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)
    val_loss /= len(val_loader.dataset)
    return val_loss

# Create a study and optimize the objective function
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print(f'Best trial: {study.best_trial.value}')
print(f'Best hyperparameters: {study.best_trial.params}')


# %%
best_params = study.best_trial.params

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

model = FinalPyrolysisNN(input_dim, output_dim, 
                         best_params['hidden_size'], 
                         best_params['n_layers'], 
                         best_params['dropout_rate'])

optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
criterion = nn.MSELoss()

# Train the final model
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

# Validate the final model
model.eval()
val_loss = 0.0
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        val_loss += loss.item() * X_batch.size(0)
val_loss /= len(val_loader.dataset)
print(f"Validation Loss: {val_loss:.4f}")

# Make predictions on training data
with torch.no_grad():
    y_train_pred = model(X_tensor).numpy()

# Calculate R^2
r2 = r2_score(y_tensor.numpy(), y_train_pred)
mae = mean_absolute_error(y_tensor.numpy(), y_train_pred)
rmse = np.sqrt(mean_squared_error(y_tensor.numpy(), y_train_pred))
print(f"R-Squared (R^2): {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")



