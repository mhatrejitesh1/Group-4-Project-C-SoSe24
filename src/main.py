import pandas as pd
import numpy as np
pip install scikit-learn
pip install openpyxl
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Load the dataset
file_path = 'C:/Users/mhatr/OneDrive/Desktop/python lab course/Jitesh Project/MGT001437_ProjectC/data.xlsx'  # Replace this with the correct file path
data = pd.read_excel(file_path)

# Set appropriate column names using the second row and remove the first two rows
data.columns = data.iloc[1]
data = data.drop([0, 1])

# Reset the index and remove columns that are completely empty or unnamed
data = data.reset_index(drop=True)
data.columns = data.columns.str.strip()
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Identify columns that should be numeric
numeric_columns = [
    'Carbon content (wt%)', 'Hydrogen content (wt%)', 'Nitrogen content (wt%)',
    'Oxygen content (wt%)', 'Sulfur content (wt%)', 'Volatile matter (wt%)',
    'Fixed carbon (wt%)', 'Ash content (wt%)', 'Reaction temperature (°C)',
    'Microwave power (W)', 'Reaction time (min)', 'Microwave absorber percentage (%)',
    'Dielectric constant of absorber (ε′)', 'Dielectric loss factor of absorber (ε“)',
    'Bio-oil yield (%)', 'Syngas yield (%)', 'Syngas composition (H₂, mol%)',
    'Syngas composition (CH₄, mol%)', 'Syngas composition (CO₂, mol%)',
    'Syngas composition (CO, mol%)', 'Biochar yield (%)', 'Biochar calorific value (MJ/kg)',
    'Biochar H/C ratio (-)', 'Biochar H/N ratio (-)', 'Biochar O/C ratio (-)'
]

# Convert identified columns to numeric, coercing errors
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Separate numeric and non-numeric data
numeric_data = data[numeric_columns]
non_numeric_data = data.drop(columns=numeric_columns)

# Debugging: Print the structure of numeric_data
print("Numeric data columns:", numeric_data.columns)
print("Numeric data head:\n", numeric_data.head())
print("Non-numeric data columns:", non_numeric_data.columns)
print("Non-numeric data head:\n", non_numeric_data.head())

# Check if numeric_data is empty
if numeric_data.empty:
    raise ValueError("No numeric columns found in the dataset to impute.")

# Initialize the IterativeImputer
imputer = IterativeImputer(random_state=0)

# Impute missing values in the numeric data
imputed_data = imputer.fit_transform(numeric_data)

# Convert the imputed numpy array back to a DataFrame with appropriate column names
imputed_df = pd.DataFrame(imputed_data, columns=numeric_data.columns)

# Concatenate the imputed numeric data with the non-numeric columns
final_data = pd.concat([non_numeric_data.reset_index(drop=True), imputed_df.reset_index(drop=True)], axis=1)

# Display the first few rows of the final cleaned and imputed dataset
print(final_data.head())

# Save the cleaned and imputed dataset to a new Excel file
final_data.to_excel('cleaned_imputed_data.xlsx', index=False)
