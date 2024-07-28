import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# Load your dataset
df = pd.read_excel('C:/Users/mhatr/OneDrive/Desktop/python lab course/Jitesh Project/cleaned_imputed_data.xlsx')

# Remove all columns that are strings
data = df.select_dtypes(include=[np.number])

# Standardize the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Perform PCA
pca = PCA(n_components=3)
principal_components = pca.fit_transform(scaled_data)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# Create a DataFrame with the principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])

# Plot the 3D PCA results
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], c='r', marker='o', label='Scores')

# Plot the loadings (arrows)
for i, (x, y, z) in enumerate(loadings):
    ax.quiver(0, 0, 0, x, y, z, color='b', arrow_length_ratio=0.1)
    ax.text(x, y, z, data.columns[i], color='b')

ax.set_title('3D PCA Plot with Loadings')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2] * 100:.1f}%)')

# Plot 95% confidence ellipse (optional, needs further calculation)
# This is a simplified version and may not match exactly
# from matplotlib.patches import Ellipse
# angle = np.arctan2(pca_df['PC2'], pca_df['PC1'])
# width = 2 * np.sqrt(pca.explained_variance_ratio_[0])
# height = 2 * np.sqrt(pca.explained_variance_ratio_[1])
# ell = Ellipse(xy=(np.mean(pca_df['PC1']), np.mean(pca_df['PC2'])),
#               width=width, height=height,
#               edgecolor='r', fc='None', lw=2, label='95% Confidence Ellipse')
# ax.add_patch(ell)

ax.legend()
plt.show()
