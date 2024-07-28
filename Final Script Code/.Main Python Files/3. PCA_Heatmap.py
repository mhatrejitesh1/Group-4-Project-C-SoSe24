import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def create_PCA_heatmap(path, title_name, file_name):
    # Read the data from the Excel file
    correlation_matrix = pd.read_excel(path, index_col=0)
    
    # Generate mask to make a triangle shaped matrix
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Figure size
    plt.figure(figsize=(16, 9))
    
    # Generate heatmap from correlation matrix and set color, initialize the temperature bar, etc.
    heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-0.5, vmax=0.5, mask=mask, linewidths=0.5, linecolor="White", cbar_kws={"shrink": .8})
    
    heatmap.set_title(f'PCA {title_name}', fontdict={'fontsize':20}, pad=12)
    #turn xaxis so make it better readable
    plt.xticks(rotation=45, ha='right')
    #makes better layout
    plt.tight_layout()
    #saves file as file name
    plt.savefig(file_name, dpi=500, bbox_inches='tight')
    plt.show()


create_PCA_heatmap('/Users/vincentkellerer/Desktop/PCA/A) Bio oil/bio oil heatmap values.xlsx', 'Bio Oil', 'PCA_Bio_oil.png')
create_PCA_heatmap('/Users/vincentkellerer/Desktop/PCA/B) Syngas/Syngas heatmap values.xlsx', 'Syngas', 'PCA_Syngas.png')
create_PCA_heatmap('/Users/vincentkellerer/Desktop/PCA/C) Bio char/bio char heatmap values.xlsx', 'Bio Char', 'PCA_Bio_char.png')