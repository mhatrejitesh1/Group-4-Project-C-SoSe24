import pandas as pd
import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt

path = 'cleaned_imputed_data.xlsx'

# Create df
df = pd.read_excel(path)

# Ensure column names are correct
df.columns = df.columns.str.strip()

# gather the column names as str for next step
column_names = df.columns.tolist()

Bio_Oil_Columns = ['Carbon content (wt%)',
 'Hydrogen content (wt%)',
 'Nitrogen content (wt%)',
 'Oxygen content (wt%)',
 'Sulfur content (wt%)',
 'Volatile matter (wt%)',
 'Fixed carbon (wt%)',
 'Ash content (wt%)',
 'Reaction temperature (°C)',
 'Microwave power (W)',
 'Reaction time (min)',
 'Microwave absorber percentage (%)',
 'Dielectric constant of absorber (ε′)',
 'Dielectric loss factor of absorber (ε“)',
 'Bio-oil yield (%)',]

Syn_Gas_Columns = ['Carbon content (wt%)',
 'Hydrogen content (wt%)',
 'Nitrogen content (wt%)',
 'Oxygen content (wt%)',
 'Sulfur content (wt%)',
 'Volatile matter (wt%)',
 'Fixed carbon (wt%)',
 'Ash content (wt%)',
 'Reaction temperature (°C)',
 'Microwave power (W)',
 'Reaction time (min)',
 'Microwave absorber percentage (%)',
 'Dielectric constant of absorber (ε′)',
 'Dielectric loss factor of absorber (ε“)',
 'Syngas yield (%)',
 'Syngas composition (H₂, mol%)',
 'Syngas composition (CH₄, mol%)',
 'Syngas composition (CO₂, mol%)',
 'Syngas composition (CO, mol%)',]

Bio_Char_Columns = ['Carbon content (wt%)',
 'Hydrogen content (wt%)',
 'Nitrogen content (wt%)',
 'Oxygen content (wt%)',
 'Sulfur content (wt%)',
 'Volatile matter (wt%)',
 'Fixed carbon (wt%)',
 'Ash content (wt%)',
 'Reaction temperature (°C)',
 'Microwave power (W)',
 'Reaction time (min)',
 'Microwave absorber percentage (%)',
 'Dielectric constant of absorber (ε′)',
 'Dielectric loss factor of absorber (ε“)',
 'Biochar yield (%)',
 'Biochar calorific value (MJ/kg)',
 'Biochar H/C ratio (-)',
 'Biochar H/N ratio (-)',
 'Biochar O/C ratio (-)']


Bio_Oil_data = df[Bio_Oil_Columns]
Syn_Gas_Data = df[Syn_Gas_Columns]
Bio_Char_Data = df[Bio_Char_Columns]

def create_spearman_heatmap(df, title_name, file_name):
   
    #put data in correlation matrix eg spearman or pearson
    correlation_matrix = df.corr(method='spearman')
    #generate maskt to make a triangle shaped matrix
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    #figure size
    plt.figure(figsize=(16, 9))
    #genaerate heatmap from correlation matrix and set colour, initialize the temperature bat etc.
    heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-0.5, vmax=0.5, mask=mask, linewidths=0.5, linecolor="White", cbar_kws={"shrink": .8})
    #title
    heatmap.set_title(f'Spearman Heatmap {title_name}', fontdict={'fontsize':20}, pad=12)
    #turn xaxis so make it better readable
    plt.xticks(rotation=45, ha='right')
    #makes better layout
    plt.tight_layout()
    #saves file as file name
    plt.savefig(file_name, dpi=500, bbox_inches='tight')
    plt.show()

create_spearman_heatmap(Bio_Oil_data, 'Bio Oil Data', 'Bio_Oil_Spearman_Heatmap.png')
create_spearman_heatmap(Syn_Gas_Data, 'Syn-Gas Data', 'Syn_Gas_Spearman_Heatmap.png')
create_spearman_heatmap(Bio_Char_Data, 'Bio Char Data', 'Bio_Char_Spearman_Heatmap.png')

