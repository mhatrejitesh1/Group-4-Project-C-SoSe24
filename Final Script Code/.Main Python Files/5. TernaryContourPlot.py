import plotly.figure_factory as ff 
import numpy as np 
import pandas as pd
import matplotlib as plt
from sklearn import preprocessing

path = 'cleaned_imputed_data.xlsx'

df = pd.read_excel(path)
df.columns = df.columns.str.strip()

Bio_Oil = ['Reaction temperature (°C)',
 'Microwave power (W)',
 'Reaction time (min)',
 'Bio-oil yield (%)']

Syn_Gas = ['Reaction temperature (°C)',
 'Microwave power (W)',
 'Reaction time (min)',
 'Syngas yield (%)']

Bio_Char = ['Reaction temperature (°C)',
 'Microwave power (W)',
 'Reaction time (min)',
 'Biochar yield (%)']

Bio_Oil_data = df[Bio_Oil]
Syn_Gas_Data = df[Syn_Gas]
Bio_Char_Data = df[Bio_Char]

def generate_ternary_contour(data,title,file_name):

	data_numpy = data.to_numpy()

	# barycentric coords: (a,b,c) 
	a = data_numpy[:,0]
	b = data_numpy[:,1]
	c = data_numpy[:,2]

	total = a + b + c
	a_normalized = a / total
	b_normalized = b / total
	c_normalized = c / total

	# values is stored in the last column 
	v = data_numpy[:,3]

	fig = ff.create_ternary_contour(
    [a_normalized, b_normalized, c_normalized], v,
    pole_labels=['Reaction temperature (°C)', 'Microwave power (W)', 'Reaction time (min)'],
    ncontours=20,
    showscale=True,
	colorscale='Viridis',
    title=f'Ternary Contour Plot {title}'	
	)
 
	fig.write_image(file_name)
	fig.show()


generate_ternary_contour(Bio_Char_Data,'hello','hello.png')