import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


# Define the paths to the CSV files
filename_1 = '/Users/ninabodenstab/Desktop/University/EPFL/Ma2/Deep Learning/Projet/cleaned_data_toxygen.csv' 
filename_2 = '/Users/ninabodenstab/Desktop/University/EPFL/Ma2/Deep Learning/Projet/HateMoji_reduced.csv'  
filename_3 = '/Users/ninabodenstab/Desktop/University/EPFL/Ma2/Deep Learning/Projet/cleaned_online-misoginy-eacl2021_reduced.csv'
filename_4 = '/Users/ninabodenstab/Desktop/University/EPFL/Ma2/Deep Learning/Projet/cleaned_Call_me_sexist_but_dataset_reduced.csv'
filename_5 = '/Users/ninabodenstab/Desktop/University/EPFL/Ma2/Deep Learning/Projet/cleaned_Detecting_Offensive_Statements_towards_Foreigners_in_Social_Media_reduced.csv'

#
# Load the CSV files into DataFrames, skipping the first row (header row)
df1 = pd.read_csv(filename_1, header=None, skiprows=1)
df2 = pd.read_csv(filename_2, header=None, skiprows=1)
df3 = pd.read_csv(filename_3, header=None, skiprows=1)
df4 = pd.read_csv(filename_4, header=None, skiprows=1)
df5 = pd.read_csv(filename_5, header=None, skiprows=1)

# Combine the DataFrames
combined_df = pd.concat([df1, df2, df3, df4, df5])

# Separate the DataFrame into two arrays: text and label
array_text = combined_df[0].values  # First column contains text data
array_label = combined_df[1].values  # Second column contains label data

# Specify the absolute path to save the cleaned dataset
cleaned_filename = '/Users/ninabodenstab/Desktop/University/EPFL/Ma2/Deep Learning/Projet/combined_dataset2.csv'
combined_df.to_csv(cleaned_filename, header=['text', 'label'], index=False) #save dataset

# Load the new combined dataset for further processing
combined_dataset3 = pd.read_csv('/Users/ninabodenstab/Desktop/University/EPFL/Ma2/Deep Learning/Projet/cleaned_dataset3.csv')  # Replace with the actual path

# Separate the DataFrame into two arrays: text and label
array_text = combined_dataset3['text'].values  
array_label = combined_dataset3['label'].values  

# Split data
text_train, text_test, label_train, label_test = train_test_split(
    array_text, array_label, test_size=0.3, random_state=42
)

# Save the training and testing sets as .npy files
np.save('/Users/ninabodenstab/Desktop/University/EPFL/Ma2/Deep Learning/Projet/text_train3.npy', text_train)
np.save('/Users/ninabodenstab/Desktop/University/EPFL/Ma2/Deep Learning/Projet/text_test3.npy', text_test)
np.save('/Users/ninabodenstab/Desktop/University/EPFL/Ma2/Deep Learning/Projet/label_train3.npy', label_train)
np.save('/Users/ninabodenstab/Desktop/University/EPFL/Ma2/Deep Learning/Projet/label_test3.npy', label_test)




