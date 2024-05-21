import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os

root = os.path.dirname(os.path.abspath(__file__))

# Define the paths to the CSV files
filename_1 = os.path.join(root, "Data", "cleaned_data_toxygen.csv") 
filename_2 = os.path.join(root, "Data", "HateMoji_reduced.csv")
filename_3 = os.path.join(root, "Data", "cleaned_online-misoginy-eacl2021_reduced.csv")
filename_4 = os.path.join(root, "Data", "cleaned_Call_me_sexist_but_dataset_reduced.csv")
filename_5 = os.path.join(root, "Data", "cleaned_Detecting_Offensive_Statements_towards_Foreigners_in_Social_Media_reduced.csv")

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
cleaned_filename = os.path.join(root, 'Split Data', "cleaned_dataset.csv")
combined_df.to_csv(cleaned_filename, header=['text', 'label'], index=False) #save dataset

# Load the new combined dataset for further processing
combined_dataset3 = pd.read_csv(os.path.join(root, 'Split Data', "cleaned_dataset.csv"))  # Replace with the actual path

def check_csv_types(csv_file, output_file):
    # Read the CSV file into a pandas DataFrame, skipping the first row
    df = pd.read_csv(csv_file, skiprows=0)
    
    # Check if the first column contains only strings
    if 'text' in df.columns:
        text_dtype = df['text'].dtype
        if text_dtype == 'object':
            print("All values in the 'text' column are strings.")
        else:
            print(f"The 'text' column contains non-string values of type {text_dtype}.")
    else:
        print("The 'text' column is not found in the DataFrame.")
    
    # Check and clean the 'label' column
    if 'label' in df.columns:
        df['label'] = pd.to_numeric(df['label'], errors='coerce')
        df = df.dropna(subset=['label'])  # Remove rows with NaN in the 'label' column
        
        # Convert floats to integers
        df['label'] = df['label'].astype(int)
        
        labels_dtype = df['label'].dtype
        if labels_dtype == 'int64':
            print("All values in the 'label' column are integers.")
        else:
            print(f"The 'label' column contains non-integer values of type {labels_dtype}.")
            print("Non-integer values in 'label' column:", df['label'].unique())
    else:
        print("The 'label' column is not found in the DataFrame.")
    
    # Save the cleaned DataFrame to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Cleaned DataFrame saved to {output_file}")

# Check the data types in the cleaned dataset
check_csv_types(cleaned_filename, cleaned_filename)

# Separate the DataFrame into two arrays: text and label
array_text = combined_dataset3['text'].values  
array_label = combined_dataset3['label'].values 

# Function to remove 'nan' rows for string data
def remove_nan_rows(text_data, label_data):
    # Assuming 'nan' is a string
    is_nan_text = np.array([str(item).strip().lower() == 'nan' for item in text_data])
    is_nan_label = np.array([str(item).strip().lower() == 'nan' or str(item).strip().lower() == 'label'  for item in label_data])
    
    valid_indices = ~(is_nan_text | is_nan_label)
    
    cleaned_text_data = text_data[valid_indices]
    cleaned_label_data = label_data[valid_indices]
    
    return cleaned_text_data, cleaned_label_data

# Remove 'nan' rows from train and test datasets
array_text, array_label = remove_nan_rows(array_text, array_label)

# Split data
text_train, text_test, label_train, label_test = train_test_split(
    array_text, array_label, test_size=0.3, random_state=42
)

# Save the training and testing sets as .npy files
np.save(os.path.join(root, 'Split Data', 'text_train.npy'), text_train)
np.save(os.path.join(root, 'Split Data', 'text_test.npy'), text_test)
np.save(os.path.join(root, 'Split Data', 'label_train.npy'), label_train)
np.save(os.path.join(root, 'Split Data', 'label_test.npy'), label_test)






