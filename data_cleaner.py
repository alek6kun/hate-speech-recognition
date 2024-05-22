import os, sys
import numpy as np
import pandas as pd
import re
from sklearn.utils import resample

root = os.path.dirname(os.path.abspath(__file__))
root_data = os.path.join(root, 'Data')

######################## call me sexist ###########################
# Function to clean text by removing <"MENTION" numbers>, hashtags, URLs, and "RT"
def clean_text_call_me_sexist(text):
    # Remove <"MENTION" numbers>
    text = re.sub(r'\bMENTION\d+\b', '', text)
    # Remove hashtags (e.g., #something)
    text = re.sub(r'#\S+', '', text)
    # Remove URLs (e.g., http:/something)
    text = re.sub(r'http\S+', '', text)
    # Remove "RT"
    text = re.sub(r'\bRT\b', '', text)
    return text.strip()

# Load the CSV file into a DataFrame
input_file = os.path.join(root_data, 'Call_me_sexist_but_dataset_reduced.csv')
df = pd.read_csv(input_file, header=None, names=['text', 'label'])

# Apply the clean_text function to the 'text' column
df['text'] = df['text'].apply(clean_text_call_me_sexist)

# Remove empty rows
df = df[df['text'].astype(bool)]

# Save the cleaned data back to a new CSV file
output_file = os.path.join(root_data, 'cleaned_Call_me_sexist_but_dataset_reduced.csv')
df.to_csv(output_file, index=False, header=False)

###################### offensive statements (German) ########################3
# Function to clean text by removing hex IDs, replacing with "Peter", and removing URLs
def clean_text_offensive_statements(text):
    # Replace hex IDs with "Peter"
    text = re.sub(r'\b[a-fA-F0-9]{16}\b', 'Peter', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    return text.strip()

# Load the CSV file into a DataFrame
input_file = os.path.join(root_data, 'Detecting_Offensive_Statements_towards_Foreigners_in_Social_Media_reduced.csv')

try:
    df = pd.read_csv(input_file, header=None, names=['text', 'label'], sep=',', encoding='utf-8', on_bad_lines='skip', engine='python')
except pd.errors.ParserError as e:
    print(f"Error parsing CSV file: {e}")
    raise

# Apply the clean_text function to the 'text' column
df['text'] = df['text'].apply(clean_text_offensive_statements)

# Remove empty rows
df = df[df['text'].astype(bool)]

# Save the cleaned data back to a new CSV file
output_file = os.path.join(root_data, 'cleaned_Detecting_Offensive_Statements_towards_Foreigners_in_Social_Media_reduced.csv')
df.to_csv(output_file, index=False, header=False)

########################## misoginy ##########################
# Function to clean text by removing URLs, short lines, lines starting with '>', and lines containing '[deleted]' or '[removed]'
def clean_text_misoginy(text):
    # Convert non-string values to empty string
    if not isinstance(text, str):
        return ''
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove lines starting with '>'
    text = re.sub(r'^>', '', text, flags=re.MULTILINE)
    # Remove lines containing '[deleted]' or '[removed]'
    text = re.sub(r'\[deleted\]|\[removed\]', '', text)
    # Remove leading and trailing whitespace
    text = text.strip()
    return text

# Load the CSV file into a DataFrame
input_file = os.path.join(root_data, 'online-misoginy-eacl2021_reduced.csv')

try:
    df = pd.read_csv(input_file, header=None, names=['text', 'label'], sep=',', encoding='utf-8', engine='python')
except pd.errors.ParserError as e:
    print(f"Error parsing CSV file: {e}")
    raise

# Apply the clean_text function to the 'text' column
df['text'] = df['text'].apply(clean_text_misoginy)
df = df[df['text'].str.len() >= 5]

# Save the cleaned data back to a new CSV file
output_file = os.path.join(root_data, 'cleaned_online-misoginy-eacl2021_reduced.csv')
df.to_csv(output_file, index=False, header=False)

#################### TOXYGEN #################
# Function to clean text by removing unwanted patterns
def clean_text_toxygen(text):
    # Remove 'b" ... "' and '\n-'
    text = re.sub(r'b\"\"|\\n-', '', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove repetitions of more than 3 of the same character
    text = re.sub(r'(\w)\1{3,}', r'\1\1\1', text)
    # Remove anything within square brackets
    text = re.sub(r'\[.*?\]', '', text)
    # Remove leading and trailing whitespace
    text = text.strip()
    return text

# Load the CSV file into a DataFrame
input_file = os.path.join(root_data, 'data_toxygen.csv')

try:
    df = pd.read_csv(input_file, header=None, names=['text', 'label'], sep=',', encoding='utf-8', engine='python')
except pd.errors.ParserError as e:
    print(f"Error parsing CSV file: {e}")
    raise

# Apply the clean_text function to the 'text' column
df['text'] = df['text'].apply(clean_text_toxygen)
df = df[df['text'].str.len() >= 5]

# Save the cleaned data back to a new CSV file
output_file =  os.path.join(root_data, 'cleaned_data_toxygen.csv')
df.to_csv(output_file, index=False, header=False)

################## fr_dataset ######################

def clean_text(text):
    # Replace hex IDs with "Peter"
    text = re.sub(r'\b[a-fA-F0-9]{16}\b', 'Peter', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove @user and @url instances
    text = re.sub(r'@user', '', text)
    text = re.sub(r'@url', '', text)
    # Remove hashtags and related words
    text = re.sub(r'#\w+', '', text)
    # Remove "rt" instances
    text = re.sub(r'\brt\b', '', text, flags=re.IGNORECASE)
    return text.strip()

input_file = os.path.join(root_data, 'fr_dataset_reduced.csv')

try:
    df = pd.read_csv(input_file, header=None, names=['text', 'label'], sep=',', encoding='utf-8', on_bad_lines='skip', engine='python')
except pd.errors.ParserError as e:
    print(f"Error parsing CSV file: {e}")
    raise

# Replace incorrect labels
df['label'] = df['label'].replace({'hateful': 1, 'normal': 0})

# Remove rows with any other labels
df = df[df['label'].isin([0, 1])]
# Remove rows where 'text' is NaN
df = df.dropna(subset=['text'])
df['text'] = df['text'].apply(clean_text)

output_file = os.path.join(root_data, 'cleaned_fr_dataset_reduced.csv')
# Save the cleaned dataset (optional)
df.to_csv(output_file, index=False)

print("Dataset cleaned and saved successfully!")

################## ar_dataset ######################

def clean_text(text):
    # Replace hex IDs with "Peter"
    text = re.sub(r'\b[a-fA-F0-9]{16}\b', 'Peter', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove @user and @url instances
    text = re.sub(r'@user', '', text)
    text = re.sub(r'@url', '', text)
    # Remove hashtags and related words
    text = re.sub(r'#\w+', '', text)
    # Remove "rt" instances
    text = re.sub(r'\brt\b', '', text, flags=re.IGNORECASE)
    return text.strip()

input_file = os.path.join(root_data, 'ar_dataset_reduced.csv')

try:
    df = pd.read_csv(input_file, header=None, names=['text', 'label'], sep=',', encoding='utf-8', on_bad_lines='skip', engine='python')
except pd.errors.ParserError as e:
    print(f"Error parsing CSV file: {e}")
    raise

# Replace incorrect labels
df['label'] = df['label'].replace({'hateful': 1, 'normal': 0})

# Remove rows with any other labels
df = df[df['label'].isin([0, 1])]
# Remove rows where 'text' is NaN
df = df.dropna(subset=['text'])
df['text'] = df['text'].apply(clean_text)

output_file = os.path.join(root_data, 'cleaned_ar_dataset_reduced.csv')
# Save the cleaned dataset (optional)
df.to_csv(output_file, index=False)

print("Dataset cleaned and saved successfully!")

################## en_dataset ######################

def clean_text(text):
    # Replace hex IDs with "Peter"
    text = re.sub(r'\b[a-fA-F0-9]{16}\b', 'Peter', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove @user and @url instances
    text = re.sub(r'@user', '', text)
    text = re.sub(r'@url', '', text)
    # Remove hashtags and related words
    text = re.sub(r'#\w+', '', text)
    # Remove "rt" instances
    text = re.sub(r'\brt\b', '', text, flags=re.IGNORECASE)
    return text.strip()

input_file = os.path.join(root_data,'en_dataset_reduced.csv' )
try:
    df = pd.read_csv(input_file, header=None, names=['text', 'label'], sep=',', encoding='utf-8', on_bad_lines='skip', engine='python')
except pd.errors.ParserError as e:
    print(f"Error parsing CSV file: {e}")
    raise

# Replace incorrect labels
df['label'] = df['label'].replace({'hateful': 1, 'normal': 0})

# Remove rows with any other labels
df = df[df['label'].isin([0, 1])]
# Remove rows where 'text' is NaN
df = df.dropna(subset=['text'])
df['text'] = df['text'].apply(clean_text)

output_file = os.path.join(root_data, 'cleaned_en_dataset_reduced.csv')
# Save the cleaned dataset (optional)
df.to_csv(output_file, index=False)

print("Dataset cleaned and saved successfully!")

from sklearn.model_selection import train_test_split
root = os.path.dirname(os.path.abspath(__file__))

# Define the paths to the CSV files
filename_1 = os.path.join(root, "Data", "cleaned_data_toxygen.csv") 
#filename_2 = os.path.join(root, "Data", "HateMoji_reduced.csv")
filename_3 = os.path.join(root, "Data", "cleaned_online-misoginy-eacl2021_reduced.csv")
filename_4 = os.path.join(root, "Data", "cleaned_Call_me_sexist_but_dataset_reduced.csv")
filename_5 = os.path.join(root, "Data", "cleaned_Detecting_Offensive_Statements_towards_Foreigners_in_Social_Media_reduced.csv")
filename_6 = os.path.join(root, "Data", "cleaned_fr_dataset_reduced.csv")
filename_7 = os.path.join(root, "Data", "cleaned_ar_dataset_reduced.csv")
filename_8 = os.path.join(root, "Data", "cleaned_en_dataset_reduced.csv")

def balance_datasets(dataset):
    # Separate the DataFrame into majority and minority classes
    class_0_dataset = dataset[dataset.iloc[:, 1] == 0]
    class_1_dataset = dataset[dataset.iloc[:, 1] == 1]
    if (len(class_0_dataset) > len(class_1_dataset)):
        # Undersample the majority class
        class_0_dataset = resample(class_0_dataset,
                                   replace=False,
                                   n_samples=len(class_1_dataset),
                                   random_state=42)
    else:
        class_1_dataset = resample(class_1_dataset,
                                   replace=False,
                                   n_samples=len(class_0_dataset),
                                   random_state=42)
    
    # Combine the undersampled majority class with the minority class
    undersampled_dataset = pd.concat([class_0_dataset, class_1_dataset])
    
    return undersampled_dataset

# Load the CSV files into DataFrames, skipping the first row (header row)
df1 = balance_datasets(pd.read_csv(filename_1, header=None, skiprows=1))
#df2 = balance_datasets(pd.read_csv(filename_2, header=None, skiprows=1))
df3 = balance_datasets(pd.read_csv(filename_3, header=None, skiprows=1))
df4 = balance_datasets(pd.read_csv(filename_4, header=None, skiprows=1))
df5 = balance_datasets(pd.read_csv(filename_5, header=None, skiprows=1))
df6 = balance_datasets(pd.read_csv(filename_6, header=None, skiprows=1))
df7 = balance_datasets(pd.read_csv(filename_7, header=None, skiprows=1))
df8 = balance_datasets(pd.read_csv(filename_8, header=None, skiprows=1))


# Combine the DataFrames
combined_df = pd.concat([df1, df3, df4, df5, df6, df7, df8])
cleaned_filename = os.path.join(root, 'Split Data', "combined_dataset.csv")
combined_df.to_csv(cleaned_filename, header=['text', 'label'], index=False) #save dataset

# Function to clean and verify the dataset
def check_csv_types(csv_file, output_file):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file, skiprows=0)
    # Remove rows where any cell is NaN
    df = df.dropna()
    # Remove rows that contain 'text' or 'label' as values in any column
    df = df[~df.apply(lambda row: row.astype(str).str.contains('text|label').any(), axis=1)]
    # Check if the 'text' column contains only strings
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
check_csv_types(cleaned_filename, cleaned_filename)

############### Separate into test train sets ##################
from sklearn.model_selection import train_test_split
# Load the new combined dataset for further processing
completeDataset = pd.read_csv(cleaned_filename)

# Function to remove 'nan' rows for string data
def remove_nan_rows(dataset):
    # Assuming 'nan' is a string
    is_nan_text = np.array([str(item).strip().lower() == 'nan' for item in dataset['text']])
    is_nan_label = np.array([str(item).strip().lower() == 'nan' for item in dataset['label']])
    
    valid_indices = ~(is_nan_text | is_nan_label)
    
    cleaned_dataset = dataset.loc[valid_indices, :]
    
    return cleaned_dataset

# Remove 'nan' rows from train and test datasets
completeDataset = remove_nan_rows(completeDataset)

# Separate the DataFrame into two arrays: text and label
array_text = completeDataset['text'].values  
array_label = completeDataset['label'].values  

# Split data
text_train, text_test, label_train, label_test = train_test_split(
    array_text, array_label, test_size=0.3, random_state=42
)

# Save the training and testing sets as .npy files
np.save(os.path.join(root, 'Split Data', 'text_train.npy'), text_train)
np.save(os.path.join(root, 'Split Data', 'text_test.npy'), text_test)
np.save(os.path.join(root, 'Split Data', 'label_train.npy'), label_train)
np.save(os.path.join(root, 'Split Data', 'label_test.npy'), label_test)
