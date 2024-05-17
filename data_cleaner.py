import os, sys
import numpy as np
import pandas as pd
import re

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