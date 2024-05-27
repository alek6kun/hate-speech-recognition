# hate-speech-recognition
*Demonstration video:* https://youtu.be/gavHuw_kwHw

This is the code and datasets used for training our hate speech classifier model with PyTorch.

There are more details about the implementation and citations in the Report.pdf.

Check out the Poster.pdf as well! It summarizes our work.


## The environment
Python 3.10.4

All necessary modules in requirements.txt

## Install SMART-mDeBERTa hate-speech recognition
```
git clone https://github.com/alek6kun/hate-speech-recognition.git
cd hate-speech-recognition
pip install -r requirements.txt
```
### Model weights trained from balanced dataset
Install https://drive.google.com/file/d/1fv3RdJxij-7nUeZQEFp5jZ1aXW6YfeDU/view?usp=drive_link and add to hate-speech-recognition folder
### Model weights trained from unbalanced dataset
Install https://drive.google.com/file/d/1FOKFn0d_KqS2U5ItxqeRXMz8ockUWkYC/view?usp=drive_link and add to hate-speech-recognition folder

## Run SMART-mDeBERTa hate-speech recognition with a GUI!
```
python gui.py
```
You can change from balanced to unbalanced by modifying the loaded model in model.py.
