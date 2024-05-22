import tkinter as tk
from model import classify_text
import speech_recognition as sr

def animate_result(is_hate_speech):
    color = "red" if is_hate_speech else "green"
    animation_canvas.delete("all")  # Clear previous drawings
    animation_canvas.config(bg=color)
    radius = 5
    while radius < 100:
        animation_canvas.create_oval(
            150-radius, 75-radius, 150+radius, 75+radius, outline=color, width=4)
        root.update()  # Update the GUI
        radius += 5
    animation_canvas.after(50)  # Pause for 50 ms

def classify_input_text(event=None):
    user_input = entry.get()
    if user_input and user_input != classify_input_text.last_input:
        classify_input_text.last_input = user_input
        result_label.config(text="Classifying...")
        button.config(state=tk.DISABLED)
        root.update()

        is_hate_speech = classify_text(user_input)
        if is_hate_speech:
            predicted_class = "Hate Speech"
        else:
            predicted_class = "Not Hate Speech"
        result_label.config(text=f"Predicted Class: {predicted_class}")
        animate_result(is_hate_speech)
        button.config(state=tk.NORMAL)
    elif not user_input:
        result_label.config(text="Please enter some text to classify.")
    else:
        result_label.config(text="Text unchanged, classification not repeated.")

classify_input_text.last_input = None

def record_and_classify():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        result_label.config(text="Listening...")
        root.update()
        audio_data = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio_data, language=selected_language.get())
            entry.delete(0, tk.END)
            entry.insert(0, text)
            classify_input_text()
        except sr.UnknownValueError:
            result_label.config(text="Sorry, I did not understand that.")
        except sr.RequestError as e:
            result_label.config(text=f"Could not request results; {e}")

root = tk.Tk()
root.title("Text Classifier")

label = tk.Label(root, text="Enter text to classify:")
label.pack(pady=10)

entry = tk.Entry(root, width=50)
entry.pack(pady=5)
entry.bind("<Return>", classify_input_text)

button = tk.Button(root, text="Classify", command=classify_input_text)
button.pack(pady=20)

result_label = tk.Label(root, text="")
result_label.pack(pady=10)

# Create a canvas for animation
animation_canvas = tk.Canvas(root, width=300, height=150, bg="white")
animation_canvas.pack(pady=20)

mic_button = tk.Button(root, text="Record and Classify", command=record_and_classify)
mic_button.pack(pady=20)

# Add language buttons
language_label = tk.Label(root, text="Select Language:")
language_label.pack(pady=10)

language_frame = tk.Frame(root)
language_frame.pack(pady=10)

languages = ["English", "French", "German", "Chinese", "Japanese"]
language_buttons = []
selected_language = tk.StringVar(value="en-US")

for language, code in zip(languages, ["en-US", "fr-FR", "de-DE", "zh-CN", "ja-JP"]):
    button = tk.Radiobutton(language_frame, text=language, variable=selected_language, value=code)
    button.pack(side=tk.LEFT, padx=5)
    language_buttons.append(button)

root.mainloop()