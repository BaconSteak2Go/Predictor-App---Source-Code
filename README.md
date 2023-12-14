
import tkinter as tk
from tkinter import filedialog, font
from PIL import Image, ImageTk
from keras.models import load_model
import numpy as np


#Load your trained model
model = load_model('Final_Model_50_Epochs.keras')



def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.convert('RGB')  # Convert the image to RGB 
        img = img.resize((64, 64), Image.ANTIALIAS)
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        prediction = model.predict(img_array)
        result = "This is not an advertisement." if prediction[0][0] >= 0.75 else "This is an advertisement."
        result_label.config(text=result)
        display_image(img)

def display_image(img):
    base_width = 250  # Set this to the desired width
    w_percent = (base_width / float(img.size[0]))
    h_size = int((float(img.size[1]) * float(w_percent)))
    img = img.resize((base_width, h_size), Image.ANTIALIAS)
    
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img  # Keep a reference

#Set up the GUI
window = tk.Tk()
window.title("Image Classification")

window.geometry('600x500')  # Set the size of the window
# window.configure(bg='green') #Sets the bg color to green
button_font = font.Font(family='Helvetica', size=14, weight='bold')

upload_button = tk.Button(window, text="Upload Image", command=upload_image)
upload_button.pack(pady=20)

image_label = tk.Label(window)
image_label.pack(pady=5)

result_label = tk.Label(window, text="", font=("Arial", 16))
result_label.pack(pady=5)

window.mainloop()
