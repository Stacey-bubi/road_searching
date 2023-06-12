import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import torch
import torchvision.transforms as transforms
from segmentation_models_pytorch import DeepLabV3Plus
import numpy as np

image_size = (1024, 1024)
num_classes = 2 

# Preprocessing function for images
preprocess_image = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load the trained model
model_path = "final_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path)
model.eval()

window = tk.Tk()
window.title("Road Segmentation")
window.geometry("1200x600") 

input_label = tk.Label(window)
input_label.pack()

mask_label = tk.Label(window)
mask_label.pack()

# Image prediction
def predict_road(image_path):
    image = Image.open(image_path).convert("RGB")
    input_image = preprocess_image(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_image)
    #    predicted_mask = (outputs > 0.5)
        predicted_mask = torch.argmin(outputs, dim=1).squeeze(0).cpu().numpy()

    mask_image = Image.fromarray(np.where(predicted_mask > 0.5, 255, 0).astype('uint8'), mode='L')
    mask_image = mask_image.resize(image_size)

    # Scale image and mask to fit the window
    width, height = window.winfo_width() // 2, window.winfo_height()
    input_image = image.resize((width, height))
    mask_image = mask_image.resize((width, height))

    # Convert the input image and mask image to Tkinter-compatible format
    input_tk = ImageTk.PhotoImage(input_image)
    mask_tk = ImageTk.PhotoImage(mask_image)

    input_label.config(image=input_tk)
    mask_label.config(image=mask_tk)

    input_label.image = input_tk
    mask_label.image = mask_tk

    window.update() 

def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

    if file_path:
        predict_road(file_path)

button = tk.Button(window, text="Open Image", command=open_image)
button.pack()

window.mainloop()
