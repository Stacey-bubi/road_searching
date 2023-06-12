import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import torch
import torchvision.transforms as transforms
from model import SegmentationModel

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
model_path = "model.pth"    # Path to the pth
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SegmentationModel(
    encoder_name="resnet50",
    encoder_depth=5,
    encoder_weights=None,
    encoder_output_stride=16,
    in_channels=3,
    classes=num_classes,
    activation='sigmoid',
    upsampling=4
).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

window = tk.Tk()
window.title("Road Segmentation")

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
        predicted_mask = (outputs > 0.5).squeeze().cpu().numpy()

    mask_image = Image.fromarray((predicted_mask * 255).astype('uint8'), mode='L')
    mask_image = mask_image.resize(image_size)

    # Convert the input image and mask image to Tkinter-compatible format
    input_tk = ImageTk.PhotoImage(image)
    mask_tk = ImageTk.PhotoImage(mask_image)
    input_label.config(image=input_tk)
    mask_label.config(image=mask_tk)
    input_label.image = input_tk
    mask_label.image = mask_tk

def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

    if file_path:
        predict_road(file_path)

button = tk.Button(window, text="Open Image", command=open_image)
button.pack()

window.mainloop()
