import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from segmentation_models_pytorch.losses import DiceLoss
import torch.optim as optim
from utils import CustomDataset
from model import SegmentationModel


# paths to the dataset
train_path = "demo/train_images/"
val_path = "demo/validation_images/"

# The image size and number of classes
image_size = (1024, 1024)
num_classes = 2 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing function for images
preprocess_image = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Preprocessing function for masks
preprocess_mask = transforms.Compose([
    transforms.ToTensor()
])

# Load the dataset
batch_size = 2
train_dataset = CustomDataset(train_path, image_size, image_transform=preprocess_image, mask_transform=preprocess_mask)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = CustomDataset(val_path, image_size, image_transform=preprocess_image, mask_transform=preprocess_mask)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

model = SegmentationModel(
    encoder_name="resnet50",
    encoder_depth=5,
    encoder_weights="imagenet",
    encoder_output_stride=16,
    in_channels=3,
    classes=num_classes,
    activation='sigmoid',
    upsampling=4
).to(device)


criterion = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=8e-5)

# Train the model
num_epochs = 12
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for i, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    train_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}")
