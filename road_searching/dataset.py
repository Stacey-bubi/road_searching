import imgaug as ia
from imgaug import augmenters as iaa
import cv2
import matplotlib.pyplot as plt
from imgaug.augmentables.segmaps import SegmentationMapOnImage

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, image_size, image_transform=None, mask_transform=None):
        self.root_dir = root_dir
        self.image_size = image_size
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.image_files = os.listdir(self.root_dir)
 
    def __len__(self):
        return len(self.image_files)
 
    def __getitem__(self, idx):
        image_filename = self.image_files[idx]
        if not image_filename.endswith('.jpg'):
            return self.__getitem__((idx + 1) % self.__len__()) # skip non-jpg files 
        mask_filename = image_filename.replace("_sat.jpg", "_mask.png")
 
        image_path = os.path.join(self.root_dir, image_filename)
        mask_path = os.path.join(self.root_dir, mask_filename)
 
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
 
        if self.image_size is not None:
            image = image.resize(self.image_size, Image.BILINEAR)
            mask = mask.resize(self.image_size, Image.NEAREST)
 
        if self.image_transform is not None:
            image = self.image_transform(image)
 
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
 
        return image, mask


def apply_augmentation(image, mask, augmentation):
    segmap = SegmentationMapOnImage(mask, shape=image.shape[:2])

    # Apply augmentation
    augmented_image, augmented_segmap = augmentation(image=image, segmentation_maps=segmap)

    # Retrieve the augmented mask as a numpy array
    augmented_mask = augmented_segmap.get_arr()

    return augmented_image, augmented_mask

image = cv2.imread('..._sat.jpg')
mask = cv2.imread('..._mask.png', cv2.IMREAD_GRAYSCALE)

# Defined augmentations
augmentations = [
    (iaa.Fliplr(1.0), 'Horizontal Flip'),
    (iaa.Flipud(1.0), 'Vertical Flip'),
    (iaa.Affine(rotate=(-45, 45)), 'Rotation'),
    (iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}), 'Scaling'),
    (iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}), 'Translation'),
    (iaa.Multiply((0.8, 1.2)), 'Brightness Adjustment'),
    (iaa.GammaContrast(gamma=(0.5, 1.5), per_channel=True), 'Contrast Adjustment')
]


fig, axes = plt.subplots(len(augmentations) + 1, 2, figsize=(10, 12))
axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Original Image')
axes[0, 1].imshow(mask, cmap='gray')
axes[0, 1].set_title('Original Mask')

for i, (augmentation, technique) in enumerate(augmentations):
    augmented_image, augmented_mask = apply_augmentation(image, mask, augmentation)
    # Display
    axes[i+1, 0].imshow(cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB))
    axes[i+1, 0].set_title(f'Augmented Image {i+1}\n({technique})')
    axes[i+1, 1].imshow(augmented_mask, cmap='gray')
    axes[i+1, 1].set_title(f'Augmented Mask {i+1}\n({technique})')

for ax in axes.flatten():
    ax.axis('off')

plt.tight_layout()

# Saved the figure with high resolution
plt.savefig('augmented_images.png', dpi=300)
plt.show()

