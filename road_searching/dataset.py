import imgaug as ia
from imgaug import augmenters as iaa
import cv2
import matplotlib.pyplot as plt
from imgaug.augmentables.segmaps import SegmentationMapOnImage

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

