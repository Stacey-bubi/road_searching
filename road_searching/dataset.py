import imgaug as ia
from imgaug import augmenters as iaa
import cv2
import matplotlib.pyplot as plt
from imgaug.augmentables.segmaps import SegmentationMapOnImage
import os

def apply_augmentation(image, mask, augmentation):
    segmap = SegmentationMapOnImage(mask, shape=image.shape[:2])

    # Apply augmentation
    augmented_image, augmented_segmap = augmentation(image=image, segmentation_maps=segmap)

    # Retrieve the augmented mask as a numpy array
    augmented_mask = augmented_segmap.get_arr()

    return augmented_image, augmented_mask

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


image_path = 'aimages'
mask_path = 'amasks'

output_path = 'augmented'

if not os.path.exists(output_path):
    os.makedirs(output_path)

image_files = os.listdir(image_path)
mask_files = os.listdir(mask_path)

for image_file, mask_file in zip(image_files, mask_files):
    image = cv2.imread(image_path + image_file)
    mask = cv2.imread(mask_path + mask_file, cv2.IMREAD_GRAYSCALE)

    # Looping through each augmentation technique
    for i, (augmentation, technique) in enumerate(augmentations):
        augmented_image, augmented_mask = apply_augmentation(image, mask, augmentation)

        cv2.imwrite(output_path + f'{image_file[:-4]}_{technique}.jpg', augmented_image)
        cv2.imwrite(output_path + f'{mask_file[:-4]}_{technique}.png', augmented_mask)

