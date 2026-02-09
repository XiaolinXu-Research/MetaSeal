import random
import torch
from PIL import ImageEnhance, ImageOps, Image, ImageFilter
from torchvision.transforms import ToTensor, ToPILImage
import torch
import random
import torchvision.transforms.functional as F
import torch.nn.functional as nnF
import cv2
import numpy as np

# Apply selective benign transformations directly on a tensor
def apply_selected_transformations(tensor_img, device, resize=False, scale = 1, angle =0, rotate=False, flip=False, brightness=False, contrast=False, blur=False, center_crop=False, crop_ratio=0):
    # 1. Resize the image (e.g., scale down by 50%)
    if resize:
        # print(tensor_img.shape)
        original_size = tensor_img.shape[-2:]  # (Height, Width)
        new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
        # print(new_size)
        tensor_img = nnF.interpolate(tensor_img, size=new_size, mode='bilinear', align_corners=True)
        # Resize back to original size to maintain consistent dimensions
        tensor_img = nnF.interpolate(tensor_img, size=tuple(original_size), mode='bilinear', align_corners=True)

    # 2. Rotate the image (e.g., by a random angle between -30 and 30 degrees)
    if rotate:
        # angle = random.uniform(-30, 30)
        img = tensor_to_pil(tensor_img)
        rotated_image = img.rotate(angle, expand=True, resample=Image.BICUBIC)

        tmp_pil = rotated_image.rotate(-angle, 
                              expand=True,  # Must match original rotation setting
                              resample=Image.BICUBIC)
        # tmp_pil.save("output_image.png")

        tensor_img = pil_to_tensor(tmp_pil, tensor_img.device)
        tensor_img = F.center_crop(tensor_img, (512, 512))
        
    # 3. Flip the image horizontally
    if flip:
        tensor_img = F.hflip(tensor_img)

    # 4. Adjust brightness (e.g., enhance by 1.2x)
    if brightness:
        tensor_img = F.adjust_brightness(tensor_img, brightness_factor=1.2)

    # 5. Adjust contrast (e.g., enhance by 1.5x)
    if contrast:
        tensor_img = F.adjust_contrast(tensor_img, contrast_factor=1.5)

    # 6. Apply Gaussian blur (e.g., with a radius of 0.5)
    if blur:
        tensor_img = F.gaussian_blur(tensor_img, kernel_size=3, sigma=0.5)

    if center_crop:
        _, _, height, width = tensor_img.shape
        crop_height = int(height * crop_ratio)
        crop_width = int(width * crop_ratio)

        # Calculate the random cropping area
        top = random.randint(0, height - crop_height)
        left = random.randint(0, width - crop_width)
        bottom = top + crop_height
        right = left + crop_width

        # Create a mask with the cropped area set to 0
        mask = torch.ones_like(tensor_img)
        mask[:, :, top:bottom, left:right] = 0  # Randomly crop the area
        tensor_img = tensor_img * mask

    return tensor_img

# Convert tensor to PIL image
def tensor_to_pil(tensor_img):
    return ToPILImage()(tensor_img.cpu().squeeze(0))

# Convert PIL image back to tensor
def pil_to_tensor(pil_img, device):
    return ToTensor()(pil_img).unsqueeze(0).to(device)



def reverse_transformations(tensor_img, transformations, device):
    # Convert tensor to PIL image for reversing transformations
    pil_img = tensor_to_pil(tensor_img)

    # Reverse transformations based on the saved parameters
    if 'compression' in transformations:
        print("Note: Compression is lossy and cannot be perfectly reversed.")

    # if 'resize' in transformations:
    #     # original_size = transformations['resize']
    #     pil_img = pil_img.resize(original_size)  # Resize back to original dimensions

    if 'rotate' in transformations:
        # angle = transformations['rotate']
        pil_img = pil_img.rotate(5)  # Rotate back by the inverse angle

    if 'flip' in transformations:
        pil_img = ImageOps.mirror(pil_img)  # Flip back horizontally

    # Convert the PIL image back to a tensor
    tensor_reversed = pil_to_tensor(pil_img, device)

    return tensor_reversed

def apply_jpeg_compression(tensor_img, quality=70):
    """
    Apply JPEG compression to a tensor image.
    Args:
    - tensor_img: The tensor representing the image (shape [1, C, H, W])
    - quality: The JPEG compression quality (1-100)
    Returns:
    - tensor_img_compressed: The tensor after applying JPEG compression
    """
    # Convert tensor to PIL Image (assuming the tensor is in range [0, 1])
    pil_image = ToPILImage()(tensor_img.cpu().squeeze(0))
    
    # Save to a BytesIO object with JPEG compression
    from io import BytesIO
    output_io = BytesIO()
    pil_image.save(output_io, format="JPEG", quality=quality)
    
    # Reload the compressed image
    compressed_image = Image.open(output_io)
    
    # Convert the compressed image back to a tensor
    tensor_img_compressed = ToTensor()(compressed_image).unsqueeze(0)
    
    return tensor_img_compressed.to(tensor_img.device)

