import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms

def otsu_thresholding(image: np.ndarray) -> np.ndarray:
    """
    Apply Otsu's thresholding to a color image by converting it to grayscale.
    Parameters: image (np.ndarray): Input color image of shape (H, W, C). 
    Returns: np.ndarray: Binary thresholded image.
    """
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return np.expand_dims(binary_image, axis=2)



def image2latent(vae, image):
    """
    Encodes an image into latent space using the VAE encoder.
    
    Args:
        vae: The variational autoencoder (VAE) model.
        image: A numpy array representing the image, with shape (B, H, W, C) and values in [0, 255].
    
    Returns:
        latent: The encoded latent representation.
    """
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image)#, dtype=torch.float16) 
    image = image / 255.0  # Normalize to [0, 1]
    image = (image - 0.5) * 2  # Scale to [-1, 1]
    image = image.permute(0, 3, 1, 2).to(vae.device)  # Convert to (B, C, H, W)
    
    with torch.no_grad():
        latent = vae.encode(image)['latent_dist'].mean  # Get latent distribution mean
    
    latent = latent * vae.config.scaling_factor  # Scale factor used in diffusion models
    return latent.detach().cpu().numpy()


def latent2image(vae, latent):
    """
    Decodes a latent into image space using the VAE decoder.
    
    Args:
        vae: The variational autoencoder (VAE) model.
        latent: A numpy array representing the latent, with shape (B, H//8, W//8, C).
        
    Returns:
        image: The decoded image representation.
    """
    if not isinstance(latent, torch.Tensor):
        latent = torch.tensor(latent)
    latent = latent.to(vae.device)
    latent = 1 / vae.config.scaling_factor * latent # 0.18215
    image = vae.decode(latent)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def save_to_image(attn_map, path):
    """
    Save the attention map to an image
    Parameters:
        attn_map: tensor (h, w)
        path: str
    Returns:
        None
    """
    attn_map = attn_map - attn_map.min()
    attn_map = attn_map / attn_map.max()
    attn_map = transforms.ToPILImage()(attn_map)
    attn_map.save(path)


def resize_attention(attn_map, H, W):
    """
    Resize the batch of attention maps to (H, W)
    Parameters:
        attn_map: tensor (b, 2, h, w) - input with 2 channels
        H: int - target height
        W: int - target width
    Returns:
        resized_attn: tensor (b, 2, H, W) - resized attention map with 2 channels
    """
    resized_attn = F.interpolate(attn_map,  # No need to add/remove channel dimension
                                size=(H, W), mode='bicubic', align_corners=False)
    return resized_attn


def normalize_map(attn_map):
    """
    Normalize the batch of attention maps across all channels
    Parameters:
        attn_map: tensor (b, 2, h, w) - input with 2 channels
    Returns:
        normalized_map: tensor (b, 2, h, w) - normalized attention map
    """
    # Normalize each channel separately
    min_val = attn_map.view(attn_map.shape[0], attn_map.shape[1], -1).min(dim=2, keepdim=True)[0].unsqueeze(2)
    max_val = attn_map.view(attn_map.shape[0], attn_map.shape[1], -1).max(dim=2, keepdim=True)[0].unsqueeze(2)
    normalized_map = (attn_map - min_val) / (max_val - min_val + 1e-6)  # Avoid division by zero
    return normalized_map


def fuse_all_attention(list_of_attn_maps, reduce=True):
    """
    Fuse all batch-wise attention maps in the list
    Parameters:
        list_of_attn_maps: list of tensor [(b, 2, h, w)] x n - input with 2 channels
    Returns:
        fused_attn: tensor (b, 2, H, W) or (n, b ,2, H, W) if not reduce - fused attention map
    """
    H, W = 64, 64  # Target height and width

    # Resize and normalize all batch-wise attention maps
    resized_maps = torch.stack([
        normalize_map(resize_attention(attn, H, W)) for attn in list_of_attn_maps
    ], dim=0)  # Shape: (n, b, 2, H, W)

    # Compute element-wise average across `n` different sources
    fused_attn = resized_maps.mean(dim=0)  # Shape: (b, 2, H, W)

    return fused_attn if reduce else resized_maps


def otsu_thresholding_batch(images: torch.Tensor) -> torch.Tensor:
    """
    Apply Otsu's thresholding to a batch of color or grayscale images.
    
    Parameters:
    images (torch.Tensor): Input batch of images with shape (b, n, h, w, c) or (b, n, h, w).
    
    Returns:
    torch.Tensor: Binary thresholded images with shape (b, n, h, w).
    """
    device = images.device
    images = images.cpu().numpy()
    if images.ndim == 4:
        images = images[..., None]  # Ensure 5D shape for consistency
    
    b, n, h, w, c = images.shape
    binary_images = np.zeros((b, n, h, w), dtype=np.uint8)
    
    for i in range(b):
        for j in range(n):
            image = images[i, j] * 255
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if c == 3 else image
            grayscale = grayscale.astype(np.uint8)
            _, binary_image = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_images[i, j, ...] = binary_image / 255
    
    return torch.tensor(binary_images, dtype=torch.float32).to(device)
