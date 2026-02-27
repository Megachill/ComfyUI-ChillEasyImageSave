"""
ChillImageSavePlus - An enhanced image save node for ComfyUI
Supports multiple formats (PNG, JPEG, WebP, TIFF, BMP) with quality control and metadata options.
"""

import os
import json
import numpy as np
from PIL import Image, PngImagePlugin, ExifTags, TiffImagePlugin
from PIL.ExifTags import TAGS

import folder_paths
import comfy.utils


class ChillImageSavePlus:
    """
    Enhanced image save node supporting multiple formats with quality control
    and optional metadata stripping.
    """

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "Chill"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "Chill"}),
                "format": (["png", "jpg", "webp", "tiff", "bmp"], {"default": "png"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1}),
                "strip_metadata": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    def save_images(self, images, filename_prefix, format, quality, strip_metadata, prompt=None, extra_pnginfo=None):
        """
        Save images in the specified format with quality control and optional metadata.

        Args:
            images: Tensor of shape (B, H, W, C) in range [0, 1]
            filename_prefix: Prefix for saved files
            format: Output format (png, jpg, webp, tiff, bmp)
            quality: Quality setting for lossy formats (1-100)
            strip_metadata: If True, strip all metadata
            prompt: Workflow prompt metadata
            extra_pnginfo: Additional PNG info metadata

        Returns:
            Dict with UI update information
        """
        # Clamp quality to valid range
        quality = max(1, min(100, quality))

        # Map format strings to PIL format names
        format_mapping = {
            "png": "PNG",
            "jpg": "JPEG",
            "webp": "WEBP",
            "tiff": "TIFF",
            "bmp": "BMP"
        }
        pil_format = format_mapping.get(format, "PNG")

        # Determine if format supports quality parameter
        lossy_formats = {"JPEG", "WEBP"}
        supports_quality = pil_format in lossy_formats

        # Determine if format supports alpha channel
        alpha_formats = {"PNG", "TIFF", "WEBP"}
        supports_alpha = pil_format in alpha_formats

        results = []

        # Process each image in the batch
        for batch_idx, image in enumerate(images):
            # Convert tensor to numpy array (H, W, C)
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # Handle alpha channel for formats that don't support it
            if not supports_alpha and img.mode in ("RGBA", "LA", "P"):
                # Create white background for transparency
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                if img.mode in ("RGBA", "LA"):
                    background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
                    img = background
                else:
                    img = img.convert("RGB")

            # Get save path
            width, height = img.size
            full_output_folder, filename, counter, subfolder, _ = \
                folder_paths.get_save_image_path(filename_prefix, self.output_dir, width, height)

            # Generate filename with correct extension
            file_extension = format if format != "jpg" else "jpg"
            file_name = f"{filename}_{counter:05d}.{file_extension}"
            file_path = os.path.join(full_output_folder, file_name)

            # Ensure we don't overwrite existing files - increment counter if file exists
            while os.path.exists(file_path):
                counter += 1
                file_name = f"{filename}_{counter:05d}.{file_extension}"
                file_path = os.path.join(full_output_folder, file_name)

            # Prepare save options
            save_kwargs = {}
            if supports_quality:
                save_kwargs["quality"] = quality
                if pil_format == "JPEG":
                    save_kwargs["optimize"] = True
                    save_kwargs["progressive"] = True
                elif pil_format == "WEBP":
                    save_kwargs["method"] = 6  # Best compression

            # Handle metadata
            if not strip_metadata:
                if pil_format == "PNG":
                    # Use PNG text chunks for metadata
                    metadata = PngImagePlugin.PngInfo()
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for key, value in extra_pnginfo.items():
                            metadata.add_text(key, json.dumps(value))
                    # Add workflow metadata
                    workflow_metadata = {"workflow": extra_pnginfo} if extra_pnginfo else None
                    if workflow_metadata:
                        metadata.add_text("workflow", json.dumps(workflow_metadata))
                    save_kwargs["pnginfo"] = metadata

                elif pil_format == "JPEG":
                    # Embed metadata in EXIF
                    exif_dict = self._create_exif_metadata(prompt, extra_pnginfo)
                    if exif_dict:
                        img.save(file_path, format=pil_format, exif=exif_dict, **save_kwargs)
                        print(f"ChillImageSavePlus: Saved {file_path} (format={format}, quality={quality})")
                        results.append({
                            "filename": file_name,
                            "subfolder": subfolder,
                            "type": self.type
                        })
                        continue

                elif pil_format == "WEBP":
                    # WebP supports EXIF metadata
                    exif_dict = self._create_exif_metadata(prompt, extra_pnginfo)
                    if exif_dict:
                        save_kwargs["exif"] = exif_dict

                elif pil_format == "TIFF":
                    # TIFF supports metadata via tags
                    tiffinfo = self._create_tiff_metadata(prompt, extra_pnginfo)
                    if tiffinfo:
                        save_kwargs["tiffinfo"] = tiffinfo

            else:
                # Strip metadata by creating fresh image from raw pixels
                if img.mode in ("RGB", "RGBA", "L"):
                    img = Image.fromarray(np.array(img))
                else:
                    img = img.convert("RGB")

            # Save the image
            try:
                img.save(file_path, format=pil_format, **save_kwargs)
                print(f"ChillImageSavePlus: Saved {file_path} (format={format}, quality={quality if supports_quality else 'N/A'})")
            except Exception as e:
                print(f"ChillImageSavePlus: Error saving {file_path}: {e}")
                raise

            results.append({
                "filename": file_name,
                "subfolder": subfolder,
                "type": self.type
            })

        return {"ui": {"images": results}}

    def _create_exif_metadata(self, prompt, extra_pnginfo):
        """
        Create EXIF metadata dictionary for JPEG/WebP formats.

        Args:
            prompt: Workflow prompt
            extra_pnginfo: Additional metadata

        Returns:
            EXIF bytes or None
        """
        try:
            # Create a minimal EXIF structure
            # UserComment tag (37510) is commonly used for embedding workflow data
            exif_dict = {}

            metadata_dict = {}
            if prompt is not None:
                metadata_dict["prompt"] = prompt
            if extra_pnginfo is not None:
                metadata_dict.update(extra_pnginfo)

            if metadata_dict:
                # Convert metadata to JSON string
                metadata_json = json.dumps(metadata_dict, separators=(',', ':'))

                # Create EXIF bytes using PIL's Exif class
                exif = Image.Exif()

                # Add UserComment tag (0x9286 = 37510)
                # UserComment format: first 8 bytes are charset identifier
                # "ASCII\x00\x00\x00" for ASCII encoding
                charset = b'ASCII\x00\x00\x00'
                user_comment = charset + metadata_json.encode('ascii')
                exif[0x9286] = user_comment

                # Add ImageDescription (0x010E = 270)
                exif[0x010E] = "ComfyUI Workflow"

                # Add Software tag (0x0131 = 305)
                exif[0x0131] = "ComfyUI - ChillImageSavePlus"

                return exif.tobytes()

        except Exception as e:
            print(f"ChillImageSavePlus: Warning - could not create EXIF metadata: {e}")

        return None

    def _create_tiff_metadata(self, prompt, extra_pnginfo):
        """
        Create TIFF metadata dictionary.

        Args:
            prompt: Workflow prompt
            extra_pnginfo: Additional metadata

        Returns:
            Dict of TIFF tags or None
        """
        try:
            tiffinfo = {}

            metadata_dict = {}
            if prompt is not None:
                metadata_dict["prompt"] = prompt
            if extra_pnginfo is not None:
                metadata_dict.update(extra_pnginfo)

            if metadata_dict:
                metadata_json = json.dumps(metadata_dict, separators=(',', ':'))

                # ImageDescription tag (270)
                tiffinfo[270] = metadata_json

                # Software tag (305)
                tiffinfo[305] = "ComfyUI - ChillImageSavePlus"

                return tiffinfo

        except Exception as e:
            print(f"ChillImageSavePlus: Warning - could not create TIFF metadata: {e}")

        return None


# Node registration
NODE_CLASS_MAPPINGS = {
    "ChillImageSavePlus": ChillImageSavePlus
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChillImageSavePlus": "Chill Image Save Plus"
}
