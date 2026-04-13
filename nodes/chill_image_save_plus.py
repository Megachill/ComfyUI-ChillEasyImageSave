"""
ChillImageSavePlus - An enhanced image save node for ComfyUI
Supports multiple formats (PNG, JPEG, WebP, TIFF, BMP) with quality control and metadata options.
"""

import os
import json
import numpy as np
from PIL import Image, PngImagePlugin, TiffImagePlugin

import folder_paths
import comfy.utils


class ChillImageSavePlus:
    """
    Enhanced image save node supporting multiple formats with quality control
    and optional metadata stripping.
    """

    # Location presets with GPS coordinates (latitude, longitude, altitude in meters)
    LOCATION_PRESETS = {
        # Europe
        "Paris, France": (48.8566, 2.3522, 35),
        "London, UK": (51.5074, -0.1278, 11),
        "Berlin, Germany": (52.5200, 13.4050, 34),
        "Rome, Italy": (41.9028, 12.4964, 37),
        "Barcelona, Spain": (41.3851, 2.1734, 12),
        "Amsterdam, Netherlands": (52.3676, 4.9041, -2),
        "Vienna, Austria": (48.2082, 16.3738, 151),
        "Prague, Czech Republic": (50.0755, 14.4378, 177),
        "Athens, Greece": (37.9838, 23.7275, 70),
        "Stockholm, Sweden": (59.3293, 18.0686, 15),
        "Oslo, Norway": (59.9139, 10.7522, 7),
        # USA
        "New York City, USA": (40.7128, -74.0060, 10),
        "Los Angeles, USA": (34.0522, -118.2437, 89),
        "Chicago, USA": (41.8781, -87.6298, 181),
        "San Francisco, USA": (37.7749, -122.4194, 16),
        "Miami, USA": (25.7617, -80.1918, 2),
        "Las Vegas, USA": (36.1699, -115.1398, 610),
        "Seattle, USA": (47.6062, -122.3321, 56),
        "Denver, USA": (39.7392, -104.9903, 1609),
        # Canada
        "Toronto, Canada": (43.6532, -79.3832, 76),
        "Vancouver, Canada": (49.2827, -123.1207, 70),
        "Montreal, Canada": (45.5017, -73.5673, 35),
        "Calgary, Canada": (51.0447, -114.0719, 1045),
        # Russia
        "Moscow, Russia": (55.7558, 37.6173, 156),
        "St. Petersburg, Russia": (59.9311, 30.3609, 4),
        # Asia
        "Tokyo, Japan": (35.6762, 139.6503, 40),
        "Beijing, China": (39.9042, 116.4074, 44),
        "Singapore": (1.3521, 103.8198, 15),
        "Bangkok, Thailand": (13.7563, 100.5018, 4),
        "Seoul, South Korea": (37.5665, 126.9780, 38),
        "Mumbai, India": (19.0760, 72.8777, 14),
        "Dubai, UAE": (25.2048, 55.2708, 5),
        "Hong Kong": (22.3193, 114.1694, 7),
        # Latin America
        "Mexico City, Mexico": (19.4326, -99.1332, 2240),
        "Rio de Janeiro, Brazil": (-22.9068, -43.1729, 2),
        "Sao Paulo, Brazil": (-23.5505, -46.6333, 760),
        "Buenos Aires, Argentina": (-34.6037, -58.3816, 25),
        "Lima, Peru": (-12.0464, -77.0428, 154),
        "Bogota, Colombia": (4.7110, -74.0721, 2556),
        "Santiago, Chile": (-33.4489, -70.6693, 521),
        "Cusco, Peru": (-13.1631, -72.5450, 3399),
        # Oceania
        "Sydney, Australia": (-33.8688, 151.2093, 3),
        "Melbourne, Australia": (-37.8136, 144.9631, 31),
        # Manual entry option
        "Manual (use custom coords)": None,
    }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "Chill"

    @classmethod
    def INPUT_TYPES(cls):
        # Get sorted location preset keys
        preset_keys = sorted([k for k in cls.LOCATION_PRESETS.keys()])

        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "Chill"}),
                "format": (["png", "jpg", "webp", "tiff", "bmp"], {"default": "png"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1}),
                "strip_metadata": ("BOOLEAN", {"default": False}),
                "gps_enabled": ("BOOLEAN", {"default": False}),
                "gps_location_preset": (
                    preset_keys,
                    {"default": "Manual (use custom coords)"},
                ),
                "gps_latitude": (
                    "FLOAT",
                    {"default": 0.0, "min": -90.0, "max": 90.0, "step": 0.000001},
                ),
                "gps_longitude": (
                    "FLOAT",
                    {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.000001},
                ),
                "gps_altitude": (
                    "FLOAT",
                    {"default": 0.0, "min": -1000.0, "max": 10000.0, "step": 0.1},
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    def save_images(
        self,
        images,
        filename_prefix,
        format,
        quality,
        strip_metadata,
        gps_enabled=False,
        gps_location_preset="Manual (use custom coords)",
        gps_latitude=0.0,
        gps_longitude=0.0,
        gps_altitude=0.0,
        prompt=None,
        extra_pnginfo=None,
    ):
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

        # Resolve GPS coordinates from preset if selected
        if gps_enabled and gps_location_preset != "Manual (use custom coords)":
            preset_coords = self.LOCATION_PRESETS.get(gps_location_preset)
            if preset_coords:
                gps_latitude, gps_longitude, gps_altitude = preset_coords

        # Map format strings to PIL format names
        format_mapping = {
            "png": "PNG",
            "jpg": "JPEG",
            "webp": "WEBP",
            "tiff": "TIFF",
            "bmp": "BMP",
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
                    background.paste(
                        img, mask=img.split()[-1] if img.mode == "RGBA" else None
                    )
                    img = background
                else:
                    img = img.convert("RGB")

            # Get save path
            width, height = img.size
            full_output_folder, filename, counter, subfolder, _ = (
                folder_paths.get_save_image_path(
                    filename_prefix, self.output_dir, width, height
                )
            )

            # Generate filename with correct extension
            file_name = f"{filename}_{counter:05d}.{format}"
            file_path = os.path.join(full_output_folder, file_name)

            # Ensure we don't overwrite existing files - increment counter if file exists
            while os.path.exists(file_path):
                counter += 1
                file_name = f"{filename}_{counter:05d}.{format}"
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
                    # Add GPS metadata if enabled
                    if gps_enabled:
                        metadata.add_text("GPSLatitude", str(gps_latitude))
                        metadata.add_text("GPSLongitude", str(gps_longitude))
                        metadata.add_text("GPSAltitude", str(gps_altitude))
                    save_kwargs["pnginfo"] = metadata

                elif pil_format in ("JPEG", "WEBP"):
                    # JPEG and WebP use EXIF for metadata
                    exif_bytes = self._create_exif_metadata(
                        prompt,
                        extra_pnginfo,
                        gps_enabled,
                        gps_latitude,
                        gps_longitude,
                        gps_altitude,
                    )
                    if exif_bytes:
                        save_kwargs["exif"] = exif_bytes

                elif pil_format == "TIFF":
                    # TIFF supports metadata via tags
                    tiffinfo = self._create_tiff_metadata(
                        prompt,
                        extra_pnginfo,
                        gps_enabled,
                        gps_latitude,
                        gps_longitude,
                        gps_altitude,
                    )
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
                print(
                    f"ChillImageSavePlus: Saved {file_path} (format={format}, quality={quality if supports_quality else 'N/A'})"
                )
            except Exception as e:
                print(f"ChillImageSavePlus: Error saving {file_path}: {e}")
                raise

            results.append(
                {"filename": file_name, "subfolder": subfolder, "type": self.type}
            )

        return {"ui": {"images": results}}

    def _decimal_to_dms(self, decimal_degrees):
        """
        Convert decimal degrees to EXIF DMS (degrees, minutes, seconds) format.

        Args:
            decimal_degrees: Decimal degrees value

        Returns:
            Tuple of 3 rationals: (degrees, minutes, seconds)
        """
        # Get absolute value for calculation
        abs_value = abs(decimal_degrees)

        # Extract degrees
        degrees = int(abs_value)

        # Extract minutes
        minutes_float = (abs_value - degrees) * 60
        minutes = int(minutes_float)

        # Extract seconds (with 2 decimal places precision)
        seconds = (minutes_float - minutes) * 60
        seconds_rational = (int(seconds * 100), 100)

        return ((degrees, 1), (minutes, 1), seconds_rational)

    def _add_gps_to_exif(self, exif, latitude, longitude, altitude):
        """
        Add GPS metadata to EXIF dictionary.

        Args:
            exif: PIL Image.Exif object
            latitude: Decimal latitude (-90 to 90)
            longitude: Decimal longitude (-180 to 180)
            altitude: Altitude in meters
        """
        # Validate GPS coordinates
        if not (-90 <= latitude <= 90):
            raise ValueError(f"Latitude {latitude} out of range [-90, 90]")
        if not (-180 <= longitude <= 180):
            raise ValueError(f"Longitude {longitude} out of range [-180, 180]")

        # Create GPS info dict
        gps_info = {}

        # GPSVersionID (0x0000) - 4 bytes: (2, 2, 0, 0)
        gps_info[0x0000] = (2, 2, 0, 0)

        # GPSLatitudeRef (0x0001) - "N" or "S"
        gps_info[0x0001] = "N" if latitude >= 0 else "S"

        # GPSLatitude (0x0002) - 3 rationals (deg, min, sec)
        gps_info[0x0002] = self._decimal_to_dms(latitude)

        # GPSLongitudeRef (0x0003) - "E" or "W"
        gps_info[0x0003] = "E" if longitude >= 0 else "W"

        # GPSLongitude (0x0004) - 3 rationals (deg, min, sec)
        gps_info[0x0004] = self._decimal_to_dms(longitude)

        # GPSAltitudeRef (0x0005) - 0 = above sea level, 1 = below
        gps_info[0x0005] = 0 if altitude >= 0 else 1

        # GPSAltitude (0x0006) - rational (absolute value)
        abs_altitude = abs(altitude)
        # Use 1 decimal place precision for altitude
        gps_info[0x0006] = (int(abs_altitude * 10), 10)

        # Add GPS info to EXIF using the GPSInfo tag (0x8825)
        exif[0x8825] = gps_info

    def _create_exif_metadata(
        self,
        prompt,
        extra_pnginfo,
        gps_enabled=False,
        gps_latitude=0.0,
        gps_longitude=0.0,
        gps_altitude=0.0,
    ):
        """
        Create EXIF metadata dictionary for JPEG/WebP formats.

        Args:
            prompt: Workflow prompt
            extra_pnginfo: Additional metadata
            gps_enabled: Whether to include GPS metadata
            gps_latitude: GPS latitude (-90 to 90)
            gps_longitude: GPS longitude (-180 to 180)
            gps_altitude: GPS altitude in meters

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

            # Create EXIF bytes using PIL's Exif class
            exif = Image.Exif()

            if metadata_dict:
                # Convert metadata to JSON string
                metadata_json = json.dumps(metadata_dict, separators=(",", ":"))

                # Add UserComment tag (0x9286 = 37510)
                # UserComment format: first 8 bytes are charset identifier
                # "ASCII\x00\x00\x00" for ASCII encoding
                charset = b"ASCII\x00\x00\x00"
                user_comment = charset + metadata_json.encode("ascii")
                exif[0x9286] = user_comment

            # Add ImageDescription (0x010E = 270)
            exif[0x010E] = "ComfyUI Workflow"

            # Add Software tag (0x0131 = 305)
            exif[0x0131] = "ComfyUI - ChillImageSavePlus"

            # Add GPS metadata if enabled
            if gps_enabled:
                self._add_gps_to_exif(exif, gps_latitude, gps_longitude, gps_altitude)

            return exif.tobytes()

        except Exception as e:
            print(f"ChillImageSavePlus: Warning - could not create EXIF metadata: {e}")

        return None

    def _create_tiff_metadata(
        self,
        prompt,
        extra_pnginfo,
        gps_enabled=False,
        gps_latitude=0.0,
        gps_longitude=0.0,
        gps_altitude=0.0,
    ):
        """
        Create TIFF metadata dictionary.

        Args:
            prompt: Workflow prompt
            extra_pnginfo: Additional metadata
            gps_enabled: Whether to include GPS metadata
            gps_latitude: GPS latitude (-90 to 90)
            gps_longitude: GPS longitude (-180 to 180)
            gps_altitude: GPS altitude in meters

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
                metadata_json = json.dumps(metadata_dict, separators=(",", ":"))

                # ImageDescription tag (270)
                tiffinfo[270] = metadata_json

            # Software tag (305)
            tiffinfo[305] = "ComfyUI - ChillImageSavePlus"

            # Add GPS metadata if enabled
            if gps_enabled:
                if not (-90 <= gps_latitude <= 90):
                    raise ValueError(f"Latitude {gps_latitude} out of range [-90, 90]")
                if not (-180 <= gps_longitude <= 180):
                    raise ValueError(
                        f"Longitude {gps_longitude} out of range [-180, 180]"
                    )

                # TIFF uses same GPS tags as EXIF via GDAL metadata
                tiffinfo[34853] = {  # GPSInfo tag
                    0: (2, 2, 0, 0),  # GPSVersionID
                    1: "N" if gps_latitude >= 0 else "S",  # GPSLatitudeRef
                    2: self._decimal_to_dms(gps_latitude),  # GPSLatitude
                    3: "E" if gps_longitude >= 0 else "W",  # GPSLongitudeRef
                    4: self._decimal_to_dms(gps_longitude),  # GPSLongitude
                    5: 0 if gps_altitude >= 0 else 1,  # GPSAltitudeRef
                    6: (int(abs(gps_altitude) * 10), 10),  # GPSAltitude
                }

            return tiffinfo if tiffinfo else None

        except Exception as e:
            print(f"ChillImageSavePlus: Warning - could not create TIFF metadata: {e}")

        return None


# Node registration
NODE_CLASS_MAPPINGS = {"ChillImageSavePlus": ChillImageSavePlus}

NODE_DISPLAY_NAME_MAPPINGS = {"ChillImageSavePlus": "Chill Image Save Plus"}
