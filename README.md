# Chill Image Save Plus for ComfyUI

An enhanced image save node for ComfyUI that extends the built-in `SaveImage` node with additional format support, quality control for lossy formats, and optional metadata stripping.

## Features

- **Multiple Format Support**: Save images in PNG, JPEG, WebP, TIFF, and BMP formats
- **Quality Control**: Adjustable quality (1-100) for lossy formats (JPEG, WebP)
- **Metadata Options**: Choose to embed or strip metadata (workflow, prompt)
- **Batch Processing**: Handles batched images with automatic counter incrementing
- **RGBA Handling**: Automatically converts RGBA images to RGB for formats that don't support alpha (JPEG, BMP)

## Installation

### Method 1: Git Clone (Recommended)

```bash
cd ComfyUI/custom_nodes
git clone git@github.com:Megachill/ComfyUI-ChillEasyImageSave.git
```

### Method 2: Manual Download

1. Download this repository as a ZIP file
2. Extract it to `ComfyUI/custom_nodes/`
3. Rename the folder to `ComfyUI-ChillEasyImageSave`

### Dependencies

The node requires Pillow >= 9.0.0 (already included with ComfyUI):

**Standard Installation:**
```bash
pip install -r requirements.txt
```

**ComfyUI Portable (Windows):**
For ComfyUI Portable installations, use the embedded Python:
```bash
../../../python_embeded/python.exe -m pip install -r requirements.txt
```
Run this command from within the `ComfyUI/custom_nodes/ComfyUI-ChillEasyImageSave/` folder.

## Usage

1. Start ComfyUI
2. The node appears in the node library under the **"Chill"** category
3. Search for **"Chill Image Save Plus"** in the node search (double-click on canvas)

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `images` | Input image tensor from a node (e.g., VAE Decode, Load Image) | Required |
| `filename_prefix` | Prefix for saved files. Supports subfolders (e.g., "outputs/test") | "Chill" |
| `format` | Output format: png, jpg, webp, tiff, bmp | "png" |
| `quality` | Quality for lossy formats (JPEG, WebP). Range: 1-100 | 95 |
| `strip_metadata` | If true, removes all metadata from output | false |

### Format Features

| Format | Quality Control | Alpha Support | Metadata Support |
|--------|-----------------|---------------|------------------|
| PNG | No | Yes | PNG text chunks |
| JPEG | Yes | No (RGB conversion) | EXIF UserComment |
| WebP | Yes | Yes | EXIF |
| TIFF | No | Yes | TIFF tags |
| BMP | No | No (RGB conversion) | None |

### Metadata

When `strip_metadata` is disabled:
- **PNG**: Embeds workflow and prompt as PNG text chunks
- **JPEG**: Embeds metadata in EXIF UserComment tag
- **WebP**: Embeds EXIF data
- **TIFF**: Embeds metadata in standard TIFF tags
- **BMP**: No metadata support

## Example Workflow

See `examples/workflow.json` for a sample ComfyUI workflow demonstrating the node.

## Tips

- **Subfolders**: Use `/` or `\` in `filename_prefix` to save to subdirectories (e.g., "batch1/output")
- **Quality Settings**:
  - JPEG: 90-100 for high quality, 75-85 for web use, 50-70 for small file size
  - WebP: Similar to JPEG but generally produces smaller files at equivalent quality
- **Transparency**: Use PNG or WebP if you need to preserve alpha channel
- **Maximum Compatibility**: Use JPEG for broadest compatibility with other software

## Compatibility

- ComfyUI: Latest version (tested with 2024 builds)
- Python: 3.10+
- Pillow: >= 9.0.0

## License

MIT License - See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and feature requests, please use the GitHub issue tracker.
