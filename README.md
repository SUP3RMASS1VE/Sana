# Sana Sprint Gradio Web UI

A Gradio web interface for the Sana Sprint text-to-image generation model. This app allows you to generate high-quality images in just 1-4 steps using the powerful [Sana Sprint 1.6B 1024px](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_1.6B_1024px) model.

## Features

- Generate high-quality 1024px images in 1-4 steps
- Automatic model download and setup on first run
- Various style presets for different aesthetics
- Adjustable parameters for customized generation
- Safety filtering to prevent harmful content generation
- Simple and intuitive user interface

## Requirements

- Python 3.8+
- CUDA-compatible GPU with at least 8GB VRAM (strongly recommended)
- Internet connection for initial setup

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/sana-sprint-gradio.git
   cd sana-sprint-gradio
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the application:

```
python app.py
```

The first time you run the app, it will:
1. Clone the Sana repository
2. Install necessary dependencies
3. Download the Sana Sprint model and safety model
4. Start the Gradio web interface

Once setup is complete, you can access the interface in your web browser at http://localhost:7860.

### Command Line Options

- `--share`: Create a public shareable link
- `--port PORT`: Specify a custom port (default: 7860)
- `--model_path PATH`: Use a local model file if already downloaded
- `--safety_model_path PATH`: Use a local safety model if already downloaded

Example:
```
python app.py --share --port 8080
```

## Interface Guide

1. **Prompt**: Enter a detailed description of the image you want to generate
2. **Style**: Choose a style preset from the dropdown menu
3. **Advanced Settings**: Adjust parameters like seed, dimensions, guidance scale, and steps
4. **Generate**: Click to create your image
5. **Examples**: Try pre-configured examples by clicking on them

## About Sana Sprint

Sana Sprint is an ultra-efficient diffusion model for text-to-image generation, developed by NVIDIA. It reduces inference steps from 20 to 1-4 while achieving state-of-the-art performance. With latencies of just 0.1s (T2I) and 0.25s (ControlNet) for 1024×1024 images on H100, Sana Sprint is ideal for AI-powered consumer applications.

[Read more about Sana Sprint on Hugging Face](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_1.6B_1024px)

## License

This project is subject to the license of the Sana Sprint model. Please refer to the [Sana repository](https://github.com/NVlabs/Sana) for more information on licensing.

## Acknowledgements

- [NVIDIA's Sana Project](https://github.com/NVlabs/Sana)
- [Hugging Face](https://huggingface.co/)
- [Gradio](https://gradio.app/) 