import os
import argparse
import torch
import gradio as gr
from pathlib import Path
import subprocess
import sys
import importlib.util
from huggingface_hub import hf_hub_download, snapshot_download
import random
import numpy as np
from PIL import Image
import platform

# Constants
# Platform-specific null device path
NULL_DEVICE = "NUL" if platform.system() == "Windows" else "/dev/null"
SANA_REPO_ID = "Efficient-Large-Model/Sana_Sprint_1.6B_1024px"
MODEL_FILENAME = "checkpoints/Sana_Sprint_1.6B_1024px.pth"
CONFIG_PATH = "configs/sana_sprint_config/1024ms/SanaSprint_1600M_1024px_allqknorm_bf16_scm_ladd.yaml"
MAX_SEED = np.iinfo(np.int32).max

# Style presets
style_list = [
    {
        "name": "(No style)",
        "prompt": "{prompt}",
    },
    {
        "name": "Cinematic",
        "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, "
        "cinemascope, moody, epic, gorgeous, film grain, grainy",
    },
    {
        "name": "Photographic",
        "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
    },
    {
        "name": "Anime",
        "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
    },
    {
        "name": "Manga",
        "prompt": "manga style {prompt} . vibrant, high-energy, detailed, iconic, Japanese comic style",
    },
    {
        "name": "Digital Art",
        "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
    },
    {
        "name": "Pixel art",
        "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
    },
    {
        "name": "Fantasy art",
        "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, "
        "majestic, magical, fantasy art, cover art, dreamy",
    },
    {
        "name": "Neonpunk",
        "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, "
        "detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, "
        "ultra detailed, intricate, professional",
    },
    {
        "name": "3D Model",
        "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
    },
]

styles = {k["name"]: (k["prompt"]) for k in style_list}
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "(No style)"

# Helper function to load modules from file path
def load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def setup_environment():
    """Set up the environment by creating necessary directories."""
    print("Setting up environment...")
    
    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

def download_model_files():
    """Download necessary model files if they don't exist."""
    print("Checking model files...")
    
    # Create cache directory
    cache_dir = Path("models")
    cache_dir.mkdir(exist_ok=True)
    
    # Download model checkpoint
    model_path = None
    try:
        print(f"Downloading model from {SANA_REPO_ID}...")
        
        # Download full repo to get all necessary files (VAE, configs, etc)
        snapshot_download(
            repo_id=SANA_REPO_ID,
            cache_dir=cache_dir,
            repo_type="model",
        )
        
        # Get specific model path
        model_path = hf_hub_download(
            repo_id=SANA_REPO_ID,
            filename=MODEL_FILENAME,
            cache_dir=cache_dir,
            repo_type="model",
        )
        print(f"Model downloaded to {model_path}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise
    
    return model_path

def randomize_seed(seed, randomize_seed):
    """Generate a random seed if randomize_seed is True."""
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def apply_style(style_name, prompt):
    """Apply the selected style to the prompt."""
    p = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", prompt)

def main():
    """Main function to run the web UI."""
    parser = argparse.ArgumentParser(description="Sana Sprint Text-to-Image Generator")
    parser.add_argument("--share", action="store_true", help="Create a shareable link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the Gradio app on")
    parser.add_argument("--model_path", type=str, help="Path to the model file (if already downloaded)")
    parser.add_argument("--config_path", type=str, help="Path to the model config file")
    args = parser.parse_args()
    
    # Set up the environment
    setup_environment()
    
    # Get absolute paths
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    app_dir = current_dir / "app"
    sana_sprint_pipeline_path = app_dir / "sana_sprint_pipeline.py"
    
    # Check if pipeline file exists
    if not sana_sprint_pipeline_path.exists():
        print(f"Error: Could not find {sana_sprint_pipeline_path}")
        sys.exit(1)
    
    # Resolve config path
    config_path = current_dir / CONFIG_PATH
    if args.config_path:
        config_path = Path(args.config_path)
    
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        print(f"Current directory: {current_dir}")
        sys.exit(1)
    
    # Load the pipeline module
    SanaSprintPipeline = load_module_from_file("sana_sprint_pipeline", sana_sprint_pipeline_path).SanaSprintPipeline
    
    # Download model files if needed
    model_path = args.model_path or download_model_files()
    
    print("Initializing Sana Sprint model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Initialize the pipeline
    pipe = SanaSprintPipeline(str(config_path))
    pipe.from_pretrained(model_path)
    pipe.register_progress_bar(gr.Progress())
    
    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Define the generation function
    @torch.no_grad()
    @torch.inference_mode()
    def generate(
        prompt,
        style=DEFAULT_STYLE_NAME,
        seed=0,
        height=1024,
        width=1024,
        guidance_scale=4.5,
        num_inference_steps=1,
        max_timesteps=1.56830,
        intermediate_timesteps=1.3,
        timesteps=None,
        randomize_seed_option=False,
        use_resolution_binning=True,
    ):
        # Apply style and prepare for generation
        seed = randomize_seed(seed, randomize_seed_option)
        generator = torch.Generator(device=device).manual_seed(seed)
        
        styled_prompt = apply_style(style, prompt)
        print(f"Processing prompt: {styled_prompt}")
        
        # Set scheduler parameters
        pipe.config.max_timesteps = max_timesteps
        pipe.config.intermediate_timesteps = intermediate_timesteps
        if isinstance(timesteps, str) and timesteps.strip():
            custom_steps = [float(t.strip()) for t in timesteps.split(",") if t.strip()]
            custom_steps.append(0.0)
            pipe.config.timesteps = custom_steps
        
        # Generate image
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        latents = pipe(
            prompt=styled_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,
            generator=generator,
            use_resolution_binning=use_resolution_binning,
        )
        end.record()
        
        torch.cuda.synchronize()
        time_taken = start.elapsed_time(end) / 1000.0  # convert to seconds
        
        # Process output
        images = []
        for img in latents:
            # Convert tensor to PIL image
            with torch.no_grad():
                img = 0.5 * (img + 1.0)
                img = img.clamp(0, 1)
                img = (img * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                img_pil = Image.fromarray(img)
                images.append(img_pil)
                
                # Save the image
                img_path = output_dir / f"sana_sprint_{seed}.png"
                img_pil.save(img_path)
        
        generation_info = (
            f"Seed: {seed}, Steps: {num_inference_steps}, "
            f"CFG scale: {guidance_scale}, Time: {time_taken:.2f}s"
        )
        
        return images[0], generation_info
    
    # Create the Gradio interface
    with gr.Blocks(title="Sana Sprint Text-to-Image Generator", css="""
        :root {
            --primary-color: #7B68EE;
            --secondary-color: #9370DB;
            --accent-color: #9370DB;
            --neutral-color: #E0E0E0;
            --gradient-from: #7B68EE;
            --gradient-to: #9370DB;
            --dark-color: #282856;
            --text-color: #282856;
        }
        
        body {
            background-image: linear-gradient(to right, var(--gradient-from), var(--gradient-to));
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .gr-panel {
            border-radius: 16px !important;
            border: none !important;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1) !important;
            background-color: rgba(255, 255, 255, 0.9) !important;
            backdrop-filter: blur(10px) !important;
        }
        
        .gr-button-primary {
            background: linear-gradient(to right, var(--gradient-from), var(--gradient-to)) !important;
            border: none !important;
            box-shadow: 0 4px 12px rgba(123, 104, 238, 0.5) !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
            color: white !important;
            font-weight: bold !important;
            min-height: 45px !important;
        }
        
        .gr-button-primary:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 16px rgba(123, 104, 238, 0.7) !important;
        }
        
        .gr-input, .gr-panel {
            border-radius: 8px !important;
        }
        
        .gr-slider {
            color: var(--primary-color) !important;
        }
        
        h1 {
            font-size: 2.5em !important;
            font-weight: 700 !important;
            color: white !important;
            text-align: center !important;
            margin-bottom: 0.5em !important;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
        }
        
        h2 {
            font-size: 1.5em !important;
            color: var(--text-color) !important;
            font-weight: 600 !important;
        }
        
        .image-container img {
            border-radius: 16px !important;
            box-shadow: 0 8px 24px rgba(0,0,0,0.15) !important;
        }
    """) as demo:
        with gr.Row():
            gr.HTML("""
                <div style="text-align: center; margin-bottom: 10px">
                    <h1>Sana Sprint Text-to-Image Generator</h1>
                    <p style="color: white; font-size: 1.2em; margin-top: -10px;">Create stunning images in just a few steps</p>
                </div>
            """)
        
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    prompt = gr.Textbox(
                        label="Your Prompt",
                        placeholder="Describe the image you want to create...",
                        lines=3,
                    )
                    style = gr.Dropdown(
                        label="Style Preset",
                        choices=STYLE_NAMES,
                        value=DEFAULT_STYLE_NAME,
                    )
                    
                    with gr.Row():
                        generate_btn = gr.Button("Generate Image", variant="primary")
                
                with gr.Accordion("Advanced Settings", open=False):
                    with gr.Group():
                        with gr.Row():
                            seed = gr.Slider(
                                label="Seed",
                                minimum=0,
                                maximum=MAX_SEED,
                                step=1,
                                value=0,
                            )
                            randomize_seed_option = gr.Checkbox(
                                label="Randomize seed",
                                value=False,
                            )
                        
                        with gr.Row():
                            height = gr.Slider(
                                label="Height",
                                minimum=512,
                                maximum=4096,
                                step=64,
                                value=1024,
                            )
                            width = gr.Slider(
                                label="Width",
                                minimum=512,
                                maximum=4096,
                                step=64,
                                value=1024,
                            )
                        
                        with gr.Row():
                            guidance_scale = gr.Slider(
                                label="Guidance Scale",
                                minimum=1.0,
                                maximum=15.0,
                                step=0.1,
                                value=4.5,
                            )
                            num_inference_steps = gr.Slider(
                                label="Steps",
                                minimum=1,
                                maximum=4,
                                step=1,
                                value=1,
                            )
                        
                        use_resolution_binning = gr.Checkbox(
                            label="Use resolution binning",
                            value=True,
                        )
                        
                        with gr.Accordion("Expert Settings", open=False):
                            max_timesteps = gr.Slider(
                                label="Max Timesteps",
                                minimum=1.0,
                                maximum=2.0,
                                step=0.01,
                                value=1.56830,
                            )
                            intermediate_timesteps = gr.Slider(
                                label="Intermediate Timesteps",
                                minimum=1.0,
                                maximum=1.5,
                                step=0.01,
                                value=1.3,
                            )
                            timesteps = gr.Textbox(
                                label="Custom Timesteps (comma-separated)",
                                placeholder="e.g. 1.5, 1.0, 0.5",
                                value="",
                            )
            
            with gr.Column(scale=1):
                with gr.Group(elem_classes="image-container"):
                    output_image = gr.Image(
                        label="Generated Image", 
                        type="pil", 
                        height=512,
                        format="png",
                        show_download_button=True,
                        elem_id="output_image"
                    )
                    generation_info = gr.Textbox(label="Generation Info", interactive=False)
        
        # Examples section with a more attractive layout
        with gr.Row():
            gr.HTML("<h2 style='margin-top: 20px; text-align: center; color: white;'>Example Prompts</h2>")
        
        with gr.Row():
            with gr.Column():
                gr.Examples(
                    examples=[
                        ["A serene landscape with mountains and a lake at sunset", "Photographic", 42],
                        ["A colorful bird perched on a branch", "Digital Art", 123],
                        ["A futuristic cityscape with flying cars", "Neonpunk", 456],
                        ["A cute cat wearing a hat", "Anime", 789],
                    ],
                    inputs=[prompt, style, seed],
                    label="Try these examples",
                )
        
        # Set up event handlers
        generate_btn.click(
            fn=generate,
            inputs=[
                prompt, style, seed, height, width, guidance_scale,
                num_inference_steps, max_timesteps, intermediate_timesteps,
                timesteps, randomize_seed_option, use_resolution_binning,
            ],
            outputs=[output_image, generation_info],
        )
    
    # Launch the Gradio app
    demo.launch(share=args.share, server_port=args.port)

if __name__ == "__main__":
    main() 
