# ComfyUI-Meissonic v1.2.2
# This custom node for ComfyUI provides functionality for background removal using various models,
# including RMBG-2.0, INSPYRENET, and BEN. It leverages deep learning techniques
# to process images and generate masks for background removal.

# License Notice:
# - RMBG-2.0: Apache-2.0 License (https://huggingface.co/briaai/RMBG-2.0)
# - INSPYRENET: MIT License (https://github.com/plemeri/InSPyReNet)
# - BEN: Apache-2.0 License (https://huggingface.co/PramaLLC/BEN)
# 
# This integration script follows GPL-3.0 License.
# When using or modifying this code, please respect both the original model licenses
# and this integration's license terms.
#
# Source: https://github.com/AILab-AI/ComfyUI-RMBG

import os
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import folder_paths
from PIL import ImageFilter
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
import shutil
import sys
import importlib.util
from tqdm import tqdm
from transformers import AutoModelForImageSegmentation

device = "cuda" if torch.cuda.is_available() else "cpu"

# Add model path
folder_paths.add_model_folder_path("meissonic", os.path.join(folder_paths.models_dir, "meissonic"))

# Model configuration
AVAILABLE_MODELS = {
    "Meissonic": {
        "type": "meissonic",
        "repo_id": "MeissonFlow/Meissonic",
        "files": {
            "model.safetensors": "text_encoder/model.safetensors",
            "transformer.safetensors": "transformer/diffusion_pytorch_model.safetensors",
            "vqvae.safetensors": "vqvae/diffusion_pytorch_model.safetensors",
        },
        "cache_dir": "MeissonicBase"
    },
    # "INSPYRENET": {
    #     "type": "inspyrenet",
    #     "repo_id": "1038lab/inspyrenet",
    #     "files": {
    #         "inspyrenet.safetensors": "inspyrenet.safetensors"
    #     },
    #     "cache_dir": "INSPYRENET"
    # },
    # "BEN": {
    #     "type": "ben",
    #     "repo_id": "PramaLLC/BEN",
    #     "files": {
    #         "model.py": "model.py",
    #         "BEN_Base.pth": "BEN_Base.pth"
    #     },
    #     "cache_dir": "BEN"
    # }
}


# Utility functions
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def handle_model_error(message):
    print(f"[MEISSONIC ERROR] {message}")
    raise RuntimeError(message)


class BaseModelLoader:
    def __init__(self):
        self.model = None
        self.current_model_version = None
        self.base_cache_dir = os.path.join(folder_paths.models_dir, "meissonic")

    def get_cache_dir(self, model_name):
        return os.path.join(self.base_cache_dir, AVAILABLE_MODELS[model_name]["cache_dir"])

    def check_model_cache(self, model_name):
        model_info = AVAILABLE_MODELS[model_name]
        cache_dir = self.get_cache_dir(model_name)

        if not os.path.exists(cache_dir):
            return False, "Model directory not found"

        missing_files = []
        for filename in model_info["files"].keys():
            if not os.path.exists(os.path.join(cache_dir, model_info["files"][filename])):
                missing_files.append(filename)

        if missing_files:
            return False, f"Missing model files: {', '.join(missing_files)}"

        return True, "Model cache verified"

    def download_model(self, model_name):
        model_info = AVAILABLE_MODELS[model_name]
        cache_dir = self.get_cache_dir(model_name)

        try:
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Downloading {model_name} model files...")

            for filename in model_info["files"].keys():
                print(f"Downloading {filename}...")
                hf_hub_download(
                    repo_id=model_info["repo_id"],
                    filename=filename,
                    local_dir=cache_dir,
                    local_dir_use_symlinks=False
                )

            return True, "Model files downloaded successfully"

        except Exception as e:
            return False, f"Error downloading model files: {str(e)}"

    def clear_model(self):
        if self.model is not None:
            self.model.cpu()
            del self.model
            self.model = None
            self.current_model_version = None
            torch.cuda.empty_cache()
            print("Model cleared from memory")


class MeissonicModel(BaseModelLoader):
    def __init__(self):
        super().__init__()

    def load_model(self, model_name):
        if self.current_model_version != model_name:
            self.clear_model()

            cache_dir = self.get_cache_dir(model_name)
            self.model = AutoModelForImageSegmentation.from_pretrained(
                cache_dir,
                trust_remote_code=True,
                local_files_only=True
            )

            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

            torch.set_float32_matmul_precision('high')
            self.model.to(device)
            self.current_model_version = model_name

    def process_image(self, images, model_name, params):
        try:
            self.load_model(model_name)

            # Prepare batch processing
            transform_image = transforms.Compose([
                transforms.Resize((params["process_res"], params["process_res"])),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            # Ensure input is in list format
            if isinstance(images, torch.Tensor):
                if len(images.shape) == 3:
                    images = [images]
                else:
                    images = [img for img in images]

            # Store original image sizes
            original_sizes = [tensor2pil(img).size for img in images]

            # Batch process transformations
            input_tensors = [transform_image(tensor2pil(img)).unsqueeze(0) for img in images]
            input_batch = torch.cat(input_tensors, dim=0).to(device)

            with torch.no_grad():
                results = self.model(input_batch)[-1].sigmoid().cpu()
                masks = []

                # Process each result and resize back to original dimensions
                for i, (result, (orig_w, orig_h)) in enumerate(zip(results, original_sizes)):
                    result = result.squeeze()
                    result = result * (1 + (1 - params["sensitivity"]))
                    result = torch.clamp(result, 0, 1)

                    # Resize back to original dimensions
                    result = F.interpolate(result.unsqueeze(0).unsqueeze(0),
                                           size=(orig_h, orig_w),
                                           mode='bilinear').squeeze()

                    masks.append(tensor2pil(result))

                return masks

        except Exception as e:
            handle_model_error(f"Error in batch processing: {str(e)}")


class Meissonic:
    """
    Meissonic Node: Non-Autoregressive Masked Image Modeling (MIM) text-to-image generation

    """

    def __init__(self):
        self.models = {
            "RMBG-2.0": MeissonicModel(),
            # "INSPYRENET": InspyrenetModel(),
            # "BEN": BENModel()
        }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MEISSONIC", {"tooltip": "The model used for denoising the input latent."}),
                # "clip": ("CLIP", {}),
                # "vae": ("VAE", {}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000,
                                  "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01,
                                  "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                # "positive": ("CONDITIONING", {
                #     "tooltip": "The conditioning describing the attributes you want to include in the image."}),
                # "negative": ("CONDITIONING", {
                #     "tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_image"
    CATEGORY = "Meissonic/Meissonic"

    def run_inference(self, pipe, prompt, negative_prompt, resolution, cfg, steps):
        return \
        pipe(prompt=prompt, negative_prompt=negative_prompt, height=resolution, width=resolution, guidance_scale=cfg,
             num_inference_steps=steps).images[0]

    def process_image(self, model, **params):
        vq_model, tokenizer, text_encoder, transformer, scheduler = model
        # base_path = folder_paths.get_folder_paths("meissonic")[0]
        # model_path = f"{base_path}/meissonic-base"
        #
        # transformer = Transformer2DModel.from_pretrained(model_path, subfolder="transformer", )
        # vq_model = VQModel.from_pretrained(model_path, subfolder="vqvae", )
        # # text_encoder = CLIPTextModelWithProjection.from_pretrained(model_path,subfolder="text_encoder",)
        # text_encoder = CLIPTextModelWithProjection.from_pretrained(  # using original text enc for stable sampling
        #     "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        # )
        # tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", )
        # scheduler = Scheduler.from_pretrained(model_path, subfolder="scheduler", )

        pipe = Pipeline(vq_model, tokenizer=tokenizer, text_encoder=text_encoder, transformer=transformer, scheduler=scheduler)
        pipe = pipe.to(device)

        steps = 32
        CFG = 9
        resolution = 512
        negative_prompt = "worst quality, low quality, low res, blurry, distortion, watermark, logo, signature, text, jpeg artifacts, signature, sketch, duplicate, ugly, identifying mark"
        positive_prompt = "A large body of water with a rock in the middle and mountains in the background."

        image = self.run_inference(pipe, positive_prompt, negative_prompt, resolution, CFG, steps)

        print(f"\n Image type: {type(image)}")

        width = 1024
        height = 1024
        color = 0xFF4500
        r = torch.full([1, height, width, 1], ((color >> 16) & 0xFF) / 0xFF)
        g = torch.full([1, height, width, 1], ((color >> 8) & 0xFF) / 0xFF)
        b = torch.full([1, height, width, 1], ((color) & 0xFF) / 0xFF)
        return (torch.cat((r, g, b), dim=-1),)
        # out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))


from core.transformer import Transformer2DModel
from core.pipeline import Pipeline
from core.scheduler import Scheduler
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from diffusers import VQModel


class CheckpointLoaderMeissonic:
    @classmethod
    def INPUT_TYPES(cls):
        paths = []
        for search_path in folder_paths.get_folder_paths("meissonic"):
            if os.path.exists(search_path):
                for root, subdir, files in os.walk(search_path, followlinks=True):
                    if "model_index.json" in files:
                        paths.append(os.path.relpath(root, start=search_path))

        return {"required": {"model_path": (paths,), }}

    RETURN_TYPES = ("MEISSONIC",)
    OUTPUT_TOOLTIPS = ("The model used for denoising latents.",
                       "The CLIP model used for encoding text prompts.",
                       "The VAE model used for encoding and decoding images to and from latent space.")
    FUNCTION = "load_checkpoint"

    CATEGORY = "loaders"
    DESCRIPTION = "Loads a MIM model checkpoint, MIM models are used to denoise latents."

    def load_checkpoint(self, model_path):
        base_path = folder_paths.get_folder_paths("meissonic")[0]
        model_path = f"{base_path}/{model_path}"

        model = Transformer2DModel.from_pretrained(model_path, subfolder="transformer", )
        vq_model = VQModel.from_pretrained(model_path, subfolder="vqvae", )
        # text_encoder = CLIPTextModelWithProjection.from_pretrained(model_path,subfolder="text_encoder",)
        text_encoder = CLIPTextModelWithProjection.from_pretrained(  # using original text enc for stable sampling
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        )
        tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", )
        scheduler = Scheduler.from_pretrained(model_path, subfolder="scheduler", )

        out = (vq_model, tokenizer, text_encoder, model, scheduler)

        return (out,)


# Node Mapping
NODE_CLASS_MAPPINGS = {
    "Meissonic": Meissonic,
    "CheckpointLoaderMeissonic": CheckpointLoaderMeissonic
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Meissonic": "Meissonic KSampler",
    "CheckpointLoaderMeissonic": "Load Meissonic Checkpoint"
}
