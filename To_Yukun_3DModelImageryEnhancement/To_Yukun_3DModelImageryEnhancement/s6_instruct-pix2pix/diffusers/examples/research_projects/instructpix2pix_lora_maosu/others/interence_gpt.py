import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionInstructPix2PixPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers.utils import  convert_state_dict_to_diffusers
from huggingface_hub import hf_hub_download
from accelerate import Accelerator

from safetensors.torch import load_file
import os

# 设置加载文件的路径
output_dir = "/mnt/e/Data/ProjectData/2025.WindowViewPressConference/s1_view_image_beautification/s6_instruct-pix2pix/finetuned/checkpoint-720/"
# lora_weights_path = f"{output_dir}/pytorch_lora_weights.safetensors"
main_weights_path = os.path.join(output_dir,"unet", "diffusion_pytorch_model.safetensors")
lora_weights_path = os.path.join(output_dir, "pytorch_lora_weights.safetensors")

# 加载预训练模型和配置
pretrained_model_name_or_path = "timbrooks/instruct-pix2pix"
revision = None
variant = None
torch_dtype = torch.float32

tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer", revision=revision)
text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder", revision=revision, variant=variant)
vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", revision=revision, variant=variant)
unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet", revision=revision)
# unet = UNet2DConditionModel.from_pretrained("/mnt/c/unet/config.json")

# 创建推理管道

pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", unet=unet)


# pipeline = StableDiffusionInstructPix2PixPipeline(
#     vae=vae,
#     text_encoder=text_encoder,
#     tokenizer=tokenizer,
#     unet=unet,
#     torch_dtype=torch_dtype
# )

# 准备生成图像的参数
prompt = "Hong Kong, buildings, clear, no restructured facade layouts, no generated contents for distortion"
original_image_path = "/mnt/e/Data/ProjectData/2025.WindowViewPressConference/s1_view_image_beautification/s7_make_datasets/s1_trainingset/image_patches_small/single/10_42_SKPPWSPXPS_3_12_3_A_Dining_114.140078_22.280819_176.97_277.993584_test.png"
num_inference_steps = 20
guidance_scale = 7
image_guidance_scale = 1.5
generator = torch.Generator(device="cuda").manual_seed(42)

# 加载原始图像
from PIL import Image
original_image = Image.open(original_image_path).convert("RGB")

# 生成图像
pipeline = pipeline.to("cuda")
edited_image = pipeline(
    prompt,
    image=original_image,
    num_inference_steps=num_inference_steps,
    image_guidance_scale=image_guidance_scale,
    guidance_scale=guidance_scale,
    generator=generator,
).images[0]

# 保存生成的图像
edited_image.save("/mnt/e/Data/ProjectData/2025.WindowViewPressConference/s1_view_image_beautification/s6_instruct-pix2pix/test/edited_image.png")

print("Generated image saved to /path/to/edited_image.png")