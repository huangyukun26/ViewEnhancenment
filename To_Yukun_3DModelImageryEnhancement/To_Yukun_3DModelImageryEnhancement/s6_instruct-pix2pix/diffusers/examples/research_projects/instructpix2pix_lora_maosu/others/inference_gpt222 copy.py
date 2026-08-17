import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionInstructPix2PixPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from huggingface_hub import hf_hub_download
from accelerate import Accelerator
from safetensors.torch import load_file
import os


from peft import LoraConfig

# 设置加载文件的路径
output_dir = "/mnt/c/checkpoint-720/"
main_weights_path = os.path.join(output_dir, "unet", "diffusion_pytorch_model.safetensors")
lora_weights_path = os.path.join(output_dir, "pytorch_lora_weights.safetensors")
optimizer_path = os.path.join(output_dir, "optimizer.bin")
random_states_path = os.path.join(output_dir, "random_states_0.pkl")
scheduler_path = os.path.join(output_dir, "scheduler.bin")
unet_config_path = os.path.join(output_dir, "unet", "config.json")

# 加载预训练模型和配置
pretrained_model_name_or_path = "timbrooks/instruct-pix2pix"
revision = None
variant = None
torch_dtype = torch.float32

tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer", revision=revision)
text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder", revision=revision, variant=variant)
vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", revision=revision, variant=variant)
unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet", revision=revision)

# 加载 scheduler、safety_checker 和 feature_extractor
scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler", revision=revision)
safety_checker = StableDiffusionInstructPix2PixPipeline.from_pretrained(pretrained_model_name_or_path, subfolder="safety_checker", revision=revision).safety_checker
feature_extractor = StableDiffusionInstructPix2PixPipeline.from_pretrained(pretrained_model_name_or_path, subfolder="feature_extractor", revision=revision).feature_extractor

# 创建并应用 LoRA 配置
unet_lora_config = LoraConfig(
    r=4,  # 假设 rank 是 8
    lora_alpha=4,  # 假设 lora_alpha 是 8
    init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
)

unet.add_adapter(unet_lora_config)

# 加载保存的UNet权重
unet.load_state_dict(load_file(main_weights_path))

# 创建推理管道
pipeline = StableDiffusionInstructPix2PixPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=scheduler,
    safety_checker=safety_checker,
    feature_extractor=feature_extractor,
)

# 加载优化器状态、随机种子状态和调度器状态
accelerator = Accelerator()
optimizer = torch.optim.AdamW(unet.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# # 加载优化器状态
# optimizer.load_state_dict(torch.load(optimizer_path))

# # 加载调度器状态
# scheduler.load_state_dict(torch.load(scheduler_path))

# 加载随机状态
# with open(random_states_path, "rb") as f:
#     random_states = torch.load(f)
#     torch.set_rng_state(random_states["torch"])
#     if "cuda" in random_states:
#         torch.cuda.set_rng_state(random_states["cuda"])

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

print("Generated image saved to /mnt/e/Data/ProjectData/2025.WindowViewPressConference/s1_view_image_beautification/s6_instruct-pix2pix/test/edited_image.png")