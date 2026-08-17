import argparse
import torch
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionInstructPix2PixPipeline, AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.utils.import_utils import is_xformers_available

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for InstructPix2Pix with LoRA.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="timbrooks/instruct-pix2pix",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--original_image_path",
        type=str,
        default="/mnt/e/Data/ProjectData/2025.WindowViewPressConference/s1_view_image_beautification/s7_make_datasets/s1_trainingset/image_patches_small/single/10_42_SKPPWSPXPS_3_12_3_A_Dining_114.140078_22.280819_176.97_277.993584_test.png",
        help="Path to the original image that you would like to edit.",
    )
    parser.add_argument(
        "--edit_prompt", 
        type=str, 
        default="Hong Kong, buildings, clear, no restructured facade layouts, no generated contents for distortion",
        help="A prompt that describes the desired edit.",
    )
    parser.add_argument(
        "--output_image_path",
        type=str,
        default="/mnt/e/Data/ProjectData/2025.WindowViewPressConference/s1_view_image_beautification/s6_instruct-pix2pix/test/111.png",
        help="Path to save the edited image.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of inference steps for the diffusion process.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7,
        help="Guidance scale for classifier-free guidance.",
    )
    parser.add_argument(
        "--image_guidance_scale",
        type=float,
        default=1.5,
        help="Guidance scale for the image condition.",
    )
    parser.add_argument(
        "--lora_weights_path",
        type=str,
        default="/mnt/e/Data/ProjectData/2025.WindowViewPressConference/s1_view_image_beautification/s6_instruct-pix2pix/finetuned/checkpoint-1120",
        help="Path to the directory containing the trained LoRA weights.",
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision)

    # Load the pipeline
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
    ).to("cuda")

    # Load the LoRA weights
    pipeline.load_lora_weights(args.lora_weights_path)

    # Load the original image
    original_image = Image.open(args.original_image_path).convert("RGB")

    # Run inference
    edited_image = pipeline(
        args.edit_prompt,
        image=original_image,
        num_inference_steps=args.num_inference_steps,
        image_guidance_scale=args.image_guidance_scale,
        guidance_scale=args.guidance_scale,
        generator=
    ).images[0]

    # Save the edited image
    edited_image.save(args.output_image_path)
    print(f"Edited image saved to {args.output_image_path}")

if __name__ == "__main__":
    main()