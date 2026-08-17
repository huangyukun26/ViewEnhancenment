from accelerate import Accelerator

import os
import argparse
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.logging import get_logger
from diffusers.training_utils import EMAModel, cast_training_params
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from contextlib import nullcontext

import PIL

import numpy as np

from packaging import version

from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils import  convert_state_dict_to_diffusers

from transformers import CLIPTextModel, CLIPTokenizer

from safetensors.torch import load_file

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for InstructPix2Pix.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="timbrooks/instruct-pix2pix",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )

    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )


    parser.add_argument(
        "--original_image_column",
        type=str,
        default="input_image",
        help="The column of the dataset containing the original image on which edits where made.",
    )
    parser.add_argument(
        "--edited_image_column",
        type=str,
        default="output_image",
        help="The column of the dataset containing the edited image.",
    )
    parser.add_argument(
        "--edit_prompt_column",
        type=str,
        default="edit_prompt",
        help="The column of the dataset containing the edit instruction.",
    )
    parser.add_argument(
        "--val_image_url",
        type=str,
        default="/mnt/e/Data/ProjectData/2025.WindowViewPressConference/s1_view_image_beautification/s8_prediction_30examples/79_full/input/",
        help="URL to the original image that you would like to edit (used during inference for debugging purposes).",
    )

    # Hong Kong, buildings, facade, clear, not restructured, generated contents for distortion
    parser.add_argument(
        "--validation_prompt", type=str, default="Hong Kong, buildings, clear, no restructured facade layouts, generated contents for distortion", help="A prompt that is sampled during training for inference."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )


    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/e/Data/ProjectData/2025.WindowViewPressConference/s1_view_image_beautification/s6_instruct-pix2pix/inference/",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )

    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )


    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")


    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )


    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action='store_false', default=True, help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )

    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank


    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def download_image(url):
    files=os.listdir(url)
    result=[]
    for file in files:
        image = PIL.Image.open(url+file)
        image = PIL.ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        result.append(image)
    return result


def test_image(args, folder,name):
    file=args.val_image_url+folder+"/"+name


    image = PIL.Image.open(file)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    return image, name

import time

def log_validation(
    pipeline,
    args,
    accelerator,
    generator,
    logger
):
    print(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device)
    # pipeline.set_progress_bar_config(disable=True)

    # # run inference
    # original_images = download_image(args.val_image_url)


# Improve the image generated from 3D photorealistic city models in Hong Kong by reducing distortion in building structures, enhancing the resolution, and using reasonable imagination to fill in missing details and enhance the scene plausibly
# Improve the image generated from 3D photorealistic city models in Hong Kong by reducing distortion in building structures, enhancing the resolution, and flattening the surfaces of buildings to create a smoother and more realistic appearance.
# Improve the image generated from 3D photorealistic city models in Hong Kong by reducing distortion in building structures, enhancing the resolution, and avoiding any imaginative additions or alterations 
    
    from IPython import embed
    embed()

    # t1=time.time()

    # folder=""
    # name="592_15_SKEYPPEAPE_1_25_14_C_MBed_114.148279_22.280424_187.6678_27.568647_patch.png"
    # validation_prompt="Improve the image generated from 3D photorealistic city models in Hong Kong by reducing distortion in building structures"
    # num_inference_steps=50
    # # Bigger, closer to the original image; Smaller than 1, more diverse; Larger than 5, closer the original image
    # image_guidance_scale=1.5

    # # Bigger, closer to prompt
    # guidance_scale=7
    # img, name=test_image(args, folder,name)
    # inference_images(1,args, img, generator, accelerator, pipeline, name, validation_prompt, num_inference_steps, image_guidance_scale, guidance_scale)

    # t2=time.time()

    # print(t2-t1)

    folder=""
    validation_prompt="Edit the image generated from 3D photorealistic city models in Hong Kong by reducing distortion in building structures and enhancing the resolution"
    num_inference_steps=50
    # Bigger, closer to the original image; Smaller than 1, more diverse; Larger than 5, closer the original image
    image_guidance_scale=1.5

    # Bigger, closer to prompt
    guidance_scale=7

    files=os.listdir(args.val_image_url)
    for file in files:
        img, name=test_image(args, folder,file)
        inference_images(1,args, img, generator, accelerator, pipeline, name, validation_prompt, num_inference_steps, image_guidance_scale, guidance_scale)

    


def inference_images(image_num,args, img, generator, accelerator, pipeline,name, validation_prompt, num_inference_steps,image_guidance_scale, guidance_scale):

    edited_images = []
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    with autocast_ctx:
        # for img1 in tqdm(original_images):
            for _ in range(image_num):
                edited_images.append(
                    pipeline(
                        validation_prompt,
                        image=img,
                        num_inference_steps=num_inference_steps,
                        image_guidance_scale=image_guidance_scale,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        width=900,
                        height=900
                    ).images[0]
                )

    for i in range(len(edited_images)):
        edited_images[i].save("/mnt/e/Data/ProjectData/2025.WindowViewPressConference/s1_view_image_beautification/s8_prediction_30examples/79_full/improved_patches/"+str(i)+"_"+name)



    # return edited_images



def unwrap_model(model, accelerator):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

def inference():

        args = parse_args()
        logger = get_logger(__name__, log_level="INFO")
        logging_dir = os.path.join(args.output_dir, args.logging_dir)

        accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)        
        
        accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )
        
        # Disable AMP for MPS.
        if torch.backends.mps.is_available():
            accelerator.native_amp = False



        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)



        if args.seed is not None:
            set_seed(args.seed)

        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
        )
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        )
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
        )


        logger.info("Initializing the InstructPix2Pix UNet from the pretrained UNet.")
        in_channels = 8
        out_channels = unet.conv_in.out_channels
        unet.register_to_config(in_channels=in_channels)

        with torch.no_grad():
            new_conv_in = nn.Conv2d(
                in_channels, out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
            )
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :in_channels, :, :].copy_(unet.conv_in.weight)
            unet.conv_in = new_conv_in

        # Freeze vae, text_encoder and unet
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        unet.requires_grad_(False)

        # referred to https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py
        # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Freeze the unet parameters before adding adapters
        unet.requires_grad_(False)


        unet_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )



        # Move unet, vae and text_encoder to device and cast to weight_dtype
        unet.to(accelerator.device, dtype=weight_dtype)

        # Add adapter and make sure the trainable params are in float32.
        unet.add_adapter(unet_lora_config)

        output_dir = "/mnt/c/large_scale/checkpoint-1440/"
        main_weights_path = os.path.join(output_dir, "unet", "diffusion_pytorch_model.safetensors")

        unet.load_state_dict(load_file(main_weights_path))


        if args.mixed_precision == "fp16":
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(unet, dtype=torch.float32)

        # Create EMA for the unet.
        if args.use_ema:
            ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)

        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warning(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")
            

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True


        if args.use_ema:
            ema_unet.to(accelerator.device)

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move text_encode and vae to gpu and cast to weight_dtype
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        vae.to(accelerator.device, dtype=weight_dtype)


        # accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if args.use_ema:
                ema_unet.copy_to(unet.parameters())

            # store only LORA layers
            unet = unet.to(torch.float32)

            pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=unwrap_model(text_encoder, accelerator),
                vae=unwrap_model(vae, accelerator),
                unet=unwrap_model(unet, accelerator),
                revision=args.revision,
                variant=args.variant,
            )

            pipeline.load_lora_weights(output_dir+"pytorch_lora_weights.safetensors")

            images = None
            if (args.val_image_url is not None) and (args.validation_prompt is not None):
                images = log_validation(
                    pipeline,
                    args,
                    accelerator,
                    generator,
                    logger
                )


                # for i in range(len(images)):
                #     images[i].save("/mnt/e/Data/ProjectData/2025.WindowViewPressConference/s1_view_image_beautification/s6_instruct-pix2pix/test/"+str(i)+".png")

        


if __name__=="__main__":
    inference()
