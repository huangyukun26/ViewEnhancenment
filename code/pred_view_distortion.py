import os
import sys
from tqdm import tqdm
import logging
import numpy as np
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils_pred import test_single_volume_windowview
from importlib import import_module
from segment_anything import sam_model_registry
from datasets.dataset_windowview_pred import Windowview_dataset

from icecream import ic


from torchvision import transforms
from datasets.dataset_landfill import transpose_self


def inference(args, multimask_output, db_config, model, test_save_path=None, split=None):
    db_test = db_config['Dataset'](base_dir=args.volume_path, list_dir=args.list_dir, split=split)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info(f'{len(testloader)} test iterations per epoch')
    print(str(len(testloader))+ 'test iterations per epoch')
    model.eval()

    for i_batch, sampled_batch in enumerate(testloader):

            try:
                h, w = sampled_batch['image'].shape[2:]
                image, label, case_name = sampled_batch['image'], sampled_batch['label'], sampled_batch['case_name'][0]
                print(case_name)

                test_single_volume_windowview(image, label, model, classes=args.num_classes, multimask_output=multimask_output,
                                                patch_size=[args.img_size, args.img_size],
                                                test_save_path=test_save_path, case=case_name)
            except ValueError as e:
                print('value_error'+case_name)






    logging.info("Testing Finished!")
    return 1


def config_to_dict(config):
    items_dict = {}
    with open(config, 'r') as f:
        items = f.readlines()
    for i in range(len(items)):
        key, value = items[i].strip().split(': ')
        items_dict[key] = value
    return items_dict

import time
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='The config file provided by the trained model')
    parser.add_argument('--volume_path', type=str, default='/lustre1/g/rec_fx/dp_paper/2025_CentalinePressConference/20250313_79bldg/input/images/')
    parser.add_argument('--dataset', type=str, default='WindowviewDistortion', help='Experiment name')
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--list_dir', type=str, default='/lustre1/g/rec_fx/data_processing/sam_windowview/ImageSets/ImageSetsAll/distortion_hk/', help='list_dir')
    parser.add_argument('--output_dir', type=str, default='/lustre1/g/rec_fx/dp_paper/2025_CentalinePressConference/20250318_viewdistortion/centaline/output/')
    parser.add_argument('--img_size', type=int, default=512, help='Input image size of the network')
    parser.add_argument('--input_size', type=int, default=224, help='The input size for training SAM model')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--is_savenii', action='store_false', default=True, help='Whether to save results during inference')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default='/lustre1/g/rec_fx/data_processing/sam_landfill/foundation_checkpoints/sam_vit_b_01ec64.pth',
                        help='Pretrained checkpoint')
    parser.add_argument('--lora_ckpt', type=str, default='/lustre1/g/rec_fx/model_train_log/sam_windowview_distortion/WindowviewDistortion_512_vit_b_0.5_0.5_2025-03-18_10-54-22/checkpoint_best.pth', help='The checkpoint from LoRA')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='Select one vit model')
    parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder')
    parser.add_argument('--split', type=str, default='centaline')    

    args = parser.parse_args()

    if args.config is not None:
        # overwtite default configurations with config file\
        config_dict = config_to_dict(args.config)
        for key in config_dict:
            setattr(args, key, config_dict[key])

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'WindowviewDistortion': {
            'Dataset': Windowview_dataset,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
            'z_spacing': 1
        }
    }
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                    num_classes=args.num_classes,
                                                                    checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                    pixel_std=[1, 1, 1])
    
    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank).cuda()

    assert args.lora_ckpt is not None
    net.load_lora_parameters(args.lora_ckpt)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    # initialize log
    log_folder = os.path.join(args.output_dir, 'test_log')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + 'log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.is_savenii:
        test_save_path = os.path.join(args.output_dir, args.split)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    t1=time.time()    
    inference(args, multimask_output, dataset_config[dataset_name], net, test_save_path, args.split)
    t2=time.time()
    print(t2-t1)
