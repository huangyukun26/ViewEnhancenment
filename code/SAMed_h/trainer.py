import argparse
import logging
import os
import random
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from utils import DiceLoss, Focal_loss
from torchvision import transforms
from icecream import ic


def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, focal_loss, dice_weight:float=0.4, focal_weight=0.4):
    low_res_logits = outputs['low_res_logits']
    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss_focal=focal_loss(low_res_logits, low_res_label_batch)

    loss = (1 - dice_weight-focal_weight) * loss_ce + dice_weight * loss_dice+focal_weight*loss_focal
    return loss, loss_ce, loss_dice, loss_focal


def trainer_landfill(args, model, snapshot_path, multimask_output, low_res):
    from evaluator import Evaluator
    from datasets.dataset_landfill import Landfill_dataset, RandomGenerator, transpose_self
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Landfill_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size], low_res=[low_res, low_res])]))
    print("The length of train set is: {}".format(len(db_train)))

    db_val = Landfill_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="val",transform=transforms.Compose([transpose_self(output_size=[args.img_size, args.img_size], low_res=[low_res, low_res])]))


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    
    valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    


    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes + 1)
    focal_loss=Focal_loss(num_classes=num_classes + 1)


    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr
    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)  # Even pass the model.parameters(), the `requires_grad=False` layers will not update
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs

    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    iterator = range(max_epoch)

    evaluator=Evaluator(2)

    best_pred=0
    for epoch_num in iterator:
        train_loss=0
        train_loss_ce=0
        train_loss_dice=0
        train_loss_focal=0

        val_loss=0

        for i_batch, sampled_batch in enumerate(tqdm(trainloader)):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # [b, c, h, w], [b, h, w]
            low_res_label_batch = sampled_batch['low_res_label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            low_res_label_batch = low_res_label_batch.cuda()
            assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'
            outputs = model(image_batch, multimask_output, args.img_size)

            loss, loss_ce, loss_dice, loss_focal = calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss,focal_loss, args.dice_param, args.focal_param)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            train_loss+=loss.item()
            train_loss_ce+=loss_ce.item()
            train_loss_dice+=loss_dice.item()
            train_loss_focal+=loss_focal.item()

            # logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

        writer.add_scalar('info/lr', lr_, epoch_num)
        writer.add_scalar('info/total_train_loss', train_loss/len(trainloader), epoch_num)
        writer.add_scalar('info/train_loss_ce', train_loss_ce/len(trainloader), epoch_num)
        writer.add_scalar('info/train_loss_dice', train_loss_dice/len(trainloader), epoch_num)
        writer.add_scalar('info/train_loss_focal', train_loss_focal/len(trainloader), epoch_num)

        print('Train Loss: %.3f' % loss)

        # validation
        model.eval()
        evaluator.reset()

        val_loss=0
        val_loss_ce=0
        val_loss_dice=0
        val_loss_focal=0

        for i_batch, sampled_batch in enumerate(tqdm(valloader)):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # [b, c, h, w], [b, h, w]
            low_res_label_batch = sampled_batch['low_res_label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            low_res_label_batch = low_res_label_batch.cuda()
            assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'
            outputs = model(image_batch, multimask_output, args.img_size)
            loss, loss_ce, loss_dice, loss_focal = calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss,focal_loss, args.dice_param, args.focal_param)


            # logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            val_loss+=loss.item()
            val_loss_ce+=loss_ce.item()
            val_loss_dice+=loss_dice.item()
            val_loss_focal+=loss_focal.item()

            output_masks = outputs['masks']
            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = out

            label=label_batch.squeeze(0).cpu().detach().numpy()

            evaluator.add_batch(label, pred)

        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

        writer.add_scalar('info/total_val_loss', val_loss/len(valloader), epoch_num)
        writer.add_scalar('info/val_loss_ce', loss_ce/len(valloader), epoch_num)
        writer.add_scalar('info/val_loss_dice', loss_dice/len(valloader), epoch_num)
        writer.add_scalar('info/val_loss_focal', loss_focal/len(valloader), epoch_num)


        writer.add_scalar('info/mIoU', mIoU, epoch_num)
        writer.add_scalar('info/Acc', Acc, epoch_num)
        writer.add_scalar('info/Acc_class', Acc_class, epoch_num)
        writer.add_scalar('info/fwIoU', FWIoU, epoch_num)

        
        print('Validation:')
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % loss)

        if mIoU > best_pred:
            best_pred = mIoU
            save_mode_path = os.path.join(snapshot_path, 'checkpoint_best.pth')
            model.save_lora_parameters(save_mode_path)


        save_interval = 20 # int(max_epoch/6)
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
