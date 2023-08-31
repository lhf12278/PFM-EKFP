
import os
import shutil
import time
import logging
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import random
import numpy as np
from utils.metric import AverageMeter, Loss, constraints_loss
from config import data_config, network_config, lr_scheduler, get_image_unique_cuhk
from train_config import config
from solver import WarmupMultiStepLR, RandomErasing
from test import test
import torch.multiprocessing as mp

mp.set_sharing_strategy('file_system')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def set_seed(args):
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def save_checkpoint(state, epoch, dst, is_best):
    filename = os.path.join(dst, "best_model") + ".pth.tar"
    torch.save(state, filename)
    if is_best:
        dst_best = os.path.join(dst, "model_best", str(epoch)) + ".pth.tar"
        shutil.copyfile(filename, dst_best)


def train(epoch, train_loader, network, optimizer, compute_loss,  args):
    train_loss = AverageMeter()
    image_pre = AverageMeter()
    text_pre = AverageMeter()

    # switch to train mode
    network.train()
    global_step = 0
    start_time = time.time()

    for step, (images, captions, labels) in enumerate(train_loader):
        (
            tokens,
            segments,
            input_masks,
            caption_length,
        ) = network.module.language_model.pre_process(captions)
        tokens = tokens.cuda()
        segments = segments.cuda()
        input_masks = input_masks.cuda()
        images = images.cuda()
        labels = labels.cuda()

        img_feats, txt_feats,img_rest,txt_rest,imgP,txtP,out_img,out_txt = network(
            images, tokens, segments, input_masks,labels
        )
        (
            cmpm_loss,
            cmpc_loss,
            idloss,
            l2loss,
            loss,
            image_precision,
            text_precision,
            pos_avg_sim,
            neg_avg_sim,
        ) = compute_loss(
            img_feats, txt_feats,img_rest,txt_rest,imgP,txtP,out_img,out_txt, labels
        )

        current_lr = []
        for params in optimizer.param_groups:
            current_lr.append(params['lr'])

        global_step += 1
        if global_step % 50 == 0:
            print(
                "Epoch: {}/{}, Step: {}/{}, Lr: {}, cmpm_loss: {:.3f}, cmpc_loss: {:.3f}, idloss: {:.3f},l2loss: {:.3f},loss: {:.3f}, Time/step: {:.4f}".format(
                    epoch,
                    args.num_epoches,
                    global_step,
                    len(train_loader),
                    "-".join([str('%.9f'%itm) for itm in sorted(current_lr)]),
                    cmpm_loss,
                    cmpc_loss,
                    idloss,
                    l2loss,
                    loss,
                    (time.time() - start_time) / 50
                )
            )
            start_time = time.time()

        # compute gradient and do ADAM step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), images.shape[0])
        image_pre.update(image_precision, images.shape[0])
        text_pre.update(text_precision, images.shape[0])

    return train_loss.avg, image_pre.avg, text_pre.avg


def main(args):

    set_seed(args)

    # transform
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transform = transforms.Compose([transforms.ToTensor(), normalize])
    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=3),
            transforms.ToTensor(),
            normalize,
        ]
    )
    cap_transform = None
    # data
    train_loader = data_config(
        args.dataset,
        args.image_dir,
        args.anno_dir,
        args.batch_size,
        "train",
        100,
        train_transform,
        cap_transform=cap_transform,
    )

    test_loader = data_config(
        args.dataset, args.image_dir, args.anno_dir, 64, "val", 100, test_transform
    )
    #test_loadertxt = data_config(
    #    args.dataset, args.image_dir, args.anno_dir, 64, "val", 100, test_transform
   # )
    unique_image = get_image_unique_cuhk(
            args.image_dir, args.anno_dir, 64, "val", 100, test_transform
        )
  
    # loss
    compute_loss = Loss(args)
    nn.DataParallel(compute_loss).cuda()

    # network
    network, optimizer = network_config(
        args, "train", compute_loss.parameters(), args.resume, args.model_path
    )

    # lr_scheduler
    scheduler = WarmupMultiStepLR(optimizer, (25, 35,45), 0.1, 0.01, 10, "linear")    # (20, 25, 35)
    ac_t2i_top1_best = 0.0
    best_epoch = 0

    for epoch in range(1, args.num_epoches + 1 - args.start_epoch):
        network.train()
        train_loss, image_precision, text_precision = train(
            args.start_epoch + epoch,
            train_loader,
            network,
            optimizer,
            compute_loss,
            args,
        )

        is_best = False
        logging.info(
            "Epoch {}/{} Finished, train_loss: {:.3f}, image_precision: {:.3f}, text_precision: {:.3f}".format(
                args.start_epoch + epoch, args.num_epoches, train_loss, image_precision, text_precision
            )
        )
        scheduler.step()

        if epoch % 5 == 0:
            (
                ac_top1_i2t,
                ac_top5_i2t,
                ac_top10_i2t,
                ac_top1_t2i,
                ac_top5_t2i,
                ac_top10_t2i,
                mAP,
                test_time,
            ) = test(test_loader, network, args, unique_image, epoch)

            state = {
                "network": network.state_dict(),
                "optimizer": optimizer.state_dict(),
                "W": compute_loss.W,
                "epoch": args.start_epoch + epoch,
            }

            if ac_top1_t2i > ac_t2i_top1_best:
                best_epoch = epoch
                ac_t2i_top1_best = ac_top1_t2i
                save_checkpoint(state, epoch, args.checkpoint_dir, is_best)

            logging.info("Text-to-Image:")
            logging.info(" R@1: {:.2f}, R@5: {:.2f}, R@10: {:.2f},mAP: {:.2f}".format(
                    ac_top1_t2i, ac_top5_t2i, ac_top10_t2i,mAP))
            logging.info("Image-to-Text:")
            logging.info(" R@1: {:.2f}, R@5: {:.2f}, R@10: {:.2f}".format(
                    ac_top1_i2t, ac_top5_i2t, ac_top10_i2t))

    logging.info("Train Finished!")
    logging.info("The best epoch:{}, the R@1 is: {:.2f}".format(best_epoch, ac_t2i_top1_best))
    logging.info(args.checkpoint_dir)


if __name__ == "__main__":
    args = config()
    main(args)
