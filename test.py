import os
import sys
import time
import shutil
import logging
import gc
import torch
import torchvision.transforms as transforms
from utils.metric import AverageMeter, compute_topk,get_mAP
from test_config import config
from config import data_config, network_config, get_image_unique_cuhk


def test(data_loader, network, args, unique_image, epoch=0):

    batch_time = AverageMeter()
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    test_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), normalize]
    )
    # switch to evaluate mode
    network.eval()
    max_size = len(data_loader.dataset.test_captions)
    img_feat_bank = torch.zeros(max_size, 2, args.feature_size)
    text_feat_bank = torch.zeros(max_size, 2, args.feature_size)
    labels_bank = torch.zeros(max_size)

    index = 0
    with torch.no_grad():
        end = time.time()
        for step, (images, captions, labels) in enumerate(data_loader):

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
            interval = images.shape[0]

            img_output, text_output = network(
                images, tokens, segments, input_masks,labels)

            img_feat_bank[index : index + interval] = img_output
            text_feat_bank[index : index + interval] = text_output
            labels_bank[index : index + interval] = labels
            batch_time.update(time.time() - end)
            end = time.time()
            index = index + interval
        unique_image = torch.tensor(unique_image) == 1

        mAP = get_mAP(
            img_feat_bank[unique_image],
            text_feat_bank,
            labels_bank[unique_image],
            labels_bank,
            )

        result, score = compute_topk(
            img_feat_bank[unique_image],
            text_feat_bank,
            labels_bank[unique_image],
            labels_bank,
            [1,5, 10],
            True,
        )
        (
            ac_top1_i2t,
            ac_top5_i2t,
            ac_top10_i2t,
            ac_top1_t2i,
            ac_top5_t2i,
            ac_top10_t2i,
        ) = result
        return (
            ac_top1_i2t,
            ac_top5_i2t,
            ac_top10_i2t,
            ac_top1_t2i,
            ac_top5_t2i,
            ac_top10_t2i,
            mAP,
            batch_time.avg,
        )

def main(args):
    # need to clear the pipeline
    # top1 & top10 need to be chosen in the same params ???
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    test_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), normalize]
    )

    test_loader = data_config(
        args.dataset, args.image_dir, args.anno_dir, 64, "test", 100, test_transform
    )
    unique_image = get_image_unique_cuhk(
            args.image_dir, args.anno_dir, 64, "test", 100, test_transform
        )
   
    ac_i2t_top1_best = 0.0
    ac_i2t_top10_best = 0.0
    ac_t2i_top1_best = 0.0
    ac_t2i_top10_best = 0.0
    ac_t2i_top5_best = 0.0
    ac_i2t_top5_best = 0.0
    i2t_models = os.listdir(args.model_path)
    i2t_models.sort()
    model_list = []
    for i2t_model in i2t_models:
        if i2t_model.split(".")[0] == "best_model":
            model_list.append((i2t_model.split(".")[0]))
        model_list.sort()
    # for i2t_model in i2t_models:
    #     if i2t_model.split(".")[0] != "model_best":
    #         model_list.append((i2t_model.split(".")[0]))
    #     model_list.sort()

    logging.info("Testing on dataset: {}".format(args.anno_dir))
    for i2t_model in model_list:
        model_file = os.path.join(args.model_path, str(i2t_model) + ".pth.tar")
        if os.path.isdir(model_file):
            continue
        epoch = i2t_model
        network, _ = network_config(args, "test", None, True, model_file)

        (
            ac_top1_i2t,
            ac_top5_i2t,
            ac_top10_i2t,
            ac_top1_t2i,
            ac_top5_t2i,
            ac_top10_t2i,
            mAP_t2i,
            test_time,
        ) = test(test_loader, network, args, unique_image)
        if ac_top1_t2i > ac_t2i_top1_best:
            ac_i2t_top1_best = ac_top1_i2t
            ac_i2t_top5_best = ac_top5_i2t
            ac_i2t_top10_best = ac_top10_i2t

            ac_t2i_top1_best = ac_top1_t2i
            ac_t2i_top5_best = ac_top5_t2i
            ac_t2i_top10_best = ac_top10_t2i
            dst_best = (
                os.path.join(args.checkpoint_dir, "model_best", str(epoch)) + ".pth.tar"
            )

        logging.info("epoch:{}".format(epoch))
        logging.info(
            "top1_t2i: {:.3f}, top5_t2i: {:.3f}, top10_t2i: {:.3f}, mAP_t2i: {:.3f},top1_i2t: {:.3f}, top5_i2t: {:.3f}, top10_i2t: {:.3f}".format(
                ac_top1_t2i,
                ac_top5_t2i,
                ac_top10_t2i,
                mAP_t2i,
                ac_top1_i2t,
                ac_top5_i2t,
                ac_top10_i2t,
            )
        )
    logging.info(
        "t2i_top1_best: {:.3f}, t2i_top5_best: {:.3f}, t2i_top10_best: {:.3f}, i2t_top1_best: {:.3f}, i2t_top5_best: {:.3f}, i2t_top10_best: {:.3f}".format(
            ac_t2i_top1_best,
            ac_t2i_top5_best,
            ac_t2i_top10_best,
            ac_i2t_top1_best,
            ac_i2t_top5_best,
            ac_i2t_top10_best,
        )
    )
    logging.info(args.model_path)
    logging.info(args.log_dir)


if __name__ == "__main__":
    args = config()
    main(args)
