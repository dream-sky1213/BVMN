import torch,gc
from torch.autograd import Variable
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from datetime import datetime
from utils.tdataloader import get_loader
from utils.utils import clip_gradient, AvgMeter, poly_lr
import torch.nn.functional as F
import torch.nn as nn
from networks.vision_mamba import MambaUnet as VIM_seg
from config import get_config

gc.collect()
torch.cuda.empty_cache()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda")
from torch.nn.modules.loss import CrossEntropyLoss
from utils import losses
torch.backends.cudnn.benchmark = True



def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()

step = 0
best_mae = 1
best_epoch = 0


def train(train_loader, model, optimizer, epoch):
    global step
    model.train()
    # ce_loss = CrossEntropyLoss()
    # dice_loss = losses.DiceLoss(1)
    loss_all = 0
    epoch_step = 0
    save_path = 'results/checkpoints/bvm_cb2/wbce{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    # loss_record1, loss_recorde, loss_record2 = AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        # ---- data prepare ----
        images, gts = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()
        # edges = Variable(edges).cuda()
        # ---- forward ----

        # pred0, probx, u  = model(images)
        pred = model(images)

        mask_loss0 = dice_loss(pred[0], gts)
        mask_loss1 = dice_loss(pred[1], gts)
        mask_loss2 = dice_loss(pred[2], gts)
        mask_loss3 = dice_loss(pred[3], gts)
        # mask_loss0 = structure_loss(pred[0], gts)
        # mask_loss1 = structure_loss(pred[1], gts)
        # mask_loss2 = structure_loss(pred[2], gts)
        # mask_loss3 = structure_loss(pred[3], gts)
        # mask_loss0 = dice_loss(pred, gts)

        # loss_unc = uncertainty_loss(probx, gts)
        mask_loss = mask_loss0 + mask_loss1 + mask_loss2 + mask_loss3
        # mask_loss = mask_loss0
        #
        # total_loss = 2 * mask_loss + loss_unc
        total_loss = mask_loss
        # ---- backward ----
        torch.autograd.set_detect_anomaly(True)
        total_loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        step = step + 1
        epoch_step = epoch_step + 1
        loss_all = loss_all + total_loss.data

        # ---- train visualization ----
        if i % 60 == 0:
            print(
                '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}],'
                'MaskLoss0: {:0.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step,
                           mask_loss0.data))
        #
        # if i % 60 == 0:
        #     print(
        #         '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}],'
        #         'MaskLoss0: {:0.4f}, '
        #         'uLoss0: {:0.4f}, '.
        #             format(datetime.now(), epoch, opt.epoch, i, total_step,
        #                    mask_loss0.data, loss_unc.data))

    if (epoch + 1) % 5 == 0 or (epoch + 1) == opt.epoch:
        torch.save(model.state_dict(), save_path + 'UBNet-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'UBNet-%d.pth' % epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
    parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
    parser.add_argument('--num_classes', type=int,  default=1,
                    help='output channel of network')
    parser.add_argument('--epoch', type=int,
                        default=150, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--decay_epoch', type=int, default=50, help='decay')
    parser.add_argument('--batch_size', type=int,
                        default=8, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=416, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--train_path', type=str,
                        default='/home/server-816/Data_Hardisk/wkk/BGNet-master/data/data_v2/TrainDataset/Nist', help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='Nist')
    parser.add_argument('--patch_size', type=list,  default=[416, 416],
                    help='patch size of network input')
                    
    parser.add_argument(
    '--cfg', type=str, default="/home/server-816/Data_Hardisk/wkk/munet/code/vmamba_tiny.yaml", help='path to config file', )
    opt = parser.parse_args()
    config = get_config(opt)

    # ---- build models ----
    model = VIM_seg(config, img_size=opt.patch_size,
                     num_classes=opt.num_classes).cuda()
    model.load_from(config)


    params = model.parameters()


    image_root = '{}/f/'.format(opt.train_path)
    gt_root = '{}/m/'.format(opt.train_path)
    # edge_root = '{}/e/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batch_size, trainsize=opt.trainsize)
    total_step = len(train_loader)
    # Conversion from epoch to step/iter
    decay_iter = opt.decay_epoch * total_step

    # Optimizers
    optimizer = torch.optim.Adam(params, opt.lr, betas=(opt.b1, opt.b2))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                                step_size=decay_iter,
                                                                gamma=0.5)
    torch.autograd.set_detect_anomaly(True)

    print("Start Training")

    for epoch in range(opt.epoch):
        poly_lr(optimizer, opt.lr, epoch, opt.epoch)
        train(train_loader, model, optimizer, epoch)
