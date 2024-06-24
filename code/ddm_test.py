import imageio
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from networks.vision_mamba import MambaUnet as VIM_seg
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from utils.tdataloader import test_dataset
# from utils.trainer import eval_mae, numpy2tensor
from config import get_config

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
parser.add_argument(
    '--cfg', type=str, default="/home/server-816/Data_Hardisk/wkk/munet/code/vmamba_tiny.yaml", help='path to config file', )

parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')
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
                        default='/home/server-816/Data_Hardisk/wkk/BGNet-master/data/data_v2/TrainDataset/IMD', help='path to train dataset')
parser.add_argument('--train_save', type=str,
                        default='IMD')
parser.add_argument('--patch_size', type=list,  default=[416, 416],
                    help='patch size of network input')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--num_classes', type=int,  default=1,
                    help='output channel of network')


parser.add_argument('--testsize', type=int, default=416, help='testing size')
parser.add_argument('--pth_path', type=str, default='/home/server-816/Data_Hardisk/wkk/munet/code/results/checkpoints/bvm_cb2/Nist//UBNet-144.pth')
opt = parser.parse_args()
config = get_config(opt)


def get_IOU(maskpath, resultpath):

    mask = maskpath
    result = resultpath
    # 计算iou7
    S1 = 0  # 交集
    S2 = 0  # 并集
    for i in range(len(mask)):
        if mask[i] > 0.5 and result[i] > 0.5:  ##0~255为由黑到白，根据图片情况自行调整
            S1 = S1 + 1
        if mask[i] > 0.5 or result[i] > 0.5:
            S2 = S2 + 1

    iou = S1 / S2
    f1 = (2 * S1) / (S1 + S2)

    return iou, f1


for _data_name in ['Sharpening']:
    # data_path = '/home/server-816/Data_Hardisk/wkk/BGNet-master/data/DID//{}'.format(_data_name)
    data_path = '/home/server-816/Data_Hardisk/wkk/BGNet-master/data//TestDataset/Nist///{}'.format(_data_name)
    # data_path = '/home/server-816/Data_Hardisk/wkk/BGNet-master/data/data_v2/TestDataset/SOTA/{}'.format(_data_name)
    save_path = './results/out/bvm_rub/{}/'.format(_data_name)
    # os.makedirs(save_path + '/uncertainty_heatmap/', exist_ok=True)
    # os.makedirs(save_path + '/frequency_heatmap/', exist_ok=True)
    opt = parser.parse_args()
    
    model = VIM_seg(config, img_size=opt.patch_size,
                     num_classes=opt.num_classes).cuda()
    model.load_from(config)
    # model = nn.DataParallel(model)
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    # os.makedirs(save_path + '/edge/', exist_ok=True)
    # os.makedirs(save_path + '/uncertainty/', exist_ok=True)
    image_root = '{}/f0.7/'.format(data_path)
    gt_root = '{}/m/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    count = 0
    F1count = 0
    IOUcount = 0
    total = test_loader.size
    print("total:",total)
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        # pred0, probx, u = model(image)
        # pred0, _, uncertainty, frequency = model(image)
        pred = model(image)
        res = pred[-1]
        # res = pred

        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        imageio.imwrite(save_path + name, (res * 255).astype(np.uint8))
        # freq = frequency.data.cpu().numpy().squeeze()
        # freq = (freq - freq.min()) / (freq.max() - freq.min() + 1e-8)
        # imageio.imwrite(save_path + '/frequency_heatmap/' + name, (freq * 255).astype(np.uint8))
        # plt.imshow(freq, cmap='jet')
        # plt.colorbar()
        # plt.axis('off')  # 不显示坐标轴
        # frequency_heatmap_save_path = save_path + '/frequency_heatmap/' + name.replace('.jpg', '.png')
        # plt.savefig(frequency_heatmap_save_path, bbox_inches='tight', pad_inches=0)
        # plt.close()

        # 处理uncertainty图谱，保存热力图
        # unc = uncertainty.data.cpu().numpy().squeeze()
        # plt.imshow(unc, cmap='jet')
        # plt.colorbar()
        # plt.axis('off')  # 不显示坐标轴
        # uncertainty_heatmap_save_path = save_path + '/uncertainty_heatmap/' + name.replace('.jpg', '.png')
        # plt.savefig(uncertainty_heatmap_save_path, bbox_inches='tight', pad_inches=0)
        # plt.close()

        # mae = eval_mae(numpy2tensor(res), numpy2tensor(gt))

        IOU, F1 = get_IOU(np.ravel(gt), np.ravel(res))
        F1count = F1count + F1
        IOUcount = IOUcount + IOU
        # count = mae + count

        print('[Eval-Test] name: {}, f1: {},iou: {}'.format(name, F1, IOU))

    F1count = F1count / total
    IOUcount = IOUcount / total
    count = count / 50
    print('[IOUcount]', IOUcount)
    print('[F1count]', F1count)
    print(opt.pth_path)

