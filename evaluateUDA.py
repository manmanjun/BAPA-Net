import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys
from collections import OrderedDict
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo

from model.deeplabv2 import Res_Deeplab
from data import get_data_path, get_loader
import torchvision.transforms as transform

from PIL import Image
import scipy.misc
from utils.loss import CrossEntropy2d

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)


MODEL = 'deeplabv2' # deeeplabv2, deeplabv3p

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="UDA evaluation script")
    parser.add_argument("-m","--model-path", type=str, default=None, required=True,
                        help="Model to evaluate")
    parser.add_argument("--class-num", type=int, default=16,
                        help="Number of classes.")
    parser.add_argument("--gpu", type=int, default=(0,),
                        help="choose gpu device.")
    parser.add_argument("--save-output-images", action="store_true",
                        help="save output images")
    return parser.parse_args()

class VOCColorize(object):
    def __init__(self, n=22):
        self.cmap = color_map(22)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        # handle void
        mask = (255 == gray_image)
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 255

        return color_image

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def get_label_vector(target, nclass):
    # target is a 3D Variable BxHxW, output is 2D BxnClass
    hist, _ = np.histogram(target, bins=nclass, range=(0, nclass-1))
    vect = hist>0
    vect_out = np.zeros((21,1))
    for i in range(len(vect)):
        if vect[i] == True:
            vect_out[i] = 1
        else:
            vect_out[i] = 0

    return vect_out

def get_iou(hist, class_num, dataset, save_path=None):
    if class_num == 19:
        mIoUs = per_class_iu(hist)

        classes = np.array(("road", "sidewalk",
            "building", "wall", "fence", "pole",
            "traffic_light", "traffic_sign", "vegetation",
            "terrain", "sky", "person", "rider",
            "car", "truck", "bus",
            "train", "motorcycle", "bicycle"))


        for i in range(class_num):
            print('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], 100*mIoUs[i]))

        print('meanIOU: ' + str(np.nanmean(mIoUs)*100) + '\n')
        if save_path:
            with open(save_path, 'w') as f:
                for i in range(class_num):
                    f.write('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], 100*mIoUs[i]) + '\n')
                f.write('meanIOU: ' + str(np.nanmean(mIoUs)*100) + '\n')
        return np.nanmean(mIoUs)*100
    elif class_num == 16:
        mIoUs = per_class_iu(hist)

        classes = np.array(("road", "sidewalk",
            "building", "wall", "fence", "pole",
            "traffic_light", "traffic_sign", "vegetation",
            "sky", "person", "rider",
            "car", "bus",
            "motorcycle", "bicycle"))


        for i in range(class_num):
            print('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], 100*mIoUs[i]))

        print('meanIOU: ' + str(np.nanmean(mIoUs)*100) + '\n')
        if save_path:
            with open(save_path, 'w') as f:
                for i in range(class_num):
                    f.write('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], 100*mIoUs[i]) + '\n')
                f.write('meanIOU: ' + str(np.nanmean(mIoUs)*100) + '\n')
        return np.nanmean(mIoUs)*100
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def evaluate(model, dataset, ignore_label=255, save_output_images=False, save_dir=None, input_size=(512,1024), ite = 0, num = 19):

    if dataset == 'cityscapes':
        num_classes = num
        data_loader = get_loader('cityscapes', num_classes)
        data_path = get_data_path('cityscapes')
        test_dataset = data_loader(data_path, img_size=input_size, img_mean = IMG_MEAN, is_transform=True, split='val')
        testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
        interp = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)


    print('Evaluating, found ' + str(len(testloader)) + ' images.')

    data_list = []
    colorize = VOCColorize()

    total_loss = []
    hist = np.zeros((num_classes, num_classes))
    for index, batch in enumerate(testloader):
        image, label, size, name, _ = batch
        size = size[0]
        #if index > 500:
        #    break
        with torch.no_grad():
            output, _= model(Variable(image).cuda())
            output = interp(output)
            output = output.cpu().data[0].numpy()

            if dataset == 'cityscapes':
                gt = np.asarray(label[0].numpy(), dtype=np.int)

            output = output.transpose(1,2,0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.int)

            hist += fast_hist(gt.flatten(), output.flatten(), num_classes)

        if (index+1) % 100 == 0:
            print('%d processed'%(index+1))

    if save_dir:
        filename = os.path.join(save_dir, f'{ite}_result.txt')
    else:
        filename = None
    mIoU = get_iou(hist, num_classes, dataset, filename)
    loss = 0
    return mIoU, loss

def main():
    """Create the model and start the evaluation process."""

    gpu0 = args.gpu

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = Res_Deeplab(num_classes=num_classes)

    checkpoint = torch.load(args.model_path)
    try:
        model.load_state_dict(checkpoint['model'])
    except:
        model = torch.nn.DataParallel(model, device_ids=args.gpu)
        model.load_state_dict(checkpoint['model'])

    model.cuda()
    model.eval()

    evaluate(model, dataset = 'cityscapes', ignore_label=ignore_label, save_output_images=args.save_output_images, save_dir=save_dir, input_size=input_size)


if __name__ == '__main__':
    args = get_arguments()

    config = torch.load(args.model_path)['config']

    num_classes = args.class_num
    input_size = (512,1024)

    ignore_label = config['ignore_label']
    save_dir = os.path.join(*args.model_path.split('/')[:-1])
    main()
