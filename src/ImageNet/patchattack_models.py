import os
import argparse
import math
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from model_clamp import mobilenet_v2


torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=60, help='number of epochs to train for')
parser.add_argument('--bs', type=int, default=16, help='number of batch size')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')

parser.add_argument('--target', type=int, default=859, help='The target class')
parser.add_argument('--p_size', type=float, default=0.05, help='patch size. E.g. 0.05 ~= 5% of image')
parser.add_argument('--step', type=float, default=2 / 255, help='patch update step')
parser.add_argument('--data', default='../ImageNet_val', help='folder of images to attack')
parser.add_argument('--save', default='patches_robust', help='folder to output images and model checkpoints')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--clp', type=float, default=1.0, help='clamp ratio')

parser.add_argument('--advp', type=int, default=0, help='if using the AdvPatch attack')
parser.add_argument('--den', type=int, default=1, help='density of clipping')
parser.add_argument('--model_type', type=int, default=0, help='model architecture, 0 is ResNet-50, 1 is Inception-V3, 2 is MobileNetV2')
args = parser.parse_args()
print(args)



class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.mu = torch.from_numpy(np.array([0.485, 0.456, 0.406])).float().cuda().unsqueeze(0).unsqueeze(2).unsqueeze(3)
        self.std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).float().cuda().unsqueeze(0).unsqueeze(2).unsqueeze(3)

        self.resnet = torchvision.models.resnet50()
        sd0 = torch.load('resnet50_clp{}.pth.tar'.format(args.clp))['state_dict']
        sd = {}
        for k, v in sd0.items():
            if k[0:len('module.resnet.')] == 'module.resnet.':
                sd[k[len('module.resnet.'):]] = v
        self.resnet.load_state_dict(sd, strict=True)


    def clamp(self, x, a=1.0):
        norm = torch.norm(x, dim=1, keepdim=True)
        thre = torch.mean(torch.mean(a * norm, dim=2, keepdim=True), dim=3, keepdim=True)
        x = x / torch.clamp_min(norm, min=1e-7)
        mask = (norm > thre).float()
        norm = norm * (1 - mask) + thre * mask
        x = x * norm
        return x

    def features(self, input):

        x = (input - self.mu) / self.std
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)



        k = 1
        for i in range(len(self.resnet.layer1)):
            b = self.resnet.layer1[i]
            x = b(x)
            if k % args.den == 0:
                x = self.clamp(x, args.clp)
            k += 1
        for i in range(len(self.resnet.layer2)):
            b = self.resnet.layer2[i]
            x = b(x)
            if k % args.den == 0:
                x = self.clamp(x, args.clp)
            k += 1
        for i in range(len(self.resnet.layer3)):
            b = self.resnet.layer3[i]
            x = b(x)
            if k % args.den == 0:
                x = self.clamp(x, args.clp)
            k += 1
        for i in range(len(self.resnet.layer4)):
            b = self.resnet.layer4[i]
            x = b(x)
            if k % args.den == 0:
                x = self.clamp(x, args.clp)
            k += 1

        return x

    def logits(self, features):
        x = self.resnet.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.resnet.fc(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class Clamp(nn.Module):
    def __init__(self, thre=1.0):
        super().__init__()
        self.thre = thre

    def forward(self, x, w=None, meth=1):
        if meth > 0:
            cam = torch.sum(x * w, dim=1, keepdim=True)
            cam = cam - torch.clamp_max(torch.min(torch.min(cam, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0], max=0)
            thre = torch.mean(torch.mean(self.thre * cam, dim=2, keepdim=True), dim=3, keepdim=True)
            x = x / torch.clamp_min(cam, min=1e-7)
            if meth == 2:
                x = x * thre
            else:
                mask = (cam > thre).float()
                norm = cam * (1 - mask) + thre * mask
                x = x * norm
        return x


class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        self.mu = torch.from_numpy(np.array([0.485, 0.456, 0.406])).float().cuda().unsqueeze(0).unsqueeze(2).unsqueeze(
            3)
        self.std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).float().cuda().unsqueeze(0).unsqueeze(2).unsqueeze(
            3)

        self.incep = torchvision.models.inception_v3(pretrained=True, aux_logits=False)
        sd0 = torch.load('incep_clp{}.pth.tar'.format(args.clp))['state_dict']
        sd = {}
        for k, v in sd0.items():
            if k[0:len('module.incep.')] == 'module.incep.':
                sd[k[len('module.incep.'):]] = v
        self.incep.load_state_dict(sd, strict=True)

    def clamp(self, x, a=1.0):
        norm = torch.norm(x, dim=1, keepdim=True)
        thre = torch.mean(torch.mean(a * norm, dim=2, keepdim=True), dim=3, keepdim=True)
        x = x / torch.clamp_min(norm, min=1e-7)
        mask = (norm > thre).float()
        norm = norm * (1 - mask) + thre * mask
        x = x * norm
        return x

    def features(self, input):
        x = (input - self.mu) / self.std
        x = self.incep._transform_input(x)
        # N x 3 x 299 x 299
        x = self.incep.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.incep.Conv2d_2a_3x3(x)
        x = self.clamp(x, args.clp)
        # N x 32 x 147 x 147
        x = self.incep.Conv2d_2b_3x3(x)
        x = self.clamp(x, args.clp)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.incep.Conv2d_3b_1x1(x)
        x = self.clamp(x, args.clp)
        # N x 80 x 73 x 73
        x = self.incep.Conv2d_4a_3x3(x)
        x = self.clamp(x, args.clp)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.incep.Mixed_5b(x)
        x = self.clamp(x, args.clp)
        # N x 256 x 35 x 35
        x = self.incep.Mixed_5c(x)
        x = self.clamp(x, args.clp)
        # N x 288 x 35 x 35
        x = self.incep.Mixed_5d(x)
        x = self.clamp(x, args.clp)
        # N x 288 x 35 x 35
        x = self.incep.Mixed_6a(x)
        x = self.clamp(x, args.clp)
        # N x 768 x 17 x 17
        x = self.incep.Mixed_6b(x)
        x = self.clamp(x, args.clp)
        # N x 768 x 17 x 17
        x = self.incep.Mixed_6c(x)
        x = self.clamp(x, args.clp)
        # N x 768 x 17 x 17
        x = self.incep.Mixed_6d(x)
        x = self.clamp(x, args.clp)
        # N x 768 x 17 x 17
        x = self.incep.Mixed_6e(x)
        x = self.clamp(x, args.clp)
        # N x 768 x 17 x 17
        aux_defined = self.incep.training and self.incep.aux_logits
        if aux_defined:
            aux = self.incep.AuxLogits(x)
        else:
            aux = None
        # N x 768 x 17 x 17
        x = self.incep.Mixed_7a(x)
        x = self.clamp(x, args.clp)
        # N x 1280 x 8 x 8
        x = self.incep.Mixed_7b(x)
        x = self.clamp(x, args.clp)
        # N x 2048 x 8 x 8
        x = self.incep.Mixed_7c(x)
        x = self.clamp(x, args.clp)
        # N x 2048 x 8 x 8
        return x

    def logits(self, features):
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(features, (1, 1))
        # N x 2048 x 1 x 1
        x = F.dropout(x, training=self.training)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.incep.fc(x)
        return x

    def forward(self, x):
        x= self.features(x)
        x = self.logits(x)
        return x


class Mobile(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu = torch.from_numpy(np.array([0.485, 0.456, 0.406])).float().cuda().unsqueeze(0).unsqueeze(2).unsqueeze(
            3)
        self.std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).float().cuda().unsqueeze(0).unsqueeze(2).unsqueeze(
            3)


        self.mobile = mobilenet_v2(clamp=args.clp)
        sd0 = torch.load('mobile_clp{}.pth.tar'.format(args.clp))['state_dict']
        sd = {}
        for k, v in sd0.items():
            if k[0:len('module.mobile.')] == 'module.mobile.':
                sd[k[len('module.mobile.'):]] = v
        self.mobile.load_state_dict(sd, strict=True)



    def features(self, input):
        x = (input - self.mu) / self.std
        x = self.mobile.features(x)
        x = self.mobile.clamp(x, args.clp)
        return x

    def logits(self, features):
        x = nn.functional.adaptive_avg_pool2d(features, 1).reshape(features.shape[0], -1)
        out = self.mobile.classifier(x)
        return out

    def forward(self, x):
        features = self.features(x)
        out = self.logits(features)
        return out



if not os.path.exists(args.save + '/' + str(args.target)):
    os.makedirs(args.save + '/' + str(args.target))

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

if args.model_type == 0:
    net = ResNet50()
elif args.model_type == 1:
    net = Inception()
else:
    net = Mobile()



if args.cuda:
    net.cuda()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

idx = np.arange(50)
training_idx = np.array([])
for i in range(1):
    training_idx = np.append(training_idx, [idx[i * 50:i * 50 + 10]])
training_idx = training_idx.astype(np.int32)

im_size = 224
if args.model_type == 1:
    im_size = 299

train_loader = torch.utils.data.DataLoader(
    dset.ImageFolder(args.data, transforms.Compose([
        transforms.Resize(round(im_size * 1.050)),
        transforms.CenterCrop(im_size),
        transforms.ToTensor(),
    ])),
    batch_size=args.bs, shuffle=False, sampler=None,
    num_workers=2, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    dset.ImageFolder(args.data, transforms.Compose([
        transforms.Resize(round(im_size * 1.000)),
        transforms.CenterCrop(im_size),
        transforms.ToTensor(),
    ])),
    batch_size=args.bs, shuffle=False, sampler=None,
    num_workers=2, pin_memory=True)



def InitPatchB(im_size, p_size, rxy=None):
    b, c, h, w = im_size
    side = math.sqrt(h * w * p_size)
    sidex = int(side)
    sidey = int(side)
    if rxy is None:
        rx = np.random.randint(0, w - sidex, size=(b, ))
        ry = np.random.randint(0, h - sidey, size=(b, ))
    else:
        rx = rxy[0] * np.ones((b,))
        ry = rxy[1] * np.ones((b,))
    patch = np.zeros([1,c,sidey,sidex])
    rx = torch.from_numpy(rx).type(torch.long)
    ry = torch.from_numpy(ry).type(torch.long)
    patch = torch.from_numpy(patch).type(torch.float32)
    if args.cuda:
        rx, ry, patch = rx.cuda(), ry.cuda(), patch.cuda()
    return patch, rx, ry



def InheritB(im_size, p_size, rxy=None):
    b, c, h, w = im_size
    side = math.sqrt(h * w * p_size)
    sidex = int(side)
    sidey = int(side)
    if rxy is None:
        rx = np.random.randint(0, w - sidex, size=(b,))
        ry = np.random.randint(0, h - sidey, size=(b,))
    else:
        rx = rxy[0] * np.ones((b,))
        ry = rxy[1] * np.ones((b,))
    rx = torch.from_numpy(rx).type(torch.long)
    ry = torch.from_numpy(ry).type(torch.long)
    if args.cuda:
        rx, ry = rx.cuda(), ry.cuda()
    return rx, ry


def get_noise(patch, im_size, rx, ry):
    b, c, h, w = im_size
    _, c, h_, w_ = patch.shape
    shift_x = rx.view(-1,1,1,1)
    shift_y = ry.view(-1,1,1,1)
    shift = torch.cat([shift_x, shift_y], dim=-1).repeat(1,h,w,1)
    idx_x = torch.arange(0, w).type(torch.long).view(1,-1).repeat(h,1)
    idx_y = torch.arange(0, h).type(torch.long).view(-1, 1).repeat(1, w)
    idx = torch.stack([idx_x, idx_y], dim=-1).view(1, h, w, 2).repeat(b,1,1,1)
    idx = idx.type(torch.float32)
    shift = shift.type(torch.float32)
    if args.cuda:
        idx = idx.cuda()
    idx = idx - shift #(N,h,w,2)
    mask_x = (idx[:, :, :, 0] >= 0).type(torch.float32) * (idx[:, :, :, 0] < w_).type(torch.float32)
    mask_y = (idx[:, :, :, 1] >= 0).type(torch.float32) * (idx[:, :, :, 1] < h_).type(torch.float32)
    mask = mask_x * mask_y #(N,h,w)
    mask = mask.view(b, 1, h, w).repeat(1, c, 1, 1)
    mask = mask.type(torch.float32)
    mask = 1.0 - mask
    idx[:, :, :, 0] = torch.clamp_max(torch.clamp_min(idx[:, :, :, 0], min=0), max=w_-1)
    idx[:, :, :, 0] = 2.0 * (idx[:, :, :, 0] / (w_ - 1.0)) - 1.0
    idx[:, :, :, 1] = torch.clamp_max(torch.clamp_min(idx[:, :, :, 1], min=0), max=h_-1)
    idx[:, :, :, 1] = 2.0 * (idx[:, :, :, 1] / (h_ - 1.0)) - 1.0
    #print(idx[0, ry[0]:ry[0]+h_, rx[0]:rx[0]+w_, :])
    patch = patch.repeat(b, 1, 1, 1)
    noise = F.grid_sample(patch, idx)
    return noise, mask



def attack(epoch, patch, rxy=None):
    net.eval()
    total = 0
    accu = 0.
    succ = 0.
    for batch_idx, (data, labels) in enumerate(train_loader):
        if args.cuda:
            data, labels = data.cuda(), labels.cuda()

        if epoch == 0 and batch_idx == 0:
            patch, rx, ry = InitPatchB(data.shape, args.p_size, rxy=rxy)
        else:
            rx, ry = InheritB(data.shape, args.p_size, rxy=rxy)

        total += 1
        pre = labels
        for count in range(1):
            patch.requires_grad = True
            noise, mask = get_noise(patch, data.shape, rx, ry)
            adv = torch.mul(data, mask) + torch.mul(noise, 1 - mask)



            adv_out = net(adv)

            if args.advp == 1:
                ys = torch.zeros_like(labels).cuda() + args.target
                losst = F.cross_entropy(adv_out, ys)
            else:
                if args.target < 1000:
                    target = adv_out[:, args.target]
                    gndtru = torch.gather(adv_out, dim=1, index=pre.unsqueeze(1)).squeeze(1)
                    losst = torch.mean(gndtru - target)
                else:
                    losst = F.cross_entropy(adv_out, labels)

            loss = losst

            loss.backward()
            grad = patch.grad.clone()
            patch.grad.data.zero_()
            patch.requires_grad = False
            patch = torch.clamp(patch - lr * args.step * torch.sign(grad), 0, 1)
        pre = torch.argmax(adv_out, dim=1)

        succ += (pre == args.target).type(torch.float32).sum()
        accu += (pre == labels).type(torch.float32).sum()


    vutils.save_image(noise,
                        args.save + '/' + str(args.target) + '/' + str(epoch) + '_noise_margin.png',
                        normalize=False)
    vutils.save_image(patch,
                        args.save + '/' + str(args.target) + '/' + str(epoch) + '_patch_margin.png',
                        normalize=False)
    vutils.save_image(mask,
                        args.save + '/' + str(args.target) + '/' + str(epoch) + '_mask_margin.png',
                        normalize=False)

    evaluate(epoch, patch)

    return patch

def evaluate(epoch, patch, rxy=None):
    net.eval()
    total = 0
    accu = 0.
    succ = 0.
    for batch_idx, (data, labels) in enumerate(train_loader):
        if args.cuda:
            data, labels = data.cuda(), labels.cuda()

        rx, ry = InheritB(data.shape, args.p_size, rxy=rxy)
        total += data.shape[0]

        for count in range(1):
            patch.requires_grad = False
            noise, mask = get_noise(patch, data.shape, rx, ry)
            adv = torch.mul(data, mask) + torch.mul(noise, 1 - mask)
            adv_out = F.softmax(net(adv), dim=1)
        pre = torch.argmax(adv_out, dim=1)

        succ += (pre == args.target).type(torch.float32).sum()
        accu += (pre == labels).type(torch.float32).sum()
    print('Epoch:{:3d}, Classify Accuracy:{:.2f}, Target Success:{:.2f}'.format(epoch, accu / total * 100, succ / total * 100))

def evaluate_clean():
    net.eval()
    total = 0
    accu = 0.
    succ = 0.
    for batch_idx, (data, labels) in enumerate(test_loader):
        if args.cuda:
            data, labels = data.cuda(), labels.cuda()

        total += 1
        for count in range(1):

            adv_out = F.softmax(net(data), dim=1)
        pre = torch.argmax(adv_out, dim=1).detach().cpu().numpy()
        accu += np.sum(pre == labels.detach().cpu().numpy())
        succ += np.sum(pre == args.target)

    print('Clean Examples: Classify Accuracy:{:.2f}, Target Success:{:.2f}'.format(accu / 100, succ / 100))


if __name__ == '__main__':
    lr = 1.0
    evaluate_clean()
    patch, _, _ = InitPatchB([1, 3, im_size, im_size], args.p_size)

    for i in range(args.epoch):
        if i == 30:
            lr = lr / 2.
        patch = attack(i, patch)
