import random
import time
import datetime
import sys
import yaml
from torch.autograd import Variable
import torch
# from visdom import Visdom
import torch.nn.functional as F
import numpy as np
from torch import distributed as dist


class DecoupleAdaptive:
    def __init__(self, ada_aug_target, ada_aug_len, update_every):
        self.ada_aug_target = ada_aug_target
        self.ada_aug_len = ada_aug_len
        self.update_every = update_every

        self.ada_update = 0
        self.ada_aug_buf = torch.tensor(0.0).cuda()
        self.r_t_stat = 0
        self.ada_aug_p = 0
        self.loss_record = []
        self.loss_mean = 0.0

    @torch.no_grad()
    def tune(self, real_pred):
        # print(real_pred)
        # print(real_pred.shape[0])
        # print(torch.sign(real_pred).sum().item())
        self.ada_aug_buf += real_pred

        self.loss_record.append(real_pred.item())

        self.ada_update += 1

        if self.ada_update % self.update_every == 0:
            self.ada_aug_buf = self.ada_aug_buf.item() / self.update_every
            # # print('\n', self.ada_aug_buf)
            # pred_signs, n_pred = self.ada_aug_buf.tolist()

            # print(pred_signs, n_pred)
            if self.ada_update <= (self.update_every * 1000):
                self.loss_mean = self.ada_aug_buf

            if 0.7 * self.loss_mean <= self.ada_aug_buf <= 1.3 * self.loss_mean:

                sign = 1

            else:
                sign = -1

            self.ada_aug_p += sign / self.ada_aug_len
            # print(self.ada_aug_p)
            self.ada_aug_p = min(self.ada_aug_target, max(0, self.ada_aug_p))
            # print(self.ada_aug_p)
            self.ada_aug_buf = torch.tensor(0.0).cuda()
            # self.ada_update = 0
        if self.ada_update % (self.update_every * 100) == 0:
            self.loss_mean = sum(self.loss_record) / len(self.loss_record)

        return self.ada_aug_p, self.loss_mean


def reduce_sum(tensor):
    if not dist.is_available():
        return tensor

    if not dist.is_initialized():
        return tensor

    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    return tensor


class Resize():
    def __init__(self, size_tuple, use_cv=True):
        self.size_tuple = size_tuple
        self.use_cv = use_cv

    def __call__(self, tensor):
        """
            Resized the tensor to the specific size

            Arg:    tensor  - The torch.Tensor obj whose rank is 4
            Ret:    Resized tensor
        """
        tensor = tensor.unsqueeze(0)

        tensor = F.interpolate(tensor, size=[self.size_tuple[0], self.size_tuple[1]])

        tensor = tensor.squeeze(0)

        return tensor  # 1, 64, 128, 128


class ToTensor():
    def __call__(self, tensor):
        tensor = np.expand_dims(tensor, 0)
        return torch.from_numpy(tensor)


def tensor2image(tensor):
    image = (127.5 * (tensor.cpu().float().numpy())) + 127.5
    image1 = image[0]
    for i in range(1, tensor.shape[0]):
        image1 = np.hstack((image1, image[i]))

    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    # print ('image1.shape:',image1.shape)
    return image1.astype(np.uint8)


class Logger():
    def __init__(self, n_epochs, batches_epoch, decouple, regist, model_name):
        # self.viz = Visdom(port= ports,env = env_name)
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}
        self.log_count = 0
        self.decouple = decouple
        self.regist = regist
        self.model_name = model_name

    def log(self, losses=None, lossdir=None):
        self.log_count += 1
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write(
            '\r%s: Epoch %03d/%03d [%04d/%04d] -- ' %
            ('', self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        with open('%s/log.txt' % lossdir, 'a') as f:
            for i, loss_name in enumerate(losses.keys()):
                if loss_name not in self.losses:
                    self.losses[loss_name] = losses[loss_name].item()
                else:
                    self.losses[loss_name] += losses[loss_name].item()

                if (i + 1) == len(losses.keys()):
                    sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name] / self.batch))
                    f.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name] / self.batch))
                else:
                    sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name] / self.batch))
                    f.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name] / self.batch))
            f.write('\n')
        # if self.decouple:
        #     sys.stdout.write('G_loss_sum: %.4f | ' % D_loss_sum.item())
        #     sys.stdout.write('loss_mean: %.4f | ' % loss_mean)
        #     sys.stdout.write('G_update_rate: %.5f | ' % D_update_rate)
        #     sys.stdout.write('update_count: %d | ' % update_count)
        # if self.regist:
        #     sys.stdout.write('r_t_stat: %.5f | ' % r_t_stat)

        batches_done = self.batches_epoch * (self.epoch - 1) + self.batch
        batches_left = self.batches_epoch * (self.n_epochs - self.epoch) + self.batches_epoch - self.batch
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left * self.mean_period / batches_done)))

        # Draw images
        # for image_name, tensor in images.items():
        #     if image_name not in self.image_windows:
        #         self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title': image_name})
        #     else:
        #         self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name],
        #                        opts={'title': image_name})

        # if self.decouple:
        #     self.viz.line(X=np.array([self.log_count]), Y=np.array([D_update_rate]), win='D_update_rate', update='append',
        #               opts={'xlabel': 'iterations', 'ylabel': 'D_update_rate',
        #                     'title': 'D_update_rate'}
        #               )

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # # Plot losses
            # for loss_name, loss in self.losses.items():
            #     if loss_name not in self.loss_windows:
            #         self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]),
            #                                                      Y=np.array([loss / self.batch]),
            #                                                      opts={'xlabel': 'epochs', 'ylabel': loss_name,
            #                                                            'title': loss_name})
            #     else:
            #         self.viz.line(X=np.array([self.epoch]), Y=np.array([loss / self.batch]),
            #                       win=self.loss_windows[loss_name], update='append')
            #     # Reset losses for next epoch
            #     self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):
    # print ('m:',m)
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


def smooothing_loss(y_pred):
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

    dx = dx * dx
    dy = dy * dy
    d = torch.mean(dx) + torch.mean(dy)
    grad = d
    return d


def write_loss_log(losses, lossdir):
    with open('%s/log.txt' % lossdir, 'a') as f:
        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in losses:
                self.losses[loss_name] = losses[loss_name].item()
            else:
                self.losses[loss_name] += losses[loss_name].item()

            if (i + 1) == len(losses.keys()):
                # sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name] / self.batch))
                f.write('%s: %.4f -- ' % (loss_name, losses[loss_name] / self.batch) + '\n')
            else:
                f.write('%s: %.4f | ' % (loss_name, losses[loss_name] / self.batch) + '\n')

