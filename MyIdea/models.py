from numpy.core.fromnumeric import size
import torch
from torch import nn
from torch.autograd import Variable


class ReptileModel(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

    def point_grad_to(self, target):
        '''
        Set .grad attribute of each parameter to be proportional
        to the difference between self and target
        '''
        for p, target_p in zip(self.parameters(), target.parameters()):
            if p.grad is None:
                if self.is_cuda():
                    p.grad = Variable(torch.zeros(p.size())).cuda()
                else:
                    p.grad = Variable(torch.zeros(p.size()))
            p.grad.data.zero_()  # not sure this is required
            p.grad.data.add_(p.data - target_p.data)  # add the difference

    def is_cuda(self):
        return next(self.parameters()).is_cuda


class VGG_MiniImagenet(ReptileModel):
    """
    A model for Omniglot classification.
    """
    def __init__(self, num_classes):
        ReptileModel.__init__(self)

        config = [
        ('conv2d', [32, 3, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 1, 0]),
        ('flatten', []),
        ('linear', [num_classes, 32 * 5 * 5])
    ]
    # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()
        for i, (name, param) in enumerate(config):
            if name is 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name is 'linear':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])


            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                            'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError
    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'

            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'


            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info
    def forward(self, x):
        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0

        for name, param in self.config:
            if name is 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name is 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2

            elif name is 'flatten':
                # print(x.shape)
                x = x.view(x.size(0), -1)
            elif name is 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name is 'relu':
                x = F.relu(x, inplace=param[0])
            elif name is 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name is 'tanh':
                x = F.tanh(x)
            elif name is 'sigmoid':
                x = torch.sigmoid(x)
            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])

            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)
        return x

    def predict(self, prob):
        __, argmax = prob.max(1)
        return argmax

    def clone(self):
        clone = VGG_MiniImagenet(self.num_classes)
        clone.load_state_dict(self.state_dict())
        if self.is_cuda():
            clone.cuda()
        return clone


class MiniImagenetModel(ReptileModel):
    """
    A model for Omniglot classification.
    """
    def __init__(self, num_classes):
        ReptileModel.__init__(self)

        self.num_classes = num_classes
        self.conv = nn.Sequential(
            # 28 x 28 - 1
            nn.Conv2d(3, 32, 3, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # nn.MaxPool2d(kernel_size=2,stride=2,padding=1),

            # 14 x 14 - 64
            nn.Conv2d(32, 32, 3, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # nn.MaxPool2d(kernel_size=2,stride=2,padding=1),

            # 7 x 7 - 64
            nn.Conv2d(32, 32, 3, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # nn.MaxPool2d(kernel_size=2,stride=2,padding=1),

            # # 4 x 4 - 64
            nn.Conv2d(32, 32, 3, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # nn.MaxPool2d(kernel_size=2,stride=1,padding=1),
            # 2 x 2 - 64


            nn.Conv2d(32, 32, 3, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # nn.MaxPool2d(kernel_size=2,stride=1,padding=1),
            # 2 x 2 - 64
        )

        self.classifier = nn.Sequential(
            # 2 x 2 x 64 = 256
            # nn.Linear(32, 32),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(32, num_classes),
            nn.LogSoftmax(1)
        )

        # print("Initial model is over!!!")

    def forward(self, x):
        
        # out = x.view(-1, 1, 32, 32)
        out = x
        # print(out.size())
        # out = self.layer1(x)
        # print(out.size())
        out = self.conv(out)
        # print(out.size())
        out = out.view(len(out), -1)
        # print(out.size())
        out = self.classifier(out)
        return out

    def predict(self, prob):
        __, argmax = prob.max(1)
        return argmax

    def clone(self):
        clone = MiniImagenetModel(self.num_classes)
        clone.load_state_dict(self.state_dict())
        if self.is_cuda():
            clone.cuda()
        return clone


if __name__ == '__main__':
    model = OmniglotModel(20)
    x = Variable(torch.zeros(5, 28*28))
    y = model(x)
    print('the size of x', x.size())
    print('the size of y', y.size())

