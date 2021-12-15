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


class Reptile_MiniImagenet(ReptileModel):
    """
    A model for Omniglot classification.
    """
    def __init__(self, num_classes):
        ReptileModel.__init__(self)
        self.num_classes =num_classes
        self.conv = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3),
                                nn.AvgPool2d(kernel_size=2),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),

                                nn.Conv2d(64, 64, kernel_size=3),
                                nn.AvgPool2d(kernel_size=2),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),

                                nn.Conv2d(64, 64, kernel_size=3),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),

                                nn.Conv2d(64, 64, kernel_size=3),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),

                                nn.MaxPool2d(3,2)

                                )
        self.fc = nn.Sequential(nn.Linear(3136, 64),
		                        nn.ReLU(inplace=True),
		                        nn.Linear(64, self.num_classes),
                                nn.LogSoftmax(1))
        
    def forward(self, x):
        out = self.conv(x)
        out = out.view(len(out), -1)
        # batch_size,nuerons = out.size()
        # print(out.size())
        out = self.fc(out)
        return out

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
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # nn.MaxPool2d(kernel_size=2,stride=2,padding=1),

            # 14 x 14 - 64
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # nn.MaxPool2d(kernel_size=2,stride=2,padding=1),

            # 7 x 7 - 64
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=1),

            # # 4 x 4 - 64
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # nn.MaxPool2d(kernel_size=2,stride=1,padding=1),
            # 2 x 2 - 64


            nn.Conv2d(32, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # nn.MaxPool2d(kernel_size=2,stride=1,padding=1),
            # 2 x 2 - 64
        )

        self.classifier = nn.Sequential(
            # 2 x 2 x 64 = 256
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
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

