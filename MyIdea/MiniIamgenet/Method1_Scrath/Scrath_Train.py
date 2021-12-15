import os
import argparse
from numpy.core.fromnumeric import ptp
import tqdm
import json
import torch

from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tensorboardX import SummaryWriter
import os
import sys
sys.path.append("./")
from MiniImagenet import TasksSet
from utils import set_learning_rate,get_loss,do_learning
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def LearnFromData(train_data,net,optimizer,Trian_or_val,scheduler):
    temp_acc = []
    temp_loss = []
    for i ,or_data in enumerate(train_data):
        # Sample minibatch
        data, labels = or_data
        data, labels  = data.to(Mydevice),labels.to(Mydevice).view(-1)
        # Forward pass
        if Trian_or_val =="train":
            prediction = net(data)
        else:
            with torch.no_grad():
                prediction = net(data)
        # Get loss
        loss = get_loss(prediction, labels)
        # acc
        argmax = net.predict(prediction)
        accuracy = (argmax == labels).float().mean()
        temp_acc.append(accuracy.data.item())
        temp_loss.append(loss.data.item())
        if Trian_or_val =="train":
            optimizer.zero_grad() #将模型参数的梯度初始化为0
            loss.backward() #反向传播计算梯度
            optimizer.step()  # 更新参数
    if Trian_or_val =="train":
        # scheduler.step(np.mean(temp_loss))
        pass
    return np.mean(temp_acc),np.mean(temp_loss),net
def do_Gradient(net,optimizer,scheduler, current_task,args):
    ### 加载数据集
    train_data,test_data = current_task
    train_data = DataLoader(train_data, args.batch, shuffle=True)
    test_data = DataLoader(test_data, args.batch, shuffle=True)
    # 训练
    net.train()
    acc_iterations = 0
    Best_loss = np.inf
    Best_net = net
    if args.Gradient_descent_iteration ==-1  or args.Gradient_descent_iteration ==0 :
        loss = np.inf
        while loss > 0.0001:
            acc,loss = LearnFromData(train_data,net,optimizer,"train",scheduler)    
    else:
        for outer_loop in range(args.Gradient_descent_iteration):  # 循环
            
            acc,loss,net = LearnFromData(train_data,net,optimizer,"train",scheduler)
            if Best_loss > loss: # save the best model
                Best_loss = loss
                Best_net = net 
                print("Best_model",loss,acc)
            # print(acc,loss)
            if acc >0.90 and acc_iterations==0:  #记录训练到0.9准确率的周期
            # if loss < 0.0001 and acc_iterations==0:  #记录训练到0.9准确率的周期
                print("acc_iterations",outer_loop+1)
                acc_iterations = outer_loop+1
    if int(acc_iterations) == 0: # 说明人为设定的微调次数太少
        acc_iterations == -1

    ### 验证
    net = Best_net
    #
    #查看tain的结果
    ####33
    net.eval()
    acc,loss,_ = LearnFromData(train_data,net,optimizer,"",scheduler)

    # print("Train res:",loss,acc)
    # 验证
    acc,loss,_ = LearnFromData(test_data,net,optimizer,"",scheduler)
    return loss,acc,acc_iterations
def get_optimizer(net, state=None):
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr,weight_decay=0.01, betas=(0, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    if state is not None:
        optimizer.load_state_dict(state)
    return optimizer,scheduler
def Update_Procedure(net,current_task,args):
    """对D这个任务（数据集）进行训练"""
    optimizer,scheduler = get_optimizer(net)   # 设置优化器
    loss,acc,end_iteraion = do_Gradient(net, optimizer,scheduler, current_task,args)
    return loss,acc,end_iteraion
# Parsing
parser = argparse.ArgumentParser('Train reptile on MiniImagenet')

# Mode
parser.add_argument('--logdir', default="./Method1_Scrath",help='Folder to store everything/load')

# - Training params
parser.add_argument('--classes', default=10, type=int, help='classes in base-task (N-way)')
parser.add_argument('--C_datapoints', default=100, type=float, help='the numbers of datapoints')
parser.add_argument('--N_meta', default=10, type=int, help='number of iterations')
parser.add_argument('--meta-iterations', default=3, type=int, help='number of meta iterations')
parser.add_argument('--Gradient_descent_iteration', default=200, type=int, help='the iterations of gradient tune')
parser.add_argument('--batch', default=50, type=int, help='minibatch size in base task')
parser.add_argument('--meta-lr', default=1., type=float, help='meta learning rate')
parser.add_argument('--lr', default=1e-3, type=float, help='base learning rate')

# - General params
parser.add_argument('--Val_precent', default=0.1, type=float, help='the numbers of datapoints')
parser.add_argument('--yy', default=0.2, type=float, help='the numbers of datapoints')
parser.add_argument('--input', default='/home/huang/Desktop/Hw/ConvexOptimization/Data/mini-imagenet', help='Path to omniglot dataset')
parser.add_argument('--cuda', default=1, type=int, help='Use cuda')

args = parser.parse_args()
Mydevice = torch.device("cuda:"+str(0) if torch.cuda.is_available else "cpu")
# Create tensorboard logger
logger = SummaryWriter(args.logdir)

# Load data
transform_image = transforms.Compose([transforms.Resize((84,84)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(45),
        # transforms.ColorJitter(contrast=1),
        transforms.ToTensor(),
])
# Resize is done by the MetaDataset because the result can be easily cached
print("Generate Task Set!")
MyTaskset = TasksSet(args,transform_image)


from models import MiniImagenetModel,Reptile_MiniImagenet

info = {}
info["loss"] = []
info["acc"] = []
info["end_iteraion"] = []
for t in tqdm.trange(len(MyTaskset)):
    # initial model
    Model_net = Reptile_MiniImagenet(args.classes)
    Model_net.to(Mydevice)
    loss,acc,end_iteraion = Update_Procedure(Model_net,MyTaskset[t],args)
    print("accuracy:%f,loss:%f,end_iteraion:%f"%(acc,loss,end_iteraion))
    info["loss"].append(loss)
    info["acc"].append(acc)
    info["end_iteraion"].append(end_iteraion)
import pandas as pd
import os
info=pd.DataFrame([info])
info.to_csv(os.path.join(args.logdir,"Res.csv"))
