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

def LearnFromData(train_data,net,optimizer,Trian_or_val):
    temp_acc = []
    temp_loss = []
    for i ,or_data in enumerate(train_data):
        # Sample minibatch
        data, labels = or_data
        data, labels  = data.to(Mydevice),labels.to(Mydevice).view(-1)
        # Forward pass
        prediction = net(data)
        # Get loss
        loss = get_loss(prediction, labels)
        # acc
        argmax = net.predict(prediction)
        accuracy = (argmax == labels).float().mean()
        temp_acc.append(accuracy.data.item())
        temp_loss.append(loss.data.item())
        # Backward pass - Update fast net
        if Trian_or_val =="train":
            optimizer.zero_grad() #将模型参数的梯度初始化为0
            loss.backward() #反向传播计算梯度
            optimizer.step()  # 更新参数
    return np.mean(temp_acc),np.mean(temp_loss)
def do_Gradient(net,optimizer, current_task,args):
    ### 加载数据集
    train_data,test_data = current_task
    train_data = DataLoader(train_data, args.batch, shuffle=True)
    test_data = DataLoader(test_data, args.batch, shuffle=True)
    # 训练
    net.train()
    acc_iterations = 0
    
    if args.Gradient_descent_iteration ==-1  or args.Gradient_descent_iteration ==0 :
        loss = np.inf
        while loss > 0.0001:
            acc,loss = LearnFromData(train_data,net,optimizer,"train")    
    else:
        for outer_loop in range(args.Gradient_descent_iteration):  # 循环
            acc,loss = LearnFromData(train_data,net,optimizer,"train")
            if np.mean(acc) >0.95 and acc_iterations==0:  #记录训练到0.9准确率的周期
            # if loss < 0.0001 and acc_iterations==0:  #记录训练到0.9准确率的周期
                print("acc_iterations",outer_loop+1)
                acc_iterations = outer_loop+1
    if int(acc_iterations) == 0: # 说明人为设定的微调次数太少
        acc_iterations == -1
    ###
    #
    #查看tain的结果
    ####33
    net.eval()
    acc,loss = LearnFromData(train_data,net,optimizer,"")

    # print("Train res:",loss,acc)
    # 验证
    acc,loss = LearnFromData(test_data,net,optimizer,"")
    return loss,acc,acc_iterations
def get_optimizer(net, state=None):
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0, 0.999))
    if state is not None:
        optimizer.load_state_dict(state)
    return optimizer

def Meta_Update(net,Tasks,t,args):
    """输入是神经网络参数和任务集合
    作用 ：meta learning
    """
    N_meta = args.N_meta
    meta_lr = args.meta_lr #* (1. - meta_iteration/float(args.meta_iterations))
    
    set_learning_rate(meta_optimizer, meta_lr)  #设置学习率
    for n_m in range(N_meta):
        # sample a task
        Select_Task_index = np.random.choice(t+1, 1, replace=False)[0] #随机选取一个任务
        meta_Task,_ = Tasks[Select_Task_index]
        Train_data = DataLoader(meta_Task, args.batch, shuffle=True)
        # Clone mode
        net = meta_net.clone()# 复制上一个任务的模型参数
        optimizer = get_optimizer(net)  #创建当前网络的优化器
        # Update fast net
        loss = do_learning(net, optimizer, Train_data, args.meta_iterations,Mydevice)
        state = optimizer.state_dict()  # save optimizer state

        # Update slow net
        meta_net.point_grad_to(net) # net是目标梯度
        meta_optimizer.step()
        torch.cuda.empty_cache()
    return meta_net

def Update_Procedure(net,current_task,args):
    """对D这个任务（数据集）进行训练"""
    optimizer = get_optimizer(net)   # 设置优化器
    loss,acc,end_iteraion = do_Gradient(net, optimizer, current_task,args)
    return loss,acc,end_iteraion
# Parsing
parser = argparse.ArgumentParser('Train reptile on MiniImagenet')

# Mode
parser.add_argument('--logdir', default="./Method2_My",help='Folder to store everything/load')

# - Training params
parser.add_argument('--classes', default=10, type=int, help='classes in base-task (N-way)')
parser.add_argument('--C_datapoints', default=100, type=float, help='the numbers of datapoints')
parser.add_argument('--N_meta', default=20, type=int, help='number of iterations')
parser.add_argument('--meta-iterations', default=10, type=int, help='number of meta iterations')
parser.add_argument('--Gradient_descent_iteration', default=200, type=int, help='the iterations of gradient tune')
parser.add_argument('--batch', default=120, type=int, help='minibatch size in base task')
parser.add_argument('--meta-lr', default=1., type=float, help='meta learning rate')
parser.add_argument('--lr', default=1e-3, type=float, help='base learning rate')

# - General params
parser.add_argument('--Val_precent', default=0.2, type=float, help='the numbers of datapoints')
parser.add_argument('--yy', default=0.2, type=float, help='the numbers of datapoints')
parser.add_argument('--input', default='/home/huang/Desktop/Hw/ConvexOptimization/Data/mini-imagenet', help='Path to omniglot dataset')
parser.add_argument('--cuda', default=1, type=int, help='Use cuda')

args = parser.parse_args()
Mydevice = torch.device("cuda:"+str(0) if torch.cuda.is_available else "cpu")
# Create tensorboard logger
logger = SummaryWriter(args.logdir)

# Load data
transform_image = transforms.Compose([transforms.Resize((84,84)),
        transforms.ToTensor(),
])
# Resize is done by the MetaDataset because the result can be easily cached
print("Generate Task Set!")
MyTaskset = TasksSet(args,transform_image)


from models import MiniImagenetModel
torch.cuda.empty_cache()
meta_net = MiniImagenetModel(args.classes)  #构建模型
meta_net.to(Mydevice)
meta_optimizer = torch.optim.SGD(meta_net.parameters(), lr=args.meta_lr)
info = {}
info["loss"] = []
info["acc"] = []
info["end_iteraion"] = []
for t in tqdm.trange(len(MyTaskset)):
    meta_net = Meta_Update(meta_net,MyTaskset,t,args)  #元学习
    loss,acc,end_iteraion = Update_Procedure(meta_net.clone(),MyTaskset[t],args)
    print("accuracy:%f,loss:%f,end_iteraion:%f"%(acc,loss,end_iteraion))
    info["loss"].append(loss)
    info["acc"].append(acc)
    info["end_iteraion"].append(end_iteraion)
import pandas as pd
import os
info=pd.DataFrame([info])
info.to_csv(os.path.join(args.logdir,"Mini.csv"))
