import re
import torch
from torch.utils import data
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import pandas as pd
from utils import list_files, list_dir

# Might need to manually download, extract, and merge
# https://github.com/brendenlake/omniglot/blob/master/python/images_background.zip
# https://github.com/brendenlake/omniglot/blob/master/python/images_evaluation.zip

class FewShot(data.Dataset):### 直接把图片扔进来就行
    '''
    Dataset for K-shot N-way classification
    '''
    def __init__(self, paths,transforms_image,select_list):
        self.paths = paths
        self.transforms_image = transforms_image
        self.select_list = select_list#.tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]['path']
        image =Image.open(path, mode='r')#.convert('L')
        image = self.transforms_image(image)
        # print(image.shape)
        # image = self.parent.cache.read_image(path, self.parent.size)
        # if self.parent.transform_image is not None:
        #     image = self.parent.transform_image(image)
        label = self.select_list.index(self.paths[idx]['label'])
        label = torch.tensor(label)
        # if self.parent.transform_label is not None:
        #     label = self.parent.transform_label(label)
        return image, label

class TasksSet:
    """任务集合"""
    def __init__(self,Parameters,transforms_image):
        '''
        :param root: folder containing alphabets for background and evaluation set
        '''
        self.root = Parameters.input  #file path 
        self.arg = Parameters
        self.datapoint = self.arg.C_datapoints
        self.transforms_image = transforms_image
        self.Path_dict = {}  #以index作为keyname
        self.Name_dict = {}
        self.Num_dict = {}    # 记录每个task的datapoint
        self.Index_dict = {}    # 记录每个task的datapoint
        self.Index_Len = 0 #记录任务的数量
        for csvfile in os.listdir(self.root):   #
            if csvfile[-3:] != "csv":
                continue   # skip fold
            full_character = os.path.join(self.root, csvfile)  # the class
            ImageName_dataframe = pd.read_csv(full_character)
            for name,Gourp_data in ImageName_dataframe.groupby(by=["label"]):
                character_idx = self.Index_Len
                self.Index_Len += 1
                full_character = name
                self.Path_dict[character_idx] = []
                self.Index_dict[character_idx] = 0  #记录的index
                self.Name_dict[character_idx] = name
                data_point_index = 0
                for filename in  Gourp_data.iloc[:,0]:
                    self.Path_dict[character_idx].append(os.path.join("/home/huang/Desktop/Hw/ConvexOptimization/Data/mini-imagenet/images", filename))
                    data_point_index += 1
                self.Num_dict[character_idx] = data_point_index
        self.GengerateTaskSet()

        
        
    def GengerateTaskSet(self,):
        """制作任务的集合"""
        Classes = self.arg.classes  #每个task的类别
        self.Count_Num_dict = self.Num_dict.copy()
        Each_class_datapoint = self.arg.C_datapoints # 每个class的数据量
        Class_list = list(range(self.Index_Len))
        Task_Set = {}  #存储的task的集合
        Task_count = 0
        while len(Class_list) > Classes: #可用的类别大于分类类别
            Task_Set[Task_count] = {} 
            Selcet_list = np.random.choice(Class_list,Classes,replace=False)
            for Class_index in Selcet_list:
                Start_index = self.Index_dict[Class_index] #开始的index
                Task_Set[Task_count][Class_index] = [Start_index,Start_index+Each_class_datapoint]
                self.Count_Num_dict[Class_index] -= Each_class_datapoint
                self.Index_dict[Class_index] = Start_index+Each_class_datapoint
            Class_list =self.DeletEle(Class_list,Selcet_list)
            Task_count +=1
        self.Task_Set = Task_Set

    def DeletEle(self,Target_list,Selcet_list):
        for Class_index in Selcet_list:
            if self.Count_Num_dict[Class_index] < self.arg.C_datapoints:  #数据量不够制作下一个类别
                Target_list.remove(Class_index)
        return Target_list

    def __len__(self,):
        return len(self.Task_Set)

    def __getitem__(self,index):
        Task = self.Task_Set[index]  #第index任务
        select_list = list(Task.keys())  #选取任务的列表
        Path_test = {}
        Path_Meta ={}
        counts_Meta =0
        counts_test =0
        Val_precent = self.arg.Val_precent  #验证集合比例
        Val_num = int( self.arg.C_datapoints*Val_precent)
        Meta_label = []
        Test_label = []
        for index_class in Task:
            Selcet_Val_index = np.random.choice( self.arg.C_datapoints,Val_num,replace=False)
            index_list = Task[index_class]
            Start_index,end_index = index_list
            path_list = self.Path_dict[index_class][Start_index:end_index]
            
            for index in range(len(path_list)):
                path_ele = path_list[index]
                if index in Selcet_Val_index:
                    Path_test[counts_test] ={}
                    Path_test[counts_test]['path'] = path_ele
                    Path_test[counts_test]['label'] = int(index_class)
                    counts_test += 1
                    Test_label.append(int(index_class))

                else:
                    Path_Meta[counts_Meta] ={}
                    Path_Meta[counts_Meta]['path'] = path_ele
                    Path_Meta[counts_Meta]['label'] = int(index_class)
                    counts_Meta += 1
                    Meta_label.append(int(index_class))
        Path_Meta_alpha  ={}
        counts_Meta_alpha = 0

        for i in Path_Meta:
            Path_Meta_alpha[counts_Meta_alpha] ={}
            Path_Meta_alpha[counts_Meta_alpha]['path'] = path_ele
            Path_Meta_alpha[counts_Meta_alpha]['label'] = int(index_class)
            counts_Meta_alpha += 1
            Path_Meta_alpha[counts_Meta] ={}
            Path_Meta_alpha[counts_Meta]['path'] = path_ele
            Path_Meta_alpha[counts_Meta]['label'] = int(index_class)
            counts_Meta += 1
        return FewShot(Path_Meta,self.transforms_image,select_list),FewShot(Path_test,self.transforms_image,select_list)









