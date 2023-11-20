import os
import torch
from Load_Replay import Load_Replay
from Load_CelebA import Load_CelebA
from Load_CASIA_SURF import Load_CASIA
from Load_OULU   import Load_OULU
from Load_MSU import Load_MSU
class data_merge(object):
    def __init__(self,root):
        super(data_merge, self).__init__()
        self.root=root
    def get_single_dataset(self,data_name="",train=True,transform=None,UUID=-1):
        if data_name =="OULU":
            data_set=Load_OULU(self.root,train=train,transforms=transform,UUID=UUID)
        elif data_name =="Replay":
            data_set=Load_Replay(self.root,train=train,transforms=transform,UUID=UUID)
        elif data_name =="CASIA":
            data_set=Load_CASIA(root_dir=self.root,train=train,transforms=transform,UUID=UUID)
        elif data_name =="MSU":
            data_set=Load_MSU(root_dir=self.root,train=train,transforms=transform,UUID=UUID)
        else:
            data_set=None
            print("error data")
        return data_set
    def get_datasets(self, train=True, protocol="1", transform=None):
        if protocol == "O_C_I_to_M":
            data_name_list_train = ["OULU", "CASIA", "Replay"]
            data_name_list_test = ["MSU"]
        elif protocol == "O_M_I_to_C":
            data_name_list_train = ["OULU", "MSU", "Replay"]
            data_name_list_test = ["CASIA"]
        elif protocol == "O_C_M_to_I":
            data_name_list_train = ["OULU", "CASIA", "MSU"]
            data_name_list_test = ["Replay"]
        elif protocol == "I_C_M_to_O":
            data_name_list_train = ["MSU", "CASIA", "Replay"]
            data_name_list_test = ["OULU"]
        elif protocol == "M_I_to_C":
            data_name_list_train = ["MSU", "Replay"]
            data_name_list_test = ["CASIA"]
        elif protocol == "M_I_to_O":
            data_name_list_train = ["MSU", "Replay"]
            data_name_list_test = ["OULU"]
        sum_n = 0
        if train:
            data_set_sum = self.get_single_dataset(data_name=data_name_list_train[0], train=True,transform=transform,UUID=0)
            sum_n = len(data_set_sum)
            for i in range(1, len(data_name_list_train)):
                data_tmp = self.get_single_dataset(data_name=data_name_list_train[i], train=True, transform=transform, UUID=i)
                data_set_sum += data_tmp
                sum_n += len(data_tmp)
        else:
            data_set_sum = {}
            for i in range(len(data_name_list_test)):
                data_tmp = self.get_single_dataset(data_name=data_name_list_test[i], train=False, transform=transform, UUID=i)
                data_set_sum[data_name_list_test[i]] = data_tmp
                sum_n += len(data_tmp)
        print("Total number: {}".format(sum_n))
        return data_set_sum

