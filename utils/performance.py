import torch
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np
class performance():

    # def get_acper_bcper_acer(predictions,labels):
    def test_roc(self,probabilities,true_labels,name):
        fpr, tpr, thresholds =roc_curve(true_labels,probabilities)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.title("ROC")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.savefig(f'{name}_ROC.png')

    def get_acper_bcper(self,predictions,labels,threshold):
        binary_predictions = (predictions > threshold).float()
        # 计算 APCER 和 BPCER
        apcer = ((binary_predictions == 0) & (labels == 1)).float().mean()
        bpcer = ((binary_predictions == 1) & (labels == 0)).float().mean()
        acer = (apcer + bpcer)/2
        return apcer,bpcer,acer


    def loss_pictrue(self,protocol,
                     loss_bec_count_list=None,
            ):
        plt.figure()
        x = np.arange(len(loss_bec_count_list))
        plt.plot(x, loss_bec_count_list, label='bec_loss')
        # plt.plot(x, loss_text_count_list, label='loss_text_count')
        # plt.plot(x, loss_difference_count_list, label='loss_difference')
        # plt.plot(x, loss_all_count_list, label='loss_all')
        plt.legend()
        plt.savefig(f'{protocol}_loss.png')
    def loss_OULU_pictrue(self,protocol,
                     loss_bec_count_list=None,
                     loss_text_count_list=None,
                     loss_difference_count_list=None,
                     loss_all_count_list=None
            ):
        plt.figure()
        x = np.arange(len(loss_bec_count_list))
        plt.plot(x, loss_bec_count_list, label='bec_loss')
        plt.plot(x, loss_text_count_list, label='loss_text_count')
        plt.plot(x, loss_difference_count_list, label='loss_difference')
        plt.plot(x, loss_all_count_list, label='loss_all')
        plt.legend()
        plt.savefig(f'{protocol}_loss.png')
    def get_threshold(self,predictions,labels):
        thresholds = np.linspace(0, 1, 100)  # 生成一系列可能的阈值
        best_threshold = None
        best_error = float('inf')
        for threshold in thresholds:
            # 对预测值进行阈值化，生成二值预测（假设大于阈值的预测为真实样本）
            binary_predictions = (predictions > threshold).float()

            # 计算 APCER 和 BPCER
            apcer = ((binary_predictions == 0) & (labels == 1)).float().mean()
            bpcer = ((binary_predictions == 1) & (labels == 0)).float().mean()

            error = apcer + bpcer

            # 更新最优阈值和最小错误值
            if error < best_error:
                best_error = error
                best_threshold = threshold

        return best_threshold