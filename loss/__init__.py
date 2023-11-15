import torch.nn.functional as F
import torch.nn as nn
import torch
def cosine_similarity_loss(input1,input2):
    #

    input1_flatten = [tensor.view(tensor.size(0), -1) for tensor in input1]
    input2_flatten = [tensor.view(tensor.size(0), -1) for tensor in input2]

    # Compute the cosine similarity for each feature
    cosine_similarities = [F.cosine_similarity(tensor1, tensor2, dim=1) for tensor1, tensor2 in
                           zip(input1_flatten, input2_flatten)]

    # Compute the mean cosine similarity
    mean_cosine_similarity = sum([cosine_similarity.mean() for cosine_similarity in cosine_similarities]) / len(
        cosine_similarities)

    return mean_cosine_similarity

