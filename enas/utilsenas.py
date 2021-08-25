# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = dict()
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res["acc{}".format(k)] = correct_k.mul_(1.0 / batch_size).item()
    return res


def reward_accuracy(output, target, topk=(1,)):
    batch_size = target.size(0)
    _, predicted = torch.max(output.data, 1)
    return (predicted == target).sum().item() / batch_size


def accuracy_binary(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)
    
    # print(f"Output {output}")
    # print(f"target {target}")

    pred = (output > 0.5).float()
    # _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # print(f"pred {pred}")
    
    # # one-hot case
    # if target.ndimension() > 1:
    #     target = target.max(1)[1]
    trueLabels = (target > 0.5).float()
    # print(f"trueLabels {trueLabels}")
    
    correct = pred.eq(trueLabels.view(1, -1).expand_as(pred))
    # print(f"correct {correct}")
    res = dict()
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res["acc{}".format(k)] = correct_k.mul_(1.0 / batch_size).item()
    return res

def corrects(output, target):
    a = list(target.size())
    y_true = torch.reshape(target, (list(target.size())[0],)) 
    y_prob = torch.reshape(output, (list(output.size())[0],)) 
    assert y_true.ndim == 1 and y_true.size() == y_prob.size()
    y_prob = y_prob > 0.5
    return (y_true == y_prob).sum().item() / y_true.size(0)

def reward_accuracy_binary(output, target, topk=(1,)):
    y_true = torch.reshape(target, (list(target.size())[0],)) 
    y_prob = torch.reshape(output, (list(output.size())[0],)) 
    assert y_true.ndim == 1 and y_true.size() == y_prob.size()
    y_prob = y_prob > 0.5
    return (y_true == y_prob).sum().item() / y_true.size(0)