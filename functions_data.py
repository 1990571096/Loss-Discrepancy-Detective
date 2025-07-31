from __future__ import print_function

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import random_split
CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
MinMaxNormalization = MinMaxScaler()

# 将模型预测损失（在这种情况下是预测熵）的分布可视化，以区分干净样本和中毒样本。
def visualize(args,losses, poisoned_vector, path):
    # 根据 poisoned_vector 将损失值分为两组，一组是干净样本的损失值（clean_loss），
    # 另一组是中毒样本的损失值（poison_loss）。
    clean_loss = losses[np.where(poisoned_vector == 0)]
    poison_loss = losses[np.where(poisoned_vector == 1)]
    fig = plt.figure()
    # X轴（entropy）：表示预测熵的值，范围从0到1
    # Y轴（PDF）：表示概率密度函数，显示具有特定熵值的样本数量。
    plt.hist([clean_loss, poison_loss], bins=20, label=['clean', 'poison'])
    plt.xlabel('entropy', fontsize=15)
    plt.ylabel('PDF', fontsize=15)
    plt.legend(fontsize=14, loc='best')
    plt.tick_params(axis='both', which='major', labelsize=14)  # 增大刻度标签字体大小
    #plt.tight_layout()
    #plt.show()
    save_dir = os.path.join('visualize', args.model_name, args.dataset, args.trigger_type)
    # 如果目录不存在，则创建目录
    os.makedirs(save_dir, exist_ok=True)
    # 拼接完整的保存路径
    save_path = os.path.join(save_dir, path)

    plt.savefig(save_path)

    plt.close(fig)

def model_cmp_loss(args, model, pre_loss, eval_loader1, poisoned_vector,test_log,isabl=False):

    model.eval()
    num_iter = (len(eval_loader1.dataset) // eval_loader1.batch_size) + 1
    losses = torch.zeros(len(eval_loader1.dataset))

    with torch.no_grad():
        for batch_idx, (inputs1, targets1, index) in enumerate(eval_loader1):
            inputs1, targets1 = inputs1.cuda(), targets1.cuda()
            logits1 = model(inputs1)
            logits_softmax1 = torch.softmax(logits1, dim=1)   # 模型输出的 Softmax 分布。
            #logits_softmax1 = logits_softmax1.float()
            for b in range(inputs1.size(0)):  # 将每个样本的损失值存储到 losses 中。  #当 Softmax 输出中包含非常小的值时，计算对数（torch.log）时可能会出现 -inf，从而导致 nan。
                losses[index[b]] = -torch.mean(torch.mul(logits_softmax1[b, :], torch.log(logits_softmax1[b, :]+ 1e-9)))
                #F.cross_entropy(logits1[b, :].unsqueeze(0), targets1[b].unsqueeze(0), reduction='none').item()  # F.cross_entropy 用的类别标签，而不是one-hot 编码
                # if np.isnan(losses[index[b]]):
                #     print(logits_softmax1[b, :])
            #sys.stdout.write('\r')
            #sys.stdout.write('| Evaluating loss Iter[%3d/%3d]\t' % (batch_idx, num_iter))
            #sys.stdout.flush()
    # 根据阈值 threshold 筛选出可信样本和可疑样本。
    losses = losses.numpy()
    losses[np.isnan(losses)] = 0
    if len(pre_loss) != 0:   # 则对当前损失值和历史损失值进行加权平均，平滑处理。
        losses = args.r * losses + (1 - args.r) * pre_loss
    else:
        losses = losses

    losses = MinMaxNormalization.fit_transform(losses.reshape(-1, 1))  # loss归一化 方便比较
    losses = np.squeeze(losses, 1)  # 去除二维数组中的单维度条目，将其转换回一维数组。

    clean_loss = losses[np.where(poisoned_vector == 0)]
    poison_loss = losses[np.where(poisoned_vector == 1)]
    clean_loss_avg = sum(clean_loss)/len(clean_loss)
    poison_loss_avg = sum(poison_loss)/len(poison_loss)
    #print("\n中毒样本与干净样本的损失值差异")
    print("\npoison_loss_avg-clean_loss_avg：",poison_loss_avg-clean_loss_avg,"|poison_loss_avg-clean_loss_avg|：",abs(poison_loss_avg-clean_loss_avg))
    if isabl:
        test_log.write('After ABL poison_loss_avg: %.2f  clean_loss_avg: %.2f  \n' % (poison_loss_avg,clean_loss_avg))
    else:
        test_log.write('No ABL  poison_loss_avg: %.2f  clean_loss_avg: %.2f  \n'  % (poison_loss_avg,clean_loss_avg))

def create_dataset_with_highest_losses(losses, dataset, percentage=0.2):
    """
    使用损失值筛选数据集，选择损失值最大的前 percentage 的数据。

    参数:
        losses (numpy.ndarray): 每个样本的损失值。
        dataset (torch.utils.data.Dataset): 原始数据集。
        percentage (float, optional): 要选择的数据百分比。默认值为 0.2（20%）。

    返回:
        subset (torch.utils.data.Subset): 包含前 percentage 损失值最大的样本的子数据集。
    """
    # 确定要选择的样本数量
    num_samples = len(dataset)
    num_to_select = int(num_samples * percentage)

    # 按损失值降序排列索引
    sorted_indices = np.argsort(-losses)  # 按损失值降序排列的索引

    # 选择前 num_to_select 个索引
    selected_indices = sorted_indices[:num_to_select]

    # 创建子数据集
    subset = torch.utils.data.Subset(dataset, selected_indices)
    subset_indices = subset.indices
    poisoned_index = subset.dataset.poisoned_vector
    return subset, subset_indices, poisoned_index


def eval_train(args,model, pre_loss, eval_loader1, poisoned_vector,epoch, threshold=None,test_log=None):
    """
    返回一个全为 1 的数组，pred1（预测干净样本的索引）
    threshold（阈值），   pre_loss（更新后的损失值）。
    评估模型在训练数据上的表现，计算每个样本的预测熵。
    根据预测熵筛选出可信样本和可疑样本。
    计算评估的准确率和召回率。
    返回筛选结果和更新后的损失值。
    """
    model.eval()
    num_iter = (len(eval_loader1.dataset) // eval_loader1.batch_size) + 1
    losses = torch.zeros(len(eval_loader1.dataset))

    with torch.no_grad():
        for batch_idx, (inputs1, targets1, index) in enumerate(eval_loader1):
            inputs1, targets1 = inputs1.cuda(), targets1.cuda()
            logits1 = model(inputs1)
            logits_softmax1 = torch.softmax(logits1, dim=1)   # 模型输出的 Softmax 分布。
            #logits_softmax1 = logits_softmax1.float()
            for b in range(inputs1.size(0)):  # 将每个样本的损失值存储到 losses 中。
                losses[index[b]] = -torch.mean(torch.mul(logits_softmax1[b, :], torch.log(logits_softmax1[b, :])))
                # if np.isnan(losses[index[b]]):
                #     print(logits_softmax1[b, :])
            #sys.stdout.write('\r')
            #sys.stdout.write('| Evaluating loss Iter[%3d/%3d]\t' % (batch_idx, num_iter))
            #sys.stdout.flush()
    # 根据阈值 threshold 筛选出可信样本和可疑样本。
    losses = losses.numpy()
    losses[np.isnan(losses)] = 0
    if len(pre_loss) != 0:   # 则对当前损失值和历史损失值进行加权平均，平滑处理。
        losses = args.r * losses + (1 - args.r) * pre_loss
    else:
        losses = losses
    pre_loss = losses
    losses = MinMaxNormalization.fit_transform(losses.reshape(-1, 1))  # loss归一化 方便比较
    losses = np.squeeze(losses, 1)  # 去除二维数组中的单维度条目，将其转换回一维数组。

    # 统计位于0.15到0.2之间的loss值的数量
    count = np.sum((losses >= 0.15) & (losses <= 0.25))   # 判断是否形成分隔
    if count/len(losses)<=0.005:
        isbreak=True
    else:
        isbreak=False
    #isbreak = False

    clean_loss = losses[np.where(poisoned_vector == 0)]
    poison_loss = losses[np.where(poisoned_vector == 1)]
    clean_loss_avg = sum(clean_loss)/len(clean_loss)
    poison_loss_avg = sum(poison_loss)/len(poison_loss)
    print("\nclean_loss_avg：",clean_loss_avg,"poison_loss_avg：",poison_loss_avg)

    visualize(args,losses, poisoned_vector, str(args.posioned_portion)+'_'+str(epoch) + '.png')

    pred1 = losses > threshold  # 可信样本筛选（损失值大于阈值）。1为干净样本

    #  加一个数据集的纯度 分离出的样本纯度
    clean_purity = np.sum((pred1 == 1) & (poisoned_vector == 0)) / np.sum(pred1 == 1)
    poisoned_purity = np.sum((pred1 == 0) & (poisoned_vector == 1)) / np.sum(pred1 == 0)
    # 计算评估判别中毒样本的准确率和召回率。
    # 预测和实际的相等的/数据集大小

    correct = np.count_nonzero(np.equal(pred1, 1-poisoned_vector)) / len(poisoned_vector)  #  中毒样本预测正确的
    Recall = np.sum((1-pred1)*poisoned_vector) / np.sum(poisoned_vector)  # 表示正确识别的中毒样本比例。 预测的中毒样本所占总数
    if test_log is not None:
        test_log.write('\nEpoch: %d\n' % epoch)
        test_log.write('clean_purity:%.5f    poisoned_purity:%.5f  \n' % (clean_purity, poisoned_purity))
        test_log.write('correct:%.5f    Recall:%.5f  \n' % (correct, Recall))
    print('eval_train acc: %.5f recall: %.5f' % (correct, Recall))

    print("Data Separation Purity Calculation")
    print('clean_purity: %.5f poisoned_purity: %.5f' % (clean_purity, poisoned_purity))
    return np.ones_like(losses), pred1, threshold, pre_loss,losses,isbreak

def finally_evaluate(args,model, pre_loss, eval_loader1, poisoned_vector,epoch, threshold=None,test_log=None,select_log_file=None):
    """
    pred1 : clean
    pred2 : poison
    """
    model.eval()
    num_iter = (len(eval_loader1.dataset) // eval_loader1.batch_size) + 1
    losses = torch.zeros(len(eval_loader1.dataset))

    with torch.no_grad():
        for batch_idx, (inputs1, targets1, index) in enumerate(eval_loader1):
            inputs1, targets1 = inputs1.cuda(), targets1.cuda()
            logits1 = model(inputs1)
            logits_softmax1 = torch.softmax(logits1, dim=1)   # 模型输出的 Softmax 分布。
            #logits_softmax1 = logits_softmax1.float()
            for b in range(inputs1.size(0)):  # 将每个样本的损失值存储到 losses 中。
                losses[index[b]] = -torch.mean(torch.mul(logits_softmax1[b, :], torch.log(logits_softmax1[b, :])))
                # if np.isnan(losses[index[b]]):
                #     print(logits_softmax1[b, :])
            #sys.stdout.write('\r')
            #sys.stdout.write('| Evaluating loss Iter[%3d/%3d]\t' % (batch_idx, num_iter))
            #sys.stdout.flush()
    # 根据阈值 threshold 筛选出可信样本和可疑样本。
    losses = losses.numpy()
    losses[np.isnan(losses)] = 0
    if len(pre_loss) != 0:   # 则对当前损失值和历史损失值进行加权平均，平滑处理。
        losses = args.r * losses + (1 - args.r) * pre_loss
    else:
        losses = losses
    pre_loss = losses
    losses = MinMaxNormalization.fit_transform(losses.reshape(-1, 1))  # loss归一化 方便比较
    losses = np.squeeze(losses, 1)  # 去除二维数组中的单维度条目，将其转换回一维数组。

    clean_loss = losses[np.where(poisoned_vector == 0)]
    poison_loss = losses[np.where(poisoned_vector == 1)]

    # clean_filename = f"clean_loss_{args.posioned_portion}.npy"
    # clean_save_path = os.path.join(select_log_file, clean_filename)
    # poison_filename = f"poison_loss_{args.posioned_portion}.npy"
    # poison_save_path = os.path.join(select_log_file, poison_filename)
    # np.save(clean_save_path,clean_loss)
    # np.save(poison_save_path, poison_loss)


    clean_loss_avg = sum(clean_loss) / len(clean_loss)
    poison_loss_avg = sum(poison_loss) / len(poison_loss)
    print("\nclean_loss_avg：", clean_loss_avg, "poison_loss_avg：", poison_loss_avg)


    visualize(args,losses, poisoned_vector, str(args.posioned_portion)+'_'+str(epoch)+ '.png')

    pred1 = losses < threshold  # poison
    pred2 = losses > 0.2   # clean
    #  加一个数据集的纯度 分离出的样本纯度
    clean_purity = np.sum((pred2 == 1) & (poisoned_vector == 0)) / np.sum(pred2 == 1)
    poisoned_purity = np.sum((pred1 == 1) & (poisoned_vector == 1)) / np.sum(pred1 == 1)
    # 计算评估判别中毒样本的准确率和召回率。
    # 预测和实际的相等的/数据集大小

    #correct = np.count_nonzero(np.equal(pred1, 1-poisoned_vector)) / len(poisoned_vector)  #  中毒样本预测正确的
    #Recall = np.sum((1-pred1)*poisoned_vector) / np.sum(poisoned_vector)  # 表示正确识别的中毒样本比例。 预测的中毒样本所占总数
    if test_log is not None:
        test_log.write('\nEpoch: %d\n' % epoch)
        test_log.write('clean_purity:%.5f    poisoned_purity:%.5f  \n' % (clean_purity, poisoned_purity))
        #test_log.write('correct:%.5f    Recall:%.5f  \n' % (correct, Recall))
    #print('eval_train acc: %.5f recall: %.5f' % (correct, Recall))

    print("Data Separation Purity Calculation")
    print('clean_purity: %.5f poisoned_purity: %.5f' % (clean_purity, poisoned_purity))
    return pred1, pred2



def warmup(args, epoch, net, optimizer, dataloader):
    # 用于对模型进行预热训练。预热训练的目的是让模型在训练初期快速适应数据，通常用于稳定训练过程和提高模型性能。
    net.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1   # 有多少个批次
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = CEloss(outputs, labels)

        #penalty = conf_penalty(outputs)
        L = loss #+ penalty

        L.backward()
        optimizer.step()
        # 在终端输出当前的训练进度和损失值。
        # sys.stdout.write('\r')
        # sys.stdout.write('%s| Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
        #                  % (args.dataset, epoch, args.num_epochs, batch_idx + 1, num_iter, loss.item()))
        # sys.stdout.flush()

def train_step_unlearning(args, epoch, net, optimizer, labeled_trainloader,poisoned_vector):
    # Input: args,训练轮次，网络，优化器，数据集加载器
    # Return：null
    # 用CE训练模型
    net.train()
    # 创建数据的子集 DataLoader
    dataset = labeled_trainloader.dataset
    subset_indices = dataset.indices
    poisoned_index = poisoned_vector   # 原来的中毒列表索引，1为中毒
    total_length = len(subset_indices)  # 长度
    #weight = 0.001 + 0.099 * (1 + math.sin(math.pi * (epoch-1) / 10)) / 2
    #weight =-0.00396 * (epoch-1)**2 + 0.0396 * (epoch-1) + 0.001
    #weight = 0.0049 * (epoch-1) + 0.001
    # weight = 0.001 + (0.05 - 0.001) / 9 * (epoch-1)
    # #weight = -0.0000796 * (epoch-1) ** 3 - 0.006766 * (epoch-1) ** 2 + 0.07562 * (epoch-1) + 0.001
    # print(f"\n Epoch {epoch}: weight = {weight:.6f}")
    # #weight = 0.001 + (0.1 - 0.001) / 10 * (epoch-1)
    # split_length = int(total_length * weight)
    # subset_, _ = random_split(dataset, [split_length, total_length-split_length])
    # subset_indices = subset_.indices
    # poisoned_index = subset_.dataset.poisoned_vector
    #
    #
    #
    # 根据子集索引统计中毒和正常样本的数量
    poisoned_count = np.sum(poisoned_index [subset_indices])
    normal_count = len(subset_indices) - poisoned_count

    # 计算纯度

    purity_poisoned = poisoned_count / total_length  # 中毒样本的纯度
    purity_normal = normal_count / total_length  # 正常样本的纯度
    print(f"\n子集中中毒样本的数量: {poisoned_count}")
    print(f"子集中正常样本的数量: {normal_count}")
    print(f"子集中中毒样本的纯度: {purity_poisoned:.4f}")
    print(f"子集中正常样本的纯度: {purity_normal:.4f}")
    print("ABL所用的样本大小:",total_length)
    #subset_loader = torch.utils.data.DataLoader(subset_, batch_size=args.batch_size, shuffle=True)

    #num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1

    for batch_idx, (inputs_x, labels_x, _) in enumerate(labeled_trainloader):
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)
        #w_x = w_x.view(-1, 1).type(torch.FloatTensor)

        inputs_x, labels_x = inputs_x.cuda(), labels_x.cuda()

        logits = net(inputs_x)
        Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * labels_x, dim=1))
        loss = Lx

        # compute gradient and do SGD step
        optimizer.zero_grad()
        (-loss).backward()
        optimizer.step()

        # sys.stdout.write('\r')
        # sys.stdout.write('%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
        #                  % (args.dataset, epoch, args.num_epochs, batch_idx + 1, num_iter,
        #                     Lx.item(), 0))
        # sys.stdout.flush()

def train_clean_model(args, epoch, net, optimizer, labeled_trainloader):
    # 用CE训练模型
    net.train()

    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
    for batch_idx, (inputs_x, _, _, labels_x, w_x) in enumerate(labeled_trainloader):
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)
        w_x = w_x.view(-1, 1).type(torch.FloatTensor)

        inputs_x, labels_x, w_x = inputs_x.cuda(), labels_x.cuda(), w_x.cuda()

        logits = net(inputs_x)
        Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * labels_x, dim=1))
        loss = Lx

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # sys.stdout.write('\r')
        # sys.stdout.write('%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
        #                  % (args.dataset, epoch, args.num_epochs, batch_idx + 1, num_iter,
        #                     Lx.item(), 0))
        # sys.stdout.flush()

def train_poisoned_model(args, epoch, net, optimizer, labeled_trainloader):
    # 用CE训练模型
    net.train()

    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
    for batch_idx, (inputs_x, _, _, labels_x, w_x) in enumerate(labeled_trainloader):
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)
        w_x = w_x.view(-1, 1).type(torch.FloatTensor)

        inputs_x, labels_x, w_x = inputs_x.cuda(), labels_x.cuda(), w_x.cuda()

        logits = net(inputs_x)
        Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * labels_x, dim=1))
        loss = Lx

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # sys.stdout.write('\r')
        # sys.stdout.write('%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
        #                  % (args.dataset, epoch, args.num_epochs, batch_idx + 1, num_iter,
        #                     Lx.item(), 0))
        # sys.stdout.flush()


def test(epoch, net1, net2, test_loader1, test_loader2, test_log,isabl=True):
    """
    Return：
    net1 在 test_loader1 上的 Accuracy,
    net2 在 test_loader2 上的 Accuracy
    """
    net1.eval()
    net2.eval()
    correct1 = 0
    correct2 = 0
    total1 = 0
    total2 = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(test_loader1):
            inputs, targets = inputs.cuda(), targets.cuda()
            if isabl:
                outputs1 = net1(inputs)
            else:
                outputs1, _ = net1(inputs)

            _, predicted1 = torch.max(outputs1, 1)
            total1 += targets.size(0)
            correct1 += predicted1.eq(targets).cpu().sum().item()

        for batch_idx, (inputs, targets, _) in enumerate(test_loader2):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs2 = net2(inputs)
            _, predicted2 = torch.max(outputs2, 1)
            total2 += targets.size(0)
            correct2 += predicted2.eq(targets).cpu().sum().item()

    acc1 = 100. * correct1 / total1
    acc2 = 100. * correct2 / total2
    print("| Test Epoch #%d\t Accuracy: %.2f%%\t Accuracy: %.2f%%" % (epoch, acc1, acc2))
    test_log.write('Epoch:%d   Accuracy1:%.2f   Accuracy2:%.2f\n' % (epoch, acc1, acc2))
    test_log.flush()
    return acc1, acc2

def save_state(epoch, net, optimizer, poisoned_index, noisy_idx, pre_loss, threshold, path):
    saved_dict = {
        "epoch": epoch,
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "poisoned_index": poisoned_index,
        "noisy_index": noisy_idx,
        "pre_loss": pre_loss,
        "threshold": threshold
    }
    torch.save(saved_dict, path)

def load_state(net, optimizer, path):
    ckpt = torch.load(path)
    net.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    poisoned_index = ckpt["poisoned_index"]
    epoch = ckpt["epoch"]
    noisy_idx = ckpt["noisy_index"]
    threshold = None
    pre_loss = None
    if "threshold" in ckpt.keys():
        threshold = ckpt["threshold"]
    if "pre_loss" in ckpt.keys():
        pre_loss = ckpt["pre_loss"]

    return epoch, poisoned_index, noisy_idx, pre_loss, threshold
