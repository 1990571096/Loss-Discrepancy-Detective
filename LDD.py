from __future__ import print_function

import os
import sys
import math
import numpy as np
import torch.backends.cudnn
from torch.cuda.amp import GradScaler
import argparse
import torch.optim as optim
import random

from torch.optim.lr_scheduler import StepLR

from dataloader_data import data_dataloader
from models.wresnet import WideResNet
from models.resnet_cifar import resnet18, resnet34
from models.preact_resnet import PreActResNet18
from models.Conv4 import ConvNet
from models.Lenet import LeNet5
from functions_data import eval_train, finally_evaluate,warmup, train_poisoned_model, test, save_state, create_dataset_with_highest_losses,load_state,train_clean_model,train_step_unlearning,train_clean_model_withssl,model_cmp_loss
import time


def  create_model(model_name,
                 pretrained=False,
                 pretrained_models_path=None,
                 n_classes=10):

    assert model_name in ['WRN-16-1', 'WRN-16-2', 'WRN-40-1', 'WRN-40-2', 'ResNet34', 'WRN-10-2', 'WRN-10-1', 'PreActResNet18', 'ResNet18', 'Conv4',"LeNet5"]
    if model_name=='WRN-16-1':
        model = WideResNet(depth=16, num_classes=n_classes, widen_factor=1, dropRate=0)
    elif model_name=='WRN-16-2':
        model = WideResNet(depth=16, num_classes=n_classes, widen_factor=2, dropRate=0)
    elif model_name=='WRN-40-1':
        model = WideResNet(depth=40, num_classes=n_classes, widen_factor=1, dropRate=0)
    elif model_name=='WRN-40-2':
        model = WideResNet(depth=40, num_classes=n_classes, widen_factor=2, dropRate=0)
    elif model_name == 'WRN-10-2':
        model = WideResNet(depth=10, num_classes=n_classes, widen_factor=2, dropRate=0)
    elif model_name == 'WRN-10-1':
        model = WideResNet(depth=10, num_classes=n_classes, widen_factor=1, dropRate=0)
    elif model_name=='ResNet34':
        model = resnet34( num_classes=n_classes)
    elif model_name == 'PreActResNet18':
        model = PreActResNet18(num_classes=n_classes)
    elif model_name == 'ResNet18':
        model = resnet18(num_classes=n_classes)
    elif model_name == 'Conv4':
        model = ConvNet(num_class=n_classes)
    elif model_name == 'LeNet5':
        model = LeNet5(num_class=n_classes)
    else:
        raise NotImplementedError

    checkpoint_epoch = None
    if pretrained:
        model_path = os.path.join(pretrained_models_path)
        print('Loading Model from {}'.format(model_path))
        checkpoint = torch.load(model_path, map_location='cpu')
        print(checkpoint.keys())
        model.load_state_dict(checkpoint['state_dict'])

        checkpoint_epoch = checkpoint['epoch']
        print("=> loaded checkpoint '{}' (epoch {}) ".format(model_path, checkpoint['epoch']))

    return model, checkpoint_epoch

# 记录程序开始运行的时间






parser = argparse.ArgumentParser(description='PyTorch Web-bird Training')
parser.add_argument('--batch_size', default=128, type=int, help='train batchsize')
parser.add_argument('--poisoned_mode', default='all2one', type=str)
parser.add_argument('--posioned_portion', default=0.6, type=float)
# 表示是否从之前的训练状态恢复
parser.add_argument('--resume', default=False, type=bool)
parser.add_argument('--trigger_label', default=0, type=int)
parser.add_argument('--trigger_type', default='WaNet', type=str)  # 只需修改的地方 badnet blended SIG CL WaNet
parser.add_argument('--trigger_path', default='./trigger/cifar10/hello_kitty.png', type=str)  # 触发器 根据中毒类型自动计算
parser.add_argument('--lr_clean', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--lr_poison', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--r', default=0.75, type=float)    # 损失加权值
parser.add_argument('--num_epochs', default=10, type=int)  # 训练轮次
parser.add_argument('--dataset', default='gtsrb', type=str)  # cifar10  gtsrb  cifar100
parser.add_argument('--seed', default=123)
#parser.add_argument('--num_class', default=100, type=int)  # 总类别数 自动计算，不指定
parser.add_argument('--model_name', default='ResNet18', type=str)  # WRN-10-1  ResNet18
parser.add_argument('--data_path', default='./dataset/', type=str, help='path to dataset')
parser.add_argument('--storage_path', default='./storage/', type=str, help='path to storage')
parser.add_argument('--isolation_ratio', type=float, default=0.1, help='ratio of isolation data')

parser.add_argument('--alpha1', default=0.1, type=float)
parser.add_argument('--alpha2', default=0.5, type=float)
args = parser.parse_args()


poison_rate=[0.6]   # 0.1,0.2,0.3,0.5,0.6   # 0.01,0.05,0.13,0.17
start_time = time.time()
data_name = [ 'gtsrb']  # 'cifar10' ,'cifar100','gtsrb'
attack_mode=['SIG']  # 'badnet','blended', 'SIG','CL', 'WaNet'
for i in poison_rate:
    args.posioned_portion = i
    for j in data_name:
        args.dataset = j
        for k in attack_mode:
            args.trigger_type=k

            if args.trigger_type=="blended":
                args.trigger_path='./trigger/cifar10/hello_kitty.png'
            elif args.trigger_type=="badnet":
                args.trigger_path = './trigger/cifar10/cifar_1.png'
            else:
                pass


            random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            np.random.seed(args.seed)
            torch.backends.cudnn.deterministic = True
            # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            print(args)

            # 创建存储路径和日志文件，用于保存模型和记录测试结果。

            test_log_path = os.path.join('checkpoint', args.model_name, args.dataset, args.trigger_type)

            # 检查路径是否存在，如果不存在则创建
            if not os.path.exists(test_log_path):
                os.makedirs(test_log_path)

            # 创建日志文件路径
            test_log_file = os.path.join(test_log_path, str(args.posioned_portion) + '.txt')  # 触发器类型作为文件名
            # if  os.path.exists(test_log_file):
            #     continue
            test_log = open(test_log_file, 'w')  # 打开（或创建）一个文本文件，用于写入测试日志

            # test_log_path = os.path.join('checkpoint', args.model_name, args.dataset)
            # if not os.path.exists(test_log_path):
            #     os.makedirs(test_log_path)
            # test_log = open(test_log_path + args.trigger_type + '.txt', 'w')  # 打开（或创建）一个文本文件，用于写入测试日志

            best_acc = 0
            asr = 1
            flag = 0
            threshold = None

            start_epoch1 = 0
            start_epoch2 = 0
            poisoned_idx = []
            noisy_idx = []

            print('| Building dataloader')
            # num_works 默认为4，windows报错
            loader = data_dataloader(args, args.batch_size, 0, args.data_path, args.trigger_label, args.poisoned_mode,
                                     args.posioned_portion, poisoned_idx, noisy_idx)
            # test_CL_loader1, test_BD_loader1 = loader.run('test_net1')  # 用于测试干净模型 net1 的干净样本和中毒样本加载器。
            # test_CL_loader2, test_BD_loader2 = loader.run('test_net2')  # 用于测试中毒模型 net2 的干净样本和中毒样本加载器。
            eval_loader2, poisoned_idx, noisy_idx = loader.run('eval_train_net2')  # 数据集没有打乱
            warmup_trainloader = loader.run('warmup')

            print('| Building net')
            # 其中一个网络（net2）用于检测被污染的样本，另一个网络（net1）用于在被污染的数据上训练干净的模型。

            # backdoor model
            net2, _ = create_model(args.model_name, n_classes=args.num_class)
            net2.cuda()


            optimizer_net2_abl = optim.SGD(net2.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
            optimizer2 = optim.Adam(net2.parameters(), lr=args.lr_poison, weight_decay=5e-4)

            scheduler = StepLR(optimizer_net2_abl, step_size=5, gamma=0.9)  # 每5轮衰减一次，衰减比例为0.9
            poisoned_vector = np.zeros(len(warmup_trainloader.dataset.data))  # 获取数据集大小
            print("中毒大小: ", len(poisoned_vector))
            for i in poisoned_idx:  # 中毒为1
                poisoned_vector[i] = 1


            model_filename = f"{args.posioned_portion}.pth"  # 假设保存为 .pth 格式
            model_save_path = os.path.join(test_log_path, model_filename)
            # if  os.path.exists(model_save_path):
            #     continue
            if args.resume==False:
                # 中毒样本检测（Warmup 阶段）
                if start_epoch2 == 0:
                    print('\nFirst Warmup Net2')
                    warmup(args, 0, net2, optimizer2, warmup_trainloader)
                    pre_loss = []

                threshold = 0.08

                for epoch in range(start_epoch2 + 1, args.num_epochs):  # 预热
                    # isabl=False
                    # print('\nWarmup To Train Net2,   ',"Current Epoch:",epoch)
                    # max(0.4 - 0.1 * epoch, 0.08)

                    threshold = max(0.3 - 0.1 * (epoch - 1), 0.08)  # 每个 epoch 开始时，调整阈值 threshold，使其逐渐减小  0.6->0.2->0.2->0.2
                    #threshold =0.2
                    print(f"\nepoch: {epoch}, threshold: {threshold:.5f}")
                    # 调用 eval_train 函数，评估 net2 的训练效果，筛选出可信样本和可疑样本
                    prob2, pred2, threshold, pre_loss, data_loss,isbreak = eval_train(args, net2, pre_loss, eval_loader2, poisoned_vector,
                                                                              epoch - 1, threshold, test_log)
                    if isbreak:
                        break
                    # 可疑样本筛选
                    dataset = eval_loader2.dataset
                    total_length = len(dataset)
                    # 筛选可信样本的索引（pred2中值为1的样本）
                    trusted_indices = np.where(pred2 == 1)[0]
                    # 获取可信样本的损失值
                    trusted_losses = data_loss[trusted_indices]
                    # ABL所用的数据集大小递增
                    weight = 0.001 + (0.01 - 0.001) / (args.num_epochs - 1) * (epoch - 1)
                   # weight =0.01
                    print(f"\n Epoch {epoch}: weight = {weight:.6f}")
                    # 按损失值升序排序可信样本的索引，以选择损失值最小的前20%
                    num_to_select = int(len(trusted_indices) * weight)
                    '''
                    按类别选最高值
                    '''
                    # 获取可信样本的类别标签
                    trusted_labels = dataset.targets[trusted_indices]  # 假设 dataset 提供标签
                    num_classes = len(set(trusted_labels))
                    max_per_class=10
                    #samples_per_class = max(math.floor(num_to_select / min(num_classes, max_per_class)),1)  # 向下取整,最多一次ABL 10个类别
                    #samples_per_class = max( samples_per_class,epoch+2)
                    samples_per_class = 1
                    #samples_per_class = max(math.floor(num_to_select / num_classes), 1)
                    # 如果类别数量大于10，则随机选择10个类别
                    '''随机选max_per_class个类别'''
                    # if num_classes > max_per_class:
                    #     selected_classes = random.sample(list(set(trusted_labels)), max_per_class)  # 将集合转换为列表
                    # else:
                    #     selected_classes = set(trusted_labels)
                    '''修改成选择数量最多的前 max_per_class 个类别'''
                    # 如果类别数量大于 max_per_class，则选择数量最多的前 max_per_class 个类别
                    if num_classes > max_per_class:
                        # 统计每个类别的样本数量
                        class_counts = {}
                        for label in trusted_labels:
                            if label not in class_counts:
                                class_counts[label] = 0
                            class_counts[label] += 1

                        # 按样本数量降序排序类别，并选择前 max_per_class 个类别
                        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
                        selected_classes = [label for label, _ in sorted_classes[:max_per_class]]
                    else:
                        selected_classes = set(trusted_labels)

                    print(f"当前轮次总类别数: {num_classes}")
                    print(f"当前轮次参与采样的类别: {selected_classes}")
                    print(f"每个类别需要采样的数量: {samples_per_class}")
                    # 为选中的每个类别创建字典存储损失值和索引
                    class_dict = {}
                    for idx, label in enumerate(trusted_labels):
                        if label in selected_classes:
                            if label not in class_dict:
                                class_dict[label] = []
                            class_dict[label].append((trusted_losses[idx], trusted_indices[idx]))

                    # 从每个类别中选择损失值最高的 samples_per_class 个样本
                    selected_indices1 = []
                    for label in class_dict:
                        # 对每个类别的损失值进行排序（按损失值从高到低）
                        sorted_samples = sorted(class_dict[label], key=lambda x: -x[0])
                        # 选择损失值最高的 samples_per_class 个索引值
                        selected_samples = [sample[1] for sample in sorted_samples[:samples_per_class]]
                        selected_indices1.extend(selected_samples)

                    # # # 调整列表并转换为 NumPy 数组（如果需要）
                    # # selected_indices = np.array(selected_indices)
                    #
                    # # sorted_indices = np.argsort(trusted_losses)[4*num_to_select:5*num_to_select]
                    # # sorted_indices = np.argsort(trusted_losses)[len(trusted_indices)//2-5 * num_to_select:len(trusted_indices)//2-4 * num_to_select]
                    sorted_indices2 = np.argsort(-trusted_losses)[:1 * num_to_select]
                    # 获取最终选择的样本索引
                    selected_indices2 = trusted_indices[sorted_indices2].tolist()
                    print("Size of selected_indices1:", len(selected_indices1))
                    print("Size of selected_indices2:", len(selected_indices2))
                    selected_indices = list(set(selected_indices1 + selected_indices2))
                    print("Size of total selected_indices:", len(selected_indices1))
                    #创建子数据集
                    ABL_dataset = torch.utils.data.Subset(dataset, selected_indices)
                    ABL_loader = torch.utils.data.DataLoader(
                        ABL_dataset,
                        batch_size=args.batch_size,
                        num_workers=0,
                        shuffle=True,
                        pin_memory=True
                    )

                    labeled_suspicious_loader, labeled_credible_trainloader = loader.run('train_net2', 1 - pred2, 1 - prob2)

                    # 筛选可信样本（pred2为1的样本）

                    print("labeled_suspicious_loader数据大小：", len(labeled_suspicious_loader.dataset))
                    print("labeled_credible_trainloader数据大小：", len(labeled_credible_trainloader.dataset))
                    print("ABL_loader数据大小：", len(ABL_dataset))

                    train_poisoned_model(args, epoch, net2, optimizer2, labeled_suspicious_loader)
                    train_step_unlearning(args, epoch, net2, optimizer_net2_abl, ABL_loader, poisoned_vector)
                    #scheduler.step()  # 更新学习率
                    print('\n')

                threshold=0.08
                #print("Epoch    ,   threshold  ", epoch, ' ' * 3, threshold)
                pred1, pred2= finally_evaluate(args, net2, pre_loss, eval_loader2, poisoned_vector,
                                                                          args.num_epochs - 1, threshold, test_log, test_log_path)
                torch.save(net2.state_dict(), model_save_path)

            else:
                net2.load_state_dict(torch.load(model_save_path))
                net2.eval()


            losses = torch.zeros(len(eval_loader2.dataset))
            with torch.no_grad():
                for batch_idx, (inputs1, targets1, index) in enumerate(eval_loader2):
                    inputs1, targets1 = inputs1.cuda(), targets1.cuda()
                    logits1 = net2(inputs1)
                    logits_softmax1 = torch.softmax(logits1, dim=1)  # 模型输出的 Softmax 分布。
                    # logits_softmax1 = logits_softmax1.float()
                    for b in range(inputs1.size(0)):  # 将每个样本的损失值存储到 losses 中。
                        losses[index[b]] = -torch.mean(torch.mul(logits_softmax1[b, :], torch.log(logits_softmax1[b, :])))
            losses = losses.numpy()
            losses_idx = np.argsort(losses)   # 计算损失值，按升序排序，得到索引
            for args.isolation_ratio in range(5, 61, 5):
                print("\nCurrent isolation_ratio :", args.isolation_ratio, "%")
                perm = losses_idx[-int(len(losses_idx) * args.isolation_ratio/100):]  # 取干净的数据集比例得到的索引
                print("Current isolation Size :", len(perm))
                clean_purity = np.sum(poisoned_vector[perm] == 0) / len(perm)
                print('clean_purity: %.2f%%' % (clean_purity * 100))

                test_log.write('\nCurrent isolation_ratio: {:d}'.format(args.isolation_ratio))
                test_log.write('\nClean_purity: {:.2f}%'.format(clean_purity * 100))
            # 记录程序结束运行的时间
            end_time = time.time()
            # 计算运行时间（单位：秒）
            elapsed_time_seconds = end_time - start_time
            # 打印运行时间
            print(f"程序运行了 {elapsed_time_seconds / 60:.2f} 分钟。")
