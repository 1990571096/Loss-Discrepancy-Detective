import sys

from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import torch.nn.functional as F
import copy
import torchvision
import numpy as np
import torch
import h5py
from PIL import Image
import matplotlib.pyplot as plt

def load_init_data(args,download, dataset_path):
    if args.dataset=="cifar10":
        train_data = datasets.CIFAR10(root=dataset_path, train=True,  download=download)
        test_data = datasets.CIFAR10(root=dataset_path, train=False, download=download)
    elif args.dataset =="cifar100":
        # 加载 CIFAR-100 数据集
        train_data = datasets.CIFAR100(root=dataset_path, train=True, download=download)
        test_data = datasets.CIFAR100(root=dataset_path, train=False, download=download)
    elif args.dataset == "gtsrb":
        dataset_path = 'dataset/gtsrb_dataset.h5'
        train_data = h5_dataset(dataset_path, True, None)  # numpy格式
        test_data = h5_dataset(dataset_path, False, None)
    # 获得类别数量
    args.num_class = len(np.unique(np.array(train_data.targets)))

    return train_data, test_data


class h5_dataset(Dataset):
    def __init__(self, path, train, transform):
        f = h5py.File(path, 'r')
        if train:
            self.data = np.vstack((np.asarray(f['X_train']), np.asarray(f['X_val']))).astype(np.uint8)
            self.targets = list(np.argmax(np.vstack((np.asarray(f['Y_train']), np.asarray(f['Y_val']))), axis=1))
        else:
            self.data = np.asarray(f['X_test']).astype(np.uint8)
            self.targets = list(np.argmax(np.asarray(f['Y_test']), axis=1))

        # 如果用户没有提供 transform，则使用默认的
        #self.transform = transform if transform is not None else default_transform
        self.num_classes = len(np.unique(self.targets))

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]

        # 应用转换操作
        if self.transform is not None:
            # 将 NumPy 数组转换为 PIL 图像
            image = Image.fromarray(image)
            image = self.transform(image)
            image = np.array(image)  # 将 PIL 图像转换回 NumPy 数组

        return (image, label)

    def __len__(self):
        return len(self.targets)

# 根据指定的触发器类型（trigger_type）和模式（mode），生成中毒数据集或测试数据集
class PoisonedDataset(Dataset):
    def __init__(self, dataset, trigger_label, trigger_type, trigger_path, poisoned_mode, portion=0.1, mode="train", poisoned_idx=[], noisy_idx=[],args=None):
        self.class_num = args.num_class
        self.args=args
        self.mode = mode
        self.poisoned_mode = poisoned_mode
        self.data = dataset.data
        self.targets = np.array(dataset.targets).astype(np.int64)
        self.poisoned_idx = poisoned_idx
        self.noisy_idx = noisy_idx

        if self.mode == 'train':
            print("## generate training data")
            if len(self.poisoned_idx) == 0 and len(self.noisy_idx) == 0:
                if trigger_type == 'badnet' or trigger_type == 'blended':   #  随机选择 portion 比例的样本作为中毒样本。
                    # np.random.permutation 是 NumPy 库中用于生成随机排列的函数
                    self.poisoned_idx = np.random.permutation(len(self.data))[0: int(len(self.data) * portion)] # 随机生成中毒列表
                    if trigger_type =='badnet':
                        self.args.trigger_path = './trigger/cifar10/cifar_1.png'
                    else:
                        self.args.trigger_path = './trigger/cifar10/hello_kitty.png'
                elif trigger_type == 'SIG' or trigger_type == 'CL':   #  从目标标签为 trigger_label 的样本中随机选择从目标标签为 trigger_label 的样本中随机选择portion 比例的样本作为中毒样本。
                    # index_target = np.where(np.array(self.targets) == trigger_label)[0]  # 干净标签攻击
                    # np.random.shuffle(index_target)  # 打乱
                    # self.poisoned_idx = index_target[0: int(len(index_target) * portion)]  #随机选一部分作为中毒的,不影响啊？

                    self.poisoned_idx = np.random.permutation(len(self.data))[0: int(len(self.data) * portion)]

                elif trigger_type == 'WaNet':
                    index = np.random.permutation(len(self.data))
                    self.noisy_ratio = 0.2
                    self.poisoned_idx = index[: int(len(self.data) * portion)]
                    self.noisy_idx = index[int(len(self.data) * portion): int(len(self.data) * (portion + self.noisy_ratio))]
                elif trigger_type == 'Dynamic':
                    index = np.random.permutation(len(self.data))
                    self.noisy_ratio = 0.1
                    self.poisoned_idx = index[: int(len(self.data) * portion)]
                    self.noisy_idx = index[int(len(self.data) * portion): int(len(self.data) * (portion + self.noisy_ratio))]
            self.data, self.targets = self.add_trigger(trigger_label, trigger_type, trigger_path)

        elif self.mode == 'Acc test':  # 干净样本测试。
            print("## generate Acc testing data")
            self.poisoned_idx = np.array([])

        elif self.mode == 'ASR test':  # 中毒样本上测试
            print("## generate ASR testing data")
            self.index_not_target = np.where(np.array(dataset.targets) != trigger_label)[0]  # 找出数据集中目标标签不等于 trigger_label 的索引
            # 通过索引进行了样本和标签提取
            self.data = self.data[self.index_not_target]
            self.targets = self.targets[self.index_not_target]
            # 生成了索引
            self.poisoned_idx = range(0, len(self.data)) #np.random.permutation(len(self.data))[0: int(len(self.data) * portion)]
            # 添加触发器
            self.data, self.targets = self.add_trigger(trigger_label, trigger_type, trigger_path)

        print("Injecting Over: %d Bad Imgs, %d Clean Imgs (%.4f)" % (len(self.poisoned_idx), len(self.data) - len(self.poisoned_idx), portion))
        # self.width, self.height, self.channels = dataset.data.shape[1:]
        # self.train_transform = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize((self.height, self.width)),
        #     transforms.RandomCrop((self.height, self.width), padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        # ])
        # self.test_transform = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize((self.height, self.width)),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        # ])

    def __getitem__(self, item):
        return None

    def __len__(self):
        return len(self.data)

    def  add_trigger(self, target_label, trigger_type, trigger_path):
        new_data = copy.deepcopy(self.data) #(50000, 32, 32, 3)
        new_targets = copy.deepcopy(self.targets)
        width, height, _ = new_data.shape[1:]

        if self.poisoned_mode == 'all2one':
            new_targets[self.poisoned_idx] = target_label
        elif self.poisoned_mode == 'all2all':
            #new_targets[self.poisoned_idx] = (new_targets[self.poisoned_idx] + 1) % self.class_num
            new_targets[self.poisoned_idx] = np.random.choice(self.class_num, size=len(self.poisoned_idx))
        new_data = self._add_trigger(new_data, trigger_type, trigger_path)

        return new_data, new_targets

    def _add_trigger(self, data, trigger_type, trigger_path):
        width, height, _ = data.shape[1:]
        # perm = self.poisoned_idx
        if trigger_type == 'badnet':
            # data[perm, width-3:width-1, height-3:height-1, :] = 255
            with open(trigger_path, "rb") as f:
                trigger_ptn = Image.open(f).convert("RGB")
            self.trigger_ptn = trigger_ptn.resize((height, width))
            self.trigger_ptn = np.array(trigger_ptn)
            self.trigger_loc = np.nonzero(self.trigger_ptn)

            for i in range(len(self.trigger_loc[0])):
                data[self.poisoned_idx, self.trigger_loc[0][i], self.trigger_loc[1][i], self.trigger_loc[2][i]] = self.trigger_ptn[self.trigger_loc[0][i], self.trigger_loc[1][i], self.trigger_loc[2][i]]
            return data
        elif trigger_type == 'blended':
            with open(trigger_path, "rb") as f:
                trigger_ptn = Image.open(f).convert("RGB")
            alpha = 0.25
            print('Blended alpha: ' + str(alpha))
            self.trigger_ptn = trigger_ptn.resize((height, width))
            for index in self.poisoned_idx:
                img = Image.fromarray(data[index])
                data[index] = np.array(Image.blend(img, self.trigger_ptn, alpha))
            return data

        elif trigger_type == 'SIG':
            self.trigger_ptn = Image.fromarray(create_SIG(data[0]))
            alpha = 0.1

            for index in self.poisoned_idx:
                img = Image.fromarray(data[index])
                data[index] = np.array(Image.blend(img, self.trigger_ptn, alpha))
            return data

        elif trigger_type == 'CL':

            data_train = np.load(f'./dataset/CL_Attack/{self.args.model_name}_{self.args.dataset}.npy')  # 用他的触发器图像确定了，像素值位置有误
            data_train = np.transpose(data_train, (0, 2, 3, 1))   #
            data[self.poisoned_idx] = data_train[self.poisoned_idx]  # 中毒的图片替换掉干净图片

            pattern = torch.zeros((1, width, width), dtype=torch.uint8)
            pattern[0, -3:, -3:] = 255
            weight = torch.zeros((1, width, width), dtype=torch.float32)
            weight[0, -3:, -3:] = 1.0
            # plt.imshow(pattern.squeeze())
            # plt.title("Pattern")
            # plt.show()

            res = weight * pattern
            weight = 1.0 - weight
            for index in self.poisoned_idx:
                # plt.imshow(data[index])
                # plt.title("Original Image")
                # plt.show()
                img = Image.fromarray(data[index])
                img = torchvision.transforms.functional.pil_to_tensor(img)
                data[index] = (weight * img + res).type(torch.uint8).permute(1, 2, 0)
                # plt.imshow(data[index])
                # plt.title("Original Image")
                # plt.show()
                #
                # sys.exit()

            return data


            # with open(trigger_path, "rb") as f:
            #     trigger_ptn = Image.open(f).convert("RGB")
            # self.trigger_ptn = np.array(trigger_ptn)
            # self.trigger_loc = np.nonzero(self.trigger_ptn)

            # if self.mode == 'train':
            #     CL_cifar10_train = np.load('./dataset/CL-cifar10/inf_32.npy')  # 用他的触发器图像确定了，像素值位置有误
            #     data[self.poisoned_idx] = CL_cifar10_train[self.poisoned_idx]  # 3000个中毒的
            #     # original_image = data[self.poisoned_idx[0]]
            # for i in range(len(self.trigger_loc[0])):
            #     data[self.poisoned_idx, self.trigger_loc[0][i], self.trigger_loc[1][i], self.trigger_loc[2][i]] = self.trigger_ptn[self.trigger_loc[0][i], self.trigger_loc[1][i], self.trigger_loc[2][i]]

            # # 选择一个中毒样本用于可视化
            # poisoned_sample_idx = self.poisoned_idx[0]  # 选择第一个中毒样本
            # poisoned_image = data[poisoned_sample_idx]  # 添加触发器后的图像
            #
            # # 显示原始图像和添加触发器后的图像
            # plt.figure(figsize=(12, 6))
            #
            # # 显示原始图像
            # plt.subplot(1, 2, 1)
            # plt.title("Original Image")
            # plt.imshow(original_image)
            # plt.axis('off')
            #
            # # 显示添加触发器后的图像
            # plt.subplot(1, 2, 2)
            # plt.title("Poisoned Image")
            # plt.imshow(poisoned_image)
            # plt.axis('off')
            #
            # plt.show()
            # print(original_image)
            # print(poisoned_image)
            return data

        elif trigger_type == 'WaNet':
            # Prepare grid
            """
            增大攻击强度：
                增大 s 的值（例如从 0.5 增大到 1.0）。
                增大 grid_rescale 的值（例如从 1 增大到 1.5）。
                增大 k 的值（例如从 4 增大到 8）。
                增大 ins 的幅度（例如将 torch.rand(1, 2, k, k) * 2 - 1 改为 torch.rand(1, 2, k, k) * 3 - 1）。
            """

            s = 1  #这是控制后门攻击强度的主要参数。增大 s 的值会增加网格变形的幅度，从而使后门触发器更加明显
            k = 4  # 增大 k 的值会生成更细粒度的网格变形，可能使后门触发器更加复杂和难以检测。
            grid_rescale = 1.1  # 增大 grid_rescale 的值会进一步放大网格变形的效果，从而使后门触发器更加明显。
            ins = torch.rand(1, 2, k, k) * 2 - 1
            ins = ins / torch.mean(torch.abs(ins))    # 增大 ins 的幅度会增加噪声的影响，从而使后门触发器更加复杂和难以检测。
            noise_grid = (
                torch.nn.functional.interpolate(ins, size=width, mode="bicubic", align_corners=True)
                .permute(0, 2, 3, 1)
            )   # 将初始噪声插值到与输入图像相同尺寸
            array1d = torch.linspace(-1, 1, steps=width)
            x, y = torch.meshgrid(array1d, array1d)
            identity_grid = torch.stack((y, x), 2)[None, ...]   #生成标准的归一化网格坐标系（范围为[-1,1]
            grid_temps = (identity_grid + s * noise_grid / width) * grid_rescale
            grid_temps = torch.clamp(grid_temps, -1, 1)

            ins = torch.rand(1, width, width, 2) * 2 - 1   # 为噪声样本添加二次随机扰动
            grid_temps2 = grid_temps + ins / width
            grid_temps2 = torch.clamp(grid_temps2, -1, 1)

            original_img = data[self.poisoned_idx[0]].copy()
            data = torch.from_numpy(data).permute(0, 3, 1, 2).to(torch.float32)    #

            # 显示原始图像
            #original_img = data[self.poisoned_idx[0]].permute(1, 2, 0).clamp(0, 255).byte().numpy()

            data[self.poisoned_idx] = F.grid_sample(data[self.poisoned_idx], grid_temps.repeat(len(self.poisoned_idx), 1, 1, 1), align_corners=True)  #对被污染的图像索引应用触发器网格
            data[self.noisy_idx] = F.grid_sample(data[self.noisy_idx], grid_temps2.repeat(len(self.noisy_idx), 1, 1, 1), align_corners=True)  # 对噪声图像索引应用噪声网格noisy_grids
            data = data.permute(0, 2, 3, 1).to(torch.uint8).numpy()  # 从(N, C, H, W)变为(N, H, W, C)，这是PyTorch中图像张量的标准格式。
            # # 显示中毒图像
            # poisoned_img = data[self.poisoned_idx[0]].copy()
            #
            # # 显示中毒前后的图像
            # plt.figure()
            #
            # # 显示原始图像
            # plt.subplot(1, 2, 1)
            # plt.title("Original")
            # plt.imshow(original_img)
            # plt.axis('off')
            #
            # # 显示中毒图像
            # plt.subplot(1, 2, 2)
            # plt.title("Poisoned")
            # plt.imshow(poisoned_img)
            # plt.axis('off')
            #
            # # 调整布局并显示图形
            # plt.tight_layout()
            # plt.show()

            return data

        elif trigger_type == 'Dynamic':
            if self.mode == 'train':
                replace_data_bd = np.load("./dataset/Dynamic/cifar10-inject1.0-target0-dynamic_train.npy", allow_pickle=True)
                for idx in self.poisoned_idx:
                    data[idx] = np.clip(replace_data_bd[idx]*255, 0, 255).astype(np.uint8)
                replace_data_cross = np.load("./dataset/Dynamic/cifar10-inject1.0-target0-dynamic_cross.npy", allow_pickle=True)
                for idx in self.noisy_idx:
                    data[idx] = np.clip(replace_data_cross[idx]*255, 0, 255).astype(np.uint8)
            else:
                replace_data_bd = np.load("./dataset/Dynamic/cifar10-inject1.0-target0-dynamic_test.npy", allow_pickle=True)[self.index_not_target]
                for idx in self.poisoned_idx:
                    data[idx] = np.clip(replace_data_bd[idx]*255, 0, 255).astype(np.uint8)
        return data


def create_SIG(img, delta=20, f=6):  # 生成正弦图像
    pattern = np.zeros_like(img)
    m = img.shape[1]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pattern[i, j] = delta * np.sin(2 * np.pi *j * f / m)

    return pattern

# def create_grids(img, num_cross, k=4, s=0.5, grid_rescale=1):
#     height = img.shape[0]
#     ins = torch.rand(1, 2, k, k) * 2 - 1
#     ins = ins / torch.mean(torch.abs(ins))
#     noise_grid = (
#         F.upsample(ins, size=height, mode="bicubic", align_corners=True)
#             .permute(0, 2, 3, 1)
#     )
#     array1d = torch.linspace(-1, 1, steps=height)  # (32)
#     x, y = torch.meshgrid(array1d, array1d)
#
#     identity_grid = torch.stack((y, x), 2)[None, ...] # (1, 32, 32, 2)
#     grid_temps = (identity_grid + s * noise_grid / height) * grid_rescale
#     bd_temps = torch.clamp(grid_temps, -1, 1)
#
#     # ins = torch.rand(num_cross, height, height, 2) * 2 - 1
#     # grid_temps2 = grid_temps.repeat(num_cross, 1, 1, 1) + ins / height
#     # noisy_temps = torch.clamp(grid_temps2, -1, 1)
#
#     return bd_temps#, noisy_temps


class data_dataset(Dataset):
    def __init__(self, dataset, mode, transform, no_transform=None, pred=[], probability=[]):
        data = dataset.data
        targets = dataset.targets
        self.poisoned_idx = dataset.poisoned_idx
        poisoned_vector = np.zeros(len(data)) # 标记哪些样本是中毒样本。
        for i in self.poisoned_idx:
            poisoned_vector[i] = 1

        self.mode = mode
        self.transform = transform
        self.no_transform = no_transform

        if self.mode == 'all' or self.mode == 'train_BD' or self.mode == 'train_BD1':  # 使用所有数据。
            self.data, self.targets = data, targets
        else:
            if self.mode == 'labeled':
                pred_idx = pred.nonzero()[0]  # pred.nonzero()：找到 pred 数组中非零元素的索引
                self.probability = [probability[i] for i in pred_idx]  # 根据 pred_idx 索引数组，从 probability 中提取可信样本的置信度
            elif self.mode == 'unlabeled':   # 选择标记为可疑样本的数据。
                pred_idx = (1 - pred).nonzero()[0]
                self.probability = [1-probability[i] for i in pred_idx]

            self.data = np.array(data)[pred_idx, :, :, :]  # 根据 pred_idx 索引数组，从原始数据集中选择对应的样本
            self.targets = [targets[i] for i in pred_idx]
            self.poisoned_vector = poisoned_vector[pred_idx]

    def __getitem__(self, item):
        if self.mode == 'all':
            img, target = self.data[item], self.targets[item]
            img = self.transform(img)
            return img, target, item
        elif self.mode == 'labeled':  #返回两个增强后的图像、一个未增强的图像、标签和中毒标记。
            img, target, poisoned = self.data[item], self.targets[item], self.poisoned_vector[item]
            img1 = self.transform(img)
            img2 = self.transform(img)
            img3 = self.no_transform(img)
            return img1, img2, img3, target, poisoned
        elif self.mode == 'unlabeled': #返回两个增强后的图像、一个未增强的图像、标签和中毒标记。
            img, target, poisoned = self.data[item], self.targets[item], self.poisoned_vector[item]
            img1 = self.transform(img)
            img2 = self.transform(img)
            img3 = self.no_transform(img)
            return img1, img2, img3, target, poisoned
        elif self.mode == 'train_BD': #返回两个增强后的图像
            img, target = self.data[item], self.targets[item]
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, target, 1
        elif self.mode == 'train_BD1': # #返回两个增强后的图像、一个未增强的图像、标签和中毒标记。
            img, target = self.data[item], self.targets[item]
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2, img, target, 1

    def __len__(self):
        return self.data.shape[0]

class data_dataloader():
    def __init__(self, args, batch_size, num_workers, data_path, trigger_label, poisoned_mode, posioned_portion, poisoned_idx, noisy_idx):
        self.args = args
        self.batch_size = batch_size
        self.num_workers = num_workers
        #  Load Data
        train_data, test_data = load_init_data(args,download=True, dataset_path=data_path)
        # 50000 生成中毒数据集和干净数据集，中毒索引
        self.train_data = PoisonedDataset(train_data, trigger_label, args.trigger_type, args.trigger_path, poisoned_mode=poisoned_mode, portion=posioned_portion, mode="train", poisoned_idx=poisoned_idx, noisy_idx=noisy_idx,args=args)
        # 10000 一个干净数据集，一个中毒数据集
        self.test_data_CL = PoisonedDataset(test_data, trigger_label, args.trigger_type, args.trigger_path, poisoned_mode=poisoned_mode, portion=0, mode="Acc test",args=args)
        self.test_data_BD = PoisonedDataset(test_data, trigger_label, args.trigger_type, args.trigger_path, poisoned_mode=poisoned_mode, portion=1, mode="ASR test",args=args)
        self.resize = 32
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.RandomCrop((32, 32), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        self.gtsrb_train_transform=transforms.Compose([
            transforms.ToPILImage(),  # 转换为 PIL 图像
            # transforms.Resize((32, 32)),  # 调整大小到 48x48
            transforms.RandomCrop((32, 32), padding=4),  # 随机裁剪
            #transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 随机调整颜色
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize([0.333, 0.333, 0.333], [0.307, 0.307, 0.307])  # 归一化
        ])


        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])

        self.gtsrb_test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.resize, self.resize)),  # 调整大小到 48x48
            transforms.ToTensor(),
            transforms.Normalize([0.333, 0.333, 0.333], [0.307, 0.307, 0.307])  # 归一化
        ])

        self.transform_noaugmentation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        self.gtsrb_transform_noaugmentation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.resize, self.resize)),  # 调整大小到 48x48
            transforms.ToTensor(),
            transforms.Normalize([0.333, 0.333, 0.333], [0.307, 0.307, 0.307])  # 归一化
        ])

        self.transform_WaNet = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.RandomCrop((32, 32), padding=4),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])

        self.gtsrb_transform_WaNet = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.resize, self.resize)),  # 调整大小到 48x48
            #transforms.RandomRotation(10),  # 随机旋转，保留标志的方向性
            transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0)),  # 随机裁剪并调整大小
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 随机调整颜色
            transforms.ToTensor(),
            transforms.Normalize([0.333, 0.333, 0.333], [0.307, 0.307, 0.307])  # 归一化
        ])

    def run(self, mode, pred=[], prob=[], batchsize=128):
        if mode == 'warmup':
            if self.args.dataset =='gtsrb':
                all_dataset = data_dataset(dataset=self.train_data, mode="all", transform=self.gtsrb_transform_noaugmentation)
            else:
                all_dataset = data_dataset(dataset=self.train_data, mode="all", transform=self.transform_noaugmentation)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=128,
                shuffle=True,
                num_workers=self.num_workers)
            return trainloader

        # elif mode == 'train_net1':
        #     labeled_dataset = data_dataset(dataset=self.train_data, mode="labeled", transform=self.train_transform, no_transform=self.transform_noaugmentation, pred=pred, probability=prob)
        #     labeled_trainloader = DataLoader(
        #         dataset=labeled_dataset,
        #         batch_size=self.batch_size,
        #         shuffle=True,
        #         num_workers=self.num_workers)
        #
        #     unlabeled_dataset = data_dataset(dataset=self.train_data, mode="unlabeled", transform=self.train_transform, no_transform=self.transform_noaugmentation, pred=pred, probability=prob)
        #     unlabeled_trainloader = DataLoader(
        #         dataset=unlabeled_dataset,
        #         batch_size=batchsize,
        #         shuffle=True,
        #         num_workers=self.num_workers)
        #     return labeled_trainloader, unlabeled_trainloader

        elif mode == 'train_net2':
            if self.args.dataset == 'gtsrb':
                labeled_dataset = data_dataset(dataset=self.train_data, mode="labeled",
                                               transform=self.gtsrb_transform_noaugmentation,
                                               no_transform=self.gtsrb_transform_noaugmentation, pred=pred, probability=prob)

            else:
                labeled_dataset = data_dataset(dataset=self.train_data, mode="labeled", transform=self.transform_noaugmentation, no_transform=self.transform_noaugmentation, pred=pred, probability=prob)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=128,
                shuffle=True,
                num_workers=self.num_workers)

            unlabeled_dataset = data_dataset(dataset=self.train_data, mode="unlabeled", transform=self.transform_noaugmentation, no_transform=self.transform_noaugmentation, pred=pred, probability=prob)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=128,
                shuffle=True,
                num_workers=self.num_workers)
            return labeled_trainloader, unlabeled_trainloader


        # elif mode == 'test_net1':
        #     test_dataset_CL = data_dataset(dataset=self.test_data_CL, mode="all", transform=self.test_transform)
        #     test_loader_CL = DataLoader(
        #         dataset=test_dataset_CL,
        #         batch_size=self.batch_size,
        #         shuffle=False,
        #         num_workers=self.num_workers)
        #     test_dataset_BD = data_dataset(dataset=self.test_data_BD, mode="all", transform=self.test_transform)
        #     test_loader_BD = DataLoader(
        #         dataset=test_dataset_BD,
        #         batch_size=self.batch_size,
        #         shuffle=False,
        #         num_workers=self.num_workers)
        #     return test_loader_CL, test_loader_BD
        #
        # elif mode == 'test_net2':
        #
        #     test_dataset_CL = data_dataset(dataset=self.test_data_CL, mode="all", transform=self.transform_noaugmentation)
        #     test_loader_CL = DataLoader(
        #         dataset=test_dataset_CL,
        #         batch_size=128,
        #         shuffle=False,
        #         num_workers=self.num_workers)
        #     test_dataset_BD = data_dataset(dataset=self.test_data_BD, mode="all", transform=self.transform_noaugmentation)
        #     test_loader_BD = DataLoader(
        #         dataset=test_dataset_BD,
        #         batch_size=128,
        #         shuffle=False,
        #         num_workers=self.num_workers)
        #     return test_loader_CL, test_loader_BD

        elif mode == 'eval_train_net2':
            if self.args.dataset == 'gtsrb':
                eval_dataset1 = data_dataset(dataset=self.train_data, mode='all',
                                             transform=self.gtsrb_transform_noaugmentation)
            else:
                eval_dataset1 = data_dataset(dataset=self.train_data,  mode='all',transform=self.transform_noaugmentation)
            eval_loader1 = DataLoader(
                dataset=eval_dataset1,
                batch_size=128,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader1, self.train_data.poisoned_idx, self.train_data.noisy_idx

        # elif mode == 'train_BD':
        #     if self.args.dataset == 'gtsrb':
        #         all_dataset = data_dataset(dataset=self.train_data, transform=self.gtsrb_train_transform, mode="train_BD")
        #     else:
        #         all_dataset = data_dataset(dataset=self.train_data, transform=self.train_transform, mode="train_BD")
        #     trainloader = DataLoader(
        #         dataset=all_dataset,
        #         batch_size=self.batch_size,
        #         shuffle=True,
        #         num_workers=self.num_workers)
        #     return trainloader

