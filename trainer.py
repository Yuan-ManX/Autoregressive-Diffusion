from typing import List
import math
from pathlib import Path
from accelerate import Accelerator
from ema_pytorch import EMA
from PIL import Image

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Module, ModuleList
from torch.utils.data import Dataset
from torchvision.utils import save_image
import torchvision.transforms as T


def exists(v):
    """
    检查变量是否存在（不为 None）。

    参数:
        v (Any): 任意变量。

    返回:
        bool: 如果 v 不为 None，则返回 True，否则返回 False。
    """
    return v is not None


def default(v, d):
    """
    如果变量存在（不为 None），则返回变量本身；否则返回默认值。

    参数:
        v (Any): 任意变量。
        d (Any): 默认值。

    返回:
        Any: 如果 v 存在，则返回 v；否则返回 d。
    """
    return v if exists(v) else d


def divisible_by(num, den):
    """
    检查一个数是否能被另一个数整除。

    参数:
        num (int): 被除数。
        den (int): 除数。

    返回:
        bool: 如果 num 能被 den 整除，则返回 True；否则返回 False。
    """
    return (num % den) == 0


def cycle(dl):
    """
    创建一个无限循环的数据加载器。

    参数:
        dl (DataLoader): 原始数据加载器。

    返回:
        Iterator[Any]: 一个无限循环的迭代器。
    """
    while True:
        for batch in dl:
            yield batch


class ImageDataset(Dataset):
    """
    图像数据集类，用于加载和处理图像数据。

    该数据集类支持从指定文件夹加载图像，应用数据增强和转换，并返回处理后的图像张量。
    """
    def __init__(
        self,
        folder: str | Path,
        image_size: int,
        exts: List[str] = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        """
        初始化图像数据集。

        参数:
            folder (str | Path): 图像文件夹路径。
            image_size (int): 图像的尺寸。
            exts (List[str], 可选): 支持的图像扩展名列表，默认为 ['jpg', 'jpeg', 'png', 'tiff']。
            augment_horizontal_flip (bool, 可选): 是否进行水平翻转数据增强，默认为 False。
            convert_image_to (str, 可选): 是否将图像转换为特定模式，例如 'RGB'，默认为 None。
        """
        super().__init__()
        if isinstance(folder, str):
            # 如果文件夹路径是字符串，则转换为 Path 对象
            folder = Path(folder)

        assert folder.is_dir()

        self.folder = folder
        self.image_size = image_size
        # 使用列表推导式生成所有图像文件的路径
        self.paths = [p for ext in exts for p in folder.glob(f'**/*.{ext}')]

        # 定义一个函数，用于将图像转换为指定的模式（如果需要）
        def convert_image_to_fn(img_type, image):
            if image.mode == img_type:
                return image

            return image.convert(img_type)

        # 如果需要转换图像模式，则创建一个部分函数；否则，使用恒等函数
        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        # 定义图像转换流程：
        # 1. 如果需要，转换图像模式。
        # 2. 调整图像大小到指定的尺寸。
        # 3. 如果需要，进行水平翻转。
        # 4. 中心裁剪图像到指定尺寸。
        # 5. 将图像转换为张量。
        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        """
        返回数据集的大小。

        Returns:
            int: 数据集的大小，即图像文件的数量。
        """
        return len(self.paths)

    def __getitem__(self, index):
        """
        获取指定索引的图像数据。

        参数:
            index (int): 图像的索引。

        返回:
            torch.Tensor: 处理后的图像张量。
        """
        # 获取指定索引的图像路径
        path = self.paths[index]
        img = Image.open(path)
        # 应用转换并返回处理后的图像张量
        return self.transform(img)


class ImageTrainer(Module):
    """
    图像训练器类，用于训练图像生成模型。

    该训练器集成了模型、数据加载器、优化器、加速器、指数移动平均（EMA）模型等功能，
    并提供了保存检查点和生成结果的功能。
    """
    def __init__(
        self,
        model,
        *,
        dataset: Dataset,
        num_train_steps = 70_000,
        learning_rate = 3e-4,
        batch_size = 16,
        checkpoints_folder: str = './checkpoints',
        results_folder: str = './results',
        save_results_every: int = 100,
        checkpoint_every: int = 1000,
        num_samples: int = 16,
        adam_kwargs: dict = dict(),
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict()
    ):
        """
        初始化图像训练器。

        参数:
            model (Module): 要训练的模型。
            dataset (Dataset): 训练数据集。
            num_train_steps (int, 可选): 训练的总步数，默认为70,000步。
            learning_rate (float, 可选): 初始学习率，默认为3e-4。
            batch_size (int, 可选): 批次大小，默认为16。
            checkpoints_folder (str, 可选): 检查点保存文件夹路径，默认为 './checkpoints'。
            results_folder (str, 可选): 结果保存文件夹路径，默认为 './results'。
            save_results_every (int, 可选): 每隔多少步保存一次结果，默认为100步。
            checkpoint_every (int, 可选): 每隔多少步保存一次检查点，默认为1000步。
            num_samples (int, 可选): 每次保存结果时生成的样本数量，默认为16。
            adam_kwargs (Dict[str, Any], 可选): Adam 优化器的关键字参数，默认为空字典。
            accelerate_kwargs (Dict[str, Any], 可选): 加速器的关键字参数，默认为空字典。
            ema_kwargs (Dict[str, Any], 可选): EMA 模型的关键字参数，默认为空字典。
        """
        super().__init__()
        # 初始化加速器
        self.accelerator = Accelerator(**accelerate_kwargs)
        # 将模型移动到加速器设备
        self.model = model

        # 如果是主进程，则初始化 EMA 模型
        if self.is_main:
            self.ema_model = EMA(
                self.model,
                forward_method_names = ('sample',),
                **ema_kwargs
            )

            self.ema_model.to(self.accelerator.device)

        # 初始化优化器
        self.optimizer = Adam(model.parameters(), lr = learning_rate, **adam_kwargs)
        # 创建数据加载器，并设置批量大小、打乱顺序和丢弃最后一批数据
        self.dl = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True)

        # 准备模型、优化器和数据加载器以进行分布式训练
        self.model, self.optimizer, self.dl = self.accelerator.prepare(self.model, self.optimizer, self.dl)

        # 保存训练总步数
        self.num_train_steps = num_train_steps

        # 设置检查点保存文件夹路径
        self.checkpoints_folder = Path(checkpoints_folder)
        # 设置结果保存文件夹路径
        self.results_folder = Path(results_folder)

        # 创建检查点和结果文件夹（如果不存在）
        self.checkpoints_folder.mkdir(exist_ok = True, parents = True)
        self.results_folder.mkdir(exist_ok = True, parents = True)

        # 设置保存检查点的频率
        self.checkpoint_every = checkpoint_every
        # 设置保存结果的频率
        self.save_results_every = save_results_every

        # 计算保存结果的行数，确保为正方形网格
        self.num_sample_rows = int(math.sqrt(num_samples))
        assert (self.num_sample_rows ** 2) == num_samples, f'{num_samples} must be a square'
        # 保存样本数量
        self.num_samples = num_samples

        assert self.checkpoints_folder.is_dir()
        assert self.results_folder.is_dir()

    @property
    def is_main(self):
        """
        检查当前进程是否为主要的训练进程。

        返回:
            bool: 如果是主要进程，则返回 True；否则返回 False。
        """
        return self.accelerator.is_main_process

    def save(self, path):
        """
        保存模型的检查点。

        参数:
            path (str): 检查点文件的路径。
        """
        if not self.is_main:
            return

        # 收集需要保存的状态字典
        save_package = dict(
            model = self.accelerator.unwrap_model(self.model).state_dict(), # 获取模型的权重
            ema_model = self.ema_model.state_dict(), # 获取 EMA 模型的权重
            optimizer = self.accelerator.unwrap_model(self.optimizer).state_dict(), # 获取优化器的状态
        )

        # 保存状态字典到指定路径
        torch.save(save_package, str(self.checkpoints_folder / path))

    def forward(self):
        """
        执行训练过程。
        """
        # 创建一个无限循环的数据加载器
        dl = cycle(self.dl)

        for ind in range(self.num_train_steps):
            # 当前步数
            step = ind + 1
            # 设置模型为训练模式
            self.model.train()

            # 获取下一个批次的数据
            data = next(dl)
            # 前向传播，计算损失
            loss = self.model(data)

            self.accelerator.print(f'[{step}] loss: {loss.item():.3f}')
            # 反向传播，计算梯度
            self.accelerator.backward(loss)

            # 更新优化器参数
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.is_main:
                # 更新 EMA 模型
                self.ema_model.update()

            self.accelerator.wait_for_everyone()

            if self.is_main:
                if divisible_by(step, self.save_results_every):
                    # 如果需要保存结果，则生成样本并保存图像
                    with torch.no_grad():
                        sampled = self.ema_model.sample(batch_size = self.num_samples)
                    # 将样本值裁剪到 [0, 1] 范围
                    sampled.clamp_(0., 1.)
                    save_image(sampled, str(self.results_folder / f'results.{step}.png'), nrow = self.num_sample_rows)

                if divisible_by(step, self.checkpoint_every):
                    self.save(f'checkpoint.{step}.pt')

            # 等待所有进程完成
            self.accelerator.wait_for_everyone()


        print('training complete')
