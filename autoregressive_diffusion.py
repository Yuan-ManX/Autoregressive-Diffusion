import math
from math import sqrt
from typing import Literal
from functools import partial
from tqdm import tqdm

import torch
from torch import nn, pi
from torch.special import expm1
import torch.nn.functional as F
from torch.nn import Module, ModuleList
import einx
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

from x_transformers import Decoder


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


def log(t, eps = 1e-20):
    """
    对输入张量进行安全的对数运算。

    参数:
        t (torch.Tensor): 输入张量。
        eps (float, 可选): 防止对数运算中出现负无穷大的最小值，默认为1e-20。

    返回:
        torch.Tensor: 对数运算后的张量。
    """
    return torch.log(t.clamp(min = eps))


def safe_div(num, den, eps = 1e-5):
    """
    对输入张量进行安全的除法运算，避免除以零。

    参数:
        num (torch.Tensor): 被除数张量。
        den (torch.Tensor): 除数张量。
        eps (float, 可选): 防止除以零的最小值，默认为1e-5。

    返回:
        torch.Tensor: 除法运算后的张量。
    """
    return num / den.clamp(min = eps)


def right_pad_dims_to(x, t):
    """
    对张量 t 进行右侧填充，使其维度与 x 一致。

    参数:
        x (torch.Tensor): 目标张量，其维度将作为参考。
        t (torch.Tensor): 需要填充的张量。

    返回:
        torch.Tensor: 填充后的张量，其维度与 x 一致。
    """
    padding_dims = x.ndim - t.ndim

    if padding_dims <= 0:
        return t

    return t.view(*t.shape, *((1,) * padding_dims))


def pack_one(t, pattern):
    """
    对张量进行打包，并返回一个用于解包的函数。

    参数:
        t (torch.Tensor): 需要打包的张量。
        pattern (Tuple[int, ...]): 打包模式，指定每个维度如何分割。

    返回:
        Tuple[torch.Tensor, Callable]: 打包后的张量和一个用于解包的函数。
    """
    packed, ps = pack([t], pattern)

    def unpack_one(to_unpack, unpack_pattern = None):
        """
        对打包后的张量进行解包。

        参数:
            to_unpack (torch.Tensor): 需要解包的张量。
            unpack_pattern (Tuple[int, ...], 可选): 解包模式，默认为 None。

        返回:
            torch.Tensor: 解包后的张量。
        """
        unpacked, = unpack(to_unpack, ps, default(unpack_pattern, pattern))
        return unpacked

    return packed, unpack_one


class AdaptiveLayerNorm(Module):
    """
    自适应层归一化（Adaptive Layer Normalization, AdaptiveLayerNorm）模块。

    该模块结合了层归一化（LayerNorm）和一个线性层，通过条件输入动态调整归一化参数。
    """
    def __init__(
        self,
        dim,
        dim_condition = None
    ):
        """
        初始化 AdaptiveLayerNorm 模块。

        参数:
            dim (int): 特征维度。
            dim_condition (int, 可选): 条件特征的维度，默认为 None。如果为 None，则使用 dim 作为条件维度。
        """
        super().__init__()
        # 如果未指定条件维度，则使用特征维度
        dim_condition = default(dim_condition, dim)

        # 初始化 LayerNorm，不使用可学习的仿射参数
        self.ln = nn.LayerNorm(dim, elementwise_affine = False)
        # 初始化线性层，将条件特征映射到 gamma 参数
        self.to_gamma = nn.Linear(dim_condition, dim, bias = False)
        # 将线性层的权重初始化为零
        nn.init.zeros_(self.to_gamma.weight)

    def forward(self, x, *, condition):
        """
        前向传播方法。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, dim)。
            condition (torch.Tensor): 条件输入张量，形状为 (batch_size, dim_condition)。

        返回:
            torch.Tensor: 归一化后的张量，形状为 (batch_size, seq_len, dim)。
        """
        # 对输入张量进行层归一化
        normed = self.ln(x)
        # 将条件输入映射到 gamma 参数
        gamma = self.to_gamma(condition)
        # 将归一化后的张量乘以 (gamma + 1)，实现自适应归一化
        return normed * (gamma + 1.)


class LearnedSinusoidalPosEmb(Module):
    """
    学习到的正弦位置嵌入（Learned Sinusoidal Positional Embedding）模块。

    该模块通过学习到的权重生成正弦和余弦位置嵌入，用于编码位置信息。
    """
    def __init__(self, dim):
        """
        初始化 LearnedSinusoidalPosEmb 模块。

        参数:
            dim (int): 嵌入维度，必须能被2整除。
        """
        super().__init__()
        assert divisible_by(dim, 2)
        # 计算一半的维度
        half_dim = dim // 2
        # 初始化学习到的权重参数
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        """
        前向传播方法。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size,)。

        返回:
            torch.Tensor: 生成的位置嵌入，形状为 (batch_size, dim)。
        """
        # 重塑张量形状为 (batch_size, 1)
        x = rearrange(x, 'b -> b 1')
        # 计算频率张量
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * pi
        # 生成正弦和余弦嵌入
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        # 将输入张量与嵌入连接起来
        fouriered = torch.cat((x, fouriered), dim = -1)
        # 返回最终的位置嵌入
        return fouriered


class MLP(Module):
    """
    多层感知机（MLP）模块，用于处理条件输入和时间嵌入。

    该 MLP 模块包含时间嵌入模块、自适应层归一化、线性层、激活函数和 dropout 层。
    """
    def __init__(
        self,
        dim_cond,
        dim_input,
        depth = 3,
        width = 1024,
        dropout = 0.
    ):
        """
        初始化 MLP 模块。

        参数:
            dim_cond (int): 条件特征的维度。
            dim_input (int): 输入特征的维度。
            depth (int, 可选): MLP 的深度，默认为3。
            width (int, 可选): MLP 的宽度，默认为1024。
            dropout (float, 可选): Dropout 概率，默认为0。
        """
        super().__init__()
        # 初始化一个空的模块列表，用于存储 MLP 的层
        layers = ModuleList([])

        # 时间嵌入模块
        self.to_time_emb = nn.Sequential(
            LearnedSinusoidalPosEmb(dim_cond),  # 使用学习到的正弦位置嵌入生成时间嵌入
            nn.Linear(dim_cond + 1, dim_cond),  # 将时间嵌入与条件输入连接，并映射到条件维度
        )

        for _ in range(depth):
            # 自适应层归一化
            adaptive_layernorm = AdaptiveLayerNorm(
                dim_input,  
                dim_condition = dim_cond
            )

            # 线性层块
            block = nn.Sequential(
                nn.Linear(dim_input, width),  # 线性层，将输入映射到宽度维度
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(width, dim_input)  # 线性层，将宽度维度映射回输入维度
            )

            # 输出 gamma 参数
            # 线性层，将条件维度映射到输入维度
            block_out_gamma = nn.Linear(dim_cond, dim_input, bias = False)
            # 将线性层的权重初始化为零
            nn.init.zeros_(block_out_gamma.weight)

            # 将所有层添加到模块列表中
            layers.append(ModuleList([
                adaptive_layernorm,
                block,
                block_out_gamma
            ]))
        
        # 保存层列表
        self.layers = layers

    def forward(
        self,
        noised,
        *,
        times,
        cond
    ):
        """
        前向传播方法。

        参数:
            noised (torch.Tensor): 输入的噪声数据，形状为 (batch_size, dim_input)。
            times (torch.Tensor): 时间输入，形状为 (batch_size,)。
            cond (torch.Tensor): 条件输入，形状为 (batch_size, dim_cond)。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, dim_input)。
        """
        assert noised.ndim == 2

        # 生成时间嵌入
        time_emb = self.to_time_emb(times)
        # 将时间嵌入与条件输入连接，并通过 SiLU 激活函数
        cond = F.silu(time_emb + cond)

        # 初始化去噪数据为输入的噪声数据
        denoised = noised

        for adaln, block, block_out_gamma in self.layers:
            # 保存残差
            residual = denoised
            # 应用自适应层归一化
            denoised = adaln(denoised, condition = cond)

            # 应用线性层块，并调整输出
            block_out = block(denoised) * (block_out_gamma(cond) + 1.)
            # 应用残差连接
            denoised = block_out + residual

        # 返回去噪后的数据
        return denoised


class ElucidatedDiffusion(Module):
    """
    阐明的扩散模型（Elucidated Diffusion）类。

    该模型实现了扩散过程的正向和逆向过程，通过逐步添加和去除噪声来生成数据。
    """
    def __init__(
        self,
        dim: int,
        net: MLP,
        *,
        num_sample_steps = 32, # number of sampling steps
        sigma_min = 0.002,     # min noise level
        sigma_max = 80,        # max noise level
        sigma_data = 0.5,      # standard deviation of data distribution
        rho = 7,               # controls the sampling schedule
        P_mean = -1.2,         # mean of log-normal distribution from which noise is drawn for training
        P_std = 1.2,           # standard deviation of log-normal distribution from which noise is drawn for training
        S_churn = 80,          # parameters for stochastic sampling - depends on dataset, Table 5 in apper
        S_tmin = 0.05,
        S_tmax = 50,
        S_noise = 1.003,
        clamp_during_sampling = True
    ):
        """
        初始化阐明的扩散模型。

        参数:
            dim (int): 数据的特征维度。
            net (MLP): 神经网络模型，用于预测噪声或数据。
            num_sample_steps (int, 可选): 采样步数，默认为32。
            sigma_min (float, 可选): 最小噪声水平，默认为0.002。
            sigma_max (float, 可选): 最大噪声水平，默认为80。
            sigma_data (float, 可选): 数据分布的标准差，默认为0.5。
            rho (float, 可选): 控制采样计划的参数，默认为7。
            P_mean (float, 可选): 训练时噪声对数正态分布的均值，默认为-1.2。
            P_std (float, 可选): 训练时噪声对数正态分布的标准差，默认为1.2。
            S_churn (float, 可选): 随机采样参数，依赖于数据集，默认为80。
            S_tmin (float, 可选): 随机采样参数，默认为0.05。
            S_tmax (float, 可选): 随机采样参数，默认为50。
            S_noise (float, 可选): 随机采样参数，默认为1.003。
            clamp_during_sampling (bool, 可选): 是否在采样期间对输出进行裁剪，默认为 True。
        """
        super().__init__()
        # 保存神经网络模型
        self.net = net
        # 保存特征维度
        self.dim = dim

        # 参数设置
         # 最小噪声水平
        self.sigma_min = sigma_min
        # 最大噪声水平
        self.sigma_max = sigma_max
        # 数据分布的标准差
        self.sigma_data = sigma_data

        # 控制采样计划的参数
        self.rho = rho

        # 训练时噪声对数正态分布的均值
        self.P_mean = P_mean
        # 训练时噪声对数正态分布的标准差
        self.P_std = P_std

        # 采样步数
        self.num_sample_steps = num_sample_steps 

        # 随机采样参数
        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

        # 是否在采样期间对输出进行裁剪
        self.clamp_during_sampling = clamp_during_sampling

    @property
    def device(self):
        """
        获取当前设备。

        Returns:
            torch.device: 当前设备（CPU 或 GPU）。
        """
        return next(self.net.parameters()).device

    def c_skip(self, sigma):
        """
        计算跳过系数 c_skip。

        参数:
            sigma (Tensor): 当前噪声水平。

        返回:
            Tensor: 计算得到的跳过系数。
        """
        return (self.sigma_data ** 2) / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        """
        计算输出系数 c_out。

        参数:
            sigma (Tensor): 当前噪声水平。

        返回:
            Tensor: 计算得到的输出系数。
        """
        return sigma * self.sigma_data * (self.sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in(self, sigma):
        """
        计算输入系数 c_in。

        参数:
            sigma (Tensor): 当前噪声水平。

        返回:
            Tensor: 计算得到的输入系数。
        """
        return 1 * (sigma ** 2 + self.sigma_data ** 2) ** -0.5

    def c_noise(self, sigma):
        """
        计算噪声系数 c_noise。

        参数:
            sigma (Tensor): 当前噪声水平。

        返回:
            Tensor: 计算得到的噪声系数。
        """
        return log(sigma) * 0.25

    def preconditioned_network_forward(self, noised_seq, sigma, *, cond, clamp = None):
        """
        预处理后的网络前向传播。

        参数:
            noised_seq (Tensor): 加噪后的序列，形状为 (batch_size, dim)。
            sigma (Tensor): 当前噪声水平，形状为 (batch_size,)。
            cond (Tensor): 条件输入，形状为 (batch_size, cond_dim)。
            clamp (bool, 可选): 是否对输出进行裁剪，默认为 None。

        返回:
            Tensor: 预处理后的网络输出，形状为 (batch_size, dim)。
        """
        # 设置是否裁剪，默认为采样期间的裁剪设置
        clamp = default(clamp, self.clamp_during_sampling)

        # 获取批次大小和设备
        batch, device = noised_seq.shape[0], noised_seq.device

        if isinstance(sigma, float):
            # 如果 sigma 是浮点数，则创建全为 sigma 的张量
            sigma = torch.full((batch,), sigma, device = device)

        # 对 sigma 进行填充，使其与 noised_seq 的维度一致
        padded_sigma = right_pad_dims_to(noised_seq, sigma)

        # 计算预处理后的输入，并传递给神经网络
        net_out = self.net(
            self.c_in(padded_sigma) * noised_seq, # 预处理后的输入
            times = self.c_noise(sigma), # 计算噪声系数作为时间输入
            cond = cond # 条件输入
        )

        # 计算最终输出
        out = self.c_skip(padded_sigma) * noised_seq +  self.c_out(padded_sigma) * net_out

        if clamp:
            # 如果需要裁剪，则对输出进行裁剪
            out = out.clamp(-1., 1.)

        # 返回预处理后的输出
        return out

    def sample_schedule(self, num_sample_steps = None):
        """
        生成采样计划中的噪声水平。

        参数:
            num_sample_steps (int, 可选): 采样步数，默认为 None。如果为 None，则使用默认的采样步数。

        返回:
            torch.Tensor: 噪声水平张量，形状为 (num_sample_steps + 1,)。
        """
        # 获取采样步数
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        # 采样步数
        N = num_sample_steps
        inv_rho = 1 / self.rho

        # 生成步数张量
        steps = torch.arange(num_sample_steps, device = self.device, dtype = torch.float32)
        sigmas = (self.sigma_max ** inv_rho + steps / (N - 1) * (self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho

        # 在末尾填充一个0，表示最后一个时间步的噪声水平为0
        sigmas = F.pad(sigmas, (0, 1), value = 0.) # last step is sigma value of 0.
        # 返回噪声水平张量
        return sigmas

    @torch.no_grad()
    def sample(self, cond, num_sample_steps = None, clamp = None):
        """
        使用扩散模型进行采样。

        参数:
            cond (Tensor): 条件输入，形状为 (batch_size, cond_dim)。
            num_sample_steps (int, 可选): 采样步数，默认为 None。如果为 None，则使用默认的采样步数。
            clamp (bool, 可选): 是否对输出进行裁剪，默认为 None。如果为 None，则使用默认的裁剪设置。

        返回:
            torch.Tensor: 生成的样本，形状为 (batch_size, dim)。
        """
        # 设置是否裁剪，默认为采样期间的裁剪设置
        clamp = default(clamp, self.clamp_during_sampling)
        # 获取采样步数
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        # 获取输出张量形状
        shape = (cond.shape[0], self.dim)

        # 生成采样计划，包括 sigma 和 gamma
        sigmas = self.sample_schedule(num_sample_steps)

        # 计算 gamma 值：
        # 如果 sigma 在 [S_tmin, S_tmax] 之间，则 gamma = min(S_churn / num_sample_steps, sqrt(2) - 1)；否则，gamma = 0。
        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / num_sample_steps, sqrt(2) - 1),
            0.
        )

        # 将 sigma 和 gamma 配对，并添加下一个 sigma 和 gamma
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # 初始化序列为噪声
        init_sigma = sigmas[0]

        seq = init_sigma * torch.randn(shape, device = self.device)

        # 逐步去噪
        for sigma, sigma_next, gamma in tqdm(sigmas_and_gammas, desc = 'sampling time step'):
            # 将张量元素转换为浮点数
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

            # 生成随机噪声
            eps = self.S_noise * torch.randn(shape, device = self.device) # stochastic sampling

            sigma_hat = sigma + gamma * sigma
            # 计算噪声扰动
            seq_hat = seq + sqrt(sigma_hat ** 2 - sigma ** 2) * eps

            # 前向传播获取去噪后的输出
            model_output = self.preconditioned_network_forward(seq_hat, sigma_hat, cond = cond, clamp = clamp)
            # 计算去噪后的输出与噪声输入的差异
            denoised_over_sigma = (seq_hat - model_output) / sigma_hat

            # 计算下一个时间步的序列
            seq_next = seq_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            # 如果不是最后一个时间步，则进行二阶修正
            if sigma_next != 0:
                # 计算下一个时间步的去噪输出
                model_output_next = self.preconditioned_network_forward(seq_next, sigma_next, cond = cond, clamp = clamp)
                # 计算去噪输出的导数
                denoised_prime_over_sigma = (seq_next - model_output_next) / sigma_next
                # 进行二阶修正
                seq_next = seq_hat + 0.5 * (sigma_next - sigma_hat) * (denoised_over_sigma + denoised_prime_over_sigma)
            # 更新序列
            seq = seq_next

        if clamp:
            # 如果需要裁剪，则对输出进行裁剪
            seq = seq.clamp(-1., 1.)

        return seq

    def loss_weight(self, sigma):
        """
        计算损失权重。

        参数:
            sigma (Tensor): 噪声水平。

        返回:
            Tensor: 计算得到的损失权重。
        """
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma * self.sigma_data) ** -2

    def noise_distribution(self, batch_size):
        """
        生成噪声分布。

        参数:
            batch_size (int): 批次大小。

        返回:
            Tensor: 生成的噪声分布，形状为 (batch_size,)。
        """
        return (self.P_mean + self.P_std * torch.randn((batch_size,), device = self.device)).exp()

    def forward(self, seq, *, cond):
        """
        前向传播方法。

        参数:
            seq (torch.Tensor): 输入序列，形状为 (batch_size, dim)。
            cond (torch.Tensor): 条件输入，形状为 (batch_size, cond_dim)。

        返回:
            torch.Tensor: 计算得到的损失值。
        """
        batch_size, dim, device = *seq.shape, self.device

        assert dim == self.dim, f'dimension of sequence being passed in must be {self.dim} but received {dim}'
        
        # 生成噪声分布
        sigmas = self.noise_distribution(batch_size)
        # 对 sigma 进行填充，使其与输入序列的维度一致
        padded_sigmas = right_pad_dims_to(seq, sigmas)

        # 生成随机噪声
        noise = torch.randn_like(seq)

        # 生成加噪后的序列
        noised_seq = seq + padded_sigmas * noise 

        # 前向传播获取去噪后的输出
        denoised = self.preconditioned_network_forward(noised_seq, sigmas, cond = cond)

        # 计算均方误差损失
        losses = F.mse_loss(denoised, seq, reduction = 'none')
        # 对损失进行平均
        losses = reduce(losses, 'b ... -> b', 'mean')

        # 乘以损失权重
        losses = losses * self.loss_weight(sigmas)

        # 返回平均损失
        return losses.mean()


class AutoregressiveDiffusion(Module):
    """
    自回归扩散模型（Autoregressive Diffusion）类。

    该模型结合了自回归模型和扩散模型，通过逐步生成序列中的每个元素来生成数据。
    """
    def __init__(
        self,
        dim,
        *,
        max_seq_len,
        depth = 8,
        dim_head = 64,
        heads = 8,
        mlp_depth = 3,
        mlp_width = None,
        dim_input = None,
        decoder_kwargs: dict = dict(),
        mlp_kwargs: dict = dict(),
        diffusion_kwargs: dict = dict(
            clamp_during_sampling = True
        )
    ):
        """
        初始化自回归扩散模型。

        参数:
            dim (int): 模型的特征维度。
            max_seq_len (int): 最大序列长度。
            depth (int, 可选): Transformer 解码器的层数，默认为8。
            dim_head (int, 可选): 每个注意力头的维度，默认为64。
            heads (int, 可选): 注意力头的数量，默认为8。
            mlp_depth (int, 可选): MLP 层的深度，默认为3。
            mlp_width (int, 可选): MLP 层的宽度，默认为 None。如果为 None，则使用 `dim`。
            dim_input (int, 可选): 输入特征的维度，默认为 None。如果为 None，则使用 `dim`。
            decoder_kwargs (Dict[str, Any], 可选): 解码器的关键字参数，默认为空字典。
            mlp_kwargs (Dict[str, Any], 可选): MLP 的关键字参数，默认为空字典。
            diffusion_kwargs (Dict[str, Any], 可选): 扩散模型的关键字参数，默认为包含 `clamp_during_sampling=True`。
        """
        super().__init__()

        # 初始化起始标记，形状为 (dim,)
        self.start_token = nn.Parameter(torch.zeros(dim))
        # 保存最大序列长度
        self.max_seq_len = max_seq_len
        # 初始化绝对位置嵌入，嵌入维度为 `dim`，词汇表大小为 `max_seq_len`
        self.abs_pos_emb = nn.Embedding(max_seq_len, dim)

        # 如果未指定输入维度，则使用模型维度
        dim_input = default(dim_input, dim)
        # 保存输入维度
        self.dim_input = dim_input
        # 初始化线性层，将输入维度映射到模型维度
        self.proj_in = nn.Linear(dim_input, dim)

        # 初始化 Transformer 解码器
        self.transformer = Decoder(
            dim = dim,
            depth = depth,
            heads = heads,
            attn_dim_head = dim_head,
            **decoder_kwargs
        )

        # 初始化去噪器，使用 MLP 实现
        self.denoiser = MLP(
            dim_cond = dim,
            dim_input = dim_input,
            depth = mlp_depth,
            width = default(mlp_width, dim),
            **mlp_kwargs
        )

        # 初始化扩散模型，使用 ElucidatedDiffusion 实现
        self.diffusion = ElucidatedDiffusion(
            dim_input,
            self.denoiser,
            **diffusion_kwargs
        )

    @property
    def device(self):
        """
        获取当前设备。

        返回:
            torch.device: 当前设备（CPU 或 GPU）。
        """
        return next(self.transformer.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        batch_size = 1,
        prompt  = None
    ):
        """
        使用自回归扩散模型进行采样。

        参数:
            batch_size (int, 可选): 批次大小，默认为1。
            prompt (Tensor, 可选): 提示序列，默认为 None。

        返回:
            torch.Tensor: 生成的样本，形状为 (batch_size, max_seq_len, dim_input)。
        """
        self.eval()

        # 重复起始标记，生成初始输入
        start_tokens = repeat(self.start_token, 'd -> b 1 d', b = batch_size)

        if not exists(prompt):
            # 如果没有提供提示序列，则创建一个空的输出张量
            out = torch.empty((batch_size, 0, self.dim_input), device = self.device, dtype = torch.float32)
        else:
            # 如果提供了提示序列，则使用提示序列作为输出
            out = prompt

        cache = None

        # 逐步生成序列
        for _ in tqdm(range(self.max_seq_len - out.shape[1]), desc = 'tokens'):
            # 将输出投影到模型维度
            cond = self.proj_in(out)
            # 将起始标记与条件输入连接起来
            cond = torch.cat((start_tokens, cond), dim = 1)
            # 添加绝对位置嵌入
            cond = cond + self.abs_pos_emb(torch.arange(cond.shape[1], device = self.device))
            # 前向传播通过 Transformer 解码器
            cond, cache = self.transformer(cond, cache = cache, return_hiddens = True)
            # 获取最后一个时间步的条件输入
            last_cond = cond[:, -1]
            # 使用扩散模型进行去噪预测
            denoised_pred = self.diffusion.sample(cond = last_cond)
            # 重塑去噪预测的形状为 (batch, 1, dim_input)
            denoised_pred = rearrange(denoised_pred, 'b d -> b 1 d')
            # 将去噪预测与当前输出连接起来
            out = torch.cat((out, denoised_pred), dim = 1)
        # 返回生成的序列
        return out

    def forward(
        self,
        seq
    ):
        """
        前向传播方法。

        参数:
            seq (torch.Tensor): 输入序列，形状为 (batch_size, seq_len, dim_input)。

        返回:
            torch.Tensor: 计算得到的扩散损失。
        """
        # 获取批次大小、序列长度和特征维度
        b, seq_len, dim = seq.shape

        assert dim == self.dim_input
        assert seq_len == self.max_seq_len

        # 将输入序列分割为输入部分和目标部分
        # 输入部分为 seq[:, :-1]，即除了最后一个时间步的所有时间步
        # 目标部分为整个序列 seq，用于预测
        seq, target = seq[:, :-1], seq

        # 投影输入序列
        # 将输入序列通过线性层进行投影，形状保持为 (b, seq_len, dim)
        seq = self.proj_in(seq)
        # 重复起始标记，并将其添加到输入序列的开头
        start_token = repeat(self.start_token, 'd -> b 1 d', b = b) # 重复起始标记，形状为 (b, 1, dim)

        # 将起始标记与输入序列连接，形状为 (b, seq_len + 1, dim)
        seq = torch.cat((start_token, seq), dim = 1)

        # 添加绝对位置嵌入，形状保持为 (b, seq_len + 1, dim)
        seq = seq + self.abs_pos_emb(torch.arange(seq_len, device = self.device))

        # 通过 Transformer 模型处理输入序列，输出条件输入，形状为 (b, seq_len + 1, dim)
        cond = self.transformer(seq) 

        # 打包批次和序列维度，以便对每个时间步应用不同的噪声水平
        # 使用 pack_one 函数对目标序列和条件输入进行打包
        target, _ = pack_one(target, '* d')  # 打包目标序列，形状为 (b * seq_len, dim)
        cond, _ = pack_one(cond, '* d')  # 打包条件输入，形状为 (b * (seq_len + 1), dim)

        # 计算扩散损失
        diffusion_loss = self.diffusion(target, cond = cond)

        # 返回扩散损失
        return diffusion_loss


def normalize_to_neg_one_to_one(img):
    """
    将图像像素值归一化到 [-1, 1] 范围。

    参数:
        img (torch.Tensor): 输入图像张量。

    返回:
        torch.Tensor: 归一化后的图像张量。
    """
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    """
    将张量值从 [-1, 1] 范围反归一化到 [0, 1] 范围。

    参数:
        t (torch.Tensor): 输入张量。

    返回:
        torch.Tensor: 反归一化后的张量。
    """
    return (t + 1) * 0.5


class ImageAutoregressiveDiffusion(Module):
    """
    图像自回归扩散模型（Image Autoregressive Diffusion）类。

    该模型将图像分割成多个小块（patch），然后使用自回归扩散模型逐块生成图像。
    """
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        channels = 3,
        model: dict = dict(),
    ):
        """
        初始化图像自回归扩散模型。

        参数:
            image_size (int): 图像的尺寸（例如，256 表示 256x256 的图像）。
            patch_size (int): 图像块的尺寸（例如，16 表示 16x16 的块）。
            channels (int, 可选): 图像的通道数，默认为3（RGB）。
            model (Dict[str, Any], 可选): 自回归扩散模型的关键字参数，默认为空字典。
        """
        super().__init__()
        assert divisible_by(image_size, patch_size)

        # 计算图像中块的总数
        num_patches = (image_size // patch_size) ** 2
        # 计算每个块的特征维度
        dim_in = channels * patch_size ** 2

        # 保存图像尺寸
        self.image_size = image_size
        # 保存块尺寸
        self.patch_size = patch_size

        # 将图像分割成块
        # 假设输入图像形状为 (batch_size, channels, height, width)
        # 重塑为 (batch_size, num_patches, channels * patch_size * patch_size)
        self.to_tokens = Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1 = patch_size, p2 = patch_size)

        # 初始化自回归扩散模型
        self.model = AutoregressiveDiffusion(
            **model,
            dim_input = dim_in,
            max_seq_len = num_patches
        )

        # 将块重新组合成图像
        # 重塑为 (batch_size, channels, height * patch_size, width * patch_size)
        self.to_image = Rearrange('b (h w) (c p1 p2) -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size, h = int(math.sqrt(num_patches)))

    def sample(self, batch_size = 1):
        """
        使用图像自回归扩散模型进行采样。

        参数:
            batch_size (int, 可选): 批次大小，默认为1。

        返回:
            torch.Tensor: 生成的图像，形状为 (batch_size, channels, height, width)。
        """
        # 使用自回归扩散模型生成块序列
        tokens = self.model.sample(batch_size = batch_size)
        # 将块序列重新组合成图像
        images = self.to_image(tokens)
        # 对图像进行反归一化
        return unnormalize_to_zero_to_one(images)

    def forward(self, images):
        """
        前向传播方法。

        参数:
            images (torch.Tensor): 输入图像，形状为 (batch_size, channels, height, width)。

        返回:
            torch.Tensor: 计算得到的损失。
        """
        # 对输入图像进行归一化
        images = normalize_to_neg_one_to_one(images)
        # 将图像分割成块
        tokens = self.to_tokens(images)
        # 将块序列传递给自回归扩散模型，并返回损失
        return self.model(tokens)
