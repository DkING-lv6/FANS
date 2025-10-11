import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td


class MLPBlock(nn.Module):
    def __init__(self, hidden_dim: int, dtype: torch.dtype = torch.float32):
        super(MLPBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.dtype = dtype

        # 初始化全连接层
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # 初始化权重
        nn.init.orthogonal_(self.fc1.weight, gain=torch.sqrt(torch.tensor(2.0)))
        nn.init.orthogonal_(self.fc2.weight, gain=torch.sqrt(torch.tensor(2.0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x


def gaussian(x):
    return torch.exp(-x * x)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, active_type: str, dtype: torch.dtype = torch.float32):
        super(ResidualBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.active_type = active_type

        # LayerNorm
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # 初始化全连接层
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 1)
        self.fc2 = nn.Linear(hidden_dim * 1, hidden_dim)

        # 初始化权重
        if self.active_type == "gaussian":
            nn.init.kaiming_normal_(self.fc1.weight.data.normal_(mean=0.0, std=0.02))
            nn.init.kaiming_normal_(self.fc2.weight.data.normal_(mean=0.0, std=0.02))
        elif self.active_type == "relu":
            nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.layer_norm(x)
        x = self.fc1(x)
        if self.active_type == "gaussian":
            x = gaussian(x)
        elif self.active_type == "relu":
            x = F.relu(x)
        x = self.fc2(x)
        return res + x


class NormalTanhPolicy(nn.Module):
    def __init__(
            self,
            action_dim: int,
            state_dependent_std: bool = True,
            kernel_init_scale: float = 1.0,
            log_std_min: float = -10.0,
            log_std_max: float = 2.0,
            dtype: torch.dtype = torch.float32,
    ):
        super(NormalTanhPolicy, self).__init__()
        self.action_dim = action_dim
        self.state_dependent_std = state_dependent_std
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.dtype = dtype

        # 均值网络
        self.mean_fc = nn.Linear(256, action_dim)
        nn.init.orthogonal_(self.mean_fc.weight, gain=kernel_init_scale)

        # 标准差网络
        if self.state_dependent_std:
            self.log_std_fc = nn.Linear(256, action_dim)
            nn.init.orthogonal_(self.log_std_fc.weight, gain=kernel_init_scale)
        else:
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, inputs: torch.Tensor, temperature: float = 1.0) -> td.Distribution:
        # 计算均值
        means = self.mean_fc(inputs)

        # 计算标准差
        if self.state_dependent_std:
            log_stds = self.log_std_fc(inputs)
        else:
            log_stds = self.log_std

        # 限制标准差范围
        log_stds = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (
                1 + torch.tanh(log_stds)
        )

        # 创建正态分布
        dist = td.Normal(loc=means, scale=torch.exp(log_stds) * temperature)

        # 使用 Tanh 变换
        dist = td.TransformedDistribution(dist, td.transforms.TanhTransform())

        return dist


class LinearCritic(nn.Module):
    def __init__(self, kernel_init_scale: float = 1.0, dtype: torch.dtype = torch.float32):
        super(LinearCritic, self).__init__()
        self.kernel_init_scale = kernel_init_scale
        self.dtype = dtype

        # 初始化全连接层
        self.fc = nn.Linear(256, 1)
        nn.init.orthogonal_(self.fc.weight, gain=kernel_init_scale)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        value = self.fc(inputs)
        return value
