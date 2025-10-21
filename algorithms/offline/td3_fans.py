
import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from fans_baseunity import MLPBlock, ResidualBlock, NormalTanhPolicy, LinearCritic

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb

wandb.login()

TensorBatch = List[torch.Tensor]

os.environ["CUDA_VISIBLE_DEVICES"]="0"

@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "halfcheetah-medium-expert-v2"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(3e6)  # Max time steps to run environment
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load

    # ensemble
    num_critics = 10
    # active_type
    actor_active_type = "relu"
    critic_active_type = "gaussian"

    # TD3
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount for
    expl_noise: float = 0.1  # Std of Gaussian exploration noise
    tau: float = 0.005  # Target network update rate
    policy_noise: float = 0.2  # Noise added to target actor during critic update
    noise_clip: float = 0.5  # Range to clip target actor noise
    policy_freq: int = 2  # Frequency of delayed actor updates
    # TD3 + BC
    alpha: float = 2.5  # Coefficient for Q function in actor loss
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    # Wandb logging
    project: str = "CORL_fans"
    group: str = "halfcheetah-medium-expert--v2"
    name: str = "fans"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
        env: gym.Env,
        state_mean: Union[np.ndarray, float] = 0.0,
        state_std: Union[np.ndarray, float] = 1.0,
        reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
                state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


class ReplayBuffer:
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            buffer_size: int,
            device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError


def set_seed(
        seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic
    # torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


@torch.no_grad()
def eval_actor(
        env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


class TD3Encoder(nn.Module):
    def __init__(self, input_dim: int, block_type: str, num_blocks: int, hidden_dim: int, active_type: str,
                 dtype: torch.dtype):
        super(TD3Encoder, self).__init__()
        self.block_type = block_type
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.active_type = active_type

        self.input_layer = nn.Linear(input_dim, hidden_dim)

        if self.block_type == "mlp":
            self.mlp_block = MLPBlock(self.hidden_dim, dtype=self.dtype)
        elif self.block_type == "residual":
            self.dense = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.residual_blocks = nn.ModuleList(
                [ResidualBlock(self.hidden_dim, self.active_type, dtype=self.dtype) for _ in range(self.num_blocks)]
            )
            self.layer_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)

        if self.block_type == "mlp":
            x = self.mlp_block(x)
        elif self.block_type == "residual":
            x = self.dense(x)
            for block in self.residual_blocks:
                x = block(x)
            x = self.layer_norm(x)
        return x


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()


class Critic(nn.Module):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            block_type: str = "residual",
            num_blocks: int = 2,
            hidden_dim: int = 512,
            critic_active_type: str = "",
            dtype: torch.dtype = torch.float32,
    ):
        super(Critic, self).__init__()


        self.encoder = TD3Encoder(
            input_dim=state_dim + action_dim,
            block_type=block_type,
            num_blocks=num_blocks,
            hidden_dim=hidden_dim,
            active_type=critic_active_type,
            dtype=dtype,
        )


        self.predictor = LinearCritic()

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:

        sa = torch.cat([state, action], dim=1)

        z = self.encoder(sa)

        q_value = self.predictor(z)
        return q_value


class TD3_BC:
    def __init__(
            self,
            max_action: float,
            actor: nn.Module,
            actor_target: nn.Module,
            actor_optimizer: torch.optim.Optimizer,
            critics: nn.Module,
            critics_target: nn.Module,
            critic_optimizer: torch.optim.Optimizer,
            num_critics: int,
            discount: float = 0.99,
            tau: float = 0.005,
            policy_noise: float = 0.2,
            noise_clip: float = 0.5,
            policy_freq: int = 2,
            alpha: float = 2.5,
            device: str = "cpu",
    ):
        self.actor = actor
        self.actor_target = actor_target
        self.actor_optimizer = actor_optimizer
        self.critics = critics
        self.critics_target = critics_target
        self.critic_optimizer = critic_optimizer
        self.num_critics = num_critics
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha

        self.total_it = 0
        self.device = device

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        state, action, reward, next_state, done = batch
        not_done = 1 - done

        with torch.no_grad():
            # Select action according to actor and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute the target Q value
            target_Qs = [critic_target(next_state, next_action) for critic_target in self.critics_target]
            target_q = torch.min(torch.stack(target_Qs), dim=0)[0]
            target_q = reward + not_done * self.discount * target_q

        # Get current Q estimates
        current_Qs = [critic(state, action) for critic in self.critics]

        # Compute critic loss
        critic_loss = sum((current_Q - target_q).pow(2).mean() for current_Q in current_Qs)

        log_dict["critic_loss"] = critic_loss.item()
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed actor updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            # pi = self.actor(state)
            # q = self.critics[0](state, pi)
            # lmbda = self.alpha / q.abs().mean().detach()

            # # actor_loss = -lmbda * q.mean() + F.mse_loss(pi, action)
            # actor_loss = -lmbda * q.mean() #remove BC
            actor_loss = -(torch.stack([critic(state, self.actor(state)) for critic in self.critics], dim=0).min(
                dim=0).values)
            log_dict["actor_loss"] = actor_loss.mean().item()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.mean().backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for i in range(len(self.critics)):
                for param, target_param in zip(self.critics[i].parameters(), self.critics_target[i].parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                # soft_update(self.critics_target[i], self.critics[i], self.tau)
            # soft_update(self.actor_target, self.actor, self.tau)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "critic": self.critic.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.critic.load_state_dict(state_dict["critic"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_target = copy.deepcopy(self.actor)

        self.total_it = state_dict["total_it"]


@pyrallis.wrap()
def train(config: TrainConfig):
    env = gym.make(config.env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    dataset = d4rl.qlearning_dataset(env)

    if config.normalize_reward:
        modify_reward(dataset, config.env)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    replay_buffer.load_d4rl_dataset(dataset)

    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)
    actor_active_type = config.actor_active_type
    critic_active_type = config.critic_active_type

    actor = Actor(state_dim, action_dim, max_action).to(config.device)
    actor_target = Actor(state_dim, action_dim, max_action).to(config.device)
    actor_target.load_state_dict(actor.state_dict())
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)

    critics = [
        Critic(state_dim, action_dim, "residual", 2, 256, critic_active_type, dtype=torch.float32).to(config.device) for
        _ in range(config.num_critics)]
    critics_target = [
        Critic(state_dim, action_dim, "residual", 2, 256, critic_active_type, dtype=torch.float32).to(config.device) for
        _ in range(config.num_critics)]
    for i in range(config.num_critics):
        critics_target[i].load_state_dict(critics[i].state_dict())
    critic_optimizer = torch.optim.Adam([param for critic in critics for param in critic.parameters()], lr=1e-4)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_target": actor_target,
        "actor_optimizer": actor_optimizer,
        "critics": critics,
        "critics_target": critics_target,
        "critic_optimizer": critic_optimizer,
        "num_critics": config.num_critics,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # TD3
        "policy_noise": config.policy_noise * max_action,
        "noise_clip": config.noise_clip * max_action,
        "policy_freq": config.policy_freq,
        # TD3 + BC
        "alpha": config.alpha,
    }

    print("---------------------------------------")
    print(f"Training FANS, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = TD3_BC(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(asdict(config))

    evaluations = []
    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores = eval_actor(
                env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
            eval_score = eval_scores.mean()
            normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
            evaluations.append(normalized_eval_score)
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
            )
            print("---------------------------------------")

            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )

            wandb.log(
                {"d4rl_normalized_score": normalized_eval_score},
                step=trainer.total_it,
            )


if __name__ == "__main__":
    train()
