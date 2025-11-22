"""
Improved MATD3 (Multi-Agent Twin Delayed Deep Deterministic Policy Gradient)
Optimized specifically for the Speaker-Listener environment with:
- Separate policies for speaker and listener (no parameter sharing)
- Proper hyperparameters for communication tasks
- Discrete communication with grounding loss
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import copy


class ReplayBuffer:
    """Multi-agent replay buffer for off-policy learning."""

    def __init__(self, capacity: int = 100000, num_agents: int = 2):
        self.capacity = capacity
        self.num_agents = num_agents
        self.buffer = []
        self.position = 0

    def push(
        self,
        obs: Dict[str, np.ndarray],
        actions: Dict[str, np.ndarray],
        rewards: Dict[str, float],
        next_obs: Dict[str, np.ndarray],
        dones: Dict[str, bool],
    ) -> None:
        """Store transition in buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (obs, actions, rewards, next_obs, dones)
        self.position = (self.position + 1) % self.capacity

    def sample(
        self, batch_size: int
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        torch.Tensor,
        Dict[str, torch.Tensor],
        torch.Tensor,
    ]:
        """Sample batch of transitions with replacement."""
        indices = np.random.randint(0, len(self.buffer), batch_size)

        batch_obs = {agent: [] for agent in ["speaker_0", "listener_0"]}
        batch_actions = {agent: [] for agent in ["speaker_0", "listener_0"]}
        batch_rewards: List[float] = []
        batch_next_obs = {agent: [] for agent in ["speaker_0", "listener_0"]}
        batch_dones: List[bool] = []

        for idx in indices:
            obs, actions, rewards, next_obs, dones = self.buffer[idx]

            for agent in ["speaker_0", "listener_0"]:
                batch_obs[agent].append(obs[agent])
                batch_actions[agent].append(actions[agent])
                batch_next_obs[agent].append(next_obs[agent])

            # Shared reward for cooperative task
            batch_rewards.append(float(sum(rewards.values())))
            batch_dones.append(bool(any(dones.values())))

        # Convert to tensors
        result_obs = {
            agent: torch.FloatTensor(np.array(batch_obs[agent]))
            for agent in ["speaker_0", "listener_0"]
        }
        result_actions = {
            agent: torch.FloatTensor(np.array(batch_actions[agent]))
            for agent in ["speaker_0", "listener_0"]
        }
        result_rewards = torch.FloatTensor(batch_rewards)
        result_next_obs = {
            agent: torch.FloatTensor(np.array(batch_next_obs[agent]))
            for agent in ["speaker_0", "listener_0"]
        }
        result_dones = torch.FloatTensor(batch_dones)

        return result_obs, result_actions, result_rewards, result_next_obs, result_dones

    def __len__(self) -> int:
        return len(self.buffer)


class ImprovedActor(nn.Module):
    """Improved actor network with layer normalization and orthogonal init."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        max_action: float = 1.0,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.max_action = max_action
        self.use_layer_norm = use_layer_norm

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        if use_layer_norm:
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)

        # Orthogonal initialization
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through actor network."""
        x = self.fc1(obs)
        if self.use_layer_norm:
            x = self.ln1(x)
        x = F.relu(x)

        x = self.fc2(x)
        if self.use_layer_norm:
            x = self.ln2(x)
        x = F.relu(x)

        # Network output in [-1, 1]
        squashed = torch.tanh(self.fc3(x))
        # Environment expects [0, 1]
        action = 0.5 * (squashed + 1.0)
        return action



class ImprovedCritic(nn.Module):
    """Twin critic networks for MATD3 with centralized training."""

    def __init__(
        self,
        num_agents: int,
        obs_dims: List[int],
        action_dims: List[int],
        hidden_dim: int = 64,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        total_obs_dim = sum(obs_dims)
        total_action_dim = sum(action_dims)
        input_dim = total_obs_dim + total_action_dim

        # Q1
        self.q1_fc1 = nn.Linear(input_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_fc3 = nn.Linear(hidden_dim, 1)

        # Q2 (twin)
        self.q2_fc1 = nn.Linear(input_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_fc3 = nn.Linear(hidden_dim, 1)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.q1_ln1 = nn.LayerNorm(hidden_dim)
            self.q1_ln2 = nn.LayerNorm(hidden_dim)
            self.q2_ln1 = nn.LayerNorm(hidden_dim)
            self.q2_ln2 = nn.LayerNorm(hidden_dim)

        # Orthogonal init
        for module in [self.q1_fc1, self.q1_fc2, self.q2_fc1, self.q2_fc2]:
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.q1_fc3.weight, gain=0.01)
        nn.init.orthogonal_(self.q2_fc3.weight, gain=0.01)

    def forward(
        self, obs_list: List[torch.Tensor], action_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through both critics."""
        x = torch.cat(obs_list + action_list, dim=-1)

        # Q1
        q1 = self.q1_fc1(x)
        if self.use_layer_norm:
            q1 = self.q1_ln1(q1)
        q1 = F.relu(q1)

        q1 = self.q1_fc2(q1)
        if self.use_layer_norm:
            q1 = self.q1_ln2(q1)
        q1 = F.relu(q1)

        q1 = self.q1_fc3(q1)

        # Q2
        q2 = self.q2_fc1(x)
        if self.use_layer_norm:
            q2 = self.q2_ln1(q2)
        q2 = F.relu(q2)

        q2 = self.q2_fc2(q2)
        if self.use_layer_norm:
            q2 = self.q2_ln2(q2)
        q2 = F.relu(q2)

        q2 = self.q2_fc3(q2)

        return q1, q2

    def q1_forward(
        self, obs_list: List[torch.Tensor], action_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass through Q1 only (for actor updates)."""
        x = torch.cat(obs_list + action_list, dim=-1)

        q1 = self.q1_fc1(x)
        if self.use_layer_norm:
            q1 = self.q1_ln1(q1)
        q1 = F.relu(q1)

        q1 = self.q1_fc2(q1)
        if self.use_layer_norm:
            q1 = self.q1_ln2(q1)
        q1 = F.relu(q1)

        q1 = self.q1_fc3(q1)
        return q1


class CommunicationModule(nn.Module):
    """
    Communication module with grounding loss for emergent protocols.

    FIX: The original implementation's "grounding" head reconstructed the
    speaker observation from a *hidden embedding* rather than the discrete
    message, which meant the message itself was not actually forced to carry
    the information. Here we reconstruct directly from the discrete message,
    making the grounding loss do what it says. 
    """

    def __init__(
        self,
        speaker_obs_dim: int,
        message_dim: int = 32,
        vocab_size: int = 20,
    ):
        super().__init__()

        # Speaker: observation -> continuous embedding
        self.speaker_encoder = nn.Sequential(
            nn.Linear(speaker_obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, message_dim),
        )

        # Discrete message generation (logits over vocabulary)
        self.message_head = nn.Linear(message_dim, vocab_size)

        # Listener: message (one-hot) -> features passed to listener policy
        self.listener_decoder = nn.Sequential(
            nn.Linear(vocab_size, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, 64),
        )

        # Grounding: reconstruct speaker observation FROM MESSAGE
        self.reconstruction_head = nn.Sequential(
            nn.Linear(vocab_size, 64),
            nn.ReLU(),
            nn.Linear(64, speaker_obs_dim),
        )

    def forward(
        self, speaker_obs: torch.Tensor, temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a discrete message and compute grounding loss.

        Returns:
            message:            [B, vocab_size] one-hot (or relaxed) message
            listener_features:  [B, 64] features for listener policy
            grounding_loss:     scalar MSE(speaker_obs, reconstructed_obs)
        """
        # Encode speaker observation to continuous embedding
        encoded = self.speaker_encoder(speaker_obs)  # [B, message_dim]

        # Logits over discrete vocabulary
        logits = self.message_head(encoded)  # [B, vocab_size]

        # Discrete message via Gumbel-Softmax during training, argmax at eval
        if self.training:
            message = F.gumbel_softmax(logits, tau=temperature, hard=True)
        else:
            message = F.one_hot(
                logits.argmax(dim=-1), num_classes=logits.size(-1)
            ).float()

        # Listener features from message
        listener_features = self.listener_decoder(message)  # [B, 64]

        # Grounding loss: reconstruct speaker obs from MESSAGE (not encoded)
        reconstructed_obs = self.reconstruction_head(message)  # [B, speaker_obs_dim]
        grounding_loss = F.mse_loss(reconstructed_obs, speaker_obs)

        return message, listener_features, grounding_loss


class ImprovedMATD3:
    """Improved MATD3 with optimizations for speaker-listener environment."""

    def __init__(
        self,
        num_agents: int,
        obs_spaces: List,
        action_spaces: List,
        lr_actor: float = 5e-4,  # Optimized for communication
        lr_critic: float = 5e-4,
        gamma: float = 0.99,
        tau: float = 0.005,  # Softer target updates
        policy_delay: int = 2,  # TD3: delayed policy updates
        noise_std: float = 0.1,
        noise_clip: float = 0.5,
        max_action: float = 1.0,
        buffer_size: int = 100000,
        batch_size: int = 64,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_communication: bool = True,
        grounding_weight: float = 0.5,
    ):
        self.num_agents = num_agents
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.max_action = max_action
        self.batch_size = batch_size
        self.device = device
        self.use_communication = use_communication
        self.grounding_weight = grounding_weight

        # Dimensions from spaces
        obs_dims = [space.shape[0] for space in obs_spaces]
        action_dims = [space.shape[0] for space in action_spaces]

        # Separate actors for speaker & listener (no parameter sharing)
        self.actors: List[ImprovedActor] = []
        self.actors_target: List[ImprovedActor] = []
        self.actor_optimizers: List[optim.Optimizer] = []

        for i in range(num_agents):
            # Listener gets additional communication features
            input_dim = obs_dims[i]
            if use_communication and i == 1:
                input_dim += 64  # features from CommunicationModule

            actor = ImprovedActor(
                obs_dim=input_dim,
                action_dim=action_dims[i],
                hidden_dim=64,
                max_action=max_action,
            ).to(device)

            actor_target = copy.deepcopy(actor)

            self.actors.append(actor)
            self.actors_target.append(actor_target)
            self.actor_optimizers.append(optim.Adam(actor.parameters(), lr=lr_actor))

        # Centralized twin critics
        self.critic = ImprovedCritic(
            num_agents=num_agents,
            obs_dims=obs_dims,
            action_dims=action_dims,
            hidden_dim=64,
        ).to(device)

        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Communication module
        if use_communication:
            self.comm_module = CommunicationModule(
                speaker_obs_dim=obs_dims[0],
                message_dim=32,
                vocab_size=20,
            ).to(device)
            self.comm_optimizer = optim.Adam(self.comm_module.parameters(), lr=lr_actor)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_size, num_agents=num_agents)

        # Counters
        self.total_steps = 0
        self.update_counter = 0

    # ------------------------------------------------------------------
    #  Action selection
    # ------------------------------------------------------------------

    def select_actions(
        self, observations: Dict[str, np.ndarray], add_noise: bool = True
    ) -> Dict[str, np.ndarray]:
        """Select actions for all agents with optional exploration noise."""
        actions: Dict[str, np.ndarray] = {}

        # Convert observations to tensors
        obs_tensors: List[torch.Tensor] = []
        for agent in ["speaker_0", "listener_0"]:
            obs = torch.FloatTensor(observations[agent]).unsqueeze(0).to(self.device)
            obs_tensors.append(obs)

        # Generate communication (no gradient during action selection)
        listener_features: Optional[torch.Tensor] = None
        if self.use_communication:
            with torch.no_grad():
                _, listener_features, _ = self.comm_module(obs_tensors[0])

        # Actor forward passes
        for i, agent in enumerate(["speaker_0", "listener_0"]):
            obs = obs_tensors[i]

            # Listener gets communication features
            if self.use_communication and i == 1 and listener_features is not None:
                obs = torch.cat([obs, listener_features], dim=-1)

            with torch.no_grad():
                action = self.actors[i](obs)

                if add_noise:
                    noise = torch.randn_like(action) * self.noise_std
                    action = action + noise
                    action = torch.clamp(action, 0.0, 1.0)


            actions[agent] = action.cpu().numpy().squeeze()

        return actions

    # ------------------------------------------------------------------
    #  Training update (TD3)
    # ------------------------------------------------------------------

    def update(self) -> Dict:
        """Update actors, critics and communication module using TD3."""
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # Sample batch
        (
            obs_batch,
            action_batch,
            reward_batch,
            next_obs_batch,
            done_batch,
        ) = self.replay_buffer.sample(self.batch_size)

        # Move to device
        for key in obs_batch:
            obs_batch[key] = obs_batch[key].to(self.device)
            action_batch[key] = action_batch[key].to(self.device)
            next_obs_batch[key] = next_obs_batch[key].to(self.device)
        reward_batch = reward_batch.to(self.device)
        done_batch = done_batch.to(self.device)

        losses: Dict[str, float] = {}

        # ---------------- Critic update ----------------
        with torch.no_grad():
            next_actions: List[torch.Tensor] = []

            # Communication for next state (no grad for targets)
            next_listener_features: Optional[torch.Tensor] = None
            if self.use_communication:
                _, next_listener_features, _ = self.comm_module(
                    next_obs_batch["speaker_0"]
                )

            for i, agent in enumerate(["speaker_0", "listener_0"]):
                next_obs = next_obs_batch[agent]

                # Listener concatenates communication features
                if self.use_communication and i == 1 and next_listener_features is not None:
                    next_obs = torch.cat([next_obs, next_listener_features], dim=-1)

                next_action = self.actors_target[i](next_obs)

                # Target policy smoothing
                noise = torch.randn_like(next_action) * self.noise_std
                noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
                next_action = torch.clamp(next_action, 0.0, 1.0)


                next_actions.append(next_action)

            # Target Q-values
            next_obs_list = [next_obs_batch[agent] for agent in ["speaker_0", "listener_0"]]
            target_q1, target_q2 = self.critic_target(next_obs_list, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward_batch.unsqueeze(1) + self.gamma * (
                1 - done_batch.unsqueeze(1)
            ) * target_q

        # Current Q-values
        obs_list = [obs_batch[agent] for agent in ["speaker_0", "listener_0"]]
        action_list = [action_batch[agent] for agent in ["speaker_0", "listener_0"]]
        current_q1, current_q2 = self.critic(obs_list, action_list)

        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        losses["critic_loss"] = float(critic_loss.item())

        # ---------------- Actor & comm update (delayed) ----------------
        if self.update_counter % self.policy_delay == 0:
            # Communication for current batch (with grad)
            listener_features: Optional[torch.Tensor] = None
            grounding_loss: Optional[torch.Tensor] = None

            if self.use_communication:
                # IMPORTANT: we zero comm grads before accumulating from actor + grounding
                self.comm_optimizer.zero_grad()
                _, listener_features, grounding_loss = self.comm_module(
                    obs_batch["speaker_0"]
                )

            actor_losses: List[torch.Tensor] = []

            for i, agent in enumerate(["speaker_0", "listener_0"]):
                # Build new joint action set (only agent i gets updated action)
                new_actions: List[torch.Tensor] = []
                for j, a in enumerate(["speaker_0", "listener_0"]):
                    if j == i:
                        agent_obs = obs_batch[a]
                        if self.use_communication and j == 1 and listener_features is not None:
                            agent_obs = torch.cat([agent_obs, listener_features], dim=-1)
                        new_action = self.actors[j](agent_obs)
                    else:
                        new_action = action_batch[a]
                    new_actions.append(new_action)

                # Standard TD3 actor loss: maximize Q1
                actor_loss = -self.critic.q1_forward(obs_list, new_actions).mean()
                actor_losses.append(actor_loss)

                # Update actor i
                self.actor_optimizers[i].zero_grad()
                # retain_graph=True because we reuse the graph across agents and comm loss
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
                self.actor_optimizers[i].step()

            # Now update communication module using both policy gradient
            # signal (through listener_features) and grounding loss.
            if self.use_communication:
                if grounding_loss is not None:
                    # Add grounding on top of policy gradient for comm_module
                    total_comm_loss = self.grounding_weight * grounding_loss
                    total_comm_loss.backward()
                    nn.utils.clip_grad_norm_(self.comm_module.parameters(), 0.5)
                    self.comm_optimizer.step()
                    losses["grounding_loss"] = float(grounding_loss.item())
                else:
                    # No grounding, but still step if there are policy grads
                    self.comm_optimizer.step()

            losses["actor_losses"] = [float(l.item()) for l in actor_losses]

            # Soft update targets
            self._soft_update_targets()

        self.update_counter += 1
        return losses

    # ------------------------------------------------------------------
    #  Target updates and utilities
    # ------------------------------------------------------------------

    def _soft_update_targets(self) -> None:
        """Soft update target networks using tau."""
        # Actors
        for i in range(self.num_agents):
            for param, target_param in zip(
                self.actors[i].parameters(), self.actors_target[i].parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

        # Critic
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def add_experience(
        self,
        obs: Dict[str, np.ndarray],
        actions: Dict[str, np.ndarray],
        rewards: Dict[str, float],
        next_obs: Dict[str, np.ndarray],
        dones: Dict[str, bool],
    ) -> None:
        """Add experience to replay buffer."""
        self.replay_buffer.push(obs, actions, rewards, next_obs, dones)
        self.total_steps += 1

    def save_checkpoint(self, filepath: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "actors": [actor.state_dict() for actor in self.actors],
            "actors_target": [target.state_dict() for target in self.actors_target],
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizers": [opt.state_dict() for opt in self.actor_optimizers],
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "total_steps": self.total_steps,
            "update_counter": self.update_counter,
        }

        if self.use_communication:
            checkpoint["comm_module"] = self.comm_module.state_dict()
            checkpoint["comm_optimizer"] = self.comm_optimizer.state_dict()

        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)

        for i in range(self.num_agents):
            self.actors[i].load_state_dict(checkpoint["actors"][i])
            self.actors_target[i].load_state_dict(checkpoint["actors_target"][i])
            self.actor_optimizers[i].load_state_dict(
                checkpoint["actor_optimizers"][i]
            )

        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

        if self.use_communication and "comm_module" in checkpoint:
            self.comm_module.load_state_dict(checkpoint["comm_module"])
            self.comm_optimizer.load_state_dict(checkpoint["comm_optimizer"])

        self.total_steps = checkpoint.get("total_steps", 0)
        self.update_counter = checkpoint.get("update_counter", 0)
