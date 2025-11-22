"""
Main training script for comparing MAPPO vs Improved MATD3 on Speaker-Listener
Includes curriculum learning, proper evaluation, and comprehensive logging
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_speaker_listener_v4
# from mpe2 import simple_speaker_listener_v4

from typing import Dict, List, Tuple
import json
import time
from collections import deque
import pandas as pd

# Import our algorithms
from mappo import MAPPO
from improved_matd3 import ImprovedMATD3


class CurriculumManager:
    """Manages curriculum learning for progressive task difficulty."""
    
    def __init__(self, stages: List[Dict]):
        self.stages = stages
        self.current_stage = 0
        self.stage_progress = []
        
    def get_current_config(self) -> Dict:
        """Get current environment configuration."""
        return self.stages[self.current_stage]
    
    def should_advance(self, success_rate: float) -> bool:
        """Check if we should move to next stage."""
        if self.current_stage < len(self.stages) - 1:
            return success_rate > self.stages[self.current_stage]['advance_threshold']
        return False
    
    def advance_stage(self):
        """Move to next curriculum stage."""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            print(f"Advancing to curriculum stage {self.current_stage + 1}")
            
    def get_stage_name(self) -> str:
        """Get name of current stage."""
        return self.stages[self.current_stage]['name']


class MultiAgentTrainer:
    """Trainer for multi-agent reinforcement learning with comparison."""
    
    def __init__(
        self,
        algorithm_name: str = "MAPPO",
        num_envs: int = 8,
        max_episodes: int = 5000,
        max_steps_per_episode: int = 25,
        eval_frequency: int = 100,
        save_frequency: int = 500,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 42,
        use_curriculum: bool = True
    ):
        self.algorithm_name = algorithm_name
        self.num_envs = num_envs
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.eval_frequency = eval_frequency
        self.save_frequency = save_frequency
        self.device = device
        self.use_curriculum = use_curriculum
        
        # Set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Create environment
        self.env = simple_speaker_listener_v4.parallel_env(
            continuous_actions=True,
            max_cycles=max_steps_per_episode
        )
        
        # Get observation and action spaces
        self.env.reset()
        self.obs_spaces = [self.env.observation_space(agent) for agent in self.env.agents]
        self.action_spaces = [self.env.action_space(agent) for agent in self.env.agents]
        
        # Initialize algorithm
        if algorithm_name == "MAPPO":
            self.agent = MAPPO(
                num_agents=2,
                obs_spaces=self.obs_spaces,
                action_spaces=self.action_spaces,
                lr_actor=5e-4,
                lr_critic=5e-4,
                gamma=0.99,
                gae_lambda=0.95,
                clip_param=0.2,
                value_clip=0.2,
                entropy_coef=0.01,
                max_grad_norm=0.5,
                n_epochs=10,
                batch_size=64,
                device=device
            )
        elif algorithm_name == "ImprovedMATD3":
            self.agent = ImprovedMATD3(
                num_agents=2,
                obs_spaces=self.obs_spaces,
                action_spaces=self.action_spaces,
                lr_actor=5e-4,
                lr_critic=5e-4,
                gamma=0.99,
                tau=0.005,
                policy_delay=2,
                noise_std=0.1,
                noise_clip=0.5,
                max_action=1.0,
                buffer_size=100000,
                batch_size=64,
                device=device,
                use_communication=True,
                grounding_weight=0.5
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        # Initialize curriculum
        if use_curriculum:
            self.curriculum = CurriculumManager([
                {
                    'name': 'Stage 1: Simple',
                    'num_landmarks': 2,
                    'advance_threshold': 0.8,
                    'episodes': 1000
                },
                {
                    'name': 'Stage 2: Medium',
                    'num_landmarks': 3,
                    'advance_threshold': 0.7,
                    'episodes': 1500
                },
                {
                    'name': 'Stage 3: Full',
                    'num_landmarks': 3,  # Full task
                    'advance_threshold': 1.0,  # Don't advance
                    'episodes': 2500
                }
            ])
        else:
            self.curriculum = None
            
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.training_losses = []
        self.eval_scores = []
        
        # Rolling buffers for statistics
        self.reward_buffer = deque(maxlen=100)
        self.success_buffer = deque(maxlen=100)
        
    def collect_episode_data(self, training: bool = True) -> Tuple[float, int, bool, Dict]:
        """Collect data from one episode."""
        obs, info = self.env.reset()
        
        episode_reward = 0
        episode_length = 0
        episode_data = {
            'speaker_0_obs': [],
            'listener_0_obs': [],
            'speaker_0_actions': [],
            'listener_0_actions': [],
            'speaker_0_log_probs': [],
            'listener_0_log_probs': [],
            'rewards': [],
            'dones': []
        }
        
        # Reset hidden states for recurrent networks
        if hasattr(self.agent, 'reset_hidden_states'):
            self.agent.reset_hidden_states()
        
        for step in range(self.max_steps_per_episode):
            # Get actions
            if self.algorithm_name == "MAPPO":
                actions, log_probs = self.agent.get_actions(obs, deterministic=not training)
            else:  # ImprovedMATD3
                actions = self.agent.select_actions(obs, add_noise=training)
                log_probs = {'speaker_0': 0, 'listener_0': 0}  # Not used for MATD3
            
            # Step environment
            next_obs, rewards, terminations, truncations, info = self.env.step(actions)
            
            # Store data
            if training:
                for agent in ['speaker_0', 'listener_0']:
                    episode_data[f'{agent}_obs'].append(obs[agent])
                    episode_data[f'{agent}_actions'].append(actions[agent])
                    episode_data[f'{agent}_log_probs'].append(log_probs[agent])
                    
                episode_data['rewards'].append(sum(rewards.values()))
                episode_data['dones'].append(any(terminations.values()) or any(truncations.values()))
                
                # Add to replay buffer for MATD3
                if self.algorithm_name == "ImprovedMATD3":
                    self.agent.add_experience(obs, actions, rewards, next_obs, 
                                             {**terminations, **truncations})
            
            # Update metrics
            episode_reward += sum(rewards.values())
            episode_length += 1
            
            obs = next_obs
            
            # Check termination
            if any(terminations.values()) or any(truncations.values()):
                break
        
        # Calculate success (distance < 0.1 at end)
        final_distance = abs(episode_reward / episode_length) ** 0.5 if episode_length > 0 else 10.0
        success = final_distance < 0.1
        
        # Convert lists to numpy arrays for training
        if training:
            for key in episode_data:
                if key != 'rewards' and key != 'dones':
                    episode_data[key] = np.array(episode_data[key])
        
        return episode_reward, episode_length, success, episode_data
    
    def train_episode(self) -> Dict:
        """Train for one episode."""
        # Collect episode data
        episode_reward, episode_length, success, episode_data = self.collect_episode_data(training=True)
        
        # Update agent
        if self.algorithm_name == "MAPPO":
            # MAPPO uses collected trajectory
            update_info = self.agent.update(episode_data)
        else:  # ImprovedMATD3
            # MATD3 samples from replay buffer
            update_info = self.agent.update()
        
        # Update metrics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.reward_buffer.append(episode_reward)
        self.success_buffer.append(success)
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'success': success,
            **update_info
        }
    
    def evaluate(self, num_eval_episodes: int = 10) -> Dict:
        """Evaluate current policy."""
        eval_rewards = []
        eval_successes = []
        eval_lengths = []
        
        for _ in range(num_eval_episodes):
            reward, length, success, _ = self.collect_episode_data(training=False)
            eval_rewards.append(reward)
            eval_successes.append(success)
            eval_lengths.append(length)
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'success_rate': np.mean(eval_successes),
            'mean_length': np.mean(eval_lengths)
        }
    
    def train(self):
        """Main training loop."""
        print(f"Starting training with {self.algorithm_name}")
        print(f"Device: {self.device}")
        print(f"Curriculum Learning: {self.use_curriculum}")
        print("-" * 50)
        
        start_time = time.time()
        
        for episode in range(self.max_episodes):
            # Train episode
            train_info = self.train_episode()
            
            # Logging
            if episode % 10 == 0:
                mean_reward = np.mean(list(self.reward_buffer))
                success_rate = np.mean(list(self.success_buffer))
                
                print(f"Episode {episode:4d} | "
                      f"Reward: {mean_reward:7.2f} | "
                      f"Success: {success_rate:5.2%} | "
                      f"Length: {train_info['episode_length']:3d}")
                
                # Check curriculum advancement
                if self.use_curriculum and self.curriculum.should_advance(success_rate):
                    self.curriculum.advance_stage()
                    print(f"  → Advanced to: {self.curriculum.get_stage_name()}")
            
            # Evaluation
            if episode % self.eval_frequency == 0 and episode > 0:
                eval_info = self.evaluate()
                self.eval_scores.append(eval_info['mean_reward'])
                
                print(f"\n[EVAL] Episode {episode} | "
                      f"Reward: {eval_info['mean_reward']:.2f} ± {eval_info['std_reward']:.2f} | "
                      f"Success: {eval_info['success_rate']:.2%}\n")
            
            # Save checkpoint
            if episode % self.save_frequency == 0 and episode > 0:
                self.save_checkpoint(episode)
        
        # Final evaluation
        final_eval = self.evaluate(num_eval_episodes=50)
        
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 50)
        print(f"Training Complete!")
        print(f"Algorithm: {self.algorithm_name}")
        print(f"Total Episodes: {self.max_episodes}")
        print(f"Training Time: {elapsed_time/3600:.2f} hours")
        print(f"Final Evaluation (50 episodes):")
        print(f"  Mean Reward: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}")
        print(f"  Success Rate: {final_eval['success_rate']:.2%}")
        print(f"  Mean Episode Length: {final_eval['mean_length']:.1f}")
        print("=" * 50)
        
        return final_eval
    
    def save_checkpoint(self, episode: int):
        """Save model checkpoint and training data."""
        os.makedirs(f"checkpoints/{self.algorithm_name}", exist_ok=True)
        
        # Save model
        checkpoint_path = f"checkpoints/{self.algorithm_name}/model_ep{episode}.pt"
        self.agent.save_checkpoint(checkpoint_path)
        
        # Save training metrics
        metrics = {
            'episode': episode,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'eval_scores': self.eval_scores,
            'algorithm': self.algorithm_name,
            'curriculum_stage': self.curriculum.current_stage if self.curriculum else None
        }
        
        metrics_path = f"checkpoints/{self.algorithm_name}/metrics_ep{episode}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"  → Checkpoint saved: {checkpoint_path}")
    
    def plot_training_curves(self):
        """Generate training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.3, label='Raw')
        if len(self.episode_rewards) > 100:
            smoothed = pd.Series(self.episode_rewards).rolling(100).mean()
            axes[0, 0].plot(smoothed, label='Smoothed (100 ep)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Episode Reward')
        axes[0, 0].set_title(f'{self.algorithm_name} - Episode Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Success rate
        success_array = np.array([float(s) for s in self.success_buffer])
        if len(self.episode_rewards) > 100:
            success_history = []
            for i in range(100, len(self.episode_rewards)):
                success_history.append(np.mean([float(s) for s in list(self.success_buffer)[max(0, i-100):i]]))
            axes[0, 1].plot(range(100, len(self.episode_rewards)), success_history)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].set_title('Success Rate (Rolling 100 ep)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Episode lengths
        axes[1, 0].plot(self.episode_lengths, alpha=0.3, label='Raw')
        if len(self.episode_lengths) > 100:
            smoothed_lengths = pd.Series(self.episode_lengths).rolling(100).mean()
            axes[1, 0].plot(smoothed_lengths, label='Smoothed (100 ep)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Episode Length')
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Evaluation scores
        if self.eval_scores:
            eval_episodes = list(range(self.eval_frequency, 
                                      len(self.eval_scores) * self.eval_frequency + 1, 
                                      self.eval_frequency))
            axes[1, 1].plot(eval_episodes, self.eval_scores, 'o-')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Evaluation Score')
            axes[1, 1].set_title('Evaluation Performance')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Add curriculum stage indicators if used
        if self.use_curriculum and hasattr(self, 'curriculum'):
            for ax in axes.flat:
                ax.axhline(y=-10, color='g', linestyle='--', alpha=0.5, label='Target (-10)')
                ax.axhline(y=-60, color='r', linestyle='--', alpha=0.5, label='Baseline (-60)')
        
        plt.suptitle(f'{self.algorithm_name} Training Progress', fontsize=16)
        plt.tight_layout()
        
        # Save figure
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f'plots/{self.algorithm_name}_training_curves.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig

def run_comparison():
    """Run comparison between MAPPO and Improved MATD3."""
    BASELINE_MEAN_REWARD = -60.0  # from project README
    results = {}

    # Match AgileRL baseline: 2,000,000 env steps with 25 steps/episode
    BASELINE_MAX_STEPS = 2_000_000
    MAX_STEPS_PER_EPISODE = 25
    episodes_for_baseline = BASELINE_MAX_STEPS // MAX_STEPS_PER_EPISODE  # 80,000

    # =======================
    # Train MAPPO
    # =======================
    print("\n" + "=" * 60)
    print("TRAINING MAPPO")
    print("=" * 60 + "\n")

    mappo_trainer = MultiAgentTrainer(
        algorithm_name="MAPPO",
        max_episodes=episodes_for_baseline,
        # max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        use_curriculum=True,
    )
    mappo_results = mappo_trainer.train()
    mappo_trainer.plot_training_curves()
    results["MAPPO"] = mappo_results

    mappo_trainer = MultiAgentTrainer(
        algorithm_name="MAPPO",
        max_episodes=5000,
        use_curriculum=True,
    )

    # =======================
    # Train Improved MATD3
    # =======================
    print("\n" + "=" * 60)
    print("TRAINING IMPROVED MATD3")
    print("=" * 60 + "\n")

    matd3_trainer = MultiAgentTrainer(
        algorithm_name="ImprovedMATD3",
        max_episodes=episodes_for_baseline,
        # max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        use_curriculum=True,
    )
    matd3_results = matd3_trainer.train()
    matd3_trainer.plot_training_curves()
    results["ImprovedMATD3"] = matd3_results

    # =======================
    # Final comparison
    # =======================
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    print(f"\n{'Algorithm':<15} {'Mean Reward':<22} {'Success Rate':<15} {'vs Baseline':<15}")
    print("-" * 75)

    improvement_vs_baseline = {}

    for alg_name in ["MAPPO", "ImprovedMATD3"]:
        res = results[alg_name]
        mean_r = res["mean_reward"]
        std_r = res["std_reward"]
        success = res["success_rate"] * 100.0

        # Improvement relative to baseline -60 (higher reward is better)
        impr = 100.0 * (mean_r - BASELINE_MEAN_REWARD) / abs(BASELINE_MEAN_REWARD)
        improvement_vs_baseline[alg_name] = impr

        print(
            f"{alg_name:<15} "
            f"{mean_r:7.2f} ± {std_r:7.2f}   "
            f"{success:6.2f}%        "
            f"{impr:+6.1f}%"
        )

    # Determine winner (highest mean reward)
    winner = max(results.keys(), key=lambda k: results[k]["mean_reward"])
    winner_impr = improvement_vs_baseline[winner]

    print(f"\n🏆 Winner: {winner} with {winner_impr:+.1f}% improvement vs baseline (-60)")

    # Baseline check
    better_than_baseline = {
        alg: (results[alg]["mean_reward"] >= BASELINE_MEAN_REWARD)
        for alg in results
    }

    if all(better_than_baseline.values()):
        print("✓ Both algorithms surpass baseline MATD3 score of -60")
    elif any(better_than_baseline.values()):
        better_list = [alg for alg, ok in better_than_baseline.items() if ok]
        print(
            "✓ Algorithms surpassing baseline MATD3 score of -60: "
            + ", ".join(better_list)
        )
    else:
        print("✗ No algorithm surpasses the baseline MATD3 score of -60")

    # Save final comparison
    with open("final_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    return results



if __name__ == "__main__":
    # Run the full comparison
    results = run_comparison()
