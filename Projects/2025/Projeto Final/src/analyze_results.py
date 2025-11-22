"""
Analysis and visualization tools for Speaker-Listener RL experiments.

This version compares ONLY:
  - BaselineMATD3  (AgileRL baseline, loaded from models/MATD3/training_scores_history.npy)
  - ImprovedMATD3  (our implementation, loaded from checkpoints/ImprovedMATD3)

All plots and metrics are computed from real data. No synthetic / illustrative
curves are used.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List
import json
import os
from matplotlib.gridspec import GridSpec

# Set style for publication-quality plots (with safe fallback)
try:
    plt.style.use("seaborn-v0_8-paper")
except Exception:
    try:
        plt.style.use("seaborn-paper")
    except Exception:
        # Fall back to default Matplotlib style
        pass

sns.set_palette("husl")


class ExperimentAnalyzer:
    """Comprehensive analysis of baseline vs Improved MATD3 experiments."""

    def __init__(self, results_dir: str = "checkpoints"):
        self.results_dir = results_dir
        self.algorithms = ["ImprovedMATD3", "BaselineMATD3"]

        # Will be overwritten by load_baseline_data() once baseline npy is found
        self.baseline_score = -60.0
        self.target_score = -10.0  # arbitrary "good performance" threshold

    # ------------------------------------------------------------------ #
    # Data loading
    # ------------------------------------------------------------------ #
    def load_training_data(self, algorithm: str) -> Dict:
        """
        Load training metrics for an algorithm from checkpoints/.
        Only used for ImprovedMATD3.
        """
        algo_dir = os.path.join(self.results_dir, algorithm)
        if not os.path.exists(algo_dir):
            raise FileNotFoundError(
                f"No checkpoint directory found for {algorithm} at {algo_dir}"
            )

        metrics_files = [f for f in os.listdir(algo_dir) if f.startswith("metrics_")]
        if not metrics_files:
            raise FileNotFoundError(
                f"No metrics_*.json files found in {algo_dir} for {algorithm}"
            )

        latest_file = sorted(metrics_files)[-1]
        with open(os.path.join(algo_dir, latest_file), "r") as f:
            data = json.load(f)

        if "episode_rewards" not in data:
            raise KeyError(
                f"'episode_rewards' key not found in {latest_file} for {algorithm}"
            )

        return data

    def load_baseline_data(self) -> Dict:
        """
        Load baseline MATD3 training scores from AgileRL run:
        models/MATD3/training_scores_history.npy

        Also updates self.baseline_score to the final-100-episode mean so that
        improvement_from_baseline is measured against the true baseline.
        """
        baseline_path = os.path.join("models", "MATD3", "training_scores_history.npy")
        if not os.path.exists(baseline_path):
            raise FileNotFoundError(
                f"Baseline file not found at {baseline_path}. "
                "Run the AgileRL baseline training or place the npy file there."
            )

        baseline_scores = np.load(baseline_path).astype(np.float32)
        if len(baseline_scores) >= 100:
            self.baseline_score = float(baseline_scores[-100:].mean())
        else:
            self.baseline_score = float(baseline_scores.mean())

        return {"episode_rewards": baseline_scores.tolist()}

    # ------------------------------------------------------------------ #
    # Statistics / tests
    # ------------------------------------------------------------------ #
    def compute_statistics(self, rewards: List[float]) -> Dict:
        """Compute comprehensive statistics for a reward series."""
        rewards = np.asarray(rewards, dtype=np.float32)
        if len(rewards) == 0:
            raise ValueError("compute_statistics() received an empty rewards list")

        final_window = rewards[-100:] if len(rewards) >= 100 else rewards

        return {
            "mean": float(np.mean(rewards)),
            "std": float(np.std(final_window)),  # std over final performance
            "min": float(np.min(rewards)),
            "max": float(np.max(rewards)),
            "median": float(np.median(rewards)),
            "q25": float(np.percentile(rewards, 25)),
            "q75": float(np.percentile(rewards, 75)),
            "final_100_mean": float(np.mean(final_window)),
            "improvement_from_baseline": float(
                (np.mean(final_window) - self.baseline_score)
                / abs(self.baseline_score)
                * 100.0
            ),
        }

    def perform_significance_test(
        self, rewards1: List[float], rewards2: List[float]
    ) -> Dict:
        """Perform statistical significance test between two algorithms."""
        rewards1 = np.asarray(rewards1, dtype=np.float32)
        rewards2 = np.asarray(rewards2, dtype=np.float32)

        final1 = rewards1[-100:] if len(rewards1) >= 100 else rewards1
        final2 = rewards2[-100:] if len(rewards2) >= 100 else rewards2

        if len(final1) < 2 or len(final2) < 2:
            raise ValueError("Not enough data for significance test")

        # Welch's t-test
        t_stat, p_value = stats.ttest_ind(final1, final2, equal_var=False)

        # Effect size (Cohen's d with pooled std)
        pooled_std = np.sqrt((np.std(final1) ** 2 + np.std(final2) ** 2) / 2.0)
        cohens_d = (np.mean(final1) - np.mean(final2)) / pooled_std if pooled_std > 0 else 0.0

        if abs(cohens_d) > 0.8:
            effect = "large"
        elif abs(cohens_d) > 0.5:
            effect = "medium"
        elif abs(cohens_d) > 0.2:
            effect = "small"
        else:
            effect = "negligible"

        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "significant": bool(p_value < 0.05),
            "effect_size": effect,
        }

    def analyze_convergence(self, rewards: List[float], window: int = 100) -> Dict:
        """Analyze convergence properties of the algorithm."""
        rewards = list(rewards)
        if len(rewards) < window:
            return {"converged": False, "convergence_episode": None}

        rolling_mean = pd.Series(rewards).rolling(window).mean()
        rolling_std = pd.Series(rewards).rolling(window).std()

        convergence_threshold = 0.1
        converged = False
        convergence_episode = None

        if len(rewards) >= window + 500:
            for i in range(window, len(rewards) - 500):
                window_std = rolling_std[i : i + 500].mean()
                window_mean = abs(rolling_mean[i : i + 500].mean())

                if window_mean > 0 and window_std / window_mean < convergence_threshold:
                    converged = True
                    convergence_episode = i
                    break

        improvement_curve = (rolling_mean - self.baseline_score) / abs(
            self.baseline_score
        )
        learning_efficiency = float(
            np.trapz(improvement_curve.dropna()) / len(rewards)
        )

        return {
            "converged": converged,
            "convergence_episode": convergence_episode,
            "learning_efficiency": learning_efficiency,
            "final_variance": float(rolling_std.iloc[-1]) if len(rolling_std) > 0 else None,
        }

    # ------------------------------------------------------------------ #
    # Plots
    # ------------------------------------------------------------------ #
    def generate_comparison_plot(
        self, save_path: str = "plots/algorithm_comparison.png"
    ):
        """Generate comparison plot: BaselineMATD3 vs ImprovedMATD3."""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Load data
        algorithm_data: Dict[str, Dict] = {}
        improved = self.load_training_data("ImprovedMATD3")
        baseline = self.load_baseline_data()
        algorithm_data["ImprovedMATD3"] = improved
        algorithm_data["BaselineMATD3"] = baseline

        # 1. Learning curves
        ax1 = fig.add_subplot(gs[0, :])
        for algo, data in algorithm_data.items():
            rewards = data["episode_rewards"]
            episodes = np.arange(len(rewards))

            ax1.plot(episodes, rewards, alpha=0.15, linewidth=0.5)

            if len(rewards) > 100:
                smoothed = pd.Series(rewards).rolling(100).mean()
                ax1.plot(episodes, smoothed, label=algo, linewidth=2)

        ax1.axhline(
            y=self.baseline_score, color="r", linestyle="--", alpha=0.5, label="Baseline mean"
        )
        ax1.axhline(
            y=self.target_score, color="g", linestyle="--", alpha=0.5, label="Target"
        )
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Episode Reward")
        ax1.set_title("Learning Curves: Baseline vs Improved MATD3")
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)

        # 2. Final performance distribution
        ax2 = fig.add_subplot(gs[1, 0])
        final_rewards: Dict[str, List[float]] = {}
        for algo, data in algorithm_data.items():
            rewards = data["episode_rewards"]
            final = rewards[-100:] if len(rewards) >= 100 else rewards
            final_rewards[algo] = final

        labels = list(final_rewards.keys())
        data_series = [final_rewards[k] for k in labels]
        positions = range(len(labels))

        bp = ax2.boxplot(
            data_series,
            labels=labels,
            positions=positions,
            widths=0.6,
            patch_artist=True,
        )
        colors = ["lightgreen", "lightgray"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

        ax2.axhline(
            y=self.baseline_score, color="r", linestyle="--", alpha=0.5
        )
        ax2.axhline(
            y=self.target_score, color="g", linestyle="--", alpha=0.5
        )
        ax2.set_ylabel("Final Episode Reward")
        ax2.set_title("Final Performance Distribution (Last 100 Episodes)")
        ax2.grid(True, alpha=0.3, axis="y")

        # 3. Convergence speed (only meaningful for ImprovedMATD3)
        ax3 = fig.add_subplot(gs[1, 1])
        convergence_data = []
        for algo, data in algorithm_data.items():
            if algo == "BaselineMATD3":
                continue
            conv_info = self.analyze_convergence(data["episode_rewards"])
            convergence_data.append(
                {
                    "Algorithm": algo,
                    "Converged": conv_info["converged"],
                    "Episode": conv_info["convergence_episode"]
                    if conv_info["converged"]
                    else len(data["episode_rewards"]),
                }
            )

        if convergence_data:
            conv_df = pd.DataFrame(convergence_data)
            x_pos = np.arange(len(conv_df))
            bars = ax3.bar(x_pos, conv_df["Episode"])

            for bar, converged in zip(bars, conv_df["Converged"]):
                bar.set_color("green" if converged else "orange")

            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(conv_df["Algorithm"])
            ax3.set_ylabel("Episodes to Convergence")
            ax3.set_title("Convergence Speed (ImprovedMATD3)")
            ax3.grid(True, alpha=0.3, axis="y")

        # 4. Success rate over time (reward > -15 as proxy)
        ax4 = fig.add_subplot(gs[1, 2])
        for algo, data in algorithm_data.items():
            if algo == "BaselineMATD3":
                continue
            rewards = data["episode_rewards"]
            window = 100
            if len(rewards) <= window:
                continue

            success_rate = []
            for i in range(window, len(rewards)):
                window_rewards = rewards[i - window : i]
                rate = np.mean([1.0 if r > -15 else 0.0 for r in window_rewards])
                success_rate.append(rate)

            if success_rate:
                ax4.plot(
                    range(window, len(rewards)),
                    success_rate,
                    label=algo,
                    linewidth=2,
                )

        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Success Rate")
        ax4.set_title("Success Rate Evolution (100-ep window, proxy)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Statistical comparison table
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis("tight")
        ax5.axis("off")

        stats_data = []
        for algo, data in algorithm_data.items():
            stats_dict = self.compute_statistics(data["episode_rewards"])
            stats_data.append(
                [
                    algo,
                    f"{stats_dict['final_100_mean']:.2f} ± {stats_dict['std']:.2f}",
                    f"{stats_dict['improvement_from_baseline']:.1f}%",
                    f"{stats_dict['min']:.2f}",
                    f"{stats_dict['max']:.2f}",
                ]
            )

        table = ax5.table(
            cellText=stats_data,
            colLabels=[
                "Algorithm",
                "Final Mean ± Std",
                "Improvement vs Baseline",
                "Min",
                "Max",
            ],
            cellLoc="center",
            loc="center",
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        for i in range(len(stats_data) + 1):
            for j in range(5):
                cell = table[(i, j)]
                if i == 0:
                    cell.set_facecolor("#4CAF50")
                    cell.set_text_props(weight="bold", color="white")
                else:
                    if j == 2:
                        value = float(stats_data[i - 1][j].replace("%", ""))
                        if value > 50:
                            cell.set_facecolor("#90EE90")
                        elif value > 0:
                            cell.set_facecolor("#FFFFE0")
                        else:
                            cell.set_facecolor("#FFB6C1")

        plt.suptitle(
            "Baseline vs Improved MATD3: Speaker-Listener Environment",
            fontsize=16,
            fontweight="bold",
        )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        return fig

    # ------------------------------------------------------------------ #
    # LaTeX table
    # ------------------------------------------------------------------ #
    def generate_latex_table(self, save_path: str = "tables/results_table.tex"):
        """Generate LaTeX table for Baseline vs Improved MATD3."""
        results: Dict[str, Dict] = {}

        baseline = self.load_baseline_data()
        improved = self.load_training_data("ImprovedMATD3")

        results["BaselineMATD3"] = self.compute_statistics(
            baseline["episode_rewards"]
        )
        results["ImprovedMATD3"] = self.compute_statistics(
            improved["episode_rewards"]
        )

        latex_table = r"""
\begin{table}[h]
\centering
\caption{Baseline vs Improved MATD3 on Speaker-Listener Environment}
\label{tab:results}
\begin{tabular}{lcccc}
\toprule
Algorithm & Final Score & Std Dev & Improvement & Best Score \\
\midrule
"""

        for algo, stats_dict in results.items():
            latex_table += (
                f"{algo} & "
                f"{stats_dict['final_100_mean']:.2f} & "
                f"{stats_dict['std']:.2f} & "
                f"{stats_dict['improvement_from_baseline']:.1f}\\% & "
                f"{stats_dict['max']:.2f} \\\\\n"
            )

        latex_table += r"""
\bottomrule
\end{tabular}
\end{table}
"""

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            f.write(latex_table)

        print(f"LaTeX table saved to {save_path}")
        return latex_table

    # ------------------------------------------------------------------ #
    # Full pipeline
    # ------------------------------------------------------------------ #
    def generate_full_report(self):
        """Generate complete analysis report for Baseline vs Improved MATD3."""
        print("Generating Comprehensive Analysis Report (MATD3 vs Improved)...")
        print("=" * 60)

        print("Creating algorithm comparison plots...")
        self.generate_comparison_plot()

        print("\nStatistical Significance Tests:")
        print("-" * 40)
        baseline = self.load_baseline_data()
        improved = self.load_training_data("ImprovedMATD3")

        sig_test = self.perform_significance_test(
            baseline["episode_rewards"], improved["episode_rewards"]
        )

        print("BaselineMATD3 vs ImprovedMATD3:")
        print(f"  t-statistic: {sig_test['t_statistic']:.3f}")
        print(f"  p-value:     {sig_test['p_value']:.5f}")
        print(f"  Cohen's d:   {sig_test['cohens_d']:.3f}")
        print(f"  Significant: {sig_test['significant']}")
        print(f"  Effect size: {sig_test['effect_size']}")

        print("\nGenerating LaTeX table for paper...")
        self.generate_latex_table()

        print("\n" + "=" * 60)
        print("Analysis complete! Check the 'plots' and 'tables' directories.")


if __name__ == "__main__":
    analyzer = ExperimentAnalyzer()
    analyzer.generate_full_report()
