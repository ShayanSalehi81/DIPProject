"""Four vector-first, publication-oriented figures for the paired study."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

PALETTE = {"clean": "#0072B2", "degraded": "#D55E00", "restored": "#009E73"}
CONDITION_ORDER = ("clean", "degraded", "restored")
CONDITION_LABELS = {"clean": "Clean", "degraded": "Degraded", "restored": "Restored"}


class PaperFigureFactory:
    """Render consistent 300-DPI PNG plus editable PDF/SVG versions."""

    def __init__(self, output_dir: Path, *, dpi: int) -> None:
        self.output_dir = output_dir
        self.dpi = dpi
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 9,
                "axes.titlesize": 10,
                "axes.labelsize": 9,
                "legend.fontsize": 8,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "figure.facecolor": "white",
                "savefig.facecolor": "white",
            }
        )

    def render_all(
        self,
        *,
        accuracy: pd.DataFrame,
        breakdown: pd.DataFrame,
        transitions: pd.DataFrame,
        wide: pd.DataFrame,
    ) -> list[str]:
        stems = [
            self.overall_accuracy(accuracy),
            self.degradation_heatmap(breakdown),
            self.severity_curve(breakdown),
            self.paired_transitions(transitions, breakdown, wide),
        ]
        return [
            str(self.output_dir / f"{stem}.{suffix}")
            for stem in stems
            for suffix in ("png", "pdf", "svg")
        ]

    def overall_accuracy(self, accuracy: pd.DataFrame) -> str:
        stem = "figure_1_overall_robustness"
        scope_order = [
            value
            for value in ("Overall", "Coarse perception", "Fine perception")
            if value in set(accuracy["scope_label"])
        ]
        fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
        x = np.arange(len(scope_order), dtype=float)
        offsets = {-1: -0.22, 0: 0.0, 1: 0.22}
        for condition_index, condition in enumerate(CONDITION_ORDER):
            rows = (
                accuracy.loc[accuracy["condition"] == condition]
                .set_index("scope_label")
                .reindex(scope_order)
            )
            y = rows["accuracy"].to_numpy(dtype=float) * 100
            low = rows["ci_low"].to_numpy(dtype=float) * 100
            high = rows["ci_high"].to_numpy(dtype=float) * 100
            ax.errorbar(
                x + offsets[condition_index - 1],
                y,
                yerr=np.vstack([y - low, high - y]),
                fmt="o",
                markersize=7,
                capsize=4,
                elinewidth=1.5,
                color=PALETTE[condition],
                label=CONDITION_LABELS[condition],
                zorder=3,
            )
            for x_value, y_value in zip(x + offsets[condition_index - 1], y, strict=True):
                ax.text(x_value, y_value + 2.2, f"{y_value:.1f}", ha="center", fontsize=7)
        ax.set_xticks(x, scope_order)
        ax.set_ylabel("Macro-average accuracy (%)")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
        ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False)
        ax.set_title(
            "Qwen3-VL robustness under paired microscopy degradation and restoration",
            pad=32,
            weight="bold",
        )
        ax.text(
            0.0,
            -0.18,
            "Points are task-macro averages; bars are paired bootstrap 95% confidence intervals.",
            transform=ax.transAxes,
            color="#4D4D4D",
            fontsize=8,
        )
        self._save(fig, stem)
        return stem

    def degradation_heatmap(self, breakdown: pd.DataFrame) -> str:
        stem = "figure_2_degradation_profile"
        data = breakdown.loc[breakdown["factor"] == "degradation_kind"].copy()
        data = data.sort_values("degradation_effect")
        labels = [str(value).replace("_", " ") for value in data["value"]]
        accuracy_matrix = data[
            ["clean_accuracy", "degraded_accuracy", "restored_accuracy"]
        ].to_numpy(float) * 100
        effect_matrix = data[
            ["degradation_effect", "restoration_gain", "residual_vs_clean"]
        ].to_numpy(float) * 100
        height = max(4.7, 0.42 * len(data) + 1.7)
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(9.2, height),
            gridspec_kw={"width_ratios": [1.0, 1.05]},
            constrained_layout=True,
        )
        image_a = axes[0].imshow(accuracy_matrix, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        axes[0].set_xticks(range(3), ["Clean", "Degraded", "Restored"])
        axes[0].set_yticks(range(len(labels)), labels)
        axes[0].set_title("A  Accuracy by condition", weight="bold", loc="left")
        self._annotate_heatmap(axes[0], accuracy_matrix, suffix="%", threshold=55)
        fig.colorbar(image_a, ax=axes[0], fraction=0.05, pad=0.03, label="Accuracy (%)")

        bound = max(10.0, float(np.nanmax(np.abs(effect_matrix))))
        norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
        image_b = axes[1].imshow(effect_matrix, cmap="RdBu", norm=norm, aspect="auto")
        axes[1].set_xticks(
            range(3),
            ["Degraded - clean", "Restored - degraded", "Restored - clean"],
            rotation=18,
        )
        axes[1].set_yticks(range(len(labels)), [f"n={int(value)}" for value in data["n"]])
        axes[1].set_title("B  Paired effects", weight="bold", loc="left")
        self._annotate_heatmap(axes[1], effect_matrix, suffix=" pp", threshold=bound * 0.55)
        fig.colorbar(image_b, ax=axes[1], fraction=0.05, pad=0.03, label="Percentage points")
        fig.suptitle(
            "Degradation-specific robustness loss and restoration utility",
            weight="bold",
            fontsize=11,
        )
        self._save(fig, stem)
        return stem

    def severity_curve(self, breakdown: pd.DataFrame) -> str:
        stem = "figure_3_severity_response"
        data = breakdown.loc[breakdown["factor"] == "severity"].copy()
        order = [value for value in ("weak", "medium", "strong") if value in set(data["value"])]
        data = data.set_index("value").reindex(order)
        x = np.arange(len(order))
        fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
        for condition in CONDITION_ORDER:
            estimate = data[f"{condition}_accuracy"].to_numpy(float) * 100
            low = data[f"{condition}_ci_low"].to_numpy(float) * 100
            high = data[f"{condition}_ci_high"].to_numpy(float) * 100
            ax.plot(
                x,
                estimate,
                marker="o",
                markersize=7,
                linewidth=2,
                color=PALETTE[condition],
                label=CONDITION_LABELS[condition],
            )
            ax.fill_between(x, low, high, color=PALETTE[condition], alpha=0.14, linewidth=0)
        for index, n_value in enumerate(data["n"]):
            ax.text(index, 2, f"n={int(n_value)}", ha="center", fontsize=7, color="#555555")
        ax.set_xticks(x, [item.title() for item in order])
        ax.set_xlabel("Synthetic degradation severity")
        ax.set_ylabel("Macro-average accuracy (%)")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
        ax.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01))
        ax.set_title(
            "Severity-robustness response and restoration trajectory",
            pad=30,
            weight="bold",
        )
        ax.text(
            0.0,
            -0.18,
            "Shaded regions are paired bootstrap 95% confidence intervals.",
            transform=ax.transAxes,
            fontsize=8,
            color="#4D4D4D",
        )
        self._save(fig, stem)
        return stem

    def paired_transitions(
        self, transitions: pd.DataFrame, breakdown: pd.DataFrame, wide: pd.DataFrame
    ) -> str:
        stem = "figure_4_paired_transitions"
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), constrained_layout=False)
        fig.subplots_adjust(left=0.08, right=0.93, top=0.84, bottom=0.28, wspace=0.30)
        pattern_order = ["111", "101", "100", "110", "011", "010", "001", "000"]
        colors = {
            "111": "#0072B2",
            "101": "#009E73",
            "100": "#D55E00",
            "110": "#CC79A7",
            "011": "#56B4E9",
            "010": "#E69F00",
            "001": "#88CCEE",
            "000": "#777777",
        }
        available_scopes = set(
            transitions.groupby("scope_label", observed=True)["count"]
            .sum()
            .loc[lambda values: values > 0]
            .index
        )
        scopes = [
            scope
            for scope in ("Overall", "Coarse perception", "Fine perception")
            if scope in available_scopes
        ]
        left = np.zeros(len(scopes), dtype=float)
        for pattern in pattern_order:
            values = []
            for scope in scopes:
                match = transitions.loc[
                    (transitions["scope_label"] == scope) & (transitions["pattern"] == pattern),
                    "fraction",
                ]
                values.append(float(match.iloc[0]) if not match.empty else 0.0)
            label_match = transitions.loc[transitions["pattern"] == pattern, "transition"]
            label = str(label_match.iloc[0]) if not label_match.empty else pattern
            axes[0].barh(scopes, values, left=left, color=colors[pattern], label=label, height=0.62)
            for y, (start, value) in enumerate(zip(left, values, strict=True)):
                if value >= 0.08:
                    axes[0].text(
                        start + value / 2,
                        y,
                        f"{value * 100:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if pattern in {"111", "100", "000"} else "black",
                    )
            left += np.asarray(values)
        axes[0].set_xlim(0, 1)
        axes[0].set_xlabel("Fraction of paired samples")
        axes[0].set_title(
            "A  Correctness trajectories (clean to degraded to restored)",
            loc="left",
            weight="bold",
        )
        axes[0].invert_yaxis()
        axes[0].legend(
            ncol=2,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            fontsize=7,
        )

        family = breakdown.loc[breakdown["factor"] == "degradation_kind"].copy()
        x = -family["degradation_effect"].to_numpy(float) * 100
        y = family["restoration_gain"].to_numpy(float) * 100
        sizes = 35 + family["n"].to_numpy(float) * 8
        recovery_scatter = axes[1].scatter(
            x,
            y,
            s=sizes,
            c=family["residual_vs_clean"].to_numpy(float) * 100,
            cmap="PiYG",
            edgecolor="white",
            linewidth=0.8,
            alpha=0.9,
        )
        limits = np.concatenate([x, y, np.array([0.0])])
        low = min(-5.0, float(np.nanmin(limits)) - 4)
        high = max(5.0, float(np.nanmax(limits)) + 4)
        axes[1].plot(
            [low, high],
            [low, high],
            linestyle="--",
            color="#555555",
            linewidth=1,
            label="Full recovery",
        )
        axes[1].axhline(0, color="#999999", linewidth=0.8)
        axes[1].axvline(0, color="#999999", linewidth=0.8)
        for index, (_, row) in enumerate(family.iterrows()):
            place_left = x[index] > high * 0.78
            axes[1].annotate(
                str(row["value"]).replace("_", " "),
                (x[index], y[index]),
                xytext=(-4 if place_left else 4, 5 if index % 2 == 0 else -10),
                textcoords="offset points",
                fontsize=6.5,
                ha="right" if place_left else "left",
            )
        axes[1].set_xlim(low, high)
        axes[1].set_ylim(low, high)
        axes[1].set_xlabel("Robustness loss: clean - degraded (pp)")
        axes[1].set_ylabel("Restoration gain: restored - degraded (pp)")
        axes[1].set_title("B  Recovery map by degradation family", loc="left", weight="bold")
        axes[1].grid(color="#E5E5E5", linewidth=0.6)
        axes[1].legend(frameon=False, loc="upper left")
        fig.colorbar(
            recovery_scatter,
            ax=axes[1],
            fraction=0.046,
            pad=0.03,
            label="Residual: restored - clean (pp)",
        )
        fig.suptitle(
            f"Paired outcomes across {len(wide)} unique Micro-Bench samples",
            weight="bold",
            fontsize=11,
        )
        self._save(fig, stem)
        return stem

    @staticmethod
    def _annotate_heatmap(
        ax: plt.Axes, matrix: np.ndarray, *, suffix: str, threshold: float
    ) -> None:
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                if not math.isfinite(value):
                    label = "NA"
                else:
                    label = f"{value:+.1f}{suffix}" if "pp" in suffix else f"{value:.1f}{suffix}"
                ax.text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if abs(value) > threshold else "black",
                )

    def _save(self, figure: plt.Figure, stem: str) -> None:
        metadata = {"Creator": "Micro-Q-Instruct Phase 1.5", "Title": stem}
        figure.savefig(self.output_dir / f"{stem}.png", dpi=self.dpi, bbox_inches="tight")
        figure.savefig(self.output_dir / f"{stem}.pdf", bbox_inches="tight", metadata=metadata)
        figure.savefig(self.output_dir / f"{stem}.svg", bbox_inches="tight", metadata=metadata)
        plt.close(figure)
