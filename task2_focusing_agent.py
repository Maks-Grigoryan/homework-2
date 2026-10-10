from __future__ import annotations

import argparse
import json
import operator
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, TypedDict

import imageio_ffmpeg
import matplotlib
import numpy as np
from langgraph.graph import END, START, StateGraph

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Circle

from local_model import LocalInstructionModel, dialogue_text


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    macropixel: int
    focus_radius: float
    background_inner: float
    background_outer: float
    sectors: int
    iterations: int
    flips: int
    noise: float = 0.06

    def encoded(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


class GraphState(TypedDict, total=False):
    dialogue: Annotated[list[dict[str, str]], operator.add]
    experiments: Annotated[list[dict[str, object]], operator.add]
    remaining: list[dict[str, object]]
    pending: dict[str, object]
    next_role: str
    decision_count: int
    general_used: bool


class FocusingTool:
    def __init__(self, output_dir: Path, seed: int = 2026) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dmd_side = 24
        self.camera_side = 48
        rng = np.random.default_rng(seed)
        shape = (self.camera_side**2, self.dmd_side**2)
        self.base = (
            rng.normal(size=shape) + 1j * rng.normal(size=shape)
        ).astype(np.complex64) / np.sqrt(2 * self.dmd_side**2)

    def macro_matrix(self, size: int) -> np.ndarray:
        side = self.dmd_side // size
        grouped = self.base.reshape(
            self.camera_side**2, side, size, side, size
        ).sum(axis=(2, 4))
        return grouped.reshape(self.camera_side**2, side**2)

    def masks(self, settings: Settings) -> tuple[np.ndarray, list[np.ndarray]]:
        yy, xx = np.mgrid[: self.camera_side, : self.camera_side]
        center = (self.camera_side - 1) / 2
        dx, dy = xx - center, yy - center
        radius = np.sqrt(dx**2 + dy**2)
        angle = (np.arctan2(dy, dx) + 2 * np.pi) % (2 * np.pi)
        focus = radius <= settings.focus_radius
        annulus = (radius >= settings.background_inner) & (
            radius < settings.background_outer
        )
        regions = []
        for index in range(settings.sectors):
            low = 2 * np.pi * index / settings.sectors
            high = 2 * np.pi * (index + 1) / settings.sectors
            regions.append(annulus & (angle >= low) & (angle < high))
        return focus, regions

    @staticmethod
    def measure(field: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
        clean = np.abs(field) ** 2
        mean = max(float(clean.mean()), 1e-12)
        sigma = noise * (np.sqrt(clean * mean) + 0.08 * mean)
        return np.clip(clean + rng.normal(0.0, sigma), 0.0, None)

    @staticmethod
    def enhancement(
        image: np.ndarray, focus: np.ndarray, regions: list[np.ndarray]
    ) -> float:
        focus_value = float(image[focus].mean())
        background = float(np.median([image[mask].mean() for mask in regions]))
        return focus_value / max(background, 1e-12)

    def run(self, settings: Settings, run_id: int) -> dict[str, object]:
        if self.dmd_side % settings.macropixel:
            raise ValueError("macropixel must divide the DMD side")
        rng = np.random.default_rng(5000 + run_id)
        matrix = self.macro_matrix(settings.macropixel)
        controls = matrix.shape[1]
        focus, regions = self.masks(settings)
        pattern = rng.integers(0, 2, size=controls, dtype=np.int8)
        field = matrix @ pattern.astype(np.float32)

        initial_shots = [self.measure(field, settings.noise, rng) for _ in range(12)]
        initial = np.mean(initial_shots, axis=0).reshape(
            self.camera_side, self.camera_side
        )
        current = self.measure(field, settings.noise, rng).reshape(
            self.camera_side, self.camera_side
        )
        score = self.enhancement(current, focus, regions)
        trace = [score]

        for _ in range(settings.iterations):
            indices = rng.choice(controls, size=min(settings.flips, controls), replace=False)
            delta = 1 - 2 * pattern[indices]
            candidate_field = field + matrix[:, indices] @ delta.astype(np.float32)
            candidate = self.measure(candidate_field, settings.noise, rng).reshape(
                self.camera_side, self.camera_side
            )
            candidate_score = self.enhancement(candidate, focus, regions)
            if candidate_score > score:
                pattern[indices] = 1 - pattern[indices]
                field = candidate_field
                score = candidate_score
            trace.append(score)

        final_shots = [self.measure(field, settings.noise, rng) for _ in range(12)]
        final = np.mean(final_shots, axis=0).reshape(
            self.camera_side, self.camera_side
        )
        initial_score = self.enhancement(initial, focus, regions)
        final_score = self.enhancement(final, focus, regions)
        stem = f"experiment_{run_id}"
        comparison_path = self.output_dir / f"{stem}_comparison.png"
        focused_path = self.output_dir / f"{stem}_focused.png"
        trace_path = self.output_dir / f"{stem}_trace.npy"
        np.save(trace_path, np.asarray(trace))
        self.save_comparison(
            initial,
            final,
            trace,
            settings,
            initial_score,
            final_score,
            comparison_path,
        )
        self.save_camera(final, settings, focused_path)
        return {
            "run": run_id,
            **asdict(settings),
            "controls": controls,
            "initial_enhancement": initial_score,
            "enhancement": final_score,
            "gain": final_score / max(initial_score, 1e-12),
            "accepted_score": score,
            "comparison_image": str(comparison_path),
            "focused_image": str(focused_path),
            "trace_file": str(trace_path),
        }

    def save_camera(self, image: np.ndarray, settings: Settings, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        shown = ax.imshow(np.log1p(image), cmap="inferno")
        center = (self.camera_side - 1) / 2
        ax.add_patch(
            Circle(
                (center, center),
                settings.focus_radius,
                fill=False,
                edgecolor="cyan",
                linewidth=1.2,
            )
        )
        ax.set_title("Focused camera image")
        ax.set_xlabel("camera x")
        ax.set_ylabel("camera y")
        fig.colorbar(shown, ax=ax, label="log(1 + intensity)")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)

    def save_comparison(
        self,
        initial: np.ndarray,
        final: np.ndarray,
        trace: list[float],
        settings: Settings,
        initial_score: float,
        final_score: float,
        path: Path,
    ) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        vmax = float(np.percentile(np.log1p(final), 99.7))
        axes[0].imshow(np.log1p(initial), cmap="inferno", vmin=0, vmax=vmax)
        axes[0].set_title(f"Initial, eta={initial_score:.2f}")
        axes[1].imshow(np.log1p(final), cmap="inferno", vmin=0, vmax=vmax)
        axes[1].set_title(f"Focused, eta={final_score:.2f}")
        axes[2].plot(trace, color="black", linewidth=1.2)
        axes[2].set_title("Search history")
        axes[2].set_xlabel("iteration")
        axes[2].set_ylabel("measured enhancement")
        for ax in axes[:2]:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(
            f"macropixel={settings.macropixel}, iterations={settings.iterations}, flips={settings.flips}"
        )
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)


def orchestrator_choice(model: LocalInstructionModel, state: GraphState) -> str:
    experiments = state.get("experiments", [])
    pending = state.get("pending", {})
    if len(experiments) >= 3:
        candidates = [
            '{"next_role":"finish","reason":"Three autonomous experiments are complete."}'
        ]
    elif pending:
        candidates = [
            '{"next_role":"executor","reason":"A complete focusing plan is ready for the tool."}',
            '{"next_role":"general","reason":"A definition is still needed before execution."}',
        ]
    else:
        candidates = [
            '{"next_role":"scientist","reason":"Choose the next focusing settings from the results."}'
        ]
        if not state.get("general_used", False):
            candidates.append(
                '{"next_role":"general","reason":"Explain the enhancement metric first."}'
            )
    system = """You are the orchestrator of a four-role optical experiment agent.
Select one candidate JSON. The scientist selects settings, the executor alone
calls the focusing tool, the general agent explains concepts, and finish is
used after three experiments. The user supplies no settings after the initial
request."""
    user = (
        f"Completed experiments: {len(experiments)}; pending settings: {pending}\n"
        f"Dialogue:\n{dialogue_text(state['dialogue'])}\nCandidates:\n"
        + "\n".join(candidates)
    )
    selected, _ = model.rank(system, user, candidates)
    return json.loads(selected)["next_role"]


def build_graph(model: LocalInstructionModel, tool: FocusingTool):
    def orchestrator(state: GraphState) -> GraphState:
        count = int(state.get("decision_count", 0))
        role = "finish" if count >= 10 else orchestrator_choice(model, state)
        text = json.dumps(
            {"next_role": role, "reason": "Model-selected next role."},
            separators=(",", ":"),
        )
        print("ORCHESTRATOR:", text)
        return {
            "dialogue": [{"role": "orchestrator", "text": text}],
            "next_role": role,
            "decision_count": count + 1,
        }

    def scientist(state: GraphState) -> GraphState:
        settings = [Settings(**item) for item in state["remaining"]]
        candidates = [item.encoded() for item in settings]
        summary = [
            {
                "macropixel": item["macropixel"],
                "controls": item["controls"],
                "enhancement": round(float(item["enhancement"]), 4),
            }
            for item in state.get("experiments", [])
        ]
        system = """You are the scientist in an autonomous wavefront-shaping agent.
Choose one candidate settings JSON. Prefer a useful progression from coarse,
fast searches toward more controllable macropixels and smaller focus regions.
Balance iteration count against the number of controls. Return one candidate
exactly; do not invent settings."""
        user = f"Previous results: {json.dumps(summary)}\nCandidates:\n" + "\n".join(
            candidates
        )
        selected, _ = model.rank(system, user, candidates)
        chosen = json.loads(selected)
        remaining = [item for item in state["remaining"] if item != chosen]
        text = "Selected settings: " + json.dumps(chosen, separators=(",", ":"))
        print("SCIENTIST:", text)
        return {
            "dialogue": [{"role": "scientist", "text": text}],
            "pending": chosen,
            "remaining": remaining,
        }

    def general(state: GraphState) -> GraphState:
        text = (
            "Enhancement is the mean focus-region intensity divided by the "
            "median mean intensity of angular regions in the nearby background."
        )
        return {
            "dialogue": [{"role": "general", "text": text}],
            "general_used": True,
        }

    def executor(state: GraphState) -> GraphState:
        settings = Settings(**state["pending"])
        run_id = len(state.get("experiments", [])) + 1
        result = tool.run(settings, run_id)
        text = (
            f"focusing_tool run={run_id}; controls={result['controls']}; "
            f"enhancement={result['enhancement']:.4f}; "
            f"gain={result['gain']:.4f}; image={result['comparison_image']}"
        )
        print("EXECUTOR:", text)
        return {
            "dialogue": [{"role": "executor", "text": text}],
            "experiments": [result],
            "pending": {},
        }

    graph = StateGraph(GraphState)
    graph.add_node("orchestrator", orchestrator)
    graph.add_node("scientist", scientist)
    graph.add_node("general", general)
    graph.add_node("executor", executor)
    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges(
        "orchestrator",
        lambda state: state["next_role"],
        {
            "scientist": "scientist",
            "general": "general",
            "executor": "executor",
            "finish": END,
        },
    )
    graph.add_edge("scientist", "orchestrator")
    graph.add_edge("general", "orchestrator")
    graph.add_edge("executor", "orchestrator")
    return graph.compile()


def make_video(
    experiments: list[dict[str, object]],
    dialogue: list[dict[str, str]],
    output_path: Path,
) -> None:
    best = max(experiments, key=lambda item: float(item["enhancement"]))
    trace = np.load(str(best["trace_file"]))
    camera = plt.imread(str(best["focused_image"]))
    events = ["USER: Focus the beam autonomously."]
    for turn in dialogue:
        text = turn["text"].replace("\n", " ")
        events.append(f"{turn['role'].upper()}: {text[:105]}")
    frames = 120
    fig = plt.figure(figsize=(12.8, 7.2), facecolor="#101418")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1, 1])
    ax_terminal = fig.add_subplot(grid[:, 0])
    ax_trace = fig.add_subplot(grid[0, 1])
    ax_camera = fig.add_subplot(grid[1, 1])
    fig.subplots_adjust(
        left=0.04, right=0.98, top=0.88, bottom=0.05, wspace=0.24, hspace=0.38
    )

    def update(frame: int):
        progress = (frame + 1) / frames
        shown_events = max(1, int(progress * len(events)))
        shown_trace = max(1, int(progress * len(trace)))
        ax_terminal.clear()
        ax_trace.clear()
        ax_camera.clear()
        ax_terminal.set_facecolor("#101418")
        ax_terminal.axis("off")
        ax_terminal.set_title(
            "Agent decision and tool call", color="white", loc="left", fontsize=14
        )
        wrapped = "\n\n".join(
            "\n".join(textwrap.wrap(event, width=62))
            for event in events[:shown_events][-7:]
        )
        ax_terminal.text(
            0.02,
            0.96,
            wrapped,
            va="top",
            ha="left",
            color="#d7e0e7",
            fontsize=8.5,
            family="monospace",
            wrap=True,
            transform=ax_terminal.transAxes,
        )
        ax_trace.plot(trace[:shown_trace], color="#ef6c35", linewidth=2)
        ax_trace.set_title("Focusing tool search", color="white")
        ax_trace.set_xlabel("iteration")
        ax_trace.set_ylabel("enhancement")
        ax_trace.set_xlim(0, len(trace))
        ax_trace.set_ylim(0, max(2.0, float(trace.max()) * 1.08))
        ax_trace.grid(alpha=0.2)
        ax_camera.imshow(camera)
        ax_camera.axis("off")
        ax_camera.set_title(
            f"Best camera image, eta={float(best['enhancement']):.2f}", color="white"
        )
        fig.suptitle("Homework 2 - autonomous focusing run", color="white", fontsize=16)
        return []

    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    animation = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
    writer = FFMpegWriter(
        fps=10, codec="libx264", bitrate=1800, extra_args=["-pix_fmt", "yuv420p"]
    )
    animation.save(output_path, writer=writer, dpi=100)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "task2")
    parser.add_argument("--video", type=Path, default=ROOT / "task2_demo.mp4")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "model_cache")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = LocalInstructionModel(args.model_id, args.cache_dir)
    tool = FocusingTool(args.output_dir)
    plans = [
        Settings(6, 2.5, 4.0, 12.0, 8, 180, 1),
        Settings(4, 2.0, 4.0, 12.0, 8, 260, 1),
        Settings(3, 1.5, 3.0, 11.0, 12, 360, 2),
        Settings(2, 1.25, 3.0, 11.0, 12, 520, 3),
    ]
    initial: GraphState = {
        "dialogue": [
            {
                "role": "user",
                "text": "Focus the beam. Choose all settings and run the experiments autonomously.",
            }
        ],
        "experiments": [],
        "remaining": [asdict(plan) for plan in plans],
        "pending": {},
        "decision_count": 0,
        "general_used": False,
    }
    final = build_graph(model, tool).invoke(initial)
    (args.output_dir / "experiments.json").write_text(
        json.dumps(final["experiments"], indent=2), encoding="utf-8"
    )
    (args.output_dir / "dialogue.json").write_text(
        json.dumps(final["dialogue"], indent=2), encoding="utf-8"
    )
    make_video(final["experiments"], final["dialogue"], args.video)
    print("Saved:", args.output_dir / "experiments.json")
    print("Saved:", args.video)


if __name__ == "__main__":
    main()
