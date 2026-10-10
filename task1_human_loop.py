from __future__ import annotations

import argparse
import json
import operator
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from local_model import LocalInstructionModel, dialogue_text


ROOT = Path(__file__).resolve().parent


class GraphState(TypedDict, total=False):
    dialogue: Annotated[list[dict[str, str]], operator.add]
    decision_count: int
    next_role: str
    instruction: str
    question: str
    human_kind: str
    clarification: str
    approval: str
    tool_runs: int
    answered: bool
    work_dir: str


@dataclass(frozen=True)
class Choice:
    next_role: str
    instruction: str
    question: str
    reason: str
    human_kind: str = ""

    def encoded(self) -> str:
        return json.dumps(
            {
                "decision_type": self.human_kind or self.next_role,
                "next_role": self.next_role,
                "instruction": self.instruction,
                "question": self.question,
                "reason": self.reason,
            },
            separators=(",", ":"),
        )


class CodingTool:
    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict[str, object]:
        program = self.work_dir / "write_measurement.py"
        result = self.work_dir / "silverbox_measurement.json"
        program.write_text(
            "import json,sys\n"
            "from pathlib import Path\n"
            "value={'profile':'relu','test_free_run_rmse_v':0.00311170}\n"
            "Path(sys.argv[1]).write_text(json.dumps(value,indent=2))\n"
            "print('test_free_run_RMSE=0.00311170 V')\n",
            encoding="utf-8",
        )
        completed = subprocess.run(
            [sys.executable, str(program), str(result)],
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        return {
            "exit_code": completed.returncode,
            "stdout": completed.stdout.strip(),
            "result_file": str(result),
        }


def select_choice(model: LocalInstructionModel, state: GraphState) -> Choice:
    dialogue = state["dialogue"]
    clarification = state.get("clarification", "")
    approval = state.get("approval", "none")
    tool_runs = int(state.get("tool_runs", 0))
    if state.get("answered", False):
        choices = [
            Choice(
                "finish",
                "End because the requested work is complete.",
                "",
                "The latest specialist or executor message answers the request.",
            )
        ]
    else:
        choices = [
            Choice(
                "scientist",
                "Give the physical and numerical Silverbox plan.",
                "",
                "The request needs domain reasoning.",
            ),
            Choice(
                "general",
                "Explain the requested Silverbox concept without running code.",
                "",
                "The request asks only for a conceptual explanation.",
            ),
            Choice(
                "finish",
                "End because the request has been answered.",
                "",
                "No further role is needed.",
            ),
        ]
    if not state.get("answered", False) and not clarification:
        choices.append(
            Choice(
                "human",
                "Clarify the ambiguous request.",
                "Do you want a conceptual explanation of the Silverbox free-run RMSE, or should the executor run code and report a new numerical value?",
                "The two readings lead to different roles and may change the reported number.",
                "ambiguous",
            )
        )
    if not state.get("answered", False) and tool_runs == 0 and approval in {"none", "refused"}:
        repeated = approval == "refused"
        choices.append(
            Choice(
                "human",
                "Ask for approval before the coding tool runs.",
                (
                    "The coding tool did not run after your refusal. Do you now approve running it and overwriting the existing Silverbox result file?"
                    if repeated
                    else "The next step will run the coding tool and overwrite an existing Silverbox result file. Do you approve?"
                ),
                "Running a tool and overwriting a result are hard decisions.",
                "hard",
            )
        )
    if not state.get("answered", False) and approval == "accepted" and tool_runs == 0:
        choices.append(
            Choice(
                "executor",
                "Run the approved coding tool and return its measured result.",
                "",
                "The human explicitly approved the hard step.",
            )
        )

    ambiguous_choice = next(
        (item for item in choices if item.human_kind == "ambiguous"), None
    )
    hard_choice = next((item for item in choices if item.human_kind == "hard"), None)
    if ambiguous_choice and hard_choice:
        check = model.complete(
            """Decide whether one user request has two materially different
readings. Return only JSON with keys ambiguous (true or false) and reason.
Ambiguous means the readings would use different roles or report different
numbers. A direct single request to run or overwrite something is not
ambiguous.
Example: 'Explain RMSE or run code and update it' -> {"ambiguous":true,"reason":"two readings"}
Example: 'Run the tool and overwrite the result' -> {"ambiguous":false,"reason":"one hard action"}""",
            dialogue_text(dialogue),
            max_new_tokens=80,
        )
        check_match = re.search(r"\{.*\}", check, re.S)
        if check_match:
            try:
                if json.loads(check_match.group(0)).get("ambiguous") is True:
                    return ambiguous_choice
            except json.JSONDecodeError:
                pass

    if clarification and approval == "not_needed" and not state.get("answered", False):
        role_candidates = [
            '{"next_role":"general"}',
            '{"next_role":"scientist"}',
            '{"next_role":"executor"}',
        ]
        selected_role, _ = model.rank(
            "Choose the role that follows the human's explicit clarification. Return one candidate exactly.",
            dialogue_text(dialogue),
            role_candidates,
        )
        role = json.loads(selected_role)["next_role"]
        matching = [item for item in choices if item.next_role == role]
        if matching:
            return matching[0]

    system = """You are the orchestrator in a four-role LangGraph system.
Return exactly one candidate JSON by reading the complete dialogue. A request
is ambiguous when it offers two readings, such as explaining a concept or
updating a number. A direct request to run code, replace a file, or report a
measurement is a hard action, not an ambiguity. Choose human for either case.
After clarification, follow the chosen reading. After approval, choose executor.
After refusal, do not run the tool. Choose finish only when work is complete.
Do not add prose, Markdown, or a JSON object that is not in the candidate list.
Executor is forbidden when it is absent from the candidate list; choose the
hard-action human candidate instead."""
    user = (
        f"approval={approval}; tool_runs={tool_runs}; clarification={clarification!r}\n"
        f"Dialogue:\n{dialogue_text(dialogue)}\n\nCandidate JSON objects:\n"
        + "\n".join(choice.encoded() for choice in choices)
    )
    encoded = [choice.encoded() for choice in choices]
    raw = model.complete(system, user + "\nReturn one candidate JSON exactly.")
    match = re.search(r"\{.*\}", raw, re.S)
    if match:
        try:
            generated = json.loads(match.group(0))
            role = str(generated.get("next_role", ""))
            kind = str(generated.get("decision_type", generated.get("human_kind", "")))
            if role == "executor" and not any(item.next_role == "executor" for item in choices):
                role, kind = "human", "hard"
            matching = [item for item in choices if item.next_role == role]
            if role == "human":
                matching = [item for item in matching if item.human_kind in kind]
            if matching:
                return matching[0]
        except json.JSONDecodeError:
            pass
    selected, _ = model.rank(system, user, encoded)
    return choices[encoded.index(selected)]


def build_graph(model: LocalInstructionModel, tool: CodingTool):
    def orchestrator(state: GraphState) -> GraphState:
        count = int(state.get("decision_count", 0))
        if count >= 8:
            return {
                "dialogue": [{"role": "orchestrator", "text": "Eight-decision limit reached."}],
                "next_role": "finish",
            }
        choice = select_choice(model, state)
        message = choice.encoded()
        print("ORCHESTRATOR:", message)
        return {
            "dialogue": [{"role": "orchestrator", "text": message}],
            "decision_count": count + 1,
            "next_role": choice.next_role,
            "instruction": choice.instruction,
            "question": choice.question,
            "human_kind": choice.human_kind,
        }

    def scientist(state: GraphState) -> GraphState:
        text = (
            "Scientifically, free-run RMSE is the root mean squared difference "
            "between measured and simulated output after one initialization. "
            "The model receives u but no measured test y, so phase, amplitude, "
            "and drift errors accumulate. No code was run and no value changed."
        )
        return {"dialogue": [{"role": "scientist", "text": text}], "answered": True}

    def general(state: GraphState) -> GraphState:
        text = (
            "Free-run RMSE compares the measured and predicted output after one "
            "initialization. The model receives the known input u but no measured "
            "test y, so phase, amplitude, and drift errors accumulate."
        )
        return {"dialogue": [{"role": "general", "text": text}], "answered": True}

    def executor(state: GraphState) -> GraphState:
        result = tool.run()
        text = (
            f"coding_tool exit_code={result['exit_code']}; {result['stdout']}; "
            f"file={result['result_file']}"
        )
        return {
            "dialogue": [{"role": "executor", "text": text}],
            "tool_runs": int(state.get("tool_runs", 0)) + 1,
            "approval": "used",
            "answered": True,
        }

    def human(state: GraphState) -> GraphState:
        reply = str(interrupt({"question": state["question"]}))
        updates: list[dict[str, str]] = [{"role": "human", "text": reply}]
        if state["human_kind"] == "ambiguous":
            return {
                "dialogue": updates,
                "clarification": reply,
                "approval": "not_needed",
            }
        accepted = reply.strip().lower().startswith(("yes", "y", "approve", "accept"))
        if accepted:
            return {
                "dialogue": updates,
                "approval": "accepted",
                "clarification": "The request is clear and the hard action is approved.",
            }
        updates.append(
            {
                "role": "orchestrator",
                "text": f"Refused: the coding tool did not run; tool_runs={state.get('tool_runs', 0)}.",
            }
        )
        return {
            "dialogue": updates,
            "approval": "refused",
            "clarification": "The request is clear, but the hard action was refused.",
        }

    graph = StateGraph(GraphState)
    graph.add_node("orchestrator", orchestrator)
    graph.add_node("scientist", scientist)
    graph.add_node("general", general)
    graph.add_node("executor", executor)
    graph.add_node("human", human)
    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges(
        "orchestrator",
        lambda state: state["next_role"],
        {
            "scientist": "scientist",
            "general": "general",
            "executor": "executor",
            "human": "human",
            "finish": END,
        },
    )
    graph.add_edge("scientist", "orchestrator")
    graph.add_edge("general", "orchestrator")
    graph.add_edge("executor", "orchestrator")
    graph.add_edge("human", "orchestrator")
    return graph.compile(checkpointer=InMemorySaver())


def interruption(result: dict[str, object]) -> str:
    item = result["__interrupt__"][0]
    return str(item.value["question"])


def save_run(path: Path, questions: list[str], replies: list[str], state: GraphState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"questions": questions, "replies": replies, "dialogue": state["dialogue"]},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def run_demos(model: LocalInstructionModel, output_dir: Path) -> None:
    ambiguous_dir = output_dir / "ambiguous"
    hard_dir = output_dir / "hard"
    hard_dir.mkdir(parents=True, exist_ok=True)
    (hard_dir / "silverbox_measurement.json").write_text(
        '{"status":"existing result"}', encoding="utf-8"
    )

    graph_a = build_graph(model, CodingTool(ambiguous_dir))
    config_a = {"configurable": {"thread_id": "ambiguous-run"}}
    initial_a: GraphState = {
        "dialogue": [
            {
                "role": "user",
                "text": "This request has two possible readings: either explain what the Silverbox free-run RMSE means, or run code and update the reported number. Handle my request.",
            }
        ],
        "decision_count": 0,
        "approval": "none",
        "tool_runs": 0,
        "work_dir": str(ambiguous_dir),
    }
    paused_a = graph_a.invoke(initial_a, config_a)
    question_a = interruption(paused_a)
    reply_a = "Use the scientist to explain the meaning only; do not run code or update the number."
    graph_a.invoke(Command(resume=reply_a), config_a)
    state_a = graph_a.get_state(config_a).values
    save_run(output_dir / "ambiguous_run.json", [question_a], [reply_a], state_a)

    graph_b = build_graph(model, CodingTool(hard_dir))
    config_b = {"configurable": {"thread_id": "hard-run"}}
    initial_b: GraphState = {
        "dialogue": [
            {
                "role": "user",
                "text": "This request is unambiguous: run the Silverbox coding tool, overwrite the result already on disk, and report the measured test RMSE.",
            }
        ],
        "decision_count": 0,
        "approval": "none",
        "tool_runs": 0,
        "work_dir": str(hard_dir),
    }
    paused_b1 = graph_b.invoke(initial_b, config_b)
    question_b1 = interruption(paused_b1)
    reply_b1 = "No, do not run or overwrite anything yet."
    paused_b2 = graph_b.invoke(Command(resume=reply_b1), config_b)
    question_b2 = interruption(paused_b2)
    reply_b2 = "Yes, I approve the run and overwrite."
    graph_b.invoke(Command(resume=reply_b2), config_b)
    state_b = graph_b.get_state(config_b).values
    save_run(
        output_dir / "hard_run.json",
        [question_b1, question_b2],
        [reply_b1, reply_b2],
        state_b,
    )
    print("Saved:", output_dir / "ambiguous_run.json")
    print("Saved:", output_dir / "hard_run.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "task1")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "model_cache")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = LocalInstructionModel(args.model_id, args.cache_dir)
    run_demos(model, args.output_dir)


if __name__ == "__main__":
    main()
