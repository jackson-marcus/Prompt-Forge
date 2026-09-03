"""Edit operators: the moves a prompt hill-climb can make.

Each operator is a deterministic rewrite of one template. It applies only when
the template lacks the thing it adds - offering a format instruction to a prompt
that already has one is a wasted suite run - and every rewrite keeps the
``{input}`` placeholder where it was, so the result is still a valid template.

Operators are deliberately orthogonal in the simulator's feature space: the
format line avoids the word "only" so it does not also count as a constraint,
and the persona line matches no feature at all. That last one is on purpose.
"You are an expert ..." is one of the most common edits people make to a
prompt, and the simulator does not reward it, which makes it the control that
exposes an acceptance rule that mistakes noise for improvement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from promptforge.llm.simulate import prompt_features
from promptforge.registry.memento import content_hash

INPUT_PLACEHOLDER = "{input}"


@dataclass(frozen=True, slots=True)
class TaskContext:
    """What an operator may know about the task: its label set and a few dev-split
    exemplars. Test-split cases are never in here."""

    labels: tuple[str, ...]
    examples: tuple[tuple[str, str], ...] = ()  # (input, label)


@dataclass(frozen=True, slots=True)
class Candidate:
    """One proposed rewrite, content-addressed like a registry snapshot."""

    edit: str
    template: str
    content_hash: str

    @classmethod
    def create(cls, edit: str, template: str) -> Candidate:
        return cls(edit=edit, template=template, content_hash=content_hash(template))


def insert_before_input(template: str, line: str) -> str:
    """Add an instruction line immediately above the line that carries ``{input}``.

    Instructions read best before the thing they instruct about; if the template
    has no placeholder line, the instruction goes at the end.
    """
    lines = template.split("\n")
    for i in range(len(lines) - 1, -1, -1):
        if INPUT_PLACEHOLDER in lines[i]:
            return "\n".join([*lines[:i], line, *lines[i:]])
    return "\n".join([*lines, line])


class EditOperator(ABC):
    name: str

    @abstractmethod
    def applicable(self, template: str, features: dict[str, bool]) -> bool:
        """Whether this edit would change something the template does not already have."""

    @abstractmethod
    def apply(self, template: str, ctx: TaskContext) -> str:
        """Return the rewritten template."""

    def propose(self, template: str, ctx: TaskContext) -> Candidate | None:
        if not self.applicable(template, prompt_features(template)):
            return None
        rewritten = self.apply(template, ctx)
        if rewritten == template:
            return None
        return Candidate.create(self.name, rewritten)


class FormatSpec(EditOperator):
    """Tell the model to answer with the bare label. Without this an exact-match
    suite scores zero no matter how often the model is right."""

    name = "format_spec"

    def applicable(self, template: str, features: dict[str, bool]) -> bool:
        return not features["has_format_spec"]

    def apply(self, template: str, ctx: TaskContext) -> str:
        options = ", ".join(ctx.labels)
        return insert_before_input(template, f"Reply with just the label, one of: {options}.")


class Constraint(EditOperator):
    name = "constraint"

    def applicable(self, template: str, features: dict[str, bool]) -> bool:
        return not features["has_constraint"]

    def apply(self, template: str, ctx: TaskContext) -> str:
        return insert_before_input(template, "Do not explain your answer.")


class FewShot(EditOperator):
    """Show one worked example per label, mined from the dev split."""

    name = "few_shot"

    def applicable(self, template: str, features: dict[str, bool]) -> bool:
        return not features["has_fewshot"]

    def apply(self, template: str, ctx: TaskContext) -> str:
        if not ctx.examples:
            return template
        block = "\n".join(f"Example: '{text}' -> {label}" for text, label in ctx.examples)
        return insert_before_input(template, block)


class Persona(EditOperator):
    """Prepend an expert persona. Popular, and inert under the simulator."""

    name = "persona"
    line = "You are an expert annotator."

    def applicable(self, template: str, features: dict[str, bool]) -> bool:
        return not template.startswith(self.line)

    def apply(self, template: str, ctx: TaskContext) -> str:
        return f"{self.line}\n{template}"


def default_operators() -> list[EditOperator]:
    return [FormatSpec(), Constraint(), FewShot(), Persona()]
