"""Skill-as-Markdown loader for Covenant Hedge Fund analysts.

Skills are structured Markdown files with YAML frontmatter that encode
analytical workflows as checklists. They are discovered by scanning the
skills/ directory and injected into LLM system prompts when the analyst
name matches the skill's analyst list.

Frontmatter schema:
    ---
    name: skill-name
    description: What this skill does
    analysts: [buffett, graham, ...]
    ---

The Markdown body below the frontmatter is the skill content -- a
human-readable, LLM-usable checklist that the analyst follows during
analysis.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class Skill:
    """A single analytical skill loaded from a Markdown file."""
    name: str
    description: str
    analysts: list[str]
    content: str
    source_path: str = ""


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from Markdown text.

    Expects the file to start with '---', followed by YAML key-value
    pairs, closed by another '---'. Returns (metadata_dict, body).

    Uses yaml.safe_load when available, falls back to simple line
    parsing for the three expected keys (name, description, analysts).
    """
    stripped = text.strip()
    if not stripped.startswith("---"):
        return {}, text

    # Split on the second '---' delimiter
    parts = stripped.split("---", 2)
    if len(parts) < 3:
        return {}, text

    raw_meta = parts[1].strip()
    body = parts[2].strip()

    # Try yaml.safe_load first
    try:
        import yaml
        meta = yaml.safe_load(raw_meta)
        if isinstance(meta, dict):
            return meta, body
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: line-by-line parsing
    meta: dict = {}
    for line in raw_meta.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()

        if key == "analysts":
            # Parse [a, b, c] list syntax
            if value.startswith("[") and value.endswith("]"):
                items = value[1:-1].split(",")
                meta[key] = [item.strip().strip("'\"") for item in items if item.strip()]
            else:
                meta[key] = [value]
        else:
            meta[key] = value

    return meta, body


def load_skills(skills_dir: str = "skills/") -> list[Skill]:
    """Scan a directory for .md skill files and return parsed Skills.

    Skips files that lack valid frontmatter (name + analysts fields).
    """
    skills_path = pathlib.Path(skills_dir)
    if not skills_path.is_dir():
        return []

    skills: list[Skill] = []
    for md_file in sorted(skills_path.glob("*.md")):
        try:
            text = md_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        meta, body = _parse_frontmatter(text)

        name = meta.get("name", "")
        analysts = meta.get("analysts", [])
        description = meta.get("description", "")

        if not name or not analysts:
            continue

        # Normalize analyst names to lowercase
        if isinstance(analysts, str):
            analysts = [analysts]
        analysts = [a.lower().strip() for a in analysts]

        skills.append(Skill(
            name=name,
            description=description,
            analysts=analysts,
            content=body,
            source_path=str(md_file),
        ))

    return skills


def get_skills_for_analyst(
    analyst_name: str,
    skills: Sequence[Skill],
) -> list[Skill]:
    """Return skills whose analyst list includes the given analyst name."""
    name = analyst_name.lower().strip()
    return [s for s in skills if name in s.analysts]


def format_skills_prompt(skills: Sequence[Skill]) -> str:
    """Format matched skills as a prompt appendix for LLM injection.

    Returns an empty string if no skills are provided, so callers can
    unconditionally append the result.
    """
    if not skills:
        return ""

    parts = ["\n\n--- ANALYTICAL SKILLS ---"]
    for skill in skills:
        parts.append(f"\n### {skill.name}")
        if skill.description:
            parts.append(f"_{skill.description}_")
        parts.append(skill.content)

    return "\n".join(parts)
