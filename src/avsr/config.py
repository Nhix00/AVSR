"""Loads config.yaml and exposes a typed cfg singleton."""
import yaml
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


class _Config:
    """Recursively wraps a dict so values are accessible as attributes."""

    def __init__(self, mapping: dict, _top_level: bool = True):
        for key, value in mapping.items():
            if isinstance(value, dict):
                # At root level, always wrap dicts as _Config for section access.
                # At deeper levels, only wrap if they contain nested dicts;
                # otherwise keep as plain dict so .items(), 'in', [] all work.
                if _top_level:
                    object.__setattr__(self, key, _Config(value, _top_level=False))
                else:
                    # Leaf dicts (all primitive values) are kept as plain Python dicts
                    # so callers can use .items(), 'in', and [] without _Config wrapping.
                    # Structural assumption: a nested dict is a "section" only if it
                    # contains at least one dict value itself.
                    if any(isinstance(v, dict) for v in value.values()):
                        object.__setattr__(self, key, _Config(value, _top_level=False))
                    else:
                        object.__setattr__(self, key, value)
            else:
                object.__setattr__(self, key, value)

    def __setattr__(self, key, value):
        raise AttributeError("cfg is read-only. Do not mutate configuration at runtime.")

    def __repr__(self):
        return f"Config({vars(self)})"


def _load() -> _Config:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.yaml not found at {_CONFIG_PATH}")
    with open(_CONFIG_PATH, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return _Config(raw)


cfg: _Config = _load()
