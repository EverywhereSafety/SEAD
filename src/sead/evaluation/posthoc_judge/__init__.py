"""Shared semantic-judge configuration, model client, projection and prompts."""

from .config import JudgeConfig, PosthocJudgeConfig, load_config

__all__ = ["JudgeConfig", "PosthocJudgeConfig", "load_config"]
