"""Compatibility alias for the shared TAC lifecycle module."""
import sys
from sead.environments.services import tac
sys.modules[__name__] = tac
