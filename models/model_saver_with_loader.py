"""
models/model_saver.py
=====================
Auto-naming model saver/loader for DQNAgent, PPOAgent, SACAgent, TD3Agent
and RARL-style DQN protagonists.

Filename format
---------------
  saved_models/<agent>_<env>_<params>_trial<N>_run<R>_<YYYYMMDD_HHMMSS>.pth

Examples
--------
  dqn_adv_frac0.30_trial0_run0_20240115_143022.pth
  ppo_irr_B0.01_trial2_run1_20240115_150301.pth
  rarl_protagonist_adv_frac1.00_trial0_run0_20240115_160000.pth
  sac_adv_frac0.50_trial3_run2_20240115_161245.pth

Usage
-----
    from models.model_saver import (
        save_agent, load_agent, find_latest_model,
        make_run_tag, adv_tag, irr_tag,
    )

    # Save
    tag = make_run_tag(agent_name='ppo', env_tag='adv_frac0.30',
                       trial=i, run=run_idx)
    save_agent(agent, tag)

    # Load exact path
    agent = PPOAgent(n_observations=3, n_actions=9, device=device)
    load_agent(agent, 'saved_models/ppo_adv_frac0.30_trial0_run0_20240115_143022.pth')

    # Or load the newest checkpoint matching a prefix
    load_agent(agent, 'ppo_adv_frac0.30_trial0_run0')
"""

from __future__ import annotations

import glob
import os
from datetime import datetime
from typing import List, Optional

import torch

SAVE_DIR = "saved_models"


def _ensure_dir() -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)


# ── Tag helpers ────────────────────────────────────────────────────────────────

def adv_tag(adversarial_fraction: float) -> str:
    """adv_tag(0.3) -> 'adv_frac0.30'"""
    return f"adv_frac{adversarial_fraction:.2f}"


def irr_tag(irrational_B: float) -> str:
    """irr_tag(0.01) -> 'irr_B0.01'"""
    return f"irr_B{irrational_B}"


def make_run_tag(agent_name: str, env_tag: str,
                 trial: int, run: int = 0) -> str:
    """
    Build a unique filename stem (no extension, no directory).

    Parameters
    ----------
    agent_name : 'dqn' | 'ppo' | 'sac' | 'td3' | 'rarl_protagonist' | 'rarl_adversary'
    env_tag    : output of adv_tag() or irr_tag()
    trial      : trial index (outer loop)
    run        : run index within the trial (default 0)
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{agent_name}_{env_tag}_trial{trial}_run{run}_{ts}"


# ── Save dispatcher ───────────────────────────────────────────────────────────

def save_agent(agent, tag: str) -> str:
    """
    Save agent weights to saved_models/<tag>.pth.
    Dispatches on the agent class name so each type saves its relevant weights.
    Returns the full path of the saved file.
    """
    _ensure_dir()
    path = os.path.join(SAVE_DIR, tag + ".pth")
    cls = type(agent).__name__

    if cls == "DQNAgent":
        torch.save(agent.policy_net.state_dict(), path)

    elif cls == "PPOAgent":
        torch.save({
            "actor": agent.actor.state_dict(),
            "critic": agent.critic.state_dict(),
        }, path)

    elif cls == "SACAgent":
        torch.save({
            "actor": agent.actor.state_dict(),
            "q1": agent.q1.state_dict(),
            "q2": agent.q2.state_dict(),
        }, path)

    elif cls == "TD3Agent":
        torch.save({
            "q1": agent.q1.state_dict(),
            "q2": agent.q2.state_dict(),
        }, path)

    else:
        # Fallback: try policy_net, then actor, then raise
        if hasattr(agent, "policy_net"):
            torch.save(agent.policy_net.state_dict(), path)
        elif hasattr(agent, "actor"):
            payload = {"actor": agent.actor.state_dict()}
            if hasattr(agent, "critic"):
                payload["critic"] = agent.critic.state_dict()
            torch.save(payload, path)
        else:
            raise TypeError(f"save_agent: unknown agent type '{cls}'")

    print(f"  [saved] {path}")
    return path


# ── Load / discovery helpers ──────────────────────────────────────────────────

def list_saved_models(pattern: str = "*.pth") -> List[str]:
    """List saved checkpoints under SAVE_DIR matching a glob pattern."""
    _ensure_dir()
    paths = sorted(glob.glob(os.path.join(SAVE_DIR, pattern)))
    return paths


def find_latest_model(tag_or_pattern: str, save_dir: str = SAVE_DIR) -> str:
    """
    Resolve a checkpoint path.

    Accepted inputs
    ---------------
    - exact path to a .pth file
    - bare filename (with or without .pth)
    - tag prefix such as 'ppo_adv_frac0.30_trial0_run0'
    - glob pattern such as 'ppo_adv_frac0.30*.pth'

    Returns the most recently modified matching file.
    """
    _ensure_dir()

    # Exact path first
    if os.path.isfile(tag_or_pattern):
        return tag_or_pattern

    candidates = []

    # saved_models/<name>.pth or saved_models/<name>
    if not os.path.isabs(tag_or_pattern):
        base = os.path.join(save_dir, tag_or_pattern)
        if os.path.isfile(base):
            return base
        if os.path.isfile(base + ".pth"):
            return base + ".pth"

    # Glob handling / prefix handling
    pattern = tag_or_pattern
    if not pattern.endswith(".pth"):
        pattern = pattern + "*.pth"

    if not os.path.isabs(pattern):
        pattern = os.path.join(save_dir, pattern)

    candidates.extend(glob.glob(pattern))

    if not candidates:
        raise FileNotFoundError(
            f"No saved model matched '{tag_or_pattern}'. Looked under '{save_dir}'."
        )

    return max(candidates, key=os.path.getmtime)


def _load_state_dict(path: str, map_location=None):
    return torch.load(path, map_location=map_location)


# ── Load dispatcher ───────────────────────────────────────────────────────────

def load_agent(agent, model_path_or_tag: str,
               *,
               device=None,
               strict: bool = True,
               eval_mode: bool = True,
               sync_targets: bool = True) -> str:
    """
    Load a saved checkpoint into an existing agent instance.

    Parameters
    ----------
    agent : instantiated agent object
    model_path_or_tag : exact path, filename, prefix tag, or glob pattern
    device : torch device / map_location override. Defaults to agent.device if present
    strict : passed to load_state_dict(..., strict=strict)
    eval_mode : put loaded networks into eval() mode when possible
    sync_targets : copy online Q weights into target nets when present

    Returns
    -------
    The resolved checkpoint path that was loaded.
    """
    resolved_path = find_latest_model(model_path_or_tag)
    map_location = device if device is not None else getattr(agent, "device", None)
    checkpoint = _load_state_dict(resolved_path, map_location=map_location)
    cls = type(agent).__name__

    if cls == "DQNAgent":
        state = checkpoint["policy_net"] if isinstance(checkpoint, dict) and "policy_net" in checkpoint else checkpoint
        agent.policy_net.load_state_dict(state, strict=strict)
        if sync_targets and hasattr(agent, "target_net"):
            agent.target_net.load_state_dict(agent.policy_net.state_dict(), strict=strict)
        if eval_mode:
            agent.policy_net.eval()
            if hasattr(agent, "target_net"):
                agent.target_net.eval()

    elif cls == "PPOAgent":
        if not isinstance(checkpoint, dict) or "actor" not in checkpoint:
            raise ValueError("PPO checkpoint must contain at least an 'actor' key.")
        agent.actor.load_state_dict(checkpoint["actor"], strict=strict)
        if "critic" in checkpoint and hasattr(agent, "critic"):
            agent.critic.load_state_dict(checkpoint["critic"], strict=strict)
        if eval_mode:
            agent.actor.eval()
            if hasattr(agent, "critic"):
                agent.critic.eval()

    elif cls == "SACAgent":
        if not isinstance(checkpoint, dict) or "actor" not in checkpoint:
            raise ValueError("SAC checkpoint must contain 'actor', 'q1', and 'q2' keys.")
        agent.actor.load_state_dict(checkpoint["actor"], strict=strict)
        if "q1" in checkpoint:
            agent.q1.load_state_dict(checkpoint["q1"], strict=strict)
        if "q2" in checkpoint:
            agent.q2.load_state_dict(checkpoint["q2"], strict=strict)
        if sync_targets:
            if hasattr(agent, "q1_tgt"):
                agent.q1_tgt.load_state_dict(agent.q1.state_dict(), strict=strict)
            if hasattr(agent, "q2_tgt"):
                agent.q2_tgt.load_state_dict(agent.q2.state_dict(), strict=strict)
        if eval_mode:
            agent.actor.eval()
            if hasattr(agent, "q1"):
                agent.q1.eval()
            if hasattr(agent, "q2"):
                agent.q2.eval()
            if hasattr(agent, "q1_tgt"):
                agent.q1_tgt.eval()
            if hasattr(agent, "q2_tgt"):
                agent.q2_tgt.eval()

    elif cls == "TD3Agent":
        if not isinstance(checkpoint, dict) or "q1" not in checkpoint:
            raise ValueError("TD3 checkpoint must contain 'q1' and 'q2' keys.")
        agent.q1.load_state_dict(checkpoint["q1"], strict=strict)
        agent.q2.load_state_dict(checkpoint["q2"], strict=strict)
        if sync_targets:
            if hasattr(agent, "q1_tgt"):
                agent.q1_tgt.load_state_dict(agent.q1.state_dict(), strict=strict)
            if hasattr(agent, "q2_tgt"):
                agent.q2_tgt.load_state_dict(agent.q2.state_dict(), strict=strict)
        if eval_mode:
            agent.q1.eval()
            agent.q2.eval()
            if hasattr(agent, "q1_tgt"):
                agent.q1_tgt.eval()
            if hasattr(agent, "q2_tgt"):
                agent.q2_tgt.eval()

    else:
        # Fallback: if the class already exposes a compatible loader, use it.
        if hasattr(agent, "load_model") and callable(getattr(agent, "load_model")):
            agent.load_model(resolved_path)
        elif hasattr(agent, "policy_net"):
            agent.policy_net.load_state_dict(checkpoint, strict=strict)
            if eval_mode:
                agent.policy_net.eval()
        elif hasattr(agent, "actor") and isinstance(checkpoint, dict) and "actor" in checkpoint:
            agent.actor.load_state_dict(checkpoint["actor"], strict=strict)
            if eval_mode:
                agent.actor.eval()
        else:
            raise TypeError(f"load_agent: unknown agent type '{cls}'")

    print(f"  [loaded] {resolved_path}")
    return resolved_path


def load_latest_agent(agent, tag_prefix: str, **kwargs) -> str:
    """Convenience wrapper around load_agent(...) for prefix-based lookup."""
    return load_agent(agent, tag_prefix, **kwargs)
