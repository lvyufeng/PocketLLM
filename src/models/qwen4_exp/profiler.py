"""Hierarchical profiler for Qwen4-Exp runtime.

Usage:
    from src.models.qwen4_exp.profiler import Profiler

    prof = Profiler(enabled=True)
    with prof.scope("forward"):
        with prof.scope("attention"):
            ...

    prof.report()  # prints tree of accumulated times
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class ScopeStats:
    """Accumulated stats for one named scope."""
    count: int = 0
    total_s: float = 0.0
    children: dict[str, ScopeStats] = field(default_factory=dict)


class Profiler:
    """Lightweight hierarchical timer with CUDA sync."""

    def __init__(self, enabled: bool = True, device=None):
        self.enabled = enabled
        self.device = device
        self.root = ScopeStats()
        self._stack: list[tuple[str, ScopeStats, float]] = []

    @contextmanager
    def scope(self, name: str):
        """Enter a named timing scope."""
        if not self.enabled:
            yield
            return

        # Navigate to current scope
        parent = self.root
        for scope_name, _, _ in self._stack:
            parent = parent.children.setdefault(scope_name, ScopeStats())

        # Create/get child
        stats = parent.children.setdefault(name, ScopeStats())

        # Sync and start
        if self.device is not None and self.device.type == "cuda":
            import torch
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()

        self._stack.append((name, stats, t0))
        try:
            yield
        finally:
            self._stack.pop()
            if self.device is not None and self.device.type == "cuda":
                import torch
                torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - t0
            stats.count += 1
            stats.total_s += elapsed

    def reset(self):
        """Clear all accumulated stats."""
        self.root = ScopeStats()
        self._stack.clear()

    def report(self, min_percent: float = 0.5) -> str:
        """Generate a tree report of accumulated times.

        Args:
            min_percent: Hide children contributing < this % of parent time.
        """
        if not self.enabled or not self.root.children:
            return "(profiler disabled or no data)"

        lines = []

        def walk(node: ScopeStats, name: str, indent: int, parent_s: float):
            # Compute self time (total - children)
            child_s = sum(c.total_s for c in node.children.values())
            self_s = node.total_s - child_s

            pct = 100.0 * node.total_s / parent_s if parent_s > 0 else 0.0
            avg_ms = 1000.0 * node.total_s / node.count if node.count > 0 else 0.0

            prefix = "  " * indent
            lines.append(
                f"{prefix}{name:30s}  {node.count:6d} calls  "
                f"{node.total_s:7.3f}s ({pct:5.1f}%)  "
                f"{avg_ms:7.2f} ms/call"
            )

            # Show self time if significant children exist
            if child_s > 0.001 and self_s / node.total_s > 0.01:
                self_pct = 100.0 * self_s / node.total_s
                lines.append(
                    f"{prefix}  {'<self>':28s}          "
                    f"{self_s:7.3f}s ({self_pct:5.1f}%)"
                )

            # Recurse into children (sorted by time desc)
            for child_name, child_stats in sorted(
                node.children.items(), key=lambda kv: kv[1].total_s, reverse=True
            ):
                if 100.0 * child_stats.total_s / node.total_s >= min_percent:
                    walk(child_stats, child_name, indent + 1, node.total_s)

        # Fake root for top-level scopes
        if self.root.children:
            total = sum(c.total_s for c in self.root.children.values())
            lines.append(f"{'=== Profiler Report ===':<50s}  Total: {total:.3f}s")
            for name, stats in sorted(
                self.root.children.items(), key=lambda kv: kv[1].total_s, reverse=True
            ):
                walk(stats, name, 0, total)

        return "\n".join(lines)
