from __future__ import annotations
from Agents.Core.agent_engine import AgentEngine
class Orchestrator:
    def __init__(self, engine: AgentEngine | None = None):
        self.engine = engine or AgentEngine()
    def run_goal(self, goal: str, **kw):
        return self.engine.run(goal)
