"""CrewProposer: runs the 3-agent sequential crew and returns raw task outputs.

A fresh crew (agents, tools, trace) is built for every request, so no memory or state
leaks between users. Any failure inside CrewAI (provider error, conversion failure,
timeout) becomes a ProposerError, which the flow counts as a failed attempt.
"""
from crewai import Agent, Crew, Process, Task

from ops_crew import schemas
from ops_crew.crew.config import CrewConfig
from ops_crew.crew.llm import LLMFactory
from ops_crew.crew.tools import ReadOnlyDataset, ToolTrace, build_tools
from ops_crew.domain.ports import ProposerError
from ops_crew.schemas import CrewRun

_TAGS = ("user_request", "tool_data")


def quote_user_text(text: str) -> str:
    """Neutralise our delimiter tags inside user text so it can't close the data block."""
    for tag in _TAGS:
        text = text.replace(f"</{tag}>", f"<\\/{tag}>").replace(f"<{tag}>", f"<\\{tag}>")
    return text


class CrewProposer:
    def __init__(self, config: CrewConfig, llm_factory: LLMFactory, dataset: ReadOnlyDataset, verbose: bool = False):
        self._config = config
        self._llms = llm_factory
        self._dataset = dataset
        self._verbose = verbose

    def build_crew(self, trace: ToolTrace) -> Crew:
        agents = {
            name: Agent(
                role=spec.role,
                goal=spec.goal,
                backstory=spec.backstory,
                llm=self._llms.for_agent(name),
                tools=build_tools(spec.tools, name, self._dataset, trace),
                max_iter=spec.max_iter,
                max_execution_time=spec.max_execution_time,
                max_retry_limit=0,         # retries are owned by the deterministic flow
                allow_delegation=False,    # sequential, no agent can hand work to another
                verbose=self._verbose,
            )
            for name, spec in self._config.agents.items()
        }
        tasks: dict[str, Task] = {}
        for name, spec in self._config.tasks.items():
            tasks[name] = Task(
                name=name,
                description=spec.description,
                expected_output=spec.expected_output,
                agent=agents[spec.agent],
                context=[tasks[c] for c in spec.context] or None,
                output_pydantic=getattr(schemas, spec.output),
            )
        return Crew(agents=list(agents.values()), tasks=list(tasks.values()), process=Process.sequential,
                    verbose=self._verbose)

    def propose(self, request: str, correlation_id: str) -> CrewRun:
        trace = ToolTrace()
        try:
            output = self.build_crew(trace).kickoff(inputs={"request": quote_user_text(request)})
        except Exception as e:
            raise ProposerError(f"{type(e).__name__}: {e}") from e
        outputs = {
            t.name: t.pydantic.model_dump_json() if t.pydantic is not None else t.raw
            for t in output.tasks_output
        }
        return CrewRun(outputs=outputs, tool_calls=trace.records)
