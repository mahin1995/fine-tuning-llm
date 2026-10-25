"""CrewPlanner: a one-agent crew that splits a request into a Plan.

Returns the raw task output; the flow validates it (domain/validation.py, domain/planning.py).
Like CrewProposer, a fresh agent and crew are built for every request, and any failure inside
CrewAI becomes a PlannerError, which the flow counts as a failed planning attempt.
"""
from crewai import Agent, Crew, Process, Task

from ops_crew import schemas
from ops_crew.crew.config import PLANNER_AGENT, PLANNER_TASK, CrewConfig
from ops_crew.crew.llm import LLMFactory
from ops_crew.crew.proposer import quote_user_text
from ops_crew.domain.ports import PlannerError


class CrewPlanner:
    def __init__(self, config: CrewConfig, llm_factory: LLMFactory, verbose: bool = False):
        self._config = config
        self._llms = llm_factory
        self._verbose = verbose

    def build_crew(self) -> Crew:
        spec = self._config.planner
        agent = Agent(
            role=spec.agent.role,
            goal=spec.agent.goal,
            backstory=spec.agent.backstory,
            llm=self._llms.for_agent(PLANNER_AGENT),
            tools=[],
            max_iter=spec.agent.max_iter,
            max_execution_time=spec.agent.max_execution_time,
            max_retry_limit=0,         # retries are owned by the deterministic flow
            allow_delegation=False,
            verbose=self._verbose,
        )
        task = Task(
            name=PLANNER_TASK,
            description=spec.task.description,
            expected_output=spec.task.expected_output,
            agent=agent,
            output_pydantic=getattr(schemas, spec.task.output),
        )
        return Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=self._verbose)

    def plan(self, request: str, correlation_id: str) -> str:
        try:
            output = self.build_crew().kickoff(inputs={"request": quote_user_text(request)})
        except Exception as e:
            raise PlannerError(f"{type(e).__name__}: {e}") from e
        task_output = output.tasks_output[0]
        return task_output.pydantic.model_dump_json() if task_output.pydantic is not None else task_output.raw
