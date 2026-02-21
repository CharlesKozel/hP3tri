from interfaces.body_plan import BodyPlanProvider, GrowthRequest
from interfaces.brain import OrganismView


class StubBodyPlan(BodyPlanProvider):
    def query_growth(
        self,
        organism: OrganismView,
        border_empty_neighbors: list[tuple[int, int]],
        genome_data: dict,
    ) -> list[GrowthRequest]:
        return []
