from interfaces.brain import BrainOutput, BrainProvider, OrganismView, SensorInputs


class StubBrain(BrainProvider):
    def __init__(self, move_direction: int = 0) -> None:
        self._move_direction = move_direction

    def evaluate(
        self,
        organism: OrganismView,
        sensors: SensorInputs,
        genome_data: dict,
    ) -> BrainOutput:
        if organism.locomotion_power > 0:
            return BrainOutput(move_direction=self._move_direction)
        return BrainOutput()
