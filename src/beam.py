from typing import Generic, TypeVar, Union, Optional, List, Tuple, Any, cast, Dict
from numpy.typing import ArrayLike
import numpy as np

StateT = TypeVar("StateT")
ResultT = TypeVar("ResultT")
T = TypeVar("T")
U = TypeVar("U")

MAX_PATH_LENGTH = 1000


class BeamNode(Generic[StateT, ResultT]):
    ParentT = Optional["BeamNode[T, U]"]
    parent: "ParentT[StateT, ResultT]" = None
    state: StateT
    score: float
    result: Optional[ResultT] = None

    def __init__(
        self,
        state: StateT,
        score: float = 1.0,
        parent: "ParentT[StateT, ResultT]" = None,
    ):
        self.state = state
        self.parent = parent
        self.score = score

    def path(self) -> "List[BeamNode[StateT, ResultT]]":
        path = [self]
        while path[0].parent is not None and len(path) < MAX_PATH_LENGTH:
            path.insert(0, path[0].parent)
        return path


class AbstractBeamSearch(Generic[StateT, ResultT]):
    beam: List[BeamNode[StateT, ResultT]]
    width: int

    def __init__(self, initial_state: StateT, width: int = 5):
        self.beam = [BeamNode(initial_state)]
        self.width = width

    def _forward(self, state: StateT) -> ResultT:
        """run the model against the current state"""
        raise NotImplemented

    def _score(self, node: BeamNode[StateT, ResultT]) -> List[Tuple[StateT, float]]:
        """create a distribution over next State"""
        raise NotImplemented

    def forward(self) -> None:
        for head in self.beam:
            head.result = self._forward(head.state)
        scoreList = [self._score(head) for head in self.beam]
        scoreMat = [[score for _, score in score] for score in scoreList]
        indices = sorted(
            [
                (p, (i, j))
                for i, scores in enumerate(scoreMat)
                for j, p in enumerate(scores)
            ],
            key=lambda t: t[0],
            reverse=True,
        )
        self.beam = [
            BeamNode(
                state=scoreList[head_i][next_j][0], score=p, parent=self.beam[head_i]
            )
            for p, (head_i, next_j) in indices[: self.width]
        ]
