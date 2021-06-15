from typing import Generic, TypeVar, Union, Optional, List, Tuple, Any, cast
from numpy.typing import ArrayLike
import numpy as np

StateT = TypeVar("StateT")
ResultT = TypeVar("ResultT")
T = TypeVar("T")
U = TypeVar("U")

MAX_PATH_LENGTH = 1000

class BeamNode(Generic[StateT, ResultT]):
    ParentT = Optional[BeamNode[T, U]]
    parent: ParentT[StateT, ResultT] = None
    state: Optional[StateT] = None
    result: Optional[ResultT] = None

    def __init__(self, state: StateT, parent: ParentT[StateT, ResultT] = None):
        self.state = state
        self.parent = parent

    def path(self) -> List[BeamNode[StateT,ResultT]]:
        path = [self]
        while path[0].parent is not None and len(path) < MAX_PATH_LENGTH:
            path.insert(0,path[0].parent)
        return path


def topk_indices(a: ArrayLike, k) -> ArrayLike:
    return np.stack(np.unravel_index(np.argsort(np.ravel(a))[..., :k]))


class AbstractBeamSearch(Generic[StateT, ResultT]):
    beam: List[BeamNode[StateT, ResultT]]
    width: int

    def __init__(self, width: int = 5, initial_state=None):
        self.beam = [BeamNode(initial_state)]
        self.width = width

    def _forward(self, state: Optional[StateT]) -> ResultT:
        """run the model against the current state"""
        raise NotImplemented

    def _score(self, result: Optional[ResultT]) -> List[Tuple[StateT, float]]:
        """create a distribution over next State"""
        raise NotImplemented

    def forward(self) -> None:
        for head in self.beam:
            head.result = self._forward(head.state)
        scoreList = [self._score(head.result) for head in self.beam]
        scoreMat = np.array([[score for _, score in score] for score in scoreList])
        # this is a blatant lie, but mypy lets us iterate the damn thing...
        indices = cast(List[List[int]], topk_indices(scoreMat, self.width))
        self.beam = [
            BeamNode(state=scoreList[head_i][next_j][0], parent=self.beam[head_i])
            for head_i, next_j in indices
        ]

