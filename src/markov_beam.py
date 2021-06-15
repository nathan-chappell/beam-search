from typing import Generic, TypeVar, List, TypeVar, Tuple, Dict, cast, Optional
from math import prod
import numpy as np

from numpy.typing import ArrayLike

StateT = TypeVar("StateT")

from beam import AbstractBeamSearch, BeamNode
from distribution import Distribution


class MarkovChain(Generic[StateT]):
    transition_matrix: Dict[StateT, Distribution[StateT]]
    current_state: StateT
    _n: int
    _indices: List[int]  # another type lie

    def __init__(
        self,
        initial_state: StateT,
        transition_matrix: Dict[StateT, Distribution[StateT]],
    ):
        self.current_state = initial_state
        self.transition_matrix = transition_matrix
        self._n = len(transition_matrix.keys())
        self._indices = cast(List[int], np.arange(self._n))

    def transition_prob(self, _from: StateT, _to: StateT) -> float:
        dist = self.transition_matrix[_from]
        return dist.prob(_to)

    def forward(self) -> None:
        r = np.random.rand()
        dist = self.transition_matrix[self.current_state]
        self.current_state = dist.sample()


class MarkovBeam(Generic[StateT], AbstractBeamSearch[StateT, Distribution[StateT]]):
    mc: MarkovChain[StateT]

    def __init__(self, mc: MarkovChain[StateT], width: int = 5):
        super().__init__(mc.current_state, width)
        self.mc = mc

    def _forward(self, state: StateT) -> Distribution[StateT]:
        return self.mc.transition_matrix[state]

    def _score(
        self, node: BeamNode[StateT, Distribution[StateT]]
    ) -> List[Tuple[StateT, float]]:
        return [
            (state, node.score * next_prob)
            for state, next_prob in cast(Distribution[StateT], node.result).dist
        ]


#
# Here's an awful diagram of a markov chain for which a greedy strategy will
# fail to get the most likely path of length three (which must go from a -> f)
#
#        .6
#      b - e
#  .6 / \   \ 1
# -> a   \.4 \_ f
#  .4 \   \  /1
#      c - d
#        1

if __name__ == "__main__":
    # and here's that ugly mother
    states = ["a", "b", "c", "d", "e", "f"]
    tm: Dict[str, Distribution[str]] = {
        "a": Distribution([("b", 0.6), ("c", 0.4)]),
        "b": Distribution([("d", 0.4), ("e", 0.6)]),
        "c": Distribution([("e", 1)]),
        "d": Distribution([("f", 1)]),
        "e": Distribution([("f", 1)]),
        "f": Distribution([("f", 1)]),
    }

    def test_beam(width):
        print(f"\n**** Testing beam: width: {width}")
        mc = MarkovChain("a", tm)
        bs = MarkovBeam(mc, width=width)
        for i in range(3):
            bs.forward()
        for i, h in enumerate(bs.beam):
            print(f"---- beam {i} ----")
            print(list(map(lambda b: (b.state, b.score), h.path())))

    for width in range(1, 4):
        test_beam(width)
