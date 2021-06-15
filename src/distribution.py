from typing import Generic, TypeVar, List, Tuple

from numpy.typing import ArrayLike
import numpy as np

T = TypeVar("T")


class Distribution(Generic[T]):
    dist: List[Tuple[T, float]]
    cum: ArrayLike

    def check_dist(self) -> None:
        assert sum(p for _, p in self.dist) == 1.0, f"bad distribution"

    def __init__(self, dist: List[Tuple[T, float]]):
        self.dist = [(t, p) for t, p in dist if p > 0]
        self.check_dist()
        self.cum = np.cumsum([p for _, p in self.dist])

    def prob(self, t: T) -> float:
        for _t, p in self.dist:
            if _t == t:
                return p
        return 0

    def sample(self) -> T:
        r = np.random.rand()
        i = np.argmax([self.cum > r])
        return self.dist[i][0]
