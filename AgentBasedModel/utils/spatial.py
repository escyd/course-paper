import math
from functools import lru_cache


@lru_cache(maxsize=4096)
def euclidean_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def transport_cost(agent_pos: tuple[int, int], market_pos: tuple[int, int], tau: float) -> float:
    return tau * euclidean_distance(agent_pos, market_pos)