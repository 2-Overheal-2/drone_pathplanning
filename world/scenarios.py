from __future__ import annotations

from typing import List

import numpy as np

from world.config import ArenaConfig
from world.objects import Obstacle


def _overlaps_xy(candidate: Obstacle, obstacles: List[Obstacle], clearance: float) -> bool:
    cmin, cmax = candidate.aabb

    for obstacle in obstacles:
        omin, omax = obstacle.aabb
        separated = (
            cmax[0] + clearance < omin[0]
            or cmin[0] - clearance > omax[0]
            or cmax[1] + clearance < omin[1]
            or cmin[1] - clearance > omax[1]
        )
        if not separated:
            return True

    return False


def _contains_point_xy(candidate: Obstacle, point, clearance: float) -> bool:
    x, y, _ = point
    minimum, maximum = candidate.aabb
    return (
        minimum[0] - clearance <= x <= maximum[0] + clearance
        and minimum[1] - clearance <= y <= maximum[1] + clearance
    )


def _uniform_size(rng: np.random.Generator, arena_axis: float, low_fraction: float, high_fraction: float, hard_min: float, hard_max: float) -> float:
    low = min(hard_min, max(0.15, arena_axis * low_fraction))
    high = min(hard_max, max(low, arena_axis * high_fraction))
    return float(rng.uniform(low, high)) if high > low else float(low)


class ScenarioGenerator:
    def generate(self, config: ArenaConfig, rng: np.random.Generator) -> List[Obstacle]:
        raise NotImplementedError


class EmptyScenario(ScenarioGenerator):
    def generate(self, config: ArenaConfig, rng: np.random.Generator) -> List[Obstacle]:
        return []


class CityScenario(ScenarioGenerator):
    """Упрощённый городской сценарий: здания + иногда колонны."""

    def generate(self, config: ArenaConfig, rng: np.random.Generator) -> List[Obstacle]:
        sx, sy, sz = config.size
        target_area = sx * sy * config.density
        occupied_area = 0.0
        result: List[Obstacle] = []

        attempts = 0
        max_attempts = max(2000, config.max_objects * 80)

        while (
            occupied_area < target_area
            and len(result) < config.max_objects
            and attempts < max_attempts
        ):
            attempts += 1

            make_column = rng.random() < 0.18

            if make_column:
                diameter = _uniform_size(rng, min(sx, sy), 0.025, 0.07, 0.25, 1.5)
                width = depth = diameter
                height = float(rng.uniform(max(0.8, sz * 0.15), max(0.81, sz * 0.75)))
                geometry = "cylinder"
                kind = "column"
                color = (0.62, 0.62, 0.62, 1.0)
            else:
                width = _uniform_size(rng, sx, 0.06, 0.20, 0.5, 7.0)
                depth = _uniform_size(rng, sy, 0.08, 0.25, 0.4, 7.0)
                height = float(rng.uniform(max(0.8, sz * 0.20), max(0.81, sz * 0.90)))
                geometry = "box"
                kind = "building"
                color = (0.45, 0.48, 0.52, 1.0)

            if width >= sx or depth >= sy:
                continue

            x = float(rng.uniform(width / 2.0, sx - width / 2.0))
            y = float(rng.uniform(depth / 2.0, sy - depth / 2.0))
            z = height / 2.0

            obstacle = Obstacle(
                kind=kind,
                geometry=geometry,
                position=np.asarray([x, y, z], dtype=float),
                dimensions=np.asarray([width, depth, height], dtype=float),
                color=color,
            )

            if _contains_point_xy(obstacle, config.start, 0.8):
                continue
            if _contains_point_xy(obstacle, config.goal, 0.8):
                continue
            if _overlaps_xy(obstacle, result, config.min_clearance):
                continue

            result.append(obstacle)
            occupied_area += obstacle.footprint_area

        return result


class ForestScenario(ScenarioGenerator):
    """Лес: вертикальные цилиндрические стволы разной высоты/толщины."""

    def generate(self, config: ArenaConfig, rng: np.random.Generator) -> List[Obstacle]:
        sx, sy, sz = config.size
        target_area = sx * sy * config.density
        occupied_area = 0.0
        result: List[Obstacle] = []

        attempts = 0
        max_attempts = max(3000, config.max_objects * 100)

        while (
            occupied_area < target_area
            and len(result) < config.max_objects
            and attempts < max_attempts
        ):
            attempts += 1

            diameter = _uniform_size(rng, min(sx, sy), 0.012, 0.045, 0.18, 1.0)
            height = float(rng.uniform(max(1.0, sz * 0.25), max(1.01, sz * 0.90)))

            if diameter >= sx or diameter >= sy:
                continue

            x = float(rng.uniform(diameter / 2.0, sx - diameter / 2.0))
            y = float(rng.uniform(diameter / 2.0, sy - diameter / 2.0))
            z = height / 2.0

            tree = Obstacle(
                kind="tree",
                geometry="cylinder",
                position=np.asarray([x, y, z], dtype=float),
                dimensions=np.asarray([diameter, diameter, height], dtype=float),
                color=(0.32, 0.20, 0.08, 1.0),
            )

            if _contains_point_xy(tree, config.start, 0.6):
                continue
            if _contains_point_xy(tree, config.goal, 0.6):
                continue
            if _overlaps_xy(tree, result, config.min_clearance * 0.45):
                continue

            result.append(tree)
            occupied_area += tree.footprint_area

        return result


def make_scenario(name: str) -> ScenarioGenerator:
    if name == "empty":
        return EmptyScenario()
    if name == "city":
        return CityScenario()
    if name == "forest":
        return ForestScenario()
    raise ValueError(f"Неизвестный scenario: {name}")
