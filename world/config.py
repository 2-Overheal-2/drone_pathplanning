from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple


Point3D = Tuple[float, float, float]


@dataclass(frozen=True)
class ArenaConfig:
    """Конфигурация геометрии и генерации мира."""

    size: Point3D = (20.0, 20.0, 10.0)
    scenario: str = "city"  # city | forest | empty
    density: float = 0.15
    seed: int = 42

    start: Point3D = (1.5, 1.5, 1.5)
    goal: Point3D = (18.0, 18.0, 3.0)

    min_clearance: float = 0.35
    max_objects: int = 250

    enable_dynamic_objects: bool = True
    num_cars: int = 1
    num_pedestrians: int = 1

    def validate(self) -> None:
        sx, sy, sz = self.size

        if not all(0.0 < value <= 50.0 for value in (sx, sy, sz)):
            raise ValueError("Каждый размер арены должен быть в диапазоне (0, 50].")

        if not 0.0 <= self.density <= 1.0:
            raise ValueError("density должна быть в диапазоне [0, 1].")

        if self.scenario not in {"city", "forest", "empty"}:
            raise ValueError("scenario должен быть 'city', 'forest' или 'empty'.")

        for name, point in (("start", self.start), ("goal", self.goal)):
            x, y, z = point
            if not (0.0 <= x <= sx and 0.0 <= y <= sy and 0.0 <= z <= sz):
                raise ValueError(f"{name}={point} находится вне арены size={self.size}.")

        if self.min_clearance < 0.0:
            raise ValueError("min_clearance не может быть отрицательным.")

        if self.max_objects < 0:
            raise ValueError("max_objects не может быть отрицательным.")

        if self.num_cars < 0 or self.num_pedestrians < 0:
            raise ValueError("Количество динамических объектов не может быть отрицательным.")

    def with_seed(self, seed: int) -> "ArenaConfig":
        return replace(self, seed=int(seed))
