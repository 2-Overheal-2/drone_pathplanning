from __future__ import annotations

from typing import Sequence

import numpy as np
import pybullet as p

from world.models.base import ObjectModel


class WireModel(ObjectModel):
    """
    Статический провод.

    Локальная ось цилиндра направлена по Z.

    Для размещения между произвольными точками
    используется wire_transform().
    """

    def __init__(
        self,
        length: float,
        radius: float = 0.025,
        color: Sequence[float] = (
            0.10,
            0.10,
            0.10,
            1.0,
        ),
    ):
        self.length = float(
            length
        )

        self.radius = float(
            radius
        )

        self.color = tuple(
            color
        )

        if self.length <= 0.0:
            raise ValueError(
                "Длина провода должна быть > 0."
            )

        if self.radius <= 0.0:
            raise ValueError(
                "Радиус провода должен быть > 0."
            )

    @property
    def kind(self) -> str:
        return "wire"

    @property
    def bounding_dimensions(
        self,
    ) -> np.ndarray:

        diameter = (
            self.radius
            * 2.0
        )

        return np.asarray(
            [
                diameter,
                diameter,
                self.length,
            ],
            dtype=float,
        )

    def spawn(
        self,
        client_id: int,
        position,
        orientation=(
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        mass: float = 0.0,
    ) -> int:

        collision = (
            p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=self.radius,
                height=self.length,
                physicsClientId=client_id,
            )
        )

        visual = (
            p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=self.radius,
                length=self.length,
                rgbaColor=self.color,
                physicsClientId=client_id,
            )
        )

        return p.createMultiBody(
            baseMass=float(mass),
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=list(position),
            baseOrientation=list(orientation),
            physicsClientId=client_id,
        )


def wire_transform(
    start: Sequence[float],
    end: Sequence[float],
):
    """
    Получить:
    - midpoint;
    - quaternion;
    - length

    для цилиндра между start и end.
    """

    start = np.asarray(
        start,
        dtype=float,
    )

    end = np.asarray(
        end,
        dtype=float,
    )

    direction = (
        end - start
    )

    length = float(
        np.linalg.norm(
            direction
        )
    )

    if length <= 1e-9:
        raise ValueError(
            "Начало и конец провода "
            "не могут совпадать."
        )

    midpoint = (
        start + end
    ) / 2.0

    dx, dy, dz = (
        direction
    )

    horizontal = float(
        np.hypot(
            dx,
            dy,
        )
    )

    yaw = float(
        np.arctan2(
            dy,
            dx,
        )
    )

    pitch = float(
        np.arctan2(
            horizontal,
            dz,
        )
    )

    orientation = (
        p.getQuaternionFromEuler(
            [
                0.0,
                pitch,
                yaw,
            ]
        )
    )

    return (
        midpoint,
        np.asarray(
            orientation,
            dtype=float,
        ),
        length,
    )