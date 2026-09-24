from __future__ import annotations

from typing import Sequence

import numpy as np
import pybullet as p

from world.models.base import ObjectModel


class BuildingModel(ObjectModel):
    """
    Простое low-poly здание.

    Состав:
    - основной корпус;
    - крыша;
    - небольшой технический блок.
    """

    def __init__(
        self,
        width: float,
        depth: float,
        height: float,
        color: Sequence[float] = (
            0.48,
            0.50,
            0.54,
            1.0,
        ),
    ):
        self.width = float(width)
        self.depth = float(depth)
        self.height = float(height)

        self.color = tuple(
            float(value)
            for value in color
        )

        if min(
            self.width,
            self.depth,
            self.height,
        ) <= 0.0:
            raise ValueError(
                "Размеры здания должны быть > 0."
            )

    @property
    def kind(self) -> str:
        return "building"

    @property
    def bounding_dimensions(
        self,
    ) -> np.ndarray:

        return np.asarray(
            [
                self.width,
                self.depth,
                self.height,
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

        main_height = (
            self.height
            * 0.90
        )

        roof_height = (
            self.height
            * 0.04
        )

        unit_height = min(
            self.height * 0.08,
            0.50,
        )

        main_position = [
            0.0,
            0.0,
            -self.height / 2.0
            + main_height / 2.0,
        ]

        roof_position = [
            0.0,
            0.0,
            self.height / 2.0
            - roof_height / 2.0,
        ]

        unit_position = [
            self.width * 0.18,
            -self.depth * 0.15,
            self.height / 2.0
            - unit_height / 2.0,
        ]

        shape_types = [
            p.GEOM_BOX,
            p.GEOM_BOX,
            p.GEOM_BOX,
        ]

        half_extents = [
            [
                self.width / 2.0,
                self.depth / 2.0,
                main_height / 2.0,
            ],
            [
                self.width / 2.0,
                self.depth / 2.0,
                roof_height / 2.0,
            ],
            [
                self.width * 0.12,
                self.depth * 0.12,
                unit_height / 2.0,
            ],
        ]

        positions = [
            main_position,
            roof_position,
            unit_position,
        ]

        orientations = [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

        body_color = list(
            self.color
        )

        roof_color = [
            max(
                0.0,
                self.color[0] - 0.10,
            ),
            max(
                0.0,
                self.color[1] - 0.10,
            ),
            max(
                0.0,
                self.color[2] - 0.10,
            ),
            1.0,
        ]

        unit_color = [
            0.32,
            0.34,
            0.36,
            1.0,
        ]

        collision = (
            p.createCollisionShapeArray(
                shapeTypes=shape_types,
                halfExtents=half_extents,
                collisionFramePositions=positions,
                collisionFrameOrientations=orientations,
                physicsClientId=client_id,
            )
        )

        visual = (
            p.createVisualShapeArray(
                shapeTypes=shape_types,
                halfExtents=half_extents,
                visualFramePositions=positions,
                visualFrameOrientations=orientations,
                rgbaColors=[
                    body_color,
                    roof_color,
                    unit_color,
                ],
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