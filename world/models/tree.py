from __future__ import annotations

from typing import Sequence

import numpy as np
import pybullet as p

from world.models.base import ObjectModel


class TreeModel(ObjectModel):
    """
    Low-poly дерево.

    Состав:
    - cylinder trunk;
    - две sphere crown.

    Сферы используются намеренно:
    геометрия очень простая и дешёвая.
    """

    def __init__(
        self,
        height: float,
        trunk_diameter: float,
        crown_diameter: float,
        trunk_color: Sequence[float] = (
            0.30,
            0.18,
            0.07,
            1.0,
        ),
        crown_color: Sequence[float] = (
            0.16,
            0.42,
            0.14,
            1.0,
        ),
    ):
        self.height = float(
            height
        )

        self.trunk_diameter = float(
            trunk_diameter
        )

        self.crown_diameter = float(
            crown_diameter
        )

        self.trunk_color = tuple(
            trunk_color
        )

        self.crown_color = tuple(
            crown_color
        )

        if min(
            self.height,
            self.trunk_diameter,
            self.crown_diameter,
        ) <= 0.0:
            raise ValueError(
                "Размеры дерева должны быть > 0."
            )

    @property
    def kind(self) -> str:
        return "tree"

    @property
    def bounding_dimensions(
        self,
    ) -> np.ndarray:

        width = max(
            self.trunk_diameter,
            self.crown_diameter,
        )

        return np.asarray(
            [
                width,
                width,
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

        trunk_height = (
            self.height
            * 0.58
        )

        trunk_radius = (
            self.trunk_diameter
            / 2.0
        )

        crown_radius = min(
            self.crown_diameter / 2.0,
            self.height * 0.24,
        )

        trunk_z = (
            -self.height / 2.0
            + trunk_height / 2.0
        )

        upper_crown_z = (
            self.height / 2.0
            - crown_radius
        )

        lower_crown_z = (
            upper_crown_z
            - crown_radius * 0.85
        )

        shape_types = [
            p.GEOM_CYLINDER,
            p.GEOM_SPHERE,
            p.GEOM_SPHERE,
        ]

        radii = [
            trunk_radius,
            crown_radius * 0.90,
            crown_radius,
        ]

        lengths = [
            trunk_height,
            0.0,
            0.0,
        ]

        half_extents = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]

        positions = [
            [
                0.0,
                0.0,
                trunk_z,
            ],
            [
                -crown_radius * 0.20,
                0.0,
                lower_crown_z,
            ],
            [
                crown_radius * 0.15,
                0.0,
                upper_crown_z,
            ],
        ]

        orientations = [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

        collision = (
            p.createCollisionShapeArray(
                shapeTypes=shape_types,
                radii=radii,
                halfExtents=half_extents,
                lengths=lengths,
                collisionFramePositions=positions,
                collisionFrameOrientations=orientations,
                physicsClientId=client_id,
            )
        )

        visual = (
            p.createVisualShapeArray(
                shapeTypes=shape_types,
                radii=radii,
                halfExtents=half_extents,
                lengths=lengths,
                visualFramePositions=positions,
                visualFrameOrientations=orientations,
                rgbaColors=[
                    list(
                        self.trunk_color
                    ),
                    list(
                        self.crown_color
                    ),
                    list(
                        self.crown_color
                    ),
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