from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import pybullet as p


class ObjectModel(ABC):
    """
    Базовый класс процедурной модели объекта.

    Модель отвечает только за:
    - геометрию;
    - visual shape;
    - collision shape;
    - размеры bounding box.

    Она не отвечает за:
    - траекторию;
    - генерацию сценария;
    - алгоритмы навигации.
    """

    @property
    @abstractmethod
    def kind(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def bounding_dimensions(self) -> np.ndarray:
        """
        Размер ограничивающего параллелепипеда:
        [x, y, z].
        """
        raise NotImplementedError

    @abstractmethod
    def spawn(
        self,
        client_id: int,
        position: Sequence[float],
        orientation: Sequence[float] = (
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        mass: float = 0.0,
    ) -> int:
        """
        Создать модель в PyBullet.

        Возвращает body_id.
        """
        raise NotImplementedError

    def world_aabb(
        self,
        position: Sequence[float],
        orientation: Sequence[float] = (
            0.0,
            0.0,
            0.0,
            1.0,
        ),
    ):
        """
        Консервативный AABB модели с учётом ориентации.

        Используется до spawn() и для нейтрального
        представления объекта.
        """

        position = np.asarray(
            position,
            dtype=float,
        )

        half = (
            self.bounding_dimensions
            / 2.0
        )

        rotation = np.asarray(
            p.getMatrixFromQuaternion(
                orientation
            ),
            dtype=float,
        ).reshape(3, 3)

        world_half = (
            np.abs(rotation)
            @ half
        )

        return (
            position - world_half,
            position + world_half,
        )


class PrimitiveModel(ObjectModel):
    """
    Простая модель старого типа.

    Поддерживает:
    - box
    - cylinder
    - sphere
    """

    def __init__(
        self,
        kind: str,
        geometry: str,
        dimensions: Sequence[float],
        color: Sequence[float] = (
            0.6,
            0.6,
            0.6,
            1.0,
        ),
    ):
        self._kind = kind
        self.geometry = geometry

        self.dimensions = np.asarray(
            dimensions,
            dtype=float,
        )

        self.color = tuple(
            float(value)
            for value in color
        )

        if self.dimensions.shape != (3,):
            raise ValueError(
                "dimensions должны иметь вид [x, y, z]."
            )

        if geometry not in {
            "box",
            "cylinder",
            "sphere",
        }:
            raise ValueError(
                f"Неизвестная geometry: {geometry}"
            )

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def bounding_dimensions(
        self,
    ) -> np.ndarray:
        return self.dimensions.copy()

    def spawn(
        self,
        client_id: int,
        position: Sequence[float],
        orientation: Sequence[float] = (
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        mass: float = 0.0,
    ) -> int:

        if self.geometry == "box":

            half = (
                self.dimensions
                / 2.0
            ).tolist()

            collision = (
                p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=half,
                    physicsClientId=client_id,
                )
            )

            visual = (
                p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=half,
                    rgbaColor=self.color,
                    physicsClientId=client_id,
                )
            )

        elif self.geometry == "cylinder":

            radius = (
                float(
                    self.dimensions[0]
                )
                / 2.0
            )

            height = float(
                self.dimensions[2]
            )

            collision = (
                p.createCollisionShape(
                    p.GEOM_CYLINDER,
                    radius=radius,
                    height=height,
                    physicsClientId=client_id,
                )
            )

            visual = (
                p.createVisualShape(
                    p.GEOM_CYLINDER,
                    radius=radius,
                    length=height,
                    rgbaColor=self.color,
                    physicsClientId=client_id,
                )
            )

        else:

            radius = (
                float(
                    self.dimensions[0]
                )
                / 2.0
            )

            collision = (
                p.createCollisionShape(
                    p.GEOM_SPHERE,
                    radius=radius,
                    physicsClientId=client_id,
                )
            )

            visual = (
                p.createVisualShape(
                    p.GEOM_SPHERE,
                    radius=radius,
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