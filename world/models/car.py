from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pybullet as p

from world.models.base import ObjectModel


class CarModel(ObjectModel):
    """
    Машина с OBJ-моделью для отображения
    и простой box-геометрией для столкновений.

    Важно:

    - колёса НЕ создаются как collision shapes;
    - колёса НЕ имеют отдельной физики;
    - вся collision geometry машины = один box;
    - OBJ используется только для визуального отображения;
    - MTL задаёт внешний вид машины.
    """

    #
    # Исходные размеры NormalCar1.obj.
    #
    # Оси исходной модели:
    #
    # X = ширина
    # Y = высота
    # Z = длина
    #
    SOURCE_WIDTH = 1.807360
    SOURCE_HEIGHT = 1.176579
    SOURCE_LENGTH = 4.220717

    SOURCE_MIN_X = -0.903680
    SOURCE_MAX_X = 0.903680

    SOURCE_MIN_Y = 0.006030
    SOURCE_MAX_Y = 1.182609

    SOURCE_MIN_Z = -2.107265
    SOURCE_MAX_Z = 2.113452

    def __init__(
        self,
        length: float = 1.8,
        width: float = 0.9,
        height: float = 0.8,
        mesh_path: str | Path | None = None,
    ):
        self.length = float(
            length
        )

        self.width = float(
            width
        )

        self.height = float(
            height
        )

        if min(
            self.length,
            self.width,
            self.height,
        ) <= 0.0:
            raise ValueError(
                "Размеры машины должны быть > 0."
            )

        if mesh_path is None:
            project_root = (
                Path(__file__)
                .resolve()
                .parents[2]
            )

            mesh_path = (
                project_root
                / "assets"
                / "cars"
                / "normal_car_1"
                / "NormalCar1.obj"
            )

        self.mesh_path = (
            Path(mesh_path)
            .resolve()
        )

        if not self.mesh_path.exists():
            raise FileNotFoundError(
                "Не найдена OBJ-модель машины:\n"
                f"{self.mesh_path}\n\n"
                "Ожидаемый путь:\n"
                "assets/cars/normal_car_1/NormalCar1.obj"
            )

        mtl_path = (
            self.mesh_path.parent
            / "NormalCar1.mtl"
        )

        if not mtl_path.exists():
            raise FileNotFoundError(
                "Не найден MTL-файл машины:\n"
                f"{mtl_path}\n\n"
                "OBJ и MTL должны находиться "
                "в одной папке."
            )

    @property
    def kind(
        self,
    ) -> str:
        return "car"

    @property
    def bounding_dimensions(
        self,
    ) -> np.ndarray:
        """
        Габарит для алгоритмов и collision checks.

        Колёса отдельно здесь не учитываются.
        """

        return np.asarray(
            [
                self.length,
                self.width,
                self.height,
            ],
            dtype=float,
        )

    def _mesh_scale(
        self,
    ) -> list[float]:
        """
        Масштабируем OBJ под заданные размеры машины.

        OBJ:
            X -> width
            Y -> height
            Z -> length
        """

        scale_x = (
            self.width
            / self.SOURCE_WIDTH
        )

        scale_y = (
            self.height
            / self.SOURCE_HEIGHT
        )

        scale_z = (
            self.length
            / self.SOURCE_LENGTH
        )

        return [
            scale_x,
            scale_y,
            scale_z,
        ]

    def _visual_offset(
        self,
    ) -> list[float]:
        """
        Центрируем модель относительно collision box.

        Нижняя точка машины должна совпадать
        с нижней гранью collision box.
        """

        scale_x, scale_y, scale_z = (
            self._mesh_scale()
        )

        source_center_x = (
            self.SOURCE_MIN_X
            + self.SOURCE_MAX_X
        ) / 2.0

        source_center_z = (
            self.SOURCE_MIN_Z
            + self.SOURCE_MAX_Z
        ) / 2.0

        #
        # После поворота:
        #
        # OBJ Z -> world X
        # OBJ X -> world Y
        # OBJ Y -> world Z
        #
        offset_x = (
            -source_center_z
            * scale_z
        )

        offset_y = (
            -source_center_x
            * scale_x
        )

        #
        # OBJ начинается почти с Y=0.
        # Совмещаем низ mesh с низом box.
        #
        offset_z = (
            -self.height / 2.0
            - self.SOURCE_MIN_Y
            * scale_y
        )

        return [
            offset_x,
            offset_y,
            offset_z,
        ]

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
        Создать машину.

        Collision:
            один обычный box.

        Visual:
            NormalCar1.obj.

        Никаких отдельных колёс
        в физической модели нет.
        """

        #
        # Простая collision geometry.
        #
        collision_shape = (
            p.createCollisionShape(
                shapeType=p.GEOM_BOX,

                halfExtents=[
                    self.length / 2.0,
                    self.width / 2.0,
                    self.height / 2.0,
                ],

                physicsClientId=client_id,
            )
        )

        #
        # Исходный OBJ использует:
        #
        # X = width
        # Y = height
        # Z = length
        #
        # Нам нужно:
        #
        # X = length
        # Y = width
        # Z = height
        #
        # Quaternion ниже выполняет
        # циклическую перестановку осей.
        #
        mesh_orientation = [
            0.5,
            0.5,
            0.5,
            0.5,
        ]

        visual_shape = (
            p.createVisualShape(
                shapeType=p.GEOM_MESH,

                fileName=str(
                    self.mesh_path
                ),

                meshScale=(
                    self._mesh_scale()
                ),

                visualFramePosition=(
                    self._visual_offset()
                ),

                visualFrameOrientation=(
                    mesh_orientation
                ),

                physicsClientId=client_id,
            )
        )

        if visual_shape < 0:
            raise RuntimeError(
                "PyBullet не смог создать "
                f"visual shape из {self.mesh_path}"
            )

        body_id = (
            p.createMultiBody(
                baseMass=float(
                    mass
                ),

                baseCollisionShapeIndex=(
                    collision_shape
                ),

                baseVisualShapeIndex=(
                    visual_shape
                ),

                basePosition=[
                    float(position[0]),
                    float(position[1]),
                    float(position[2]),
                ],

                baseOrientation=[
                    float(orientation[0]),
                    float(orientation[1]),
                    float(orientation[2]),
                    float(orientation[3]),
                ],

                physicsClientId=client_id,
            )
        )

        return body_id