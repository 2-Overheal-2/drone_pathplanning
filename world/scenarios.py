from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from world.config import ArenaConfig
from world.objects import Obstacle


Block = Tuple[
    float,
    float,
    float,
    float,
]


def _obstacles_overlap_xy(
    candidate: Obstacle,
    obstacles: Sequence[Obstacle],
    clearance: float,
) -> bool:
    """
    Проверка пересечения объектов по XY
    с дополнительным clearance.
    """

    candidate_min, candidate_max = (
        candidate.aabb
    )

    for obstacle in obstacles:

        obstacle_min, obstacle_max = (
            obstacle.aabb
        )

        separated = (
            candidate_max[0] + clearance
            < obstacle_min[0]

            or candidate_min[0] - clearance
            > obstacle_max[0]

            or candidate_max[1] + clearance
            < obstacle_min[1]

            or candidate_min[1] - clearance
            > obstacle_max[1]
        )

        if not separated:
            return True

    return False


def _point_inside_obstacle_xy(
    obstacle: Obstacle,
    point,
    clearance: float,
) -> bool:
    """
    Проверить, находится ли start/goal
    внутри объекта или слишком близко к нему.
    """

    obstacle_min, obstacle_max = (
        obstacle.aabb
    )

    x = float(
        point[0]
    )

    y = float(
        point[1]
    )

    return (
        obstacle_min[0] - clearance
        <= x
        <= obstacle_max[0] + clearance

        and obstacle_min[1] - clearance
        <= y
        <= obstacle_max[1] + clearance
    )


def _near_start_or_goal(
    obstacle: Obstacle,
    config: ArenaConfig,
) -> bool:

    return (
        _point_inside_obstacle_xy(
            obstacle,
            config.start,
            config.start_goal_clearance,
        )
        or
        _point_inside_obstacle_xy(
            obstacle,
            config.goal,
            config.start_goal_clearance,
        )
    )


def _clamp_height_range(
    value_range,
    arena_height: float,
):
    """
    Не позволяем объектам быть выше арены.
    """

    minimum = min(
        float(value_range[0]),
        arena_height,
    )

    maximum = min(
        float(value_range[1]),
        arena_height,
    )

    if maximum < minimum:
        minimum = maximum

    return (
        minimum,
        maximum,
    )


class EmptyScenario:
    """
    Пустая арена.
    """

    def generate(
        self,
        config: ArenaConfig,
        rng: np.random.Generator,
    ) -> List[Obstacle]:

        return []


class CityScenario:
    """
    Городской сценарий.

    Арена разбивается на кварталы.

    Между кварталами остаются пустые
    полосы шириной road_width.

    Внутри кварталов генерируются:
    - buildings
    - columns
    """

    def generate(
        self,
        config: ArenaConfig,
        rng: np.random.Generator,
    ) -> List[Obstacle]:

        size_x, size_y, size_z = (
            config.size
        )

        blocks = self._generate_blocks(
            config
        )

        if not blocks:
            return []

        target_area = (
            size_x
            * size_y
            * config.density
        )

        obstacles: List[
            Obstacle
        ] = []

        occupied_area = 0.0

        attempts = 0

        max_attempts = max(
            1000,
            config.max_objects * 80,
        )

        while (
            occupied_area
            < target_area
            and len(obstacles)
            < config.max_objects
            and attempts
            < max_attempts
        ):
            attempts += 1

            block = blocks[
                int(
                    rng.integers(
                        0,
                        len(blocks),
                    )
                )
            ]

            obstacle = (
                self._create_object_in_block(
                    block=block,
                    config=config,
                    rng=rng,
                )
            )

            if obstacle is None:
                continue

            if _near_start_or_goal(
                obstacle,
                config,
            ):
                continue

            clearance = max(
                config.min_clearance,
                config.building_spacing,
            )

            if _obstacles_overlap_xy(
                obstacle,
                obstacles,
                clearance,
            ):
                continue

            obstacles.append(
                obstacle
            )

            occupied_area += (
                obstacle.footprint_area
            )

        return obstacles

    def _generate_blocks(
        self,
        config: ArenaConfig,
    ) -> List[Block]:
        """
        Создать прямоугольные городские кварталы.

        Между соседними кварталами остаётся
        пустое пространство road_width.
        """

        size_x, size_y, _ = (
            config.size
        )

        block_size = float(
            config.city_block_size
        )

        road_width = float(
            config.road_width
        )

        blocks: List[
            Block
        ] = []

        x0 = 0.0

        while x0 < size_x:

            x1 = min(
                x0 + block_size,
                size_x,
            )

            y0 = 0.0

            while y0 < size_y:

                y1 = min(
                    y0 + block_size,
                    size_y,
                )

                width = (
                    x1 - x0
                )

                depth = (
                    y1 - y0
                )

                if (
                    width > 0.5
                    and depth > 0.5
                ):
                    blocks.append(
                        (
                            x0,
                            x1,
                            y0,
                            y1,
                        )
                    )

                y0 += (
                    block_size
                    + road_width
                )

            x0 += (
                block_size
                + road_width
            )

        return blocks

    def _create_object_in_block(
        self,
        block: Block,
        config: ArenaConfig,
        rng: np.random.Generator,
    ) -> Obstacle | None:

        (
            block_x0,
            block_x1,
            block_y0,
            block_y1,
        ) = block

        block_width = (
            block_x1
            - block_x0
        )

        block_depth = (
            block_y1
            - block_y0
        )

        create_column = (
            rng.random()
            < config.column_probability
        )

        if create_column:

            diameter = float(
                rng.uniform(
                    *config.column_diameter_range
                )
            )

            if (
                diameter
                >= block_width
                or diameter
                >= block_depth
            ):
                return None

            (
                min_height,
                max_height,
            ) = _clamp_height_range(
                config.column_height_range,
                config.size[2],
            )

            height = float(
                rng.uniform(
                    min_height,
                    max_height,
                )
            )

            x = float(
                rng.uniform(
                    block_x0
                    + diameter / 2.0,

                    block_x1
                    - diameter / 2.0,
                )
            )

            y = float(
                rng.uniform(
                    block_y0
                    + diameter / 2.0,

                    block_y1
                    - diameter / 2.0,
                )
            )

            return Obstacle(
                kind="column",

                geometry="cylinder",

                position=np.asarray(
                    [
                        x,
                        y,
                        height / 2.0,
                    ],
                    dtype=float,
                ),

                dimensions=np.asarray(
                    [
                        diameter,
                        diameter,
                        height,
                    ],
                    dtype=float,
                ),

                color=(
                    0.55,
                    0.55,
                    0.58,
                    1.0,
                ),
            )

        width = float(
            rng.uniform(
                *config.building_width_range
            )
        )

        depth = float(
            rng.uniform(
                *config.building_depth_range
            )
        )

        if (
            width
            >= block_width
            or depth
            >= block_depth
        ):
            return None

        (
            min_height,
            max_height,
        ) = _clamp_height_range(
            config.building_height_range,
            config.size[2],
        )

        height = float(
            rng.uniform(
                min_height,
                max_height,
            )
        )

        x = float(
            rng.uniform(
                block_x0
                + width / 2.0,

                block_x1
                - width / 2.0,
            )
        )

        y = float(
            rng.uniform(
                block_y0
                + depth / 2.0,

                block_y1
                - depth / 2.0,
            )
        )

        #
        # Небольшая вариативность цвета,
        # пока модели остаются примитивами.
        #
        shade = float(
            rng.uniform(
                0.38,
                0.62,
            )
        )

        return Obstacle(
            kind="building",

            geometry="box",

            position=np.asarray(
                [
                    x,
                    y,
                    height / 2.0,
                ],
                dtype=float,
            ),

            dimensions=np.asarray(
                [
                    width,
                    depth,
                    height,
                ],
                dtype=float,
            ),

            color=(
                shade,
                shade,
                min(
                    1.0,
                    shade + 0.04,
                ),
                1.0,
            ),
        )


class ForestScenario:
    """
    Лесной сценарий.

    Деревья могут располагаться:
    - равномерно;
    - группами вокруг cluster centers.

    Дополнительно создаются пустые поляны.
    """

    def generate(
        self,
        config: ArenaConfig,
        rng: np.random.Generator,
    ) -> List[Obstacle]:

        size_x, size_y, size_z = (
            config.size
        )

        target_area = (
            size_x
            * size_y
            * config.density
        )

        cluster_centers = (
            self._generate_cluster_centers(
                config,
                rng,
            )
        )

        clearings = (
            self._generate_clearings(
                config,
                rng,
            )
        )

        obstacles: List[
            Obstacle
        ] = []

        occupied_area = 0.0

        attempts = 0

        max_attempts = max(
            2000,
            config.max_objects * 100,
        )

        while (
            occupied_area
            < target_area
            and len(obstacles)
            < config.max_objects
            and attempts
            < max_attempts
        ):
            attempts += 1

            position_xy = (
                self._sample_tree_position(
                    config=config,
                    rng=rng,
                    cluster_centers=(
                        cluster_centers
                    ),
                )
            )

            if position_xy is None:
                continue

            x, y = position_xy

            if self._inside_clearing(
                x,
                y,
                clearings,
            ):
                continue

            diameter = float(
                rng.uniform(
                    *config.tree_diameter_range
                )
            )

            (
                min_height,
                max_height,
            ) = _clamp_height_range(
                config.tree_height_range,
                size_z,
            )

            height = float(
                rng.uniform(
                    min_height,
                    max_height,
                )
            )

            radius = (
                diameter
                / 2.0
            )

            if not (
                radius
                <= x
                <= size_x - radius

                and radius
                <= y
                <= size_y - radius
            ):
                continue

            tree = Obstacle(
                kind="tree",

                geometry="cylinder",

                position=np.asarray(
                    [
                        x,
                        y,
                        height / 2.0,
                    ],
                    dtype=float,
                ),

                dimensions=np.asarray(
                    [
                        diameter,
                        diameter,
                        height,
                    ],
                    dtype=float,
                ),

                color=(
                    0.28,
                    0.18,
                    0.07,
                    1.0,
                ),
            )

            if _near_start_or_goal(
                tree,
                config,
            ):
                continue

            if _obstacles_overlap_xy(
                tree,
                obstacles,
                config.min_clearance,
            ):
                continue

            obstacles.append(
                tree
            )

            occupied_area += (
                tree.footprint_area
            )

        return obstacles

    def _generate_cluster_centers(
        self,
        config: ArenaConfig,
        rng: np.random.Generator,
    ):

        centers = []

        size_x, size_y, _ = (
            config.size
        )

        for _ in range(
            config.forest_cluster_count
        ):

            centers.append(
                np.asarray(
                    [
                        rng.uniform(
                            0.0,
                            size_x,
                        ),

                        rng.uniform(
                            0.0,
                            size_y,
                        ),
                    ],
                    dtype=float,
                )
            )

        return centers

    def _generate_clearings(
        self,
        config: ArenaConfig,
        rng: np.random.Generator,
    ):

        clearings = []

        size_x, size_y, _ = (
            config.size
        )

        for _ in range(
            config.forest_clearing_count
        ):

            clearings.append(
                (
                    float(
                        rng.uniform(
                            0.0,
                            size_x,
                        )
                    ),

                    float(
                        rng.uniform(
                            0.0,
                            size_y,
                        )
                    ),

                    float(
                        config.forest_clearing_radius
                    ),
                )
            )

        return clearings

    def _sample_tree_position(
        self,
        config: ArenaConfig,
        rng: np.random.Generator,
        cluster_centers,
    ):
        """
        Часть деревьев располагается
        около cluster center.

        Остальные распределяются
        равномерно по лесу.
        """

        size_x, size_y, _ = (
            config.size
        )

        use_cluster = (
            bool(cluster_centers)
            and rng.random()
            < config.forest_cluster_probability
        )

        if not use_cluster:

            return (
                float(
                    rng.uniform(
                        0.0,
                        size_x,
                    )
                ),

                float(
                    rng.uniform(
                        0.0,
                        size_y,
                    )
                ),
            )

        center = cluster_centers[
            int(
                rng.integers(
                    0,
                    len(cluster_centers),
                )
            )
        ]

        #
        # Normal distribution даёт более
        # естественное сгущение около центра.
        #
        x = float(
            rng.normal(
                center[0],
                config.forest_cluster_radius,
            )
        )

        y = float(
            rng.normal(
                center[1],
                config.forest_cluster_radius,
            )
        )

        return (
            x,
            y,
        )

    @staticmethod
    def _inside_clearing(
        x: float,
        y: float,
        clearings,
    ) -> bool:

        for (
            clearing_x,
            clearing_y,
            radius,
        ) in clearings:

            dx = (
                x
                - clearing_x
            )

            dy = (
                y
                - clearing_y
            )

            if (
                dx * dx
                + dy * dy
                <= radius * radius
            ):
                return True

        return False


def make_scenario(
    name: str,
):
    """
    Factory сценариев.
    """

    if name == "empty":
        return EmptyScenario()

    if name == "city":
        return CityScenario()

    if name == "forest":
        return ForestScenario()

    raise ValueError(
        f"Неизвестный scenario: {name}"
    )