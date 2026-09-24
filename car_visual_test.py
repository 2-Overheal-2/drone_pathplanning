import time

import pybullet as p
import pybullet_data

from world.models import CarModel


def main():
    client_id = p.connect(
        p.GUI
    )

    try:
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(),
            physicsClientId=client_id,
        )

        p.setGravity(
            0.0,
            0.0,
            -9.81,
            physicsClientId=client_id,
        )

        p.loadURDF(
            "plane.urdf",
            physicsClientId=client_id,
        )

        car = CarModel(
            length=4.2,
            width=1.8,
            height=1.2,
        )

        print(
            "Mesh:",
            car.mesh_path,
        )

        print(
            "Bounding dimensions:",
            car.bounding_dimensions,
        )

        car_id = car.spawn(
            client_id=client_id,
            position=[
                0.0,
                0.0,
                0.6,
            ],
        )

        print(
            "Car body ID:",
            car_id,
        )

        p.resetDebugVisualizerCamera(
            cameraDistance=7.0,
            cameraYaw=45.0,
            cameraPitch=-25.0,
            cameraTargetPosition=[
                0.0,
                0.0,
                0.8,
            ],
            physicsClientId=client_id,
        )

        print()
        print(
            "Car visual test running."
        )

        print(
            "Close the PyBullet window to finish."
        )

        while p.isConnected(
            client_id
        ):
            p.stepSimulation(
                physicsClientId=client_id,
            )

            time.sleep(
                1.0 / 240.0
            )

    finally:
        if p.isConnected(
            client_id
        ):
            p.disconnect(
                client_id
            )


if __name__ == "__main__":
    main()