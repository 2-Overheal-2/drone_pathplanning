import numpy as np
import pybullet as p
import math
from typing import List, Dict, Tuple, Union

class DynamicBranch:
    """Class representing a dynamic branch-like obstacle that moves up and down"""
    
    def __init__(self, 
                 position: List[float], 
                 dimensions: List[float],
                 movement_range: float = 1.5,
                 movement_direction: str = "down",
                 speed: float = 0.03,
                 color: List[float] = None):
        """
        Initialize a dynamic branch obstacle
        
        Args:
            position: [x, y, z] center position
            dimensions: [length, width, height] dimensions of the branch
            movement_range: How far the branch moves
            movement_direction: Initial movement direction ("up" or "down")
            speed: Movement speed factor
            color: RGBA color (defaults to brown)
        """
        self.position = np.array(position, dtype=np.float32)
        self.dimensions = dimensions
        self.movement_range = movement_range
        self.speed = speed
        self.color = color if color is not None else [0.6, 0.3, 0.1, 0.9]  # Brown by default
        self.movement_direction = movement_direction
        
        # Initial position is the center position
        self.center_position = self.position.copy()
        
        # Initialize obstacle in PyBullet
        self.obstacle_id = None
        self.step_count = 0
        
        # Set initial phase based on direction
        self.phase_offset = 0 if movement_direction == "down" else math.pi
        
    def create_in_pybullet(self):
        """Create the branch in PyBullet simulation"""
        try:
            branch_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[self.dimensions[0]/2, self.dimensions[1]/2, self.dimensions[2]/2]
            )
            branch_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[self.dimensions[0]/2, self.dimensions[1]/2, self.dimensions[2]/2],
                rgbaColor=self.color
            )
            self.obstacle_id = p.createMultiBody(
                baseMass=0,  # Static obstacle (mass=0)
                baseCollisionShapeIndex=branch_shape,
                baseVisualShapeIndex=branch_visual,
                basePosition=self.position
            )
            return self.obstacle_id
        except Exception as e:
            print(f"Error creating branch: {e}")
            return None
        
    def update(self):
        """Update branch position with vertical oscillation"""
        try:
            self.step_count += 1
            
            # Calculate vertical offset using sine wave with phase offset
            vertical_offset = self.movement_range * math.sin(self.step_count * self.speed + self.phase_offset)
            
            # Update position (only z-coordinate changes)
            self.position[2] = self.center_position[2] + vertical_offset
            
            # Update position in PyBullet
            if self.obstacle_id is not None:
                p.resetBasePositionAndOrientation(
                    self.obstacle_id,
                    self.position,
                    [0, 0, 0, 1]  # Default orientation (no rotation)
                )
        except Exception as e:
            print(f"Error updating branch: {e}")
    
    def get_position_and_dimensions(self):
        """Get current position and dimensions for collision detection"""
        return {
            "position": self.position.tolist(),
            "dimensions": self.dimensions,
            "type": "box"
        }

def create_dynamic_branches(static_obstacles: List[Dict], 
                           arena_size: float = 10.0,
                           num_branches: int = 5,
                           branch_length: float = 2.0,
                           branch_width: float = 0.2,
                           branch_height: float = 0.2) -> List[DynamicBranch]:
    """
    Create dynamic branch obstacles
    
    Args:
        static_obstacles: List of static obstacles with position and dimensions
        arena_size: Size of the arena
        num_branches: Number of branches to create
        branch_length: Length of each branch
        branch_width: Width of each branch
        branch_height: Height of each branch
        
    Returns:
        List of DynamicBranch objects
    """
    # This function is not used in the current implementation
    # Branches are created directly in StaticArena class
    return []
