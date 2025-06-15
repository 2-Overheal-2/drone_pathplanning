import numpy as np
import pybullet as p
from typing import List, Dict, Tuple
from utils.dynamic_branches import create_dynamic_branches, DynamicBranch

class StaticArena:
    """
    Creates a static arena with floor and obstacles for UAV navigation
    """
    def __init__(self, arena_size: float = 11.0, include_dynamic_branches: bool = True, num_branches: int = 5, num_obstacles: int = 3):
        self.arena_size_x = arena_size
        self.arena_size_y = 3.0
        self.obstacle_ids = []
        self.floor_id = None
        self.include_dynamic_branches = include_dynamic_branches
        self.num_branches = num_branches
        self.num_obstacles = num_obstacles
        self.dynamic_branches = []
        
    def create_in_pybullet(self):
        """Create the arena in PyBullet simulation"""
        
        # Clear any existing IDs
        self.obstacle_ids = []
        
        # Create floor (just visual, no collision)
        floor_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.arena_size_x/2, self.arena_size_y/2, 0.01],
            rgbaColor=[0.7, 0.7, 0.7, 1.0]
        )
        self.floor_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=floor_visual,
            basePosition=[self.arena_size_x/2, self.arena_size_y/2, 0]
        )
        
        # Create static obstacles (boxes)
        obstacle_height = 6.0
        obstacle_positions = [
            # Format: (x, y, z, length, width, height, color)
            # First obstacle: (2,2) to (3,3)
            (2.5, 2.5, obstacle_height/2, 
             1.0, 1.0, obstacle_height, [0.8, 0.2, 0.2, 0.8]),  # Red obstacle
            
            # Second obstacle: (4,0) to (6,1)
            (5.0, 0.5, obstacle_height/2, 
             2.0, 1.0, obstacle_height, [0.2, 0.8, 0.2, 0.8]),  # Green obstacle
            
            # Third obstacle: (7,2) to (9,3)
            (8.0, 2.5, obstacle_height/2, 
             2.0, 1.0, obstacle_height, [0.2, 0.2, 0.8, 0.8]),  # Blue obstacle
        ]
        
        for x, y, z, length, width, height, color in obstacle_positions:
            obstacle_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[length/2, width/2, height/2]
            )
            obstacle_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[length/2, width/2, height/2],
                rgbaColor=color
            )
            obstacle_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=obstacle_shape,
                baseVisualShapeIndex=obstacle_visual,
                basePosition=[x, y, z]
            )
            self.obstacle_ids.append(obstacle_id)
        
        # Create dynamic branches with specific positions and movements
        if self.include_dynamic_branches:            
            # Create branches with specific configurations
            branch_configs = [
                # First branch at (2.5, 2, 5), moves down 2 points
                {"position": [2.5, 2.0, 5.0], "dimensions": [0.2, 2.0, 0.2], 
                 "movement_range": 2.0, "movement_direction": "up", "speed": 0.001},
                
                # Second branch at (4.5, 1, 2), moves up 1 point
                {"position": [4.5, 1.0, 2.0], "dimensions": [0.2, 2.0, 0.2], 
                 "movement_range": 1.0, "movement_direction": "up", "speed": 0.001},
                
                # Third branch at (5.5, 1, 5.5), moves down 1.5 points
                {"position": [5.5, 1.0, 5.5], "dimensions": [0.2, 2, 0.2], 
                 "movement_range": 1.5, "movement_direction": "up", "speed": 0.0015},
                
                # Fourth branch at (7.5, 2, 3), moves up 2.5 points
                {"position": [7.5, 2.0, 3.0], "dimensions": [0.2, 2, 0.2], 
                 "movement_range": 2.5, "movement_direction": "down"                
, "speed": 0.0015},
                
                # Fifth branch at (9.5, 2, 4), moves down 2.5 points
                {"position": [8.5, 2.0, 4.0], "dimensions": [0.2, 2.0, 0.2], 
                 "movement_range": 2.5, "movement_direction": "up", "speed": 0.0015}
            ]
            
            for config in branch_configs:
                branch = DynamicBranch(
                    position=config["position"],
                    dimensions=config["dimensions"],
                    movement_range=config["movement_range"],
                    movement_direction=config["movement_direction"],
                    speed=config["speed"],
                    color=[0.8, 0.4, 0.0, 0.9]  # Brown color
                )
                branch_id = branch.create_in_pybullet()
                if branch_id is not None:
                    self.obstacle_ids.append(branch_id)
                    self.dynamic_branches.append(branch)
            
        # Return all created object IDs
        all_ids = [self.floor_id] + self.obstacle_ids
        return all_ids
    
    def update_dynamic_objects(self):
        """Update positions of dynamic objects"""
        if self.include_dynamic_branches and self.dynamic_branches:
            for branch in self.dynamic_branches:
                branch.update()
        
    def get_obstacle_positions(self) -> List[Dict]:
        """
        Get positions and dimensions of all obstacles for collision detection
        
        Returns:
            List of dictionaries with obstacle information
        """
        obstacles = []
        
        # Add static obstacles
        obstacle_height = 6.0
        obstacle_positions = [
            {"position": [2.5, 2.5, obstacle_height/2], 
             "dimensions": [1.0, 1.0, obstacle_height], "type": "box"},
            
            {"position": [5.0, 0.5, obstacle_height/2], 
             "dimensions": [2.0, 1.0, obstacle_height], "type": "box"},
            
            {"position": [8.0, 2.5, obstacle_height/2], 
             "dimensions": [2.0, 1.0, obstacle_height], "type": "box"}
        ]
        
        obstacles.extend(obstacle_positions)
        
        # Add dynamic branches if they exist
        if self.include_dynamic_branches and self.dynamic_branches:
            for branch in self.dynamic_branches:
                obstacles.append(branch.get_position_and_dimensions())
        
        return obstacles
