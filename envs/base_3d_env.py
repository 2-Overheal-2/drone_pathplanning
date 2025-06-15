import pybullet as p
import numpy as np
from gymnasium.spaces.box import Box
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from typing import List, Dict, Union, Optional
import traceback
import time
import os
from utils.static_arena import StaticArena
from utils.dynamic_branches import DynamicBranch

class Base3DEnv(BaseRLAviary):
  """
  Custom 3D environment for UAV path planning with obstacles.
  Extends the BaseRLAviary from gym-pybullet-drones.
  """
  def __init__(self,
               drone_model: DroneModel = DroneModel.CF2X,
               initial_xyzs=None,
               physics: Physics = Physics.PYB,
               freq: int = 240,
               gui: bool = False,
               target: list = None,
               arena_size: float = 11.0,
               num_branches: int = 5,
               episode_length: float = 13.0,):
    
    # Initialize starting position - keep the default position

    initial_xyzs = np.array([[1.5, 1.5, 3]])
    
    # Store initial position for resets
    self.initial_xyzs = initial_xyzs.copy()
    
    # Initialize attributes that will be used in _setup_environment
    # IMPORTANT: These must be initialized before calling super().__init__
    self.obstacle_ids = []
    self.wall_ids = []
    self.target_id = None
    self.static_arena = None
    self.drone_marker_id = None
    self.obstacle_data = []
    
    # Environment parameters
    # Set target at the end of the corridor
    self.EPISODE_LEN_SEC = episode_length  # Set episode length
    self.gui = gui
    
    # Statistics
    self.target_reached_count = 0
    self.collision_count = 0
    self.episode_count = 0
    self.out_of_bounds_count = 0
    self.timeout_count = 0
    self.tilted_count = 0
    self.first_obstacle_count = 0
    self.second_obstacle_count = 0
    
    # Video recording attributes
    self.recording = False
    self.video_id = None
    self.video_directory = "videos"
    self.video_path = None
    
    # Boundary awareness - track how close to boundaries the drone is
    self.boundary_margin = 0.3  # Distance from boundary to start penalizing
    self.arena_lenght = 11
    self.arena_width = 3
    self.arena_height = 6.3
    self.dangerous_height = 0.5
    self.target = np.array(target if target is not None else [9, 1.5, 4.5], dtype=np.float32)
    
    # Store parameters for reset
    self._init_params = {
        'drone_model': drone_model,
        'initial_xyzs': initial_xyzs.copy(),
        'physics': physics,
        'freq': freq,
        'gui': gui,
        'target': target.copy() if target is not None else None,
        'episode_length': episode_length
    }
    
    # Call BaseRLAviary constructor
    super().__init__(
        drone_model=drone_model,
        initial_xyzs=initial_xyzs,
        physics=physics,
        pyb_freq=freq,
        ctrl_freq= 30,
        gui=gui,
        record=False,
        obs=ObservationType.KIN,
        act=ActionType.RPM
    )
    
    # Define action and observation spaces
    self.action_space = self._actionSpace()
    
    # Observation space includes drone state and target position
    # Create a temporary observation to determine the actual shape
    self.observation_space = self._observationSpace()
    self.TARGET_POS = target
    self.racing_setup = { 0: [[3, 0, 0.5],[3, 2, 6.3]],
                            1: [[6, 1, 0.5], [6, 3, 6.3]],
                            2: [[9, 0, 0.5], [9, 2, 6.3]]
                            }

    self.passing_flag = [False, False, False]
    self.prev_pos = self._getDroneStateVector(0)[:3] 
    # Setup environment after initialization
    self._setup_environment()
    print(f"GUI enabled: {self.gui}")
    print("Using static arena with rectangular obstacles")

  def _addObstacles(self):
    """Override the parent class method to use our custom setup"""
    # This method is called by the parent class during initialization
    # We'll handle obstacle creation in our _setup_environment method
    pass

  def _setup_environment(self):
    """Creates obstacles and target in the simulation"""        
    
    # Create target (visual marker)
    # target_shape = p.createVisualShape(
    #     p.GEOM_SPHERE,
    #     radius=0.3,
    #     rgbaColor=[1, 0, 0, 1]
    # )
    # self.target_id = p.createMultiBody(
    #     baseVisualShapeIndex=target_shape,
    #     basePosition=self.target
    # )
    
    # Only create arena objects if rendering is enabled
        # Create static arena with walls and rectangular obstacles
    self.static_arena = StaticArena(
                arena_size=self.arena_lenght,
                include_dynamic_branches=True,
                num_branches=5,
                num_obstacles=3
            )
    created_ids = self.static_arena.create_in_pybullet()
    self.obstacle_ids.extend(created_ids)
        
        # Update obstacle data for collision detection
    self.obstacle_data = self.static_arena.get_obstacle_positions()
        
        # Add a visual marker for the drone
    self._setup_drone_marker()
        

  def _remove_objects(self):
    """Remove all objects from the simulation safely"""
        
    if not hasattr(p, 'getNumBodies'):
        print("PyBullet not initialized, skipping object removal")
        return
        
    # Get all existing bodies in the simulation
    try:
        existing_bodies = set()
        for i in range(p.getNumBodies()):
            existing_bodies.add(p.getBodyUniqueId(i))
    except Exception as e:
        print(f"Error getting bodies: {e}")
        return
    
    # Remove target if it exists
    if self.target_id is not None and self.target_id in existing_bodies:
        try:
            p.removeBody(self.target_id)
        except Exception as e:
            print(f"Error removing target: {e}")
        self.target_id = None
    
    # Remove drone marker if it exists
    if hasattr(self, 'drone_marker_id') and self.drone_marker_id is not None and self.drone_marker_id in existing_bodies:
        try:
            p.removeBody(self.drone_marker_id)
        except Exception as e:
            print(f"Error removing drone marker: {e}")
        self.drone_marker_id = None
    
    # Remove obstacles only if they exist
    for obj_id in self.obstacle_ids:
        if obj_id in existing_bodies:
            try:
                p.removeBody(obj_id)
            except Exception as e:
                print(f"Error removing obstacle {obj_id}: {e}")
    
    # Clear obstacle IDs list
    self.obstacle_ids = []
    
    # Mark environment as not set up

  def _computeObs(self):
    """Compute the observation vector"""
    try:
        # Get observation from parent class
        obs = super()._computeObs()
          
        return obs
          
    except Exception as e:
        print(f"Error computing observation: {e}")
        print(traceback.format_exc())
        return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        

  

  def _check_box_collision(self, drone_pos, box_pos, box_dims):
    """
    Check if drone collides with a box obstacle
    
    Args:
        drone_pos: [x, y, z] drone position
        box_pos: [x, y, z] box center position
        box_dims: [length, width, height] box dimensions
        
    Returns:
        True if collision detected, False otherwise
    """
    # Calculate half dimensions
    half_length, half_width, half_height = box_dims[0]/2, box_dims[1]/2, box_dims[2]/2
    
    # Calculate closest point on box to drone
    closest_x = max(box_pos[0] - half_length, min(drone_pos[0], box_pos[0] + half_length))
    closest_y = max(box_pos[1] - half_width, min(drone_pos[1], box_pos[1] + half_width))
    closest_z = max(box_pos[2] - half_height, min(drone_pos[2], box_pos[2] + half_height))
    
    # Calculate distance between closest point and drone
    distance = np.linalg.norm(np.array([closest_x, closest_y, closest_z]) - drone_pos)
    
    # Drone radius (approximate)
    drone_radius = 0.1
    
    # Check collision
    return distance < drone_radius

  def _computeTerminated(self):
    """Determine if episode is terminated (success or collision)"""
    try:
        state = self._getDroneStateVector(0)
        drone_pos = np.array([state[0], state[1], state[2]])

        # Check if target reached
        if (drone_pos[0]>self.target[0]) and (0<drone_pos[1]<self.arena_width) and (self.dangerous_height<drone_pos[2]<self.arena_height):
            self.target_reached_count += 1
            print("Target reached!")
            return True

        # Check for collisions with obstacles
        collision = False
        
        # Check static obstacles based on their type
        for obs in self.obstacle_data:
            if obs["type"] == "box":
                # Box obstacle
                if self._check_box_collision(drone_pos, obs["position"], obs["dimensions"]):
                    collision = True
                    break

        # Check if collision occurred
        if collision:
            self.collision_count += 1
            print("Collision detected!")
            print(f"Drone position: {drone_pos}")
            return True
            
        return False
    except Exception as e:
        print(f"Error in _computeTerminated: {e}")
        return False

  def _computeTruncated(self):
    """Determine if episode is truncated (out of bounds or timeout)"""
    try:
        state = self._getDroneStateVector(0)
        drone_pos = np.array([state[0], state[1], state[2]])

        # Check if out of bounds
        if (drone_pos[0] < 0 or drone_pos[0] > self.arena_lenght or
            drone_pos[1] < 0 or drone_pos[1] > self.arena_width or
            drone_pos[2] < self.dangerous_height or drone_pos[2] > self.arena_height):
            self.out_of_bounds_count += 1
            print(f"Out of bounds! Position: {drone_pos}")
            return True
        
        # Check if max time exceeded
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            self.timeout_count += 1
            print("Timeout!")
            return True
        if (abs(state[7]) > 0.6  or abs(state[8]) > 0.6  # Truncate when the drone is too tilted
        ):
            self.tilted_count += 1
            print("Tilted")
            return True
        
        return False
    except Exception as e:
        print(f"Error in _computeTruncated: {e}")
        return False

  def _computeReward(self):
    """Calculate reward based on distance to target and obstacles"""
    try:
        state = self._getDroneStateVector(0)
        drone_pos = np.array([state[0], state[1], state[2]]) 
        reward = 0
        ang_vel=state[13:16]
        linear_velocity = state[10:13]
        prev_pos = getattr(self, 'prev_drone_pos', drone_pos)
        self.prev_drone_pos = drone_pos.copy()
        cur_dist = np.linalg.norm(np.array(self.TARGET_POS) - drone_pos)
        prev_dist = np.linalg.norm(np.array(self.TARGET_POS) - prev_pos)
        on_way_reward = 4.5*(prev_dist - cur_dist) - 0.001 * np.linalg.norm(ang_vel)
        for obs in self.obstacle_data: 
            # Calculate distance to box surface (approximate
            closest_x = max(obs["position"][0] - obs["dimensions"][0]/2, 
                                min(drone_pos[0], obs["position"][0] + obs["dimensions"][0]/2))
            closest_y = max(obs["position"][1] - obs["dimensions"][1]/2, 
                                        min(drone_pos[1], obs["position"][1] + obs["dimensions"][1]/2))
            closest_z = max(obs["position"][2] - obs["dimensions"][2]/2, 
                                        min(drone_pos[2], obs["position"][2] + obs["dimensions"][2]/2))
                            
            closest_point = np.array([closest_x, closest_y, closest_z])
            dist_to_surface = np.linalg.norm(drone_pos - closest_point)
                            
                            # Penalty for getting too close
            if dist_to_surface < 0.4:
                on_way_reward -= 0.001*(0.4 - dist_to_surface)
        # # Time penalty to encourage efficiency
        passing = False
        for i, key in enumerate(self.racing_setup.keys()):
            if self.passing_flag[i]:
                continue
            if ((self.racing_setup[key][0][0] < drone_pos[0]) and 
                (self.racing_setup[key][0][1] < drone_pos[1] < self.racing_setup[key][1][1]) and
                (self.dangerous_height < drone_pos[2] < self.arena_height)):
                passing = True
                if i == 0: self.first_obstacle_count += 1
                if i == 1: self.second_obstacle_count += 1
                self.passing_flag[i] = True
                break
        collide = False
        for obs in self.obstacle_data:            
                if obs["type"] == "box":
                # Box obstacle
                    if self._check_box_collision(drone_pos, obs["position"], obs["dimensions"]):
                        collide = True
        if passing:
            print("passing")
            print(drone_pos)
            print(self.passing_flag)
            reward = 20
        elif collide:
            reward = -10
        # Out of bounds penalty
        if (drone_pos[0] < 0 or drone_pos[0] > self.arena_lenght or
            drone_pos[1] < 0 or drone_pos[1] > self.arena_width or
            drone_pos[2] < self.dangerous_height or drone_pos[2] > self.arena_height):
            reward = -10
        if (abs(state[7]) > .6 or abs(state[8]) > .6 # Truncate when the drone is too tilted
        ):
            reward = -10
        else:
            reward = on_way_reward

        return float(reward)  # Ensure reward is a float
    except Exception as e:
        print(f"Error in _computeReward: {e}")
        return 0.0

  def step(self, action):
    """Execute one step in the environment"""
    try:
        # Update dynamic branches if using static arena
        if hasattr(self, 'static_arena') and self.static_arena and hasattr(self.static_arena, 'update_dynamic_objects'):
            self.static_arena.update_dynamic_objects()
    
        # Execute step in parent class
        obs, reward, terminated, truncated, info = super().step(action)
    
        # Update step counter
        self.step_counter += 1

        # Update drone marker position if GUI is enabled
        if self.gui and hasattr(self, 'drone_marker_id') and self.drone_marker_id is not None:
            drone_pos = self._getDroneStateVector(0)[:3]
            p.resetBasePositionAndOrientation(
                self.drone_marker_id,
                drone_pos,
                [0, 0, 0, 1]
            )
        
            # Optionally update camera to follow drone
            p.resetDebugVisualizerCamera(
            cameraDistance=1.0,
            cameraYaw=90,
            cameraPitch=-30,
            cameraTargetPosition=[11,1.5,7]
        )
    
        # Return observation, reward, done flag, and info
        return obs, reward, terminated, truncated, info
    
    except Exception as e:
        print(f"Error during environment step: {e}")
        print(traceback.format_exc())
        # Return default values in case of error
        if hasattr(self, 'observation_space'):
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, {}
    
  def reset(self, seed=None, options=None):
    """Reset the environment with optimized performance"""
    self.episode_count += 1
    print(f"\n--- Episode {self.episode_count} ---")
    # Standard reset: Use parent class reset and recreate environment
    # Call parent reset
    obs, info = super().reset(seed=seed, options=options)
            
    # Remove existing objects
    self._remove_objects()
            
    # Recreate environment objects
    self._setup_environment()
            
    self.step_counter = 0

    self.passing_flag = [False, False, False]

    print(f"\n Target Reached {self.target_reached_count}")
    print(f"\n First Obstacle {self.first_obstacle_count}")
    print(f"\n Second Obstacle {self.second_obstacle_count}")
    print(f"\n Out of Bounds {self.out_of_bounds_count}")
    print(f"\n TimeOut {self.timeout_count}")
    print(f"\n Tilted {self.tilted_count}")
    print(f"\n Collision {self.collision_count}")
            
    return obs, info

  def _computeInfo(self):
    """Return additional information"""
    return {
        "target_position": self.target.tolist(),
        "target_reached_count": self.target_reached_count,
        "collision_count": self.collision_count,
        "out_of_bounds_count": self.out_of_bounds_count,
        "timeout_count": self.timeout_count,
        "episode_count": self.episode_count
    }
    
  def get_stats(self):
    """Get environment statistics"""
    return {
        "target_reached_count": self.target_reached_count,
        "collision_count": self.collision_count,
        "out_of_bounds_count": self.out_of_bounds_count,
        "timeout_count": self.timeout_count,
        "episode_count": self.episode_count,
        "success_rate": self.target_reached_count / max(1, self.episode_count)
    }

  # Add this method to make the drone more visible
  def _setup_drone_marker(self):
    """Creates a visual marker to make the drone more visible"""
    if self.gui:
        self.drone_marker = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.3,  # Larger visual radius
            rgbaColor=[0, 1, 0, 0.7]  # Green semi-transparent
        )
        self.drone_marker_id = p.createMultiBody(
            baseMass=0,  # Zero mass so it doesn't affect physics
            baseVisualShapeIndex=self.drone_marker,
            basePosition=[0, 0, 0]  # Will be updated in step()
        )
        print(f"Created drone marker with ID: {self.drone_marker_id}")

  # Video recording methods
  def start_video_recording(self, video_path=None):
    """Start recording video of the simulation"""
    if not self.gui:
        print("Warning: Cannot record video in headless mode. Set gui=True to enable recording.")
        return False
    
    # Create videos directory if it doesn't exist
    if not os.path.exists(self.video_directory):
        os.makedirs(self.video_directory)
    
    # Generate video path if not provided
    if video_path is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        video_path = os.path.join(self.video_directory, f"uav_episode_{timestamp}.mp4")
    
    # Store video path
    self.video_path = video_path
    
    try:
        # Set camera parameters for better recording
        p.resetDebugVisualizerCamera(
            cameraDistance=5.0,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[self.arena_lenght/2, self.arena_width/2, 2.0]
        )
        
        # Start video recording
        self.video_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, self.video_path)
        self.recording = True
        print(f"Started video recording to {self.video_path}")
        return True
    except Exception as e:
        print(f"Error starting video recording: {e}")
        return False

  def stop_video_recording(self):
    """Stop recording video"""
    if hasattr(self, 'recording') and self.recording and hasattr(self, 'video_id') and self.video_id is not None:
        try:
            p.stopStateLogging(self.video_id)
            self.recording = False
            self.video_id = None
            print(f"Video recording saved to {self.video_path}")
            return True
        except Exception as e:
            print(f"Error stopping video recording: {e}")
            return False
    return False
  
