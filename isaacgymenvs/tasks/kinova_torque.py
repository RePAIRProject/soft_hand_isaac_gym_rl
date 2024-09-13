import os
import math
import numpy as np
import random
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.gymutil import AxesGeometry
from isaacgym.gymutil import draw_lines
from isaacgym.torch_utils import tf_vector, quat_conjugate
from isaacgym.torch_utils import quat_mul, to_torch, tensor_clamp

from isaacgymenvs.utils.torch_jit_utils import unscale_transform
from isaacgymenvs.tasks.base.vec_task import VecTask

# from kinova.kinova_isaac_utils import KinovaIsaac
# from kinova.kinova_isaac_utils import sim2real_joints
# from utils.angles import r2d
import torch
    
from pyqb.pyqb import PySoftHandMimicPlugin


class KinovaTorque(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render, wandb_activate):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.robot_position_noise = self.cfg["env"]["robotPositionNoise"]
        self.robot_rotation_noise = self.cfg["env"]["robotRotationNoise"]
        self.robot_dof_noise = self.cfg["env"]["robotDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.use_real_kinova = self.cfg["env"]["useRealKinova"]

         # Create dicts to pass to reward function
        self.reward_settings = {
            "r_dist_scale": self.cfg["env"]["distRewardScale"],
            "r_rot_scale": self.cfg["env"]["rotationRewardScale"],
            "r_fintip_scale": self.cfg["env"]["fingertipRewardScale"],
            "r_fintip_dist_scale": self.cfg["env"]["fingertipDistanceScale"],
            "r_lift_scale": self.cfg["env"]["liftRewardScale"],
            "r_lift_height_scale": self.cfg["env"]["liftHeightRewardScale"],
            "r_actions_reg_scale": self.cfg["env"]["actionsRegularizationRewardScale"],
            "r_object_rot_scale": self.cfg["env"]["objectRotation"],
        }

        # Arm controller type
        self.arm_control_type = self.cfg["env"]["armControlType"]
        assert self.arm_control_type in {"pos", "vel"},\
            "Invalid control type specified. Must be one of: {pos, vel}"

        # Hand controller type
        self.hand_control_type = self.cfg["env"]["handControlType"]
        assert self.hand_control_type in {"binary", "effort"},\
            "Invalid control type specified. Must be one of: {binary, effort}"

        # dimensions
        num_obs = 17
        num_actions = 8 # 7 dof for arm + 1 dof for the hand
        self.cfg["env"]["numObservations"] = num_obs
        self.cfg["env"]["numActions"] = num_actions

        # Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed

        # Tensor placeholders
        self._root_state = None             # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None                     # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._contact_forces = None     # Contact forces in sim
        self._eef_state = None  # end effector state (at grasping point)
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._arm_control = None  # Tensor buffer for controlling arm
        self._gripper_control = None  # Tensor buffer for controlling gripper
        self._pos_control = None            # Position actions
        self._vel_control = None         # Torque actions
        self._effort_control = None         # Torque actions
        self._global_indices = None         # Unique indices corresponding to all envs in flattened array
        self.up_axis = "z"

        self.prev_u_hand = 0.0


        super().__init__(config=self.cfg, rl_device=rl_device,
                sim_device=sim_device, graphics_device_id=graphics_device_id,
                headless=headless,
                virtual_screen_capture=virtual_screen_capture,
                force_render=force_render)

        # Kinova + Seed defaults
        # robot_default_dof_pos = [0., 0., 0., 0., 0., 0., 0.] + [0.0] * 33       # zero position
        robot_default_dof_pos = [0, 0.5, 0, 1.85, 0, -0.8, -1.6] + [0.0] * 33   # home position
        # robot_default_dof_pos = [0, 0.5, 0, 1.85, 0, -0.8, -1.6] + [0.0] * 33   # home position
        # robot_default_dof_pos = [0, 0.5, 0, 1.85, 0, -0.8, -1.6] + [0.0] * 34   # home position
        # robot_default_dof_pos = [0, 0.5, 0, 1.85, 0, -1.0, 1.57+0.35] + [0.0] * 34   # hand facing up
        robot_default_dof_pos[36] = 0.0 # 1.57
        self.robot_default_dof_pos = to_torch(robot_default_dof_pos, device=self.device)

        if self.use_real_kinova:
            from kinova.kinova_api import KinovaArm
            self.real_arm = KinovaArm()
            self.kinova_isaac = KinovaIsaac(real_arm = self.real_arm,
                                            sim_joint_states = self._dof_state[:, :7, :],
                                            agent_idx = 0,
                                            max_speed = 80,
                                            stop_condition = 0.05)
            
        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()

        self.initial_fresco_height = torch.mean(self.states["object_pos"][:, 2]).cpu().numpy().item()

        self.hand_virtual_joint_idxs = [8, 10, 12, 15, 17, 19, 22, 24, 26, 29, 31, 33, 36, 38]
        self.hand_real_joint_idxs = [7, 9, 11, 13, 14, 16, 18, 20, 21, 23, 25, 27, 28, 30, 32, 34, 35, 37, 39]
        self.knuckle_joint_idxs = [35, 7, 21, 28, 14]  # timrl 

        self.all_envs = [self.gym.get_env(self.sim, i) for i in range(self.num_envs)]
        self.robot_target_positions = torch.zeros(self.num_envs, self.num_dofs)   
        self.robot_target_velocities = torch.zeros(self.num_envs, self.num_dofs)   
        self.robot_target_efforts = torch.zeros(self.num_envs, self.num_dofs) 

        self.arm_upper_limits_urdf = [3.1416, 2.25, 3.1416, 2.58, 3.1416, 2.09, 3.1416]
        self.arm_lower_limits_urdf = [-3.1416, -2.25, -3.1416, -2.58, -3.1416, -2.09, -3.1416]
        max_vel = 0.6727 * 2
        self.arm_velocity_limits_urdf = to_torch([max_vel, max_vel, max_vel, max_vel, max_vel, max_vel, max_vel], device=self.device)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(-1.5, -1.5, 1.0)
            cam_target = gymapi.Vec3(0.0, 0.0, 1.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        
        self.hand = torch.zeros(self.num_envs, 1) # -2.1
        self.direction = 1
        self.hand_closed = False

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets")
        robot_asset_file = "urdf/qb_hand/urdf/arm_qbhand_iit.urdf"

        # == Load robot asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        asset_options.override_com = True
        asset_options.override_inertia = True

        robot_asset = self.gym.load_asset(self.sim, asset_root, robot_asset_file, asset_options)

        self.num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.num_robot_dofs = self.gym.get_asset_dof_count(robot_asset)

        print("num robot bodies: ", self.num_robot_bodies)
        print("num robot dofs: ", self.num_robot_dofs)

        # set robot dof properties
        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)
        self.robot_dof_lower_limits = []
        self.robot_dof_upper_limits = []
 
        for i in range(self.num_robot_dofs):
            self.robot_dof_lower_limits.append(robot_dof_props['lower'][i])
            self.robot_dof_upper_limits.append(robot_dof_props['upper'][i])

        # arm properties
        if self.arm_control_type == 'vel':
            # For velocity control set stiffness to 0 and damping to any value
            # Source: https://docs.omniverse.nvidia.com/isaacsim/latest/advanced_tutorials/tutorial_advanced_joint_tuning.html
            robot_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_VEL)
            robot_dof_props["armature"][:7].fill(0.001)
            robot_dof_props["stiffness"][:7].fill(0.0)
            robot_dof_props["damping"][:7].fill(10.0)
            robot_dof_props["effort"][:7].fill(30.00)
        if self.arm_control_type == 'pos':
            # How to tune stiffness and damping
            # https://docs.omniverse.nvidia.com/isaacsim/latest/advanced_tutorials/tutorial_advanced_joint_tuning.html
            # 1. Choose an armature/inertia
            # 2. Set damping to 0
            # 3. Move arm from home position to zero position with stiffness 1.0
            # 4. Increases stiffness until arm goes to target position and doesn't oscilate much around it
            # 5. Add damping 1 order of magnitude below stiffness to remove the remaining oscilations
            robot_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
            robot_dof_props["armature"][:7].fill(0.001)
            robot_dof_props["stiffness"][:7].fill(200)
            robot_dof_props["damping"][:7].fill(40.0)
            robot_dof_props["friction"][:7].fill(0.0)
            robot_dof_props["effort"][:7].fill(30.00)

        # hand properties
        if self.hand_control_type == 'effort':
            robot_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_EFFORT)
            robot_dof_props["armature"][7:].fill(0.01)
            robot_dof_props["stiffness"][7:].fill(0) 
            # robot_dof_props["damping"][7:].fill(8.0)
            robot_dof_props["damping"][7:].fill(80.0)
            robot_dof_props["friction"][7:].fill(0.0)
            # robot_dof_props["effort"][7:].fill(0.1)
            robot_dof_props["effort"][7:].fill(1.0)
        if self.hand_control_type == 'binary':
            robot_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
            robot_dof_props["armature"][7:].fill(0.01)
            robot_dof_props["stiffness"][7:].fill(80.0)
            robot_dof_props["damping"][7:].fill(8.0)
            robot_dof_props["friction"][7:].fill(0.0)
            robot_dof_props["effort"][7:].fill(10)
            
        self.robot_dof_lower_limits = to_torch(self.robot_dof_lower_limits, device=self.device)
        self.robot_dof_upper_limits = to_torch(self.robot_dof_upper_limits, device=self.device)
        self.robot_dof_lower_limits[[0, 2, 4, 6]] = -3.14
        self.robot_dof_upper_limits[[0, 2, 4, 6]] = 3.14
        
        if self.hand_control_type == 'binary':
            self.u_hand_min = to_torch([0.0], device=self.device)
            self.u_hand_max = to_torch([math.pi/4], device=self.device)
        else:
            self.u_hand_min = to_torch([-3.0], device=self.device)
            self.u_hand_max = to_torch([3.0], device=self.device)

        
        # Define start pose for robot
        robot_start_pose = gymapi.Transform()
        # robot_start_pose.p = gymapi.Vec3(0.072, 0.52, 0.27) # (robot_radius(4.5cm) +2.7cm, robot position relative to table, table_heigt + black_base_height)
        robot_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.27)
        robot_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # == Create table asset
        # table_pos = [0.5, 0.5, 0.3]
        table_length = 1.8 # 0.8
        table_width = 1.8
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_opts.disable_gravity = True
        table_asset = self.gym.create_box(self.sim, *[table_length, table_width, table_thickness], table_opts) # using real table dims
        # table_dof_props = self.gym.get_asset_dof_properties(table_asset)

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(table_length/2 - 0.075, -table_width/2 + 0.515, 0.25 - table_thickness/2)
        # table_start_pose.p = gymapi.Vec3(.7, 0., 0.25)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # == Load object asset
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.001
        asset_options.fix_base_link = False  # True
        asset_options.thickness = 0.0001
        asset_options.disable_gravity = False
        asset_options.use_mesh_materials = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.override_inertia = True
        asset_options.override_com = True
        asset_options.flip_visual_attachments = False
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.convex_hull_downsampling = 30

        object_asset = self.gym.load_asset(self.sim, asset_root,
                                           "urdf/qb_hand/urdf/fresco.urdf",
                                           asset_options)
        
        object_shape_props = self.gym.get_asset_rigid_shape_properties(object_asset)
        object_shape_props[0].friction = 1.0
        
        init_object_height = 0.25
        # self.object_default_state = torch.tensor([0.75, 0.40, 0.8, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # hand facing up
        self.object_default_state = torch.tensor([0.6, -0.4, init_object_height, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # fixed pos
        # self.object_default_state = torch.tensor([0.6, -0.4, 0.32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # fixed pos
        # self.object_default_state = torch.tensor([0.5, -0.6, 0.3, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0.6, -0.4, init_object_height)
        # object_start_pose.p = gymapi.Vec3(0.6, -0.4, 0.32)
        # object_start_pose.p = gymapi.Vec3(0.4, 0.0, 0.01)
        object_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        

        # compute aggregate size
        num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        num_robot_shapes = self.gym.get_asset_rigid_shape_count(robot_asset)
        num_table_bodies = self.gym.get_asset_rigid_body_count(table_asset)
        num_table_shapes = self.gym.get_asset_rigid_shape_count(table_asset)
        num_object_bodies = self.gym.get_asset_rigid_body_count(object_asset)
        num_object_shapes = self.gym.get_asset_rigid_shape_count(object_asset)
        max_agg_bodies = num_robot_bodies + num_table_bodies + num_object_bodies
        max_agg_shapes = num_robot_shapes + num_table_shapes + num_object_shapes
        
        self.robots = []
        self.envs = []

        # Create environments
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            robot_actor = self.gym.create_actor(env_ptr, robot_asset, robot_start_pose, "robot", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, robot_actor, robot_dof_props)

            # Create table
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 0, 0)
            
            # Create object actor
            self._object_id = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 0, 0)
                    
            self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.robots.append(robot_actor)

        self.pyqb = PySoftHandMimicPlugin(num_envs, robot_asset, self.sim, self.gym.find_asset_dof_index, self.gym.get_sim_time, device=self.device)
        
        self._init_object_state = torch.zeros(self.num_envs, 13, device=self.device)
        
        # == Arm contacts
        arm_body_names = ['base_link', 'shoulder_link', 'HalfArm1_link', 'half_arm_2_link', 
                          'forearm_link', 'spherical_wrist_1_link', 'spherical_wrist_2_link', 'bracelet_with_vision_link']
        self.rigid_body_arm_inds = torch.zeros(len(arm_body_names), dtype=torch.long, device=self.device)
        for i, n in enumerate(arm_body_names):
            self.rigid_body_arm_inds[i] = self.gym.find_actor_rigid_body_handle(env_ptr, robot_actor, n)

        # == Hand Contacts
        hand_body_names = ['right_hand_v1_2_research_palm_link',          'right_hand_v1_2_research_index_knuckle_link',  # 'right_hand_v1_2_research_base_link',            
                           'right_hand_v1_2_research_index_proximal_link',  'right_hand_v1_2_research_index_middle_link',  'right_hand_v1_2_research_little_knuckle_link',  
                           'right_hand_v1_2_research_little_proximal_link', 'right_hand_v1_2_research_little_middle_link', 'right_hand_v1_2_research_middle_knuckle_link',  
                           'right_hand_v1_2_research_middle_proximal_link', 'right_hand_v1_2_research_middle_middle_link', 'right_hand_v1_2_research_ring_knuckle_link',
                           'right_hand_v1_2_research_ring_proximal_link',   'right_hand_v1_2_research_ring_middle_link',   'right_hand_v1_2_research_thumb_knuckle_link', 
                           'right_hand_v1_2_research_thumb_proximal_link']
        self.rigid_body_hand_inds = torch.zeros(len(hand_body_names), dtype=torch.long, device=self.device)
        for i, n in enumerate(hand_body_names):
            self.rigid_body_hand_inds[i] = self.gym.find_actor_rigid_body_handle(env_ptr, robot_actor, n)

        # == Fingertips Contacts
        fingertips_body_names = ['right_hand_v1_2_research_thumb_distal_link', 'right_hand_v1_2_research_index_distal_link', 'right_hand_v1_2_research_ring_distal_link',  
                                 'right_hand_v1_2_research_little_distal_link', 'right_hand_v1_2_research_middle_distal_link']
        self.rigid_body_fingertip_inds = torch.zeros(len(fingertips_body_names), dtype=torch.long, device=self.device)
        for i, n in enumerate(fingertips_body_names):
            self.rigid_body_fingertip_inds[i] = self.gym.find_actor_rigid_body_handle(env_ptr, robot_actor, n)
        
        # == Fresco Contacts
        fresco_body_names = ["fresco"]
        self.rigid_body_fresco_inds = torch.zeros(len(fresco_body_names), dtype=torch.long, device=self.device)
        self.rigid_body_fresco_inds[0] = self.gym.find_actor_rigid_body_handle(env_ptr, self._object_id, "fresco")

        # == Table Contacts
        table_body_names = ["box"]
        self.rigid_body_table_inds = torch.zeros(len(table_body_names), dtype=torch.long, device=self.device)
        self.rigid_body_table_inds[0] = self.gym.find_actor_rigid_body_handle(env_ptr, table_actor, "box")
       
        # Setup data
        self.init_data()
        

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        robot_handle = 0

        self.handles = {
            "grip_site": self.gym.find_actor_rigid_body_handle(env_ptr,
                                                               robot_handle,
                                                               "qbhand_end_effector_link"),
            "fftip": self.gym.find_actor_rigid_body_handle(env_ptr,
                                                           robot_handle,
                                                           "right_hand_v1_2_research_index_distal_link"),
            "mitip": self.gym.find_actor_rigid_body_handle(env_ptr,
                                                           robot_handle,
                                                           "right_hand_v1_2_research_middle_distal_link"),
            "thtip": self.gym.find_actor_rigid_body_handle(env_ptr,
                                                           robot_handle,
                                                           "right_hand_v1_2_research_thumb_distal_link"),
        }

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._dof_force_tensor = gymtorch.wrap_tensor(_dof_force_tensor).view(self.num_envs, -1)
        self._contact_forces = gymtorch.wrap_tensor(_net_contact_forces).view(self.num_envs, -1, 3)

        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._eef_state = self._rigid_body_state[:, self.handles["grip_site"], :]
        self._fftip_state = self._rigid_body_state[:, self.handles["fftip"], :]
        self._mitip_state = self._rigid_body_state[:, self.handles["mitip"], :]
        self._thtip_state = self._rigid_body_state[:, self.handles["thtip"], :]
        self._object_state = self._root_state[:, self._object_id, :]

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._vel_control = torch.zeros_like(self._pos_control)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize indices
        self._global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)

        self.down_axis = to_torch([0, 0, -1], device=self.device).repeat((self.num_envs, 1))
        self.grasp_up_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))

    def _update_states(self):
        self.states.update({
            # Robot
            "q_arm": self._q[:, :7],
            "q_hand": self._q[:, 7:],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            "fftip_pos" : self._fftip_state[:, :3],
            "mitip_pos" : self._mitip_state[:, :3],
            "thtip_pos" : self._thtip_state[:, :3],
            # Object
            "object_quat": self._object_state[:, 3:7],
            "object_pos": self._object_state[:, :3],
            "object_pos_relative": self._object_state[:, :3] - self._eef_state[:, :3],
            "object_fftip_pos_relative": self._object_state[:, :3] - self._fftip_state[:, :3],
            "to_height": 1.2 - self._object_state[:, 2].unsqueeze(1)
        })

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Refresh states
        self._update_states()

    def compute_observations(self):
        self._refresh()
        obs = ["object_pos", "object_quat", "object_pos_relative", "eef_pos", "eef_quat"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        maxs = {ob: torch.max(self.states[ob]).item() for ob in obs}

        return self.obs_buf

    def reset_idx(self, env_ids):
        self.i = 0
        if self.use_real_kinova:
            default_angles = sim2real_joints(self.robot_default_dof_pos[:7])
            self.real_arm.move_angular(default_angles)

        pos = tensor_clamp(self.robot_default_dof_pos.unsqueeze(0),
            self.robot_dof_lower_limits.unsqueeze(0), self.robot_dof_upper_limits)
        
        self.prev_u_hand = 0.0
        
        # add randomization to the joints of kinova
        pos = pos.repeat((env_ids.shape[0], 1))
        # pos[:, :7] += 0.1 # * torch.rand(env_ids.shape[0], 7).to(self.device)

        # Reset object states by sampling random poses
        self._reset_init_object_state(env_ids=env_ids)
        self._object_state[env_ids] = self.object_default_state.cuda() # self._init_object_state[env_ids]

        # Reset the internal obs accordingly
        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids, :] = pos
        self._vel_control[env_ids, :] = torch.zeros_like(pos)
        self._effort_control[env_ids, :] = torch.zeros_like(pos)

        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._pos_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_velocity_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._vel_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._effort_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        # Update object states
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, -1].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _reset_init_object_state(self, env_ids):
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_cube_state = self.object_default_state.repeat(num_resets, 1).to(device=self.device)
        sampled_cube_state[:, :2] = sampled_cube_state[:, :2] + \
                                    1.0 * self.start_position_noise * \
                                    (torch.rand(num_resets, 2, device=self.device) - 0.5)

        # Sample rotation value
        if self.start_rotation_noise > 0:
            aa_rot = torch.zeros(num_resets, 3, device=self.device)
            aa_rot[:, 2] = 2.0 * self.start_rotation_noise * (torch.rand(num_resets, device=self.device) - 0.5)
            sampled_cube_state[:, 3:7] = quat_mul(axisangle2quat(aa_rot), sampled_cube_state[:, 3:7])
    
        # Lastly, set these sampled values as the new init state
        self._init_object_state[env_ids, :] = sampled_cube_state


    def pre_physics_step(self, actions):
                
        self.actions = actions.clone().to(self.device)

        u_arm, u_hand = self.actions[:, :7], self.actions[:, 7:]
        
        if self.arm_control_type == "vel":
            u_arm = unscale_transform(u_arm,
                                      -self.arm_velocity_limits_urdf,
                                      self.arm_velocity_limits_urdf)
            u_arm = tensor_clamp(u_arm,
                                 -self.arm_velocity_limits_urdf,
                                 self.arm_velocity_limits_urdf)    
        elif self.arm_control_type == "pos":
            u_arm = unscale_transform(u_arm,
                                    self.robot_dof_lower_limits[:7],
                                    self.robot_dof_upper_limits[:7])
            u_arm = tensor_clamp(u_arm,
                                self.robot_dof_lower_limits[:7],
                                self.robot_dof_upper_limits[:7])
        
        u_hand = unscale_transform(u_hand, self.u_hand_min, self.u_hand_max)
        u_hand = tensor_clamp(u_hand, self.u_hand_min, self.u_hand_max)
        
        # Current pos and vel state of each DOF
        pos_tensor = self._dof_state[:, :, 0]
        vel_tensor = self._dof_state[:, :, 1]
        
        # == Arm control
        move_arm = True
        if move_arm:
            if self.arm_control_type == 'vel':
                self._vel_control[:, :7] = u_arm # * 0.0
                # self._vel_control[:, 7:] = vel_tensor[:, 7:]
                self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(self._vel_control))
            if self.arm_control_type == 'pos':
                self._pos_control[:, :7] = u_arm # * 0.0
                pos_tensor[:, :7] = u_arm
                # self._pos_control[:, 7:] = pos_tensor[:, 7:]
                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))

        # u_hand = torch.ones([self.num_envs, 1], device=self.device) * self.prev_u_hand
        # if self.prev_u_hand < 0.1:
        #     self.prev_u_hand += 0.02

        # == Hand control
        move_hand = True
        if move_hand:
            if self.hand_control_type == 'binary':
                # self._pos_control[:, :7] = pos_tensor[:, :7]                  # make sure arm joints position is not affected here
                self._pos_control[:, self.hand_real_joint_idxs] = u_hand
                self._pos_control[:, self.hand_virtual_joint_idxs] = 0.0      # don't move the virtual joints
                self._pos_control[:, self.knuckle_joint_idxs[1:]] = 0.0       # don't move the knuckle joints
                self._pos_control[:, 35:36] = u_hand * 2.0                    # the only knuckle we move is the thumb, which has 2x the range of other joints
                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
            if self.hand_control_type == 'effort':
                self.hand_default_dof_pos = torch.zeros_like(pos_tensor, dtype=torch.float, device=self.device, requires_grad=False)
                tau = self.pyqb.OnUpdateSoftSyn(u_hand[:, 0], pos_tensor, vel_tensor)
                K = 30.0 # or divide hand effort by 10 and this one also by 10
                self._effort_control = K * torch.tensor(tau, device=self.device, dtype=torch.float32)
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))

        if self.use_real_kinova:
            self.kinova_isaac.replicate_sim_movement()

    
    def post_physics_step(self):
        self.progress_buf += 1
     
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()       
        self.compute_reward(self.actions) 
  
    def compute_reward(self, actions):

        self.rew_buf[:], self.reset_buf[:], reward_dict = compute_robot_reward(
                self.reset_buf, self.progress_buf, self.device, self.actions, self.states,
                self.grasp_up_axis, self.down_axis,
                self.num_envs,
                self.reward_settings, self.max_episode_length,
                self._contact_forces, self.rigid_body_arm_inds, self.rigid_body_fingertip_inds,
                self.rigid_body_hand_inds, self.rigid_body_table_inds,
                self.rigid_body_fresco_inds, self.initial_fresco_height)
    

#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def compute_robot_reward(
    reset_buf, progress_buf, device, actions, states, grasp_up_axis, down_axis,
    num_envs, reward_settings, max_episode_length, contact_forces, arm_inds, 
    fingertip_inds, hand_inds, table_inds, fresco_inds, initial_fresco_height):

    # type: (Tensor, Tensor, str, Tensor, Dict[str, Tensor], Tensor, Tensor, int, Dict[str, float], float,  Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]
    
        rewards = torch.zeros(num_envs, device=device)
        reward_dict = {"Distance Reward":               torch.zeros(num_envs, device=device),
                        "Rotation Reward":              torch.zeros(num_envs, device=device),
                        "Fingertips Reward":            torch.zeros(num_envs, device=device),
                        "Fingert Reward":               torch.zeros(num_envs, device=device),
                        "Thumb Reward":                 torch.zeros(num_envs, device=device),
                        "Lift Reward":                  torch.zeros(num_envs, device=device),
                        "Lift Height Reward":           torch.zeros(num_envs, device=device),
                        "Action Regularization Reward": torch.zeros(num_envs, device=device),
                        "Thumb Middle Reward":          torch.zeros(num_envs, device=device),
                        "Fingertips contact bonus":     torch.zeros(num_envs, device=device),
                        "Table contact bonus":          torch.zeros(num_envs, device=device),
                        "Object Rotation Reward":       torch.zeros(num_envs, device=device)}


        # distance from grasp link to the object  
        d_eef = torch.norm(states["object_pos_relative"], dim=-1)
        dist_reward =  1 - torch.tanh(2 * d_eef)
        
        # distance from fingertips to the object
        d_fftip = torch.norm(states["fftip_pos"] - states["object_pos"], dim=-1)
        d_thtip = torch.norm(states["thtip_pos"] - states["object_pos"], dim=-1)
        fintip_reward = 1 - torch.tanh(reward_settings["r_fintip_dist_scale"] * (d_fftip + d_thtip) / 2.)
        
        # grasp axis should look down
        grasp_axis = tf_vector(states["eef_quat"], grasp_up_axis)
        dot = torch.bmm(grasp_axis.view(num_envs, 1, 3), down_axis.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        rot_reward = torch.sign(dot) * dot ** 2

        # reward for lifting object
        object_height = states["object_pos"][:, 2]
        object_lifted = object_height > (initial_fresco_height + 0.04)  # 0.45  # 0.3  # here is the reward that checks if object is lifted
        lift_reward = object_lifted		# 1 for lifted; 0 for not
        lift_height = object_height - (initial_fresco_height + 0.05)  # 0.25		0.05 table thickness.
        object_grasped = object_height > 0.35

        # lift_height = torch.where(lift_height < 0, 0.0, lift_height)
        
        # Above the table bonus
        # object_above = (states["object_pos"][:, 2] - 0.25) > 0.015
        # lift_bonus_reward = torch.zeros_like(d_fftip)
        # lift_bonus_reward = torch.where(object_above, lift_bonus_reward + 0.5, lift_bonus_reward) 

        # Object to goal height distance
        # og_d = torch.norm(states["to_height"], p=2, dim=-1)
        # og_dist_reward = torch.zeros_like(d_fftip)
        # og_dist_reward = torch.where(object_above, 1.0 / (0.04 + og_d), og_dist_reward)

        # Bonus if object is near goal height
        # og_bonus_reward = torch.zeros_like(og_dist_reward)
        # og_bonus_reward = torch.where(og_d <= 0.04, og_bonus_reward + 0.5, og_bonus_reward)

        # object rotation (object_quat) 
        # check the z axis of the object and the grasp to see if it matches
        
        # use object_lifted as an 'if' to apply on when the object is lifted
        grasp_axis = tf_vector(states["object_quat"], grasp_up_axis)
        dot = torch.bmm(grasp_axis.view(num_envs, 1, 3), grasp_up_axis.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        objrot_reward = object_grasped * torch.sign(dot) * dot ** 2

        # fingertips contact reward
        num_fingertip_contacts = torch.sum(torch.norm(contact_forces[:, fingertip_inds, :], dim=2) > 1.0, dim=-1)
        switch = (torch.abs(torch.norm(contact_forces[:, fresco_inds, :], dim=-1)) > (9.81 * 0.268 * 3)).squeeze()
        fingertips_contacts_reward = 0.20 * num_fingertip_contacts * switch

        # == Nothing makes contact with the table bonus
        table_collision = (torch.norm(contact_forces[:, table_inds, :], dim=-1)).squeeze()
                            # (torch.norm(contact_forces[:, table_inds, :], dim=-1) == 0.0).squeeze()
        table_collision_penalty = 1.0 * table_collision # 5.0
        # print("Switch:", switch[0], "Table:", table_collision[0])

        # Regularization on the actions of the arm
        action_penalty = torch.sum(actions[:, :7] ** 2, dim=-1)
        
        action_incentive = torch.sum(actions[:, 7:] ** 2, dim=-1)

        rewards = reward_settings["r_dist_scale"] * dist_reward \
                + reward_settings["r_rot_scale"] * rot_reward \
                + reward_settings["r_fintip_scale"] * fintip_reward \
                + reward_settings["r_lift_scale"] * lift_reward \
                + reward_settings["r_lift_height_scale"] * lift_height \
                + reward_settings["r_object_rot_scale"] * objrot_reward \
                - reward_settings["r_actions_reg_scale"] * action_penalty \
                # + 0.1 * action_incentive \
                # + 10 * fingertips_contacts_reward \
                # - table_collision_penalty \
                
        # Return rewards in dict for debugging
        reward_dict["Distance Reward"]              = reward_settings["r_dist_scale"] * dist_reward
        reward_dict["Rotation Reward"]              = reward_settings["r_rot_scale"] * rot_reward
        reward_dict["Fingertips Reward"]            = reward_settings["r_fintip_scale"] * fintip_reward
        reward_dict["Fingert Reward"]               = reward_settings["r_fintip_scale"] * d_fftip
        reward_dict["Thumb Reward"]                 = reward_settings["r_fintip_scale"] * d_thtip
        reward_dict["Lift Reward"]                  = reward_settings["r_lift_scale"] * lift_reward
        reward_dict["Lift Height Reward"]           = reward_settings["r_lift_height_scale"] * lift_height
        reward_dict["Action Regularization Reward"] = reward_settings["r_actions_reg_scale"] * action_penalty
        reward_dict["Object Rotation Reward"]       = reward_settings["r_object_rot_scale"] * objrot_reward
        
        # # == Compute resets
        # # Object below table height
        object_below = (0.1 - states["object_pos"][:, 2]) > 0.04
        reset_buf = torch.where(object_below, torch.ones_like(reset_buf), reset_buf)
        
        # # - Arm makes contact with something
        # arm_collision = torch.any(torch.norm(contact_forces[:, arm_inds, :], dim=2) > 1.0, dim=1)
        # reset_buf = torch.where(arm_collision, torch.ones_like(reset_buf), reset_buf)

        # Max episodes
        reset_buf = torch.where((progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)


        return rewards, reset_buf, reward_dict

    # dist_to_target = torch.norm(states["object_pos_relative"], dim=-1)
    # dist_reward = (1 - torch.tanh(1.0 * dist_to_target))

    # # Regularization on the actions
    # action_penalty = torch.sum(actions ** 2, dim=-1)

    # rewards = reward_settings["r_dist_scale"] * dist_reward \
    #           - reward_settings["r_actions_reg_scale"] * action_penalty

    # # == Compute resets

    # # Arm makes contact with something
    # arm_collision = torch.any(torch.norm(contact_forces[:, arm_inds, :], dim=2) > 1.0, dim=1)
    # reset_buf = torch.where(arm_collision, torch.ones_like(reset_buf), reset_buf)

    # # Hand makes contact with the table (except middle finger tip)
    # hand_collision = torch.any(torch.norm(contact_forces[:, hand_inds, :], dim=2) > 1.0, dim=1)
    # reset_buf = torch.where(hand_collision, torch.ones_like(reset_buf), reset_buf)

    # # Max episodes
    # reset_buf = torch.where((progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)

    # # Return rewards in dict for debugging
    # reward_dict = {"Distance Reward": reward_settings["r_dist_scale"] * dist_reward,
    #                "Action Regularization Reward": reward_settings["r_actions_reg_scale"] * action_penalty}


    # return rewards, reset_buf, reward_dict

@torch.jit.script
def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

@torch.jit.script
def random_pos(num: int, device: str) -> torch.Tensor:
    radius = 0.6
    height = 0.3
    # sample theta of point
    theta = 2 * np.pi * torch.rand(num, dtype=torch.float, device=device)

    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    z = torch.tensor([height], device=device).repeat((num, 1))

    return torch.cat((x[:, None], y[:, None], z), dim=-1)

    
@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat

@torch.jit.script
def to_rads(x):
    return (x * 3.14159265359) / 180.


@torch.jit.script
def scale(x, lower, upper):
    return (0.5 * (x + 1.0) * (upper - lower) + lower)

# @torch.jit.script
def remap(x, l1, h1, l2, h2):
    return l2 + (x - l1) * (h2 - l2) / (h1 - l1)
