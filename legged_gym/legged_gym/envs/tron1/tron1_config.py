""" Basic model configs for Unitree Go2 """
import numpy as np
import os.path as osp

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# go2_action_scale = 0.5
# go2_const_dof_range = dict(
#     Hip_max= 1.0472,
#     Hip_min= -1.0472,
#     Front_Thigh_max= 3.4907,
#     Front_Thigh_min= -1.5708,
#     Rear_Thingh_max= 4.5379,
#     Rear_Thingh_min= -0.5236,
#     Calf_max= -0.83776,
#     Calf_min= -2.7227,
# )

class Tron1RoughCfg( LeggedRobotCfg ):
    class env:
        num_envs = 4096
        num_observations = None # No use, use obs_components
        num_privileged_obs = None # No use, use privileged_obs_components

        use_lin_vel = False # to be decided
        num_actions = 8
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

        obs_components = [
            "lin_vel",  # 3
            "ang_vel",  # 3
            "projected_gravity",    # 3
            "commands", # 3
            "dof_pos",  # 8
            "dof_vel",  # 8
            "last_actions", # 8
            "height_measurements",  #
        ]

    class sensor:
        class proprioception:
            obs_components = ["ang_vel", "projected_gravity", "commands", "dof_pos", "dof_vel"]
            latency_range = [0.005, 0.045] # [s]
            latency_resampling_time = 5.0 # [s]

    class terrain:
        selected = "TerrainPerlin"
        mesh_type = None
        measure_heights = True
        # x: [-0.5, 1.5], y: [-0.5, 0.5] range for go2
        measured_points_x = [i for i in np.arange(-0.5, 1.51, 0.1)]
        measured_points_y = [i for i in np.arange(-0.5, 0.51, 0.1)]
        horizontal_scale = 0.025 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 5 # [m]
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 4.
        terrain_width = 4.
        num_rows= 16 # number of terrain rows (levels)
        num_cols = 16 # number of terrain cols (types)
        slope_treshold = 1.

        TerrainPerlin_kwargs = dict(
            zScale= 0.07,
            frequency= 10,
        )

    # class terrain:
    #     selected = None
    #     mesh_type = "plane"
    #     measure_heights = True
    #     static_friction = 1.0
    #     dynamic_friction = 1.0
    #     restitution = 0.
    #     measured_points_x = [i for i in np.arange(-0.5, 1.51, 0.1)]
    #     measured_points_y = [i for i in np.arange(-0.5, 0.51, 0.1)]
    #     curriculum = False
    
    class commands( LeggedRobotCfg.commands ):
        heading_command = False
        resampling_time = 5. # [s]
        lin_cmd_cutoff = 0.2
        ang_cmd_cutoff = 0.2
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-0.5, 0.5]
            ang_vel_yaw = [-1.0, 1.0]

    class init_state( LeggedRobotCfg.init_state ):
        # pos = [0.0, 0.0, 0.7 + 0.1664] # [0.0, 0.0, 0.7 + 0.1664]  # x,y,z [m]
        pos = [0.0, 0.0, 0.7 + 0.1664] # [0.0, 0.0, 0.7 + 0.1664]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # target angles when action = 0.0
            "abad_L_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "ankle_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_R_Joint": 0.0,
            "ankle_R_Joint": 0.0,
        }

    class gait:
        num_gait_params = 4
        resampling_time = 5  # time before command are changed[s]
        touch_down_vel = 0.0

        class ranges:
            frequencies = [1.0, 1.5]
            # frequencies = [1.0 + 0.3, 1.5 + 0.3]
            offsets = [0.5, 0.5]  # offset is hard to learn
            # durations = [0.3, 0.8]  # small durations(<0.4) is hard to learn
            # frequencies = [2, 2]
            # offsets = [0.5, 0.5]
            durations = [0.5, 0.5]
            swing_height = [0.5, 0.15]

    class control( LeggedRobotCfg.control ):
        stiffness = {
            "abad_L_Joint": 45,
            "hip_L_Joint": 45,
            "knee_L_Joint": 45,
            "abad_R_Joint": 45,
            "hip_R_Joint": 45,
            "knee_R_Joint": 45,
            "ankle_L_Joint": 45,
            "ankle_R_Joint": 45,
        }  # [N*m/rad]
        damping = {
            "abad_L_Joint": 1.5,
            "hip_L_Joint": 1.5,
            "knee_L_Joint": 1.5,
            "abad_R_Joint": 1.5,
            "hip_R_Joint": 1.5,
            "knee_R_Joint": 1.5,
            "ankle_L_Joint":  0.8,
            "ankle_R_Joint":  0.8,
        }  # [N*m*s/rad]
        action_scale = 0.25
        computer_clip_torque = False
        motor_clip_torque = True

        

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/tron1/urdf/tron1.urdf'
        name = "tron1"
        foot_name = "ankle"
        foot_radius = 0.0

        penalize_contacts_on = ["knee", "hip"]
        terminate_after_contacts_on = ["abad", "base"]
        replace_cylinder_with_capsule = False
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        
         # sdk_dof_range = go2_const_dof_range
        # dof_velocity_override = 35.

    class termination:
        termination_terms = [
            "roll",
            "pitch",
        ]

        roll_kwargs = dict(
            threshold= 3.0, # [rad]
        )
        pitch_kwargs = dict(
            threshold= 3.0, # [rad] # for leap, jump
        )

    class domain_rand( LeggedRobotCfg.domain_rand ):
        randomize_com = True
        class com_range:
            x = [-0.2, 0.2]
            y = [-0.1, 0.1]
            z = [-0.05, 0.05]
        friction_range = [0.2, 1.6]
        added_mass_range = [-0.5, 2]
        randomize_friction = True
        randomize_base_mass = False

        randomize_motor = False
        leg_motor_strength_range = [0.8, 1.2]
        
        ######## 增加对初始位姿的鲁棒性  ###########
        # init_base_pos_range = dict(
        #     x= [0.05, 0.6],
        #     y= [-0.25, 0.25],
        # )
        # init_base_rot_range = dict(
        #     roll= [-0.75, 0.75],
        #     pitch= [-0.75, 0.75],
        # )
        # init_base_vel_range = dict(
        #     x= [-0.2, 1.5],
        #     y= [-0.2, 0.2],
        #     z= [-0.2, 0.2],
        #     roll= [-1., 1.],
        #     pitch= [-1., 1.],
        #     yaw= [-1., 1.],
        # )
        # init_dof_vel_range = [-5, 5]
        ########################################

        push_robots = True 
        max_push_vel_xy = 0.5 # [m/s]
        push_interval_s = 2

    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            # tracking_lin_vel = 3.
            tracking_lin_vel_x = 3.0
            tracking_lin_vel_y = 3.0

            tracking_ang_vel = 1.
            # base_height = -20.0
            # energy_substeps = -2e-5
            stand_still = -1.5
            # dof_error = -0.01
            # added for tron1

            lin_vel_z = -0.5
            ang_vel_xy = -0.05
            torques = -0.00008
            dof_acc = -2.5e-7
            action_rate = -0.01
            dof_pos_limits = -2.0
            collision = -10
            action_smooth = -0.01
            orientation = -20
            feet_distance = -200
            feet_regulation = -0.15
            tracking_contacts_shaped_force = -2.0 
            tracking_contacts_shaped_vel = -2.0
            feet_contact_forces = -0.0035
            ankle_torque_limits = -0.1
            power = -2e-3
            
            # penalty for hardware safety
            # exceed_dof_pos_limits = -0.4
            # exceed_torque_limits_l1norm = -0.4
            # dof_vel_limits = -0.4
            # dof_error_named = -1.

        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        clip_reward = 100
        clip_single_reward = 5
        tracking_sigma = 0.2  # tracking reward = exp(-error^2/sigma)
        ang_tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        height_tracking_sigma = 0.01
        soft_dof_pos_limit = (
            0.95  # percentage of urdf limits, values above this limit are penalized
        )
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 0.8
        base_height_target_min = 0.56 + 0.1
        base_height_target_max = 0.75 + 0.1
        feet_height_target = 0.10
        min_feet_distance = 0.19
        max_feet_distance = 0.23
        max_contact_force = 100.0  # forces above this value are penalized
        kappa_gait_probs = 0.05
        gait_force_sigma = 25.0 # tracking reward = exp(-error^2/sigma)
        gait_vel_sigma = 0.25   # tracking reward = exp(-error^2/sigma)
        gait_height_sigma = 0.005   # tracking reward = exp(-error^2/sigma)

        about_landing_threshold = 0.07

        # dof_error_names = ["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"]
        # only_positive_rewards = False
        # soft_dof_vel_limit = 0.9
        # soft_dof_pos_limit = 0.9
        # soft_torque_limit = 0.9

    # class normalization( LeggedRobotCfg.normalization ):
    #     class obs_scales( LeggedRobotCfg.normalization.obs_scales ):
    #         lin_vel = 2.0
    #     height_measurements_offset = -0.2
    #     clip_actions_method = None # let the policy learns not to exceed the limits

    class noise( LeggedRobotCfg.noise ):
        add_noise = False

    class viewer( LeggedRobotCfg.viewer ):
        pos = [-1., -1., 0.4]
        lookat = [0., 0., 0.3]

    class sim( LeggedRobotCfg.sim ):
        body_measure_points = { # transform are related to body frame
            "base": dict(
                x= [i for i in np.arange(-0.10, 0.10, 0.03)],
                y= [i for i in np.arange(-0.1, 0.05, 0.03)],
                z= [i for i in np.arange(-0.15, 0.03, 0.03)],
                transform= [0., 0., 0.005, 0., 0., 0.],
            ),

            "abad": dict(
                x= [i for i in np.arange(-0.16, -0.035, 0.03)],
                y= [i for i in np.arange(-0.1, 0.05, 0.03)],
                z= [i for i in np.arange(-0.03, 0.18, 0.03)],
                transform= [-0.1, 0.02, -0.1,   0., 1.57079632679, 0.],
            ),
            "hip": dict(
                x= [i for i in np.arange(-0.15, 0.25, 0.03)],
                y= [i for i in np.arange(-0.043, 0.04, 0.03)],
                z= [i for i in np.arange(-0.1, -0.05, 0.03)],
                transform= [0., 0., -0.1,   0., 2.1, 0.],
            ),
            "knee": dict(
                x= [i for i in np.arange(-0.22, 0.15, 0.03)],
                y= [i for i in np.arange(-0.015, 0.03, 0.03)],
                z= [i for i in np.arange(-0.02, 0.03, 0.03)],
                transform= [0.1, 0.,  -0.15,   0., 0.95, 0.],
            ),

            "ankle": dict(
                x= [i for i in np.arange(-0.088, 0.1, 0.03)],
                y= [i for i in np.arange(-0.036, 0.036, 0.03)],
                z= [i for i in np.arange(-0.05, 0.015, 0.03)],
                transform= [0., 0., 0., 0., 0., 0.],
            ),
        }

logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class Tron1RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        clip_min_std = 0.2
        learning_rate = 5.e-4
        optimizer_class_name = "AdamW"

    class policy( LeggedRobotCfgPPO.policy ):
        actor_hidden_dims = [128, 64, 32] # [256, 128, 64, 32]
        critic_hidden_dims = [128, 64, 32] # [256, 128, 64, 32]
        
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

        # configs for estimator module
        estimator_obs_components = [
            "ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos",
            "dof_vel",
            "last_actions",
        ]
        estimator_target_components = ["lin_vel"]
        replace_state_prob = 1.0
        class estimator_kwargs:
            hidden_sizes = [128, 64]
            nonlinearity = "CELU"
        # configs for (critic) encoder
        encoder_component_names = ["height_measurements"]
        encoder_class_name = "MlpModel"
        class encoder_kwargs:
            hidden_sizes = [128, 64]
            nonlinearity = "CELU"
        encoder_output_size = 32
        critic_encoder_component_names = ["height_measurements"]
        init_noise_std = 0.5
        # configs for policy: using recurrent policy with GRU
        rnn_type = 'gru'
        mu_activation = None

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "EncoderStateAcRecurrent"
        algorithm_class_name = "EstimatorPPO"
        experiment_name = "rough_Tron1"
        
        resume = False
        load_run = None

        # run_name = "".join(["Tron1",
        #     ("_pEnergy" + np.format_float_scientific(Tron1RoughCfg.rewards.scales.energy_substeps, precision= 1, trim= "-") if Tron1RoughCfg.rewards.scales.energy_substeps != 0 else ""),
        #     ("_pDofErr" + np.format_float_scientific(Tron1RoughCfg.rewards.scales.dof_error, precision= 1, trim= "-") if Tron1RoughCfg.rewards.scales.dof_error != 0 else ""),
        #     # ("_pDofErrN" + np.format_float_scientific(Tron1RoughCfg.rewards.scales.dof_error_named, precision= 1, trim= "-") if Tron1RoughCfg.rewards.scales.dof_error_named != 0 else ""),
        #     ("_pStand" + np.format_float_scientific(Tron1RoughCfg.rewards.scales.stand_still, precision= 1, trim= "-") if Tron1RoughCfg.rewards.scales.stand_still != 0 else ""),
        #     ("_noResume" if not resume else "_from" + "_".join(load_run.split("/")[-1].split("_")[:2])),
        #     ("_lr" + np.format_float_scientific(algorithm.learning_rate, precision= 1, trim= "-") if algorithm.learning_rate != 0 else ""),
        # ])
        

        max_iterations = 5000
        save_interval = 100
        log_interval = 1
