""" Basic model configs for Unitree Go2 """
import numpy as np
import os.path as osp

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1RoughCfg( LeggedRobotCfg ):
    class env:
        num_envs = 4096
        num_observations = None # No use, use obs_components
        num_privileged_obs = None # No use, use privileged_obs_components

        use_lin_vel = False # to be decided
        num_actions = 12
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

        obs_components = [
            "lin_vel",  # 3
            "ang_vel",  # 3
            "projected_gravity",    # 3
            "commands", # 3
            "dof_pos",  # 10
            "dof_vel",  # 10
            "last_actions", # 10
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
        measured_points_x = [i for i in np.arange(-0.5, 1.51, 0.1)]
        measured_points_y = [i for i in np.arange(-0.5, 0.51, 0.1)]
        horizontal_scale = 0.025 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 5 # [m]
        curriculum = True
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
            frequency= 5,
        )

        curriculum_perlin= True

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
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
           'left_knee_joint' : 0.3,       
           'left_ankle_pitch_joint' : -0.2,     
           'left_ankle_roll_joint' : 0,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
           'right_knee_joint' : 0.3,                                             
           'right_ankle_pitch_joint': -0.2,                              
           'right_ankle_roll_joint' : 0,       
           'torso_joint' : 0.
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        action_scale = 0.25
        computer_clip_torque = False
        motor_clip_torque = True        

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_12dof.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        foot_radius = 0.0
        penalize_contacts_on = ["hip", "knee"]

        terminate_after_contacts_on = ["pelvis"]
        # terminate_after_contacts_on = ["base"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False

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
        randomize_com = False
        class com_range:
            x = [-0.2, 0.2]
            y = [-0.1, 0.1]
            z = [-0.05, 0.05]

        randomize_friction = True
        friction_range = [0.1, 1.25]

        randomize_base_mass = False
        added_mass_range = [-1., 3.]

        randomize_motor = False
        leg_motor_strength_range = [0.8, 1.2]

        push_robots = False 
        push_interval_s = 5
        max_push_vel_xy = 1.5 # [m/s]
        
        # ####### 增加对初始位姿的鲁棒性  ###########
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
        # #######################################        

    class rewards( LeggedRobotCfg.rewards ):
        dof_error_names = ["left_hip_yaw_joint","right_hip_yaw_joint","left_hip_roll_joint","right_hip_roll_joint"]
        min_feet_distance = 0.3
        only_positive_rewards = False
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        class scales:
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            ang_vel_xy = -0.3
            orientation = -1.0
            lin_vel_z = -2.0

            dof_acc = -2.5e-7
            dof_vel = -1e-3
            
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0
            contact = 0.18

            # # added for parlin terrain
            feet_air_time = 10.0
            single_contact = 1
            # dof_error_named = -0.1

    # class normalization( LeggedRobotCfg.normalization ):
    #     class obs_scales( LeggedRobotCfg.normalization.obs_scales ):
    #         lin_vel = 2.0
    #     height_measurements_offset = -0.2
    #     clip_actions_method = None # let the policy learns not to exceed the limits

    class noise( LeggedRobotCfg.noise ):
        add_noise = False

    class viewer( LeggedRobotCfg.viewer ):
        pos = [-4., -1., 2.4]
        lookat = [0., 0., 0.3]

    class sim( LeggedRobotCfg.sim ):
        body_measure_points = { # transform are related to body frame
            "hip_yaw": dict(
                x= [i for i in np.arange(-0.1, 0.02, 0.02)],
                y= [i for i in np.arange(-0.06, 0.06, 0.02)],
                z= [i for i in np.arange(-0.15, 0.2, 0.02)],
                transform= [0.05, 0., 0.,  0., 0., 0.],
            ),
            "knee": dict(
                x= [i for i in np.arange(-0.29, 0.06, 0.02)],
                y= [i for i in np.arange(-0.028, 0.06, 0.02)],
                z= [i for i in np.arange(-0.06, 0.07, 0.02)],
                transform= [0., 0., -0.25, 0., 1.6, 0.],
            ),
            "ankle": dict(
                x= [i for i in np.arange(-0.05, 0.18, 0.02)],
                y= [i for i in np.arange(-0.036, 0.036, 0.03)],
                z= [i for i in np.arange(-0.07, 0., 0.02)],
                transform= [-0.01, 0., 0., 0., 0., 0.],
            ),
        }

logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class G1RoughCfgPPO( LeggedRobotCfgPPO ):
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
        experiment_name = "rough_G1"
        
        resume = False
        load_run = None

        # run_name = "".join(["H1",
        #     ("_pEnergy" + np.format_float_scientific(G1RoughCfg.rewards.scales.energy_substeps, precision= 1, trim= "-") if G1RoughCfg.rewards.scales.energy_substeps != 0 else ""),
        #     ("_pDofErr" + np.format_float_scientific(G1RoughCfg.rewards.scales.dof_error, precision= 1, trim= "-") if G1RoughCfg.rewards.scales.dof_error != 0 else ""),
        #     # ("_pDofErrN" + np.format_float_scientific(G1RoughCfg.rewards.scales.dof_error_named, precision= 1, trim= "-") if G1RoughCfg.rewards.scales.dof_error_named != 0 else ""),
        #     ("_pStand" + np.format_float_scientific(G1RoughCfg.rewards.scales.stand_still, precision= 1, trim= "-") if G1RoughCfg.rewards.scales.stand_still != 0 else ""),
        #     ("_noResume" if not resume else "_from" + "_".join(load_run.split("/")[-1].split("_")[:2])),
        #     ("_lr" + np.format_float_scientific(algorithm.learning_rate, precision= 1, trim= "-") if algorithm.learning_rate != 0 else ""),
        # ])
        

        max_iterations = 8000
        save_interval = 500
        log_interval = 1
