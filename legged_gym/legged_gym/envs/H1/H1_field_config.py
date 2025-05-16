""" Config to train the whole parkour oracle policy """
import numpy as np
from os import path as osp
from collections import OrderedDict

from legged_gym.envs.H1.H1_config import H1RoughCfg, H1RoughCfgPPO

class H1FieldCfg( H1RoughCfg ):
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

    class init_state( H1RoughCfg.init_state ):
        pos = [0.0, 0.0, 1.0 + 0.05] # x,y,z [m]
        zero_actions = False

    class sensor( H1RoughCfg.sensor):
        class proprioception( H1RoughCfg.sensor.proprioception ):
            # latency_range = [0.0, 0.0]
            latency_range = [0.005, 0.045] # [s]

    class terrain( H1RoughCfg.terrain ):
        num_rows = 10
        num_cols = 4
        selected = "BarrierTrack"
        slope_treshold = 20.
        max_init_terrain_level = 1
        pad_unavailable_info = True
        BarrierTrack_kwargs = dict(
            options= [
                "jump",
                "leap",
                "hurdle",
                "down",
                "tilted_ramp",
                "stairsup",
                "stairsdown",
                "discrete_rect",
                "slope",
                "wave",
            ], # each race track will permute all the options
            hurdle= dict(
                height= 2,
                depth= [0.2, 0.2],
                width_range= [0.3, 4.5],  # 障碍物宽度范围（y轴）
                position_range= 0,  # 障碍物位置范围（相对于道路中心的偏移量）
            ),

            jump= dict(
                height= [0.05, 0.5],
                depth= [0.1, 0.3],
                # fake_offset= 0.1,
            ),
            leap= dict(
                length= [0.05, 0.8],
                depth= [0.5, 0.8],
                height= 0.2, # expected leap height over the gap
                # fake_offset= 0.1,
            ),
            down= dict(
                height= [0.1, 0.6],
                depth= [0.3, 0.5],
            ),
            tilted_ramp= dict(
                tilt_angle= [0.2, 0.5],
                switch_spacing= 0.,
                spacing_curriculum= False,
                overlap_size= 0.2,
                depth= [-0.1, 0.1],
                length= [0.6, 1.2],
            ),
            slope= dict(
                slope_angle= [0.2, 0.42],
                length= [1.2, 2.2],
                use_mean_height_offset= True,
                face_angle= [-3.14, 0, 1.57, -1.57],
                no_perlin_rate= 0.2,
                length_curriculum= True,
            ),
            slopeup= dict(
                slope_angle= [0.2, 0.42],
                length= [1.2, 2.2],
                use_mean_height_offset= True,
                face_angle= [-0.2, 0.2],
                no_perlin_rate= 0.2,
                length_curriculum= True,
            ),
            slopedown= dict(
                slope_angle= [0.2, 0.42],
                length= [1.2, 2.2],
                use_mean_height_offset= True,
                face_angle= [-0.2, 0.2],
                no_perlin_rate= 0.2,
                length_curriculum= True,
            ),
            stairsup= dict(
                height= [0.1, 0.3],
                length= [0.3, 0.5],
                residual_distance= 0.05,
                num_steps= [3, 19],
                num_steps_curriculum= True,
            ),
            stairsdown= dict(
                height= [0.1, 0.3],
                length= [0.3, 0.5],
                num_steps= [3, 19],
                num_steps_curriculum= True,
            ),
            discrete_rect= dict(
                max_height= [0.05, 0.2],
                max_size= 0.6,
                min_size= 0.2,
                num_rects= 10,
            ),
            wave= dict(
                amplitude= [0.1, 0.15], # in meter
                frequency= [0.6, 1.0], # in 1/meter
            ),
            track_width= 3.2,
            track_block_length= 2.4,
            wall_thickness= (0.01, 0.6),
            wall_height= [-0.5, 2.0],
            add_perlin_noise= True,
            border_perlin_noise= True,
            border_height= 0.,
            virtual_terrain= False,
            draw_virtual_terrain= True,
            engaging_next_threshold= 0.8,
            engaging_finish_threshold= 0.,
            curriculum_perlin= True,
            no_perlin_threshold= 0.1,
            randomize_obstacle_order= True,
            n_obstacles_per_track= 1,
            
        )

    class commands( H1RoughCfg.commands ):
        # a mixture of command sampling and goal_based command update allows only high speed range
        # in x-axis but no limits on y-axis and yaw-axis
        lin_cmd_cutoff = 0.2
        class ranges( H1RoughCfg.commands.ranges ):
            # lin_vel_x = [0.6, 1.8]
            lin_vel_x = [-0.6, 2.0]


        #############################################################################################
        is_goal_based = True
        class goal_based:
            # the ratios are related to the goal position in robot frame
            x_ratio = None # sample from lin_vel_x range
            y_ratio = 1.2
            yaw_ratio = 1.
            follow_cmd_cutoff = True
            x_stop_by_yaw_threshold = 1. # stop when yaw is over this threshold [rad]        
        #############################################################################################

        is_avoid_obstacles = True

    class asset( H1RoughCfg.asset ):
        terminate_after_contacts_on = ["base"]
        penalize_contacts_on = ["pelvis","knee", "hip","ankle"]

    class termination( H1RoughCfg.termination ):
        roll_kwargs = dict(
            threshold= 1.4, # [rad]
        )
        pitch_kwargs = dict(
            threshold= 1.6, # [rad]
        )
        timeout_at_border = True
        timeout_at_finished = False

    class rewards( H1RoughCfg.rewards ):
        collision_tracking_weight = 5  # 碰撞时的基础权重
        class scales:
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            # lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            # base_height = -10.0
            dof_acc = -2.5e-7
            feet_air_time = 0.0
            collision = -1.0
            action_rate = -0.01
            torques = 0.0
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0
            contact_no_vel = -0.2
            # feet_swing_height = -20.0
            contact = 0.18

            # # ADDED Reward
            # collision = -0.05
            # lazy_stop = -3.

            # # penetration penalty
            # penetrate_depth = -0.05

    class noise( H1RoughCfg.noise ):
        add_noise = False

    class curriculum:
        penetrate_depth_threshold_harder = 100
        penetrate_depth_threshold_easier = 200
        no_moveup_when_fall = True
    
logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class H1FieldCfgPPO( H1RoughCfgPPO ):
    class algorithm( H1RoughCfgPPO.algorithm ):
        entropy_coef = 0.0

    class runner( H1RoughCfgPPO.runner ):
        experiment_name = "field_H1"

        resume = True
        load_run = osp.join(logs_root, "rough_H1",
            "May04_21-19-33_unitree",
        )

        # run_name = "".join(["H1_",
        #     ("{:d}skills".format(len(H1FieldCfg.terrain.BarrierTrack_kwargs["options"]))),
        #     ("_pEnergy" + np.format_float_scientific(-H1FieldCfg.rewards.scales.energy_substeps, precision=2)),
        #     # ("_pDofErr" + np.format_float_scientific(-H1FieldCfg.rewards.scales.dof_error, precision=2) if getattr(H1FieldCfg.rewards.scales, "dof_error", 0.) != 0. else ""),
        #     # ("_pHipDofErr" + np.format_float_scientific(-H1FieldCfg.rewards.scales.dof_error_named, precision=2) if getattr(H1FieldCfg.rewards.scales, "dof_error_named", 0.) != 0. else ""),
        #     # ("_pStand" + np.format_float_scientific(H1FieldCfg.rewards.scales.stand_still, precision=2)),
        #     # ("_pTerm" + np.format_float_scientific(H1FieldCfg.rewards.scales.termination, precision=2) if hasattr(H1FieldCfg.rewards.scales, "termination") else ""),
        #     ("_pTorques" + np.format_float_scientific(H1FieldCfg.rewards.scales.torques, precision=2) if hasattr(H1FieldCfg.rewards.scales, "torques") else ""),
        #     # ("_pColl" + np.format_float_scientific(H1FieldCfg.rewards.scales.collision, precision=2) if hasattr(H1FieldCfg.rewards.scales, "collision") else ""),
        #     ("_pLazyStop" + np.format_float_scientific(H1FieldCfg.rewards.scales.lazy_stop, precision=2) if hasattr(H1FieldCfg.rewards.scales, "lazy_stop") else ""),
        #     # ("_trackSigma" + np.format_float_scientific(H1FieldCfg.rewards.tracking_sigma, precision=2) if H1FieldCfg.rewards.tracking_sigma != 0.25 else ""),
        #     # ("_pPenV" + np.format_float_scientific(-H1FieldCfg.rewards.scales.penetrate_volume, precision=2)),
        #     ("_pPenD" + np.format_float_scientific(-H1FieldCfg.rewards.scales.penetrate_depth, precision=2)),
        #     # ("_pTorqueL1" + np.format_float_scientific(-H1FieldCfg.rewards.scales.exceed_torque_limits_l1norm, precision=2)),
        #     ("_penEasier{:d}".format(H1FieldCfg.curriculum.penetrate_depth_threshold_easier)),
        #     ("_penHarder{:d}".format(H1FieldCfg.curriculum.penetrate_depth_threshold_harder)),
        #     # ("_leapMin" + np.format_float_scientific(H1FieldCfg.terrain.BarrierTrack_kwargs["leap"]["length"][0], precision=2)),
        #     ("_leapHeight" + np.format_float_scientific(H1FieldCfg.terrain.BarrierTrack_kwargs["leap"]["height"], precision=2)),
        #     ("_motorTorqueClip" if H1FieldCfg.control.motor_clip_torque else ""),
        #     # ("_noMoveupWhenFall" if H1FieldCfg.curriculum.no_moveup_when_fall else ""),
        #     ("_noResume" if not resume else "_from" + "_".join(load_run.split("/")[-1].split("_")[:2])),
        # ])

        max_iterations = 800000
        save_interval = 500
        log_interval = 1
        