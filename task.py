import numpy as np
from physics_sim import PhysicsSim

class Task():
    """
    Task (environment) that defines the goal and provides feedback to the agent.

    The task to consider here is Take-off.
    The quadcoputer starts a little above of (x, y, z) = (0, 0, 0).
    The target is set to (x, y, z) = (0, 0, 10.0).

    """
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None):
        """

        Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent

        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6 # size of the state space (with action repeats)
        self.action_size = 4 # size of the action space
        self.action_low = 0.0 # lower bound for the action space
        self.action_high = 900.0 # upper bound for the action space

        # the target position
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])
        # the target angle
        self.target_angle = np.array([0., 0., 0.])

    def get_reward_ddgp(self):
        """
        Uses current pose of sim to return reward and done
        (so that once the quadcopter reaches the height of the target,
        the episode ends. )

        """

        ## the following one is the best one March 26
        reward = - 0.20*(abs(self.sim.pose[:2] - self.target_pos[:2])).sum()
        reward += 2.4 - 1.2*abs(self.sim.pose[2] - self.target_pos[2])
        # reward += 2.4 - 1.2*abs(self.sim.pose[2] - self.target_pos[2])

        # since the angles in self.sim.pose[3:] take value in [0, 2*pi]
        # I will rewrite them such that they take value in range [-pi, pi]
        # and then introduce the reward
        angles_around_0 = np.array([0.0, 0.0, 0.0])
        for i in range(3):
            if self.sim.pose[i+3] <= np.pi:
                angles_around_0[i] = self.sim.pose[i+3]
            else:
                angles_around_0[i] = self.sim.pose[i+3] - 2.0*np.pi
        reward += - 0.20*(abs(angles_around_0[0:3]-self.target_angle)).sum()

        done = False
        if(self.sim.pose[2] >= self.target_pos[2]):
            reward += 200.0
            done = True

        return reward, done

    def step_ddpg(self, rotor_speeds):
        """
        Uses action to obtain next state, reward, done.
        For DDPG algorithm.

        """
        reward = 0.0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            delta_reward, done_height = self.get_reward_ddgp()
            reward += delta_reward
            # once the quadcoper reaches the height of the target, end the episode
            if done_height:
                done = done_height
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)

        return next_state, reward, done

    def reset(self):
        """
        Reset the sim to start a new episode.

        """
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state

    def get_reward(self):
        """
        THIS ONE IS USED FOR THE DEMOMSTRATION PART OF THE NOTEBOOK.
        Uses current pose of sim to return reward and done
        (so that once the quadcopter reaches the height of the target,
        the episode ends. )

        """
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward

    def step(self, rotor_speeds):
        """
        THIS ONE IS USED FOR THE DEMONSTRATION PART OF THE NOTEBOOK.
        Uses action to obtain next state, reward, done.

        """
        reward = 0.0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)

        return next_state, reward, done
