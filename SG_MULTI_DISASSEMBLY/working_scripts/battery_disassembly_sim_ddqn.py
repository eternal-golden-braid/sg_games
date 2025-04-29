import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from mpl_toolkits.mplot3d import Axes3D

class BatteryDisassemblyEnv:
    """
    Environment class for the battery disassembly task.
    This environment models a workstation with a battery module and two robots:
    - Franka robot (Leader): Equipped with a two-finger gripper for unbolting operations
    - UR10 robot (Follower): Equipped with vacuum suction for sorting and pick-and-place
    """
    def __init__(self, parameters):
        """
        Initialize the battery disassembly environment.
        
        Parameters:
        - parameters: Dictionary containing environment parameters
        """
        self.rng = np.random.default_rng(parameters['seed'])
        self.task_id = parameters['task_id']
        
        # Load the task board and properties
        self.task_board, self.task_prop = self.task_reader(self.task_id)
        self.curr_board = np.copy(self.task_board)
        
        # Define robot properties
        self.franka_pos = np.array([0.5, -0.3, 0.5])  # Base position of Franka robot
        self.ur10_pos = np.array([-0.5, -0.3, 0.5])   # Base position of UR10 robot
        
        # Define workspace properties
        self.battery_pos = np.array([0.0, 0.0, 0.1])  # Position of the battery module
        self.bin_positions = {
            'screws': np.array([0.3, 0.4, 0.1]),
            'cells': np.array([-0.3, 0.4, 0.1]),
            'casings': np.array([0.0, 0.5, 0.1])
        }
        
        # Task completion tracking
        self.completed_tasks = []
        
        # Robot state
        self.franka_state = {'position': self.franka_pos, 'gripper_open': True, 'holding': None}
        self.ur10_state = {'position': self.ur10_pos, 'suction_active': False, 'holding': None}
        
        # Task timing and resource tracking
        self.time_step = 0
        self.max_time_steps = parameters.get('max_time_steps', 100)
        
        # Robot kinematic constraints
        self.franka_workspace_radius = 0.8
        self.ur10_workspace_radius = 1.0
        
        # Task failure probabilities (uncertainty modeling)
        self.franka_failure_prob = parameters.get('franka_failure_prob', 0.1)
        self.ur10_failure_prob = parameters.get('ur10_failure_prob', 0.1)
        
    def task_reader(self, task_id):
        """
        Read the task information from the configuration files.
        """
        # For this simulation, we'll create a custom battery disassembly task
        
        # Task board represents the spatial arrangement of components to be disassembled
        # 0: Empty space
        # 1-4: Top screws (requires unbolting by Franka)
        # 5-8: Side screws (requires unbolting by Franka)
        # 9-12: Battery cells (requires pick-and-place by UR10)
        # 13-16: Casing components (requires collaborative effort)
        task_board = np.array([
            [1, 2, 3, 4],
            [9, 10, 11, 12],
            [17, 17, 18, 18],
            [5, 6, 7, 8],
            [13, 14, 15, 16]
        ])
        
        # Task properties define the characteristics of each task
        # type 1: Leader-specific tasks (unbolting by Franka)
        # type 2: Follower-specific tasks (pick-and-place by UR10)
        # type 3: Sequential tasks (can be done by either, but one is more efficient)
        # type 4: Collaborative tasks (requires both robots)
        
        # Shape indicates the physical size/complexity (affects timing)
        # l_succ/f_succ are success probabilities for leader/follower
        task_prop = {
            'type': np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4]),
            'shape': np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2]),
            'l_succ': np.array([0.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]),
            'f_succ': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.9, 0.9, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7])
        }
        
        return task_board, task_prop

    def get_task_info(self):
        """
        Get task information for initializing the learning algorithms.
        """
        info = {}
        info['task_id'] = self.task_id
        info['dims'] = self.task_board.shape[1]
        info['dimAl'] = self.task_board.shape[1] + 1  # +1 for "do nothing" action
        info['dimAf'] = self.task_board.shape[1] + 1  # +1 for "do nothing" action
        info['dimal'] = 1
        info['dimaf'] = 1
        info['task_prop'] = self.task_prop
        return info

    def get_current_state(self):
        """
        Get the current state of the environment.
        
        Returns:
        - First row of the current board (simplified state representation)
        - Complete current board
        """
        return np.copy(self.curr_board[0, :]), np.copy(self.curr_board)
    
    def set_env(self, board):
        """
        Set the environment to a specific board configuration.
        """
        self.curr_board = np.copy(board)
    
    def reset_env(self):
        """
        Reset the environment to the initial state.
        """
        self.curr_board = np.copy(self.task_board)
        self.completed_tasks = []
        self.time_step = 0
        self.franka_state = {'position': self.franka_pos, 'gripper_open': True, 'holding': None}
        self.ur10_state = {'position': self.ur10_pos, 'suction_active': False, 'holding': None}
    
    def step(self, al, af):
        """
        Execute one step in the environment based on leader and follower actions.
        
        Parameters:
        - al: Leader action (Franka robot)
        - af: Follower action (UR10 robot)
        """
        # Simulate if task is completed by the leader (Franka)
        if al == -1:
            tl, tl_done = 0, False  # Leader does nothing
        else:
            tl = self.curr_board[0, al]
            if tl == 0:
                tl_done = False  # Task already completed or invalid
            else:
                # Check if task is within Franka's capabilities and workspace
                if self.is_task_feasible(tl, 'leader'):
                    tl_done = True if self.rng.uniform() < self.task_prop['l_succ'][tl] else False
                else:
                    tl_done = False
        
        # Simulate if task is completed by the follower (UR10)
        if af == -1:
            tf, tf_done = 0, False  # Follower does nothing
        else:
            tf = self.curr_board[0, af]
            if tf == 0:
                tf_done = False  # Task already completed or invalid
            else:
                # Check if task is within UR10's capabilities and workspace
                if self.is_task_feasible(tf, 'follower'):
                    tf_done = True if self.rng.uniform() < self.task_prop['f_succ'][tf] else False
                else:
                    tf_done = False
        
        # Update the task board based on the simulated results
        self.update_board(tl, tl_done, tf, tf_done)
        
        # Update robot positions based on actions
        if tl_done or al != -1:
            self.update_robot_position('leader', al)
        
        if tf_done or af != -1:
            self.update_robot_position('follower', af)
        
        # Increment time step
        self.time_step += 1
    
    def is_task_feasible(self, task_id, robot):
        """
        Check if a task is feasible for the given robot based on capabilities and workspace constraints.
        
        Parameters:
        - task_id: ID of the task to check
        - robot: 'leader' for Franka, 'follower' for UR10
        
        Returns:
        - Boolean indicating if the task is feasible
        """
        # Check robot capability based on task type
        task_type = self.task_prop['type'][task_id]
        
        if robot == 'leader':
            # Leader can do type 1, 3, and 4 tasks
            if task_type not in [1, 3, 4]:
                return False
                
            # Check if Franka can reach the task
            # In a realistic scenario, this would involve inverse kinematics checks
            task_pos = self.get_task_position(task_id)
            dist = np.linalg.norm(task_pos - self.franka_state['position'])
            return dist <= self.franka_workspace_radius
            
        elif robot == 'follower':
            # Follower can do type 2, 3, and 4 tasks
            if task_type not in [2, 3, 4]:
                return False
                
            # Check if UR10 can reach the task
            task_pos = self.get_task_position(task_id)
            dist = np.linalg.norm(task_pos - self.ur10_state['position'])
            return dist <= self.ur10_workspace_radius
    
    def get_task_position(self, task_id):
        """
        Get the 3D position of a task based on its ID.
        
        In a realistic scenario, this would map task IDs to actual 
        positions on the battery module.
        """
        # Find the task coordinates in the board
        coords = np.argwhere(self.curr_board == task_id)
        if len(coords) == 0:
            return None
        
        row, col = coords[0]
        
        # Map the 2D coordinates to 3D positions relative to the battery position
        x = self.battery_pos[0] + (col - self.curr_board.shape[1]/2) * 0.1
        y = self.battery_pos[1] + (row - self.curr_board.shape[0]/2) * 0.1
        z = self.battery_pos[2] + 0.05  # Slight offset from battery surface
        
        return np.array([x, y, z])
    
    def update_robot_position(self, robot, action):
        """
        Update the position of a robot based on its action.
        
        Parameters:
        - robot: 'leader' for Franka, 'follower' for UR10
        - action: The robot's action
        """
        if action == -1:
            # No movement for "do nothing" action
            return
        
        task_pos = self.get_task_position(self.curr_board[0, action])
        if task_pos is None:
            # No valid task position
            return
        
        if robot == 'leader':
            # Move Franka to the task position
            self.franka_state['position'] = task_pos
            # Update gripper state based on task type
            task_id = self.curr_board[0, action]
            if task_id > 0:
                task_type = self.task_prop['type'][task_id]
                self.franka_state['gripper_open'] = task_type != 1  # Close gripper for unbolting
                self.franka_state['holding'] = task_id if task_type == 1 else None
        else:
            # Move UR10 to the task position
            self.ur10_state['position'] = task_pos
            # Update suction state based on task type
            task_id = self.curr_board[0, action]
            if task_id > 0:
                task_type = self.task_prop['type'][task_id]
                self.ur10_state['suction_active'] = task_type == 2  # Activate suction for pick-and-place
                self.ur10_state['holding'] = task_id if task_type == 2 else None
    
    def update_board(self, tl, tl_done, tf, tf_done):
        """
        Update the task board based on completed tasks.
        
        Parameters:
        - tl: Leader's task ID
        - tl_done: Whether leader's task was completed
        - tf: Follower's task ID
        - tf_done: Whether follower's task was completed
        """
        # Handle the case where both robots choose the same task
        if tl == tf:
            if tl != 0 and self.task_prop['type'][tl] == 4 and tl_done and tf_done:
                # Both robots successfully complete a collaborative task
                idx = np.where(self.curr_board[0] == tl)[0]
                self.curr_board[0, idx] = 0
                self.completed_tasks.append(tl)
            # Otherwise, no update (either both did nothing or both failed)
        else:
            # Handle leader's task
            if tl != 0 and tl_done and self.task_prop['type'][tl] != 4:
                idx = np.where(self.curr_board[0] == tl)[0]
                self.curr_board[0, idx] = 0
                self.completed_tasks.append(tl)
            
            # Handle follower's task
            if tf != 0 and tf_done and self.task_prop['type'][tf] != 4:
                idx = np.where(self.curr_board[0] == tf)[0]
                self.curr_board[0, idx] = 0
                self.completed_tasks.append(tf)
        
        # Update subsequent rows (task dependencies)
        for i in range(self.task_board.shape[0] - 1):
            curr_row, next_row = self.curr_board[i, :], self.curr_board[i+1, :]
            
            # Find tasks that may drop from the next row
            task_list = []
            idx = np.where(curr_row == 0)[0]
            for j in idx:
                task_id = next_row[j]
                if task_id !=0 and task_id not in task_list:    # task 0 does not count
                    task_list.append(task_id)

            # Check for tasks that can now be accessed
            mod_flag = False
            for ti in task_list:
                idx = np.where(next_row == ti)[0]
                if np.all(curr_row[idx] == 0):
                    curr_row[idx] = ti
                    next_row[idx] = 0
                    mod_flag = True
            
            # If no modifications were made, no need to update future rows
            if not mod_flag:
                break
    
    def reward(self, s, al, af):
        """
        Calculate rewards for leader and follower based on their actions.
        
        Parameters:
        - s: Current state (first row of the board)
        - al: Leader's action
        - af: Follower's action
        
        Returns:
        - rl, rf: Rewards for leader and follower
        """
        # Determine task IDs corresponding to the actions
        tl = 0 if al == -1 else s[al]
        tf = 0 if af == -1 else s[af]
        
        # Initialize rewards
        rl, rf = 0, 0
        
        # Both choose zero task or idle
        if tl == 0 and tf == 0:
            if al == -1 and af == -1:
                rl, rf = -0.5, -0.5  # Both idle (slight penalty)
            elif al == -1 and af != -1:
                rl, rf = 0, -1       # Leader idle, follower attempts empty task
            elif al != -1 and af == -1:
                rl, rf = -1, 0       # Leader attempts empty task, follower idle
            else:
                rl, rf = -2, -2      # Both attempt empty tasks (larger penalty)
        
        # Both choose the same non-zero task
        elif tl == tf and tl != 0 and tf != 0:
            if self.task_prop['type'][tl] == 4:
                rl, rf = 2, 2        # Collaborative task (higher reward)
            else:
                rl, rf = -1, -1      # Collision on non-collaborative task (penalty)
        
        # Choose different tasks
        else:
            # Process leader's reward
            if tl == 0:
                rl = 0 if al == -1 else -1  # Either idle or empty task
            elif self.task_prop['type'][tl] in [1, 3]:
                rl = 1               # Leader-appropriate task
            else:
                rl = -1              # Leader-inappropriate task
            
            # Process follower's reward
            if tf == 0:
                rf = 0 if af == -1 else -1  # Either idle or empty task
            elif self.task_prop['type'][tf] in [2, 3]:
                rf = 1               # Follower-appropriate task
            else:
                rf = -1              # Follower-inappropriate task
        
        return float(rl), float(rf)
    
    def is_done(self):
        """
        Check if the task is complete (all tasks processed).
        """
        return np.all(self.curr_board == 0)
    
    def render(self, ax=None, mode='human'):
        """
        Render the current state of the environment.
        
        Parameters:
        - ax: Matplotlib axis for rendering
        - mode: Rendering mode ('human' for visualization, 'rgb_array' for image)
        
        Returns:
        - Matplotlib axis with the rendered environment
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        # Clear previous plot
        ax.clear()
        
        # Plot workstation surface
        x = np.linspace(-0.6, 0.6, 10)
        y = np.linspace(-0.4, 0.6, 10)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        ax.plot_surface(X, Y, Z, alpha=0.3, color='gray')
        
        # Plot battery module
        battery_size = 0.2
        ax.bar3d(
            self.battery_pos[0] - battery_size/2,
            self.battery_pos[1] - battery_size/2,
            0,
            battery_size, battery_size, self.battery_pos[2],
            color='silver', alpha=0.7
        )
        
        # Plot bins
        bin_size = 0.1
        for bin_name, bin_pos in self.bin_positions.items():
            ax.bar3d(
                bin_pos[0] - bin_size/2,
                bin_pos[1] - bin_size/2,
                0,
                bin_size, bin_size, bin_pos[2],
                color='lightblue', alpha=0.5
            )
            ax.text(bin_pos[0], bin_pos[1], bin_pos[2] + 0.02, bin_name, 
                   horizontalalignment='center', verticalalignment='bottom')
        
        # Visualize tasks on the battery
        for i in range(self.task_board.shape[0]):
            for j in range(self.task_board.shape[1]):
                task_id = self.curr_board[i, j]
                if task_id > 0:
                    task_pos = self.get_task_position(task_id)
                    task_type = self.task_prop['type'][task_id]
                    
                    # Color based on task type
                    if task_type == 1:
                        color = 'blue'        # Leader tasks
                    elif task_type == 2:
                        color = 'green'       # Follower tasks
                    elif task_type == 3:
                        color = 'orange'      # Sequential tasks
                    else:
                        color = 'red'         # Collaborative tasks
                    
                    # Plot task markers
                    ax.scatter(
                        task_pos[0], task_pos[1], task_pos[2],
                        color=color, s=100, alpha=0.8
                    )
                    
                    ax.text(
                        task_pos[0], task_pos[1], task_pos[2] + 0.02,
                        f'T{task_id}', 
                        horizontalalignment='center',
                        verticalalignment='bottom'
                    )
        
        # Plot robots
        # Franka robot (leader)
        self._plot_robot(
            ax, self.franka_state['position'], 
            'Franka', 'blue', 
            self.franka_state['gripper_open'], 
            self.franka_state['holding']
        )
        
        # UR10 robot (follower)
        self._plot_robot(
            ax, self.ur10_state['position'], 
            'UR10', 'green', 
            not self.ur10_state['suction_active'], 
            self.ur10_state['holding']
        )
        
        # Add legend explaining task types
        ax.text(0.6, 0.5, 0.1, 'Task Types:', fontweight='bold')
        ax.scatter(0.6, 0.45, 0.1, color='blue', s=50)
        ax.text(0.65, 0.45, 0.1, 'Type 1: Leader (Unbolting)')
        
        ax.scatter(0.6, 0.4, 0.1, color='green', s=50)
        ax.text(0.65, 0.4, 0.1, 'Type 2: Follower (Pick & Place)')
        
        ax.scatter(0.6, 0.35, 0.1, color='orange', s=50)
        ax.text(0.65, 0.35, 0.1, 'Type 3: Sequential Tasks')
        
        ax.scatter(0.6, 0.3, 0.1, color='red', s=50)
        ax.text(0.65, 0.3, 0.1, 'Type 4: Collaborative Tasks')
        
        # Set plot limits and labels
        ax.set_xlim([-0.8, 0.8])
        ax.set_ylim([-0.5, 0.7])
        ax.set_zlim([0, 0.8])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Battery Disassembly Simulation - Time Step: {self.time_step}')
        
        # Show completed tasks
        if self.completed_tasks:
            completed_str = 'Completed: ' + ', '.join([f'T{t}' for t in self.completed_tasks[-5:]])
            if len(self.completed_tasks) > 5:
                completed_str += f'... (+{len(self.completed_tasks)-5} more)'
            ax.text2D(0.05, 0.95, completed_str, transform=ax.transAxes)
        
        plt.draw()
        return ax
    
    def _plot_robot(self, ax, position, name, color, gripper_open, holding):
        """
        Plot a robot in the environment.
        
        Parameters:
        - ax: Matplotlib axis
        - position: Robot position [x, y, z]
        - name: Robot name
        - color: Robot color
        - gripper_open: Boolean indicating if gripper is open
        - holding: ID of the task being held (if any)
        """
        # Plot robot base
        ax.bar3d(
            position[0] - 0.05,
            position[1] - 0.05,
            0,
            0.1, 0.1, position[2] - 0.1,
            color=color, alpha=0.5
        )
        
        # Plot end-effector
        ax.scatter(
            position[0], position[1], position[2],
            color=color, s=150, alpha=0.8
        )
        
        # Show robot name
        ax.text(
            position[0], position[1], position[2] + 0.05,
            name, 
            horizontalalignment='center',
            verticalalignment='bottom'
        )
        
        # Indicate gripper state
        gripper_state = "Open" if gripper_open else "Closed"
        ax.text(
            position[0], position[1], position[2] - 0.05,
            gripper_state, 
            horizontalalignment='center',
            verticalalignment='top',
            fontsize=8
        )
        
        # Indicate held object
        if holding is not None:
            ax.text(
                position[0], position[1], position[2] - 0.1,
                f"Holding: T{holding}", 
                horizontalalignment='center',
                verticalalignment='top',
                fontsize=8
            )
        
        # Plot a line from the base to the end-effector
        ax.plot(
            [position[0], position[0]],
            [position[1], position[1]],
            [0, position[2]],
            color=color, linewidth=2, alpha=0.6
        )


class StackelbergGameSimulation:
    """
    Main simulation class for the Stackelberg game between two robots.
    """
    def __init__(self, parameters):
        """
        Initialize the simulation.
        
        Parameters:
        - parameters: Dictionary containing simulation parameters
        """
        self.env = BatteryDisassemblyEnv(parameters)
        self.device = parameters.get('device', 'cpu')
        
        # Initialize leader Q-network (Franka robot)
        self.leader = self._init_leader(parameters)
        
        # Initialize follower Q-network (UR10 robot)
        self.follower = self._init_follower(parameters)
        
        # Training parameters
        self.n_episodes = parameters.get('episode_size', 1000)
        self.n_steps_per_episode = parameters.get('step_per_episode', 40)
        self.batch_size = parameters.get('batch_size', 32)
        self.buffer_size = parameters.get('buffer_size', 256)
        
        # Initialize replay buffer
        self.buffer = ReplayBuffer(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            seed=parameters.get('seed', 0)
        )
        
        # Statistics tracking
        self.training_stats = {
            'leader_rewards': [],
            'follower_rewards': [],
            'completion_steps': [],
            'completion_rates': []
        }
        
    def _init_leader(self, parameters):
        """
        Initialize the leader (Franka robot) agent.
        """
        env_info = self.env.get_task_info()
        
        # In a real implementation, this would initialize a Q-network
        # For this simulation, we'll use a simplified model
        leader = {
            'policy': np.zeros((env_info['dims'] + 1, env_info['dims'] + 1)),
            'learning_rate': parameters.get('leader', {}).get('learning_rate', 0.001),
            'gamma': parameters.get('leader', {}).get('reward_decay', 0.9),
            'epsilon': parameters.get('leader', {}).get('epsilon', 0.1)
        }
        
        return leader
    
    def _init_follower(self, parameters):
        """
        Initialize the follower (UR10 robot) agent.
        """
        env_info = self.env.get_task_info()
        
        # In a real implementation, this would initialize a Q-network
        # For this simulation, we'll use a simplified model
        follower = {
            'policy': np.zeros((env_info['dims'] + 1, env_info['dims'] + 1)),
            'learning_rate': parameters.get('follower', {}).get('learning_rate', 0.001),
            'gamma': parameters.get('follower', {}).get('reward_decay', 0.9),
            'epsilon': parameters.get('follower', {}).get('epsilon', 0.1)
        }
        
        return follower
    
    def compute_stackelberg_equilibrium(self, state):
        """
        Compute the Stackelberg equilibrium given the current state.
        
        In a Stackelberg game, the leader moves first, and the follower responds optimally.
        The leader anticipates the follower's response when making its decision.
        
        Parameters:
        - state: Current environment state
        
        Returns:
        - leader_action, follower_action: Equilibrium actions
        """
        env_info = self.env.get_task_info()
        dims = env_info['dims']
        
        # For demonstration purposes, we'll use a heuristic strategy
        # In a real implementation, this would use the learned Q-networks
        
        # Available actions (including "do nothing" as -1)
        action_space = list(range(-1, dims))
        
        # Create reward matrices for all possible action combinations
        leader_rewards = np.zeros((len(action_space), len(action_space)))
        follower_rewards = np.zeros((len(action_space), len(action_space)))
        
        for i, al in enumerate(action_space):
            for j, af in enumerate(action_space):
                rl, rf = self.env.reward(state, al, af)
                leader_rewards[i, j] = rl
                follower_rewards[i, j] = rf
        
        # Compute Stackelberg equilibrium
        # For each leader action, find the follower's best response
        follower_best_response = np.argmax(follower_rewards, axis=1)
        
        # Choose the leader action that maximizes its reward given the follower's best response
        leader_value = np.array([leader_rewards[i, follower_best_response[i]] for i in range(len(action_space))])
        leader_action_idx = np.argmax(leader_value)
        
        # Get the corresponding actions
        leader_action = action_space[leader_action_idx]
        follower_action = action_space[follower_best_response[leader_action_idx]]
        
        return leader_action, follower_action
    
    def run_episode(self, training=True, render=False):
        """
        Run a single episode of the simulation.
        
        Parameters:
        - training: Whether to update the agent policies
        - render: Whether to render the environment
        
        Returns:
        - Dictionary containing episode statistics
        """
        self.env.reset_env()
        state, _ = self.env.get_current_state()
        
        total_leader_reward = 0
        total_follower_reward = 0
        steps = 0
        
        # Create figure for rendering if needed
        if render:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            plt.ion()  # Turn on interactive mode
        
        # Run the episode
        for step in range(self.n_steps_per_episode):
            # Compute Stackelberg equilibrium actions
            leader_action, follower_action = self.compute_stackelberg_equilibrium(state)
            
            # Apply exploration if training
            if training:
                if np.random.rand() < self.leader['epsilon']:
                    leader_action = np.random.choice(list(range(-1, self.env.task_board.shape[1])))
                
                if np.random.rand() < self.follower['epsilon']:
                    follower_action = np.random.choice(list(range(-1, self.env.task_board.shape[1])))
            
            # Get rewards and update environment
            leader_reward, follower_reward = self.env.reward(state, leader_action, follower_action)
            self.env.step(leader_action, follower_action)
            next_state, _ = self.env.get_current_state()
            
            # Store experience in replay buffer if training
            if training:
                experience = np.concatenate((
                    state, 
                    np.array([leader_action, follower_action, leader_reward, follower_reward]), 
                    next_state
                ))
                self.buffer.add(experience)
            
            # Update statistics
            total_leader_reward += leader_reward
            total_follower_reward += follower_reward
            steps += 1
            
            # Render if requested
            if render:
                self.env.render(ax)
                plt.draw()
                plt.pause(0.1)  # Short pause to update display
            
            # Check if episode is done
            if self.env.is_done():
                break
            
            # Update state
            state = next_state
        
        if render:
            plt.ioff()  # Turn off interactive mode
        
        # Return episode statistics
        return {
            'leader_reward': total_leader_reward,
            'follower_reward': total_follower_reward,
            'steps': steps,
            'completion': self.env.is_done()
        }
    
    def train(self, n_episodes=None, render_interval=None):
        """
        Train the agents using Stackelberg learning.
        
        Parameters:
        - n_episodes: Number of episodes to train (uses default if None)
        - render_interval: How often to render an episode (None for no rendering)
        
        Returns:
        - Training statistics
        """
        if n_episodes is None:
            n_episodes = self.n_episodes
        
        print(f"Starting training for {n_episodes} episodes...")
        
        for episode in range(n_episodes):
            # Render occasionally if requested
            render = render_interval is not None and episode % render_interval == 0
            
            # Run the episode
            stats = self.run_episode(training=True, render=render)
            
            # Store statistics
            self.training_stats['leader_rewards'].append(stats['leader_reward'])
            self.training_stats['follower_rewards'].append(stats['follower_reward'])
            self.training_stats['completion_steps'].append(stats['steps'])
            self.training_stats['completion_rates'].append(float(stats['completion']))
            
            # Print progress more frequently for long training runs
            if episode % 10 == 0 or (n_episodes > 100 and episode % 50 == 0):
                print(f"Episode {episode}/{n_episodes}: "
                      f"Leader Reward = {stats['leader_reward']:.2f}, "
                      f"Follower Reward = {stats['follower_reward']:.2f}, "
                      f"Steps = {stats['steps']}")
                
            # Save checkpoint for long training runs
            if n_episodes >= 1000 and episode > 0 and episode % 200 == 0:
                print(f"Saving checkpoint at episode {episode}...")
                # Create a checkpoint of training statistics
                checkpoint = {
                    'episode': episode,
                    'leader_rewards': self.training_stats['leader_rewards'],
                    'follower_rewards': self.training_stats['follower_rewards'],
                    'completion_steps': self.training_stats['completion_steps'],
                    'completion_rates': self.training_stats['completion_rates']
                }
                try:
                    import pickle
                    import os
                    # Create checkpoints directory if it doesn't exist
                    os.makedirs('checkpoints', exist_ok=True)
                    # Save checkpoint
                    with open(f'checkpoints/training_checkpoint_ep{episode}.pkl', 'wb') as f:
                        pickle.dump(checkpoint, f)
                except Exception as e:
                    print(f"Warning: Could not save checkpoint: {e}")
        
        print("Training complete!")
        return self.training_stats
    
    def evaluate(self, n_episodes=10, render=False):
        """
        Evaluate the trained agents.
        
        Parameters:
        - n_episodes: Number of episodes to evaluate
        - render: Whether to render the evaluation episodes
        
        Returns:
        - Evaluation statistics
        """
        eval_stats = {
            'leader_rewards': [],
            'follower_rewards': [],
            'completion_steps': [],
            'completion_rates': []
        }
        
        print(f"Evaluating for {n_episodes} episodes...")
        
        for episode in range(n_episodes):
            stats = self.run_episode(training=False, render=render)
            
            eval_stats['leader_rewards'].append(stats['leader_reward'])
            eval_stats['follower_rewards'].append(stats['follower_reward'])
            eval_stats['completion_steps'].append(stats['steps'])
            eval_stats['completion_rates'].append(float(stats['completion']))
            
            print(f"Evaluation Episode {episode+1}/{n_episodes}: "
                  f"Leader Reward = {stats['leader_reward']:.2f}, "
                  f"Follower Reward = {stats['follower_reward']:.2f}, "
                  f"Steps = {stats['steps']}")
        
        # Calculate averages
        avg_leader_reward = np.mean(eval_stats['leader_rewards'])
        avg_follower_reward = np.mean(eval_stats['follower_rewards'])
        avg_steps = np.mean(eval_stats['completion_steps'])
        completion_rate = np.mean(eval_stats['completion_rates']) * 100
        
        print(f"Evaluation Results (over {n_episodes} episodes):")
        print(f"Average Leader Reward: {avg_leader_reward:.2f}")
        print(f"Average Follower Reward: {avg_follower_reward:.2f}")
        print(f"Average Steps: {avg_steps:.2f}")
        print(f"Task Completion Rate: {completion_rate:.1f}%")
        
        return eval_stats
    
    def visualize_training_stats(self):
        """
        Visualize the training statistics.
        
        Returns:
        - Matplotlib figure with training plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot rewards
        axes[0, 0].plot(self.training_stats['leader_rewards'], label='Leader')
        axes[0, 0].plot(self.training_stats['follower_rewards'], label='Follower')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].set_title('Rewards per Episode')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot completion steps
        axes[0, 1].plot(self.training_stats['completion_steps'])
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].set_title('Completion Steps per Episode')
        axes[0, 1].grid(True)
        
        # Plot completion rates (using a moving average)
        window_size = min(50, len(self.training_stats['completion_rates']))
        completion_rates = np.array(self.training_stats['completion_rates'])
        moving_avg = np.convolve(completion_rates, np.ones(window_size)/window_size, mode='valid')
        axes[1, 0].plot(moving_avg * 100)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Completion Rate (%)')
        axes[1, 0].set_title(f'Task Completion Rate (Moving Avg, Window={window_size})')
        axes[1, 0].grid(True)
        
        # Plot cumulative rewards
        cum_leader_rewards = np.cumsum(self.training_stats['leader_rewards'])
        cum_follower_rewards = np.cumsum(self.training_stats['follower_rewards'])
        axes[1, 1].plot(cum_leader_rewards, label='Leader')
        axes[1, 1].plot(cum_follower_rewards, label='Follower')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Cumulative Reward')
        axes[1, 1].set_title('Cumulative Rewards')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        return fig


class ReplayBuffer:
    """
    Simple replay buffer for storing and sampling experiences.
    """
    def __init__(self, buffer_size, batch_size, seed):
        """
        Initialize the replay buffer.
        
        Parameters:
        - buffer_size: Maximum size of the buffer
        - batch_size: Size of batches to sample
        - seed: Random seed for sampling
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.buffer = []
    
    def __len__(self):
        """
        Get the current size of the buffer.
        """
        return len(self.buffer)
    
    def add(self, experience):
        """
        Add an experience to the buffer.
        
        Parameters:
        - experience: Experience to add (numpy array)
        """
        if len(self.buffer) == 0:
            self.buffer = experience[np.newaxis, :]
        elif len(self.buffer) < self.buffer_size:
            self.buffer = np.vstack((self.buffer, experience[np.newaxis, :]))
        else:
            # Replace the oldest experience
            self.buffer = np.vstack((self.buffer[1:], experience[np.newaxis, :]))
    
    def sample(self, batch_size=None):
        """
        Sample a batch of experiences from the buffer.
        
        Parameters:
        - batch_size: Size of batch to sample (uses default if None)
        
        Returns:
        - Batch of experiences (numpy array)
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer contains {len(self.buffer)} experiences, but requested batch size is {batch_size}")
        
        indices = self.rng.choice(len(self.buffer), batch_size, replace=False)
        return self.buffer[indices]


def run_simulation():
    """
    Run a demonstration of the battery disassembly simulation.
    """
    # Define simulation parameters
    parameters = {
        'task_id': 1,
        'seed': 42,
        'device': 'cpu',
        'batch_size': 32,
        'buffer_size': 256,
        'update_target_step': 50,
        'episode_size': 1000,  # Updated to 1000 episodes
        'step_per_episode': 40,
        'max_time_steps': 100,
        'franka_failure_prob': 0.1,
        'ur10_failure_prob': 0.1,
        'leader': {
            'learning_rate': 1e-4,
            'momentum': 0.6,
            'reward_decay': 0.9,
            'epsilon': 0.1,
            'tau': 0.1,
            'n1_feature': 64,
            'n2_feature': 64
        },
        'follower': {
            'learning_rate': 1e-4,
            'momentum': 0.6,
            'reward_decay': 0.9,
            'epsilon': 0.1,
            'tau': 0.1,
            'n1_feature': 64,
            'n2_feature': 64
        }
    }
    
    # Create the simulation
    sim = StackelbergGameSimulation(parameters)
    
    # Full training run with 1000 episodes (no rendering for speed)
    print("Starting full training with 1000 episodes...")
    train_stats = sim.train(n_episodes=1000, render_interval=None)
    
    # Evaluate the trained policy (no rendering to avoid errors)
    eval_stats = sim.evaluate(n_episodes=5, render=False)
    
    # Visualize training statistics (static plot, should work fine)
    fig = sim.visualize_training_stats()
    
    return sim


if __name__ == "__main__":
    # Set a specific backend that should be more stable
    import matplotlib
    matplotlib.use('TkAgg')  # You can also try 'Agg' for non-interactive use
    
    # Display backend information
    print(f"Using matplotlib backend: {matplotlib.get_backend()}")
    
    try:
        run_simulation()
    except Exception as e:
        print(f"Error running simulation: {e}")
        import traceback
        traceback.print_exc()