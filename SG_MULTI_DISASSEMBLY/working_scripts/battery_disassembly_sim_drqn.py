import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import random
import pickle


class RecurrentQNetwork(nn.Module):
    """
    Deep Recurrent Q-Network implementation using LSTM.
    This handles temporal dependencies through recurrent layers.
    """
    def __init__(self, input_dim, action_dim_1, action_dim_2, hidden_size=64, lstm_layers=1):
        super(RecurrentQNetwork, self).__init__()
        self.input_dim = input_dim
        self.action_dim_1 = action_dim_1  # Leader action space size
        self.action_dim_2 = action_dim_2  # Follower action space size
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU()
        )
        
        # LSTM layer for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )
        
        # Output layer for Q-values
        self.output_layer = nn.Linear(hidden_size, action_dim_1 * action_dim_2)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights with small random values."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, state, hidden_state=None):
        """
        Forward pass through the network.
        
        Parameters:
        - state: Batch of state sequences [batch_size, seq_len, state_dim]
        - hidden_state: Initial hidden state for LSTM
        
        Returns:
        - Q-values for all action pairs
        - Final hidden state
        """
        batch_size, seq_len, _ = state.shape
        
        # Extract features
        features = self.feature_extractor(state.view(-1, self.input_dim))
        features = features.view(batch_size, seq_len, self.hidden_size)
        
        # Pass through LSTM
        if hidden_state is None:
            lstm_out, hidden_state = self.lstm(features)
        else:
            lstm_out, hidden_state = self.lstm(features, hidden_state)
        
        # Generate Q-values
        q_values = self.output_layer(lstm_out)
        
        return q_values, hidden_state
    
    def get_q_values(self, state, hidden_state=None):
        """
        Get Q-values for a single state.
        
        Parameters:
        - state: Single state tensor [state_dim]
        - hidden_state: Hidden state for LSTM
        
        Returns:
        - Q-values for all action pairs
        - Updated hidden state
        """
        # Add batch and sequence dimensions if not present
        if len(state.shape) == 1:
            state = state.unsqueeze(0).unsqueeze(0)  # [1, 1, state_dim]
        elif len(state.shape) == 2:
            state = state.unsqueeze(0)  # [1, seq_len, state_dim]
        
        # Forward pass
        q_values, new_hidden_state = self.forward(state, hidden_state)
        
        # Return last timestep's Q-values
        return q_values[:, -1, :].view(self.action_dim_1, self.action_dim_2), new_hidden_state


class SequenceReplayBuffer:
    """
    Replay buffer for storing and sampling sequences of experiences.
    """
    def __init__(self, buffer_size, sequence_length, state_dim, batch_size, seed):
        self.buffer_size = buffer_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.buffer = []
        self.episode_buffer = []
    
    def __len__(self):
        """Get the current size of the buffer."""
        return len(self.buffer)
    
    def add(self, experience):
        """
        Add an experience to the episode buffer.
        
        Parameters:
        - experience: Experience to add [state, a1, a2, r1, r2, next_state]
        """
        self.episode_buffer.append(experience)
    
    def end_episode(self):
        """
        End the current episode and transfer sequences to the main buffer.
        """
        if len(self.episode_buffer) == 0:
            return
        
        # Add overlapping sequences from the episode to the buffer
        for i in range(max(1, len(self.episode_buffer) - self.sequence_length + 1)):
            sequence = self.episode_buffer[i:i+self.sequence_length]
            if len(sequence) < self.sequence_length:
                # Pad shorter sequences
                padding = [sequence[-1]] * (self.sequence_length - len(sequence))
                sequence.extend(padding)
            
            self.buffer.append(sequence)
            
            # Maintain buffer size
            if len(self.buffer) > self.buffer_size:
                self.buffer.pop(0)
        
        self.episode_buffer = []
    
    def sample(self, batch_size=None):
        """
        Sample a batch of sequences from the buffer.
        
        Parameters:
        - batch_size: Size of batch to sample (uses default if None)
        
        Returns:
        - Batch of sequence experiences
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer contains {len(self.buffer)} sequences, but requested batch size is {batch_size}")
        
        indices = self.rng.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]


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


class StackelbergDRQNAgent:
    """
    Agent implementation using Deep Recurrent Q-Networks for Stackelberg games.
    """
    def __init__(self, state_dim, action_dim_leader, action_dim_follower, 
                 hidden_size=64, sequence_length=8, device='cpu', learning_rate=1e-4,
                 gamma=0.9, epsilon=0.1, epsilon_decay=0.995, epsilon_min=0.01,
                 tau=0.01, update_every=10, seed=42):
        """
        Initialize the Stackelberg DRQN agent.
        
        Parameters:
        - state_dim: Dimension of the state space
        - action_dim_leader: Dimension of the leader's action space
        - action_dim_follower: Dimension of the follower's action space
        - hidden_size: Hidden layer size in the recurrent network
        - sequence_length: Length of sequences for training
        - device: Device to run the model on (cpu or cuda)
        - learning_rate: Learning rate for optimizer
        - gamma: Discount factor for future rewards
        - epsilon: Exploration rate
        - epsilon_decay: Rate at which epsilon decays over time
        - epsilon_min: Minimum value for epsilon
        - tau: Soft update parameter for target network
        - update_every: How often to update the target network
        - seed: Random seed
        """
        self.state_dim = state_dim
        self.action_dim_leader = action_dim_leader
        self.action_dim_follower = action_dim_follower
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.tau = tau
        self.update_every = update_every
        self.seed = seed
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize leader and follower networks
        self.leader_online = RecurrentQNetwork(
            state_dim, action_dim_leader, action_dim_follower, 
            hidden_size).to(device)
        self.leader_target = RecurrentQNetwork(
            state_dim, action_dim_leader, action_dim_follower, 
            hidden_size).to(device)
        self.follower_online = RecurrentQNetwork(
            state_dim, action_dim_leader, action_dim_follower, 
            hidden_size).to(device)
        self.follower_target = RecurrentQNetwork(
            state_dim, action_dim_leader, action_dim_follower, 
            hidden_size).to(device)
        
        # Initialize target networks with same weights as online networks
        self.leader_target.load_state_dict(self.leader_online.state_dict())
        self.follower_target.load_state_dict(self.follower_online.state_dict())
        
        # Initialize optimizers
        self.leader_optimizer = optim.Adam(self.leader_online.parameters(), lr=learning_rate)
        self.follower_optimizer = optim.Adam(self.follower_online.parameters(), lr=learning_rate)
        
        # Initialize hidden states
        self.leader_hidden = None
        self.follower_hidden = None
        
        # Initialize training step counter
        self.t_step = 0
    
    def compute_stackelberg_equilibrium(self, state):
        """
        Compute Stackelberg equilibrium using the current Q-networks.
        
        Parameters:
        - state: Current environment state
        
        Returns:
        - leader_action, follower_action: Equilibrium actions
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        # Get Q-values for all possible action pairs
        leader_q_values, self.leader_hidden = self.leader_online.get_q_values(
            state_tensor, self.leader_hidden)
        follower_q_values, self.follower_hidden = self.follower_online.get_q_values(
            state_tensor, self.follower_hidden)
        
        # Convert to numpy for easier manipulation
        leader_q = leader_q_values.detach().cpu().numpy()
        follower_q = follower_q_values.detach().cpu().numpy()
        
        # Compute the follower's best response for each leader action
        follower_best_responses = np.argmax(follower_q, axis=1)
        
        # Leader chooses the action that maximizes its utility given follower's best response
        leader_values = np.array([leader_q[i, follower_best_responses[i]] 
                                 for i in range(self.action_dim_leader)])
        leader_action = np.argmax(leader_values)
        
        # Get the corresponding follower action (best response)
        follower_action = follower_best_responses[leader_action]
        
        # Convert from index to actual action (-1 to n-1)
        return leader_action - 1, follower_action - 1
    
    def reset_hidden_states(self):
        """Reset the hidden states for both agents."""
        self.leader_hidden = None
        self.follower_hidden = None
    
    def act(self, state, epsilon=None):
        """
        Select actions according to epsilon-greedy policy.
        
        Parameters:
        - state: Current environment state
        - epsilon: Exploration rate (uses default if None)
        
        Returns:
        - leader_action, follower_action: Selected actions
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        # With probability epsilon, select random actions
        if np.random.random() < epsilon:
            leader_action = np.random.randint(-1, self.action_dim_leader - 1)
            follower_action = np.random.randint(-1, self.action_dim_follower - 1)
            return leader_action, follower_action
        
        # Otherwise, compute and return Stackelberg equilibrium actions
        return self.compute_stackelberg_equilibrium(state)
    
    def update(self, experiences):
        """
        Update the Q-networks using a batch of experiences.
        
        Parameters:
        - experiences: List of (state, action, reward, next_state, done) tuples
        """
        # Convert experiences to tensors
        states = []
        leader_actions = []
        follower_actions = []
        leader_rewards = []
        follower_rewards = []
        next_states = []
        dones = []
        
        # Process each sequence of experiences
        for sequence in experiences:
            seq_states = []
            seq_leader_actions = []
            seq_follower_actions = []
            seq_leader_rewards = []
            seq_follower_rewards = []
            seq_next_states = []
            seq_dones = []
            
            for exp in sequence:
                s, a_l, a_f, r_l, r_f, s_next = exp
                # Convert actions to indices (add 1 to handle -1 actions)
                a_l_idx = a_l + 1
                a_f_idx = a_f + 1
                # Determine if state is terminal
                done = np.all(s_next == 0)
                
                seq_states.append(s)
                seq_leader_actions.append(a_l_idx)
                seq_follower_actions.append(a_f_idx)
                seq_leader_rewards.append(r_l)
                seq_follower_rewards.append(r_f)
                seq_next_states.append(s_next)
                seq_dones.append(done)
            
            states.append(seq_states)
            leader_actions.append(seq_leader_actions)
            follower_actions.append(seq_follower_actions)
            leader_rewards.append(seq_leader_rewards)
            follower_rewards.append(seq_follower_rewards)
            next_states.append(seq_next_states)
            dones.append(seq_dones)
        
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        leader_actions = torch.tensor(leader_actions, dtype=torch.long).to(self.device)
        follower_actions = torch.tensor(follower_actions, dtype=torch.long).to(self.device)
        leader_rewards = torch.tensor(leader_rewards, dtype=torch.float).to(self.device)
        follower_rewards = torch.tensor(follower_rewards, dtype=torch.float).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)
        
        batch_size, seq_len = states.shape[0], states.shape[1]
        
        # Compute Q-values for current states
        leader_q_values, _ = self.leader_online(states)
        follower_q_values, _ = self.follower_online(states)
        
        # Reshape for easier indexing
        leader_q_values = leader_q_values.view(batch_size, seq_len, self.action_dim_leader, self.action_dim_follower)
        follower_q_values = follower_q_values.view(batch_size, seq_len, self.action_dim_leader, self.action_dim_follower)
        
        # Gather Q-values for taken actions
        leader_q = leader_q_values.gather(2, leader_actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.action_dim_follower))
        leader_q = leader_q.gather(3, follower_actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, -1))
        follower_q = follower_q_values.gather(2, leader_actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.action_dim_follower))
        follower_q = follower_q.gather(3, follower_actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, -1))
        
        leader_q = leader_q.squeeze(-1).squeeze(-1)
        follower_q = follower_q.squeeze(-1).squeeze(-1)
        
        # Compute target Q-values for next states
        with torch.no_grad():
            # Get next state Q-values from target networks
            next_leader_q, _ = self.leader_target(next_states)
            next_follower_q, _ = self.follower_target(next_states)
            
            # Reshape for easier manipulation
            next_leader_q = next_leader_q.view(batch_size, seq_len, self.action_dim_leader, self.action_dim_follower)
            next_follower_q = next_follower_q.view(batch_size, seq_len, self.action_dim_leader, self.action_dim_follower)
            
            # For each batch and sequence step, compute Stackelberg equilibrium
            leader_targets = torch.zeros_like(leader_rewards)
            follower_targets = torch.zeros_like(follower_rewards)
            
            for b in range(batch_size):
                for s in range(seq_len):
                    if dones[b, s]:
                        # If done, target is just the reward
                        leader_targets[b, s] = leader_rewards[b, s]
                        follower_targets[b, s] = follower_rewards[b, s]
                    else:
                        # Compute follower's best response for each leader action
                        follower_best_responses = torch.argmax(next_follower_q[b, s], dim=1)
                        
                        # Leader chooses action to maximize its utility given follower's response
                        leader_values = torch.stack([
                            next_leader_q[b, s, a, follower_best_responses[a]] 
                            for a in range(self.action_dim_leader)
                        ])
                        leader_action = torch.argmax(leader_values)
                        follower_action = follower_best_responses[leader_action]
                        
                        # Compute target values using Bellman equation
                        leader_targets[b, s] = leader_rewards[b, s] + self.gamma * next_leader_q[b, s, leader_action, follower_action]
                        follower_targets[b, s] = follower_rewards[b, s] + self.gamma * next_follower_q[b, s, leader_action, follower_action]
        
        # Compute loss for both agents
        leader_loss = torch.nn.functional.mse_loss(leader_q, leader_targets)
        follower_loss = torch.nn.functional.mse_loss(follower_q, follower_targets)
        
        # Optimize leader network
        self.leader_optimizer.zero_grad()
        leader_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.leader_online.parameters(), 1)  # Gradient clipping
        self.leader_optimizer.step()
        
        # Optimize follower network
        self.follower_optimizer.zero_grad()
        follower_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.follower_online.parameters(), 1)  # Gradient clipping
        self.follower_optimizer.step()
        
        # Soft update target networks
        self.t_step += 1
        if self.t_step % self.update_every == 0:
            self.soft_update(self.leader_online, self.leader_target)
            self.soft_update(self.follower_online, self.follower_target)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return leader_loss.item(), follower_loss.item()
    
    def soft_update(self, online_model, target_model):
        """
        Soft update of target network parameters.
        θ_target = τ*θ_online + (1 - τ)*θ_target
        
        Parameters:
        - online_model: Online network
        - target_model: Target network
        """
        for target_param, online_param in zip(target_model.parameters(), online_model.parameters()):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data)
    
    def save(self, path):
        """
        Save the agent's state.
        
        Parameters:
        - path: Directory to save to
        """
        os.makedirs(path, exist_ok=True)
        
        torch.save(self.leader_online.state_dict(), f"{path}/leader_online.pt")
        torch.save(self.leader_target.state_dict(), f"{path}/leader_target.pt")
        torch.save(self.follower_online.state_dict(), f"{path}/follower_online.pt")
        torch.save(self.follower_target.state_dict(), f"{path}/follower_target.pt")
        
        params = {
            "epsilon": self.epsilon,
            "t_step": self.t_step
        }
        
        with open(f"{path}/params.pkl", "wb") as f:
            pickle.dump(params, f)
    
    def load(self, path):
        """
        Load the agent's state.
        
        Parameters:
        - path: Directory to load from
        """
        self.leader_online.load_state_dict(torch.load(f"{path}/leader_online.pt"))
        self.leader_target.load_state_dict(torch.load(f"{path}/leader_target.pt"))
        self.follower_online.load_state_dict(torch.load(f"{path}/follower_online.pt"))
        self.follower_target.load_state_dict(torch.load(f"{path}/follower_target.pt"))
        
        with open(f"{path}/params.pkl", "rb") as f:
            params = pickle.load(f)
            self.epsilon = params["epsilon"]
            self.t_step = params["t_step"]


class StackelbergDRQNSimulation:
    """
    Main simulation class for the Stackelberg game using DRQN.
    """
    def __init__(self, parameters):
        """
        Initialize the simulation.
        
        Parameters:
        - parameters: Dictionary containing simulation parameters
        """
        self.env = BatteryDisassemblyEnv(parameters)
        self.device = parameters.get('device', 'cpu')
        
        # Extract environment information
        env_info = self.env.get_task_info()
        state_dim = env_info['dims']
        action_dim_leader = env_info['dimAl']
        action_dim_follower = env_info['dimAf']
        
        # Initialize agent
        self.agent = StackelbergDRQNAgent(
            state_dim=state_dim,
            action_dim_leader=action_dim_leader,
            action_dim_follower=action_dim_follower,
            hidden_size=parameters.get('hidden_size', 64),
            sequence_length=parameters.get('sequence_length', 8),
            device=self.device,
            learning_rate=parameters.get('learning_rate', 1e-4),
            gamma=parameters.get('gamma', 0.9),
            epsilon=parameters.get('epsilon', 0.1),
            epsilon_decay=parameters.get('epsilon_decay', 0.995),
            epsilon_min=parameters.get('epsilon_min', 0.01),
            tau=parameters.get('tau', 0.01),
            update_every=parameters.get('update_every', 10),
            seed=parameters.get('seed', 42)
        )
        
        # Initialize replay buffer
        self.buffer = SequenceReplayBuffer(
            buffer_size=parameters.get('buffer_size', 10000),
            sequence_length=parameters.get('sequence_length', 8),
            state_dim=state_dim,
            batch_size=parameters.get('batch_size', 32),
            seed=parameters.get('seed', 42)
        )
        
        # Training parameters
        self.n_episodes = parameters.get('episode_size', 1000)
        self.n_steps_per_episode = parameters.get('step_per_episode', 40)
        self.batch_size = parameters.get('batch_size', 32)
        
        # Statistics tracking
        self.training_stats = {
            'leader_rewards': [],
            'follower_rewards': [],
            'completion_steps': [],
            'completion_rates': [],
            'leader_losses': [],
            'follower_losses': []
        }
    
    def generate_initial_buffer(self, n_episodes=10):
        """
        Generate initial experiences using random actions.
        
        Parameters:
        - n_episodes: Number of episodes to generate
        """
        print("Generating initial experiences...")
        
        for episode in range(n_episodes):
            self.env.reset_env()
            state, _ = self.env.get_current_state()
            
            for step in range(self.n_steps_per_episode):
                # Choose random actions
                leader_action = np.random.randint(-1, self.env.task_board.shape[1])
                follower_action = np.random.randint(-1, self.env.task_board.shape[1])
                
                # Get rewards and update environment
                leader_reward, follower_reward = self.env.reward(state, leader_action, follower_action)
                self.env.step(leader_action, follower_action)
                next_state, _ = self.env.get_current_state()
                
                # Store experience
                experience = [state, leader_action, follower_action, leader_reward, follower_reward, next_state]
                self.buffer.add(experience)
                
                # Check if done
                if self.env.is_done():
                    break
                
                state = next_state
            
            # End episode in buffer
            self.buffer.end_episode()
        
        print(f"Initial buffer size: {len(self.buffer)}")
    
    def train(self, n_episodes=None, render_interval=None):
        """
        Train the agents using DRQN.
        
        Parameters:
        - n_episodes: Number of episodes to train (uses default if None)
        - render_interval: How often to render an episode (None for no rendering)
        
        Returns:
        - Training statistics
        """
        if n_episodes is None:
            n_episodes = self.n_episodes
        
        # Generate initial experiences if buffer is empty
        if len(self.buffer) < self.batch_size:
            self.generate_initial_buffer()
        
        print(f"Starting training for {n_episodes} episodes...")
        
        for episode in range(n_episodes):
            # Reset environment and agent hidden states
            self.env.reset_env()
            self.agent.reset_hidden_states()
            state, _ = self.env.get_current_state()
            
            episode_leader_reward = 0
            episode_follower_reward = 0
            episode_leader_losses = []
            episode_follower_losses = []
            steps = 0
            
            # Create figure for rendering if needed
            if render_interval is not None and episode % render_interval == 0:
                render = True
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                plt.ion()  # Turn on interactive mode
            else:
                render = False
            
            # Run the episode
            for step in range(self.n_steps_per_episode):
                # Select actions using current policy
                leader_action, follower_action = self.agent.act(state)
                
                # Get rewards and update environment
                leader_reward, follower_reward = self.env.reward(state, leader_action, follower_action)
                self.env.step(leader_action, follower_action)
                next_state, _ = self.env.get_current_state()
                
                # Store experience
                experience = [state, leader_action, follower_action, leader_reward, follower_reward, next_state]
                self.buffer.add(experience)
                
                # Update statistics
                episode_leader_reward += leader_reward
                episode_follower_reward += follower_reward
                steps += 1
                
                # Update networks if enough experiences are available
                if len(self.buffer) >= self.batch_size:
                    experiences = self.buffer.sample(self.batch_size)
                    leader_loss, follower_loss = self.agent.update(experiences)
                    episode_leader_losses.append(leader_loss)
                    episode_follower_losses.append(follower_loss)
                
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
            
            # End episode in buffer
            self.buffer.end_episode()
            
            if render:
                plt.ioff()  # Turn off interactive mode
            
            # Store episode statistics
            self.training_stats['leader_rewards'].append(episode_leader_reward)
            self.training_stats['follower_rewards'].append(episode_follower_reward)
            self.training_stats['completion_steps'].append(steps)
            self.training_stats['completion_rates'].append(float(self.env.is_done()))
            
            if episode_leader_losses:
                self.training_stats['leader_losses'].append(np.mean(episode_leader_losses))
                self.training_stats['follower_losses'].append(np.mean(episode_follower_losses))
            
            # Print progress
            if episode % 10 == 0 or (n_episodes > 100 and episode % 50 == 0):
                print(f"Episode {episode}/{n_episodes}: "
                      f"Leader Reward = {episode_leader_reward:.2f}, "
                      f"Follower Reward = {episode_follower_reward:.2f}, "
                      f"Steps = {steps}, "
                      f"Epsilon = {self.agent.epsilon:.3f}")
            
            # Save checkpoint for long training runs
            if n_episodes >= 1000 and episode > 0 and episode % 200 == 0:
                print(f"Saving checkpoint at episode {episode}...")
                self.agent.save(f"checkpoints/drqn_episode_{episode}")
                
                # Save training statistics
                checkpoint = {
                    'episode': episode,
                    'leader_rewards': self.training_stats['leader_rewards'],
                    'follower_rewards': self.training_stats['follower_rewards'],
                    'completion_steps': self.training_stats['completion_steps'],
                    'completion_rates': self.training_stats['completion_rates'],
                    'leader_losses': self.training_stats['leader_losses'],
                    'follower_losses': self.training_stats['follower_losses']
                }
                
                try:
                    os.makedirs('checkpoints', exist_ok=True)
                    with open(f'checkpoints/drqn_stats_ep{episode}.pkl', 'wb') as f:
                        pickle.dump(checkpoint, f)
                except Exception as e:
                    print(f"Warning: Could not save checkpoint: {e}")
        
        print("Training complete!")
        
        # Save final model
        self.agent.save("checkpoints/drqn_final")
        
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
        
        # Set agent to evaluation mode (epsilon=0)
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0
        
        for episode in range(n_episodes):
            # Reset environment and agent hidden states
            self.env.reset_env()
            self.agent.reset_hidden_states()
            state, _ = self.env.get_current_state()
            
            episode_leader_reward = 0
            episode_follower_reward = 0
            steps = 0
            
            # Create figure for rendering if needed
            if render:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                plt.ion()  # Turn on interactive mode
            
            # Run the episode
            for step in range(self.n_steps_per_episode):
                # Select actions using current policy (no exploration)
                leader_action, follower_action = self.agent.act(state, epsilon=0)
                
                # Get rewards and update environment
                leader_reward, follower_reward = self.env.reward(state, leader_action, follower_action)
                self.env.step(leader_action, follower_action)
                next_state, _ = self.env.get_current_state()
                
                # Update statistics
                episode_leader_reward += leader_reward
                episode_follower_reward += follower_reward
                steps += 1
                
                # Render if requested
                if render:
                    self.env.render(ax)
                    plt.draw()
                    plt.pause(0.2)  # Longer pause to view the simulation
                
                # Check if episode is done
                if self.env.is_done():
                    break
                
                # Update state
                state = next_state
            
            if render:
                plt.ioff()  # Turn off interactive mode
            
            # Store episode statistics
            eval_stats['leader_rewards'].append(episode_leader_reward)
            eval_stats['follower_rewards'].append(episode_follower_reward)
            eval_stats['completion_steps'].append(steps)
            eval_stats['completion_rates'].append(float(self.env.is_done()))
            
            print(f"Evaluation Episode {episode+1}/{n_episodes}: "
                  f"Leader Reward = {episode_leader_reward:.2f}, "
                  f"Follower Reward = {episode_follower_reward:.2f}, "
                  f"Steps = {steps}")
        
        # Restore agent's original epsilon
        self.agent.epsilon = original_epsilon
        
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
        fig, axes = plt.subplots(3, 2, figsize=(12, 14))
        
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
        
        # Plot losses if available
        if self.training_stats.get('leader_losses'):
            axes[2, 0].plot(self.training_stats['leader_losses'], label='Leader')
            axes[2, 0].plot(self.training_stats['follower_losses'], label='Follower')
            axes[2, 0].set_xlabel('Episode')
            axes[2, 0].set_ylabel('Loss')
            axes[2, 0].set_title('TD Losses')
            axes[2, 0].legend()
            axes[2, 0].grid(True)
            
            # Plot epsilon decay
            episodes = np.arange(len(self.training_stats['leader_losses']))
            epsilon_values = 0.1 * np.power(0.995, episodes)
            axes[2, 1].plot(episodes, epsilon_values)
            axes[2, 1].set_xlabel('Episode')
            axes[2, 1].set_ylabel('Epsilon')
            axes[2, 1].set_title('Exploration Rate Decay')
            axes[2, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        return fig


def run_drqn_simulation():
    """
    Run a simulation using the DRQN implementation.
    """
    # Define simulation parameters
    parameters = {
        'task_id': 1,
        'seed': 42,
        'device': 'cpu',
        'batch_size': 32,
        'buffer_size': 10000,
        'sequence_length': 8,
        'update_every': 10,
        'episode_size': 1000,
        'step_per_episode': 40,
        'max_time_steps': 100,
        'franka_failure_prob': 0.1,
        'ur10_failure_prob': 0.1,
        'hidden_size': 64,
        'learning_rate': 1e-4,
        'gamma': 0.9,
        'epsilon': 0.1,
        'epsilon_decay': 0.995,
        'epsilon_min': 0.01,
        'tau': 0.01
    }
    
    # Create the simulation
    sim = StackelbergDRQNSimulation(parameters)
    
    # Train the agent
    print("Starting DRQN training with 1000 episodes...")
    train_stats = sim.train(n_episodes=1000, render_interval=None)
    
    # Evaluate the trained policy
    eval_stats = sim.evaluate(n_episodes=5, render=False)
    
    # Visualize training statistics
    fig = sim.visualize_training_stats()
    
    return sim


if __name__ == "__main__":
    # Set a specific backend that should be more stable
    import matplotlib
    matplotlib.use('TkAgg')  # You can also try 'Agg' for non-interactive use
    
    # Display backend information
    print(f"Using matplotlib backend: {matplotlib.get_backend()}")
    
    try:
        run_drqn_simulation()
    except Exception as e:
        print(f"Error running simulation: {e}")
        import traceback
        traceback.print_exc()