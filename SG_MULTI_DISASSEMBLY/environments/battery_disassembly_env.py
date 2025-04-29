import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class BatteryDisassemblyEnv:
    """
    Environment class for the battery disassembly task with three robots.
    This environment models a workstation with a battery module and three robots:
    - Franka robot (Leader): Equipped with a two-finger gripper for unbolting operations
    - UR10 robot (Follower 1): Equipped with vacuum suction for sorting and pick-and-place
    - Kuka robot (Follower 2): Equipped with specialized tools for casing and connections
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
        self.franka_pos = np.array([0.5, -0.3, 0.5])   # Base position of Franka robot (Leader)
        self.ur10_pos = np.array([-0.5, -0.3, 0.5])    # Base position of UR10 robot (Follower 1)
        self.kuka_pos = np.array([0.0, -0.5, 0.5])     # Base position of Kuka robot (Follower 2)
        
        # Define workspace properties
        self.battery_pos = np.array([0.0, 0.0, 0.1])  # Position of the battery module
        self.bin_positions = {
            'screws': np.array([0.3, 0.4, 0.1]),
            'cells': np.array([-0.3, 0.4, 0.1]),
            'casings': np.array([0.0, 0.5, 0.1]),
            'connectors': np.array([0.3, -0.4, 0.1])  # New bin for connectors
        }
        
        # Task completion tracking
        self.completed_tasks = []
        
        # Robot states
        self.franka_state = {'position': self.franka_pos, 'gripper_open': True, 'holding': None}
        self.ur10_state = {'position': self.ur10_pos, 'suction_active': False, 'holding': None}
        self.kuka_state = {'position': self.kuka_pos, 'tool_active': False, 'holding': None}
        
        # Task timing and resource tracking
        self.time_step = 0
        self.max_time_steps = parameters.get('max_time_steps', 100)
        
        # Robot kinematic constraints
        self.franka_workspace_radius = 0.8
        self.ur10_workspace_radius = 1.0
        self.kuka_workspace_radius = 0.9
        
        # Task failure probabilities (uncertainty modeling)
        self.franka_failure_prob = parameters.get('franka_failure_prob', 0.1)
        self.ur10_failure_prob = parameters.get('ur10_failure_prob', 0.1)
        self.kuka_failure_prob = parameters.get('kuka_failure_prob', 0.1)
        
    def task_reader(self, task_id):
        """
        Read the task information from the configuration files.
        Extended for three-robot scenario with more task types.
        """
        # Task board represents the spatial arrangement of components to be disassembled
        # 0: Empty space
        # 1-4: Top screws (requires unbolting by Franka)
        # 5-8: Side screws (requires unbolting by Franka)
        # 9-12: Battery cells (requires pick-and-place by UR10)
        # 13-16: Casing components (requires specialized tools by Kuka)
        # 17-20: Connectors (requires collaborative effort between UR10 and Kuka)
        # 21-22: Complex assemblies (requires all three robots)
        task_board = np.array([
            [1, 2, 3, 4],
            [9, 10, 11, 12],
            [17, 18, 19, 20],
            [5, 6, 7, 8],
            [13, 14, 15, 16],
            [21, 21, 22, 22]
        ])
        
        # Task properties define the characteristics of each task
        # type 1: Leader-specific tasks (unbolting by Franka)
        # type 2: Follower1-specific tasks (pick-and-place by UR10)
        # type 3: Follower2-specific tasks (casing work by Kuka)
        # type 4: Collaborative tasks between Follower1 and Follower2
        # type 5: Tasks requiring Leader and one Follower
        # type 6: Complex tasks requiring all three robots
        
        # Create a type array matching the size of the largest task ID
        max_task_id = np.max(task_board)
        type_array = np.zeros(max_task_id + 1, dtype=int)
        
        # Assign task types
        type_array[1:9] = 1      # Franka tasks (screws)
        type_array[9:13] = 2     # UR10 tasks (battery cells)
        type_array[13:17] = 3    # Kuka tasks (casing)
        type_array[17:21] = 4    # UR10 + Kuka collaborative tasks (connectors)
        type_array[21:23] = 6    # All three robots (complex assemblies)
        
        # Success probabilities for each robot on different task types
        l_succ = np.zeros(max_task_id + 1)
        f1_succ = np.zeros(max_task_id + 1)
        f2_succ = np.zeros(max_task_id + 1)
        
        # Set success probabilities based on task types
        # Type 1: Leader (Franka) tasks
        l_succ[type_array == 1] = 0.9
        f1_succ[type_array == 1] = 0.0
        f2_succ[type_array == 1] = 0.0
        
        # Type 2: Follower 1 (UR10) tasks
        l_succ[type_array == 2] = 0.0
        f1_succ[type_array == 2] = 0.9
        f2_succ[type_array == 2] = 0.0
        
        # Type 3: Follower 2 (Kuka) tasks
        l_succ[type_array == 3] = 0.0
        f1_succ[type_array == 3] = 0.0
        f2_succ[type_array == 3] = 0.9
        
        # Type 4: Follower 1 + Follower 2 collaborative tasks
        l_succ[type_array == 4] = 0.0
        f1_succ[type_array == 4] = 0.7
        f2_succ[type_array == 4] = 0.7
        
        # Type 5: Leader + Follower collaborative tasks (not in this board)
        
        # Type 6: All three robots collaborative tasks
        l_succ[type_array == 6] = 0.7
        f1_succ[type_array == 6] = 0.7
        f2_succ[type_array == 6] = 0.7
        
        # Shape indicates the physical size/complexity (affects timing)
        shape_array = np.ones(max_task_id + 1, dtype=int)
        shape_array[0] = 0  # Empty space has no shape
        shape_array[type_array == 6] = 3  # Complex tasks have larger shape value
        
        task_prop = {
            'type': type_array,
            'shape': shape_array,
            'l_succ': l_succ,
            'f1_succ': f1_succ,
            'f2_succ': f2_succ
        }
        
        return task_board, task_prop

    def get_task_info(self):
        """
        Get task information for initializing the learning algorithms.
        """
        info = {}
        info['task_id'] = self.task_id
        info['dims'] = self.task_board.shape[1]
        info['dimAl'] = self.task_board.shape[1] + 1   # +1 for "do nothing" action
        info['dimAf1'] = self.task_board.shape[1] + 1  # +1 for "do nothing" action
        info['dimAf2'] = self.task_board.shape[1] + 1  # +1 for "do nothing" action
        info['dimal'] = 1
        info['dimaf1'] = 1
        info['dimaf2'] = 1
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
        self.kuka_state = {'position': self.kuka_pos, 'tool_active': False, 'holding': None}
    
    def step(self, al, af1, af2):
        """
        Execute one step in the environment based on all three robots' actions.
        
        Parameters:
        - al: Leader action (Franka robot)
        - af1: Follower 1 action (UR10 robot)
        - af2: Follower 2 action (Kuka robot)
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
        
        # Simulate if task is completed by follower 1 (UR10)
        if af1 == -1:
            tf1, tf1_done = 0, False  # Follower 1 does nothing
        else:
            tf1 = self.curr_board[0, af1]
            if tf1 == 0:
                tf1_done = False  # Task already completed or invalid
            else:
                # Check if task is within UR10's capabilities and workspace
                if self.is_task_feasible(tf1, 'follower1'):
                    tf1_done = True if self.rng.uniform() < self.task_prop['f1_succ'][tf1] else False
                else:
                    tf1_done = False
        
        # Simulate if task is completed by follower 2 (Kuka)
        if af2 == -1:
            tf2, tf2_done = 0, False  # Follower 2 does nothing
        else:
            tf2 = self.curr_board[0, af2]
            if tf2 == 0:
                tf2_done = False  # Task already completed or invalid
            else:
                # Check if task is within Kuka's capabilities and workspace
                if self.is_task_feasible(tf2, 'follower2'):
                    tf2_done = True if self.rng.uniform() < self.task_prop['f2_succ'][tf2] else False
                else:
                    tf2_done = False
        
        # Update the task board based on the simulated results
        self.update_board(tl, tl_done, tf1, tf1_done, tf2, tf2_done)
        
        # Update robot positions based on actions
        if tl_done or al != -1:
            self.update_robot_position('leader', al)
        
        if tf1_done or af1 != -1:
            self.update_robot_position('follower1', af1)
            
        if tf2_done or af2 != -1:
            self.update_robot_position('follower2', af2)
        
        # Increment time step
        self.time_step += 1
    
    def is_task_feasible(self, task_id, robot):
        """
        Check if a task is feasible for the given robot based on capabilities and workspace constraints.
        
        Parameters:
        - task_id: ID of the task to check
        - robot: 'leader', 'follower1', or 'follower2'
        
        Returns:
        - Boolean indicating if the task is feasible
        """
        # Check robot capability based on task type
        task_type = self.task_prop['type'][task_id]
        
        if robot == 'leader':
            # Leader can do type 1 tasks, and participate in type 5 & 6 collaborative tasks
            if task_type not in [1, 5, 6]:
                return False
                
            # Check if Franka can reach the task
            task_pos = self.get_task_position(task_id)
            if task_pos is None:
                return False
            dist = np.linalg.norm(task_pos - self.franka_state['position'])
            return dist <= self.franka_workspace_radius
            
        elif robot == 'follower1':
            # Follower1 can do type 2 tasks, and participate in type 4, 5 (with leader), & 6 collaborative tasks
            if task_type not in [2, 4, 5, 6]:
                return False
                
            # Check if UR10 can reach the task
            task_pos = self.get_task_position(task_id)
            if task_pos is None:
                return False
            dist = np.linalg.norm(task_pos - self.ur10_state['position'])
            return dist <= self.ur10_workspace_radius
            
        elif robot == 'follower2':
            # Follower2 can do type 3 tasks, and participate in type 4, 5 (with leader), & 6 collaborative tasks
            if task_type not in [3, 4, 5, 6]:
                return False
                
            # Check if Kuka can reach the task
            task_pos = self.get_task_position(task_id)
            if task_pos is None:
                return False
            dist = np.linalg.norm(task_pos - self.kuka_state['position'])
            return dist <= self.kuka_workspace_radius
    
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
        - robot: 'leader', 'follower1', or 'follower2'
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
        
        elif robot == 'follower1':
            # Move UR10 to the task position
            self.ur10_state['position'] = task_pos
            # Update suction state based on task type
            task_id = self.curr_board[0, action]
            if task_id > 0:
                task_type = self.task_prop['type'][task_id]
                self.ur10_state['suction_active'] = task_type == 2  # Activate suction for pick-and-place
                self.ur10_state['holding'] = task_id if task_type == 2 else None
        
        elif robot == 'follower2':
            # Move Kuka to the task position
            self.kuka_state['position'] = task_pos
            # Update tool state based on task type
            task_id = self.curr_board[0, action]
            if task_id > 0:
                task_type = self.task_prop['type'][task_id]
                self.kuka_state['tool_active'] = task_type == 3  # Activate tool for casing work
                self.kuka_state['holding'] = task_id if task_type == 3 else None
    
    def update_board(self, tl, tl_done, tf1, tf1_done, tf2, tf2_done):
        """
        Update the task board based on completed tasks.
        
        Parameters:
        - tl: Leader's task ID
        - tl_done: Whether leader's task was completed
        - tf1: Follower1's task ID
        - tf1_done: Whether follower1's task was completed
        - tf2: Follower2's task ID
        - tf2_done: Whether follower2's task was completed
        """
        # Handle single robot tasks
        if tl != 0 and tl_done and self.task_prop['type'][tl] == 1:
            idx = np.where(self.curr_board[0] == tl)[0]
            self.curr_board[0, idx] = 0
            self.completed_tasks.append(tl)
            
        if tf1 != 0 and tf1_done and self.task_prop['type'][tf1] == 2:
            idx = np.where(self.curr_board[0] == tf1)[0]
            self.curr_board[0, idx] = 0
            self.completed_tasks.append(tf1)
            
        if tf2 != 0 and tf2_done and self.task_prop['type'][tf2] == 3:
            idx = np.where(self.curr_board[0] == tf2)[0]
            self.curr_board[0, idx] = 0
            self.completed_tasks.append(tf2)
        
        # Handle collaborative tasks between the two followers (type 4)
        if tf1 == tf2 and tf1 != 0:
            task_type = self.task_prop['type'][tf1]
            if task_type == 4 and tf1_done and tf2_done:
                idx = np.where(self.curr_board[0] == tf1)[0]
                self.curr_board[0, idx] = 0
                self.completed_tasks.append(tf1)
        
        # Handle three-robot collaborative tasks (type 6)
        if tl == tf1 and tf1 == tf2 and tl != 0:
            task_type = self.task_prop['type'][tl]
            if task_type == 6 and tl_done and tf1_done and tf2_done:
                idx = np.where(self.curr_board[0] == tl)[0]
                self.curr_board[0, idx] = 0
                self.completed_tasks.append(tl)
        
        # Update subsequent rows (task dependencies)
        for i in range(self.task_board.shape[0] - 1):
            curr_row, next_row = self.curr_board[i, :], self.curr_board[i+1, :]
            
            # Find tasks that may drop from the next row
            task_list = []
            idx = np.where(curr_row == 0)[0]
            for j in idx:
                task_id = next_row[j]
                if task_id != 0 and task_id not in task_list:    # task 0 does not count
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
    
    def reward(self, s, al, af1, af2):
        """
        Calculate rewards for all three robots based on their actions.
        
        Parameters:
        - s: Current state (first row of the board)
        - al: Leader's action
        - af1: Follower1's action
        - af2: Follower2's action
        
        Returns:
        - rl, rf1, rf2: Rewards for leader, follower1, and follower2
        """
        # Determine task IDs corresponding to the actions
        tl = 0 if al == -1 else s[al]
        tf1 = 0 if af1 == -1 else s[af1]
        tf2 = 0 if af2 == -1 else s[af2]
        
        # Initialize rewards
        rl, rf1, rf2 = 0, 0, 0
        
        # All robots idle
        if tl == 0 and tf1 == 0 and tf2 == 0:
            if al == -1 and af1 == -1 and af2 == -1:
                rl, rf1, rf2 = -0.5, -0.5, -0.5  # All idle (slight penalty)
            else:
                # One or more robots attempting empty tasks
                rl = 0 if al == -1 else -1
                rf1 = 0 if af1 == -1 else -1
                rf2 = 0 if af2 == -1 else -1
        
        # Check for collaborative task situations
        
        # Two-robot follower collaboration (type 4)
        if tf1 == tf2 and tf1 != 0 and self.task_prop['type'][tf1] == 4:
            rf1, rf2 = 2, 2  # Higher reward for collaborative task
            rl = 0 if al == -1 else -1  # Leader should not interfere
        
        # Three-robot collaboration (type 6)
        elif tl == tf1 and tf1 == tf2 and tl != 0 and self.task_prop['type'][tl] == 6:
            rl, rf1, rf2 = 3, 3, 3  # Highest reward for full collaboration
        
        # Handle non-collaborative or individual tasks
        else:
            # Process leader's reward
            if tl == 0:
                rl = 0 if al == -1 else -1  # Either idle or empty task
            elif self.task_prop['type'][tl] == 1:
                rl = 1  # Leader-appropriate task
            else:
                rl = -1  # Leader-inappropriate task
            
            # Process follower1's reward
            if tf1 == 0:
                rf1 = 0 if af1 == -1 else -1  # Either idle or empty task
            elif self.task_prop['type'][tf1] == 2:
                rf1 = 1  # Follower1-appropriate task
            else:
                rf1 = -1  # Follower1-inappropriate task
                
            # Process follower2's reward
            if tf2 == 0:
                rf2 = 0 if af2 == -1 else -1  # Either idle or empty task
            elif self.task_prop['type'][tf2] == 3:
                rf2 = 1  # Follower2-appropriate task
            else:
                rf2 = -1  # Follower2-inappropriate task
        
        # Additional penalties for task conflicts (non-collaborative tasks)
        if tl == tf1 and tl != 0 and self.task_prop['type'][tl] not in [5, 6]:
            rl -= 1
            rf1 -= 1
            
        if tl == tf2 and tl != 0 and self.task_prop['type'][tl] not in [5, 6]:
            rl -= 1
            rf2 -= 1
            
        if tf1 == tf2 and tf1 != 0 and self.task_prop['type'][tf1] != 4:
            rf1 -= 1
            rf2 -= 1
        
        return float(rl), float(rf1), float(rf2)
    
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
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
        
        # Clear previous plot
        ax.clear()
        
        # Plot workstation surface
        x = np.linspace(-0.8, 0.8, 10)
        y = np.linspace(-0.6, 0.6, 10)
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
                    if task_pos is not None:
                        task_type = self.task_prop['type'][task_id]
                        
                        # Color based on task type
                        if task_type == 1:
                            color = 'blue'        # Leader tasks
                        elif task_type == 2:
                            color = 'green'       # Follower1 tasks
                        elif task_type == 3:
                            color = 'orange'      # Follower2 tasks
                        elif task_type == 4:
                            color = 'purple'      # Follower1 + Follower2 collaborative tasks
                        elif task_type == 5:
                            color = 'magenta'     # Leader + Follower collaborative tasks
                        else:  # type 6
                            color = 'red'         # All three robots
                        
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
            'Franka (L)', 'blue', 
            self.franka_state['gripper_open'], 
            self.franka_state['holding']
        )
        
        # UR10 robot (follower1)
        self._plot_robot(
            ax, self.ur10_state['position'], 
            'UR10 (F1)', 'green', 
            not self.ur10_state['suction_active'], 
            self.ur10_state['holding']
        )
        
        # Kuka robot (follower2)
        self._plot_robot(
            ax, self.kuka_state['position'], 
            'Kuka (F2)', 'orange', 
            not self.kuka_state['tool_active'], 
            self.kuka_state['holding']
        )
        
        # Add legend explaining task types
        ax.text(0.7, 0.5, 0.1, 'Task Types:', fontweight='bold')
        ax.scatter(0.7, 0.45, 0.1, color='blue', s=50)
        ax.text(0.75, 0.45, 0.1, 'Type 1: Leader (Unbolting)')
        
        ax.scatter(0.7, 0.4, 0.1, color='green', s=50)
        ax.text(0.75, 0.4, 0.1, 'Type 2: Follower1 (Pick & Place)')
        
        ax.scatter(0.7, 0.35, 0.1, color='orange', s=50)
        ax.text(0.75, 0.35, 0.1, 'Type 3: Follower2 (Casing)')
        
        ax.scatter(0.7, 0.3, 0.1, color='purple', s=50)
        ax.text(0.75, 0.3, 0.1, 'Type 4: F1 + F2 Collaborative')
        
        ax.scatter(0.7, 0.25, 0.1, color='red', s=50)
        ax.text(0.75, 0.25, 0.1, 'Type 6: All Three Robots')
        
        # Set plot limits and labels
        ax.set_xlim([-0.8, 0.8])
        ax.set_ylim([-0.6, 0.7])
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
        - gripper_open: Boolean indicating if gripper/tool is inactive
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
        
        # Indicate tool state
        tool_state = "Inactive" if gripper_open else "Active"
        ax.text(
            position[0], position[1], position[2] - 0.05,
            tool_state, 
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