# Student agent: Add your own agent here
from copy import deepcopy
from agents.agent import Agent
from store import register_agent
import numpy as np
import sys

moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
# Opposite Directions
opposites = {0: 2, 1: 3, 2: 0, 3: 1}
max_steps = 0
tree_depth = 3
root = None

class GameState():
    def __init__(self, chess_board, my_pos, adv_pos, turn, depth, parent=None):
        self.board = chess_board
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.turn = turn
        self.children = []
        self.parent = parent
        self.depth = depth
        self.eval = None
    
    def evaluate_state(self):
        """
        Uses a heuristic to evaluate the current state of the board.
        """
        self.eval = 0

def set_barrier(c_board, r, c, dir):
    board = deepcopy(c_board)
    # Set the barrier to True
    board[r, c, dir] = True
    # Set the opposite barrier to True
    move = moves[dir]
    board[r + move[0], c + move[1], opposites[dir]] = True
    return board

def get_next_states(state):
    global max_steps, moves, tree_depth
    if (state.turn == 0):
        my_pos = state.my_pos
        adv_pos = state.adv_pos
    else:
        my_pos = state.adv_pos
        adv_pos = state.my_pos

    depth_reached = False

    state_queue = [(my_pos, 0)]
    visited = {tuple(my_pos)}
    walled_states = []

    while state_queue:
        cur_pos, cur_step = state_queue.pop()
        r, c = cur_pos
        if cur_step == 0:
            for d in range(4):
                if state.board[r,c,d]:
                    continue
                new_board = set_barrier(state.board, my_pos[0], my_pos[1], d)
                
                if (state.turn == 0):
                    new_state = GameState(new_board, my_pos, adv_pos, 1-state.turn, state.depth + 1, state)
                else:
                    new_state = GameState(new_board, adv_pos, my_pos, 1-state.turn, state.depth + 1, state)
                
                state.children.append(new_state)

                new_state.parent = state

                if (new_state.depth == tree_depth):
                    new_state.evaluate_state()
                    depth_reached = True

        if cur_step == max_steps:
            break

        for dir, move in enumerate(moves):
            if state.board[r, c, dir]:
                continue
            next_pos = (cur_pos[0] + move[0], cur_pos[1] + move[1])
            if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                continue

            visited.add(tuple(next_pos))
            state_queue.append((next_pos, cur_step + 1))

            for d in range(4):
                if state.board[next_pos[0], next_pos[1], d]:
                    continue
                walled_states.append((next_pos[0], next_pos[1], d))

                new_board = set_barrier(state.board, next_pos[0], next_pos[1], d)
                
                if (state.turn == 0):
                    new_state = GameState(new_board, next_pos, adv_pos, 1-state.turn, state.depth + 1, state)
                else:
                    new_state = GameState(new_board, adv_pos, next_pos, 1-state.turn, state.depth + 1, state)
                
                state.children.append(new_state)

                new_state.parent = state

                if (new_state.depth == tree_depth):
                    new_state.evaluate_state()
                    depth_reached = True

    return state.children, depth_reached

@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        global max_steps, root, tree_depth
        max_steps = max_step

        cb_copy = deepcopy(chess_board)
        my_pos_copy = deepcopy(my_pos)
        adv_pos_copy = deepcopy(adv_pos)

        root = GameState(cb_copy, my_pos_copy, adv_pos_copy, 0, 0)
        depth_reached = False
        state_queue = [root]

        while state_queue:
            curr = state_queue.pop()
            new_states, depth_reached = get_next_states(curr)
            if (depth_reached):
                break
            else:
                for s in new_states:
                    state_queue.append(s)

        # dummy return
        return my_pos, self.dir_map["u"]

