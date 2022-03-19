# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import numpy as np
import sys

moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
max_steps = 0;


class GameState():
    def __init__(self, chess_board, p1_pos, p2_pos):
        self.chess_board = chess_board
        self.p1_pos = p1_pos
        self.p2_pos = p2_pos
        self.next_states = []


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
        global max_steps
        max_steps = max_step
        states = get_states(chess_board, my_pos, adv_pos, max_step)
        print(len(states[1]))
        # dummy return
        return my_pos, self.dir_map["u"]

def get_states(chess_board, start_pos, adv_pos, max_step):
    # BFS
    state_queue = [(start_pos, 0)]
    visited = {tuple(start_pos)}
    walled_states = []
    while state_queue:
        cur_pos, cur_step = state_queue.pop(0)
        r, c = cur_pos
        if cur_step == max_step:
            break
        for dir, move in enumerate(moves):
            if chess_board[r, c, dir]:
                continue
            
            next_pos = (cur_pos[0] + move[0], cur_pos[1] + move[1])
            if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                continue

            visited.add(tuple(next_pos))
            for d in range(4):
                if not chess_board[r, c, d]:
                    walled_states.append({next_pos, d})
            state_queue.append((next_pos, cur_step + 1))

    return walled_states

