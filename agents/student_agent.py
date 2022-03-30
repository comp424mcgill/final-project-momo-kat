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
    """
    """
    def __init__(self, chess_board, p0_pos, p1_pos, turn, depth, parent=None):
        self.board = chess_board
        self.p0_pos = p0_pos
        self.p1_pos = p1_pos
        self.turn = turn
        self.children = []
        self.parent = parent
        self.depth = depth
        self.isLeaf = False
        self.isTerminal = False
        self.eval = None
    
    def evaluate_state(self):
        """
        Uses a heuristic to evaluate the current state of the board.
        """
        if (self.isTerminal):
            # do not calculate a heuristic, get the actual score
            pass    # TODO remove after implementation
        else:
            # use heuristic
            self.eval = 0

        ##prioritze distance closer to the center of the board
        
        ##prioritize fewer number of moves that opposit player has

        ## prioritize walls in the direction of the opposing player.. unless losing?

        ## prioritize moves that are close to the opponent max steps. (otherwise they can 360 around you)

        ##keep track of closed areas so the "center" of the board might change

        ## def need a way to keep track of walls that are close to finishing, but maybe not if heuristic is good and depth is good too


        ##method called on a game, always assume it's the player's turn to start with
    def minimax(self):
        if self.depth==0 or self.isLeaf==True :
            return self.evaluate_state(self)
        if self.turn==0: ##our turn
            self.eval=-100000 #-inf
            for child in self.children :
                self.eval= max(self.eval, self.minimax(child))
            return self.eval
        else : ##min player
            self.eval=100000
            for child in self.children :
                self.eval= min(self.eval, self.minimax(child))
            return self.eval
        




        

    def check_endgame(self):
        """
        Check if the game ends and compute the current score of the agents.

        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        player_1_score : int
            The score of player 1.
        player_2_score : int
            The score of player 2.
        """
        # Union-Find
        board_size = len(self.board)
        father = dict()
        for r in range(board_size):
            for c in range(board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(
                    moves[1:3]
                ):  # Only check down and right
                    if self.board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_r = find(tuple(self.p0_pos))
        p1_r = find(tuple(self.p1_pos))
        return p0_r != p1_r


def set_barrier(c_board, r, c, dir):
    """
    """
    board = deepcopy(c_board)
    # Set the barrier to True
    board[r, c, dir] = True
    # Set the opposite barrier to True
    move = moves[dir]
    board[r + move[0], c + move[1], opposites[dir]] = True
    return board

def add_walls_to_position(state, r, c, my_pos, adv_pos):
    """
    """
    depth_reached = False
    for d in range(4):
        if state.board[r,c,d]:
            continue
        new_board = set_barrier(state.board, r, c, d)
        
        if (state.turn == 0):
            p0, p1 = my_pos, adv_pos
        else:
            p0, p1 = adv_pos, my_pos
        
        new_state = GameState(new_board, deepcopy(p0), deepcopy(p1), 1-state.turn, state.depth + 1, state)
        
        state.children.append(new_state)

        new_state.parent = state

        if (new_state.depth == tree_depth):
            new_state.isLeaf = True
            depth_reached = True

    return depth_reached

def get_next_states(state):
    """
    """
    global max_steps, moves, tree_depth
    if (state.turn == 0):
        my_pos = state.p0_pos
        adv_pos = state.p1_pos
    else:
        my_pos = state.p1_pos
        adv_pos = state.p0_pos

    state_queue = [(my_pos, 0)]
    visited = {tuple(my_pos)}

    depth_reached = add_walls_to_position(state, my_pos[0], my_pos[1], my_pos, adv_pos)

    while state_queue and not depth_reached:
        cur_pos, cur_step = state_queue.pop()
        r, c = cur_pos

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

            depth_reached = add_walls_to_position(state, next_pos[0], next_pos[1], my_pos, adv_pos)

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
                    if (s.check_endgame()):
                        s.isLeaf = True
                        s.isTerminal = True
                    else:
                        state_queue.append(s)

        root.minimax(root)
        for child in root.children:
            if child.eval == root.eval:
            ## this is the best move
                ## look where we put the wall in that game state, chose that wall to put
                for i in range(4):
                    if(root.board[child.p0_pos[0][child.p0_pos[1]][i]]!=child.board[child.p0_pos[0][child.p0_pos[1]][i]]):
                        return child.p0_pos, i
        #return my_pos, self.dir_map["u"]

