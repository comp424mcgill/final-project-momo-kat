# Student agent: Add your own agent here
from copy import deepcopy
from agents.agent import Agent
from store import register_agent
import numpy as np

moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
# Opposite Directions
opposites = {0: 2, 1: 3, 2: 0, 3: 1}
max_steps = 0
tree_depth = 3
root = None


class GameState:
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
        if self.isTerminal:
            # do not calculate a heuristic, get the actual score

            # if it is a terminal state then maybe we didn't reach this from our move, check who's turn it is and
            # check if we won
            self.eval = -100  # assume it's our loss move for now
        else:
            # use heuristic
            self.eval = 0

        
        ###param1: prioritize distances close to center
        xCenterDistance,yCenterDistance= self.distanceToCenter()
        # prioritize distance closer to the center of the board
        param1=-(xCenterDistance+yCenterDistance)

        # if we know there's a wall, the 'center' of the board should change
        # keep track of closed areas so the "center" of the board might change



        # prioritize fewer number of moves that opposite player has
        # calculated how many moves can the opposite player do from this position

        # prioritize walls in the direction of the opposing player... unless losing?
        # check if there's a wall between you and the opponent

        # prioritize moves that are close to the opponent max steps. (otherwise they can go 360 around you)
        # to see

    ##indicates by how much we won the game
    ## TODO
    def score(self):
        end,p0_score,p1_score=self.check_endgame()
        return p0_score-p1_score


    
    #return a tuple ( x, y) where x,y represents the absolute distance with respect the center in x and in y.
    # ex: for 5x5 board and pos (2,1) we will return (abs(3-2),abs(3-1))
    def distanceToCenter(self)-> tuple:
        boardx,boardy, _ = self.board.shape
        
        ##TODO: check where our position actually is

        mex,mey= self.p0_pos
        mex=mex+1;mey=mey+1 

        if (boardx % 2) == 0:
            ##in the case of even number, the board has 4 centers
            #find closest center:
            if mex > boardx/2:
                boardxCenter=boardx/2+1 
            else:
                boardxCenter=boardx/2

            if mey > boardy/2:
                boardyCenter=boardx/2+1 
            else:
                boardyCenter=boardx/2


            xDist=(mex)-(boardxCenter) #check x distance with respect to xCenter
            yDist=(mey)-(boardyCenter) #check y distance with respect to yCenter
            
        else:
            xDist=(mex)-(boardx/2 + 1) #check x distance with respect to xCenter
            yDist=(mey)-(boardy/2 + 1) #check y distance with respect to yCenter

        return (abs(xDist),abs(yDist))

    ##calculates how many moves the opposit player has
    ##TODO
    def oppenentMoves(self)-> int:
        boardx,boardy, _ = self.board.shape
            
        return 0

    def minimax(self) -> int:

        if self.depth == tree_depth or self.isLeaf:
            self.evaluate_state()
            #  print("value of depth ",tree_depth,": ",self.eval)
            return self.eval

        elif self.turn == 0:  # our turn
            self.eval = -100000  # -inf
            for child in self.children:
                self.eval = max(self.eval, child.minimax())
            #    print("Max: value of depth ",self.depth,": ",max(self.eval, child.minimax()))

            return self.eval
        else:  # min player
            self.eval = 100000
            for child in self.children:
                self.eval = min(self.eval, child.minimax())
            #   print("Min: value of depth ",self.depth,": ",max(self.eval, child.minimax()))

            return self.eval

    # displays the values of the tree by height
    def traverse(self):
        this_level = [self]
        h = 0
        print("level ", h)

        while this_level:
            next_level = list()
            for n in this_level:
                print(n.eval, end=" ")
                next_level = next_level + n.children
            h = h + 1
            print()
            print("level ", h)
            this_level = next_level

    # displays the values of the tree by height. It also indicates which nodes belong to which parent
    def traverse_children(self):
        this_level = [[self]]
        h = 0
        print("level ", h)

        while this_level:
            next_level = []
            i = 0
            print(len(this_level), " nodes on this level\n")
            for item in this_level:
                print(i, ":", ", ", len(item), end=" \n")
                for n in item:
                    print(n.eval, end=" ")
                    next_level.append(n.children)
                print("\n")
                i = i + 1

            h = h + 1
            print()
            if h > tree_depth:
                return
            print("level ", h)

            this_level = next_level

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
                for direction, move in enumerate(
                        moves[1:3]
                ):  # Only check down and right
                    if self.board[r, c, direction + 1]:
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
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        return p0_r != p1_r,p0_score,p1_score


def set_barrier(c_board, r, c, direction):
    """
    """
    board = deepcopy(c_board)
    # Set the barrier to True
    board[r, c, direction] = True
    # Set the opposite barrier to True
    move = moves[direction]
    board[r + move[0], c + move[1], opposites[direction]] = True
    return board


def add_walls_to_position(state, r, c, my_pos, adv_pos):
    """
    """
    states = []
    for d in range(4):
        if state.board[r, c, d]:
            continue
        new_board = set_barrier(state.board, r, c, d)

        if state.turn == 0:
            p0, p1 = my_pos, adv_pos
        else:
            p0, p1 = adv_pos, my_pos

        new_state = GameState(new_board, deepcopy(p0), deepcopy(p1), 1 - state.turn, state.depth + 1, state)

        new_state.parent = state

        if new_state.depth <= tree_depth:
            states.append(new_state)

    return states


def get_next_states(state):
    """
    """
    global max_steps, moves, tree_depth
    next_states = []
    if state.turn == 0:
        my_pos = state.p0_pos
        adv_pos = state.p1_pos
    else:
        my_pos = state.p1_pos
        adv_pos = state.p0_pos

    state_queue = [(my_pos, 0)]
    visited = {tuple(my_pos)}

    these_next_states = add_walls_to_position(state, my_pos[0], my_pos[1], my_pos, adv_pos)

    next_states.extend(these_next_states)

    while state_queue:
        cur_pos, cur_step = state_queue.pop()
        r, c = cur_pos

        if cur_step == max_steps:
            break

        for direction, move in enumerate(moves):
            if state.board[r, c, direction]:
                continue
            next_pos = (cur_pos[0] + move[0], cur_pos[1] + move[1])
            if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                continue

            visited.add(tuple(next_pos))
            state_queue.append((next_pos, cur_step + 1))

            those_next_states = add_walls_to_position(state, next_pos[0], next_pos[1], my_pos, adv_pos)

            next_states.extend(those_next_states)
    return next_states


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
        state_queue = [root]

        while state_queue:
            curr = state_queue.pop()
            new_states = get_next_states(curr)
            for s in new_states:
                curr.children.append(s)
                endGame,_,_=s.check_endgame()
                if endGame:
                    s.isLeaf = True
                    s.isTerminal = True
                elif s.depth == tree_depth:
                    s.isLeaf = True
                else:
                    state_queue.append(s)

        print()
        print('state of tree before minimax... ')
        root.traverse_children()
        print()
        print("calculating ...", end="")
        root.minimax()
        root.traverse_children()  # will show the tree after minimax
        print('state of tree before minimax... ')
        
        for child in root.children:
            if child.eval == root.eval:
            ## this is the best move
                ## look where we put the wall in that game state, chose that wall to put
                for i in range(4):
                    if(root.board[child.p0_pos[0][child.p0_pos[1]][i]]!=child.board[child.p0_pos[0][child.p0_pos[1]][i]]):
                        return child.p0_pos, i
        return my_pos, self.dir_map["u"]
