# Student agent: Add your own agent here
from copy import deepcopy
from agents.agent import Agent
from store import register_agent
import numpy as np
import colorama
from colorama import Fore

moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
# Opposite Directions
opposites = {0: 2, 1: 3, 2: 0, 3: 1}
max_steps = 0
tree_depth = 2
currentMove = 0
counterForUs=0
root = None
import time

def setupTreeDepth(chess_board):
    boardSize,_, _ = chess_board.shape
    global tree_depth
    global currentMove
    global counterForUs
    playingAgainstUs=False ## set to true if you are playing against yourself, aka studentagent v studentagent
    if playingAgainstUs:
        
        if counterForUs ==1 :
            if currentMove!=0:currentMove-=1
            counterForUs=0
        elif currentMove!=0:
            counterForUs+=1
        
    if boardSize==4:
        if currentMove < 3: #first  moves 
            tree_depth=3
        elif currentMove < 4: #third move
            tree_depth=4
        elif currentMove < 5: #4th move
            tree_depth=5
        else:
            tree_depth=6
    
    elif boardSize==5:
        if currentMove <1:
            tree_depth=2
        elif currentMove < 8: #first 8 moves 
            tree_depth=3
        else:
            tree_depth=4
    
    elif boardSize==6:
        if currentMove < 5:
            tree_depth=2
        else:
            tree_depth=3
    
    elif boardSize==7:
        if currentMove < 15: #first  moves 
            tree_depth=2
        else: #else
            tree_depth=3
        
    elif boardSize==8:
        if currentMove < 25: #first  moves 
            tree_depth=2
        else: #else
            tree_depth=3
    
    elif boardSize==9:
        if currentMove < 40: #first  moves 
            tree_depth=2
        else: #else
            tree_depth=3
    elif boardSize==10:
        if currentMove < 10:
            tree_depth=1
        elif currentMove < 75: #first  moves 
            tree_depth=2
        else: #else
            tree_depth=3
    currentMove=currentMove+1
    

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
        self.p0s = None
        self.p1s= None
        self.eval = None

    def evaluate_state(self):
        """
        Uses a heuristic to evaluate the current state of the board.
        """
        if self.isTerminal:
            
            score= self.p0s-self.p1s #the bigger the gap the better

            self.eval = score *1000  # assume it's our loss move for now
        else:

            ###param1: prioritize distances close to center 
            param1=self.distanceToCenter()

            # param2: number of moves for us or opposing player
            # param2= self.oppenentMoves()
            # if self.depth%2==0:
            #     ## then the opponent just played their move we should have a high number indicating our choices of move
            #     param2=param2
            # else:
            #     #then we just played, opponent should have a low number, so prioritize small numbers
            #     param2=-param2
             # use heuristic parameter optimizations:
            self.eval =  param1*5

    
    #return a tuple ( x, y) where x,y represents the absolute distance with respect the center in x and in y.
    # ex: for 5x5 board and pos (2,1) we will return (abs(3-2),abs(3-1))
    def distanceToCenter(self)-> tuple:
        
        boardx,boardy, _ = self.board.shape
        #for normalization, check what is the maximum distance from center in this case
        
        if (boardx % 2) == 0:
            maxDistance = (boardx/2 -1 )*2 
        else:
            maxDistance = int(boardx/2) * 2
        
        

        #our positoin
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

        xCenterDistance,yCenterDistance =(abs(xDist),abs(yDist))
        param1=-(xCenterDistance+yCenterDistance)
        param1=param1/maxDistance  ## for example if distance to center is 5 and maxdistance is 10 we returned -0.5

        return param1

    def oppenentMoves(self)-> int:
        
        ##for normalization: check what is normally considered to be the maximum number of moves in this checkboard size
        boardx,boardy, _ = self.board.shape
        stepSize= (boardx + 1) // 2
        maxMoves=0
        if boardx %2==0:
            maxMoves= stepSize*stepSize + ((stepSize+1)*(stepSize+1)) ##counting the diagonals in the area 
            maxMoves=maxMoves-2 #since the board is even
            maxMoves=maxMoves*4 #4 choices per side of the board
        else:
            maxMoves= stepSize*stepSize + ((stepSize+1)*(stepSize+1)) ##counting the diagonals in the area 
            maxMoves=maxMoves-4 #since the board is odd
            maxMoves=maxMoves*4 #4 choices per side of the board

        return len(get_next_states(self))/maxMoves

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
        debug=True
        this_level = [[self]]
        h = 0
        print("level ", h)

        while this_level:
            next_level = []
            i = 0
            if debug:
                nodesinThisLevel=0
                for item in this_level:
                    nodesinThisLevel=nodesinThisLevel+len(item)
                print(nodesinThisLevel, " nodes on this level\n")
            for item in this_level:
                print(i, ": ", len(item), end=" \n")
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
        returns:
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

    #resets current move if start of game
    def check_startgame(self):
        global currentMove
        totalWalls= 0
        board_size ,_, _ = self.board.shape
        for r in range(board_size):
            for c in range(board_size):
                for direction in range(4):
                    if self.board[r, c, direction ] == True:
                        totalWalls+=1
        totalWalls=(totalWalls-(board_size*4))/2

        if totalWalls==max_steps*2 or totalWalls==max_steps*2+1:
            currentMove=0
            return True
        else: 
            return False

        


        

########helper methods for next moves

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

            those_next_states = add_walls_to_position(state, next_pos[0], next_pos[1], next_pos, adv_pos)

            next_states.extend(those_next_states)
    return next_states



@register_agent("student_agent")
class StudentAgent(Agent):

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.autoplay = True

    def step(self, chess_board, my_pos, adv_pos, max_step):
        
        start = time.time()
        
        global max_steps, root, tree_depth
        max_steps = max_step

        cb_copy = deepcopy(chess_board)
        my_pos_copy = deepcopy(my_pos)
        adv_pos_copy = deepcopy(adv_pos)   

        root = GameState(cb_copy, my_pos_copy, adv_pos_copy, 0, 0)
        state_queue = [root]
        root.check_startgame()
        setupTreeDepth(chess_board) 

        while state_queue:
            curr = state_queue.pop()
            new_states = get_next_states(curr)
            for s in new_states:
                curr.children.append(s)
                endGame,p0s,p1s=s.check_endgame()
                if endGame:
                    s.isLeaf = True
                    s.isTerminal = True
                    s.p0s=p0s
                    s.p1s=p1s
                elif s.depth == tree_depth:
                    s.isLeaf = True
                else:
                    state_queue.append(s)

     
        root.minimax()
        
        for child in root.children:
            if child.eval == root.eval:
            ## this is the best move
                ## look where we put the wall in that game state, chose that wall to put
                for i in range(4):

                    if(root.board[child.p0_pos[0],child.p0_pos[1],i]!=child.board[child.p0_pos[0],child.p0_pos[1],i]):
                        # print("returned: ",child.p0_pos, i)
                        end = time.time()
                        if(end - start)>=2:
                            print(Fore.RED+"tree depth:",tree_depth,",boardsize:", len(cb_copy))
                            print(Fore.RED+"Move",currentMove-1,"took time:",end - start)
                        else:
                            print(Fore.WHITE+"tree depth:",tree_depth,",boardsize:", len(cb_copy))
                            print(Fore.WHITE+"Move",currentMove-1,"took time:",end - start)
                        return child.p0_pos, i

