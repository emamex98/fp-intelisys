#------------------------------------------------------------------------------------------------------------------
#   Tic Tac Toe game.
#
#   This code is an adaptation of the Tic Tac Toe bot described in:
#   Artificial intelligence with Python. Alberto Artasanchez and Prateek Joshi. 2nd edition, 2020, 
#   editorial Pack. Chapter 13.
#
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
#   Imports
#------------------------------------------------------------------------------------------------------------------

from easyAI import TwoPlayersGame, Human_Player, AI_Player, Negamax
from itertools import groupby, chain
#------------------------------------------------------------------------------------------------------------------
#   Class definitions
#------------------------------------------------------------------------------------------------------------------

NONE = '.'
RED = 'R'
YELLOW = 'Y'

def diagonalsPos (matrix, cols, rows):
    """Get positive diagonals, going from bottom-left to top-right."""
    for di in ([(j, i - j) for j in range(cols)] for i in range(cols + rows -1)):
        yield [matrix[i][j] for i, j in di if i >= 0 and j >= 0 and i < cols and j < rows]

def diagonalsNeg (matrix, cols, rows):
    """Get negative diagonals, going from top-left to bottom-right."""
    for di in ([(j, i - cols + j + 1) for j in range(cols)] for i in range(cols + rows - 1)):
        yield [matrix[i][j] for i, j in di if i >= 0 and j >= 0 and i < cols and j < rows]

class ConnectFour(TwoPlayersGame):
    """ Class that is used to play the TIC TAC TOE game. """

    def __init__(self, players, cols = 7, rows = 6, requiredToWin = 4):
        """ 
            This constructor initializes the game according to the specified players.

            players : The list with the player objects.
        """
        self.cols = cols
        self.rows = rows
        self.win = requiredToWin
        # Define the players
        self.players = players

        # Define who starts the game
        self.nplayer = 1 

        # Define the board
        self.board = [[NONE] * rows for _ in range(cols)]
    
    def show(self):
        """Print the board."""
        print('  '.join(map(str, range(self.cols))))
        for y in range(self.rows):
            print('  '.join(str(self.board[x][y]) for x in range(self.cols)))
        print()

    def possible_moves(self):
        """ This method returns the possible moves according to the current game state. """        
        return [x for x in range(self.cols) if self.board[x][0]==NONE]
    
    def make_move(self, move):
        """ 
            This method executes the specified move.

            move : The move to execute.
        """
        #self.board[int(move) - 1] = self.nplayer
        c = self.board[move]

        if c[0] != NONE:
            # raise Exception('Column is full')
            print("*** Column is full ***")
        i = -1
        while c[i] != NONE:
            i -= 1
        c[i] = self.player.name

    def getWinner (self):
        """Get the winner on the current board."""
        lines = (
            self.board, # columns
            zip(*self.board), # rows
            diagonalsPos(self.board, self.cols, self.rows), # positive diagonals
            diagonalsNeg(self.board, self.cols, self.rows) # negative diagonals
        )

        for line in chain(*lines):
            for color, group in groupby(line):
                if color != NONE and len(list(group)) >= self.win:
                    return color

    def loss_condition(self):
        """ This method returns whether the opponent has three in a line. """
        w = self.getWinner()
        if w:
            return True
        else:
            return False 
    
    def is_over(self):
        """ This method returns whether the game is over. """
        return (self.possible_moves() == []) or self.loss_condition()
        
    def scoring(self):
        """ This method computes the game score (-100 for loss condition, 0 otherwise). """
        return -100 if self.loss_condition() else 0

#------------------------------------------------------------------------------------------------------------------
#   Main function
#------------------------------------------------------------------------------------------------------------------
def main():

    # Search algorithm of the AI player
    algorithm = Negamax(7)

    # Start the game
    ConnectFour([Human_Player('R'), AI_Player(algorithm, 'Y')]).play()

if __name__ == '__main__':
    main()

#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------