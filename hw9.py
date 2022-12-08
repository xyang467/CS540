import random
from copy import deepcopy

class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]
        
    def succ(self, state, current):
        successors = []
        count = 0
        empty = set()
        place = set()
        
        for i in range(5):
            for j in range(5):
                if state[i][j] == ' ':
                    empty.add((i,j))
                elif state[i][j] == current:
                    count+= 1
                    place.add((i,j))
                else:
                    count+= 1

        # drop phase            
        if count < 8:
            for i,j in empty:
                s = deepcopy(state)
                s[i][j] = current
                successors.append(s)
            
        # continued gameplay
        else:
            for i,j in place:
                adj = set()
                keep = set()
                adj.add((i-1,j-1))
                adj.add((i-1,j))
                adj.add((i-1,j+1))
                adj.add((i,j-1))
                adj.add((i,j+1))
                adj.add((i+1,j-1))
                adj.add((i+1,j))
                adj.add((i+1,j+1))
                for a in adj:
                    if not ((-1 in a) or (5 in a)):
                        keep.add(a)
                inter = keep.intersection(empty)
                empty.difference(inter)
                for x,y in inter:
                    s = deepcopy(state)
                    s[x][y] = current
                    s[i][j] = " "
                    successors.append(s)
        return successors
    
    def heuristic_game_value(self, state):
        if self.game_value(state) != 1 and self.game_value(state) != -1:
            my = []
            opp = []
            for row in state:
                for i in range(2):
                    check = [row[i],row[i + 1],row[i + 2],row[i + 3]]
                    my.append(check.count(self.my_piece))
                    opp.append(check.count(self.opp))

            for col in range(5):
                for i in range(2):
                    check = [state[i][col],state[i + 1][col],state[i + 2][col],state[i + 3][col]]
                    my.append(check.count(self.my_piece))
                    opp.append(check.count(self.opp))


            for d in range(2):
                for d2 in range(2):
                    check = [state[d][d2], state[d+1][d2+1], state[d+2][d2+2], state[d+3][d2+3]]
                    my.append(check.count(self.my_piece))
                    opp.append(check.count(self.opp))


            for x in range(2):
                for x2 in range(3,5):
                    check = [state[x][x2],state[x+1][x2-1], state[x+2][x2-2], state[x+3][x2-3]]
                    my.append(check.count(self.my_piece))
                    opp.append(check.count(self.opp))

            for y in range(4):
                for y2 in range(4):
                    check = [state[y][y2], state[y][y2+1], state[y+1][y2+1], state[y+1][y2]]
                    my.append(check.count(self.my_piece))
                    opp.append(check.count(self.opp))
            
            my_max = max(my)
            opp_max = max(opp)
            
            if my_max == opp_max:
                return 0
            elif my_max > opp_max:
                return my_max/4
            else:
                return -opp_max/4
            
            
    def max_value(self,state,d):
        if self.game_value(state) == 1:
            return 1,state
        elif self.game_value(state) == -1:
            return -1,state
        elif d == 0:
            return self.heuristic_game_value(state),state
        else:
            alpha = float('-Inf')
            st = state
            for s in self.succ(state,self.my_piece):
                a = self.min_value(s,d-1)[0]
                if a > alpha:
                    alpha = a
                    st = s
        return alpha,st
    
    def min_value(self,state,d):
        if self.game_value(state) == 1:
            return 1,state
        elif self.game_value(state) == -1:
            return -1,state
        elif d == 0:
            return self.heuristic_game_value(state),state
        else:
            beta = float('Inf')
            for s in self.succ(state, self.opp):
                b = self.max_value(s,d-1)[0]
                if b < beta:
                    beta = b
                    st = s
        return beta,st

        
    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """
        count = 0
        for i in range(5):
            for j in range(5):
                if state[i][j] != ' ':
                    count += 1
        if count < 8:
            drop_phase = True 
        else:
            drop_phase = False
        move = []
        v,change = self.max_value(state,3)
        if not drop_phase:
            for i in range(5):
                for j in range(5):
                    if state[i][j] != change[i][j]:
                        if state[i][j] != " ":
                            move.insert(1, (i, j)) 
                        else:
                             move.insert(0, (i, j))
            return move
        if drop_phase:
            for i in range(5):
                for j in range(5):
                    if state[i][j] != change[i][j]:
                        move.insert(0, (i, j))
                        return move
    
    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row) + ": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")
        
        
    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner

        TODO: complete checks for diagonal and box wins
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i + 1] == row[i + 2] == row[i + 3]:
                    return 1 if row[i] == self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i + 1][col] == state[i + 2][col] == state[i + 3][col]:
                    return 1 if state[i][col] == self.my_piece else -1
                
        # check \ diagonal wins
        for d in range(2):
            for d2 in range(2):
                if state[d][d2] != ' ' and state[d][d2] == state[d+1][d2+1] == state[d+2][d2+2] == state[d+3][d2+3]:
                    return 1 if state[d][d2] == self.my_piece else -1

        # check / diagonal wins
        for x in range(2):
            for x2 in range(3,5):
                if state[x][x2] != ' ' and state[x][x2]==state[x+1][x2-1] == state[x+2][x2-2] == state[x+3][x2-3]:
                    return 1 if state[x][x2] == self.my_piece else -1
        
        # check box wins
        for y in range(4):
            for y2 in range(4):
                if state[y][y2] != ' ' and state[y][y2] == state[y][y2+1]==state[y+1][y2+1] == state[y+1][y2]:
                    return 1 if state[y][y2] == self.my_piece else -1
                
        return 0  # no winner yet


############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece + " moved at " + chr(move[0][1] + ord("A")) + str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp + "'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0]) - ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece + " moved from " + chr(move[1][1] + ord("A")) + str(move[1][0]))
            print("  to " + chr(move[0][1] + ord("A")) + str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp + "'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0]) - ord("A")),
                                      (int(move_from[1]), ord(move_from[0]) - ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()
