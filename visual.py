class GridVisual:
    def show_board(grid):
        ROWS = grid.ROWS-1
        COLS = grid.COLS -1

        for x in range( ROWS - 1, 0,-1):
            print('----------------------')
            out = '| '
            for y in range(1, COLS):
                if (x, y) == grid.state:
                    out += '2 | '  # 2 for the robot location
                elif grid.board[x][y] == 3:
                    out += 'G | '  # for the goal cell
                elif grid.board[x][y] == 1:
                    out += '# | '  # draw # for walls or obstacles
                elif grid.board[x][y] == 0:
                    out += '  | '  # 0  == 'space' for empty cell
            print(out)
        print('-----------------------')
