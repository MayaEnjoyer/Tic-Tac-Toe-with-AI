import pygame
import numpy as np
import random

pygame.init()

WIDTH, HEIGHT = 300, 350
LINE_WIDTH = 5
BOARD_ROWS, BOARD_COLS = 3, 3
SQUARE_SIZE = WIDTH // BOARD_COLS
CIRCLE_RADIUS = SQUARE_SIZE // 3
CIRCLE_WIDTH = 15
CROSS_WIDTH = 25
SPACE = SQUARE_SIZE // 4

BG_COLOR = (28, 170, 156)
LINE_COLOR = (23, 145, 135)
CIRCLE_COLOR = (239, 231, 200)
CROSS_COLOR = (66, 66, 66)
BUTTON_COLOR = (200, 50, 50)
BUTTON_TEXT_COLOR = (255, 255, 255)

font = pygame.font.Font(None, 40)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tic Tac Toe")
screen.fill(BG_COLOR)

board = np.zeros((BOARD_ROWS, BOARD_COLS))
Q_table = {}


def board_to_state(board):
    """Converts a board to a string: -1 -> '2', 0 -> '0', 1 -> '1'"""
    return ''.join('2' if x == -1 else '0' if x == 0 else '1' for x in board.flatten())


def get_available_actions(state):
    """Converts the status bar back to an array and returns the available actions"""
    board = np.array([int(x) if x != '2' else -1 for x in state]).reshape((BOARD_ROWS, BOARD_COLS))
    return [(row, col) for row in range(BOARD_ROWS) for col in range(BOARD_COLS) if board[row, col] == 0]


def get_Q(state, action):
    """Gets the Q value for the state and action"""
    return Q_table.get((state, action), 0)


def update_Q(state, action, reward, next_state, alpha=0.1, gamma=0.9):
    """Updates the Q-value"""
    max_next_Q = max([get_Q(next_state, a) for a in get_available_actions(next_state)], default=0)
    current_Q = get_Q(state, action)
    Q_table[(state, action)] = current_Q + alpha * (reward + gamma * max_next_Q - current_Q)


def choose_action(state, epsilon=0.2):
    """Selects an action based on a Q-table or randomly"""
    available_actions = get_available_actions(state)
    if not available_actions:
        return None

    if random.uniform(0, 1) > epsilon:
        opponent = -1
        block_move = block_opponent(board, opponent)
        if block_move:
            return block_move

    if random.uniform(0, 1) < epsilon:
        return random.choice(available_actions)

    return max(available_actions, key=lambda a: get_Q(state, a))


def get_reward(winner, player, is_defense=False):
    """Returns the reward added for blocking or winning"""
    if winner == player:
        return 1
    elif winner == 0:
        return 0.5
    elif winner == -player:
        if is_defense:
            return 0.7
        return -1
    else:
        return 0


def train_AI(episodes=100000):
    """Trains AI by playing many games"""
    for _ in range(episodes):
        board = np.zeros((BOARD_ROWS, BOARD_COLS))
        player = 1
        state = board_to_state(board)
        while True:
            action = choose_action(state)
            if action is None:
                break
            board[action[0], action[1]] = player
            next_state = board_to_state(board)
            winner = check_winner()
            reward = get_reward(winner, player)
            update_Q(state, action, reward, next_state)
            if winner is not None:
                break
            state = next_state
            player *= -1


def block_opponent(board, player):
    """Attempts to block the opponent's victory if possible"""
    opponent = -player
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if board[row, col] == 0:
                board[row, col] = opponent
                if check_winner() == opponent:
                    board[row, col] = player
                    return (row, col)
                board[row, col] = 0
    return None


def draw_lines():
    for row in range(1, BOARD_ROWS):
        pygame.draw.line(screen, LINE_COLOR, (0, row * SQUARE_SIZE), (WIDTH, row * SQUARE_SIZE), LINE_WIDTH)
    for col in range(1, BOARD_COLS):
        pygame.draw.line(screen, LINE_COLOR, (col * SQUARE_SIZE, 0), (col * SQUARE_SIZE, BOARD_ROWS * SQUARE_SIZE),
                         LINE_WIDTH)


def draw_figures():
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if board[row, col] == 1:  # X
                pygame.draw.line(screen, CROSS_COLOR,
                                 (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SPACE),
                                 (col * SQUARE_SIZE + SQUARE_SIZE - SPACE, row * SQUARE_SIZE + SQUARE_SIZE - SPACE),
                                 CROSS_WIDTH)
                pygame.draw.line(screen, CROSS_COLOR,
                                 (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SQUARE_SIZE - SPACE),
                                 (col * SQUARE_SIZE + SQUARE_SIZE - SPACE, row * SQUARE_SIZE + SPACE), CROSS_WIDTH)
            elif board[row, col] == -1:  # O
                pygame.draw.circle(screen, CIRCLE_COLOR,
                                   (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2),
                                   CIRCLE_RADIUS, CIRCLE_WIDTH)


def draw_restart_button():
    pygame.draw.rect(screen, BUTTON_COLOR, (50, 310, 200, 40))
    text = font.render('Restart', True, BUTTON_TEXT_COLOR)
    screen.blit(text, (100, 315))


def restart_game():
    global board, player, game_over
    board = np.zeros((BOARD_ROWS, BOARD_COLS))
    player = 1
    game_over = False
    screen.fill(BG_COLOR)
    draw_lines()
    draw_restart_button()


def check_winner():
    for row in range(BOARD_ROWS):
        if abs(np.sum(board[row, :])) == 3:
            return board[row, 0]
    for col in range(BOARD_COLS):
        if abs(np.sum(board[:, col])) == 3:
            return board[0, col]
    if abs(np.sum(np.diag(board))) == 3:
        return board[0, 0]
    if abs(np.sum(np.diag(np.fliplr(board)))) == 3:
        return board[0, BOARD_COLS - 1]
    if is_full():
        return 0
    return None


def is_full():
    return not (board == 0).any()


def mark_square(row, col, player):
    """We put a cross or a zero on the field"""
    board[row, col] = player

train_AI()
draw_lines()
draw_restart_button()

player = 1
game_over = False

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouseX = event.pos[0]
            mouseY = event.pos[1]
            if 50 <= mouseX <= 250 and 310 <= mouseY <= 350:
                restart_game()
            if not game_over and mouseY < SQUARE_SIZE * BOARD_ROWS:
                clicked_row = mouseY // SQUARE_SIZE
                clicked_col = mouseX // SQUARE_SIZE
                if board[clicked_row, clicked_col] == 0:
                    mark_square(clicked_row, clicked_col, player)
                    winner = check_winner()
                    if winner is not None:
                        game_over = True
                        if winner == 0:
                            print('Draw!')
                        else:
                            print(f"Won {'Cross' if winner == 1 else 'Zero'}!")
                    player *= -1

        if not game_over and player == -1:
            state = board_to_state(board)
            action = choose_action(state, epsilon=0.1)
            mark_square(action[0], action[1], player)
            winner = check_winner()
            if winner is not None:
                game_over = True
                if winner == -1:
                    print("AI won!")
                elif winner == 0:
                    print('Draw!')
            player *= -1

        screen.fill(BG_COLOR)
        draw_lines()
        draw_figures()
        draw_restart_button()
        pygame.display.update()
