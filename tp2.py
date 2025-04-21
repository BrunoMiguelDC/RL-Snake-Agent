import os
import time

from snake_agent import EpsilonGreedySnakeAgent, UCBSnakeAgent
from snake_game import SnakeGame

from keras.optimizers import Adam
from keras.losses import MeanSquaredError
import numpy as np
import matplotlib.pyplot as plt
import imageio

from game_demo import plot_board

ACTIONS = [-1, 0, 1]
action_name = {-1: 'Turn left', 0: 'Straight ahead', 1: 'Turn right'}


def input_parser():
    print("-------- Snake Game using RL --------")

    print()
    print("Input board configuration")
    board_width = int(input("Board width: "))
    board_height = int(input("Board height: "))
    border_size = int(input("Border size: "))
    max_grass = float(input("Max grass: "))
    grass_growth = float(input("Grass growth: "))

    print()
    print("Input RL agent\'s hyperparameters")
    max_eps = int(input("Max number of episodes when training the RL agent: "))
    gamma = float(input("Gamma: "))
    exploration_strategy = int(
        input("Exploration Strategy\n1. Epsilon Greedy\n2. Upper Confidence Bound\nChoice(1/2): "))
    epsilon = 0
    epsilon_decay = 0
    epsilon_min = 0
    confidence = 0
    if exploration_strategy == 1:
        epsilon = float(input("Initial Epsilon: "))
        epsilon_decay = float(input("Decay epsilon by: "))
        epsilon_min = float(input("Minimum epsilon: "))
    else:
        confidence = float(input("Confidence: "))
    learning_rate = float(input("Learning rate: "))
    use_replay_buffer = input("Use replay buffer?(y/n) ").lower() == "y"
    replay_buff_size = 0
    batch_size = 0
    if use_replay_buffer:
        replay_buff_size = int(input("Replay buffer size: "))
        batch_size = int(input("Batch size to sample from replay buffer: "))
    use_target_network = input("Use target network?(y/n) ").lower() == "y"
    make_gif = input("Do you want to make gif of training?(y/n) ").lower() == "y"

    return board_width, board_height, border_size, max_grass, grass_growth, \
        max_eps, gamma, exploration_strategy, confidence, epsilon, epsilon_decay, epsilon_min, \
        learning_rate, use_replay_buffer, replay_buff_size, batch_size, use_target_network, make_gif


def initial_replay_buffer(game, max_frames=20):
    print("Filling replay buffer...")
    # (y,x)
    actions_dict = {0: [(0, -1), (-1, 0), (0, 1)],  # -1, 0, 1
                    1: [(-1, 0), (0, 1), (1, 0)],
                    2: [(0, 1), (1, 0), (0, -1)],
                    3: [(1, 0), (0, -1), (-1, 0)]
                    }

    def manhattan(x1, y1, x2, y2):
        return np.abs(x1 - x2) + np.abs(y1 - y2)

    def heuristic(apple, head, tail, direction):
        y_app, x_app = apple[0]
        y_head, x_head = head
        direction_list = actions_dict[direction]
        best = None
        best_action = -1
        for i in range(len(direction_list)):
            y, x = direction_list[i]
            aux_x = x_head + x
            aux_y = y_head + y
            if (aux_y, aux_x) in tail:
                continue
            distance = manhattan(aux_x, aux_y, x_app, y_app)
            if best is None or distance < best:
                best = distance
                best_action = i - 1
        return best_action

    replay_buffer = []
    frame = 0

    while frame < max_frames:
        board, reward, done, info = game.reset()
        while not done:
            frame += 1
            score, apple, head, tail, direction = game.get_state()

            action = heuristic(apple, head, tail, direction)

            next_board, reward, done, info = game.step(action)
            # plot_board(f'{frame}.png', next_board, action_name[action])

            replay_buffer.append((np.expand_dims(np.asarray(board).astype(np.float64), axis=0) / 255,
                                  action,
                                  reward,
                                  np.expand_dims(np.asarray(next_board).astype(np.float64), axis=0) / 255,
                                  done))
            board = next_board
            if frame == max_frames:
                break
    print(f"Replay buffer filled with {max_frames} examples")
    return replay_buffer


def train_snake_agent(env, agent, max_eps, batch_size, make_gif=False, exploration_strategy=1):
    print("Starting snake training...")
    filenames = []
    scores = []
    epsilons = []
    epsilons_per_ep = []
    steps = []
    time_per_ep = []
    images = []
    i = 0
    total_time = 0
    tot_steps = 0
    for episode in range(max_eps):
        print('#' * 55)
        print(f"Episode = {episode}")
        if exploration_strategy == 1:
            print(f"epsilon:{agent.epsilon():.2f}")
            epsilons_per_ep.append(agent.epsilon())
        board, reward, done, info = env.reset()
        n_steps = 0
        start_time = time.time()
        while not done:
            # choose action
            action = agent.act(board, tot_steps)
            tot_steps += 1

            if make_gif and episode == max_eps - 1:
                filename = f'{i}.png'
                filenames.append(filename)
                plot_board(f'{i}.png', board, f'{action_name[action]}', f'ep:{episode}')
                i += 1

            # make action in env
            next_board, reward, done, info = env.step(action)
            n_steps += 1
            # train deep q network
            agent.train(board, action, reward, next_board, done, batch_size)
            if exploration_strategy == 1:
                epsilons.append(agent.epsilon())
                agent.update_epsilon()

            board = next_board
            if episode % 150 == 0:  # every 10 episodes update target
                agent.align_target_model()

            if done:
                break
        end_time = time.time()
        elapsed_time = (end_time - start_time)
        time_per_ep.append(round(elapsed_time))
        total_time += elapsed_time
        steps.append(n_steps)
        score = info['score']
        scores.append(score)
        print(f"Num steps = {n_steps}")
        print(f"Time = {elapsed_time // 60}min {round(elapsed_time)}sec")
        print(f"Score = {score}")
        print('#' * 55)

    if make_gif:
        for filename in filenames:
            image = imageio.v2.imread(filename)
            images.append(image)

        imageio.mimsave('./train.gif', images, fps=5)

        for filename in set(filenames):
            os.remove(filename)
    print(f"Finished training after {total_time // 60}min")
    return scores, epsilons, epsilons_per_ep, steps, time_per_ep


def plot_stats(max_eps, scores, epsilons, epsilons_per_ep, n_steps, time_per_ep, subplots=False,
               exploration_strategy=1):
    if subplots:
        pass
    else:
        plt.plot(range(max_eps), scores)
        # plt.xticks(ticks=range(max_eps), labels=list(range(max_eps)))
        plt.xlabel("episodes")
        plt.ylabel("scores")
        plt.show()

        plt.plot(range(max_eps), n_steps)
        plt.xlabel("episodes")
        plt.ylabel("num_steps")
        plt.show()

        plt.plot(range(max_eps), time_per_ep)
        plt.xlabel("episodes")
        plt.ylabel("seconds")
        plt.show()

        if exploration_strategy == 1:
            plt.plot(epsilons)
            plt.xlabel("total steps")
            plt.ylabel("epsilon")
            plt.show()

            plt.plot(range(max_eps), epsilons_per_ep)
            plt.xlabel("episodes")
            plt.ylabel("epsilon")
            plt.show()


def main():
    print("-------- START --------")
    use_default = input("Use default hyperparameters?(y/n) ") == 'y'
    if use_default:
        # Default hyperparameters
        board_width = 14
        board_height = 14
        border_size = 1
        max_grass = 0.05
        grass_growth = 0.001
        max_eps = 500
        gamma = 0.9
        exploration_strategy = 1
        epsilon = 1.0 if exploration_strategy == 1 else 0
        epsilon_decay = 0.0002 if exploration_strategy == 1 else 0  # reaches epsilon_min in 20 eps
        epsilon_min = 0.1 if exploration_strategy == 1 else 0
        confidence = 0.95 if exploration_strategy == 2 else 0
        learning_rate = 0.001
        use_replay_buffer = True
        use_target_dqn = True
        replay_buff_size = 50000
        batch_size = 1024 if use_replay_buffer else 0
        make_gif = False
    else:
        board_width, board_height, border_size, max_grass, grass_growth, max_eps, gamma, \
            exploration_strategy, confidence, epsilon, epsilon_decay, epsilon_min, learning_rate, \
            use_replay_buffer, replay_buff_size, batch_size, use_target_dqn, make_gif = input_parser()

    loss = MeanSquaredError()
    optimizer = Adam(learning_rate=learning_rate)

    env = SnakeGame(board_width, board_height, border=border_size, max_grass=max_grass, grass_growth=grass_growth)

    board, reward, done, info = env.reset()

    if use_replay_buffer:
        if exploration_strategy == 1:
            agent = EpsilonGreedySnakeAgent(ACTIONS, board.shape, loss, optimizer, gamma, epsilon,
                                            epsilon_decay, epsilon_min, use_replay=use_replay_buffer,
                                            initial_replay_buff=initial_replay_buffer(env, max_frames=replay_buff_size),
                                            replay_buff_size=replay_buff_size,
                                            use_target_dqn=use_target_dqn)
        else:
            agent = UCBSnakeAgent(ACTIONS, board.shape, loss, optimizer, gamma, confidence,
                                  use_replay=use_replay_buffer,
                                  initial_replay_buff=initial_replay_buffer(env, max_frames=replay_buff_size),
                                  replay_buff_size=replay_buff_size,
                                  use_target_dqn=use_target_dqn)

    else:
        if exploration_strategy == 1:
            agent = EpsilonGreedySnakeAgent(ACTIONS, board.shape, loss, optimizer, gamma, epsilon, epsilon_decay,
                                            epsilon_min, use_replay=use_replay_buffer, use_target_dqn=use_target_dqn)
        else:
            agent = UCBSnakeAgent(ACTIONS, board.shape, loss, optimizer, gamma, confidence,
                                  use_replay=use_replay_buffer, use_target_dqn=use_target_dqn)

    scores, epsilons, epsilons_per_ep, n_steps, time_per_ep = train_snake_agent(env, agent, max_eps, batch_size,
                                                                                make_gif, exploration_strategy)
    plot_stats(max_eps, scores, epsilons, epsilons_per_ep, n_steps, time_per_ep)
    print("-------- END --------")


if __name__ == '__main__':
    main()
