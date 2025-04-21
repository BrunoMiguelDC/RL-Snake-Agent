#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 07:35:58 2021
Updated on Fri May 12 07:35:00 2023

"""

import matplotlib.pyplot as plt
from snake_game import SnakeGame
import numpy as np


def plot_board_image(board, text=None):
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(board)
    plt.axis('off')
    if text is not None:
        plt.gca().text(3, 3, text, fontsize=45, color='yellow')
    fig.canvas.draw()
    rgba_buf = fig.canvas.buffer_rgba()
    (w, h) = fig.canvas.get_width_height()
    rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((h, w, 4))
    return rgba_arr


def plot_board(file_name, board, text=None, ep=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(board)
    plt.axis('off')
    if text is not None:
        plt.gca().text(3, 3, text, fontsize=45, color='yellow')
    if ep is not None:
        plt.gca().text(0, 1, ep, fontsize=45, color='yellow')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


def snake_demo(actions):
    game = SnakeGame(30, 30, border=1)
    board, reward, done, info = game.reset()
    action_name = {-1: 'Turn left', 0: 'Straight ahead', 1: 'Turn right'}
    plot_board('0.png', board, 'Start')
    for frame, action in enumerate(actions):
        board, reward, done, info = game.step(action)
        plot_board(f'{frame + 1}.png', board, action_name[action])
