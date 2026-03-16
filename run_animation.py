"""
Standalone animation script - no Jupyter required!
Run with: python run_animation.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

print("Loading data...")
tracking = pd.read_csv('data/raw/week1.csv')
plays = pd.read_csv('data/raw/plays.csv')

selected_game_id = 2021090900
selected_play_id = 1267

print("Filtering to selected play...")
play_tracking = tracking[
    (tracking['gameId'] == selected_game_id) & 
    (tracking['playId'] == selected_play_id)
].copy()

print(f"Found {len(play_tracking)} tracking rows")

# Normalize play direction
def normalize_play_direction(df):
    df_normalized = df.copy()
    going_left = df_normalized['playDirection'] == 'left'
    df_normalized.loc[going_left, 'x'] = 120 - df_normalized.loc[going_left, 'x']
    df_normalized.loc[going_left, 'y'] = 53.3 - df_normalized.loc[going_left, 'y']
    df_normalized.loc[going_left, 'o'] = (df_normalized.loc[going_left, 'o'] + 180) % 360
    df_normalized.loc[going_left, 'dir'] = (df_normalized.loc[going_left, 'dir'] + 180) % 360
    df_normalized.loc[going_left, 'playDirection'] = 'right'
    return df_normalized

play_tracking = normalize_play_direction(play_tracking)
print("Normalized play direction to left-to-right")

# Get frames
frames = sorted(play_tracking['frameId'].unique())
print(f"Total frames: {len(frames)}")

# Initialize figure
fig, ax = plt.subplots(figsize=(15, 8))

offense_scatter = ax.scatter([], [], c='red', s=200, marker='o', 
                            edgecolors='darkred', linewidth=2,
                            label='Offense (TB)', zorder=3)
defense_scatter = ax.scatter([], [], c='blue', s=200, marker='s',
                            edgecolors='darkblue', linewidth=2,
                            label='Defense (DAL)', zorder=3)
ball_scatter = ax.scatter([], [], c='brown', s=150, marker='D',
                         edgecolors='black', linewidth=2,
                         label='Football', zorder=4)

ax.set_xlim(0, 120)
ax.set_ylim(0, 53.3)
ax.set_xlabel('X Coordinate (yards) — Field Length', fontsize=12, weight='bold')
ax.set_ylabel('Y Coordinate (yards) — Field Width', fontsize=12, weight='bold')
ax.set_aspect('equal', adjustable='box')
ax.grid(True, alpha=0.3, linestyle='--')

ax.axhline(y=0, color='green', linewidth=3, alpha=0.3)
ax.axhline(y=53.3, color='green', linewidth=3, alpha=0.3)
ax.axvline(x=10, color='orange', linewidth=2, alpha=0.3, linestyle='--')
ax.axvline(x=110, color='orange', linewidth=2, alpha=0.3, linestyle='--')

ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

title = ax.text(0.5, 1.05, '', transform=ax.transAxes,
               ha='center', fontsize=14, weight='bold')

def init():
    offense_scatter.set_offsets(np.empty((0, 2)))
    defense_scatter.set_offsets(np.empty((0, 2)))
    ball_scatter.set_offsets(np.empty((0, 2)))
    title.set_text('')
    return offense_scatter, defense_scatter, ball_scatter, title

def update(frame_idx):
    current_frame = frames[frame_idx]
    frame_data = play_tracking[play_tracking['frameId'] == current_frame]
    
    offense = frame_data[frame_data['team'] == 'TB']
    defense = frame_data[frame_data['team'] == 'DAL']
    ball = frame_data[frame_data['team'] == 'football']
    
    offense_positions = offense[['x', 'y']].values
    defense_positions = defense[['x', 'y']].values
    ball_position = ball[['x', 'y']].values
    
    offense_scatter.set_offsets(offense_positions)
    defense_scatter.set_offsets(defense_positions)
    ball_scatter.set_offsets(ball_position)
    
    title.set_text(f'Play Animation — Frame {current_frame} of {frames[-1]}')
    
    return offense_scatter, defense_scatter, ball_scatter, title

print("\nCreating animation...")
anim = FuncAnimation(fig, update, frames=len(frames),
                    init_func=init, interval=100, 
                    blit=True, repeat=True)

print("Animation ready! Displaying window...")
print("\nControls: Close the window to exit")
plt.show()
