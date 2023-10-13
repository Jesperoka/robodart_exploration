import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers

def test_utils_import():
    print("\n\nimport was successful")


def save_video(frames, filename=None, fps=30, show=False):
    """
    Convert a list of frames (numpy arrays) into a video using matplotlib.

    Parameters:
    - frames: List of numpy arrays representing images.
    - filename: Name of the file where the video will be saved. If None, video won't be saved.
    - fps: Frames per second for the output video.
    - show: If True, display the video. 

    Returns:
    - None
    """
    
    fig, ax = plt.subplots()
    im = ax.imshow(frames[0])
    plt.axis('off')  # Turn off axis numbers and ticks

    def update(frame):
        im.set_array(frame)
        return [im]

    ani = FuncAnimation(fig, update, frames=frames, blit=True, repeat=False)

    if filename:
        # Save the video to a file. You may need to have ffmpeg installed.
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(filename, writer=writer)

    if show:
        plt.show()

    plt.close(fig)

# Example usage:
# frames = [np.random.random((100, 100, 3)) for _ in range(100)]  # Example frames (random noise)
# frames_to_video(frames, filename="output.mp4", fps=30, show=True)

