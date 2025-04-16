import numpy as np
import matplotlib.pyplot as plt
from behavior import Behavior
import os

def visualize_behavior(path, save_dir=None):
    """
    Create comprehensive behavior visualizations for a given dataset.
    
    Parameters:
    -----------
    path : str
        Path to the folder containing behavior data
    save_dir : str, optional
        Directory to save the plots. If None, plots will be displayed but not saved.
    """
    # Load behavior data
    behavior = Behavior(path)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Performance over sessions
    ax1 = plt.subplot(2, 2, 1)
    behavior.plot_performance_over_sessions()
    ax1.set_title('Performance Over Sessions')
    
    # 2. Left/Right performance
    ax2 = plt.subplot(2, 2, 2)
    behavior.plot_LR_performance_over_sessions()
    ax2.set_title('Left/Right Performance')
    
    # 3. Early lick rate
    ax3 = plt.subplot(2, 2, 3)
    behavior.plot_early_lick()
    ax3.set_title('Early Lick Rate')
    
    # 4. Learning progression
    ax4 = plt.subplot(2, 2, 4)
    behavior.learning_progression(window=50)
    ax4.set_title('Learning Progression')
    
    plt.tight_layout()
    
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, 'behavior_summary.png'))
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    # Example usage
    path = input("Enter the path to your behavior data folder: ")
    save_dir = input("Enter directory to save plots (press Enter to display only): ")
    if save_dir.strip() == "":
        save_dir = None
    visualize_behavior(path, save_dir) 