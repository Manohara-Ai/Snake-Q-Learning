import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores1, mean_scores1, scores2, mean_scores2):
    """
    Plot scores and mean scores for two agents.

    :param scores1: List of scores for Agent 1.
    :param mean_scores1: List of mean scores for Agent 1.
    :param scores2: List of scores for Agent 2.
    :param mean_scores2: List of mean scores for Agent 2.
    """
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    
    # Plot Agent 1 scores
    plt.plot(scores1, label="Agent 1 Scores", color="blue")
    plt.plot(mean_scores1, label="Agent 1 Mean Score", linestyle="--", color="cyan")
    
    # Plot Agent 2 scores
    plt.plot(scores2, label="Agent 2 Scores", color="green")
    plt.plot(mean_scores2, label="Agent 2 Mean Score", linestyle="--", color="lime")

    plt.ylim(ymin=0)
    plt.legend(loc="upper left")
    
    # Annotate final scores
    if scores1:
        plt.text(len(scores1) - 1, scores1[-1], str(scores1[-1]), color="blue")
        plt.text(len(mean_scores1) - 1, mean_scores1[-1], str(mean_scores1[-1]), color="cyan")
    if scores2:
        plt.text(len(scores2) - 1, scores2[-1], str(scores2[-1]), color="green")
        plt.text(len(mean_scores2) - 1, mean_scores2[-1], str(mean_scores2[-1]), color="lime")
    
    plt.show(block=False)
    plt.pause(0.1)
