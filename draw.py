import matplotlib.pyplot as plt

def plot_algorithm_times(minU, times_list, labels, colors):
    """
    Plots the execution times of different algorithms and displays values on each point.

    Parameters:
        minU (list): The list of minU values.
        times_list (list of lists): A list containing time arrays for each algorithm.
        labels (list): A list of labels for each algorithm.
        colors (list): A list of colors for each algorithm.
    """
    for times, label, color in zip(times_list, labels, colors):
        plt.plot(minU, times, marker='o', color=color, label=label)  # Plot each algorithm with its label and color

        # Display value on each data point
        for x, y in zip(minU, times):
            plt.text(x, y, f'{y}', ha='center', va='bottom')  # Display time value at each point

    # Adding labels and title
    plt.xlabel('minU')
    plt.ylabel('Time (s)')
    plt.title('Comparison of Execution Times for Algorithms')
    plt.legend()

    # Show plot
    plt.grid(True)
    plt.show()

# Example usage
minU = [1, 5, 10, 20, 60]
times1 = [3, 35, 80, 200, 600]
times2 = [0.3, 3, 8, 10, 40]
plot_algorithm_times(minU, [times1, times2], ['Algorithm 1', 'Algorithm 2'], ['blue', 'red'])
