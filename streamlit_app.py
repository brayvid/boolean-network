import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from matplotlib.animation import FuncAnimation, PillowWriter

# RandomBooleanNetwork class definition
class RandomBooleanNetwork:
    def __init__(self, state, chart, rule):
        self.state = state
        self.chart = chart
        self.rule = rule
        self.nodes = len(self.state)

    def go(self):
        if len(self.state) != len(self.chart):
            st.error("Invalid chart size.")
            return None
        if np.power(2, len(self.state) - 1) != len(self.rule):
            st.error("Invalid rule size.")
            return None
        states = [self.state.copy()]
        max_iterations = 100  # Limit the maximum number of iterations to avoid infinite loops
        for _ in np.arange(max_iterations):
            self.update()
            states.append(self.state.copy())
            # Check if steady state is reached (no change in the last two states)
            if np.array_equal(states[-1], states[-2]):
                break
        return np.array(states)

    def update(self):
        nextState = np.zeros(self.nodes)
        for node in np.arange(self.nodes):
            connected = self.chart[node]
            substate = np.zeros(self.nodes - 1)
            for i in np.arange(len(connected)):
                substate[i] = self.state[connected[i]]
            for j in np.arange(len(connected), self.nodes - 1):
                substate[j] = 0
            nextState[node] = self.rule[int("".join(str(int(x)) for x in substate), 2)]
        self.state = nextState

# Streamlit App
st.title("Random Boolean Network")

# Display author information
st.markdown("""Blake Rayvid - [https://github.com/brayvid/boolean-network](https://github.com/brayvid/boolean-network)""")

# User inputs
k = st.slider("Number of Nodes", min_value=2, max_value=10, value=6)
s = st.number_input("Random Seed", value=42, min_value=0)

# Initialize Random Boolean Network
np.random.seed(s)
state = np.random.randint(2, size=k)
chart = [np.random.choice(np.setdiff1d(range(k), [i], assume_unique=True),
                          size=np.random.randint(1, k), replace=False) for i in range(k)]
rule = np.random.randint(2, size=int(np.power(2, k - 1)))

rbn = RandomBooleanNetwork(state, chart, rule)
states = rbn.go()

if states is not None:
    st.write("State Pattern")

    # Dynamically adjust height based on the number of nodes, with a cap on height
    height = min(1 + k * 0.2, 6)  # Adjust the height to be proportional but not too tall
    
    # Fixed width for consistent layout
    width_in_inches = 8  # Set a fixed width in inches for the figure

    # Custom colormap for yellow and purple
    cmap = mcolors.ListedColormap(['purple', 'yellow'])

    # Display the state pattern as a horizontal heatmap with dynamic height
    fig, ax = plt.subplots(figsize=(width_in_inches, height))  # Adjust the height dynamically
    ax.imshow(states.T, aspect='auto', cmap=cmap, interpolation='nearest')  # Use custom colormap
    ax.set_ylabel("Node")
    ax.set_xlabel("Time Step")
    ax.set_yticks(np.arange(states.shape[1]))
    ax.set_yticklabels(np.arange(1, states.shape[1] + 1))  # Start labels at 1
    st.pyplot(fig)

    # Animation function
    def update(num, data, scat):
        colors = ['yellow' if state else 'purple' for state in data[num]]
        scat.set_color(colors)
        return scat,

    # Create a NetworkX graph for visualization
    G = nx.DiGraph()
    for i in range(k):
        G.add_node(i)
    for i, conns in enumerate(chart):
        for conn in conns:
            G.add_edge(i, conn)

    # Adjust node labels to start at 1 instead of 0
    labels = {i: str(i + 1) for i in range(k)}

    fig, ax = plt.subplots(figsize=(width_in_inches, width_in_inches))  # Keep the animation square with same width as state pattern
    pos = nx.shell_layout(G)  # Use shell layout for the graph
    scat = nx.draw_networkx_nodes(G, pos, node_color=['yellow' if state else 'purple' for state in states[0]], 
                                  node_size=500, ax=ax)  # Removed cmap parameter
    nx.draw_networkx_edges(G, pos, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_color='black')  # Use custom labels

    ani = FuncAnimation(fig, update, frames=len(states), fargs=(states, scat), interval=1000)  # 1 frame per second

    # Save animation to display as GIF
    gif_path = "rbn_animation.gif"
    ani.save(gif_path, writer=PillowWriter(fps=1))  # Ensure it saves at 1 FPS
    
    st.write("Network Evolution")
    st.image(gif_path)
