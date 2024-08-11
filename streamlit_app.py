import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
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

# User inputs
k = st.slider("Number of Nodes", min_value=2, max_value=20, value=6)
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
    st.write("State Pattern:")

    # Display the state pattern as a horizontal heatmap with reduced height
    fig, ax = plt.subplots(figsize=(10, 2))  # Adjust the figsize to make it shorter
    ax.imshow(states.T, aspect='auto', cmap='binary', interpolation='nearest')
    ax.set_ylabel("Node")
    ax.set_xlabel("Time Step")
    st.pyplot(fig)

    # Animation function
    def update(num, data, scat):
        scat.set_array(data[num])
        return scat,

    # Create a NetworkX graph for visualization
    G = nx.DiGraph()
    for i in range(k):
        G.add_node(i)
    for i, conns in enumerate(chart):
        for conn in conns:
            G.add_edge(i, conn)

    fig, ax = plt.subplots()
    pos = nx.spring_layout(G)
    scat = nx.draw_networkx_nodes(G, pos, node_color='orange', node_size=500, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax, font_color='black')

    ani = FuncAnimation(fig, update, frames=len(states), fargs=(states, scat), interval=200)

    # Save animation to display as GIF
    gif_path = "rbn_animation.gif"
    ani.save(gif_path, writer=PillowWriter(fps=5))
    
    st.write("Network Animation:")
    st.image(gif_path)
