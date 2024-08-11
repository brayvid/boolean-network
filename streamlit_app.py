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

    def go(self, n):
        if len(self.state) != len(self.chart):
            st.error("Invalid chart size.")
            return None
        if np.power(2, len(self.state) - 1) != len(self.rule):
            st.error("Invalid rule size.")
            return None
        states = [self.state.copy()]
        for t in np.arange(n):
            self.update()
            states.append(self.state.copy())
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
st.title("Random Boolean Network Simulation")

# User inputs
n = st.slider("Number of Iterations (n)", min_value=1, max_value=100, value=50)
k = st.slider("Number of Nodes (k)", min_value=2, max_value=20, value=6)
s = st.number_input("Random Seed (s)", value=0, min_value=0)

# Initialize Random Boolean Network
np.random.seed(s)
state = np.random.randint(2, size=k)
chart = [np.random.choice(np.setdiff1d(range(k), [i], assume_unique=True),
                          size=np.random.randint(1, k), replace=False) for i in range(k)]
rule = np.random.randint(2, size=int(np.power(2, k - 1)))

rbn = RandomBooleanNetwork(state, chart, rule)
states = rbn.go(n)

if states is not None:
    st.write("Generated States Over Time:")

    # Display the state pattern as a heatmap
    fig, ax = plt.subplots()
    ax.imshow(states, cmap='binary', interpolation='nearest')
    ax.set_xlabel("Node")
    ax.set_ylabel("Time Step")
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
    scat = nx.draw_networkx_nodes(G, pos, node_color=states[0], cmap='binary', ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax)

    ani = FuncAnimation(fig, update, frames=n, fargs=(states, scat), interval=200)

    # Save animation to display as GIF
    gif_path = "/mnt/data/rbn_animation.gif"
    ani.save(gif_path, writer=PillowWriter(fps=5))
    
    st.write("Network Evolution Animation:")
    st.image(gif_path)
