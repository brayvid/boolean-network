import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from matplotlib.animation import FuncAnimation, PillowWriter
from flask import Flask, render_template, request, url_for, send_from_directory
import base64
from io import BytesIO
import time # For cache busting GIF

app = Flask(__name__)
STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
os.makedirs(STATIC_FOLDER, exist_ok=True) # Ensure static folder exists
app.config['STATIC_FOLDER'] = STATIC_FOLDER


# RandomBooleanNetwork class definition (remains the same as before)
class RandomBooleanNetwork:
    def __init__(self, state, chart, rule):
        self.state = state
        self.chart = chart
        self.rule = rule
        self.nodes = len(self.state)
        self.error_message = None 

    def go(self):
        if len(self.state) != len(self.chart):
            self.error_message = "Invalid chart size."
            return None
        
        expected_rule_len = int(np.power(2, self.nodes - 1))
        if expected_rule_len != len(self.rule) :
            # Handle k=1 case where nodes-1 = 0, 2^0 = 1. rule length should be 1.
            if self.nodes == 1 and expected_rule_len == 1 and len(self.rule) == 1:
                pass # This is fine for k=1, rule depends on 0 inputs (constant)
            else:
                self.error_message = f"Invalid rule size. Expected {expected_rule_len}, got {len(self.rule)} for {self.nodes} nodes."
                return None

        states_list = [self.state.copy()]
        seen_states = {tuple(self.state)}
        max_iterations = 100
        for _ in np.arange(max_iterations):
            self.update()
            if self.error_message:
                return None
            current_state_tuple = tuple(self.state)
            if current_state_tuple in seen_states:
                break
            seen_states.add(current_state_tuple)
            states_list.append(self.state.copy())
        return np.array(states_list)

    def update(self):
        nextState = np.zeros(self.nodes)
        if self.nodes == 0: # Should not happen with k_min=2
            self.error_message = "Cannot update with zero nodes."
            return

        k_rule = self.nodes - 1 # Number of inputs expected by the rule logic
        if k_rule < 0: k_rule = 0 # For single node case, rule is based on 0 inputs

        for node_idx in np.arange(self.nodes):
            connected_to_node = self.chart[node_idx]
            num_actual_inputs = len(connected_to_node)
            
            rule_input_vector = np.zeros(k_rule if k_rule > 0 else 0) # Handle k_rule=0 for single node

            if k_rule > 0: # Only populate if rule expects inputs
                for i in range(k_rule):
                    if i < num_actual_inputs:
                        rule_input_vector[i] = self.state[connected_to_node[i]]
                    else:
                        rule_input_vector[i] = 0 
            
            rule_index_str = "".join(str(int(x)) for x in rule_input_vector) if k_rule > 0 else "" # Empty string for 0 inputs
            
            try:
                # If k_rule is 0 (single node network), rule_index is 0 (accesses the single rule value)
                rule_index = int(rule_index_str, 2) if rule_index_str else 0
                
                if rule_index >= len(self.rule):
                    self.error_message = (f"Rule index {rule_index} out of bounds for rule of len {len(self.rule)}. "
                                          f"Input vector: {rule_input_vector} from connected {connected_to_node} (node {node_idx})")
                    return
                nextState[node_idx] = self.rule[rule_index]
            except ValueError:
                self.error_message = f"Could not convert rule input '{rule_index_str}' to int."
                return

        self.state = nextState

# Helper function to generate RBN outputs
def generate_rbn_visuals(k_val, s_val, static_dir_path):
    """Generates RBN states, heatmap, and GIF."""
    output = {
        "heatmap_data": None, "gif_filename": None, 
        "timestamp": None, "error_message": None
    }

    np.random.seed(s_val)
    state = np.random.randint(2, size=k_val)
    
    chart = []
    for i in range(k_val):
        possible_influencers = np.setdiff1d(np.arange(k_val), [i])
        max_possible_connections = len(possible_influencers)
        
        if k_val == 1: # Special case for a single node (no influencers)
            num_connections = 0
        elif max_possible_connections == 0 : # Should not happen if k_val > 1
             num_connections = 0
        else: # k_val > 1 and has possible influencers
            # Number of incoming connections to node i. Max is k_val-1. Min is 1.
            num_connections = np.random.randint(1, max_possible_connections + 1)
        
        if num_connections > 0:
            selected_influencers = np.random.choice(
                possible_influencers, size=num_connections, replace=False
            )
        else:
            selected_influencers = np.array([])
        chart.append(selected_influencers)

    # Rule size: For k_val nodes, rule lookup uses k_val-1 inputs (or 0 for k_val=1).
    k_for_rule_length = k_val - 1
    if k_for_rule_length < 0: k_for_rule_length = 0 # For k_val=1, k_for_rule_length=0, 2^0=1 rule entry
    
    rule_len = int(np.power(2, k_for_rule_length))
    if rule_len <= 0 and k_val > 0 : # Should theoretically only be non-positive if k_for_rule_length is problematic
         output["error_message"] = f"Calculated rule length is {rule_len}, which is invalid for k={k_val}."
         return output
    rule = np.random.randint(2, size=rule_len)

    rbn = RandomBooleanNetwork(state.copy(), chart, rule)
    states_history = rbn.go()

    if rbn.error_message:
        output["error_message"] = rbn.error_message
        return output
    if states_history is None or len(states_history) == 0:
        output["error_message"] = "Network simulation failed to produce states."
        return output

    # --- Generate Heatmap ---
    try:
        height = min(1 + k_val * 0.3, 6)
        width_in_inches = 8
        cmap = mcolors.ListedColormap(['purple', 'yellow'])

        fig_heatmap, ax_heatmap = plt.subplots(figsize=(width_in_inches, height))
        ax_heatmap.imshow(states_history.T, aspect='auto', cmap=cmap, interpolation='nearest')
        ax_heatmap.set_ylabel("Node")
        ax_heatmap.set_xlabel("Time Step")
        ax_heatmap.set_yticks(np.arange(states_history.shape[1]))
        ax_heatmap.set_yticklabels(np.arange(1, states_history.shape[1] + 1))
        
        img_buffer = BytesIO()
        fig_heatmap.savefig(img_buffer, format="png", bbox_inches='tight')
        plt.close(fig_heatmap)
        img_buffer.seek(0)
        output["heatmap_data"] = base64.b64encode(img_buffer.read()).decode('utf-8')
    except Exception as e:
        output["error_message"] = f"Error generating heatmap: {e}"
        # Continue to try generating GIF if heatmap fails

    # --- Generate Animation GIF ---
    try:
        G = nx.DiGraph()
        for i in range(k_val): G.add_node(i) 
        
        for i_node_influenced, influencers in enumerate(chart):
            for j_influencer_node in influencers:
                G.add_edge(j_influencer_node, i_node_influenced)
        
        labels = {i: str(i + 1) for i in range(k_val)}
        anim_fig_size = 8 

        fig_anim, ax_anim = plt.subplots(figsize=(anim_fig_size, anim_fig_size))
        pos = nx.shell_layout(G)
        
        initial_node_colors = ['yellow' if s else 'purple' for s in states_history[0]]
        
        scat = nx.draw_networkx_nodes(G, pos, node_color=initial_node_colors,
                                      node_size=500, ax=ax_anim)
        nx.draw_networkx_edges(G, pos, ax=ax_anim, arrowstyle='->', arrowsize=10)
        nx.draw_networkx_labels(G, pos, ax=ax_anim, labels=labels, font_color='black')

        def update_anim_func(num, data_hist, scat_nodes_ref):
            node_colors = ['yellow' if node_s else 'purple' for node_s in data_hist[num]]
            scat_nodes_ref.set_color(node_colors)
            return scat_nodes_ref,

        ani = FuncAnimation(fig_anim, update_anim_func, frames=len(states_history), 
                            fargs=(states_history, scat), interval=1000, blit=True)
        
        gif_path = os.path.join(static_dir_path, "rbn_animation.gif")
        ani.save(gif_path, writer=PillowWriter(fps=1))
        plt.close(fig_anim)
        output["gif_filename"] = "rbn_animation.gif"
        output["timestamp"] = int(time.time())
    except Exception as e:
        current_error = output["error_message"] or ""
        if current_error: current_error += " "
        output["error_message"] = f"{current_error}Error generating animation: {e}"
    
    return output


@app.route('/', methods=['GET', 'POST'])
def index():
    default_k = 6
    default_s = 42

    # Initialize with defaults, these might be overridden
    k_to_use = default_k
    s_to_use = default_s
    
    # Variables to pass to template
    view_params = {
        "k_value": k_to_use,
        "s_value": s_to_use,
        "heatmap_data": None,
        "gif_filename": None,
        "timestamp": None,
        "error_message": None
    }

    if request.method == 'POST':
        try:
            k_from_form = int(request.form.get('k', default_k))
            s_from_form = int(request.form.get('s', default_s))

            if not (2 <= k_from_form <= 10):
                view_params["error_message"] = "Number of Nodes (K) must be between 2 and 10."
                # Keep form values if valid, otherwise defaults for rendering
                view_params["k_value"] = k_from_form 
                view_params["s_value"] = s_from_form
            else:
                k_to_use = k_from_form
                s_to_use = s_from_form
                view_params["k_value"] = k_to_use # Update view_params with validated form values
                view_params["s_value"] = s_to_use

        except ValueError:
            view_params["error_message"] = "Invalid input for K or S. Please enter numbers."
            # k_to_use, s_to_use remain defaults
            
    # Generate visuals if no critical form error, or if it's a GET request
    if not view_params["error_message"] or request.method == 'GET':
        # For GET, k_to_use and s_to_use are already defaults
        # For POST, they are validated form values (or defaults if form was invalid but not range error)
        rbn_results = generate_rbn_visuals(k_to_use, s_to_use, app.config['STATIC_FOLDER'])
        
        view_params["heatmap_data"] = rbn_results["heatmap_data"]
        view_params["gif_filename"] = rbn_results["gif_filename"]
        view_params["timestamp"] = rbn_results["timestamp"]
        # If rbn_results has an error, it will override any previous error_message
        if rbn_results["error_message"]:
            view_params["error_message"] = rbn_results["error_message"]
            
    return render_template('index.html', **view_params)


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                           'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)