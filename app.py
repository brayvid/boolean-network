import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from matplotlib.animation import FuncAnimation, FFMpegWriter
from flask import Flask, render_template, request, url_for, send_from_directory
import time

# --- Optimization Imports ---
from flask_caching import Cache
from flask_compress import Compress
from flask_assets import Environment, Bundle

# --- App Initialization ---
app = Flask(__name__, instance_relative_config=True) # Enable instance folder
Compress(app) # Enable Gzip compression
os.makedirs(app.instance_path, exist_ok=True) # Ensure instance folder exists

# --- Caching Configuration (Production-Ready) ---
# Switched to FileSystemCache and specified a dedicated, git-ignored directory.
cache_config = {
    "CACHE_TYPE": "FileSystemCache",
    "CACHE_DIR": os.path.join(app.instance_path, "cache"), # Store cache in instance/cache/
    "CACHE_DEFAULT_TIMEOUT": 3600 # Cache items for 1 hour
}
app.config.from_mapping(cache_config)
cache = Cache(app)

# --- Asset Management for CSS/JS Minification ---
assets = Environment(app)
css = Bundle('css/main.css', filters='cssmin', output='gen/packed.css')
assets.register('all_css', css)

# --- Static Folder & Generated Media Setup ---
STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
app.config['STATIC_FOLDER'] = STATIC_FOLDER

# --- NEW: Define a dedicated, git-ignored folder for generated media ---
GENERATED_MEDIA_FOLDER = os.path.join(STATIC_FOLDER, 'generated_media')
os.makedirs(GENERATED_MEDIA_FOLDER, exist_ok=True) # Ensure this folder exists


# The RandomBooleanNetwork class remains exactly the same.
# ... (paste your RandomBooleanNetwork class here) ...
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
            if self.nodes == 1 and expected_rule_len == 1 and len(self.rule) == 1:
                pass 
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
        if self.nodes == 0:
            self.error_message = "Cannot update with zero nodes."
            return

        k_rule = self.nodes - 1
        if k_rule < 0: k_rule = 0

        for node_idx in np.arange(self.nodes):
            connected_to_node = self.chart[node_idx]
            num_actual_inputs = len(connected_to_node)
            
            rule_input_vector = np.zeros(k_rule if k_rule > 0 else 0)

            if k_rule > 0:
                for i in range(k_rule):
                    if i < num_actual_inputs:
                        rule_input_vector[i] = self.state[connected_to_node[i]]
                    else:
                        rule_input_vector[i] = 0 
            
            rule_index_str = "".join(str(int(x)) for x in rule_input_vector) if k_rule > 0 else ""
            
            try:
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


# --- Modified generation function to save to a subdirectory ---
@cache.memoize()
def generate_rbn_visuals(k_val, s_val):
    """
    Generates RBN states and saves visuals to a dedicated subdirectory.
    """
    print(f"CACHE MISS: Generating visuals for k={k_val}, s={s_val}")

    file_identifier = f"k{k_val}_s{s_val}"
    
    # --- CHANGE: Prepend the subdirectory to the filenames ---
    # The URL will be static/generated_media/heatmap_...
    # The file path will be /path/to/project/static/generated_media/heatmap_...
    output = {
        "heatmap_filename": os.path.join('generated_media', f"heatmap_{file_identifier}.png"),
        "animation_filename": os.path.join('generated_media', f"rbn_animation_{file_identifier}.mp4"),
        "error_message": None
    }

    # The RBN logic is unchanged...
    np.random.seed(s_val)
    # ... (rest of RBN logic from your original file) ...
    state = np.random.randint(2, size=k_val)
    chart = []
    for i in range(k_val):
        possible_influencers = np.setdiff1d(np.arange(k_val), [i])
        max_possible_connections = len(possible_influencers)
        if k_val == 1: num_connections = 0
        elif max_possible_connections == 0 : num_connections = 0
        else: num_connections = np.random.randint(1, max_possible_connections + 1)
        
        if num_connections > 0:
            selected_influencers = np.random.choice(possible_influencers, size=num_connections, replace=False)
        else:
            selected_influencers = np.array([])
        chart.append(selected_influencers)

    k_for_rule_length = k_val - 1
    if k_for_rule_length < 0: k_for_rule_length = 0
    rule_len = int(np.power(2, k_for_rule_length))
    if rule_len <= 0 and k_val > 0:
         output["error_message"] = f"Calculated rule length is {rule_len}, which is invalid for k={k_val}."
         return output
    rule = np.random.randint(2, size=rule_len)

    rbn = RandomBooleanNetwork(state.copy(), chart, rule)
    states_history = rbn.go()

    if rbn.error_message or states_history is None or len(states_history) == 0:
        output["error_message"] = rbn.error_message or "Network simulation failed."
        return output

    # --- Generate and SAVE Heatmap to the subdirectory ---
    try:
        # --- CHANGE: Use the new folder for the full file path ---
        heatmap_path = os.path.join(STATIC_FOLDER, output["heatmap_filename"])
        
        if not os.path.exists(heatmap_path):
            height = min(1 + k_val * 0.3, 6)
            width_in_inches = 8
            cmap = mcolors.ListedColormap(['#4B0082', '#FFD700'])
            fig_heatmap, ax_heatmap = plt.subplots(figsize=(width_in_inches, height))
            ax_heatmap.imshow(states_history.T, aspect='auto', cmap=cmap, interpolation='nearest')
            ax_heatmap.set_ylabel("Node")
            ax_heatmap.set_xlabel("Time Step")
            ax_heatmap.set_yticks(np.arange(states_history.shape[1]))
            ax_heatmap.set_yticklabels(np.arange(1, states_history.shape[1] + 1))
            fig_heatmap.savefig(heatmap_path, format="png", bbox_inches='tight')
            plt.close(fig_heatmap)
    except Exception as e:
        output["error_message"] = f"Error generating heatmap: {e}"

    # --- Generate and SAVE Animation MP4 to the subdirectory ---
    try:
        # --- CHANGE: Use the new folder for the full file path ---
        animation_path = os.path.join(STATIC_FOLDER, output["animation_filename"])

        if not os.path.exists(animation_path):
            G = nx.DiGraph()
            for i in range(k_val): G.add_node(i) 
            for i_node, influencers in enumerate(chart):
                for j_influencer in influencers: G.add_edge(j_influencer, i_node)
            
            labels = {i: str(i + 1) for i in range(k_val)}
            fig_anim, ax_anim = plt.subplots(figsize=(8, 8))
            pos = nx.shell_layout(G)
            node_colors = ['#FFD700' if s else '#4B0082' for s in states_history[0]]
            scat = nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, ax=ax_anim)
            nx.draw_networkx_edges(G, pos, ax=ax_anim, arrowstyle='->', arrowsize=10)
            nx.draw_networkx_labels(G, pos, ax=ax_anim, labels=labels, font_color='white')

            def update_anim_func(num, data, scat_nodes):
                colors = ['#FFD700' if s else '#4B0082' for s in data[num]]
                scat_nodes.set_color(colors)
                return scat_nodes,

            ani = FuncAnimation(fig_anim, update_anim_func, frames=len(states_history), fargs=(states_history, scat), interval=500, blit=True)
            ani.save(animation_path, writer=FFMpegWriter(fps=2, bitrate=1800))
            plt.close(fig_anim)
    except Exception as e:
        current_error = output.get("error_message", "") or ""
        if current_error: current_error += " "
        output["error_message"] = f"{current_error}Error generating animation: {e}"
    
    return output

# --- The index view remains unchanged. url_for() handles subdirectories automatically. ---
@app.route('/', methods=['GET', 'POST'])
def index():
    # ... (view logic is exactly the same as before) ...
    default_k = 6
    default_s = 42
    k_to_use = default_k
    s_to_use = default_s
    
    view_params = {
        "k_value": k_to_use, "s_value": s_to_use,
        "heatmap_filename": None, "animation_filename": None,
        "error_message": None
    }

    if request.method == 'POST':
        try:
            k_from_form = int(request.form.get('k', default_k))
            s_from_form = int(request.form.get('s', default_s))

            if not (2 <= k_from_form <= 10):
                view_params["error_message"] = "Number of Nodes (K) must be between 2 and 10."
                view_params["k_value"] = k_from_form 
                view_params["s_value"] = s_from_form
            else:
                k_to_use = k_from_form
                s_to_use = s_from_form
                view_params["k_value"] = k_to_use
                view_params["s_value"] = s_to_use
        except ValueError:
            view_params["error_message"] = "Invalid input for K or S. Please enter numbers."
            
    if not view_params["error_message"]:
        rbn_results = generate_rbn_visuals(k_to_use, s_to_use)
        view_params.update(rbn_results)
            
    return render_template('index.html', **view_params)


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                           'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)