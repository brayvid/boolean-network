import os
import io
import base64
import tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from matplotlib.animation import FuncAnimation, FFMpegWriter
from flask import Flask, render_template, request, url_for, send_from_directory

from flask_compress import Compress
from flask_assets import Environment, Bundle

app = Flask(__name__, instance_relative_config=True)
Compress(app)

assets = Environment(app)
css = Bundle('css/main.css', filters='cssmin', output='gen/packed.css')
assets.register('all_css', css)

STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
MEDIA_PATH = os.getenv('MEDIA_PATH', os.path.join(STATIC_FOLDER, 'generated_media'))
os.makedirs(MEDIA_PATH, exist_ok=True)

@app.route('/media/<path:filename>')
def serve_media(filename):
    return send_from_directory(MEDIA_PATH, filename)

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
        
        expected_rule_len = int(np.power(2, self.nodes - 1)) if self.nodes > 0 else 1
        if self.nodes == 1 and len(self.rule) == 1:
            pass
        elif expected_rule_len != len(self.rule):
            self.error_message = f"Invalid rule size. Expected {expected_rule_len}, got {len(self.rule)} for {self.nodes} nodes."
            return None

        states_list = [self.state.copy()]
        seen_states = {tuple(self.state)}
        max_iterations = 100
        for _ in np.arange(max_iterations):
            self.update()
            if self.error_message: return None
            current_state_tuple = tuple(self.state)
            if current_state_tuple in seen_states: break
            seen_states.add(current_state_tuple)
            states_list.append(self.state.copy())
        return np.array(states_list)

    def update(self):
        nextState = np.zeros(self.nodes, dtype=int)
        if self.nodes == 0: return

        k_rule = self.nodes - 1 if self.nodes > 1 else 0

        for node_idx in np.arange(self.nodes):
            connected_to_node = self.chart[node_idx]
            num_actual_inputs = len(connected_to_node)
            
            if k_rule > 0:
                rule_input_vector = np.zeros(k_rule, dtype=int)
                for i in range(k_rule):
                    if i < num_actual_inputs: rule_input_vector[i] = self.state[connected_to_node[i]]
                rule_index = int("".join(map(str, rule_input_vector)), 2)
            else:
                rule_index = 0

            if rule_index >= len(self.rule):
                self.error_message = f"Rule index {rule_index} out of bounds."
                return
            nextState[node_idx] = self.rule[rule_index]
        self.state = nextState

def generate_rbn_visuals(k_val, s_val):
    print(f"Requesting visuals for k={k_val}, s={s_val}")

    output = {
        "heatmap_data": None,   # Will hold filename OR base64 string
        "animation_data": None, # Will hold filename OR base64 string
        "is_static": False,     # Tells template how to render
        "error_message": None
    }

    # === OPTIMIZATION: DEFAULT CASE ===
    # If it's the default, we save to disk and reuse it forever.
    if k_val == 6 and s_val == 42:
        file_identifier = f"k{k_val}_s{s_val}"
        heatmap_filename = f"heatmap_{file_identifier}.png"
        animation_filename = f"rbn_animation_{file_identifier}.mp4"
        heatmap_path = os.path.join(MEDIA_PATH, heatmap_filename)
        animation_path = os.path.join(MEDIA_PATH, animation_filename)

        # If files exist, return them immediately
        if os.path.exists(heatmap_path) and os.path.exists(animation_path):
            print("Default case found on disk. Serving from file system.")
            output["is_static"] = True
            output["heatmap_data"] = heatmap_filename
            output["animation_data"] = animation_filename
            return output
        
        # If not, we continue below to generate them, but set a flag to save them at the end.
        save_to_disk = True
    else:
        save_to_disk = False

    # === GENERATION LOGIC ===
    np.random.seed(s_val)
    state = np.random.randint(2, size=k_val)
    chart = []
    for i in range(k_val):
        if k_val > 1:
            possible_influencers = np.setdiff1d(np.arange(k_val), [i])
            num_connections = np.random.randint(1, len(possible_influencers) + 1)
            selected_influencers = np.random.choice(possible_influencers, size=num_connections, replace=False)
        else:
            selected_influencers = np.array([])
        chart.append(selected_influencers)

    k_for_rule = k_val - 1 if k_val > 1 else 0
    rule_len = 2**k_for_rule
    rule = np.random.randint(2, size=rule_len)

    rbn = RandomBooleanNetwork(state.copy(), chart, rule)
    states_history = rbn.go()

    if rbn.error_message or states_history is None or len(states_history) == 0:
        output["error_message"] = rbn.error_message or "Network simulation failed."
        return output

    # 1. Generate Heatmap
    try:
        height = min(1 + k_val * 0.3, 6)
        cmap = mcolors.ListedColormap(['#4B0082', '#FFD700'])
        fig_heatmap, ax_heatmap = plt.subplots(figsize=(8, height))
        ax_heatmap.imshow(states_history.T, aspect='auto', cmap=cmap, interpolation='nearest')
        ax_heatmap.set_ylabel("Node"); ax_heatmap.set_xlabel("Time Step")
        ax_heatmap.set_yticks(np.arange(k_val)); ax_heatmap.set_yticklabels(np.arange(1, k_val + 1))
        
        if save_to_disk:
            fig_heatmap.savefig(heatmap_path, format="png", bbox_inches='tight')
            output["heatmap_data"] = heatmap_filename
        else:
            # Save to memory buffer
            buf = io.BytesIO()
            fig_heatmap.savefig(buf, format="png", bbox_inches='tight')
            buf.seek(0)
            output["heatmap_data"] = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()
        
        plt.close(fig_heatmap)
    except Exception as e:
        output["error_message"] = f"Error generating heatmap: {e}"
        return output

    # 2. Generate Animation
    try:
        G = nx.DiGraph()
        for i in range(k_val): G.add_node(i)
        for i_node, influencers in enumerate(chart):
            for j_influencer in influencers: G.add_edge(j_influencer, i_node)
        
        fig_anim, ax_anim = plt.subplots(figsize=(8, 8))
        pos = nx.circular_layout(G)
        node_colors = ['#FFD700' if s else '#4B0082' for s in states_history[0]]
        scat = nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, ax=ax_anim)
        nx.draw_networkx_edges(G, pos, ax=ax_anim, arrowstyle='->', arrowsize=20, node_size=500)
        nx.draw_networkx_labels(G, pos, ax=ax_anim, labels={i: str(i + 1) for i in range(k_val)}, font_color='white', font_size=12)
        
        ani = FuncAnimation(fig_anim, lambda n: scat.set_color(['#FFD700' if s else '#4B0082' for s in states_history[n]]), frames=len(states_history), interval=500)
        
        if save_to_disk:
            ani.save(animation_path, writer=FFMpegWriter(fps=2, bitrate=1800))
            output["animation_data"] = animation_filename
            output["is_static"] = True
        else:
            # FFMpeg needs a real file path to write to initially, we can't stream directly to pipe with Matplotlib easily.
            # We create a temp file, read it, and delete it immediately.
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                ani.save(tmp_file.name, writer=FFMpegWriter(fps=2, bitrate=1800))
                tmp_file_path = tmp_file.name
            
            # Read the temp file into memory
            with open(tmp_file_path, 'rb') as f:
                video_bytes = f.read()
                output["animation_data"] = base64.b64encode(video_bytes).decode('utf-8')
            
            # Delete the temp file immediately
            os.remove(tmp_file_path)
            output["is_static"] = False

        plt.close(fig_anim)
    except Exception as e:
        output["error_message"] = f'{output.get("error_message", "")} Error generating animation: {e}'

    return output

@app.route('/', methods=['GET', 'POST'])
def index():
    k_to_use = request.form.get('k', 6, type=int)
    s_to_use = request.form.get('s', 42, type=int)
    
    if request.method == 'GET':
        k_to_use, s_to_use = 6, 42
    
    view_params = {"k_value": k_to_use, "s_value": s_to_use, "error_message": None}

    if not (1 <= k_to_use <= 10):
        view_params["error_message"] = "Number of Nodes (K) must be between 1 and 10."
    else:
        rbn_results = generate_rbn_visuals(k_to_use, s_to_use)
        if rbn_results.get("error_message"):
            view_params["error_message"] = rbn_results["error_message"]
        else:
            # Pass results directly to template
            view_params["is_static"] = rbn_results["is_static"]
            view_params["heatmap_data"] = rbn_results["heatmap_data"]
            view_params["animation_data"] = rbn_results["animation_data"]

    return render_template('index.html', **view_params)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(STATIC_FOLDER, 'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)