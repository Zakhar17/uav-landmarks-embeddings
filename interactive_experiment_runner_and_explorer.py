import os
import json
import logging
import base64
import io
import random
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from PIL import Image, ImageDraw, ImageFont

import dash
from dash import dcc, html, Input, Output, State, ALL, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# =====================================================
# 1) CONFIG & LOGGING
# =====================================================
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

# Satellite config
EMBEDDINGS_NPY_ARRAY_PATH = "data/vpair_embeddings/landmarks_satellite_max_urban/embeddings.npy"
EMBEDDINGS_METADATA_CSV_PATH = "data/vpair_embeddings/landmarks_satellite_max_urban/info.csv"

# EMBEDDINGS_NPY_ARRAY_PATH = "data/vpair_embeddings/landmarks_satellite_avg_urban/embeddings.npy"
# EMBEDDINGS_METADATA_CSV_PATH = "data/vpair_embeddings/landmarks_satellite_avg_urban/info.csv"

# EMBEDDINGS_NPY_ARRAY_PATH = "data/vpair_embeddings/landmarks_satellite_sum_urban/embeddings.npy"
# EMBEDDINGS_METADATA_CSV_PATH = "data/vpair_embeddings/landmarks_satellite_sum_urban/info.csv"

IMAGES_ROOT_FOLDER_PATH = "data/vpair/reference_views"

# Drone config (UPDATED PATHS)
DRONE_EMBEDDINGS_NPY_ARRAY_PATH = "data/vpair_embeddings/landmarks_drone_max_urban/embeddings.npy"
DRONE_EMBEDDINGS_METADATA_CSV_PATH = "data/vpair_embeddings/landmarks_drone_max_urban/info.csv"

# DRONE_EMBEDDINGS_NPY_ARRAY_PATH = "data/vpair_embeddings/landmarks_drone_avg_urban/embeddings.npy"
# DRONE_EMBEDDINGS_METADATA_CSV_PATH = "data/vpair_embeddings/landmarks_drone_avg_urban/info.csv"

# DRONE_EMBEDDINGS_NPY_ARRAY_PATH = "data/vpair_embeddings/landmarks_drone_sum_urban/embeddings.npy"
# DRONE_EMBEDDINGS_METADATA_CSV_PATH = "data/vpair_embeddings/landmarks_drone_sum_urban/info.csv"

DRONE_IMAGES_ROOT_FOLDER_PATH = "data/vpair/queries"

RESULTS_FOLDER = "data/landmarks_ui_results"
# RESULTS_FOLDER = "data/landmarks_avg_ui_results"
# RESULTS_FOLDER = "data/landmarks_sum_ui_results"

os.makedirs(RESULTS_FOLDER, exist_ok=True)

EMBEDDING_COORDS_ROOT = "data/embedding_coords_cache"
# EMBEDDING_COORDS_ROOT = "data/embedding_avg_coords_cache"
# EMBEDDING_COORDS_ROOT = "data/embedding_sum_coords_cache"

os.makedirs(EMBEDDING_COORDS_ROOT, exist_ok=True)

LAYER_TO_DIM = {
    0: 16,
    1: 32,
    2: 64,
    3: 64,
    4: 128, 
    5: 128,
    6: 128,
    7: 256,
    8: 256,
    9: 256,
    10: 256
}

def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_embeddings_for_layers(embeddings: np.ndarray, layers: list, layer_to_dim: dict = LAYER_TO_DIM) -> np.ndarray:
    invalid_layers = [layer for layer in layers if layer not in layer_to_dim]
    if invalid_layers:
        raise ValueError(f"Layers {invalid_layers} are not present in layer_to_dim mapping.")

    layer_start_indices = {}
    current_index = 0
    for layer in sorted(layer_to_dim.keys()):
        layer_start_indices[layer] = current_index
        current_index += layer_to_dim[layer]

    selected_columns = []
    for layer in layers:
        start = layer_start_indices[layer]
        dim = layer_to_dim[layer]
        selected_columns.extend(range(start, start + dim))
    return embeddings[:, selected_columns]

def generate_experiment_folder(conf_threshold, layer_list, algo_name, algo_params):
    layer_str = "_".join(map(str, layer_list)) if layer_list else "all"
    params_sorted = sorted(algo_params.items(), key=lambda x: x[0])
    params_str = "_".join(f"{k}-{v}" for k, v in params_sorted)
    folder_name = f"conf-{conf_threshold}_layers-{layer_str}_{algo_name}_{params_str}"
    return folder_name

def generate_embedding_config_folder(conf_threshold, layer_list):
    layer_str = "_".join(map(str, layer_list)) if layer_list else "all"
    folder_name = f"conf-{conf_threshold}_layers-{layer_str}"
    return folder_name

def draw_boxes_on_image(image_path, instances_info):
    """
    Draw bounding boxes for one or more instances on a single image.
    If label_val == -1 => red bounding box, else blue bounding box.
    The font size is set to 30 for clearer labeling.
    """
    im = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    # Use a bigger default font
    font = ImageFont.load_default(size=30)
    try:
        # Pillow doesn't actually implement load_default(size=30) in many versions,
        # so you can either attempt it or fallback. We'll attempt.
        font.size = 30
    except Exception as e:
        pass

    for info in instances_info:
        x0, y0, x1, y1 = info["box"]
        instance_id = info["instance_id"]
        label_val = info.get("label_val", 0)
        color = "red" if label_val == -1 else "blue"

        draw.rectangle([(x0, y0), (x1, y1)], outline=color, width=3)
        label_txt = f"ID:{instance_id}"
        draw.text((x0, y0), label_txt, fill=color, font=font)
    return im

# =====================================================
# 3) BASE CLASSES & ALGORITHMS
# =====================================================
class BaseClusteringMethod:
    @classmethod
    def get_name(cls):
        raise NotImplementedError

    @classmethod
    def get_param_definition(cls):
        raise NotImplementedError

    def __init__(self, **params):
        self.params = params

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class DBSCANMethod(BaseClusteringMethod):
    @classmethod
    def get_name(cls):
        return "DBSCAN"

    @classmethod
    def get_param_definition(cls):
        return {
            "eps": {"type": "float", "default": 0.5},
            "min_samples": {"type": "int", "default": 5}
        }

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        eps = float(self.params.get("eps", 0.5))
        min_samples = int(self.params.get("min_samples", 5))
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(embeddings)
        return labels

class IsolationForestMethod(BaseClusteringMethod):
    @classmethod
    def get_name(cls):
        return "IsolationForest"

    @classmethod
    def get_param_definition(cls):
        return {
            "n_estimators": {"type": "int", "default": 500},
            "contamination": {"type": "float", "default": 0.01}
        }

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        n_estimators = int(self.params.get("n_estimators", 500))
        contamination = float(self.params.get("contamination", 0.01))
        iso = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
        predicted = iso.fit_predict(embeddings)
        labels = np.where(predicted == 1, 0, -1)
        return labels

AVAILABLE_ALGORITHMS = [IsolationForestMethod, DBSCANMethod]
ALGO_NAME_TO_CLASS = {algo.get_name(): algo for algo in AVAILABLE_ALGORITHMS}

# =====================================================
# 4) DASH APP SETUP
# =====================================================
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)
app.title = "Landmark Detection UI"

# -----------------------------------------------------
# 5A) TABS: Launch + Visualize
# -----------------------------------------------------
launch_tab_content = dbc.Container([
    html.H2("Launch Landmark Detection Algorithm", className="my-3"),

    html.Div([
        html.Label("Select Layers to Use:"),
        dcc.Dropdown(
            id="layer-selection-dropdown",
            options=[{"label": f"Layer {l}", "value": l} for l in LAYER_TO_DIM.keys()],
            value=[0, 1],
            multi=True
        )
    ], className="mb-3"),

    html.Div([
        html.Label("Confidence Threshold:"),
        dcc.Slider(
            id="confidence-threshold-slider",
            min=0.0,
            max=1.0,
            step=0.01,
            value=0.5,
            marks={i/10: str(i/10) for i in range(0, 11)}
        ),
        html.Div(id="confidence-threshold-value", style={"marginTop": 10}),
    ], className="mb-3"),

    html.Div([
        html.Label("Algorithm:"),
        dcc.Dropdown(
            id="algo-selection-dropdown",
            options=[{"label": algo.get_name(), "value": algo.get_name()} for algo in AVAILABLE_ALGORITHMS],
            value=AVAILABLE_ALGORITHMS[0].get_name(),
            clearable=False
        )
    ], className="mb-3"),

    html.Div(id="algo-params-div", className="mb-3"),

    dbc.Button("Run Algorithm", id="run-algo-button", color="primary"),
    dcc.Loading(
        id="loading-launch",
        children=[html.Div(id="launch-status", style={"marginTop": 20})],
        type="default"
    ),
], fluid=True, style={"width": "100%"})

visualize_tab_content = dbc.Container([
    html.H2("Visualise Results", className="my-3"),

    html.Div([
        html.Label("Select a Previous Run Configuration:"),
        dcc.Dropdown(
            id="previous-run-dropdown",
            options=[],
            value=None,
            placeholder="Select a run"
        )
    ], className="mb-3"),

    html.Div([
        html.Label("Sampling fraction of non-outliers:"),
        dcc.Dropdown(
            id="sampling-fraction-dropdown",
            options=[
                {"label": "All", "value": "all"},
                {"label": "5%",  "value": "0.05"},
                {"label": "10%", "value": "0.10"},
                {"label": "20%", "value": "0.20"},
            ],
            value="0.05",
            clearable=False
        )
    ], className="mb-3"),

    html.Div([
        html.Label("Show t-SNE plot?"),
        dcc.Checklist(
            id="tsne-toggle-checklist",
            options=[{"label": "", "value": "show_tsne"}],
            value=[]
        )
    ], className="mb-3"),

    dbc.Button("Load & Show Results", id="load-visualization-button", color="secondary"),
    dcc.Loading(
        id="loading-visuals",
        children=[
            html.Div(id="visuals-container", children=[])
        ],
        type="default"
    ),

    html.Hr(),
    # Single-image visualization
    html.H3("Visualise bounding boxes for a single image", className="my-3"),
    html.Div([
        html.Label("Image filename:"),
        dcc.Input(
            id="single-image-filename",
            type="text",
            placeholder="Enter image filename (e.g. 00001.png)",
            style={"width": "300px"}
        ),
    ], className="mb-2"),
    dbc.Button("Show bounding boxes for this image", id="show-image-button", color="primary"),
    html.Div(id="single-image-display", style={"marginTop": 20})

], fluid=True, style={"width": "100%"})

# -----------------------------------------------------
# 5B) TAB: Performance evaluation
# -----------------------------------------------------
performance_tab_content = dbc.Container([
    html.H2("Performance Evaluation", className="my-3"),

    html.Div([
        html.Label("Select a Previous Run Configuration:"),
        dcc.Dropdown(
            id="previous-run-dropdown-eval",
            options=[],
            value=None,
            placeholder="Select a run for evaluation"
        )
    ], className="mb-3"),

    dbc.Button("Suggest evaluation sample", id="suggest-sample-button", color="secondary"),
    html.Div(id="eval-samples-container", children=[], style={"marginTop": 20}),

    html.Hr(),
    dbc.Button("Calculate metrics", id="calc-metrics-button", color="primary"),
    html.Div(id="metrics-display", style={"marginTop": 20}),
    html.Hr(),
    html.H4("Retrieval Visualisation"),
    html.Div(id="retrieval-visuals-container", style={"marginTop": 20})

], fluid=True, style={"width": "100%"})

# -----------------------------------------------------
# 5C) MAIN LAYOUT
# -----------------------------------------------------
app.layout = html.Div([
    dbc.Tabs([
        dbc.Tab(launch_tab_content, label="Launch landmark detection algorithm", tab_id="launch-tab"),
        dbc.Tab(visualize_tab_content, label="Visualise results", tab_id="visualize-tab"),
        dbc.Tab(performance_tab_content, label="Performance evaluation", tab_id="eval-tab")
    ], id="main-tabs", active_tab="launch-tab"),

    # A store for evaluation data
    dcc.Store(id="evaluation-store", data={"rows": []})
], style={"width": "100%"})


# -----------------------------------------------------
# 6) CALLBACKS (LAUNCH + VISUALIZE)
# -----------------------------------------------------
@app.callback(
    Output("confidence-threshold-value", "children"),
    Input("confidence-threshold-slider", "value")
)
def update_conf_threshold_display(val):
    return f"Confidence Threshold = {val:.2f}"

@app.callback(
    Output("algo-params-div", "children"),
    Input("algo-selection-dropdown", "value")
)
def update_algo_params_ui(algo_name):
    if not algo_name:
        return []
    algo_class = ALGO_NAME_TO_CLASS[algo_name]
    param_def = algo_class.get_param_definition()
    controls = []
    for param_name, param_info in param_def.items():
        label = html.Label(param_name, style={"font-weight": "bold", "marginRight": "8px"})
        default_val = param_info.get("default", 0)
        input_ctrl = dcc.Input(
            id={"type": "algo-param-input", "paramName": param_name},
            type="number",
            value=default_val,
            style={"width": "10rem"}
        )
        controls.append(html.Div([label, input_ctrl], className="mb-2"))
    return controls

@app.callback(
    Output("launch-status", "children"),
    Input("run-algo-button", "n_clicks"),
    State("layer-selection-dropdown", "value"),
    State("confidence-threshold-slider", "value"),
    State("algo-selection-dropdown", "value"),
    State({"type": "algo-param-input", "paramName": ALL}, "value"),
    State({"type": "algo-param-input", "paramName": ALL}, "id"),
    prevent_initial_call=True
)
def run_algorithm(n_clicks,
                  selected_layers,
                  conf_threshold,
                  algo_name,
                  param_values,
                  param_ids):
    if not n_clicks:
        return dash.no_update

    embeddings = np.load(EMBEDDINGS_NPY_ARRAY_PATH)
    df_meta = pd.read_csv(EMBEDDINGS_METADATA_CSV_PATH)
    mask = df_meta["conf"] >= conf_threshold
    df_meta_filtered = df_meta[mask].copy()
    embeddings_filtered = embeddings[mask.values]

    if selected_layers:
        embeddings_filtered = get_embeddings_for_layers(embeddings_filtered, selected_layers, LAYER_TO_DIM)

    algo_params = {}
    for val, pid in zip(param_values, param_ids):
        param_name = pid["paramName"]
        algo_params[param_name] = val

    algo_class = ALGO_NAME_TO_CLASS[algo_name]
    algo_instance = algo_class(**algo_params)
    labels = algo_instance.fit_predict(embeddings_filtered)

    run_folder_name = generate_experiment_folder(conf_threshold, selected_layers, algo_name, algo_params)
    run_folder_path = os.path.join(RESULTS_FOLDER, run_folder_name)
    ensure_folder(run_folder_path)

    np.save(os.path.join(run_folder_path, "cluster_labels.npy"), labels)
    df_meta_filtered.to_csv(os.path.join(run_folder_path, "filtered_metadata.csv"), index=False)

    with open(os.path.join(run_folder_path, "params.json"), "w") as f:
        json.dump({
            "algo_name": algo_name,
            "params": algo_params,
            "confidence_threshold": conf_threshold,
            "layers": selected_layers
        }, f, indent=2)

    msg = f"Run completed. Results saved to {run_folder_path}"
    return html.Div([html.P("Algorithm run completed."), html.P(msg)])


@app.callback(
    Output("previous-run-dropdown", "options"),
    Input("main-tabs", "active_tab")
)
def update_previous_runs_options(active_tab):
    if active_tab != "visualize-tab":
        return []
    runs = []
    for folder in os.listdir(RESULTS_FOLDER):
        run_path = os.path.join(RESULTS_FOLDER, folder)
        if os.path.isdir(run_path):
            runs.append({"label": folder, "value": folder})
    return sorted(runs, key=lambda x: x["label"])


@app.callback(
    Output("visuals-container", "children"),
    Input("load-visualization-button", "n_clicks"),
    State("previous-run-dropdown", "value"),
    State("sampling-fraction-dropdown", "value"),
    State("tsne-toggle-checklist", "value"),
    prevent_initial_call=True
)
def load_and_show_results(n_clicks, selected_run, sampling_value, tsne_toggle_vals):
    if not selected_run:
        return "No run selected."
    
    run_folder_path = os.path.join(RESULTS_FOLDER, selected_run)
    label_path = os.path.join(run_folder_path, "cluster_labels.npy")
    if not os.path.exists(label_path):
        return "Cluster results not found in the selected run."

    labels = np.load(label_path)
    df_meta_filtered = pd.read_csv(os.path.join(run_folder_path, "filtered_metadata.csv"))
    df_meta_filtered.reset_index(drop=True, inplace=True)

    with open(os.path.join(run_folder_path, "params.json"), "r") as f:
        config_json = json.load(f)
    conf_threshold = config_json["confidence_threshold"]
    used_layers = config_json["layers"]

    embeddings_all = np.load(EMBEDDINGS_NPY_ARRAY_PATH)
    df_meta_all = pd.read_csv(EMBEDDINGS_METADATA_CSV_PATH)
    mask = df_meta_all["conf"] >= conf_threshold
    embeddings_filtered = embeddings_all[mask.values]
    if used_layers:
        embeddings_filtered = get_embeddings_for_layers(embeddings_filtered, used_layers, LAYER_TO_DIM)

    # Store PCA/t-SNE in folder keyed by embeddings config
    emb_config_folder_name = generate_embedding_config_folder(conf_threshold, used_layers)
    emb_config_folder_path = os.path.join(EMBEDDING_COORDS_ROOT, emb_config_folder_name)
    ensure_folder(emb_config_folder_path)

    pca_coords_path = os.path.join(emb_config_folder_path, "pca_coords.npy")
    tsne_coords_path = os.path.join(emb_config_folder_path, "tsne_coords.npy")

    if os.path.exists(pca_coords_path):
        pca_coords = np.load(pca_coords_path)
        logging.debug(f"Loaded existing PCA coords from {pca_coords_path}")
    else:
        logging.debug("Computing PCA coords...")
        pca = PCA(n_components=2, random_state=42)
        pca_coords = pca.fit_transform(embeddings_filtered)
        np.save(pca_coords_path, pca_coords)

    compute_tsne = ("show_tsne" in tsne_toggle_vals)
    tsne_coords = None
    if compute_tsne:
        if os.path.exists(tsne_coords_path):
            logging.debug(f"Loaded existing t-SNE coords from {tsne_coords_path}")
            tsne_coords = np.load(tsne_coords_path)
        else:
            logging.debug("Computing t-SNE coords...")
            tsne = TSNE(n_components=2, random_state=42)
            tsne_coords = tsne.fit_transform(embeddings_filtered)
            np.save(tsne_coords_path, tsne_coords)

    df_plot = df_meta_filtered.copy()
    df_plot["label"] = labels
    df_plot["pca_x"] = pca_coords[:, 0]
    df_plot["pca_y"] = pca_coords[:, 1]
    if compute_tsne and tsne_coords is not None:
        df_plot["tsne_x"] = tsne_coords[:, 0]
        df_plot["tsne_y"] = tsne_coords[:, 1]

    df_plot["orig_index"] = df_plot.index

    if sampling_value != "all":
        frac = float(sampling_value)
        logging.info(f"Sampling fraction for non-outliers: {frac*100:.1f}%")
        is_outlier = df_plot["label"] == -1
        df_outliers = df_plot[is_outlier]
        df_inliers = df_plot[~is_outlier]
        if frac < 1.0:
            df_inliers = df_inliers.sample(frac=frac, random_state=42)
        df_plot = pd.concat([df_inliers, df_outliers], axis=0)
    else:
        logging.info("Displaying ALL points (including all inliers and outliers).")

    unique_labels = df_plot["label"].unique().tolist()
    unique_labels = [lbl for lbl in unique_labels if lbl != -1] + [-1]

    fig_pca = go.Figure()
    fig_tsne = go.Figure()

    palette = px.colors.qualitative.Plotly
    color_map = {}
    inlier_ids = [cid for cid in unique_labels if cid != -1]
    for i, cid in enumerate(inlier_ids):
        color_map[cid] = palette[i % len(palette)]
    color_map[-1] = "orange"

    def add_cluster_trace(fig, df_subset, cid, xcol, ycol):
        if cid == -1:
            fig.add_trace(go.Scatter(
                x=df_subset[xcol],
                y=df_subset[ycol],
                mode="markers",
                marker=dict(size=10, color="orange", line=dict(width=1, color="black")),
                text=df_subset["orig_index"].astype(str),
                hovertext=df_subset.to_json(orient="records"),
                name="Outliers (-1)"
            ))
        else:
            fig.add_trace(go.Scatter(
                x=df_subset[xcol],
                y=df_subset[ycol],
                mode="markers",
                marker=dict(size=6,
                             color= color_map[cid], 
                             line=dict(width=1, color="black")),
                text=df_subset["orig_index"].astype(str),
                hovertext=df_subset.to_json(orient="records"),
                name=f"Cluster {cid}"
            ))

    for cid in unique_labels:
        sub = df_plot[df_plot["label"] == cid]
        add_cluster_trace(fig_pca, sub, cid, "pca_x", "pca_y")

    # figure size logic
    if compute_tsne and tsne_coords is not None:
        pca_width, pca_height = 800, 600
    else:
        pca_width, pca_height = 1000, 700

    fig_pca.update_layout(
        title="PCA Plot",
        clickmode="event+select",
        hovermode="closest",
        showlegend=False,
        width=pca_width,
        height=pca_height
    )

    if compute_tsne and tsne_coords is not None:
        for cid in unique_labels:
            sub = df_plot[df_plot["label"] == cid]
            add_cluster_trace(fig_tsne, sub, cid, "tsne_x", "tsne_y")
        fig_tsne.update_layout(
            title="t-SNE Plot",
            clickmode="event+select",
            hovermode="closest",
            showlegend=False,
            width=800,
            height=600
        )

    tsne_display_style = {}
    if not compute_tsne or tsne_coords is None:
        tsne_display_style = {"display": "none"}

    content = html.Div([
        dbc.Row([
            dbc.Col(
                dcc.Graph(
                    figure=fig_pca,
                    id="pca-graph"
                ),
                width=6 if compute_tsne else 12,
                style={"textAlign": "center"}
            ),
            dbc.Col(
                dcc.Graph(
                    figure=fig_tsne,
                    id="tsne-graph",
                    style=tsne_display_style
                ),
                width=6,
                style={"textAlign": "center"}
            ),
        ], style={"width": "100%"}),
        html.Hr(),
        html.Div(id="selected-images-container", children=[], style={"width": "100%"})
    ], style={"width": "100%"})

    return content


# 6.6 existing callback: display selected images from PCA/TSNE
@app.callback(
    Output("selected-images-container", "children"),
    [Input("pca-graph", "selectedData"),
     Input("tsne-graph", "selectedData")],
    State("previous-run-dropdown", "value"),
    prevent_initial_call=True
)
def display_selected_images(pca_selected, tsne_selected, selected_run):
    if not selected_run:
        return []

    selected_orig_indices = set()
    for sel_data in [pca_selected, tsne_selected]:
        if sel_data and "points" in sel_data:
            for pt in sel_data["points"]:
                idx_str = pt.get("text")
                if idx_str is not None:
                    selected_orig_indices.add(int(idx_str))

    if not selected_orig_indices:
        return []

    run_folder_path = os.path.join(RESULTS_FOLDER, selected_run)
    df_meta_filtered = pd.read_csv(os.path.join(run_folder_path, "filtered_metadata.csv"))
    df_meta_filtered.reset_index(drop=True, inplace=True)
    labels = np.load(os.path.join(run_folder_path, "cluster_labels.npy"))

    sub_df = df_meta_filtered.iloc[list(selected_orig_indices)]
    children = []
    grouped = sub_df.groupby("image")
    for image_name, group_rows in grouped:
        instance_info = []
        for i, row in group_rows.iterrows():
            box = (row["box_0"], row["box_1"], row["box_2"], row["box_3"])
            label_val = labels[i] if i < len(labels) else 0
            instance_info.append({
                "box": box,
                "instance_id": i,
                "label_val": label_val
            })
        image_path = os.path.join(IMAGES_ROOT_FOLDER_PATH, image_name)
        if os.path.exists(image_path):
            annotated = draw_boxes_on_image(image_path, instance_info)
            buf = io.BytesIO()
            annotated.save(buf, format="PNG")
            data = base64.b64encode(buf.getvalue()).decode("utf-8")
            children.append(html.Div([
                html.Img(src=f"data:image/png;base64,{data}", style={"maxWidth": "600px", "maxHeight": "600px"}),
                html.P(f"Image file: {image_name}")
            ], style={"margin": "20px", "display": "inline-block"}))

    return children


# -----------------------------------------------------
# 7) NEW: Single-Image Visualization
# -----------------------------------------------------
@app.callback(
    Output("single-image-display", "children"),
    Input("show-image-button", "n_clicks"),
    State("previous-run-dropdown", "value"),
    State("single-image-filename", "value"),
    prevent_initial_call=True
)
def show_single_image(n_clicks, selected_run, filename):
    """
    The user enters a filename, e.g. "00001.png".
    We load run_folder's filtered_metadata.csv, cluster_labels.npy,
    find all instances with that filename, and draw bounding boxes.
    """
    if not selected_run or not filename:
        return "Please select a run and enter a valid filename."

    run_folder_path = os.path.join(RESULTS_FOLDER, selected_run)
    # load metadata & labels
    filtered_meta_path = os.path.join(run_folder_path, "filtered_metadata.csv")
    labels_path = os.path.join(run_folder_path, "cluster_labels.npy")

    if not (os.path.exists(filtered_meta_path) and os.path.exists(labels_path)):
        return "Cluster results or metadata not found in the selected run."

    df_meta_filtered = pd.read_csv(filtered_meta_path)
    df_meta_filtered.reset_index(drop=True, inplace=True)
    labels = np.load(labels_path)

    # find all rows with image == filename
    sub_df = df_meta_filtered[df_meta_filtered["image"] == filename].copy()
    if sub_df.empty:
        return f"No instances found for image: {filename}"

    # gather bounding boxes
    instance_info = []
    for i, row in sub_df.iterrows():
        box = (row["box_0"], row["box_1"], row["box_2"], row["box_3"])
        label_val = labels[i] if i < len(labels) else 0
        instance_info.append({
            "box": box,
            "instance_id": i,
            "label_val": label_val
        })

    # check if image file actually exists
    image_path = os.path.join(IMAGES_ROOT_FOLDER_PATH, filename)
    if not os.path.exists(image_path):
        return f"Image file '{filename}' not found on disk."

    annotated = draw_boxes_on_image(image_path, instance_info)
    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode("utf-8")

    # return the annotated image
    return html.Div([
        html.Img(src=f"data:image/png;base64,{data}", style={"maxWidth": "600px", "maxHeight": "600px"}),
        html.P(f"Image file: {filename}")
    ], style={"margin": "20px", "display": "inline-block"})


# -----------------------------------------------------
# 7) PERFORMANCE EVALUATION TAB CALLBACKS
# -----------------------------------------------------

@app.callback(
    Output("previous-run-dropdown-eval", "options"),
    Input("main-tabs", "active_tab")
)
def update_eval_run_options(active_tab):
    if active_tab != "eval-tab":
        return []
    runs = []
    for folder in os.listdir(RESULTS_FOLDER):
        run_path = os.path.join(RESULTS_FOLDER, folder)
        if os.path.isdir(run_path):
            runs.append({"label": folder, "value": folder})
    return sorted(runs, key=lambda x: x["label"])


# 7.2 SUGGEST EVALUATION SAMPLE => update store
@app.callback(
    Output("evaluation-store", "data"),
    Input("suggest-sample-button", "n_clicks"),
    State("previous-run-dropdown-eval", "value"),
    State("evaluation-store", "data"),
    prevent_initial_call=True
)
def suggest_evaluation_sample(n_clicks, selected_run, store_data):
    if not selected_run:
        return store_data

    run_folder_path = os.path.join(RESULTS_FOLDER, selected_run)
    meta_path = os.path.join(run_folder_path, "filtered_metadata.csv")
    labels_path = os.path.join(run_folder_path, "cluster_labels.npy")
    param_path = os.path.join(run_folder_path, "params.json")

    if not (os.path.exists(meta_path) and os.path.exists(labels_path) and os.path.exists(param_path)):
        return store_data

    # load run config to see which threshold was used
    with open(param_path, "r") as f:
        cfg = json.load(f)
    conf_threshold = cfg["confidence_threshold"]

    df_sat = pd.read_csv(meta_path).reset_index(drop=True)
    sat_labels = np.load(labels_path)
    df_sat["label"] = sat_labels

    # images that have at least one -1 building
    group_lbls = df_sat.groupby("image")["label"].apply(lambda s: (s == -1).any())
    candidate_images = group_lbls[group_lbls].index.tolist()

    # exclude images used
    already_shown = [row["image_name"] for row in store_data["rows"]]
    candidate_images = [img for img in candidate_images if img not in already_shown]
    if not candidate_images:
        return store_data  # no more samples

    chosen_image = random.choice(candidate_images)

    # gather sat building info
    sub_sat = df_sat[df_sat["image"] == chosen_image].copy()

    # gather drone building info
    df_drone = pd.read_csv(DRONE_EMBEDDINGS_METADATA_CSV_PATH).reset_index(drop=True)
    # filter by same threshold
    mask_drone = df_drone["conf"] >= conf_threshold
    df_drone = df_drone[mask_drone].reset_index(drop=True)

    sub_drone = df_drone[df_drone["image"] == chosen_image].copy()

    row_data = {
        "image_name": chosen_image,
        "sat_bldgs": sub_sat.to_dict(orient="records"),
        "drone_bldgs": sub_drone.to_dict(orient="records"),
        "annotated_pairs": [],
        "show_dropdowns": False
    }
    store_data["rows"].append(row_data)
    return store_data


# 7.3 ANNOTATE / CREATE PAIR => update store
@app.callback(
    Output("evaluation-store", "data", allow_duplicate=True),
    [Input({"type": "annotate-button", "index": ALL}, "n_clicks"),
     Input({"type": "create-pair-button", "index": ALL}, "n_clicks")],
    [State("previous-run-dropdown-eval", "value"),
     State("evaluation-store", "data"),
     State({"type": "satellite-id-dropdown", "index": ALL}, "value"),
     State({"type": "drone-id-dropdown", "index": ALL}, "value")],
    prevent_initial_call=True
)
def handle_annotation_buttons(annotate_clicks, create_clicks,
                              selected_run, store_data,
                              sat_values_list, drone_values_list):
    ctx = callback_context
    if not ctx.triggered:
        return store_data

    tid = ctx.triggered[0]["prop_id"].split(".")[0]
    if not tid.startswith("{"):
        return store_data

    tid_dict = eval(tid)  # e.g. {"type":"annotate-button","index":0}
    row_index = tid_dict["index"]
    button_type = tid_dict["type"]

    rows = store_data["rows"]
    if row_index >= len(rows):
        return store_data

    row_data = rows[row_index]

    if button_type == "annotate-button":
        row_data["show_dropdowns"] = True

    elif button_type == "create-pair-button":
        sat_id = sat_values_list[row_index]
        drone_id = drone_values_list[row_index]
        if sat_id is None or drone_id is None:
            return store_data

        run_folder_path = os.path.join(RESULTS_FOLDER, selected_run)
        df_sat_run = pd.read_csv(os.path.join(run_folder_path, "filtered_metadata.csv")).reset_index(drop=True)
        sat_labels = np.load(os.path.join(run_folder_path, "cluster_labels.npy"))

        pair_type = "standard_building"
        if 0 <= sat_id < len(sat_labels):
            if sat_labels[sat_id] == -1:
                pair_type = "landmark_building"

        pair_obj = {
            "sat_id": sat_id,
            "drone_id": drone_id,
            "pair_type": pair_type
        }
        row_data["annotated_pairs"].append(pair_obj)
        row_data["show_dropdowns"] = False

    return store_data


#############################
# ROW DELETION CALLBACK
#############################
@app.callback(
    Output("evaluation-store", "data", allow_duplicate=True),
    Input({"type": "delete-row-button", "index": ALL}, "n_clicks"),
    State("evaluation-store", "data"),
    prevent_initial_call=True
)
def delete_evaluation_row(delete_clicks, store_data):
    """
    Remove exactly one row from store_data["rows"] if the user actually
    clicked the "Delete" button. We do this by scanning which element of
    `delete_clicks` is > 0, and removing the corresponding row index.

    This avoids any reliance on callback_context, ensuring that
    "Annotate" clicks do NOT interfere with row deletion.
    """
    if not delete_clicks:
        # No "delete" clicks recorded => do nothing
        return store_data

    rows = store_data.get("rows", [])

    # Find the first index i where delete_clicks[i] is non-zero
    # That means the user pressed the delete button for row i
    for i, nclicks in enumerate(delete_clicks):
        if nclicks and nclicks > 0:
            # Remove row i if valid, then stop
            if 0 <= i < len(rows):
                rows.pop(i)
            break

    return store_data

# 7.4 RENDER EVAL ROWS => triggered by store or run changes
@app.callback(
    Output("eval-samples-container", "children"),
    Input("evaluation-store", "data"),
    Input("previous-run-dropdown-eval", "value")
)
def render_eval_rows(store_data, selected_run):
    if not selected_run:
        return "No run selected for evaluation."

    rows = store_data.get("rows", [])
    if not rows:
        return "No samples suggested yet."
    return build_eval_rows_html(rows, selected_run)


#########################
# BUILD_EVAL_ROWS_HTML #
#########################

def build_eval_rows_html(rows, selected_run):
    """
    Build the HTML for each suggested evaluation sample row, including:
      - Satellite image
      - Drone image
      - Annotated pairs
      - "Annotate" button
      - "Delete" button
      - If "show_dropdowns" is True => show the pair-creation dropdowns
    """
    run_folder_path = os.path.join(RESULTS_FOLDER, selected_run)
    labels_path = os.path.join(run_folder_path, "cluster_labels.npy")
    if not os.path.exists(labels_path):
        return [html.Div("No cluster_labels found.")]

    sat_labels = np.load(labels_path)
    df_sat = pd.read_csv(os.path.join(run_folder_path, "filtered_metadata.csv")).reset_index(drop=True)
    df_drone_global = pd.read_csv(DRONE_EMBEDDINGS_METADATA_CSV_PATH).reset_index(drop=True)

    def render_image(image_path, instance_info, div_id):
        if not os.path.exists(image_path):
            return html.Div([html.P(f"Image not found: {image_path}")])
        annotated_img = draw_boxes_on_image(image_path, instance_info)
        buf = io.BytesIO()
        annotated_img.save(buf, format="PNG")
        data = base64.b64encode(buf.getvalue()).decode("utf-8")
        return html.Div([
            html.Img(src=f"data:image/png;base64,{data}", style={"maxWidth": "400px"}),
        ], style={"textAlign": "center"}, id=div_id)

    out = []
    for row_idx, row_data in enumerate(rows):
        image_name = row_data["image_name"]
        sub_sat = row_data["sat_bldgs"]
        sub_drone = row_data["drone_bldgs"]
        annotated_pairs = row_data["annotated_pairs"]
        show_dd = row_data.get("show_dropdowns", False)

        # Build bounding box data for satellite
        sat_instance_info = []
        for irec in sub_sat:
            matches = df_sat.index[
                (df_sat["image"] == irec["image"]) &
                (df_sat["box_0"] == irec["box_0"]) &
                (df_sat["box_1"] == irec["box_1"]) &
                (df_sat["box_2"] == irec["box_2"]) &
                (df_sat["box_3"] == irec["box_3"])
            ].tolist()
            sat_id_idx = matches[0] if matches else 0
            label_val = sat_labels[sat_id_idx] if 0 <= sat_id_idx < len(sat_labels) else 0
            sat_instance_info.append({
                "box": (irec["box_0"], irec["box_1"], irec["box_2"], irec["box_3"]),
                "instance_id": sat_id_idx,
                "label_val": label_val
            })
        sat_img_path = os.path.join(IMAGES_ROOT_FOLDER_PATH, image_name)
        sat_div = render_image(sat_img_path, sat_instance_info, f"sat-image-{row_idx}")

        # Build bounding box data for drone
        drone_instance_info = []
        for irec in sub_drone:
            matches = df_drone_global.index[
                (df_drone_global["image"] == irec["image"]) &
                (df_drone_global["box_0"] == irec["box_0"]) &
                (df_drone_global["box_1"] == irec["box_1"]) &
                (df_drone_global["box_2"] == irec["box_2"]) &
                (df_drone_global["box_3"] == irec["box_3"])
            ].tolist()
            dr_id_idx = matches[0] if matches else 0
            drone_instance_info.append({
                "box": (irec["box_0"], irec["box_1"], irec["box_2"], irec["box_3"]),
                "instance_id": dr_id_idx,
                "label_val": 0
            })
        drone_img_path = os.path.join(DRONE_IMAGES_ROOT_FOLDER_PATH, image_name)
        drone_div = render_image(drone_img_path, drone_instance_info, f"drone-image-{row_idx}")

        # Show existing pairs
        pair_list_html = []
        for pair in annotated_pairs:
            pair_list_html.append(html.Div([
                html.P(
                    f"Pair: satellite_id={pair['sat_id']} | drone_id={pair['drone_id']} | type={pair['pair_type']}"
                )
            ]))

        # The annotation dropdowns (hidden by default, appear if show_dd=True)
        style_dd = {"display": "block"} if show_dd else {"display": "none"}

        # Build the building ID dropdown options
        sat_options = []
        for sinfo in sat_instance_info:
            sat_options.append({
                "label": f"ID {sinfo['instance_id']} (label={sinfo['label_val']})",
                "value": sinfo["instance_id"]
            })
        drone_options = []
        for dinfo in drone_instance_info:
            drone_options.append({
                "label": f"ID {dinfo['instance_id']}",
                "value": dinfo["instance_id"]
            })

        row_html = html.Div([
            html.H4(f"Sample #{row_idx+1}: {image_name}"),
            dbc.Row([
                dbc.Col(sat_div, width=6),
                dbc.Col(drone_div, width=6),
            ]),
            html.Div(pair_list_html),
            # Annotate button
            dbc.Button(
                "Annotate",
                id={"type": "annotate-button", "index": row_idx},
                n_clicks=0,
                style={"marginBottom": "10px", "marginRight": "10px"}
            ),
            # NEW: "Delete row" button
            dbc.Button(
                "Delete",
                id={"type": "delete-row-button", "index": row_idx},
                color="danger",
                style={"marginBottom": "10px"}
            ),

            # The hidden or shown block with satellite/drone dropdowns + "Create evaluation pair"
            html.Div([
                dcc.Dropdown(
                    id={"type": "satellite-id-dropdown", "index": row_idx},
                    options=sat_options,
                    placeholder="Select satellite building ID"
                ),
                dcc.Dropdown(
                    id={"type": "drone-id-dropdown", "index": row_idx},
                    options=drone_options,
                    placeholder="Select drone building ID"
                ),
                dbc.Button(
                    "Create evaluation pair",
                    id={"type": "create-pair-button", "index": row_idx},
                    color="primary",
                    style={"marginTop": "5px"}
                )
            ], style=style_dd),
            html.Hr()
        ], style={"border": "1px solid #ccc", "padding": "10px", "marginBottom": "20px"})

        out.append(row_html)
    return out


##############################
# CALCULATE METRICS CALLBACK
##############################
@app.callback(
    Output("metrics-display", "children"),
    Output("retrieval-visuals-container", "children"),
    Input("calc-metrics-button", "n_clicks"),
    State("previous-run-dropdown-eval", "value"),
    State("evaluation-store", "data"),
    prevent_initial_call=True
)
def calculate_metrics(n_clicks, selected_run, store_data):
    """
    For each annotated pair => compute top-1, top-5 retrieval success
    Distinguish "landmark_building" vs "standard_building"
    Then produce final metrics + retrieval visualizations.
    """
    if not selected_run:
        return ("No run selected for evaluation.", [])

    # 1) Load run info
    run_folder_path = os.path.join(RESULTS_FOLDER, selected_run)
    meta_path = os.path.join(run_folder_path, "filtered_metadata.csv")
    labels_path = os.path.join(run_folder_path, "cluster_labels.npy")
    param_path = os.path.join(run_folder_path, "params.json")

    if not (os.path.exists(meta_path) and os.path.exists(labels_path) and os.path.exists(param_path)):
        return ("Run data not found.", [])

    with open(param_path, "r") as f:
        cfg = json.load(f)
    conf_threshold = cfg["confidence_threshold"]
    used_layers = cfg["layers"]

    # 2) Rebuild satellite embeddings
    embeddings_all_sat = np.load(EMBEDDINGS_NPY_ARRAY_PATH)
    df_meta_all_sat = pd.read_csv(EMBEDDINGS_METADATA_CSV_PATH)
    mask_sat = df_meta_all_sat["conf"] >= conf_threshold
    df_sat_filtered = df_meta_all_sat[mask_sat].copy().reset_index(drop=True)
    sat_emb_filtered = embeddings_all_sat[mask_sat.values]
    if used_layers:
        sat_emb_filtered = get_embeddings_for_layers(sat_emb_filtered, used_layers, LAYER_TO_DIM)

    # Rebuild drone embeddings
    embeddings_all_drone = np.load(DRONE_EMBEDDINGS_NPY_ARRAY_PATH)
    df_meta_all_drone = pd.read_csv(DRONE_EMBEDDINGS_METADATA_CSV_PATH)
    mask_drone = df_meta_all_drone["conf"] >= conf_threshold
    df_drone_filtered = df_meta_all_drone[mask_drone].copy().reset_index(drop=True)
    dr_emb_filtered = embeddings_all_drone[mask_drone.values]
    if used_layers:
        dr_emb_filtered = get_embeddings_for_layers(dr_emb_filtered, used_layers, LAYER_TO_DIM)

    def l2_dist(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    # 3) Gather all pairs
    all_pairs = []
    for row_data in store_data["rows"]:
        for pair in row_data["annotated_pairs"]:
            # pair = {"sat_id":..., "drone_id":..., "pair_type":...}
            # We'll also keep the image_name if needed
            image_name = row_data["image_name"]
            all_pairs.append({
                "sat_id": pair["sat_id"],
                "drone_id": pair["drone_id"],
                "pair_type": pair["pair_type"],
                "image_name": image_name
            })

    if not all_pairs:
        return ("No pairs annotated yet!", [])

    # 4) Separate by pair_type
    landmark_pairs = [p for p in all_pairs if p["pair_type"] == "landmark_building"]
    standard_pairs = [p for p in all_pairs if p["pair_type"] == "standard_building"]

    # We'll define some helper functions to find the index in sat_emb_filtered / dr_emb_filtered.
    def find_sat_embedding_idx(sat_id):
        # 'sat_id' is the row in the satellite "filtered_metadata.csv" for the run
        df_run = pd.read_csv(meta_path).reset_index(drop=True)
        if sat_id < 0 or sat_id >= len(df_run):
            return None
        # bounding box
        row = df_run.iloc[sat_id]
        image = row["image"]
        b0, b1, b2, b3 = row["box_0"], row["box_1"], row["box_2"], row["box_3"]

        # find the match in df_sat_filtered
        match_idx = df_sat_filtered.index[
            (df_sat_filtered["image"] == image) &
            (df_sat_filtered["box_0"] == b0) &
            (df_sat_filtered["box_1"] == b1) &
            (df_sat_filtered["box_2"] == b2) &
            (df_sat_filtered["box_3"] == b3)
        ].tolist()
        if match_idx:
            return match_idx[0]
        return None

    def find_drone_embedding_idx(dr_id):
        # 'dr_id' is the row in the "df_drone_global" after threshold
        if dr_id < 0 or dr_id >= len(df_drone_filtered):
            return None
        return dr_id

    # Evaluate a list of pairs => returns (top1_acc, top5_acc, retrieval_details)
    def evaluate_pairs(pair_list):
        correct_top1 = 0
        correct_top5 = 0
        details = []
        total = len(pair_list)
        for pair in pair_list:
            sat_embed_idx = find_sat_embedding_idx(pair["sat_id"])
            dr_embed_idx = find_drone_embedding_idx(pair["drone_id"])
            if sat_embed_idx is None or dr_embed_idx is None:
                # can't evaluate
                details.append({
                    "pair": pair,
                    "topk": [],
                    "true_id": None
                })
                continue

            query_vec = dr_emb_filtered[dr_embed_idx]
            # compute L2-dist to all satellite embeddings
            dist_list = []
            for i in range(len(sat_emb_filtered)):
                dist_val = l2_dist(query_vec, sat_emb_filtered[i])
                dist_list.append((dist_val, i))
            dist_list.sort(key=lambda x: x[0])
            top5 = dist_list[:5]
            top5_ids = [x[1] for x in top5]

            is_top1 = (top5_ids[0] == sat_embed_idx)
            is_top5 = (sat_embed_idx in top5_ids)
            if is_top1:
                correct_top1 += 1
            if is_top5:
                correct_top5 += 1

            details.append({
                "pair": pair,
                "topk": top5_ids,  # top-5 retrieved indices
                "true_id": sat_embed_idx
            })

        if total > 0:
            top1_acc = correct_top1 / total
            top5_acc = correct_top5 / total
        else:
            top1_acc = 0
            top5_acc = 0

        return top1_acc, top5_acc, details

    # Evaluate each group
    land_top1, land_top5, land_details = evaluate_pairs(landmark_pairs)
    std_top1, std_top5, std_details = evaluate_pairs(standard_pairs)

    summary_text = (f"Landmark buildings: top-1={land_top1:.2f}, top-5={land_top5:.2f} | "
                    f"Standard buildings: top-1={std_top1:.2f}, top-5={std_top5:.2f}")

    # 5) Build retrieval visualization
    # We'll show "landmark_building" first, then "standard_building"
    retrieval_layout = []
    retrieval_layout += build_retrieval_rows(land_details, "landmark_building", df_sat_filtered, sat_emb_filtered,
                                             df_drone_filtered, dr_emb_filtered)
    retrieval_layout += build_retrieval_rows(std_details, "standard_building", df_sat_filtered, sat_emb_filtered,
                                             df_drone_filtered, dr_emb_filtered)

    return summary_text, retrieval_layout


##########################################
# ADD THIS HELPER FOR THE RETRIEVAL VISUAL
##########################################
def build_retrieval_rows(details, category_name, df_sat_filtered, sat_emb_filtered,
                         df_drone_filtered, dr_emb_filtered):
    """
    For each detail => produce a row with:
      - Drone image bounding the single annotated building
      - True satellite bounding the single building
      - 5 bounding boxes for top-5 retrieved
    """
    out = []
    for det in details:
        row_div = build_single_retrieval_row(det, category_name,
                                             df_sat_filtered, sat_emb_filtered,
                                             df_drone_filtered, dr_emb_filtered)
        out.append(row_div)
    return out


def build_single_retrieval_row(det, category_name, df_sat_filtered, sat_emb_filtered,
                               df_drone_filtered, dr_emb_filtered):
    """
    For a single pair's retrieval result, display:
      1) Drone image bounding the single annotated building
      2) True satellite image bounding the single correct building
      3) 5 bounding-box images for top-5 retrieved satellite results
    """
    pair = det["pair"]              # { "sat_id", "drone_id", "pair_type", "image_name" }
    topk_ids = det["topk"]          # list of 5 sat_embed_idx
    sat_idx_true = det["true_id"]   # correct satellite embedding index (or None)

    style_div = {"display": "inline-block", "margin": "5px", "verticalAlign": "top"}

    # 1) Drone building bounding box
    drone_img = generate_bounding_box_image_drone(pair["drone_id"], df_drone_filtered)

    # 2) The "true" satellite bounding box
    sat_true_img = ""
    if sat_idx_true is not None and sat_idx_true >= 0 and sat_idx_true < len(df_sat_filtered):
        sat_true_img = generate_bounding_box_image_sat(sat_idx_true, df_sat_filtered)

    # 3) The top-5 retrieved satellite bounding boxes
    top5_imgs = []
    for rank, idx in enumerate(topk_ids):
        top5_imgs.append({
            "rank": rank+1,
            "image_div": generate_bounding_box_image_sat(idx, df_sat_filtered),
            "embed_idx": idx
        })

    # Build the row layout
    drone_div = html.Div([
        html.H5(f"Drone building, type={category_name}"),
        html.P(f"drone_id={pair['drone_id']}"),
        drone_img  # the actual bounding box image
    ], style=style_div)

    sat_true_div = html.Div([
        html.H5("True Satellite building"),
        html.P(f"sat_embed_idx={sat_idx_true if sat_idx_true is not None else 'N/A'}"),
        sat_true_img
    ], style=style_div)

    top5_divs = []
    for item in top5_imgs:
        top5_divs.append(html.Div([
            html.P(f"Rank {item['rank']} => sat_embed_idx={item['embed_idx']}"),
            item["image_div"]
        ], style={"margin": "5px", "textAlign": "center"}))

    top5_container = html.Div(top5_divs, style={"display": "inline-block", "verticalAlign": "top"})

    return html.Div([
        html.Hr(),
        drone_div,
        sat_true_div,
        top5_container
    ], style={"border": "1px solid #ccc", "marginBottom": "10px", "padding": "5px"})


###########################################
# HELPER: generate drone bounding box
###########################################
def generate_bounding_box_image_drone(dr_id, df_drone_filtered):
    """
    Given a drone embedding index (dr_id) in the thresholded df_drone_filtered,
    build a bounding-box image for that single building only.
    Returns an html.Img Div with base64-encoded data.
    """
    if dr_id < 0 or dr_id >= len(df_drone_filtered):
        return html.Div("Drone building index out of range")

    row = df_drone_filtered.iloc[dr_id]
    image_name = row["image"]
    box = (row["box_0"], row["box_1"], row["box_2"], row["box_3"])

    image_path = os.path.join(DRONE_IMAGES_ROOT_FOLDER_PATH, image_name)
    if not os.path.exists(image_path):
        return html.Div([html.P(f"Drone image not found: {image_path}")])

    # Single bounding box
    instance_info = [{
        "box": box,
        "instance_id": dr_id,
        # If you want to color outliers, you'd check label, but for simplicity just label=0
        "label_val": 0
    }]
    annotated = draw_boxes_on_image(image_path, instance_info)
    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode("utf-8")

    return html.Img(src=f"data:image/png;base64,{data}", style={"maxWidth": "200px"})


###########################################
# HELPER: generate satellite bounding box
###########################################
def generate_bounding_box_image_sat(sat_idx, df_sat_filtered):
    """
    Given a satellite embedding index (sat_idx) in the thresholded df_sat_filtered,
    build a bounding-box image for that single building only.
    Returns an html.Img Div with base64-encoded data.
    """
    if sat_idx < 0 or sat_idx >= len(df_sat_filtered):
        return html.Div("Satellite building index out of range")

    row = df_sat_filtered.iloc[sat_idx]
    image_name = row["image"]
    box = (row["box_0"], row["box_1"], row["box_2"], row["box_3"])
    label_val = 0  # or -1 if you want to check something else

    image_path = os.path.join(IMAGES_ROOT_FOLDER_PATH, image_name)
    if not os.path.exists(image_path):
        return html.Div([html.P(f"Satellite image not found: {image_path}")])

    instance_info = [{
        "box": box,
        "instance_id": sat_idx,
        "label_val": label_val
    }]
    annotated = draw_boxes_on_image(image_path, instance_info)
    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode("utf-8")

    return html.Img(src=f"data:image/png;base64,{data}", style={"maxWidth": "200px"})


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Parameters for prediction')
    parser.add_argument('-p', '--port', type=int, required=False, default=8050)
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    logging.info("Starting Dash server...")
    app.run_server(debug=True, port=args.port)