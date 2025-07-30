# Unsupervised Knowledge Extraction of Distinctive Landmarks from Earth Imagery Using Deep Feature Outliers for Robust UAV Geo-Localization

This project provides the necessary code and resources to reproduce the research presented in the paper "Unsupervised Knowledge Extraction of Distinctive Landmarks from Earth Imagery Using Deep Feature Outliers for Robust UAV Geo-Localization". The goal of this research is to extract distinctive landmarks from Earth imagery for robust UAV geo-localization using deep feature outliers.

## Project Structure

The project is organized as follows:

* `1_create_and_visualise_embeddings.ipynb`: This Jupyter notebook is used for creating and visualizing embeddings from the Earth imagery. It's a foundational step to understand the feature representation.
* `2_visualise_retrieved_buildings_for_article.ipynb`: This notebook focuses on visualizing the retrieved buildings, which is crucial for the analysis and presentation of results in the accompanying article.
* `3_calculate_proxy_metrics_to_select_best_embeddings.ipynb`: This notebook contains the code to calculate various proxy metrics. These metrics are used to evaluate and select the best performing embeddings for the given task.
* `interactive_experiment_runner_and_explorer.py`: A Python script that provides an interactive interface to run experiments and explore the results. This is particularly useful for dynamic analysis and visualization.
* `model/best.pt`: This directory contains the pre-trained model weights (`best.pt`) used in the project. These weights are essential for running the inference and reproducing the results.
* `requirements.txt`: Lists all the Python dependencies required to run the project.
* `data/`: This directory is intended to store the `vpair` dataset, which is necessary for running some parts of the project, especially for visualizing buildings.

## Setup and Installation

To set up the project environment, follow these steps:

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/your-repository/uav-landmarks-embeddings.git
    cd uav-landmarks-embeddings
    ```

    (Note: Replace `https://github.com/your-repository/uav-landmarks-embeddings.git` with the actual repository URL if this is a public repository, otherwise, this step is for illustrative purposes.)

2. **Create a Conda Environment (Recommended):**
    It is highly recommended to use a virtual environment to manage dependencies. If you don't have Anaconda or Miniconda installed, please install it first.

    ```bash
    conda create -n uav_env python=3.9 # Or your preferred Python version
    conda activate uav_env
    ```

3. **Install Dependencies:**
    Install all the required Python packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

## Data Preparation

To visualize buildings and run certain parts of the analysis, you need to download the `vpair` dataset and place it within a `data` folder in the project root.

1. **Create the `data` directory:**

    ```bash
    mkdir data
    ```

2. **Download `vpair` dataset:** (Please provide specific instructions for downloading the `vpair` dataset here, as it's not publicly available through a simple command.)
    Once downloaded, ensure the dataset structure within the `data` directory is correctly set up as expected by the notebooks.

## Running the Project

### Jupyter Notebooks

You can run the Jupyter notebooks to follow the research steps interactively.

1. **Start Jupyter Lab/Notebook:**

    ```bash
    jupyter lab
    # or
    jupyter notebook
    ```

    Your web browser will open with the Jupyter interface.

2. **Navigate and Run:**
    Open the `.ipynb` files (`1_create_and_visualise_embeddings.ipynb`, `2_visualise_retrieved_buildings_for_article.ipynb`, `3_calculate_proxy_metrics_to_select_best_embeddings.ipynb`) and execute the cells in sequence.

### Interactive Experiment Runner and Explorer

To run the interactive experiment runner and explorer:

```bash
python -m interactive_experiment_runner_and_explorer
```

This will launch an interactive application, likely a Dash application given the dependencies (`dash`, `dash-bootstrap-components`).

## Reproducibility

By following the steps outlined above, you should be able to reproduce the experimental results and analyses presented in the research paper "Unsupervised Knowledge Extraction of Distinctive Landmarks from Earth Imagery Using Deep Feature Outliers for Robust UAV Geo-Localization".
