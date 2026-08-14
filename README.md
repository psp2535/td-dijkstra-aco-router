# IM-VRM Reproduction Project (Phase 1 Baseline)

This repository contains a modular, lightweight Python project structure for reproducing the baseline preprocessing pipeline of the **IM-VRM** (*Intelligent Multi-Depot Vehicle Routing and Management*) model using the Microsoft **T-Drive** taxi trajectory dataset.

The IM-VRM framework was introduced in:
> *"An Intelligent Multi-Depot Vehicle Routing and Management Model for Smart Cities,"*  
> **IEEE Transactions on Intelligent Transportation Systems** (Volume 26, Issue 6, June 2025).

---

## 📖 Theoretical Primitives

This phase implements the exact data cleaning, kinematic modeling, trip segmentation, spatial binned congestion profiling, grid transition graph construction, and fleet metadata creation required for the routing model.

### 1. Geospatial & Kinematic Calculations
For consecutive GPS logs $P_1(\text{lat}_1, \text{lon}_1)$ and $P_2(\text{lat}_2, \text{lon}_2)$ recorded at times $t_1$ and $t_2$:
*   **Geodetic Distance ($\Delta d$):** Calculated using the Haversine formula:
    $$a = \sin^2\left(\frac{\Delta \text{lat}}{2}\right) + \cos(\text{lat}_1)\cos(\text{lat}_2)\sin^2\left(\frac{\Delta \text{lon}}{2}\right)$$
    $$\Delta d = 2 R \arcsin(\sqrt{a})$$
    where $R = 6,371,000\text{ meters}$.
*   **Velocity ($v$):** Computed as $v = \Delta d / \Delta t$, where $\Delta t = t_2 - t_1$. If $v > v_{\text{limit}}$ (e.g., 120 km/h), the destination point is filtered out as a GPS anomaly.
*   **Acceleration ($a$):** Computed as $a = \frac{v_2 - v_1}{\Delta t}$.

### 2. Green Emission & Fuel Estimation
To model ecological routing costs, we implement a physics-based load-dependent fuel consumption model:
*   **Payload Correction:** 
    $$F_{\text{adj}} = F_{\text{base}} \times \left(1 + \lambda_{\text{load}} \times \frac{\text{Payload}_{\text{kg}}}{100}\right)$$
    where $F_{\text{base}}$ is the base fuel rate (L/100km) and $\lambda_{\text{load}}$ is the load penalty factor.
*   **Speed Efficiency Curve:** Vehicles consume fuel less efficiently in stop-and-go conditions or at extreme speeds due to aerodynamic drag. We scale consumption by a U-shaped efficiency multiplier $\eta(v)$:
    *   $\eta(v) = 2.0$ for $v < 10\text{ km/h}$ (crawl/congestion)
    *   $\eta(v) = 1.4$ for $10 \le v < 30\text{ km/h}$ (city traffic)
    *   $\eta(v) = 1.0$ for $30 \le v < 60\text{ km/h}$ (optimum cruise range)
    *   $\eta(v) = 1.25$ for $60 \le v < 90\text{ km/h}$ (highway speed)
    *   $\eta(v) = 1.6$ for $v \ge 90\text{ km/h}$ (drag-dominated)
*   **Step Consumption (L):** $\text{Fuel} = \frac{F_{\text{adj}}}{100} \times \left(\frac{\Delta d}{1000}\right) \times \eta(v)$ (with idle consumption applied when $v \approx 0$).
*   **Carbon Emissions (g):** $\text{CO}_2 = \text{Fuel} \times \text{CO}_{2,\text{per\_liter}}$.

### 3. Spatial Grid Congestion Labeling
Beijing is binned into an $N \times M$ grid.
*   **Free Flow Speed ($V_{\text{ff}}$):** Defined per grid cell as the 90th percentile of all speeds observed in that cell.
*   **Congestion Ratio ($R_{\text{cong}}$):** For any hour of the day $h$, the ratio is $R_{\text{cong}} = V_{\text{avg}}(h) / V_{\text{ff}}$.
*   **Discrete Levels:**
    *   **Level 0 (FreeFlow):** $R_{\text{cong}} \ge 0.7$
    *   **Level 1 (Moderate):** $0.4 \le R_{\text{cong}} < 0.7$
    *   **Level 2 (Congested):** $R_{\text{cong}} < 0.4$

### 4. Grid Transition Graph
Nodes correspond to visited grid cells $(r, c)$. A directed edge is created from $(r_1, c_1) \to (r_2, c_2)$ if a taxi actually transitions between the two cells in consecutive time steps. Edges are weighted by transition counts and average transition speeds.

---

## 📁 Repository Structure

```
T-drive Taxi Trajectories/
├── config/
│   └── default_config.yaml     # Bounding boxes, thresholds, grid sizes, vehicle parameters
├── data/
│   ├── raw/                    # Data directory for raw T-Drive .txt files
│   └── processed/              # Processed grid statistics, clean trajectories, trip segments
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Basic trajectory cleaning and stats exploration
│   └── 02_grid_analysis.ipynb     # Spatial grid binning and congestion analysis
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── cleaning.py         # Parsing raw logs, geospatial bounds, speed threshold filters
│   │   └── segmentation.py     # Splitting trajectories into individual trips
│   ├── features/
│   │   ├── __init__.py
│   │   ├── spatial_grid.py     # Map coordinates to NxM grids, aggregate speeds per cell
│   │   ├── metrics.py          # Delta distance, delta time, velocity, and emissions calculation
│   │   └── vehicle.py          # Generator for synthetic multi-depot fleet details & capacity metadata
│   ├── graph/
│   │   ├── __init__.py
│   │   └── grid_graph.py       # Construct spatial cell nodes and transition edges
│   └── utils/
│       ├── __init__.py
│       └── helpers.py          # Haversine distance, YAML config loader, logging setup
├── outputs/                    # Output directory for logs, grid speed matrices, and visualizations
├── main.py                     # Entry point for baseline processing pipeline
├── requirements.txt            # Minimal dependencies (numpy, pandas, matplotlib, pyyaml, tqdm)
└── README.md                   # Instructions for data setup, pipeline stages, and reproduction
```

---

## ⚙️ Setup and Installation

1.  **Clone the workspace** and ensure Python 3.8+ is installed.
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Setup Dataset:**  
    By default, the pipeline expects the raw T-Drive trajectory text files inside the directory specified in `config/default_config.yaml` (default: `release/taxi_log_2008_by_id`). Make sure the raw `.txt` files are located there.

---

## 🚀 Running the Pipeline

You can run the full pipeline or isolate specific stages using the `--step` argument.

### 1. Run the Entire Preprocessing & Modeling Baseline
Executes all processing, congestion labeling, graph transitions, and synthetic fleet generation for the default subset of 50 taxis:
```bash
python main.py --step all --num-taxis 50
```

### 2. Run Individual Steps
*   **Step 1: Trajectory Cleaning, Trip Segmentation & Kinematic features**
    ```bash
    python main.py --step preprocess --num-taxis 50
    ```
    Outputs processed data to `data/processed/clean_trips.csv`.

*   **Step 2: Calculate Grid-based Congestion Levels**
    ```bash
    python main.py --step congestion
    ```
    Outputs congestion profiles to `data/processed/grid_congestion_stats.csv`.

*   **Step 3: Construct Grid transition topology**
    ```bash
    python main.py --step graph
    ```
    Outputs graph connectivity to `data/processed/graph_nodes.json` and `data/processed/graph_edges.csv`.

*   **Step 4: Generate Synthetic Delivery Fleet Profiles**
    ```bash
    python main.py --step fleet
    ```
    Outputs depot coordinates and vehicle configurations to `data/processed/synthetic_fleet.json`.

---

## 🧪 Pipeline Verification

To guarantee correctness, run the programmatic test suite:
```bash
python verify_pipeline.py
```
This runs the pipeline on a small 5-taxi subset and automatically checks the output schema matching, row sanity, and directory structure.
