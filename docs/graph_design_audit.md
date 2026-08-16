# Graph Design Audit Before Dynamic Wind Edges

This audit finalizes the graph design contract before implementing
dynamic wind edges. It reviews current `main` graph scripts and the
`Nirika-work` graph scripts 01-07 without merging that branch.

## Corrected Static Foundation Status

Graph scripts 01-04 have now been corrected and the static graph
foundation has been regenerated from the sensor-qualified node registry.
The first raw dynamic wind-edge stage is also implemented, while graph
snapshots and GNN training remain unimplemented.

Validation results:

```text
canonical nodes: 56
unique dataset_name: true
unique pm25_sensor_id: true
model-usable train+validation nodes: 51
missing coordinates: 0

distance matrix: 56x56, symmetric, zero diagonal
complete undirected distance edges: 1,540

bearing matrix: 56x56, directed
complete directed bearing edges: 3,080

KNN K: 5
static undirected candidate pairs: 188
static directed candidate edges: 376
adjacency directed edges: 376
static edge rows: 376
candidate pairs missing reverse direction: 0
```

## Dynamic Wind Edge Implementation Status

The first dynamic wind-edge stage has now been implemented in:

```text
scripts/graph/05_dynamic_edge_weights.py
```

It uses the corrected `static_graph.csv` candidate edge set,
`station_mapping.csv` model-usable flags, and hourly source-node wind
from `data/processed/featured/`. It does not build graph snapshots,
sliding windows, or any GNN model.

Generated outputs:

```text
data/processed/graph/dynamic_edge_weights.csv
data/processed/graph/dynamic_edge_weights_summary.csv
data/processed/graph/dynamic_edge_weights_validation.csv
data/processed/graph/dynamic_supervised_degree.csv
data/processed/graph/dynamic_reverse_direction_check.csv
```

Validation results:

```text
lambda_d: 1.930 km
timestamp count: 47,988
total rows: 2,919,724
candidate edges: 376
supervised candidate edges: 326
active-edge percentage: 47.522%
zero-weight percentage: 52.478%
missing-wind percentage: 0.000%
calm-wind percentage: 1.867%
all validation checks passed: true
```

Supervised 51-node subgraph after filtering `supervised_edge=True`:

```text
min out-degree: 4
median out-degree: 6
max out-degree: 9
isolated nodes: 0
```

The lowest-degree supervised node is
`Tarakeswor (SC-14)-GD Labs`, with out-degree 4 and in-degree 4. This is
not problematic enough to change KNN silently.

## Scope

Reviewed scripts:

```text
scripts/graph/01_station_mapping.py
scripts/graph/02_distance_matrix.py
scripts/graph/03_bearing_matrix.py
scripts/graph/04_static_graph.py
scripts/graph/05_dynamic_edge_weights.py
scripts/graph/06_graph_snapshots.py
scripts/graph/07_sliding_windows.py
```

Findings:

- `Nirika-work` scripts 01-04 match `main`.
- Scripts 05-07 are empty placeholders in both `main` and
  `Nirika-work`.
- Do not merge `Nirika-work` as-is.

## Reproducible Audit

A documentation/audit helper was added:

```text
scripts/22_graph_design_audit.py
scripts/analysis/graph_design_audit.py
```

It writes ignored audit outputs under:

```text
data/processed/graph/design_audit/
```

## Original Node Identity Finding

Current canonical/featured data is sensor-qualified where needed. The
graph must therefore not use human station name alone as identity.

Audit summary:

```text
metadata rows: 56
unique human station names: 54
unique PM2.5 sensors: 56
featured datasets: 56
model-usable train+validation datasets: 51
current station_mapping nodes: 54
```

The mismatch comes from two issues:

- three Kathmandu University PM2.5 sensors share the same human station
  name and are collapsed by the current `StationMapper`;
- current mapping uses raw station names, while featured dataset names
  use sanitized names and sensor-qualified names for duplicates.

The old `station_mapping.csv` was therefore not safe for graph modeling.
It dropped distinct PM2.5 sensors and could not join one-to-one with
featured dataset files.

Implemented policy:

- maintain a canonical 56-row graph node registry keyed by
  `dataset_name`, not by human `station`;
- include `node_id`, `dataset_name`, human `station`, `location_id`,
  `pm25_sensor_id`, `latitude`, and `longitude`;
- for the first supervised graph model, use the 51 model-usable nodes
  that have train and validation datasets;
- keep the five non-model-usable featured nodes in the canonical
  registry but exclude them from the first supervised training graph
  until they have usable train/validation sequences.

The five currently non-model-usable featured nodes are:

```text
Kathmandu University__sensor_15286458
Kathmandu University__sensor_15286975
Kathmandu University__sensor_15286980
Pulchowk (SC-15)-GD Labs
Tarakeswor (SC-15)- GD Labs
```

## Coordinate Mapping

The recommended 56-row node registry is one-to-one:

```text
dataset_name unique: true
pm25_sensor_id unique: true
missing coordinates: 0
dataset names with multiple coordinate pairs: 0
```

Node IDs must be assigned by sorted `dataset_name` or another explicitly
documented deterministic order. The order must not depend on filesystem
iteration order.

## Distance And Bearing

The original formulas were correct for the old 54-node mapping:

```text
distance matrix shape: 54x54
distance edges: 1431 / expected undirected complete edges 1431
distance matrix symmetric: true
distance diagonal zero: true
max distance recalculation error: 0.0 km

bearing matrix shape: 54x54
bearing edges: 2862 / expected directed complete edges 2862
bearing diagonal zero: true
max bearing recalculation error: 0.0 degrees
max reverse-bearing 180-degree error: 0.13 degrees
```

The small reverse-bearing error is expected from spherical geometry and
rounding; reverse initial bearings are approximately but not exactly
180 degrees apart.

Implemented correction before dynamic edges:

- recompute distance and bearing matrices from the corrected
  sensor-qualified node registry;
- keep bearings directed as A to B;
- keep distances symmetric.

## Static KNN Graph

The old static graph used `K_NEIGHBORS = 5` and symmetrized the
adjacency matrix.

Audit summary for current 54-node artifacts:

```text
static edge rows: 270
adjacency directed edges after symmetrization: 362
adjacency undirected edges: 181
static undirected pairs: 181
static missing reverse rows: 92
adjacency directed edges missing from static edge rows: 92
```

So the current adjacency and edge CSV do not represent the same directed
edge set. The CSV stores each source's original five nearest outgoing
neighbors, while the adjacency stores the symmetric union.

Implemented candidate-edge representation:

- build an undirected candidate pair set from the symmetric union of KNN
  neighbors;
- expand every undirected pair `{A, B}` into two directed candidates:
  `A -> B` and `B -> A`;
- store both directed rows with their own A-to-B bearing;
- dynamic wind weights should be computed on this directed candidate
  edge table, not on the current one-way static CSV.

For the corrected 56-node canonical graph, this produces 188 undirected
candidate pairs and 376 directed candidate rows.

## Dynamic Wind Edge Formula

Meteorological wind direction is the direction from which wind blows.
For transport, convert it to the direction toward which air is moving:

```text
transport_direction_A(t) = (wind_direction_A(t) + 180) % 360
```

For candidate directed edge `A -> B` at timestamp `t`:

```text
theta_AB = bearing from source A to target B, degrees
v_A(t) = wind_speed at source A, km/h
d_AB = distance from A to B, km
delta_AB(t) = min(
    abs(transport_direction_A(t) - theta_AB),
    360 - abs(transport_direction_A(t) - theta_AB)
)

alignment_AB(t) = max(0, cos(delta_AB(t) in radians))
speed_factor_A(t) = v_A(t) / (v_A(t) + 5)
distance_factor_AB = exp(-d_AB / lambda_d)

raw_weight_AB(t) =
    candidate_AB
    * alignment_AB(t)
    * speed_factor_A(t)
    * distance_factor_AB
```

Where:

- `candidate_AB` is 1 only if `A -> B` is in the directed candidate edge
  table;
- `alignment_AB(t)` is 1 when transport points exactly from A to B, 0
  when wind is perpendicular or pointing away, and never negative;
- `speed_factor_A(t)` smoothly downweights calm wind and saturates as
  wind becomes stronger;
- `5 km/h` is a fixed moderate-wind scale, not a tuned hyperparameter;
- `distance_factor_AB` downweights longer candidate edges;
- `lambda_d` should be the median distance of the directed candidate
  edge table, computed once from static geography after the node set is
  fixed.

Optional model input normalization:

```text
dynamic_weight_AB(t) =
    raw_weight_AB(t) / sum_B raw_weight_AB(t)
```

Use this row normalization only if the downstream GNN expects outgoing
edge weights to sum to one. Otherwise keep `raw_weight` and let the model
consume it directly. In either case, save the raw components so the
weight is auditable.

## Whose Wind Controls A To B?

Use source-node wind: `wind_speed_A(t)` and `wind_direction_A(t)`.

Reason: the edge `A -> B` represents possible transport of pollution
leaving source A toward target B. Source wind is the most defensible
local proxy for whether air at A is moving toward B. Target wind may be
useful later as a node feature or secondary modifier, but it should not
control the primary transport direction for `A -> B`.

## Missing And Edge Cases

Near-zero wind:

- if `wind_speed_A(t) < 0.5 km/h`, set `raw_weight_AB(t) = 0`;
- keep the edge row with a `calm_wind` flag so missing and calm are not
  confused.

Wind points away from B:

- if `delta_AB(t) >= 90 degrees`, `alignment_AB(t) = 0`, so the dynamic
  edge weight is 0.

PM2.5 missing at a node:

- PM2.5 missing does not prevent computing the wind edge itself;
- node features and supervised targets must carry masks;
- do not impute PM2.5 just to keep a graph snapshot;
- supervised loss should include only nodes with valid target PM2.5.

Weather missing:

- if source wind speed or direction is missing, set dynamic weight to
  null or 0 with a `missing_source_wind` flag;
- prefer retaining the row plus flags over silently dropping edges,
  because dropped edges can change graph shape across time.

Stations do not share exactly the same usable timestamps:

- build graph snapshots on a global hourly timestamp index;
- require weather availability for dynamic edges at that timestamp;
- use node masks for missing PM2.5/current features/targets;
- for the first graph model, restrict supervised training/evaluation to
  timestamps where the selected 51-node cohort has the required model
  features and target policy defined, or use explicit node-level masks.

## Expected Dynamic Edge Schema

Recommended row schema:

```text
timestamp
source_node_id
target_node_id
source_dataset_name
target_dataset_name
distance_km
bearing_deg
source_wind_speed
source_wind_direction_from_deg
source_transport_direction_deg
alignment_angle_deg
alignment
speed_factor
distance_factor
raw_dynamic_weight
dynamic_weight
calm_wind
missing_source_wind
edge_active
```

The implemented dynamic edge stage keeps the raw components and
`raw_dynamic_weight` only. Row-normalized `dynamic_weight` is deliberately
not emitted yet; add it later only if the downstream GNN layer requires
outgoing edge weights to sum to one.

## Graph Snapshot Implementation Status

`scripts/graph/06_graph_snapshots.py` now builds the first supervised
graph snapshot synchronization artifacts from:

```text
data/metadata/station_mapping.csv
data/processed/featured/
data/processed/graph/dynamic_edge_weights.csv
```

The snapshot stage uses the 51 `model_usable` nodes and preserves their
canonical node IDs from the 56-node registry. It does not renumber nodes.
It also does not impute missing values, row-normalize edges, build
sliding windows, or train a graph model.

Node features at timestamp `t`:

```text
pm2_5
hour_sin
hour_cos
month_sin
month_cos
temperature
humidity
pressure
dew_point
wind_u
wind_v
```

Target:

```text
residual_pm25(t+1) = pm2_5(t+1) - pm2_5(t)
```

Every node row records `node_exists`, `input_valid`,
`target_exists_t_plus_1`, `target_valid`, and
`snapshot_supervised_usable`. The target-valid mask excludes the three
global chronological split-boundary crossings where `t+1` would fall in a
different split.

Generated compact artifacts:

```text
data/processed/graph/snapshots/supervised_nodes.csv
data/processed/graph/snapshots/snapshot_nodes.csv.gz
data/processed/graph/snapshots/snapshot_edges.csv.gz
data/processed/graph/snapshots/snapshot_timestamp_summary.csv
data/processed/graph/snapshots/snapshot_policy_summary.csv
data/processed/graph/snapshots/snapshot_validation.csv
data/processed/graph/snapshots/snapshot_valid_node_distribution.csv
data/processed/graph/snapshots/snapshot_continuous_runs.csv
```

Synchronization result:

```text
global hourly timestamps: 47,987
node snapshot rows: 2,447,337
edge snapshot rows: 2,659,101
strict usable timestamps: 0
masked usable timestamps: 30,067
masked node-target sequences: 201,608
```

The strict policy requiring all 51 nodes to have valid inputs and valid
t+1 targets is therefore not viable. The recommended first GNN dataset
policy is the masked fixed-graph policy: keep all 51 node IDs in every
snapshot and use explicit input/target masks for learning and
evaluation.

Coverage:

```text
valid input nodes per timestamp: min 0, median 1, max 43
valid target nodes per timestamp: min 0, median 1, max 43
timestamps with 51 valid input nodes: 0
timestamps with >=45 valid input nodes: 0
timestamps with >=40 valid input nodes: 47
timestamps with >=30 valid input nodes: 1,921
```

Validation passed: global timestamps are hourly, targets are exactly
t+1, fixed 51-node identity is preserved, no future node features are
used, every dynamic edge is a supervised static candidate, source/target
edge IDs map to the correct nodes, and raw dynamic weights remain
unchanged and non-negative.

## Code Corrected Before Dynamic Edges

1. `scripts/graph/01_station_mapping.py`

   Corrected to preserve duplicate human station names by using the
   canonical sensor-qualified `dataset_name` identity.

2. `scripts/graph/02_distance_matrix.py`

   Corrected to read the sensor-qualified mapping and preserve
   `dataset_name`/sensor identity in edge outputs.

3. `scripts/graph/03_bearing_matrix.py`

   Corrected to read the sensor-qualified mapping and output directed
   bearings for sensor-qualified node IDs.

4. `scripts/graph/04_static_graph.py`

   Corrected so adjacency and static edge CSV represent the same directed
   expansion of the symmetric KNN union.

5. `scripts/graph/05_dynamic_edge_weights.py`

   Implemented. It computes raw source-wind-controlled dynamic edge
   weights over the corrected directed static candidates.

6. `scripts/graph/06_graph_snapshots.py`

   Implemented. It constructs fixed 51-node graph snapshots with
   explicit node/target masks and supervised dynamic edge attachments.

7. `scripts/graph/07_sliding_windows.py`

   Still pending. It should remain unimplemented until the masked graph
   snapshot dataset loader/batching design is reviewed.

## Final Recommendation

Dynamic wind edges and graph snapshot synchronization are now
implemented on the corrected static foundation. The station-name mapping
has been replaced with a sensor-qualified canonical node registry,
distance/bearing/static candidate edges have been regenerated, and the
static edge CSV and adjacency matrix describe the same directed
candidate set.

For the first graph model, use the 51 train+validation model-usable nodes
as the supervised node set while preserving the full 56-node canonical
registry for reproducibility and future data expansion. Use the masked
fixed-graph snapshot policy, because strict all-node synchronization
produces zero usable timestamps.
