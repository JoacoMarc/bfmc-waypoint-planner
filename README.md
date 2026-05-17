# BFMC Waypoint Route Planner

Script standalone para BFMC 2026: calcula la mejor ruta para pasar por la mayor cantidad de waypoints (dots de scoring) en 10 minutos, empezando desde el random start area. Modela el problema como un Orienteering Problem y genera un SVG con la ruta + JSON con la secuencia de waypoints y lanelets.

## Instalación

```bash
cd bfmc-waypoint-planner
pip install -r requirements.txt
```

## Uso

```bash
python tools/plan_bfmc_route.py \
  --osm data/lanelet2_map_FINAL_RandomStartingArea.osm \
  --time-budget 600 \
  --output-json data/outputs/bfmc_plan.json \
  --output-svg data/outputs/bfmc_plan.svg
```

Flags principales:
- `--osm PATH`: Archivo OSM Lanelet2 (con waypoints como `multipolygon` y start area como `way area=yes`).
- `--time-budget SECONDS`: Presupuesto de tiempo total (default 600).
- `--speed-urban M/S`: Velocidad urbana (default 0.2).
- `--speed-highway M/S`: Velocidad highway (default 0.4).
- `--efficiency FACTOR`: Factor de eficiencia 0-1 (default 0.7).
- `--penalty-stopline SECONDS`: Penalty por stopline (default 4.0).
- `--penalty-intersection SECONDS`: Penalty por intersección (default 1.5).
- `--start-pose AUTO|x,y,yaw`: Pose inicial. `AUTO` calcula el mejor; o pasar `--start-pose -7.5,-3.0,1.57`.
- `--solver greedy|sa|hybrid`: Solver (default `hybrid`).
- `--seed N`: Seed RNG (default 42).
- `--verbose`: Logs detallados.

## Outputs

- `data/outputs/bfmc_plan.svg`: SVG con track, waypoints (azul si visitados, gris si descartados), ruta dibujada, start area y pose inicial.
- `data/outputs/bfmc_plan.json`: Plan estructurado con `waypoint_sequence` (orden de visita) y `lanelet_sequence` (lanelet IDs en orden).

## Estructura

```
bfmc-waypoint-planner/
├── src/                          # Módulos
│   ├── osm_parser.py             # Parser OSM Lanelet2
│   ├── geometry.py               # Utilidades geométricas
│   ├── topology.py               # Centerlines + sucesores
│   ├── waypoint_anchor.py        # Anclar waypoints a lanelets
│   ├── cost_matrix.py            # Dijkstra all-pairs + cache
│   ├── orienteering_solver.py    # Greedy + 2-opt + Or-opt + SA
│   ├── plan_writer.py            # JSON output
│   └── visualizer_svg.py         # SVG output
├── tools/plan_bfmc_route.py      # CLI
├── tests/                        # Tests pytest
└── data/                         # OSM + outputs
```
