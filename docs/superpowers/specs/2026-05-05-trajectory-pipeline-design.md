# Design: Trajectory Pipeline — vecraspy + mapgod

**Datum:** 2026-05-05

## Überblick

Zwei-Paket-Lösung für den Workflow: Punkte aus GeoPackage lesen → nach ID gruppieren → Trajektorien bauen → auf Raster plotten (statisch oder animiert).

- **vecraspy** übernimmt die funktionale Datenverarbeitung (Lesen, Gruppieren, Strukturieren).
- **mapgod** übernimmt die Visualisierung (statischer Plot, Animation).
- Die Pakete sind über den `Trajectory`-Dataclass verbunden — kein gegenseitiger Import.

---

## vecraspy: `src/vecraspy/vector.py`

### Datenstruktur

```python
@dataclass
class Trajectory:
    id: Any              # Wert aus der ID-Spalte, oder None
    points: GeoDataFrame # geordnete Punkte, original CRS erhalten
```

### Funktionen

```python
def read_points(
    path: Path | str,
    layer: str | None = None,
) -> GeoDataFrame
```
Liest einen Point-Layer aus einem GeoPackage via `geopandas.read_file`. `layer` wird an geopandas weitergegeben (optional, für Multi-Layer-gpkg).

```python
def group_by_id(
    gdf: GeoDataFrame,
    id_col: str,
) -> dict[Any, GeoDataFrame]
```
Splittet ein GeoDataFrame nach eindeutigen Werten in `id_col`. Gibt ein Dict zurück: `{id_value: GeoDataFrame}`.

```python
def build_trajectory(
    gdf: GeoDataFrame,
    sort_col: str | None = None,
) -> Trajectory
```
Erstellt eine `Trajectory` aus einem GeoDataFrame. Wenn `sort_col` angegeben, werden Zeilen vorher nach dieser Spalte aufsteigend sortiert. `id` wird aus `gdf.attrs.get("id")` gezogen, sonst `None`.

```python
def build_trajectories(
    path: Path | str,
    id_col: str | None = None,
    sort_col: str | None = None,
    layer: str | None = None,
) -> list[Trajectory]
```
Convenience-Funktion: kombiniert `read_points → group_by_id → build_trajectory` in einem Aufruf. Wenn `id_col=None`, wird der gesamte Datensatz als eine einzige Trajektorie behandelt.

### Exports (`__init__.py`)

`Trajectory`, `build_trajectories` werden öffentlich exportiert. Die Einzelfunktionen (`read_points`, `group_by_id`, `build_trajectory`) sind importierbar aber nicht im `__all__`.

---

## mapgod: `src/mapgod/trajectory.py`

### Funktionen

```python
def plot_trajectory(
    raster: Path | str,
    trajectories: Trajectory | list[Trajectory],
    *,
    band: int = 1,
    cmap: str = "gray",
    point_style: dict | None = None,
    line_style: dict | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 8),
) -> tuple[Figure, Axes]
```
Statischer Plot: GeoTIFF als Hintergrund (via `rasterio`), Trajektorie(n) als Linie + markierter Endpunkt. Mehrere Trajektorien werden in unterschiedlichen Farben geplottet (matplotlib color cycle). Gibt `(fig, ax)` zurück für weitere Anpassungen.

```python
def animate_trajectory(
    raster: Path | str,
    trajectories: Trajectory | list[Trajectory],
    *,
    band: int = 1,
    cmap: str = "gray",
    point_style: dict | None = None,
    line_style: dict | None = None,
    interval: int = 100,
    output: Path | str | None = None,
    figsize: tuple[float, float] = (10, 8),
    title: str | None = None,
    fps: int = 10,
) -> FuncAnimation
```
Animation via `matplotlib.animation.FuncAnimation`. Pro Frame wird ein Punkt mehr aufgedeckt: bisherige Punkte als Linie, aktueller Punkt als Marker. Bei mehreren Trajektorien laufen alle parallel — Frameanzahl = Länge der längsten Trajektorie, kürzere Trajektorien bleiben am letzten Punkt stehen. `output=None` zeigt die Animation direkt. `.gif` und `.mp4` werden als Ausgabeformat unterstützt (Pillow für GIF, ffmpeg für MP4). Gibt das `FuncAnimation`-Objekt zurück.

### CRS-Handling

`mapgod` reprojiziiert die Trajektorienpunkte intern in das CRS des Rasters (via `geopandas.to_crs`), bevor Koordinaten extrahiert werden. Kein manuelles CRS-Management nötig.

### Exports (`__init__.py`)

`plot_trajectory`, `animate_trajectory` werden öffentlich exportiert.

---

## Abhängigkeiten

| Paket | Neue Abhängigkeit |
|-------|------------------|
| vecraspy | `geopandas` (bereits vorhanden) |
| mapgod | `rasterio`, `matplotlib`, `geopandas` (für CRS-Reprojizierung) |

`mapgod` importiert `Trajectory` aus `vecraspy` zur Typ-Annotation (optional, kann auch als `TYPE_CHECKING`-only Import bleiben um zirkuläre Abhängigkeiten zu vermeiden).

---

## Beispiel-Workflow

```python
from vecraspy import build_trajectories
from mapgod import animate_trajectory

trajectories = build_trajectories(
    "tracks.gpkg",
    id_col="track_id",
    sort_col="timestamp",
)

animate_trajectory(
    "dem.tif",
    trajectories,
    output="tracks.gif",
    interval=80,
)
```
