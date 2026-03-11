"""
Generate realistic GeoJSON polygon boundaries for 11 areas in Accra, Ghana.
Uses shapely to create irregular polygons with 6-10 vertices per neighborhood.
"""

import json
import math
import random
from shapely.geometry import Polygon, mapping


def make_irregular_polygon(center_lon, center_lat, radius_deg, n_vertices=8, seed=None):
    """
    Create an irregular polygon by placing vertices around a center point
    with randomized radii and angle offsets to simulate real neighborhood shapes.

    Args:
        center_lon: longitude of center (degrees)
        center_lat: latitude of center (degrees)
        radius_deg: approximate radius in degrees
        n_vertices: number of vertices (6-10 recommended)
        seed: random seed for reproducibility
    """
    rng = random.Random(seed)

    # Aspect correction: 1 degree latitude ~ 111 km, 1 degree longitude ~ 111*cos(lat) km
    lat_rad = math.radians(center_lat)
    lon_scale = math.cos(lat_rad)  # lon degrees are shorter physically

    coords = []
    # Spread angles evenly, then add small jitter
    base_angles = [i * (360.0 / n_vertices) for i in range(n_vertices)]

    for i, base_angle in enumerate(base_angles):
        # Jitter the angle slightly (up to ±15 degrees)
        angle_jitter = rng.uniform(-15, 15)
        angle = math.radians(base_angle + angle_jitter)

        # Randomize radius between 55% and 130% of base radius
        r_factor = rng.uniform(0.55, 1.30)
        r = radius_deg * r_factor

        # Apply aspect ratio correction so polygon looks right on the ground
        dlon = (r / lon_scale) * math.sin(angle)
        dlat = r * math.cos(angle)

        coords.append((center_lon + dlon, center_lat + dlat))

    # Close the polygon
    coords.append(coords[0])
    return Polygon(coords)


def deg_radius(km):
    """Convert kilometers to approximate degrees (using ~111 km/degree)."""
    return km / 111.0


# ---------------------------------------------------------------------------
# Define neighborhoods
# ---------------------------------------------------------------------------

# (name, type_label, center_lat, center_lon, half_width_km, n_vertices, seed)
neighborhoods = [
    # Mid-Market Districts
    {
        "name": "Spintex Road",
        "type": "district",
        "lat": 5.620,
        "lon": -0.128,
        "km": 2.0,        # ~4 km wide corridor → radius ~2 km
        "n_vertices": 9,
        "seed": 101,
    },
    {
        "name": "Adenta",
        "type": "district",
        "lat": 5.712,
        "lon": -0.168,
        "km": 2.5,        # ~5 km spread
        "n_vertices": 8,
        "seed": 102,
    },
    {
        "name": "Tema",
        "type": "district",
        "lat": 5.668,
        "lon": 0.017,
        "km": 3.5,        # ~7 km grid city
        "n_vertices": 10,
        "seed": 103,
    },
    {
        "name": "Dome",
        "type": "district",
        "lat": 5.650,
        "lon": -0.235,
        "km": 2.0,        # ~4 km suburb
        "n_vertices": 7,
        "seed": 104,
    },
    {
        "name": "Kasoa",
        "type": "district",
        "lat": 5.534,
        "lon": -0.420,
        "km": 3.0,        # ~6 km peri-urban
        "n_vertices": 9,
        "seed": 105,
    },
    # Prime Areas
    {
        "name": "East Legon",
        "type": "prime",
        "lat": 5.636,
        "lon": -0.151,
        "km": 1.25,       # ~2.5 km upscale estate
        "n_vertices": 8,
        "seed": 201,
    },
    {
        "name": "Cantonments",
        "type": "prime",
        "lat": 5.587,
        "lon": -0.186,
        "km": 1.0,        # ~2 km diplomatic enclave
        "n_vertices": 7,
        "seed": 202,
    },
    {
        "name": "Airport Residential",
        "type": "prime",
        "lat": 5.605,
        "lon": -0.166,
        "km": 0.9,        # ~1.8 km near airport
        "n_vertices": 7,
        "seed": 203,
    },
    {
        "name": "Labone / Roman Ridge",
        "type": "prime",
        "lat": 5.574,
        "lon": -0.174,
        "km": 0.75,       # ~1.5 km coastal upscale
        "n_vertices": 6,
        "seed": 204,
    },
    {
        "name": "Dzorwulu / Abelenkpe",
        "type": "prime",
        "lat": 5.597,
        "lon": -0.210,
        "km": 1.0,        # ~2 km
        "n_vertices": 8,
        "seed": 205,
    },
    {
        "name": "Trasacco Valley",
        "type": "prime",
        "lat": 5.662,
        "lon": -0.135,
        "km": 1.0,        # ~2 km gated estate
        "n_vertices": 7,
        "seed": 206,
    },
]

# ---------------------------------------------------------------------------
# Build GeoJSON FeatureCollection
# ---------------------------------------------------------------------------

features = []

for nbhd in neighborhoods:
    poly = make_irregular_polygon(
        center_lon=nbhd["lon"],
        center_lat=nbhd["lat"],
        radius_deg=deg_radius(nbhd["km"]),
        n_vertices=nbhd["n_vertices"],
        seed=nbhd["seed"],
    )

    # Validate polygon
    if not poly.is_valid:
        poly = poly.buffer(0)  # attempt repair
    if not poly.is_valid:
        raise RuntimeError(f"Could not create valid polygon for {nbhd['name']}")

    feature = {
        "type": "Feature",
        "geometry": mapping(poly),
        "properties": {
            "name": nbhd["name"],
            "type": nbhd["type"],
            "lat": nbhd["lat"],
            "lon": nbhd["lon"],
        },
    }
    features.append(feature)

geojson = {
    "type": "FeatureCollection",
    "features": features,
}

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------

output_path = "/teamspace/studios/this_studio/Zillow-Home-Value-Index-Prediction/data/accra_boundaries.geojson"

with open(output_path, "w") as f:
    json.dump(geojson, f, indent=2)

print(f"GeoJSON written to: {output_path}")
print(f"Total features: {len(features)}")
print()

# Quick validation summary
for feat in features:
    props = feat["properties"]
    geom = feat["geometry"]
    n_coords = len(geom["coordinates"][0])
    print(f"  [{props['type']:8s}] {props['name']:30s}  vertices={n_coords-1}  "
          f"center=({props['lat']:.3f}, {props['lon']:.3f})")

print()
print("Done.")
