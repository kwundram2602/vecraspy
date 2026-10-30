"""Tests for vecraspy.convert."""

import json

import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from vecraspy.convert import (
    aoi_to_bbox,
    aoi_to_geojson,
    aoi_to_wkt,
    bbox_to_aoi,
    bbox_to_wkt,
    geojson_to_aoi,
    wkt_to_aoi,
    wkt_to_bbox,
)
from vecraspy.vector import get_stac_spatial_extent

BBOX = (7.1, 51.9, 7.3, 52.0)


def test_bbox_round_trip():
    aoi = bbox_to_aoi(BBOX)
    assert aoi.crs == "EPSG:4326"
    assert len(aoi) == 1
    assert aoi_to_bbox(aoi) == pytest.approx(BBOX)


@pytest.mark.parametrize(
    "text",
    ["7.1,51.9,7.3,52.0", "7.1, 51.9, 7.3, 52", "[7.1 51.9 7.3 52]", "7.1;51.9;7.3;52"],
)
def test_bbox_string_parsing(text):
    assert wkt_to_bbox(bbox_to_wkt(text)) == pytest.approx(BBOX)


@pytest.mark.parametrize(
    "bbox", [(1, 2, 3), "1,2,3,4,5", (3, 0, 1, 1), (0, 1, 1, 1), "a,b,c,d"]
)
def test_bad_bbox_raises(bbox):
    with pytest.raises(ValueError):
        bbox_to_aoi(bbox)


def test_wkt_round_trip():
    wkt = "POLYGON ((0 0, 2 0, 2 1, 0 1, 0 0))"
    aoi = wkt_to_aoi(wkt, crs="EPSG:25832")
    assert aoi.crs == "EPSG:25832"
    assert aoi_to_wkt(aoi) == wkt


def test_invalid_wkt_raises():
    with pytest.raises(ValueError):
        wkt_to_aoi("POLYGON ((0 0, 1")


def test_aoi_to_wkt_unions_features_and_rounds():
    gdf = gpd.GeoDataFrame(
        geometry=[box(0, 0, 1, 1), box(1, 0, 2.123456789, 1)], crs="EPSG:4326"
    )
    wkt = aoi_to_wkt(gdf, precision=2)
    assert wkt.startswith("POLYGON")
    assert "2.12 " in wkt
    assert "2.123" not in wkt


def test_aoi_to_wkt_reprojects():
    aoi = bbox_to_aoi((400000, 5750000, 401000, 5751000), crs="EPSG:25832")
    minx, miny, maxx, maxy = wkt_to_bbox(aoi_to_wkt(aoi, crs="EPSG:4326"))
    assert 7 < minx < maxx < 8
    assert 51 < miny < maxy < 52


def test_aoi_to_bbox_reprojects_and_matches_stac_extent():
    aoi = bbox_to_aoi((400000, 5750000, 401000, 5751000), crs="EPSG:25832")
    bbox = aoi_to_bbox(aoi, crs="EPSG:4326")
    assert get_stac_spatial_extent(aoi) == list(bbox)
    assert aoi_to_bbox(aoi) == pytest.approx((400000, 5750000, 401000, 5751000))


def test_reproject_without_crs_raises():
    with pytest.raises(ValueError, match="no CRS"):
        aoi_to_wkt(box(0, 0, 1, 1), crs="EPSG:4326")


def test_accepts_geometry_series_and_path(tmp_path):
    geom = box(0, 0, 1, 1)
    path = tmp_path / "aoi.gpkg"
    gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326").to_file(path)
    expected = (0.0, 0.0, 1.0, 1.0)
    assert aoi_to_bbox(geom) == expected
    assert aoi_to_bbox(gpd.GeoSeries([geom])) == expected
    assert aoi_to_bbox(path) == expected
    assert aoi_to_bbox(str(path)) == expected


def test_empty_aoi_raises():
    with pytest.raises(ValueError, match="no geometry"):
        aoi_to_bbox(gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"))


def test_geojson_round_trip():
    aoi = bbox_to_aoi(BBOX)
    text = aoi_to_geojson(aoi)
    assert json.loads(text)["type"] == "Polygon"
    back = geojson_to_aoi(text)
    assert back.crs == "EPSG:4326"
    assert aoi_to_bbox(back) == pytest.approx(BBOX)


def test_aoi_to_geojson_defaults_to_wgs84():
    aoi = bbox_to_aoi((400000, 5750000, 401000, 5751000), crs="EPSG:25832")
    coords = json.loads(aoi_to_geojson(aoi, precision=3))["coordinates"][0]
    assert all(7 < x < 8 and 51 < y < 52 for x, y in coords)
    assert all(round(x, 3) == x for x, _ in coords)


def test_geojson_feature_collection_keeps_properties():
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "a"},
                "geometry": {"type": "Point", "coordinates": [1, 2]},
            },
            {
                "type": "Feature",
                "properties": {"name": "b"},
                "geometry": {"type": "Point", "coordinates": [3, 4]},
            },
        ],
    }
    aoi = geojson_to_aoi(json.dumps(fc))
    assert list(aoi["name"]) == ["a", "b"]
    assert aoi.geometry.iloc[1] == Point(3, 4)
    assert len(geojson_to_aoi(fc["features"][0])) == 1


@pytest.mark.parametrize(
    "text", ["{not json", '{"coordinates": [1, 2]}', '{"type": "Foo"}']
)
def test_invalid_geojson_raises(text):
    with pytest.raises(ValueError):
        geojson_to_aoi(text)


def test_public_exports():
    import vecraspy

    for name in [
        "aoi_to_bbox",
        "aoi_to_geojson",
        "aoi_to_wkt",
        "bbox_to_aoi",
        "bbox_to_wkt",
        "geojson_to_aoi",
        "wkt_to_aoi",
        "wkt_to_bbox",
    ]:
        assert hasattr(vecraspy, name)
