import sys
import collections
from osgeo import gdal
import geojson
import shapely.geometry
from geopandas import GeoSeries
from psting import opening, extract_contours, simplify, parents_in_hierarchy, featurize
import numpy as np
from PIL import Image

class BuildingExtract(object):
    def __init__(self, kernel_opening = 10, simplify_threshold = 0.001):
        """
        Adapted from: https://github.com/mapbox/robosat 
        """
        self.kernel_opening = kernel_opening
        self.simplify_threshold = simplify_threshold
        self.features = []

    def extract(self, mask_img):
        mask = gdal.Open(mask_img)
        gt = mask.GetGeoTransform()
        self.prj = mask.GetProjection()
        mask = mask.ReadAsArray()
        mask = opening(mask, self.kernel_opening)
        multipolygons, hierarchy = extract_contours(mask)

        if hierarchy is None:
            return

        assert len(hierarchy) == 1, "always single hierarchy for all polygons in multipolygon"
        hierarchy = hierarchy[0]

        assert len(multipolygons) == len(hierarchy), "polygons and hierarchy in sync"

        polygons = [simplify(polygon, self.simplify_threshold) for polygon in multipolygons]
        #polygons = [polygon for polygon in multipolygons]
        # All child ids in hierarchy tree, keyed by root id.
        features = collections.defaultdict(set)

        for i, (polygon, node) in enumerate(zip(polygons, hierarchy)):
            if len(polygon) < 3:
                print("Warning: simplified feature no longer valid polygon, skipping", file=sys.stderr)
                continue

            _, _, _, parent_idx = node

            ancestors = list(parents_in_hierarchy(i, hierarchy))

            # Only handles polygons with a nesting of two levels for now => no multipolygons.
            if len(ancestors) > 1:
                print("Warning: polygon ring nesting level too deep, skipping", file=sys.stderr)
                continue

            # A single mapping: i => {i} implies single free-standing polygon, no inner rings.
            # Otherwise: i => {i, j, k, l} implies: outer ring i, inner rings j, k, l.
            root = ancestors[-1] if ancestors else i

            features[root].add(i)

        for outer, inner in features.items():
            rings = [featurize(polygons[outer], gt)]

            # In mapping i => {i, ..} i is not a child.
            children = inner.difference(set([outer]))

            for child in children:
                rings.append(featurize(polygons[child], gt))

            assert 0 < len(rings), "at least one outer ring in a polygon"

            geometry = geojson.Polygon(rings)
            shape = shapely.geometry.shape(geometry)

            if shape.is_valid:
                #self.features.append(geojson.Feature(geometry=geometry))
                self.features.append(geometry)
            else:
                print("Warning: extracted feature is not valid, skipping", file=sys.stderr)

    def save(self, out):
        p = GeoSeries(self.features)
        #print(self.prj)
        p.to_file(out,crs=self.prj)
        # collection = geojson.FeatureCollection(self.features)
        # # w = shapefile.Writer(out)
        # # w.autoBalance = 1
        # # w.field('class_name', 'C', '40')
        # # features = collection['features']
        # # #print(features)
        # # for feature in features:
        # #     w.poly(feature['geometry']['coordinates'])
        # #     w.record('building')
        # # w.close()    
        # with open(out, "w") as fp:
        #     geojson.dump(collection, fp)



def get_polygons(pred_masks_path = '..//building_extraction//output//combine_crop_seg//crop2pred.tif', 
                polygons_path = '..//building_extraction//output//shp//crop2.shp',
                kernel_opening = 3, simplify_threshold = 0.00001):
    
    """Generate GeoJSON polygons from predicted masks
    Args:
      pred_masks_path: directory where the predicted mask are saved
      polygons_path: path to GeoJSON file to store features in
      kernel_opening: the opening morphological operation's kernel size in pixel
      simplify_threshold: the simplification accuracy as max. percentage of the arc length, in [0, 1]
    """
    bldg_extract = BuildingExtract(kernel_opening = kernel_opening, simplify_threshold = simplify_threshold)
    #mask = np.array(Image.open(pred_masks_path).convert("P"), dtype=np.uint8)
    bldg_extract.extract(pred_masks_path)
    bldg_extract.save(polygons_path)

#get_polygons()
# import geopandas as gpd
# tif = gdal.Open('C://code//building_extraction//test//crop2.tif')
# gt = tif.GetGeoTransform()
# prj = tif.GetProjection()
# gdf = gpd.read_file('C://code//building_extraction//output//shp//crop2.geojson')
# gdf.to_file('C://code//building_extraction//output//shp//crop2.shp',crs=prj)
