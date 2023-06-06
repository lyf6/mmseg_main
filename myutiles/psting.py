import os
import cv2
import numpy as np
import collections
import functools
import pyproj
import shapely.ops

from PIL import Image
from rtree.index import Index, Property






def probs_load(path,anchors, threshold = None):
    """ load and addapt the predicted probabilities
    Adapted from: https://github.com/mapbox/robosat  
    Args:
      path: path of the probs file
      anchors: evenly spaced numbers in [0,1]
      threshold: threshold used to curtail the probabilities 

    Returns:
      numpy array of probabilities of each class (i.e., background and foreground)
    """
    # Note: assumes binary case and probability sums up to one.

    quantized = np.array(Image.open(path).convert("P"))

    # (H, W, 1) -> (1, W, H)
    foreground = np.rollaxis(np.expand_dims(anchors[quantized], axis=0), axis=0)
    if threshold:
        foreground[foreground < threshold] = 0
    background = np.rollaxis(1. - foreground, axis=0)

    # (1, W, H) + (1, W, H) -> (2, W, H)
    return np.concatenate((background, foreground), axis=0)

def opening(mask, kernel_size):
    """ Morphologycal transforations of erosion the dilatation which removes small objects
    Args:
      mask: the binary mask to transform
      eps: the opening kernel size, in pixel

    Returns:
      The transformed mask
    """
    struct = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, struct)

def extract_contours(mask):
    """ Find contours objects in the mask
    From: https://github.com/mapbox/robosat 
    Args:
      mask: the binary mask to transform
      eps: the opening kernel size, in pixel

    Returns:
      The detected contours as a list of points and the contour hierarchy
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

def simplify(polygon, eps):
    """Simplifies a polygon to minimize the polygon's vertices
    From: https://github.com/mapbox/robosat 

    Args:
      polygon: the polygon made up of a list of vertices.
      eps: the approximation accuracy as max. percentage of the arc length, in [0, 1]
    Returns:
      The simplified polygon
    """

    assert 0 <= eps <= 1, "approximation accuracy is percentage in [0, 1]"

    epsilon = eps * cv2.arcLength(polygon, closed=True)
    return cv2.approxPolyDP(polygon, epsilon=epsilon, closed=True)

def parents_in_hierarchy(node, tree):
    """Walks a hierarchy tree upwards from a starting node collecting all nodes on the way.
    From: https://github.com/mapbox/robosat 

    Args:
      node: the index for the starting node in the hierarchy.
      tree: the hierarchy tree containing tuples of (next, prev, first child, parent) ids.

    Yields:
      The node ids on the upwards path in the hierarchy tree.
    """

    def parent(n):
        # next, prev, fst child, parent
        return n[3]

    at = tree[node]
    up = parent(at)

    while up != -1:
        index = up
        at = tree[index]
        up = parent(at)

        assert index != node, "upward path does not include starting node"

        yield index


def featurize(polygon, gt):
    feature = []
    x_min = gt[0]
    x_size = gt[1]
    y_min = gt[3]
    y_size = gt[5]
    feature = []
    for point in polygon:
        px, py = point[0]
        x = px*x_size + x_min
        y = py*y_size + y_min
        feature.append([x,y])
    assert feature, "at least one location in polygon"
    feature.append(feature[0])  # polygons are closed

    return feature


# def featurize(tile, polygon, shape):
#     """Transforms polygons in image pixel coordinates into world coordinates.
#     From: https://github.com/mapbox/robosat 

#     Args:
#       tile: the tile this polygon is in for coordinate calculation.
#       polygon: the polygon to transform from pixel to world coordinates.
#       shape: the image's max x and y coordinates.

#     Returns:
#       The closed polygon transformed into world coordinates.
#     """

#     xmax, ymax = shape

#     feature = []

#     for point in polygon:
#         px, py = point[0]
#         dx, dy = px / xmax, py / ymax

#         feature.append(pixel_to_location(tile, dx, 1. - dy))

#     assert feature, "at least one location in polygon"
#     feature.append(feature[0])  # polygons are closed

#     return feature







class UndirectedGraph:
    """Simple undirected graph.
    From: https://github.com/mapbox/robosat 

    Note: stores edges; can not store vertices without edges.
    """

    def __init__(self):
        """Creates an empty `UndirectedGraph` instance.
        """

        # Todo: We might need a compressed sparse row graph (i.e. adjacency array)
        # to make this scale. Let's circle back when we run into this limitation.
        self.edges = collections.defaultdict(set)

    def add_edge(self, s, t):
        """Adds an edge to the graph.

        Args:
          s: the source vertex.
          t: the target vertex.

        Note: because this is an undirected graph for every edge `s, t` an edge `t, s` is added.
        """

        self.edges[s].add(t)
        self.edges[t].add(s)

    def targets(self, v):
        """Returns all outgoing targets for a vertex.

        Args:
          v: the vertex to return targets for.

        Returns:
          A list of all outgoing targets for the vertex.
        """

        return self.edges[v]

    def vertices(self):
        """Returns all vertices in the graph.

        Returns:
          A set of all vertices in the graph.
        """

        return self.edges.keys()

    def empty(self):
        """Returns true if the graph is empty, false otherwise.

        Returns:
          True if the graph has no edges or vertices, false otherwise.
        """
        return len(self.edges) == 0

    def dfs(self, v):
        """Applies a depth-first search to the graph.

        Args:
          v: the vertex to start the depth-first search at.

        Yields:
          The visited graph vertices in depth-first search order.

        Note: does not include the start vertex `v` (except if an edge targets it).
        """

        stack = []
        stack.append(v)

        seen = set()

        while stack:
            s = stack.pop()

            if s not in seen:
                seen.add(s)

                for t in self.targets(s):
                    stack.append(t)

                yield s

    def components(self):
        """Computes connected components for the graph.

        Yields:
          The connected component sub-graphs consisting of vertices; in no particular order.
        """

        seen = set()

        for v in self.vertices():
            if v not in seen:
                component = set(self.dfs(v))
                component.add(v)

                seen.update(component)

                yield component




# def project(shape, source, target):
#     """Projects a geometry from one coordinate system into another.
#     From: https://github.com/mapbox/robosat

#     Args:
#       shape: the geometry to project.
#       source: the source EPSG spatial reference system identifier.
#       target: the target EPSG spatial reference system identifier.

#     Returns:
#       The projected geometry in the target coordinate system.
#     """

#     transformer = pyproj.Transformer.from_crs(source, target)
#     return shapely.ops.transform(transformer.transform, shape)

def project_0(shape, source, target):
    """Projects a geometry from one coordinate system into another.
    This function is an adaptation to bypass a bug in pyproj package
    Args:
      shape: the geometry to project.
      source: the source EPSG spatial reference system identifier.
      target: the target EPSG spatial reference system identifier.

    Returns:
      The projected geometry in the target coordinate system.
    """
    with warnings.catch_warnings(): # To exclude the warnings of Proj deprecation
        warnings.simplefilter("ignore")
        proj_in = pyproj.Proj(init=source)
        proj_out = pyproj.Proj(init=target)
    project_fun = pyproj.Transformer.from_proj(proj_in, proj_out).transform

    return shapely.ops.transform(project_fun, shape)

def project(shape, source, target):
    """Projects a geometry from one coordinate system into another.
    Args:
      shape: the geometry to project.
      source: the source EPSG spatial reference system identifier.
      target: the target EPSG spatial reference system identifier.

    Returns:
      The projected geometry in the target coordinate system.
    """
    with warnings.catch_warnings(): # To exclude the warnings of Proj deprecation
        warnings.simplefilter("ignore")
        proj_in = CRS(source)
        proj_out = CRS(target)
    project_fun = pyproj.Transformer.from_crs(proj_in, proj_out).transform

    return shapely.ops.transform(project_fun, shape)


def union(shapes):
    """Returns the union of all shapes.
    From: https://github.com/mapbox/robosat

    Args:
      shapes: the geometries to merge into one.

    Returns:
      The union of all shapes as one shape.
    """

    assert shapes

    def fn(lhs, rhs):
        return lhs.union(rhs)

    return functools.reduce(fn, shapes)


def make_index(shapes):
    """Creates an index for fast and efficient spatial queries.
    From: https://github.com/mapbox/robosat

    Args:
      shapes: shapely shapes to bulk-insert bounding boxes for into the spatial index.

    Returns:
      The spatial index created from the shape's bounding boxes.
    """

    # Todo: benchmark these for our use-cases
    prop = Property()
    prop.dimension = 2
    prop.leaf_capacity = 1000
    prop.fill_factor = 0.9

    def bounded():
        for i, shape in enumerate(shapes):
            yield (i, shape.bounds, None)

    return Index(bounded(), properties=prop)

