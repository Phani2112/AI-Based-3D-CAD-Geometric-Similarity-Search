import math
import torch
from OCP.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCP.BRepGProp import BRepGProp
from OCP.GeomAbs import (
    GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Torus,
    GeomAbs_Sphere, GeomAbs_BezierSurface, GeomAbs_BSplineSurface, 
    GeomAbs_SurfaceOfRevolution, GeomAbs_SurfaceOfExtrusion, GeomAbs_OffsetSurface,
    GeomAbs_OtherSurface
)
from OCP.GeomAbs import (
    GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse, GeomAbs_Hyperbola,
    GeomAbs_Parabola, GeomAbs_BezierCurve, GeomAbs_BSplineCurve, GeomAbs_OtherCurve
)
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCP.TopExp import TopExp
from OCP.TopoDS import TopoDS
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_IndexedMapOfShape
from OCP.gp import gp_Pnt, gp_Vec

SURFACE_TYPES = {
    GeomAbs_Plane: 0,
    GeomAbs_Cylinder: 1,
    GeomAbs_Cone: 2,
    GeomAbs_Torus: 3,
    GeomAbs_Sphere: 4,
    GeomAbs_BezierSurface: 5,
    GeomAbs_BSplineSurface: 6,
    GeomAbs_SurfaceOfRevolution: 7,
    GeomAbs_SurfaceOfExtrusion: 8,
    GeomAbs_OffsetSurface: 9,
}

CURVE_TYPES = {
    GeomAbs_Line: 0,
    GeomAbs_Circle: 1,
    GeomAbs_Ellipse: 2,
    GeomAbs_Hyperbola: 3,
    GeomAbs_Parabola: 4,
    GeomAbs_BezierCurve: 5,
    GeomAbs_BSplineCurve: 6,
}

def _one_hot(value, num_classes):
    return [1.0 if i == value else 0.0 for i in range(num_classes)]

class BRepGraph:
    def __init__(self, x, edge_attr, edge_index):
        self.x = x
        self.edge_attr = edge_attr
        self.edge_index = edge_index
        self.num_nodes = x.shape[0]
        self.num_edges = edge_index.shape[1]

class BRepGraphBuilder:
    def build(self, shape):
        nodes = []
        edges = []
        edge_attrs = []
        
        faces = shape.Faces
        for face in faces:
            adaptor = BRepAdaptor_Surface(face, True)
            surf_type = SURFACE_TYPES.get(adaptor.GetType(), 0)
            one_hot = _one_hot(surf_type, 10)
            normal, tangent = self._surface_vectors(adaptor)
            node_feat = one_hot + normal + tangent
            nodes.append(node_feat)

        face_map = TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(shape._shape, TopAbs_FACE, face_map)

        edge_to_faces = TopTools_IndexedDataMapOfShapeListOfShape()
        TopExp.MapShapesAndAncestors_s(shape._shape, TopAbs_EDGE, TopAbs_FACE, edge_to_faces)

        for idx in range(1, edge_to_faces.Extent() + 1):
            face_ids = [
                face_map.FindIndex(face_shape) - 1
                for face_shape in edge_to_faces.FindFromIndex(idx)
                if face_map.FindIndex(face_shape) > 0
            ]
            face_ids = sorted(set(face_ids))
            if len(face_ids) < 2:
                continue

            edge = TopoDS.Edge_s(edge_to_faces.FindKey(idx))
            curve = BRepAdaptor_Curve(edge)
            direction = self._curve_direction(curve)
            
            curve_type = CURVE_TYPES.get(curve.GetType(), 0)
            one_hot = _one_hot(curve_type, 7)
            
            props = GProp_GProps()
            BRepGProp.LinearProperties_s(edge, props)
            length = float(props.Mass())
            
            edge_attr_feat = one_hot + direction + [length]

            for i in range(len(face_ids)):
                for j in range(i + 1, len(face_ids)):
                    edges.append([face_ids[i], face_ids[j]])
                    edges.append([face_ids[j], face_ids[i]])
                    edge_attrs.append(edge_attr_feat)
                    edge_attrs.append(edge_attr_feat)

        x = torch.tensor(nodes, dtype=torch.float32)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32) if edge_attrs else torch.empty((0, 11), dtype=torch.float32)

        return BRepGraph(x, edge_attr, edge_index)

    def _surface_vectors(self, surface):
        u = self._midpoint(surface.FirstUParameter(), surface.LastUParameter())
        v = self._midpoint(surface.FirstVParameter(), surface.LastVParameter())
        if u is None or v is None:
            return [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]

        point = gp_Pnt()
        d1u = gp_Vec()
        d1v = gp_Vec()
        try:
            surface.D1(u, v, point, d1u, d1v)
            normal = d1u.Crossed(d1v)
            return self._normalize_vec(normal, [0.0, 0.0, 1.0]), self._normalize_vec(d1u, [1.0, 0.0, 0.0])
        except Exception:
            return [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]

    def _curve_direction(self, curve):
        parameter = self._midpoint(curve.FirstParameter(), curve.LastParameter())
        if parameter is None:
            return [1.0, 0.0, 0.0]

        point = gp_Pnt()
        tangent = gp_Vec()
        try:
            curve.D1(parameter, point, tangent)
            return self._normalize_vec(tangent, [1.0, 0.0, 0.0])
        except Exception:
            return [1.0, 0.0, 0.0]

    @staticmethod
    def _midpoint(first, last):
        if not math.isfinite(first) or not math.isfinite(last):
            return None
        return (float(first) + float(last)) / 2.0

    @staticmethod
    def _normalize_vec(vector, fallback):
        magnitude = float(vector.Magnitude())
        if magnitude <= 1e-12 or not math.isfinite(magnitude):
            return fallback
        return [
            float(vector.X()) / magnitude,
            float(vector.Y()) / magnitude,
            float(vector.Z()) / magnitude,
        ]