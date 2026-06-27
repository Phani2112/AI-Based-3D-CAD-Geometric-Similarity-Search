from OCP.STEPControl import STEPControl_Reader
from OCP.IFSelect import IFSelect_RetDone
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE
from OCP.TopoDS import TopoDS

class STEPParser:
    def parse(self, filepath):
        reader = STEPControl_Reader()
        status = reader.ReadFile(filepath)
        if status == IFSelect_RetDone:
            reader.TransferRoots()
            shape = reader.OneShape()
            if not shape.IsNull():
                return ParsedShape(shape)
        return None

class ParsedShape:
    def __init__(self, shape):
        self._shape = shape
        self.Faces = []
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More():
            face = TopoDS.Face_s(exp.Current())
            self.Faces.append(face)
            exp.Next()