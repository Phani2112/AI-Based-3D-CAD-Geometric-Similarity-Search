from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import topods

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
            face = topods.Face(exp.Current())
            self.Faces.append(face)
            exp.Next()