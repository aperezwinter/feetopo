import gmsh, sys
import numpy as np
import src.lib.mesh as mesh

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)

s1 = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
gmsh.model.occ.synchronize()  # synchronize model
v1_dimTags = gmsh.model.occ.extrude([(2, s1)], 0, 0, 2, [10], [1], True)
gmsh.model.occ.synchronize()

gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 5e-1)

gmsh.model.mesh.setOrder(1)
gmsh.model.mesh.generate(3)

# ... and save it to disk
gmsh.write(fileName="test.msh")

# Launch the GUI to see the results:
#if "-nopopup" not in sys.argv:
#    gmsh.fltk.run()

gmsh.model.remove()
gmsh.finalize()

myMesh = mesh.Mesh("test.msh")
print(myMesh)
