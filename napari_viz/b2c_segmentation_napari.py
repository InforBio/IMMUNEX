import napari
import numpy as np

viewer = napari.Viewer()
coords = np.random.rand(10000, 2) * 1000
viewer.add_points(coords, name="Test Points", size=2)
napari.run()
