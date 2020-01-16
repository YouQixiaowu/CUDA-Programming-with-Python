import numpy as np
import plotly.offline as py
import plotly.graph_objs as go

class Material(object):
    def __init__(self, unitcell, lattice, supercell=[1,1,1]):
        unitcell = np.array(unitcell)
        lattice = np.array(lattice)
        supercell = supercell
        trace = go.Scatter3d(
            x = unitcell[0, :],
            y = unitcell[1, :],
            z = unitcell[2, :],
            mode='markers',
        )

        fig = go.Figure(data=[trace])
        py.plot(fig, filename='a')

if __name__ == "__main__":
    unitcell = [
        [0.0, 0.0, 0.5, 0.5],
        [0.0, 0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5, 0.0],
        ]
    lattice = [
        [5.385, 0.000, 0.000],
        [0.000, 5.385, 0.000],
        [0.000, 0.000, 5.385],
        ]
    supercell = [2,3,4]
    Graphene = Material(
        unitcell = unitcell,
        lattice = lattice,
        supercell = supercell,
        )