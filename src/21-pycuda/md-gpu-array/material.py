import numpy as np

class Material(object):
    '''
    usage:\n
    Unicellular information (Replacing 'fractional' with 'coordinate' represents using the true coordinates of the atoms).\n
    unitcell = {
        'lattice' : [
            [5.000, 0.000, 0.000],
            [0.000, 5.000, 0.000],
            [0.000, 0.000, 5.000],
            ],
        'fractional' : [
            ['B', 11, [0.3, 0.0, 0.0]],
            ['C', 12, [0.4, 0.4, 0.0]],
            ['C', 12, [0.5, 0.8, 0.0]],
            ['N', 14, [0.6, 0.3, 0.0]],
            ],
        }
    Supercell multiple(default: [1,1,1]).\n
    supercell = [10, 20, 30]
    '''
    def __init__(self, unitcell:dict, supercell:list=[1,1,1]):

        if not isinstance(unitcell, dict):
            raise ValueError(self.__doc__)

        if 'lattice' in unitcell:
            unitcell_lattice = np.array(unitcell['lattice'])
            if not unitcell_lattice.shape == (3,3):
                raise ValueError('The shape of lattice matrix must be (3,3).')
        else:
            raise ValueError(self.__doc__)

        if 'coordinate' in unitcell:
            unitcell_tpye = [atom[0] for atom in unitcell['coordinate']]
            unitcell_mass = [atom[1] for atom in unitcell['coordinate']]
            unitcell_coordinate = np.array([atom[2] for atom in unitcell['coordinate']]).T
        elif 'fractional' in unitcell:
            unitcell_tpye = [atom[0] for atom in unitcell['fractional']]
            unitcell_mass = [atom[1] for atom in unitcell['fractional']]
            unitcell_coordinate = unitcell_lattice.T.dot(
                np.array([atom[2] for atom in unitcell['fractional']]).T)
        else:
            raise ValueError(self.__doc__)

        self.atomic_number = int(np.prod(supercell) * unitcell_coordinate.shape[1])
        self.atomic_mass = np.tile(unitcell_mass, (np.prod(supercell), 1)).T.ravel()
        self.lattice = np.tile(supercell, (3,1)) * unitcell_lattice
        grid_x, grid_y, grid_z = np.meshgrid(
            list(range(supercell[0])),list(range(supercell[1])),list(range(supercell[2])))
        grid_coordinate = unitcell_lattice.T.dot([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
        coor = np.tile(grid_coordinate, (unitcell_coordinate.shape[1], 1)) + \
            np.tile(unitcell_coordinate.T.ravel(), (np.prod(supercell), 1)).T
        self.coordinate = np.array(
            [coor[0::3].ravel(), coor[1::3].ravel(), coor[2::3].ravel()])
