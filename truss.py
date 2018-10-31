#!python3

# ##----------------------------------------## #
#         Author: M. Burak Yesilyurt           #
#      Truss Optimization by Employing         #
#             Genetic Algorithms               #
# ##----------------------------------------## #

# Importing necessary modules
# To run the code below, imported python packages must be installed.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


class Truss:
    # Truss Class.
    def __init__(self, param):
        self.h1 = param["h1"]  # Shoulder Height of the Truss in mm
        self.h2 = param["h2"]  # Peak Height of the Truss in mm
        self.l = param["l"]  # Half of the Span Length of the Truss
        self.nd = param["nd"]  # Number of Divisions of the Truss
        self.dia = param["dia"]  # Alignment of the Diagonals for Each Bay

        self.B = param["B"]  # Height of SHS member
        self.t = param["t"]  # Wall Thickness of SHS member in mm
        self.q = param["q"]  # Line Load on the Truss in kN/m

        self.nn = 2 * self.nd + 2  # Numer of Nodes
        self.nm = 4 * self.nd + 1  # Number of Members
        self.lines = []  # Line list for plotting
        self.j_forces = np.zeros((self.nn,2)) # Joint Force Matrix
        self.f = np.zeros((self.nn*2)) # Dof Force Matrix - will be mapped from j_forces

        xvals = np.array([np.hstack((np.linspace(0, self.l, self.nd + 1), np.linspace(0, self.l, self.nd + 1)))]).T
        # Nodal coordinates - in X axis

        yvals = np.array([np.hstack((np.zeros(self.nd + 1), np.linspace(self.h1, self.h2, self.nd + 1)))]).T
        # Nodal coordinates - in Y axis

        self.n_coord = np.hstack((xvals, yvals)) # Nodal Coordinates are stacked in n_coord
        self.n_conn = np.zeros((self.nm, 2)) # Connectivity matrix
        self.n_bound = np.array([[0, 1], [0, 2], [self.nd, 1], [self.nd * 2 + 1, 1]]) # Introducing boundary cnds.
        self.n_dof = np.zeros((self.nn, 2)) # Nodal Dof matrix - info on which dof is assigned to which joint
        self.geo_props = np.zeros((self.nm, 5)) # Geometric props. of each member.
        self.mate_props = np.zeros((self.nm, 2)) # Material props. of each member.

        # Populating connectivity matrix
        for i in range(self.nd):
            self.n_conn[i] = [i, i + 1]

        for i in range(self.nd):
            self.n_conn[i + self.nd] = [i + self.nd + 1, i + self.nd + 2]

        for i in range(self.nd + 1):
            self.n_conn[i + 2 * self.nd] = [i, i + self.nd + 1]

        for i in range(self.nd):
            if self.dia[i]:
                self.n_conn[i + 3 * self.nd + 1] = [i, i + self.nd + 2]
            else:
                self.n_conn[i + 3 * self.nd + 1] = [i + self.nd + 1, i + 1]
        self.n_conn = self.n_conn.astype(int)

        # Populating dof matrix - restrained dofs are shifted to the end.
        count = 0
        for n in range(self.nn):
            for j in range(1, 3):
                if not np.equal(self.n_bound, [[n, j]]).all(axis=1).any():
                    print(np.equal(self.n_bound, [[n, j]]).all(axis=1).any(),n,j)
                    self.n_dof[n, j - 1] = int(count)
                    count += 1
        for i in self.n_bound:
            node = i[0]
            dof = i[1]
            self.n_dof[node, dof - 1] = int(count)
            count += 1
        # Material properties
        E = 210000  # N/mm2
        fy = 355  # N/mm2

        for i in range(self.nm):
            self.mate_props[i] = [E, fy]

        # Geometric Properties
        A, I, r = shs_props(self.B, self.t)

        for x, y in enumerate(self.n_conn):
            i = int(y[0])
            j = int(y[1])

            ix = self.n_coord[i, 0]
            iy = self.n_coord[i, 1]

            jx = self.n_coord[j, 0]
            jy = self.n_coord[j, 1]

            dx = jx - ix
            dy = jy - iy

            L = np.sqrt(dx ** 2 + dy ** 2)
            alpha = np.arctan2(dy, dx)
            self.geo_props[x] = [A, I, r, L, alpha]

        # Assigning point loads to to top chord joints. First and last joints loaded half of the interior joints.
        p = self.q*self.l/(self.nd + 1)
        self.j_forces[self.nd,1] = p/2
        self.j_forces[2 * self.nd - 1, 1] = p / 2

        for i in range(self.nd+1, 2 * self.nd-1):
            self.j_forces[i,1] = p

        # Mapping joint forces to dof forces. Difference is due to boundary contitions
        for n, i in enumerate(self.n_dof):
            self.f[int(i[0])] = self.j_forces[n,0]
            self.f[int(i[1])] = self.j_forces[n,1]

    def truss_geo(self, offset=0):
        # this method is for creating drawing
        lines = []
        for i in self.n_conn:
            node_i = int(i[0])
            node_j = int(i[1])
            coord_ix = self.n_coord[node_i, 0]
            coord_iz = self.n_coord[node_i, 1] + offset
            coord_jx = self.n_coord[node_j, 0]
            coord_jz = self.n_coord[node_j, 1] + offset
            coord_i = (coord_ix, coord_iz)
            coord_j = (coord_jx, coord_jz)
            lines.append([coord_i, coord_j])
        return lines

    def stiffness(self):
        # Assembling stiffness matrix. Usual procedure.

        k_stiff = np.zeros((2*self.nn, 2*self.nn))
        k_loc = np.zeros((4, 4))
        r = np.zeros((4, 4))

        for i in range(self.nm):
            nodes = self.n_conn[i]
            dofs = self.n_dof[nodes,:].reshape(4)

            A = self.geo_props[i, 0]
            L = self.geo_props[i, 3]
            alpha = self.geo_props[i, 4]
            E = self.mate_props[i, 0]

            k11 = E*A/L
            k_loc[0, 0] = k11
            k_loc[2, 2] = k11
            k_loc[0, 2] = -k11
            k_loc[2, 0] = -k11

            r[0, 0] = np.cos(alpha)
            r[0, 1] = np.sin(alpha)
            r[1, 0] = -np.sin(alpha)
            r[1, 1] = np.cos(alpha)
            r[2, 2] = np.cos(alpha)
            r[2, 3] = np.sin(alpha)
            r[3, 2] = -np.sin(alpha)
            r[3, 3] = np.cos(alpha)

            k_gl = np.dot(np.dot(r.T, k_loc), r)

            for x in range(4):
                for y in range(4):
                    dof1 = int(dofs[x])
                    dof2 = int(dofs[y])
                    k_stiff[dof1, dof2] += k_gl[x,y]

        return  k_stiff


def analyze(nn, nm, conn, dof, mate, geo, nforce):
    k_stiff = np.zeros((2 * nn, 2 * nn))
    k_loc = np.zeros((4, 4))
    r = np.zeros((4, 4))

    for i in range(nm):
        nodes = conn[i]
        dofs = dof[nodes, :].reshape(4)

        A = geo[i, 0]
        L = geo[i, 3]
        alpha = geo[i, 4]
        E = mate[i, 0]

        k11 = E * A / L
        k_loc[0, 0] = k11
        k_loc[2, 2] = k11
        k_loc[0, 2] = -k11
        k_loc[2, 0] = -k11

        r[0, 0] = np.cos(alpha)
        r[0, 1] = np.sin(alpha)
        r[1, 0] = -np.sin(alpha)
        r[1, 1] = np.cos(alpha)
        r[2, 2] = np.cos(alpha)
        r[2, 3] = np.sin(alpha)
        r[3, 2] = -np.sin(alpha)
        r[3, 3] = np.cos(alpha)

        k_gl = np.dot(np.dot(r.T, k_loc), r)

        for x in range(4):
            for y in range(4):
                dof1 = dofs[x]
                dof2 = dofs[y]
                k_stiff[dof1, dof2] += k_gl[x, y]

    return k_stiff


def shs_props(B, t):
    sec_A = B * B - (B - 2 * t) * (B - 2 * t)
    sec_I = (B ** 4 - (B - 2 * t) ** 4) / 12
    sec_r = np.sqrt(sec_I / sec_A)
    return sec_A, sec_I, sec_r


parameters = {"h1": 2000, "h2": 4000, "l": 10000, "nd": 5,
              "dia": np.random.randint(2, size=5), "B": 250,
              "t": 6, "q":10}

T1 = Truss(parameters)

T1.stiffness()
lines = T1.truss_geo()

ax = plt.axes()
ax.set_xlim(-1000, 10000 + 1000)
ax.set_ylim(-1000, 1 * (12000 + 1000))
segments = LineCollection(lines, linewidths=2)
ax.add_collection(segments)
plt.show()
