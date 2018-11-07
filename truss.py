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
from numpy.linalg import inv
import random as rnd


class Truss:
    # Truss Class.
    def __init__(self, DNA, param):
        self.point = 0  # Fitness point set to zero at initiation

        self.h1 = DNA[0]  # Shoulder Height of the Truss in mm
        self.h2 = DNA[1]  # Peak Height of the Truss in mm
        self.nd = DNA[2]  # Number of Divisions of the Truss
        self.dia = DNA[3]  # Alignment of the Diagonals for Each Bay

        self.B = [shs_catalog[int(i)][0] for i in DNA[4]]  # Height of SHS member
        self.t = [shs_catalog[int(i)][1] for i in DNA[4]]  # Wall Thickness of SHS member in mm

        self.L = param[0]  # Half of the Span Length of the Truss
        self.q = param[1]  # Line Load on the Truss in kN/m

        self.nn = 2 * self.nd + 2  # Numer of Nodes
        self.nm = 4 * self.nd + 1  # Number of Members
        self.Lines = []  # Line list for plotting
        self.j_forces = np.zeros((self.nn, 2))  # Joint Force Matrix
        self.f = np.zeros((self.nn * 2, 1))  # Dof Force Vector - will be mapped from j_forces
        self.u = np.zeros((self.nn * 2, 1))  # Deformation Field
        self.p = np.zeros((self.nm, 1))  # Member Axial Forces
        self.util = np.zeros((self.nm, 1))  # Member Utilization Ratios

        xvals = np.array([np.hstack((np.linspace(0, self.L, self.nd + 1), np.linspace(0, self.L, self.nd + 1)))]).T
        # Nodal coordinates - in X axis

        yvals = np.array([np.hstack((np.zeros(self.nd + 1), np.linspace(self.h1, self.h2, self.nd + 1)))]).T
        # Nodal coordinates - in Y axis

        self.n_coord = np.hstack((xvals, yvals))  # Nodal Coordinates are stacked in n_coord
        self.n_conn = np.zeros((self.nm, 2))  # Connectivity matrix
        self.n_bound = np.array([[0, 1], [0, 2], [self.nd, 1], [self.nd * 2 + 1, 1]])  # Introducing boundary cnds.
        self.n_dof = np.zeros((self.nn, 2))  # Nodal Dof matrix - info on which dof is assigned to which joint
        self.m_type = np.zeros((self.nm, 1))  # Member Type bot chord :0, top chord :1, post :2, diagonal :3
        self.geo_props = []  # Geometric props. of each member.
        self.mate_props = []  # Material props. of each member.
        self.sec_props = []  # Section props. of each member.

        # Populating connectivity matrix
        for i in range(self.nd):
            self.n_conn[i] = [i, i + 1]
            if i < self.nd / 2:
                mt = 0
            else:
                mt = 1
            self.m_type[i] = mt

        for i in range(self.nd):
            self.n_conn[i + self.nd] = [i + self.nd + 1, i + self.nd + 2]
            if i < self.nd / 2:
                mt = 0
            else:
                mt = 1
            self.m_type[i + self.nd] = 2 + mt

        for i in range(self.nd + 1):
            self.n_conn[i + 2 * self.nd] = [i, i + self.nd + 1]
            if i < self.nd / 2:
                mt = 0
            else:
                mt = 1
            self.m_type[i + 2 * self.nd] = 4 + mt

        for i in range(self.nd):

            if i < self.nd / 2:
                mt = 0
            else:
                mt = 1
            self.m_type[i + 3 * self.nd + 1] = 6 + mt

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
            self.mate_props.append((E, fy))

        for i in range(self.nm):
            mt = int(self.m_type[i])
            A, I, r = shs_props(self.B[mt], self.t[mt])
            self.sec_props.append((self.B[mt], self.t[mt], A, I, r))

        # Geometric Properties
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
            self.geo_props.append((L, alpha))

        # Assigning point loads to to top chord joints. First and last joints loaded half of the interior joints.
        p = self.q * self.L / self.nd
        self.j_forces[self.nd + 1, 1] = p / 2
        self.j_forces[2 * self.nd + 1, 1] = p / 2

        for i in range(self.nd + 2, 2 * self.nd + 1):
            self.j_forces[i, 1] = p

        # Mapping joint forces to dof forces. Difference is due to boundary conditions
        # Assembling Force Vector
        for n, i in enumerate(self.n_dof):
            self.f[int(i[0]), 0] = self.j_forces[n, 0]
            self.f[int(i[1]), 0] = self.j_forces[n, 1]

        self.W = 0
        for n,i in enumerate(self.sec_props):
            L, alpha = self.geo_props[n]
            B, t, A, I, ir = i
            self.W += L*A*7.85e-3

    def truss_geo(self, offset=0, deformed=False, ud=0, scale=20):
        # this method is for creating drawing
        u = np.zeros((self.nn * 2, 1))
        if deformed:
            u = ud
        lines = []
        for i in self.n_conn:
            node_i = int(i[0])
            node_j = int(i[1])
            dofi_1 = int(self.n_dof[node_i, 1])
            dofj_0 = int(self.n_dof[node_j, 0])
            dofj_1 = int(self.n_dof[node_j, 1])
            dofi_0 = int(self.n_dof[node_i, 0])
            coord_ix = self.n_coord[node_i, 0] + u[dofi_0, 0] * scale
            coord_iz = self.n_coord[node_i, 1] + offset + u[dofi_1, 0] * scale
            coord_jx = self.n_coord[node_j, 0] + u[dofj_0, 0] * scale
            coord_jz = self.n_coord[node_j, 1] + offset + u[dofj_1, 0] * scale
            coord_i = (coord_ix, coord_iz)
            coord_j = (coord_jx, coord_jz)
            lines.append([coord_i, coord_j])
        return lines

    def stiffness(self):
        # Assembling stiffness matrix. Usual procedure.
        k_stiff = np.zeros((2 * self.nn, 2 * self.nn))
        k_loc = np.zeros((4, 4))
        r = np.zeros((4, 4))

        for i in range(self.nm):
            nodes = self.n_conn[i]
            dofs = self.n_dof[nodes, :].reshape(4)

            L, alpha = self.geo_props[i]
            B, t, A, I, ir = self.sec_props[i]
            E, fy = self.mate_props[i]

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
                    dof1 = int(dofs[x])
                    dof2 = int(dofs[y])
                    k_stiff[dof1, dof2] += k_gl[x, y]

        return k_stiff

    def analyze(self):
        k_loc = np.zeros((4, 4))
        r = np.zeros((4, 4))
        for i, n in enumerate(self.sec_props):
            nodes = self.n_conn[i]
            dofs = self.n_dof[nodes, :].reshape(4).astype(int)
            u_mem = self.u[dofs, :]

            L, alpha = self.geo_props[i]
            B, t, A, I, ir = n
            E, fy = self.mate_props[i]

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

            f = np.dot(k_loc, np.dot(r, u_mem))
            self.p[i] = (f[2] - f[0]) / 2
            self.util[i] = mem_design(self.p[i], n, self.geo_props[i], self.mate_props[i])


def disp(f, k, n_bc):
    n_dof = len(f)
    kff = k[:-n_bc, :-n_bc]
    # kfr = k[:-n_bc, -n_bc:]
    krf = k[-n_bc:, :-n_bc]
    # krr = k[-n_bc:, -n_bc:]

    ff = f[:-n_bc]
    ff.shape = (n_dof - n_bc, 1)
    # fr = f[-n_bc:]
    # fr.shape = (n_bc,1)

    ur = np.zeros((n_bc, 1))

    uf = np.dot(inv(kff), ff)  # - np.dot(kfr, ur)))
    r = np.dot(krf, uf)  # + np.dot(krr, ur) - fr

    return uf


# def stiff(nn, nm, conn, dof, mate, geo):
#     # Stiffness assembly, force assembly and matrix inversion
#     k_stiff = np.zeros((2 * nn, 2 * nn))
#     k_loc = np.zeros((4, 4))
#     r = np.zeros((4, 4))
#
#     for i in range(nm):
#         nodes = conn[i]
#         dofs = dof[nodes, :].reshape(4)
#
#         A = geo[i, 0]
#         L = geo[i, 3]
#         alpha = geo[i, 4]
#         E = mate[i, 0]
#
#         k11 = E * A / L
#         k_loc[0, 0] = k11
#         k_loc[2, 2] = k11
#         k_loc[0, 2] = -k11
#         k_loc[2, 0] = -k11
#
#         r[0, 0] = np.cos(alpha)
#         r[0, 1] = np.sin(alpha)
#         r[1, 0] = -np.sin(alpha)
#         r[1, 1] = np.cos(alpha)
#         r[2, 2] = np.cos(alpha)
#         r[2, 3] = np.sin(alpha)
#         r[3, 2] = -np.sin(alpha)
#         r[3, 3] = np.cos(alpha)
#
#         k_gl = np.dot(np.dot(r.T, k_loc), r)
#
#         for x in range(4):
#             for y in range(4):
#                 dof1 = dofs[x]
#                 dof2 = dofs[y]
#                 k_stiff[dof1, dof2] += k_gl[x, y]
#
#     return k_stiff


def shs_props(B, t):
    sec_A = B * B - (B - 2 * t) * (B - 2 * t)
    sec_I = (B ** 4 - (B - 2 * t) ** 4) / 12
    sec_r = np.sqrt(sec_I / sec_A)
    return sec_A, sec_I, sec_r


def mem_design(N, sec, geo, mat):
    B, t, A, I, r = sec
    L, alpha = geo
    E, fy = mat

    if N > 0:
        Pn = 0.9 * A * fy
        return N / Pn
    else:
        b = B - 2 * t
        lmd = b / t
        lmdr = 1.4 * np.sqrt(E / fy)
        Fe = (np.pi ** 2) * E / (L / r)
        if L / r < 4.71 * np.sqrt(E / fy):
            Fcr = fy * (0.658 ** (fy / Fe))
        else:
            Fcr = 0.877 * Fe

        if lmd < lmdr * np.sqrt(E / fy):
            # Section fully compact
            Pn = 0.9 * A * Fcr
            return abs(N / Pn)
        else:
            c1 = 0.18
            c2 = 1.31
            Fel = c2 * lmdr * fy / lmd
            U = (1 - c1 * np.sqrt(Fel / fy)) * np.sqrt(Fel / fy)
            Ae = U * A
            Pn = 0.9 * Ae * Fcr
            return abs(N / Pn)


def population(size, param, const, seed=0):
    if seed:
        rnd.seed(seed)
    pop = []
    for _ in range(size):
        h1 = rnd.randrange(param[0][0], param[0][1], param[0][2])
        h2 = rnd.randrange(max(h1,param[1][0]), param[1][1], param[1][2])
        n_div = rnd.randint(param[2][0], param[2][1])
        division = [rnd.randrange(0, 3, 1) for j in range(n_div)]
        sec = []
        cnt = len(shs_catalog)
        for i in range(8):
            sec.append(rnd.randint(0, cnt-1))

        pop.append(Truss([h1,h2,n_div,division,sec],const))
    return pop


def pop_analyze(trusses):
    wg = np.zeros((len(trusses),2))
    for n,i in enumerate(trusses):
        wg[n] = [n,i.W]
        i.analyze()
    ws = wg[wg[:,1].argsort()]

    w_pnt = 1000
    w_decr = w_pnt/len(trusses)
    util_pnt = 1000
    el_pnt = -10

    for n,i in enumerate(ws):
        trusses[int(i[0])].point -= w_decr * n  # Weight Point

    for i in trusses:

        for j in i.util:
            i.point += util_pnt * j * np.sign(0.95 - j)  #  Utilization Point

        i.point += el_pnt * i.nm  # Member Count Point

        print(i.point)






def fitness(mem):
    out = 0




shs_catalog = [[20, 2], [30, 2], [40, 2], [40, 3], [40, 4], [50, 2], [50, 3], [50, 4], [50, 5], [60, 3], [60, 2],
               [60, 4], [70, 3], [60, 5], [70, 4],
               [70, 5], [80, 3], [80, 4], [80, 5], [80, 6], [90, 3], [90, 4], [90, 5], [90, 6], [100, 5], [100, 4],
               [100, 6], [120, 5], [120, 4],
               [120, 6], [120, 8], [140, 4], [140, 5], [140, 6], [140, 8], [140, 10], [150, 5], [150, 6], [150, 8],
               [160, 4], [150, 10],
               [160, 5], [160, 8], [160, 6], [160, 10], [180, 6], [180, 8], [180, 10], [180, 12.5], [200, 6], [200, 8],
               [200, 10], [200, 12.5],
               [220, 8], [220, 12.5], [220, 10], [250, 6], [250, 10], [250, 8], [250, 12.5], [260, 8], [260, 10],
               [260, 12.5], [300, 6],
               [300, 8], [300, 10], [300, 12.5], [350, 8], [350, 10], [400, 10], [350, 12.5], [400, 12.5]]


parameters = ([1000,2000,250],[1000,4000,250],[3,10])
constraints = (10000,-20)

Trusses = population(100,parameters,constraints)


pop_analyze(Trusses)

T1 = Trusses[0]
k = T1.stiffness()
u = np.vstack((disp(T1.f, k, 4), np.zeros((4, 1))))
T1.u = u

lines = T1.truss_geo()

ax = plt.axes()
ax.set_xlim(-1000, 10000 + 1000)
ax.set_ylim(-3000, 1 * (12000 + 1000))
segments = LineCollection(lines, linewidths=2)
ax.add_collection(segments)

lines = T1.truss_geo(deformed=True, ud=u)

segments = LineCollection(lines, linewidths=2,color="r")
ax.add_collection(segments)

plt.show()
