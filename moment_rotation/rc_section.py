import o3seespy as o3
import numpy as np
import o3seespy.extensions


def get_moment_curvature(axial_load=100., max_curve=0.001, num_incr=500):
    osi = o3.OpenSeesInstance(ndm=2, ndf=3, state=3)

    fc = 4.0
    # e_mod = 57000.0 * np.sqrt(fc * 1000.0) / 1e3

    conc_conf = o3.uniaxial_material.Concrete01(osi, fpc=-5.0, epsc0=-0.005, fpcu=-3.5, eps_u=-0.02)
    conc_unconf = o3.uniaxial_material.Concrete01(osi, fpc=-fc, epsc0=-0.002, fpcu=0.0, eps_u=-0.006)
    rebar = o3.uniaxial_material.Steel01(osi, fy=60.0, e0=30000.0, b=0.02)

    h = 18.0
    b = 18.0
    cover = 2.5
    gj = 1.0E10
    nf_core_y = 8
    nf_core_z = 8
    nf_cover_y = 10
    nf_cover_z = 10
    n_bars = 3
    bar_area = 0.79

    edge_y = h / 2.0
    edge_z = b / 2.0
    core_y = edge_y - cover
    core_z = edge_z - cover

    sect = o3.section.Fiber(osi, gj=gj)
    # define the core patch
    o3.patch.Quad(osi, conc_conf, nf_core_z, nf_core_y,  # core, counter-clockwise (diagonals at corners)
                  crds_i=[-core_y, core_z],
                  crds_j=[-core_y, -core_z],
                  crds_k=[core_y, -core_z],
                  crds_l=[core_y, core_z])

    o3.patch.Quad(osi, conc_unconf, 1, nf_cover_y,  # right cover, counter-clockwise (diagonals at corners)
                  crds_i=[-edge_y, edge_z],
                  crds_j=[-core_y, core_z],
                  crds_k=[core_y, core_z],
                  crds_l=[edge_y, edge_z])
    o3.patch.Quad(osi, conc_unconf, 1, nf_cover_y,  # left cover
                  crds_i=[-core_y, -core_z],
                  crds_j=[-edge_y, -edge_z],
                  crds_k=[edge_y, -edge_z],
                  crds_l=[core_y, -core_z])
    o3.patch.Quad(osi, conc_unconf, nf_cover_z, 1,  # bottom cover
                  crds_i=[-edge_y, edge_z],
                  crds_j=[-edge_y, -edge_z],
                  crds_k=[-core_y, -core_z],
                  crds_l=[-core_y, core_z])
    o3.patch.Quad(osi, conc_unconf, nf_cover_z, 1,  # top cover
                  crds_i=[core_y, core_z],
                  crds_j=[core_y, -core_z],
                  crds_k=[edge_y, -edge_z],
                  crds_l=[edge_y, edge_z])

    o3.layer.Straight(osi, rebar, n_bars, bar_area, start=[-core_y, core_z], end=[-core_y, -core_z])
    o3.layer.Straight(osi, rebar, n_bars, bar_area, start=[core_y, core_z], end=[core_y, -core_z])

    spacing_y = 2 * core_y / (n_bars - 1)
    remaining_bars = n_bars - 2
    o3.layer.Straight(osi, rebar, remaining_bars, bar_area,
                      start=[core_y - spacing_y, core_z],
                      end=[-core_y + spacing_y, core_z])
    o3.layer.Straight(osi, rebar, remaining_bars, bar_area,
                      start=[core_y - spacing_y, -core_z],
                      end=[-core_y + spacing_y, -core_z])

    n1 = o3.node.Node(osi, 0.0, 0.0)
    n2 = o3.node.Node(osi, 0.0, 0.0)
    o3.Fix3DOF(osi, n1, 1, 1, 1)
    o3.Fix3DOF(osi, n2, 0, 1, 0)
    ele = o3.element.ZeroLengthSection(osi, [n1, n2], sect)

    nd = o3.recorder.NodeToArrayCache(osi, n2, dofs=[3], res_type='disp')
    nm = o3.recorder.NodeToArrayCache(osi, n1, dofs=[3], res_type='reaction')

    ts = o3.time_series.Constant(osi)
    o3.pattern.Plain(osi, ts)
    o3.Load(osi, n2, load_values=[axial_load, 0.0, 0.0])

    o3.system.BandGeneral(osi)
    o3.numberer.Plain(osi)
    o3.constraints.Plain(osi)
    o3.test.NormUnbalance(osi, tol=1.0e-9, max_iter=10)
    o3.algorithm.Newton(osi)
    o3.integrator.LoadControl(osi, incr=0.0)
    o3.analysis.Static(osi)
    o3.analyze(osi, 1)

    #
    ts = o3.time_series.Linear(osi)
    o3.pattern.Plain(osi, ts)
    o3.Load(osi, n2, load_values=[0.0, 0.0, 1.0])

    d_cur = max_curve / num_incr

    o3.integrator.DisplacementControl(osi, n2, o3.cc.DOF2D_ROTZ, d_cur, 1, d_cur, d_cur)
    o3.analyze(osi, num_incr)
    o3.wipe(osi)
    curvature = nd.collect()
    moment = -nm.collect()
    return moment, curvature


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    mom, curve = get_moment_curvature()
    plt.plot(mom)
    plt.show()
