import numpy as np
import o3seespy as o3


def create():
    """
    Check slider behaviour when orientated horizontally
    Returns
    -------

    """

    osi = o3.OpenSeesInstance(ndm=3, ndf=3)
    top_node = o3.node.Node(osi, 0, 0, 0)
    above_slider_node = o3.node.Node(osi, 0, 0, 0)
    below_slider_node = o3.node.Node(osi, 0, 0, 0)

    o3.Fix3DOF(osi, below_slider_node, o3.cc.FIXED, o3.cc.FIXED, o3.cc.FIXED)
    kn = 1e4
    p_mat = o3.uniaxial_material.Elastic(osi, kn)
    p_mat_l = o3.uniaxial_material.Elastic(osi, kn / 1e6)
    m_mat = o3.uniaxial_material.Elastic(osi, 1e10)
    top_ele = o3.element.ZeroLength(osi, [top_node, above_slider_node], [p_mat, p_mat, p_mat], [o3.cc.X, o3.cc.Y, o3.cc.DOF3D_Z])

    coh = 10.0
    print('coh: ', coh)
    frc_nds = [above_slider_node, below_slider_node]  # correct order
    # frc_nds = [below_slider_node, above_slider_node]
    fric_ele = o3.element.ZeroLengthContact3D(osi, frc_nds, kn, kt=kn, mu=0, c=coh, direction=2)
    # fric_ele = o3.element.ZeroLength(osi, frc_nds, [p_mat, p_mat, p_mat], [o3.cc.X, o3.cc.Y, o3.cc.DOF3D_Z])
    o3.element.ZeroLength(osi, frc_nds, [p_mat_l, p_mat_l, p_mat_l], [o3.cc.X, o3.cc.Y, o3.cc.DOF3D_Z])

    ts = o3.time_series.Linear(osi, 1)
    o3.pattern.Plain(osi, ts)
    o3.Load(osi, top_node, [0.0, -50.0, 0.0])  # vertical load

    tol = 1.0e-3
    o3.constraints.Transformation(osi)
    o3.numberer.RCM(osi)
    o3.system.FullGeneral(osi)
    n_steps_gravity = 100
    rate = 1. / n_steps_gravity
    o3.integrator.LoadControl(osi, rate, num_iter=10)
    o3.test_check.NormDispIncr(osi, tol, 10)
    o3.algorithm.Linear(osi)
    o3.analysis.Static(osi)
    o3.analyze(osi, n_steps_gravity)

    print('beam: ', o3.get_ele_response(osi, top_ele, 'force'))
    print('ndisp: ', o3.get_node_disps(osi, top_node, [o3.cc.X, o3.cc.Y, o3.cc.DOF2D_ROTZ]))
    print('fric: ', o3.get_ele_response(osi, fric_ele, 'force'))
    o3.load_constant(osi, time=0)

    ts = o3.time_series.Linear(osi, 1)
    o3.pattern.Plain(osi, ts)
    o3.SP(osi, top_node, o3.cc.X, [1])

    o3.analyze(osi, 100)

    print('beam: ', o3.get_ele_response(osi, top_ele, 'force'))
    print('fric: ', o3.get_ele_response(osi, fric_ele, 'force'))
    fresp = o3.get_ele_response(osi, fric_ele, 'force')

    print(fresp[0], coh)
    assert np.isclose(fresp[0], coh, rtol=0.01)


if __name__ == '__main__':
    create()

