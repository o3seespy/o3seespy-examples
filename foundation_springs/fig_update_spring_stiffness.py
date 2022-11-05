import numpy as np
import o3seespy as o3


def create(use_single_ele=0):
    """
    Check slider behaviour when orientated horizontally
    Returns
    -------

    """
    osi = o3.OpenSeesInstance(ndm=2, ndf=3)

    top_node = o3.node.Node(osi, 0, 5)
    above_spring_node = o3.node.Node(osi, 0, 0)
    below_spring_node = o3.node.Node(osi, 0, 0)
    o3.Fix3DOF(osi, below_spring_node, o3.cc.FIXED, o3.cc.FIXED, o3.cc.FIXED)

    tran = o3.geom_transf.Linear2D(osi)
    beam_ele = o3.element.ElasticBeamColumn2D(osi, [top_node, above_spring_node], 0.01, 200.0e9, iz=1.0e-4, transf=tran)

    k1 = 1e10
    p_mat = o3.uniaxial_material.Elastic(osi, k1, eneg=1e1)
    h_mat = o3.uniaxial_material.Elastic(osi, k1, eneg=k1)
    m_mat = o3.uniaxial_material.Elastic(osi, k1, eneg=k1)

    if use_single_ele:
        sl_ele = o3.element.ZeroLength(osi, [above_spring_node, below_spring_node], [h_mat, p_mat, m_mat],
                                       [o3.cc.X, o3.cc.Y, o3.cc.DOF2D_ROTZ])
    else:
        sl_ele = o3.element.ZeroLength(osi, [above_spring_node, below_spring_node], [p_mat, m_mat],
                                       [o3.cc.Y, o3.cc.DOF2D_ROTZ])
        sl_ele_h = o3.element.ZeroLength(osi, [above_spring_node, below_spring_node], [h_mat], [o3.cc.X])
    ts = o3.time_series.Linear(osi, 1)
    o3.pattern.Plain(osi, ts)
    o3.Load(osi, top_node, [0.0, -50.0, 0.0])

    tol = 1.0e-3
    o3.constraints.Transformation(osi)
    o3.numberer.RCM(osi)
    o3.system.BandGeneral(osi)
    n_steps_gravity = 10
    rate = 1. / n_steps_gravity
    o3.integrator.LoadControl(osi, rate, num_iter=10)
    o3.test_check.NormDispIncr(osi, tol, 10)
    o3.algorithm.Linear(osi)
    o3.analysis.Static(osi)
    o3.analyze(osi, n_steps_gravity)

    print('beam: ', o3.get_ele_response(osi, beam_ele, 'force'))
    print('ndisp: ', o3.get_node_disps(osi, top_node, [o3.cc.X, o3.cc.Y, o3.cc.DOF2D_ROTZ]))
    print('fric: ', o3.get_ele_response(osi, sl_ele, 'force'))
    o3.load_constant(osi, time=0)

    ts = o3.time_series.Linear(osi, 1)
    o3.pattern.Plain(osi, ts)
    # o3.SP(osi, top_node, o3.cc.X, [1])
    o3.Load(osi, top_node, [20, 0.0, 0.0])
    o3.analyze(osi, int(1 / rate))
    o3.load_constant(osi, time=0)

    print('beam: ', o3.get_ele_response(osi, beam_ele, 'force'))
    print('fric: ', o3.get_ele_response(osi, sl_ele, 'force'))
    fresp = o3.get_ele_response(osi, sl_ele, 'force')
    print('ndisp: ', o3.get_node_disps(osi, above_spring_node, [o3.cc.X, o3.cc.Y, o3.cc.DOF2D_ROTZ]))
    if use_single_ele:
        h_mat.set_e_mod(k1 * 0.01, eles=[sl_ele])  # Note: this updates all spring dofs
    else:
        h_mat.set_e_mod(k1 * 0.01, eles=[sl_ele_h])
    o3.analyze(osi, 10)
    fd_disp = o3.get_node_disps(osi, above_spring_node, [o3.cc.X, o3.cc.Y, o3.cc.DOF2D_ROTZ])
    print('ndisp: ', fd_disp)
    if use_single_ele:
        sl_resp = o3.get_ele_response(osi, sl_ele, 'force')
    else:
        sl_resp = np.array(o3.get_ele_response(osi, sl_ele, 'force')) + np.array(o3.get_ele_response(osi, sl_ele_h, 'force'))
    ks = np.array(sl_resp)[:3] / np.array(fd_disp)
    print('k: ', ks)
    print('sl force: ', sl_resp)
    if use_single_ele:
        assert np.isclose(ks[0], 1e8)
        assert np.isclose(ks[1], 1e8)
        assert np.isclose(ks[2], 1e8)
    else:
        assert np.isclose(ks[0], 1e8)
        assert np.isclose(ks[1], 1e10)
        assert np.isclose(ks[2], 1e10)


if __name__ == '__main__':
    create(1)

