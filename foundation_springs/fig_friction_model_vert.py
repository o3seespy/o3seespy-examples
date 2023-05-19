import numpy as np
import o3seespy as o3


def create():
    """
    Check slider behaviour when orientated horizontally
    Returns
    -------

    """
    osi = o3.OpenSeesInstance(ndm=2, ndf=3)

    top_node = o3.node.Node(osi, 0, 5)
    above_slider_node = o3.node.Node(osi, 0, 0)
    below_slider_node = o3.node.Node(osi, 0, 0)
    o3.Fix3DOF(osi, below_slider_node, o3.cc.FIXED, o3.cc.FIXED, o3.cc.FIXED)

    tran = o3.geom_transf.Linear2D(osi)
    beam_ele = o3.element.ElasticBeamColumn2D(osi, [top_node, above_slider_node], 0.01, 200.0e9, iz=1.0e-4, transf=tran)

    phi_interf = 30.0
    mu = np.tan(np.radians(phi_interf))
    print('mu: ', mu)
    sf_frn = o3.friction_model.Coulomb(osi, mu)
    k_add_shear = 1e12
    p_mat = o3.uniaxial_material.Elastic(osi, 1e1, eneg=1e10)
    m_mat = o3.uniaxial_material.Elastic(osi, 1e10)
    ory = [0, -1, 0, 1, 0, 0]
    fric_ele = o3.element.FlatSliderBearing2D(osi, [above_slider_node, below_slider_node], sf_frn, k_add_shear, p_mat=p_mat,
                              mz_mat=m_mat, orient=ory)
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
    print('fric: ', o3.get_ele_response(osi, fric_ele, 'force'))
    o3.load_constant(osi, time=0)

    ts = o3.time_series.Linear(osi, 1)
    o3.pattern.Plain(osi, ts)
    o3.SP(osi, top_node, o3.cc.X, [1])
    o3.analyze(osi, 100)

    print('beam: ', o3.get_ele_response(osi, beam_ele, 'force'))
    print('fric: ', o3.get_ele_response(osi, fric_ele, 'force'))
    fresp = o3.get_ele_response(osi, fric_ele, 'force')
    frat = fresp[0] / -fresp[1]
    print(frat, mu)
    assert np.isclose(frat, mu, rtol=0.01)


if __name__ == '__main__':
    create()

