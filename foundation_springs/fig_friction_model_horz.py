import numpy as np
import o3seespy as o3


def create():
    """
    Check slider behaviour when orientated horizontally
    Returns
    -------

    """
    osi = o3.OpenSeesInstance(ndm=2, ndf=3)

    left_node = o3.node.Node(osi, 5, 0)
    left_of_slider_node = o3.node.Node(osi, 0, 0)
    right_of_slider_node = o3.node.Node(osi, 0.0, 0.0)
    o3.Fix3DOF(osi, right_of_slider_node, o3.cc.FIXED, o3.cc.FIXED, o3.cc.FIXED)

    tran = o3.geom_transf.Linear2D(osi)
    beam_ele = o3.element.ElasticBeamColumn2D(osi, [left_node, left_of_slider_node], 0.01, 200.0e9, iz=1.0e-4, transf=tran)

    phi_interf = 30.0
    mu = np.tan(np.radians(phi_interf))
    print('mu: ', mu)
    sf_frn = o3.friction_model.Coulomb(osi, mu)
    k_add_shear = 1e12
    p_mat = o3.uniaxial_material.Elastic(osi, 1e12)
    ory = [1, 0, 0, 0, 1, 0]
    fric_ele = o3.element.FlatSliderBearing2D(osi, [left_of_slider_node, right_of_slider_node], sf_frn, k_add_shear, p_mat=p_mat,
                              mz_mat=p_mat, orient=ory)
    ts = o3.time_series.Linear(osi, 1)
    o3.pattern.Plain(osi, ts)
    o3.Load(osi, left_node, [50.0, 0.0, 0.0])

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
    print('fric: ', o3.get_ele_response(osi, fric_ele, 'force'))
    o3.load_constant(osi, time=0)

    ts = o3.time_series.Linear(osi, 1)
    o3.pattern.Plain(osi, ts)
    o3.SP(osi, left_node, o3.cc.Y, [1])
    o3.analyze(osi, 100)

    print('beam: ', o3.get_ele_response(osi, beam_ele, 'force'))
    print('fric: ', o3.get_ele_response(osi, fric_ele, 'force'))
    bresp = o3.get_ele_response(osi, beam_ele, 'force')
    print(bresp[1] / bresp[0], mu)
    fresp = o3.get_ele_response(osi, fric_ele, 'force')
    frat = fresp[1] / fresp[0]
    print(frat, mu)
    assert np.isclose(frat, mu, rtol=0.01)


if __name__ == '__main__':
    create()

