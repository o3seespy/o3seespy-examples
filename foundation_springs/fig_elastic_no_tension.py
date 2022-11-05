import numpy as np
import o3seespy as o3


def create(show):
    """
    Check ENT behaviour when orientated vertically
    Returns
    -------

    """
    osi = o3.OpenSeesInstance(ndm=2, ndf=3)

    above_conn_node = o3.node.Node(osi, 0, 0)
    below_conn_node = o3.node.Node(osi, 0, 0)
    o3.Fix3DOF(osi, above_conn_node, o3.cc.FIXED, o3.cc.FREE, o3.cc.FIXED)
    o3.Fix3DOF(osi, below_conn_node, o3.cc.FIXED, o3.cc.FIXED, o3.cc.FIXED)

    use_ent = 1
    if use_ent:
        p_mat = o3.uniaxial_material.ENT(osi, 1e4)
    else:
        p_mat = o3.uniaxial_material.Elastic(osi, 1e1, eneg=1e4)
    sec_mat = o3.uniaxial_material.Elastic(osi, 1e4)
    ory = [1, 0, 0, 0, 1, 0]  # When pressing down - there is no support (default)
    # ory = [-1, 0, 0, 0, -1, 0]  # When pressing down - there is support
    main_ele = o3.element.ZeroLength(osi, [above_conn_node, below_conn_node], [p_mat], dirs=[o3.cc.Y], orient=ory)
    sec_ele = o3.element.ZeroLength(osi, [above_conn_node, below_conn_node], [sec_mat], dirs=[o3.cc.Y], orient=ory)
    ts = o3.time_series.Path(osi, time=[0, 10, 30, 1e4], values=[0, 1, -1, -1])
    o3.pattern.Plain(osi, ts)
    o3.Load(osi, above_conn_node, [0.0, -50.0, 0.0])

    tol = 1.0e-3
    o3.constraints.Transformation(osi)
    o3.numberer.RCM(osi)
    o3.system.BandGeneral(osi)
    rate = 0.1 * 10
    o3.integrator.LoadControl(osi, rate, num_iter=10)
    o3.test_check.NormDispIncr(osi, tol, 10)
    o3.algorithm.Linear(osi)
    o3.analysis.Static(osi)
    print(o3.get_ele_response(osi, main_ele, 'xaxis'))
    print(o3.get_ele_response(osi, main_ele, 'yaxis'))
    print(o3.get_ele_response(osi, main_ele, 'zaxis'))
    # return
    mf = [o3.get_ele_response(osi, main_ele, 'force')[1]]
    sf = [o3.get_ele_response(osi, sec_ele, 'force')[1]]
    disp = [o3.get_node_disp(osi, above_conn_node, o3.cc.Y)]
    for i in range(30):
        o3.analyze(osi, 1)
        mf.append(o3.get_ele_response(osi, main_ele, 'force')[1])
        sf.append(o3.get_ele_response(osi, sec_ele, 'force')[1])
        disp.append(o3.get_node_disp(osi, above_conn_node, o3.cc.Y))
    mf = np.array(mf)
    sf = np.array(sf)

    assert np.isclose(np.interp(10, np.arange(len(mf)), mf), 0.0)
    assert np.isclose(np.interp(30, np.arange(len(mf)), mf), 25.0)

    if show:
        import matplotlib.pyplot as plt
        bf, ax = plt.subplots(nrows=3)
        ax[0].plot(disp)
        ax[1].plot(mf, c='r')
        ax[1].plot(sf, c='b')
        ax[1].plot(sf + mf, c='purple', ls='--')
        ax[2].plot(disp, mf)
        ax[2].plot(disp, sf)
        ax[2].plot(disp, sf + mf)

        plt.show()

    print('main: ', o3.get_ele_response(osi, main_ele, 'force'))
    print('sec: ', o3.get_ele_response(osi, sec_ele, 'force'))
    print('ndisp: ', o3.get_node_disps(osi, above_conn_node, [o3.cc.X, o3.cc.Y, o3.cc.DOF2D_ROTZ]))


if __name__ == '__main__':
    create(show=1)


