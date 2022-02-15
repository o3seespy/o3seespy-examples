import numpy as np
import o3seespy as o3


def run(etype, xi):
    rho = 10.0
    e_mod = 100.0e3
    nu = 0.3
    osi = o3.OpenSeesInstance(ndm=2, ndf=2)
    n_node_rows = 2

    ele_width = 1.0
    ele_thick = 1.0
    ele_h = 1.0
    node_depths = np.arange(n_node_rows) * ele_h
    grav = 0.0
    eles = []
    sn = [[o3.node.Node(osi, 0, 0), o3.node.Node(osi, ele_width, 0)]]
    for i in range(1, n_node_rows):
        # Establish left and right nodes
        sn.append([o3.node.Node(osi, 0, -node_depths[i]), o3.node.Node(osi, ele_width, -node_depths[i])])
        # set x and y dofs equal for left and right nodes
        if i != n_node_rows - 1:  # Note this is crucial otherwise explicit solver does not compute correctly!!!
            o3.EqualDOF(osi, sn[i][0], sn[i][1], [o3.cc.X, o3.cc.Y])

        mat = o3.nd_material.ElasticIsotropic(osi, e_mod, nu, rho)
        nodes = [sn[i][0], sn[i][1], sn[i-1][1], sn[i-1][0]]  # anti-clockwise
        eles.append(o3.element.SSPquad(osi, nodes, mat, o3.cc.PLANE_STRAIN, ele_thick, 0.0, -grav))

    # Fix base nodes
    o3.Fix2DOF(osi, sn[-1][0], o3.cc.FREE, o3.cc.FIXED)
    o3.Fix2DOF(osi, sn[-1][1], o3.cc.FREE, o3.cc.FIXED)
    o3.EqualDOF(osi, sn[-1][0], sn[-1][1], [o3.cc.X])
    # Define dashpot nodes
    dashpot_node_1 = o3.node.Node(osi, 0, -node_depths[-1])
    o3.Fix2DOF(osi, dashpot_node_1, o3.cc.FIXED, o3.cc.FIXED)
    dashpot_node_2 = sn[-1][0]

    g_mod = e_mod / (2 * (1 + nu))
    v_s = np.sqrt(g_mod / rho)
    c_base = ele_width * rho * v_s
    dashpot_mat = o3.uniaxial_material.Viscous(osi, c_base, alpha=1.)
    o3.element.ZeroLength(osi, [dashpot_node_1, dashpot_node_2], mats=[dashpot_mat], dirs=[o3.cc.DOF2D_X])
    # spring_mat = o3.uniaxial_material.Elastic(osi, e_mod * 0.1)
    # o3.element.ZeroLength(osi, [dashpot_node_l, dashpot_node_2], mats=[spring_mat], dirs=[o3.cc.DOF2D_X])


    lam = 2 * g_mod * nu / (1 - 2 * nu)
    mu = g_mod
    v_dil = np.sqrt((lam + 2 * mu) / rho)
    min_dt = ele_h / v_dil
    ts0 = o3.time_series.Linear(osi)
    o3.pattern.Plain(osi, ts0)
    o3.Load(osi, sn[-1][0], [1., 0.])
    print('min_dt: ', min_dt)

    o3.constraints.Transformation(osi)
    o3.test_check.NormDispIncr(osi, tol=1.0e-6, max_iter=10)
    o3.numberer.RCM(osi)
    if etype == 'implicit':
        o3.system.ProfileSPD(osi)
        o3.algorithm.NewtonLineSearch(osi, 0.75)
        o3.integrator.Newmark(osi, 0.5, 0.25)
    else:
        o3.algorithm.Linear(osi, factor_once=True)
        if etype == 'newmark_explicit':
            o3.system.ProfileSPD(osi)
            o3.integrator.NewmarkExplicit(osi, gamma=0.5)
        elif etype == 'central_difference':
            o3.system.FullGeneral(osi)
            o3.integrator.CentralDifference(osi)
        elif etype == 'explicit_difference':
            o3.system.Diagonal(osi)
            o3.integrator.ExplicitDifference(osi)
        else:
            raise ValueError(etype)
    ndp = np.ceil(np.log10(min_dt))
    if 0.5 * 10 ** ndp < min_dt:
        dt = 0.5 * 10 ** ndp
    elif 0.2 * 10 ** ndp < min_dt:
        dt = 0.2 * 10 ** ndp
    elif 0.1 * 10 ** ndp < min_dt:
        dt = 0.1 * 10 ** ndp
    else:
        raise ValueError(min_dt, 0.1 * 10 ** ndp)
    print('dt: ', min_dt, dt)

    o3.analysis.Transient(osi)
    ttotal = 15  # s
    inc = 2
    nn = int(ttotal / dt / inc)
    print(nn)
    time = []
    disp1 = []
    disp2 = []
    for i in range(nn):
        time.append(o3.get_time(osi))
        disp1.append(o3.get_node_disp(osi, sn[0][0], dof=o3.cc.X))
        disp2.append(o3.get_node_disp(osi, sn[-1][0], dof=o3.cc.X))
        o3.analyze(osi, inc, dt)

    return time, disp1, disp2


if __name__ == '__main__':
    # run('implicit', 0.05)

    import matplotlib.pyplot as plt
    bf, ax = plt.subplots(nrows=2)
    xi = 0.2
    et = 'implicit'
    t, d1, d2 = run(et, xi)
    ax[0].plot(t, d1, label=et)
    ax[1].plot(t, d2, ls='-')
    et = 'central_difference'
    t, d1, d2 = run(et, xi)
    ax[0].plot(t, d1, ls='--', label=et)
    ax[1].plot(t, d2, ls='--')
    # et = 'explicit_difference'
    # t, d1, d2 = run(et, xi)
    # ax[0].plot(t, d1, ls='-.', label=et)
    # ax[1].plot(t, d2, ls='-.')
    # et = 'newmark_explicit'
    # t, d1, d2 = run(et, xi)
    # ax[0].plot(t, d1, ls=':', label=et)
    # ax[1].plot(t, d2, ls=':')
    ax[0].legend()
    name = __file__.replace('.py', '')
    name = name.split("fig_")[-1]
    bf.suptitle(name)
    bf.savefig(f'figs/{name}.png', dpi=90)
    plt.show()

