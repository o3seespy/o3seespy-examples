import numpy as np
import o3seespy as o3


def run(etype, xi):
    mass = 100.0
    k = 100.0
    omega = np.sqrt(k / mass)
    c = 2 * xi * k / omega
    osi = o3.OpenSeesInstance(ndm=1, ndf=1)
    mat = o3.uniaxial_material.Elastic(osi, k)
    dashpot_mat = o3.uniaxial_material.Viscous(osi, big_c=c, alpha=1.)

    n0 = o3.node.Node(osi, 0.0)
    n0a = o3.node.Node(osi, 0.0)
    n1 = o3.node.Node(osi, 0.0)
    n2 = o3.node.Node(osi, 0.0)

    o3.Mass2D(osi, n1, x_mass=mass)
    o3.Mass2D(osi, n2, x_mass=mass)
    o3.Fix1DOF(osi, n0, x=o3.cc.FIXED)

    o3.element.ZeroLength(osi, [n1, n2], mats=[mat], dirs=[o3.cc.X])
    if xi > 0.0:
        o3.element.ZeroLength(osi, [n0, n0a], mats=[dashpot_mat], dirs=[o3.cc.X])
    o3.EqualDOF(osi, n1, n0a, dofs=[o3.cc.X])
    ts0 = o3.time_series.Linear(osi)
    o3.pattern.Plain(osi, ts0)
    o3.Load(osi, n1, [10])
    min_dt = 2. / omega
    print('min_dt: ', min_dt)

    o3.constraints.Transformation(osi)
    o3.test_check.NormDispIncr(osi, tol=1.0e-6, max_iter=10)
    o3.numberer.RCM(osi)
    if etype == 'implicit':
        o3.system.ProfileSPD(osi)
        o3.algorithm.NewtonLineSearch(osi, 0.75)
        o3.integrator.Newmark(osi, 0.5, 0.25)
        dt = min_dt
    else:
        o3.algorithm.Linear(osi, factor_once=True)
        if etype == 'newmark_explicit':
            o3.system.ProfileSPD(osi)
            o3.integrator.NewmarkExplicit(osi, gamma=0.5)
            explicit_dt = min_dt / 1
        elif etype == 'central_difference':
            o3.system.FullGeneral(osi)
            o3.integrator.CentralDifference(osi)
            explicit_dt = min_dt / 1
        elif etype == 'explicit_difference':
            o3.system.Diagonal(osi)
            o3.integrator.ExplicitDifference(osi)
            explicit_dt = min_dt / 1
        else:
            raise ValueError(etype)
        ndp = np.ceil(np.log10(explicit_dt))
        if 0.5 * 10 ** ndp < explicit_dt:
            dt = 0.5 * 10 ** ndp
        elif 0.2 * 10 ** ndp < explicit_dt:
            dt = 0.2 * 10 ** ndp
        elif 0.1 * 10 ** ndp < explicit_dt:
            dt = 0.1 * 10 ** ndp
        else:
            raise ValueError(explicit_dt, 0.1 * 10 ** ndp)
        print('explicit_dt: ', explicit_dt, dt)

    o3.analysis.Transient(osi)
    ttotal = 150  # s
    inc = 2
    nn = int(ttotal / dt / inc)
    print(nn)
    time = []
    disp1 = []
    disp2 = []
    for i in range(nn):
        time.append(o3.get_time(osi))
        disp1.append(o3.get_node_disp(osi, n1, dof=o3.cc.X))
        disp2.append(o3.get_node_disp(osi, n2, dof=o3.cc.X))
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
    et = 'explicit_difference'
    t, d1, d2 = run(et, xi)
    ax[0].plot(t, d1, ls='-.', label=et)
    ax[1].plot(t, d2, ls='-.')
    et = 'newmark_explicit'
    t, d1, d2 = run(et, xi)
    ax[0].plot(t, d1, ls=':', label=et)
    ax[1].plot(t, d2, ls=':')
    ax[0].legend()
    plt.show()

