
import sfsimodels
import eqsig

import numpy as np

import o3seespy as o3


def run_analysis(etype, asig):
    osi = o3.OpenSeesInstance(ndm=2, ndf=3)
    nodes = [o3.node.Node(osi, 0.0, 0.0),
             o3.node.Node(osi, 5.5, 0.0),
             o3.node.Node(osi, 0.0, 3.3),
             o3.node.Node(osi, 5.5, 3.3)
             ]
    o3.Mass2D(osi, nodes[2], 1e5, 1e5, 1e6)
    o3.Mass2D(osi, nodes[3], 1e5, 1e5, 1e6)
    o3.Fix3DOF(osi, nodes[0], o3.cc.FIXED, o3.cc.FIXED, o3.cc.FIXED)
    o3.Fix3DOF(osi, nodes[1], o3.cc.FIXED, o3.cc.FIXED, o3.cc.FIXED)

    steel_mat = o3.uniaxial_material.Steel01(osi, 300.0e6, 200.0e9, b=0.02)

    # o3.element.DispBeamColumn(osi, [nodes[2], nodes[3]], )
    tran = o3.geom_transf.Linear2D(osi)
    o3.element.ElasticBeamColumn2D(osi, [nodes[2], nodes[3]], 0.01, 200.0e9, iz=1.0e-4, transf=tran)
    o3.element.ElasticBeamColumn2D(osi, [nodes[0], nodes[2]], 0.01, 200.0e9, iz=1.0e-4, transf=tran)
    o3.element.ElasticBeamColumn2D(osi, [nodes[1], nodes[3]], 0.01, 200.0e9, iz=1.0e-4, transf=tran)

    a_series = o3.time_series.Path(osi, dt=asig.dt, values=-1 * asig.values)  # should be negative
    o3.pattern.UniformExcitation(osi, dir=o3.cc.X, accel_series=a_series)

    xi = 0.04
    angular_freqs = np.array(o3.get_eigen(osi, solver='fullGenLapack', n=4)) ** 0.5
    print('angular_freqs: ', angular_freqs)
    periods = 2 * np.pi / angular_freqs
    print('periods: ', periods)

    omega0 = 0.2
    omega1 = 10.0
    a1 = 2. * xi / (omega0 + omega1)
    a0 = a1 * omega0 * omega1

    beta_k = 2 * xi / angular_freqs[0]
    o3.rayleigh.Rayleigh(osi, alpha_m=a0, beta_k=a1, beta_k_init=0.0, beta_k_comm=0.0)

    # o3.ModalDamping(osi, [xi, xi])

    o3.constraints.Transformation(osi)
    o3.test_check.NormDispIncr(osi, tol=1.0e-5, max_iter=35, p_flag=0)
    o3.numberer.RCM(osi)
    if etype == 'implicit':
        o3.algorithm.Newton(osi)
        o3.system.FullGeneral(osi)
        o3.integrator.Newmark(osi, gamma=0.5, beta=0.25)
        dt = 0.01
    else:
        o3.algorithm.Linear(osi)

        if etype == 'newmark_explicit':
            o3.system.FullGeneral(osi)
            o3.integrator.NewmarkExplicit(osi, gamma=0.5)
            explicit_dt = periods[-1] / np.pi / 8
        elif etype == 'central_difference':
            o3.system.FullGeneral(osi)
            o3.integrator.CentralDifference(osi)
            explicit_dt = periods[-1] / np.pi / 8  # 0.5 is a factor of safety
        elif etype == 'hht_explicit':
            o3.integrator.HHTExplicit(osi, alpha=0.5)
            explicit_dt = periods[-1] / np.pi / 8
        elif etype == 'explicit_difference':
            # o3.opy.system('Diagonal')
            o3.system.Diagonal(osi)
            o3.integrator.ExplicitDifference(osi)
            explicit_dt = periods[-1] / np.pi / 4
        else:
            raise ValueError(etype)
        print('explicit_dt: ', explicit_dt)
        dt = explicit_dt
    o3.analysis.Transient(osi)

    roof_disp = o3.recorder.NodeToArrayCache(osi, nodes[2], dofs=[o3.cc.X], res_type='disp')
    time = o3.recorder.TimeToArrayCache(osi)
    ttotal = 10.0
    o3.analyze(osi, int(ttotal / dt), dt)
    o3.wipe(osi)
    return time.collect(), roof_disp.collect()


def run():
    import all_paths as ap
    import matplotlib.pyplot as plt
    record_filename = 'short_motion_dt0p01.txt'
    motion_step = 0.01
    rec = np.loadtxt(ap.MODULE_DATA_PATH + f'gms/{record_filename}', skiprows=2)

    xi = 0.05

    acc_signal = eqsig.AccSignal(rec, motion_step)

    etypes = ['implicit', 'central_difference', 'newmark_explicit', 'explicit_difference']  # ,
    ls = ['-', '--', ':', '-.']
    for i, etype in enumerate(etypes):
        time, roof_d = run_analysis(etype, acc_signal)
        plt.plot(time, roof_d, label=etype, ls=ls[i])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    run()