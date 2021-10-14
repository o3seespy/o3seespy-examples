import o3seespy as o3
import math
import numpy as np
from bwplot import cbox, lsbox
import pandas as pd
import all_paths as ap
import timeit


def run_ts_custom_strain(mat, esig_v0, strains, osi=None, nu_dyn=None, target_d_inc=0.00001, k0=None, etype='newmark_explicit',
                         handle='silent', verbose=0, opyfile=None, dss=False, plain_strain=True, min_n=10, nl=True):
    # if dss:
    #     raise ValueError('dss option is not working')
    damp = 0.05
    omega0 = 0.2
    omega1 = 20.0
    a1 = 2. * damp / (omega0 + omega1)
    a0 = a1 * omega0 * omega1
    if osi is None:
        osi = o3.OpenSeesInstance(ndm=2, ndf=2)
        mat.build(osi)

    # Establish nodes
    h_ele = 1.
    nodes = [
        o3.node.Node(osi, 0.0, 0.0),
        o3.node.Node(osi, h_ele, 0.0),
        o3.node.Node(osi, h_ele, h_ele),
        o3.node.Node(osi, 0.0, h_ele)
    ]

    # Fix bottom node
    o3.Fix2DOF(osi, nodes[0], o3.cc.FIXED, o3.cc.FIXED)
    if k0 is None:
        o3.Fix2DOF(osi, nodes[1], o3.cc.FIXED, o3.cc.FIXED)
        # Set out-of-plane DOFs to be slaved
        o3.EqualDOF(osi, nodes[2], nodes[3], [o3.cc.X, o3.cc.Y])
    else:  # control k0 with node forces
        o3.Fix2DOF(osi, nodes[1], o3.cc.FIXED, o3.cc.FREE)

    if plain_strain:
        oop = 'PlaneStrain'
    else:
        oop = 'PlaneStress'

    ele = o3.element.SSPquad(osi, nodes, mat, oop, 1, 0.0, 0.0)

    # o3.rayleigh.Rayleigh(osi, a0, a1, 0.0, 0.0)
    angular_freqs = np.array(o3.get_eigen(osi, solver='fullGenLapack', n=2)) ** 0.5
    print('angular_freqs: ', angular_freqs)
    periods = 2 * np.pi / angular_freqs

    print('periods: ', periods)
    o3.constraints.Transformation(osi)
    o3.test_check.NormDispIncr(osi, tol=1.0e-6, max_iter=35, p_flag=0)
    o3.numberer.RCM(osi)
    if etype == 'implicit':
        o3.algorithm.Newton(osi)
        o3.system.SparseGeneral(osi)
        o3.integrator.Newmark(osi, gamma=0.5, beta=0.25)
        dt = 0.01
    else:
        o3.algorithm.Linear(osi, factor_once=True)
        o3.system.FullGeneral(osi)
        if etype == 'newmark_explicit':
            o3.integrator.NewmarkExplicit(osi, gamma=0.6)
            explicit_dt = periods[0] / np.pi / 8
        elif etype == 'central_difference':
            o3.integrator.CentralDifference(osi)
            explicit_dt = periods[0] / np.pi / 8  # 0.5 is a factor of safety
        elif etype == 'hht_explicit':
            o3.integrator.HHTExplicit(osi, alpha=0.5)
            explicit_dt = periods[0] / np.pi / 8
        elif etype == 'explicit_difference':
            o3.integrator.ExplicitDifference(osi)
            explicit_dt = periods[0] / np.pi / 32
        else:
            raise ValueError(etype)
        print('explicit_dt: ', explicit_dt)
        dt = explicit_dt
    o3.analysis.Transient(osi)

    o3.update_material_stage(osi, mat, stage=0)

    # dt = 0.00001
    tload = 20
    n_steps = tload / dt
    # Add static vertical pressure and stress bias
    time_series = o3.time_series.Path(osi, time=[0, tload, 1e10], values=[0, 1, 1])
    o3.pattern.Plain(osi, time_series)
    # ts0 = o3.time_series.Linear(osi, factor=1)
    # o3.pattern.Plain(osi, ts0)

    if k0:
        o3.Load(osi, nodes[2], [-esig_v0 / 2, -esig_v0 / 2])
        o3.Load(osi, nodes[3], [esig_v0 / 2, -esig_v0 / 2])
        o3.Load(osi, nodes[1], [-esig_v0 / 2, 0])
        # node 0 is fixed
    else:
        o3.Load(osi, nodes[2], [0, -esig_v0 / 2])
        o3.Load(osi, nodes[3], [0, -esig_v0 / 2])

    print('Apply init stress to elastic element')
    o3.analyze(osi, num_inc=n_steps, dt=dt)
    stresses = o3.get_ele_response(osi, ele, 'stress')
    print('init_stress0: ', stresses)
    o3.load_constant(osi, tload)

    if hasattr(mat, 'update_to_nonlinear') and nl:
        print('set to nonlinear')
        mat.update_to_nonlinear()
        o3.analyze(osi, 10000, dt=dt)
    # if not nl:
    #     mat.update_to_linear()
    if nu_dyn is not None:
        mat.set_nu(nu_dyn, eles=[ele])
        o3.analyze(osi, 10000, dt=dt)

    # o3.extensions.to_py_file(osi)
    stresses = o3.get_ele_response(osi, ele, 'stress')
    print('init_stress1: ', stresses)

    # Prepare for reading results
    exit_code = None
    stresses = o3.get_ele_response(osi, ele, 'stress')
    if dss:
        o3.gen_reactions(osi)
        force0 = o3.get_node_reaction(osi, nodes[2], o3.cc.DOF2D_X)
        force1 = o3.get_node_reaction(osi, nodes[3], o3.cc.DOF2D_X)
        # force2 = o3.get_node_reaction(osi, nodes[0], o3.cc.DOF2D_X)
        stress = [force1 + force0]
        strain = [o3.get_node_disp(osi, nodes[2], dof=o3.cc.DOF2D_X)]
        sxy_ind = None
        gxy_ind = None
        # iforce0 = o3.get_node_reaction(osi, nodes[0], o3.cc.DOF2D_X)
        # iforce1 = o3.get_node_reaction(osi, nodes[1], o3.cc.DOF2D_X)
        # iforce2 = o3.get_node_reaction(osi, nodes[2], o3.cc.DOF2D_X)
        # iforce3 = o3.get_node_reaction(osi, nodes[3], o3.cc.DOF2D_X)
        # print(iforce0, iforce1, iforce2, iforce3, stresses[2])
    else:
        ro = o3.recorder.load_recorder_options()
        import pandas as pd
        df = pd.read_csv(ro)
        mat_type = ele.mat.type
        dfe = df[(df['mat'] == mat_type) & (df['form'] == oop)]
        df_sxy = dfe[dfe['recorder'] == 'stress']
        outs = df_sxy['outs'].iloc[0].split('-')
        sxy_ind = outs.index('sxy')

        df_gxy = dfe[dfe['recorder'] == 'strain']
        outs = df_gxy['outs'].iloc[0].split('-')
        gxy_ind = outs.index('gxy')
        stress = [stresses[sxy_ind]]
        cur_strains = o3.get_ele_response(osi, ele, 'strain')
        strain = [cur_strains[gxy_ind]]

    time_series = o3.time_series.Path(osi, time=[0, tload, 1e10], values=[0, 1, 1])
    o3.pattern.Plain(osi, time_series)
    disps = list(np.array(strains) * 1)
    d_per_dt = 0.01
    diff_disps = np.diff(disps, prepend=0)
    time_incs = np.abs(diff_disps) / d_per_dt
    approx_n_steps = time_incs / dt
    time_incs = np.where(approx_n_steps < 800, 800 * dt, time_incs)
    approx_n_steps = time_incs / dt
    assert min(approx_n_steps) >= 8, approx_n_steps
    curr_time = o3.get_time(osi)
    times = list(np.cumsum(time_incs) + curr_time)
    disps.append(disps[-1])
    times.append(1e10)

    disps = list(disps)
    n_steps_p2 = int((times[-2] - curr_time) / dt) + 10

    print('n_steps: ', n_steps_p2)
    times.insert(0, curr_time)
    disps.insert(0, 0.0)

    init_disp = o3.get_node_disp(osi, nodes[2], dof=o3.cc.X)

    disps = list(np.array(disps) + init_disp)
    ts0 = o3.time_series.Path(osi, time=times, values=disps, factor=1)
    pat0 = o3.pattern.Plain(osi, ts0)
    o3.SP(osi, nodes[2], dof=o3.cc.X, dof_values=[1])
    o3.SP(osi, nodes[3], dof=o3.cc.X, dof_values=[1])
    print('init_disp: ', init_disp)
    print('path -times: ', times)
    print('path -values: ', disps)

    v_eff = [stresses[1]]
    h_eff = [stresses[0]]
    time = [o3.get_time(osi)]
    for i in range(int(n_steps_p2 / 200)):
        print(i / (n_steps_p2 / 200))
        fail = o3.analyze(osi, 200, dt=dt)
        o3.gen_reactions(osi)

        stresses = o3.get_ele_response(osi, ele, 'stress')
        v_eff.append(stresses[1])
        h_eff.append(stresses[0])
        if dss:
            o3.gen_reactions(osi)
            force0 = o3.get_node_reaction(osi, nodes[2], o3.cc.DOF2D_X)
            force1 = o3.get_node_reaction(osi, nodes[3], o3.cc.DOF2D_X)
            stress.append(force1 + force0)
            strain.append(o3.get_node_disp(osi, nodes[2], dof=o3.cc.DOF2D_X))
        else:
            stress.append(stresses[sxy_ind])
            cur_strains = o3.get_ele_response(osi, ele, 'strain')
            strain.append(cur_strains[gxy_ind])
        time.append(o3.get_time(osi))

        if fail:
            break

    return np.array(stress), np.array(strain)-init_disp, np.array(v_eff), np.array(h_eff), np.array(time), exit_code


def run_example(show=0):
    import o3seespy as o3

    osi = None

    if show:
        import matplotlib.pyplot as plt
        bf, sps = plt.subplots(nrows=2)
        peak_strains = [0.0005, 0.0001, 0.0007]
        peak_strains = np.array([0.0005, 0.0001, 0.0007]) * 1
        esig_v0 = 100.0e3
        poissons_ratio = 0.3
        g_mod = 30.0e6
        unit_dry_mass = 1.6e3
        etypes = ['implicit', 'newmark_explicit', 'explicit_difference', 'central_difference']
        # etypes = ['implicit']
        for i, etype in enumerate(etypes):
            e_mod = 2 * g_mod * (1 + poissons_ratio)
            mat = o3.nd_material.ElasticIsotropic(osi, e_mod=e_mod, nu=poissons_ratio, rho=unit_dry_mass)
            stime = timeit.timeit()
            ss, es, vp, hp, time, error = run_ts_custom_strain(mat, esig_v0=esig_v0, strains=peak_strains, osi=osi,
                                                 target_d_inc=1.0e-4, nl=True, dss=True, etype=etype)
            etime = timeit.timeit()
            atime = etime - stime
            # ss /= 1e3
            data = {'ss': ss, 'es': es, 'vp': vp, 'hp': hp}
            dfo = pd.DataFrame.from_dict(data)

            sps[0].plot(es, ss, label=f'{etype}', c=cbox(i), ls=lsbox(i))
            sps[1].plot(time, ss, label=f'{etype}', c=cbox(i), ls=lsbox(i))
        ylims = np.array(sps[0].get_ylim())
        sps[0].plot(ylims / g_mod, ylims, c='k', ls=':', lw=0.7)
        sps[0].legend()
        sps[1].legend()

        plt.show()


if __name__ == '__main__':
    run_example(show=1)
