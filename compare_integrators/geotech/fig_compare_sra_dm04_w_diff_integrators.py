import o3seespy as o3
import copy
import sfsimodels as sm
import numpy as np
import eqsig
import all_paths as ap
# for linear analysis comparison
import liquepy as lq


def site_response(sp, asig, freqs=(0.5, 10), xi=0.03, dy=0.5, analysis_time=None, outs=None,
                  rec_dt=None, etype='implicit', forder=1.0):
    """
    Run seismic analysis of a soil profile

    Parameters
    ----------
    sp: sfsimodels.SoilProfile object
        A soil profile
    asig: eqsig.AccSignal object
        An acceleration signal

    Returns
    -------

    """
    if analysis_time is None:
        analysis_time = asig.time[-1]
    if outs is None:
        outs = {'ACCX': [0]}  # Export the horizontal acceleration at the surface

    osi = o3.OpenSeesInstance(ndm=2, ndf=2, state=3)
    assert isinstance(sp, sm.SoilProfile)
    sp.gen_split(props=['shear_vel', 'unit_mass'], target=dy)
    thicknesses = sp.split["thickness"]
    n_node_rows = len(thicknesses) + 1
    node_depths = np.cumsum(sp.split["thickness"])
    node_depths = np.insert(node_depths, 0, 0)
    ele_depths = (node_depths[1:] + node_depths[:-1]) / 2
    unit_masses = sp.split["unit_mass"] / forder

    grav = 9.81

    ele_width = min(thicknesses)
    total_soil_nodes = len(thicknesses) * 2 + 2

    # Define nodes and set boundary conditions for simple shear deformation
    # Start at top and build down?
    sn = [[o3.node.Node(osi, 0, 0), o3.node.Node(osi, ele_width, 0)]]
    for i in range(1, n_node_rows):
        # Establish left and right nodes
        sn.append([o3.node.Node(osi, 0, -node_depths[i]),
                    o3.node.Node(osi, ele_width, -node_depths[i])])
        # set x and y dofs equal for left and right nodes
        o3.EqualDOF(osi, sn[i][0], sn[i][1], [o3.cc.X, o3.cc.Y])

    # Fix base nodes
    o3.Fix2DOF(osi, sn[-1][0], o3.cc.FIXED, o3.cc.FIXED)
    o3.Fix2DOF(osi, sn[-1][1], o3.cc.FIXED, o3.cc.FIXED)

    # define materials
    ele_thick = 1.0  # m
    soil_mats = []
    eles = []
    prev_id = -1
    for i in range(len(thicknesses)):
        y_depth = ele_depths[i]

        sl_id = sp.get_layer_index_by_depth(y_depth)
        sl = sp.layer(sl_id)
        mat = sl.o3_mat
        if sl_id != prev_id:
            mat.build(osi)
            soil_mats.append(mat)
            prev_id = sl_id

        # def element
        nodes = [sn[i+1][0], sn[i+1][1], sn[i][1], sn[i][0]]  # anti-clockwise
        # eles.append(o3.element.SSPquad(osi, nodes, mat, o3.cc.PLANE_STRAIN, ele_thick, 0.0, grav * unit_masses[i]))
        eles.append(o3.element.Quad(osi, nodes, mat=mat, otype=o3.cc.PLANE_STRAIN, thick=ele_thick, rho=unit_masses[i], b1=0.0, b2=-grav))

    for i, soil_mat in enumerate(soil_mats):
        if hasattr(soil_mat, 'update_to_linear'):
            print('Update model to linear')
            soil_mat.update_to_linear()

    # Gravity analysis
    o3.constraints.Transformation(osi)
    o3.test_check.NormDispIncr(osi, tol=1.0e-5, max_iter=30, p_flag=0)
    o3.algorithm.Newton(osi)
    o3.numberer.RCM(osi)
    o3.system.ProfileSPD(osi)
    o3.integrator.Newmark(osi, 5./6, 4./9)  # include numerical damping
    o3.analysis.Transient(osi)
    omega_1 = 2 * np.pi * freqs[0]
    omega_2 = 2 * np.pi * freqs[1]
    xi_grav = 0.2
    a0 = 2 * xi_grav * omega_1 * omega_2 / (omega_1 + omega_2)
    a1 = 2 * xi_grav / (omega_1 + omega_2)
    o3.rayleigh.Rayleigh(osi, a0, 0, a1, 0)
    o3.analyze(osi, 200, 1.)

    for i, soil_mat in enumerate(soil_mats):
        if hasattr(soil_mat, 'update_to_nonlinear'):
            print('Update model to nonlinear')
            soil_mat.update_to_nonlinear()
    show_grav = 0
    if show_grav:
        nx = o3.recorder.NodeToArrayCache(osi, sn[0][0], dofs=[o3.cc.X, o3.cc.Y], res_type='disp', close_on_write=True)
        o3.analyze(osi, 2000, 0.005)
        import matplotlib.pyplot as plt
        plt.plot(nx.collect())
        plt.show()
    # reset time and analysis
    o3.rayleigh.Rayleigh(osi, 0, 0, 0, 0)
    o3.set_time(osi, 0.0)


    o3.wipe_analysis(osi)

    ods = {}
    for otype in outs:
        if otype == 'ACCX':

            ods['ACCX'] = []
            if isinstance(outs['ACCX'], str) and outs['ACCX'] == 'all':
                ods['ACCX'] = o3.recorder.NodesToArrayCache(osi, nodes=sn[:][0], dofs=[o3.cc.X], res_type='accel', dt=rec_dt)
            else:
                for i in range(len(outs['ACCX'])):
                    ind = np.argmin(abs(node_depths - outs['ACCX'][i]))
                    ods['ACCX'].append(o3.recorder.NodeToArrayCache(osi, node=sn[ind][0], dofs=[o3.cc.X], res_type='accel', dt=rec_dt))
        if otype == 'TAUXY':
            ods['TAUXY'] = []
            if isinstance(outs['TAUXY'], str) and outs['TAUXY'] == 'all':
                ods['TAUXY'] = o3.recorder.ElementsToArrayCache(osi, eles=eles, arg_vals=['stress'], dt=rec_dt)
            else:
                for i in range(len(outs['TAU'])):
                    ind = np.argmin(abs(ele_depths - outs['TAU'][i]))
                    ods['TAU'].append(o3.recorder.ElementToArrayCache(osi, ele=eles[ind], arg_vals=['stress'], dt=rec_dt))

        if otype == 'STRS':
            ods['STRS'] = []
            if isinstance(outs['STRS'], str) and outs['STRS'] == 'all':
                ods['STRS'] = o3.recorder.ElementsToArrayCache(osi, eles=eles, arg_vals=['strain'], dt=rec_dt)
            else:
                for i in range(len(outs['STRS'])):
                    ind = np.argmin(abs(ele_depths - outs['STRS'][i]))
                    ods['STRS'].append(o3.recorder.ElementToArrayCache(osi, ele=eles[ind], arg_vals=['strain'], dt=rec_dt))

    # Define the dynamic analysis
    acc_series = o3.time_series.Path(osi, dt=asig.dt, values=asig.values)
    o3.pattern.UniformExcitation(osi, dir=o3.cc.X, accel_series=acc_series)

    # Run the dynamic analysis

    n = 4
    # omegas = np.array(o3.get_eigen(osi, solver='fullGenLapack', n=n)) ** 0.5  # DO NOT USE fullGenLapack
    omegas = np.array(o3.get_eigen(osi, n=n)) ** 0.5
    periods = 2 * np.pi / omegas
    print('response_periods: ', periods)

    o3.constraints.Transformation(osi)
    o3.test_check.NormDispIncr(osi, tol=1.0e-6, max_iter=10, p_flag=2)
    o3.numberer.RCM(osi)
    if etype == 'implicit':
        # o3.algorithm.Newton(osi)
        o3.system.ProfileSPD(osi)
        # o3.integrator.Newmark(osi, gamma=0.5, beta=0.25)
        # o3.algorithm.NewtonLineSearch(osi, 0.75)
        o3.algorithm.Newton(osi)
        o3.integrator.Newmark(osi, 0.5, 0.25)
        dt = 0.001
    else:
        o3.algorithm.Linear(osi)
        if etype == 'newmark_explicit':
            o3.system.ProfileSPD(osi)
            o3.integrator.NewmarkExplicit(osi, gamma=0.5)
            explicit_dt = periods[-1] / np.pi / 64
        elif etype == 'central_difference':
            o3.system.ProfileSPD(osi)
            o3.integrator.CentralDifference(osi)
            explicit_dt = periods[-1] / np.pi / 64
        elif etype == 'hht_explicit':
            o3.integrator.HHTExplicit(osi, alpha=0.5)
            explicit_dt = periods[-1] / np.pi / 8
        elif etype == 'explicit_difference':
            # o3.system.FullGeneral(osi)
            o3.system.Diagonal(osi)
            o3.integrator.ExplicitDifference(osi)
            explicit_dt = periods[-1] / np.pi / 64
        else:
            raise ValueError(etype)
        print('explicit_dt: ', explicit_dt)
        dt = explicit_dt

    if etype in ['newmark_explicit', 'central_difference', 'implicit']:  # Does not support modal damping
        omega_1 = 2 * np.pi * freqs[0]
        omega_2 = 2 * np.pi * freqs[1]
        a0 = 2 * xi * omega_1 * omega_2 / (omega_1 + omega_2)
        a1 = 2 * xi / (omega_1 + omega_2)
        o3.rayleigh.Rayleigh(osi, a0, 0, a1, 0)
    else:
        o3.ModalDamping(osi, [xi])
    o3.analysis.Transient(osi)
    for i in range(300):
        print(i)
        o3.analyze(osi, 10, dt)
    # o3.analyze(osi, int(analysis_time / dt), dt)

    o3.wipe(osi)
    out_dict = {}
    for otype in ods:
        if isinstance(ods[otype], list):
            out_dict[otype] = []
            for i in range(len(ods[otype])):
                out_dict[otype].append(ods[otype][i].collect())
            out_dict[otype] = np.array(out_dict[otype])
        else:
            out_dict[otype] = ods[otype].collect().T
    out_dict['time'] = np.arange(0, analysis_time, dt)

    return out_dict


def gen_mz_toyoura_sand_dafalias_and_manzari_2004():
    sl = lq.num.models.ManzariDafaliasModel()
    sl.name = 'mz_toyoura_sand_dafalias_and_manzari_2004'
    sl.g0 = 125.0
    sl.poissons_ratio = 0.05  # Note kementzetzidis_2018 consider v=0.3
    sl.m_c = 1.25
    sl.c_c = 0.712
    sl.lambda_c = 0.019
    sl.e_0 = 0.934
    sl.ksi = 0.7
    sl.m_yield = 0.01
    sl.h_0 = 7.05
    sl.c_h = 0.968
    sl.n_b = 1.1
    sl.a_0 = 0.704
    sl.n_d = 3.5
    sl.z_max = 4
    sl.c_z = 600
    sl.e_min = 0.574  # Fioravante:2015ea
    sl.e_max = 0.923  # Fioravante:2015ea
    sl.specific_gravity = 2.65
    sl.p_atm = 101.0e3
    return sl


def run():
    soil_profile = sm.SoilProfile()
    soil_profile.height = 10.0
    xi = 0.03

    sl = gen_mz_toyoura_sand_dafalias_and_manzari_2004()
    sl.relative_density = 0.6
    soil_profile.add_layer(0.0, sl)
    sl.xi = xi

    # Define o3 soil object
    nu_init = 0.05
    f_order = 1.0
    soil_mat = o3.nd_material.ManzariDafalias(None, g0=sl.g0, nu=nu_init, e_init=sl.e_curr, m_c=sl.m_c,
                                              c_c=sl.c_c, lambda_c=sl.lambda_c, e_0=sl.e_0, ksi=sl.ksi,
                                              p_atm=sl.p_atm / f_order,
                                              m_yield=sl.m_yield, h_0=sl.h_0, c_h=sl.c_h, n_b=sl.n_b, a_0=sl.a_0,
                                              n_d=sl.n_d, z_max=sl.z_max, c_z=sl.c_z, den=sl.unit_dry_mass / f_order)
    sl.o3_mat = soil_mat

    sl_base = sm.Soil()
    vs = 350.
    unit_mass = 1700.0
    sl_base.g_mod = vs ** 2 * unit_mass
    sl_base.poissons_ratio = 0.0
    sl_base.unit_dry_weight = unit_mass * 9.8
    sl_base.xi = xi  # for linear analysis
    e_mod = 2 * sl_base.g_mod * (1 + sl_base.poissons_ratio)
    sl_base.o3_mat = o3.nd_material.ElasticIsotropic(None, e_mod=e_mod, nu=sl_base.poissons_ratio, rho=sl_base.unit_dry_mass)
    soil_profile.add_layer(8., sl_base)
    gm_scale_factor = 0.5
    record_filename = 'short_motion_dt0p01.txt'
    in_sig = eqsig.load_asig(ap.MODULE_DATA_PATH + 'gms/' + record_filename, m=gm_scale_factor)

    # analysis with pysra
    od = lq.sra.run_pysra(soil_profile, in_sig, odepths=np.array([0.0]), wave_field='within')
    pysra_sig = eqsig.AccSignal(od['ACCX'][0], in_sig.dt)

    import matplotlib.pyplot as plt
    from bwplot import cbox
    bf, sps = plt.subplots(nrows=3, figsize=(6, 8))

    in_sig.smooth_fa_frequencies = in_sig.fa_frequencies[1:]
    pysra_sig.smooth_fa_frequencies = in_sig.fa_frequencies[1:]
    sps[0].plot(in_sig.time, in_sig.values, c='k', label='Input')
    sps[0].plot(pysra_sig.time, pysra_sig.values, c='r', label='pysra')
    sps[1].plot(in_sig.fa_frequencies, abs(in_sig.fa_spectrum), c='k')
    sps[1].plot(pysra_sig.fa_frequencies, abs(pysra_sig.fa_spectrum), c='r')
    # pysra_h = pysra_sig.smooth_fa_spectrum / in_sig.smooth_fa_spectrum
    # sps[2].plot(pysra_sig.smooth_fa_frequencies, pysra_h, c='r')

    # analysis with O3
    etypes = ['implicit', 'explicit_difference', 'central_difference', 'newmark_explicit']
    # etypes = ['central_difference']
    etypes = ['implicit']
    # etypes = ['explicit_difference']
    ls = ['-', '--', ':', '-.']
    outs = {'ACCX': [0], 'TAUXY': 'all', 'STRS': 'all'}
    for i, etype in enumerate(etypes):
        outputs_exp = site_response(soil_profile, in_sig, freqs=(0.5, 10), xi=xi, etype=etype, outs=outs)
        resp_dt = outputs_exp['time'][2] - outputs_exp['time'][1]
        surf_sig = eqsig.AccSignal(outputs_exp['ACCX'][0], resp_dt)
        surf_sig.smooth_fa_frequencies = in_sig.fa_frequencies[1:]
        sps[0].plot(surf_sig.time, surf_sig.values, c=cbox(i), label=etype, ls=ls[i])
        sps[1].plot(surf_sig.fa_frequencies, abs(surf_sig.fa_spectrum), c=cbox(i), ls=ls[i])
        # h = surf_sig.smooth_fa_spectrum / in_sig.smooth_fa_spectrum
        # sps[2].plot(surf_sig.smooth_fa_frequencies, h, c=cbox(i), ls=ls[i])
        sps[2].plot(outputs_exp['STRS'][6], outputs_exp['TAUXY'][6])

    # sps[2].axhline(1, c='k', ls='--')
    sps[1].set_xlim([0, 20])

    sps[0].legend(prop={'size': 6})
    plt.show()

    # assert np.isclose(o3_surf_vals, pysra_sig.values, atol=0.01, rtol=100).all()


if __name__ == '__main__':
    run()

