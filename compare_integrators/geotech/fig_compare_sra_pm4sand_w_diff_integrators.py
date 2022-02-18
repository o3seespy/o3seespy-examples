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
    t_min = min(sp.split["thickness"] / sp.split['shear_vel']) / 4
    req_dt = t_min / np.pi  # undamped case, damped case: 2 / omega * (sqrt(1+xi^2) - xi)
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
        eles.append(o3.element.SSPquad(osi, nodes, mat, o3.cc.PLANE_STRAIN, ele_thick, 0.0, -grav))

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
    for i in range(400):
        o3.analyze(osi, 1, 0.1)
        # print('disps: ', o3.get_node_disp(osi, sn[0][0]))

    for i, soil_mat in enumerate(soil_mats):
        if hasattr(soil_mat, 'update_to_nonlinear'):
            print('Update model to nonlinear')
            soil_mat.update_to_nonlinear()
    for ele in eles:
        o3.set_parameter(osi, value=0, eles=[ele], args=['FirstCall', ele.mat.tag])
    print('run nonlinear gravity analysis')
    verbose = 0
    for i in range(400):
        o3.analyze(osi, 1, 0.001)
        if verbose:
            print('disps: ', o3.get_node_disp(osi, sn[0][0]))
            print('s0:', o3.get_ele_response(osi, eles[0], 'stress'))
            print('s1:', o3.get_ele_response(osi, eles[-1], 'stress'))
    print('finished nonlinear gravity analysis')

    # reset time and analysis
    o3.set_time(osi, 0.0)
    o3.wipe_analysis(osi)

    n = 12
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
        o3.algorithm.NewtonLineSearch(osi, 0.75)
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
            # o3.opy.system('Diagonal')
            # o3.system.FullGeneral(osi)
            o3.system.Diagonal(osi)
            o3.integrator.ExplicitDifference(osi)
            explicit_dt = periods[-1] / np.pi / 64
            explicit_dt = req_dt / 10
        else:
            raise ValueError(etype)
        print('explicit_dt: ', explicit_dt)
        ndp = np.ceil(np.log10(explicit_dt))
        if 0.5 * 10 ** ndp < explicit_dt:
            dt = 0.5 * 10 ** ndp
        elif 0.2 * 10 ** ndp < explicit_dt:
            dt = 0.2 * 10 ** ndp
        elif 0.1 * 10 ** ndp < explicit_dt:
            dt = 0.1 * 10 ** ndp
        else:
            raise ValueError(explicit_dt, 0.1 * 10 ** ndp)
    use_modal_damping = 0
    if not use_modal_damping:
        omega_1 = 2 * np.pi * freqs[0]
        omega_2 = 2 * np.pi * freqs[1]
        a0 = 2 * xi * omega_1 * omega_2 / (omega_1 + omega_2)
        a1 = 2 * xi / (omega_1 + omega_2)
        o3.rayleigh.Rayleigh(osi, a0, 0, a1, 0)
    else:
        o3.ModalDamping(osi, [xi])
    o3.analysis.Transient(osi)

    o3.test_check.NormDispIncr(osi, tol=1.0e-7, max_iter=10)
    rec_dt = 0.001

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
        if otype == 'TAU':
            ods['TAU'] = []
            if isinstance(outs['TAU'], str) and outs['TAU'] == 'all':
                ods['TAU'] = o3.recorder.ElementsToArrayCache(osi, eles=eles, arg_vals=['stress'], dt=rec_dt)
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
    ods['time'] = o3.recorder.TimeToArrayCache(osi, dt=rec_dt)

    acc_series = o3.time_series.Path(osi, dt=asig.dt, values=asig.values)
    o3.pattern.UniformExcitation(osi, dir=o3.cc.X, accel_series=acc_series)

    # Run the dynamic analysis

    inc = 1
    if etype != 'implicit':
        inc = 10
    o3.record(osi)
    while o3.get_time(osi) < analysis_time:
        print(o3.get_time(osi))
        print('s-mid:', o3.get_ele_response(osi, eles[5], 'stress'))
        # print('s1:', o3.get_ele_response(osi, eles[-1], 'stress'))
        if o3.analyze(osi, inc, dt):
            print('failed')
            break

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
    # out_dict['time'] = np.arange(0, analysis_time, rec_dt)

    return out_dict


def run():
    forder = 1.0e3
    # forder = 1
    soil_profile = sm.SoilProfile()

    sl = lq.num.models.PM4Sand()
    soil_profile.add_layer(0.0, sl)
    unit_mass = 16.0e3 / 9.8
    esig_v0 = 100.0e3
    sl.phi = 33.0
    g_mod = 31.8e6
    sl.poissons_ratio = 0.3
    sl.set_g0_mod_at_v_eff_stress(esig_v0, g_mod)
    sl.g0_mod = 316.
    sl.h_po = 0.833
    sl.relative_density = 0.6
    sl.unit_dry_weight = unit_mass * 9.8
    sl.specific_gravity = 2.65
    sl.xi = 0.03  # used in pysra if linear

    sl.o3_mat = o3.nd_material.PM4Sand(None, sl.relative_density, sl.g0_mod, sl.h_po, nu=sl.poissons_ratio,
                                 den=sl.unit_dry_mass / forder, p_atm=101.0e3 / forder)

    sl_base = sm.Soil()
    vs = 150.
    unit_mass = 1700.0
    sl_base.g_mod = vs ** 2 * unit_mass
    sl_base.poissons_ratio = 0.0
    sl_base.unit_dry_weight = unit_mass * 9.8
    sl_base.xi = 0.02  # for linear analysis

    e_mod = 2 * sl_base.g_mod * (1 + sl_base.poissons_ratio)
    sl_base.o3_mat = o3.nd_material.ElasticIsotropic(None, e_mod=e_mod / forder, nu=sl_base.poissons_ratio, rho=sl_base.unit_dry_mass / forder)
    # soil_profile.add_layer(5.1, sl_base)
    soil_profile.height = 10.0
    gm_scale_factor = 0.5
    record_filename = 'short_motion_dt0p01.txt'
    in_sig = eqsig.load_asig(ap.MODULE_DATA_PATH + 'gms/' + record_filename, m=gm_scale_factor)

    # analysis with pysra
    sl.sra_type = 'hyperbolic'
    sl_base.sra_type = 'linear'
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
    pysra_h = pysra_sig.smooth_fa_spectrum / in_sig.smooth_fa_spectrum
    sps[2].plot(pysra_sig.smooth_fa_frequencies, pysra_h, c='r')

    # analysis with O3
    etypes = ['implicit', 'explicit_difference', 'central_difference', 'newmark_explicit']
    etypes = ['implicit'] #, 'explicit_difference']
    # etypes = ['central_difference']
    # etypes = ['implicit', 'explicit_difference']
    etypes = ['explicit_difference']
    ls = ['-', '--', ':', '-.']

    for i, etype in enumerate(etypes):
        outputs_exp = site_response(soil_profile, in_sig, freqs=(0.5, 10), xi=0.03, etype=etype,
                                    forder=forder, rec_dt=in_sig.dt, analysis_time=None)
        resp_dt = (outputs_exp['time'][-1] - outputs_exp['time'][0]) / (len(outputs_exp['time']) - 1)
        # resp_dt = (outputs_exp['time'][1] - outputs_exp['time'][0])
        surf_sig = eqsig.AccSignal(outputs_exp['ACCX'][0], resp_dt)
        surf_sig.smooth_fa_frequencies = in_sig.fa_frequencies[1:]
        sps[0].plot(surf_sig.time, surf_sig.values, c=cbox(i), label=etype, ls=ls[i])
        sps[1].plot(surf_sig.fa_frequencies, abs(surf_sig.fa_spectrum), c=cbox(i), ls=ls[i])
        h = surf_sig.smooth_fa_spectrum / in_sig.smooth_fa_spectrum
        sps[2].plot(surf_sig.smooth_fa_frequencies, h, c=cbox(i), ls=ls[i])

    sps[2].axhline(1, c='k', ls='--')
    sps[1].set_xlim([0, 20])

    sps[0].legend(prop={'size': 6})
    plt.show()


if __name__ == '__main__':
    run()
