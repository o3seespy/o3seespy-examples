import o3seespy as o3
import copy
import sfsimodels as sm
import numpy as np
import eqsig
import all_paths as ap
# for linear analysis comparison
import liquepy as lq


def site_response(sp, asig, freqs=(0.5, 10), xi=0.03, dtype='rayleigh', analysis_dt=0.001, dy=0.5, analysis_time=None, outs=None, rec_dt=None):
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
    if rec_dt is None:
        rec_dt = analysis_dt

    osi = o3.OpenSeesInstance(ndm=2, ndf=2, state=3)
    assert isinstance(sp, sm.SoilProfile)
    sp.gen_split(props=['shear_vel', 'unit_mass'], target=dy)
    thicknesses = sp.split["thickness"]
    n_node_rows = len(thicknesses) + 1
    node_depths = np.cumsum(sp.split["thickness"])
    node_depths = np.insert(node_depths, 0, 0)
    ele_depths = (node_depths[1:] + node_depths[:-1]) / 2

    grav = 9.81

    k0 = 0.5
    pois = k0 / (1 + k0)

    newmark_gamma = 0.5
    newmark_beta = 0.25

    ele_width = min(thicknesses)

    # Define nodes and set boundary conditions for simple shear deformation
    # Start at top and build down?
    sn = [[o3.node.Node(osi, 0, 0), o3.node.Node(osi, ele_width, 0)]]
    for i in range(1, n_node_rows):
        # Establish left and right nodes
        sn.append([o3.node.Node(osi, 0, -node_depths[i]), o3.node.Node(osi, ele_width, -node_depths[i])])
        # set x and y dofs equal for left and right nodes
        o3.EqualDOF(osi, sn[i][0], sn[i][1], [o3.cc.X, o3.cc.Y])

    # Fix base nodes
    o3.Fix2DOF(osi, sn[-1][0], o3.cc.FIXED, o3.cc.FIXED)
    o3.Fix2DOF(osi, sn[-1][1], o3.cc.FIXED, o3.cc.FIXED)

    # define materials
    ele_thick = 1.0  # m
    soil_mats = []
    prev_pms = [0, 0, 0]
    eles = []
    for i in range(len(thicknesses)):
        y_depth = ele_depths[i]
        sl_id = sp.get_layer_index_by_depth(y_depth)
        sl = sp.layer(sl_id)
        # Define material
        e_mod = 2 * sl.g_mod * (1 + sl.poissons_ratio)
        umass = sl.unit_dry_mass
        nu = sl.poissons_ratio
        pms = [e_mod, nu, umass]
        changed = 0
        for pp in range(len(pms)):
            if not np.isclose(pms[pp], prev_pms[pp]):
                changed = 1
        if changed:
            mat = o3.nd_material.ElasticIsotropic(osi, e_mod=e_mod, nu=nu, rho=umass)
            soil_mats.append(mat)

        # def element
        nodes = [sn[i+1][0], sn[i+1][1], sn[i][1], sn[i][0]]  # anti-clockwise
        eles.append(o3.element.SSPquad(osi, nodes, mat, o3.cc.PLANE_STRAIN, ele_thick, 0.0, grav * umass))

    # Static analysis
    o3.constraints.Transformation(osi)
    o3.test_check.NormDispIncr(osi, tol=1.0e-5, max_iter=30, p_flag=0)
    o3.algorithm.Newton(osi)
    o3.numberer.RCM(osi)
    o3.system.ProfileSPD(osi)
    o3.integrator.Newmark(osi, 5./6, 4./9)  # include numerical damping
    o3.analysis.Transient(osi)
    o3.analyze(osi, 40, 1.)

    # reset time and analysis
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

    # Run the dynamic analysis
    o3.algorithm.Newton(osi)
    o3.system.SparseGeneral(osi)
    o3.numberer.RCM(osi)
    o3.constraints.Transformation(osi)
    o3.integrator.Newmark(osi, newmark_gamma, newmark_beta)

    if dtype == 'rayleigh':
        omega_1 = 2 * np.pi * freqs[0]
        omega_2 = 2 * np.pi * freqs[1]
        a0 = 2 * xi * omega_1 * omega_2 / (omega_1 + omega_2)
        a1 = 2 * xi / (omega_1 + omega_2)
        o3.rayleigh.Rayleigh(osi, a0, a1, 0, 0)
    else:
        n = 20
        omegas = np.array(o3.get_eigen(osi, n=n)) ** 0.5
        o3.ModalDamping(osi, [xi])

    o3.analysis.Transient(osi)

    o3.test_check.NormDispIncr(osi, tol=1.0e-7, max_iter=10)

    # Define the dynamic analysis
    acc_series = o3.time_series.Path(osi, dt=asig.dt, values=asig.values)
    o3.pattern.UniformExcitation(osi, dir=o3.cc.X, accel_series=acc_series)

    while o3.get_time(osi) < analysis_time:
        print(o3.get_time(osi))
        if o3.analyze(osi, 1, analysis_dt):
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
    out_dict['time'] = np.arange(0, analysis_time, rec_dt)

    return out_dict


def run():
    soil_profile = sm.SoilProfile()
    soil_profile.height = 30.0
    xi = 0.03

    sl = sm.Soil()
    vs = 250.
    unit_mass = 1700.0
    sl.g_mod = vs ** 2 * unit_mass
    sl.poissons_ratio = 0.3
    sl.unit_dry_weight = unit_mass * 9.8
    sl.xi = xi  # for linear analysis
    sl.sra_type = 'linear'
    soil_profile.add_layer(0, sl)

    sl_base = sm.Soil()
    vs = 350.
    unit_mass = 1700.0
    sl_base.g_mod = vs ** 2 * unit_mass
    sl_base.poissons_ratio = 0.0
    sl_base.unit_dry_weight = unit_mass * 9.8
    sl_base.xi = xi  # for linear analysis
    sl_base.sra_type = 'linear'
    soil_profile.add_layer(10., sl_base)
    soil_profile.height = 20.0
    gm_scale_factor = 1.5
    record_filename = 'test_motion_dt0p01.txt'
    in_sig = eqsig.load_asig(ap.MODULE_DATA_PATH + 'gms/' + record_filename, m=gm_scale_factor)

    # analysis with pysra
    od = lq.sra.run_pysra(soil_profile, in_sig, odepths=np.array([0.0]), wave_field='within')
    ind = in_sig.npts
    pysra_sig = eqsig.AccSignal(od['ACCX'][0][:ind], in_sig.dt)

    dtype = 'rayleigh'
    dtype = 'modal'
    outputs = site_response(soil_profile, in_sig, freqs=(0.5, 10), xi=xi, analysis_dt=0.005, dtype=dtype)
    resp_dt = outputs['time'][2] - outputs['time'][1]
    surf_sig = eqsig.AccSignal(outputs['ACCX'][0], resp_dt)

    o3_surf_vals = np.interp(pysra_sig.time, surf_sig.time, surf_sig.values) + in_sig.values
    surf_sig = eqsig.AccSignal(surf_sig.values + np.interp(surf_sig.time, in_sig.time, in_sig.values), surf_sig.dt)

    show = 1

    if show:
        import matplotlib.pyplot as plt
        from bwplot import cbox

        in_sig.smooth_fa_frequencies = in_sig.fa_frequencies[1:]
        surf_sig.smooth_fa_frequencies = in_sig.fa_frequencies[1:]
        pysra_sig.smooth_fa_frequencies = in_sig.fa_frequencies[1:]

        bf, sps = plt.subplots(nrows=3)

        sps[0].plot(in_sig.time, in_sig.values, c='k', label='Input')
        sps[0].plot(surf_sig.time, surf_sig.values, c=cbox(0), label='o3')
        sps[0].plot(pysra_sig.time, pysra_sig.values, c=cbox(1), label='pysra')

        sps[1].plot(in_sig.fa_frequencies, abs(in_sig.fa_spectrum), c='k')
        sps[1].plot(surf_sig.fa_frequencies, abs(surf_sig.fa_spectrum), c=cbox(0))
        sps[1].plot(pysra_sig.fa_frequencies, abs(pysra_sig.fa_spectrum), c=cbox(1))

        h = surf_sig.smooth_fa_spectrum / in_sig.smooth_fa_spectrum
        sps[2].plot(surf_sig.smooth_fa_frequencies, h, c=cbox(0))
        pysra_h = pysra_sig.smooth_fa_spectrum / in_sig.smooth_fa_spectrum
        sps[2].plot(pysra_sig.smooth_fa_frequencies, pysra_h, c=cbox(1))
        sps[2].axhline(1, c='k', ls='--')
        sps[1].set_xlim([0, 20])
        sps[2].set_xlim([0, 20])
        sps[0].legend()
        name = __file__.replace('.py', '')
        name = name.split("fig_")[-1]
        bf.suptitle(name)
        bf.savefig(f'figs/{name}.png', dpi=90)
        plt.show()

    assert np.isclose(o3_surf_vals, pysra_sig.values, atol=0.01, rtol=100).all()


if __name__ == '__main__':
    run()
