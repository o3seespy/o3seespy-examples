import o3seespy as o3
import o3seespy.extensions
import copy
import sfsimodels as sm
import numpy as np
import eqsig
import all_paths as ap
import liquepy as lq

import matplotlib.pyplot as plt
from bwplot import cbox


def site_response(sp, asig, freqs=(0.5, 10), xi=0.03, analysis_dt=0.001, dy=0.5,
                  analysis_time=None, outs=None, rec_dt=None, ff_bc=True, x_eles=1):
    """
    Run seismic analysis of a soil profile that has a compliant base

    Parameters
    ----------
    sp: sfsimodels.SoilProfile object
        A soil profile
    asig: eqsig.AccSignal object
        An acceleration signal
    fc_bc: bool
        If true then use fre-field columns as boundaries, else use tied boundaries
    x_eles: int
        Number of elements in the x-direction

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
    unit_masses = sp.split["unit_mass"] / 1e3

    grav = 9.81
    omega_1 = 2 * np.pi * freqs[0]
    omega_2 = 2 * np.pi * freqs[1]
    a0 = 2 * xi * omega_1 * omega_2 / (omega_1 + omega_2)
    a1 = 2 * xi / (omega_1 + omega_2)

    newmark_gamma = 0.5
    newmark_beta = 0.25

    # Define nodes and set boundary conditions for simple shear deformation
    # Start at top and build down?
    nx = np.linspace(0, 0.5 * x_eles, x_eles + 1)
    sn = [[o3.node.Node(osi, x, 0) for x in nx]]
    for i in range(1, n_node_rows):
        # Establish left and right nodes
        sn.append([o3.node.Node(osi, x, -node_depths[i]) for x in nx])
        # set x and y dofs equal for left and right nodes
        if i < n_node_rows - 1:
            if ff_bc:  # Use free-field columns as boundary conditions
                o3.EqualDOF(osi, sn[i][0], sn[i][1], [o3.cc.X, o3.cc.Y])
                o3.EqualDOF(osi, sn[i][-2], sn[i][-1], [o3.cc.X, o3.cc.Y])
            else:
                o3.EqualDOF(osi, sn[i][0], sn[i][-1], [o3.cc.X, o3.cc.Y])
    sn = np.array(sn)
    # Fix base nodes
    o3.Fix2DOFMulti(osi, sn[-1], o3.cc.FREE, o3.cc.FIXED)

    # Define dashpot nodes
    dashpot_node_l = o3.node.Node(osi, 0, -node_depths[-1])
    dashpot_node_2 = o3.node.Node(osi, 0, -node_depths[-1])
    o3.Fix2DOF(osi, dashpot_node_l, o3.cc.FIXED, o3.cc.FIXED)
    o3.Fix2DOF(osi, dashpot_node_2, o3.cc.FREE, o3.cc.FIXED)

    # define equal DOF for dashpot and soil base nodes
    o3.EqualDOFMulti(osi, r_node=sn[-1, 0], c_nodes=sn[-1, 1:], dofs=[o3.cc.X])
    o3.EqualDOF(osi, sn[-1][0], dashpot_node_2, [o3.cc.X])

    # define materials
    ele_thick = 1.0  # m
    soil_mats = []
    prev_args = []
    prev_kwargs = {}
    prev_sl_type = None
    eles = []
    area = 0
    for i in range(len(thicknesses)):
        y_depth = ele_depths[i]

        sl_id = sp.get_layer_index_by_depth(y_depth)
        sl = sp.layer(sl_id)
        app2mod = {}
        if y_depth > sp.gwl:
            umass = sl.unit_sat_mass / 1e3
        else:
            umass = sl.unit_dry_mass / 1e3
        # Define material
        sl_class = o3.nd_material.ElasticIsotropic
        sl.e_mod = 2 * sl.g_mod * (1 + sl.poissons_ratio) / 1e3
        app2mod['rho'] = 'unit_moist_mass'
        overrides = {'nu': sl.poissons_ratio, 'unit_moist_mass': umass}

        args, kwargs = o3.extensions.get_o3_kwargs_from_obj(sl, sl_class, custom=app2mod, overrides=overrides)
        changed = 0
        if sl.type != prev_sl_type or len(args) != len(prev_args) or len(kwargs) != len(prev_kwargs):
            changed = 1
        else:
            for j, arg in enumerate(args):
                if not np.isclose(arg, prev_args[j]):
                    changed = 1
            for pm in kwargs:
                if pm not in prev_kwargs or not np.isclose(kwargs[pm], prev_kwargs[pm]):
                    changed = 1

        if changed:
            mat = sl_class(osi, *args, **kwargs)
            prev_sl_type = sl.type
            prev_args = copy.deepcopy(args)
            prev_kwargs = copy.deepcopy(kwargs)

            soil_mats.append(mat)
        for j in range(len(nx) - 1):
            if j in [0, len(nx) - 2]:
                if ff_bc:
                    cur_ele_thick = ele_thick * 1000
                else:
                    cur_ele_thick = ele_thick
            else:
                cur_ele_thick = ele_thick
            if i == 0:
                area += cur_ele_thick * (sn[0][j+1].x - sn[0][j].x)
            # def element
            nodes = [sn[i+1][j], sn[i+1][j + 1], sn[i][j + 1], sn[i][j]]  # anti-clockwise
            eles.append(o3.element.SSPquad(osi, nodes, mat, o3.cc.PLANE_STRAIN, cur_ele_thick, 0.0, -grav))

    model_eff_width = area
    base_sl = sp.layer(sp.n_layers)
    c_base = model_eff_width * base_sl.unit_dry_mass / 1e3 * sp.get_shear_vel_at_depth(sp.height)
    dashpot_mat = o3.uniaxial_material.Viscous(osi, c_base, alpha=1.)
    o3.element.ZeroLength(osi, [dashpot_node_l, dashpot_node_2], mats=[dashpot_mat], dirs=[o3.cc.DOF2D_X])

    # Static analysis
    o3.constraints.Transformation(osi)
    o3.test_check.NormDispIncr(osi, tol=1.0e-4, max_iter=30, p_flag=0)
    o3.algorithm.Newton(osi)
    o3.numberer.RCM(osi)
    o3.system.ProfileSPD(osi)
    o3.integrator.Newmark(osi, newmark_gamma, newmark_beta)
    o3.analysis.Transient(osi)
    o3.analyze(osi, 40, 1.)

    for i in range(len(soil_mats)):
        if isinstance(soil_mats[i], o3.nd_material.PM4Sand) or isinstance(soil_mats[i], o3.nd_material.PressureIndependMultiYield):
            o3.update_material_stage(osi, soil_mats[i], 1)
    o3.analyze(osi, 50, 0.5)

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

    # Define the dynamic analysis
    ts_obj = o3.time_series.Path(osi, dt=asig.dt, values=asig.velocity * 1, factor=c_base)
    o3.pattern.Plain(osi, ts_obj)
    o3.Load(osi, sn[-1][0], [1., 0.])

    # Run the dynamic analysis
    o3.algorithm.Newton(osi)
    o3.system.SparseGeneral(osi)
    o3.numberer.RCM(osi)
    o3.constraints.Transformation(osi)
    o3.integrator.Newmark(osi, newmark_gamma, newmark_beta)
    o3.rayleigh.Rayleigh(osi, a0, a1, 0, 0)
    o3.analysis.Transient(osi)

    o3.test_check.EnergyIncr(osi, tol=1.0e-7, max_iter=10)
    o3.extensions.to_py_file(osi, 'simple_1d_comp_mod.py')
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

    sl = sm.Soil()
    vs = 250.
    unit_mass = 1700.0
    sl.g_mod = vs ** 2 * unit_mass
    sl.poissons_ratio = 0.3
    sl.unit_dry_weight = unit_mass * 9.8
    sl.xi = 0.01  # for linear analysis
    soil_profile.add_layer(0, sl)

    sl_base = sm.Soil()
    vs = 350.
    unit_mass = 1700.0
    sl_base.g_mod = vs ** 2 * unit_mass
    sl_base.poissons_ratio = 0.3
    sl_base.unit_dry_weight = unit_mass * 9.8
    sl_base.xi = 0.01  # for linear analysis
    soil_profile.add_layer(19., sl_base)
    soil_profile.height = 20.0

    record_filename = 'short_motion_dt0p01.txt'
    in_sig = eqsig.load_asig(ap.MODULE_DATA_PATH + 'gms/' + record_filename, m=0.5)

    # analysis with pysra
    od = lq.sra.run_pysra(soil_profile, in_sig, odepths=np.array([0.0]), wave_field='outcrop')
    pysra_sig = eqsig.AccSignal(od['ACCX'][0], in_sig.dt)

    bf, sps = plt.subplots(nrows=3)

    sps[0].plot(in_sig.time, in_sig.values, c='k', label='Input')

    sps[0].plot(pysra_sig.time, pysra_sig.values, c=cbox('mid grey'), label='pysra')

    sps[1].plot(in_sig.fa_frequencies, abs(in_sig.fa_spectrum), c='k')

    sps[1].plot(pysra_sig.fa_frequencies, abs(pysra_sig.fa_spectrum), c=cbox('mid grey'))
    sps[1].set_xlim([0, 20])
    in_sig.smooth_fa_frequencies = in_sig.fa_frequencies[1:]
    pysra_sig.smooth_fa_frequencies = in_sig.fa_frequencies[1:]
    pysra_h = pysra_sig.smooth_fa_spectrum / in_sig.smooth_fa_spectrum
    sps[2].plot(pysra_sig.smooth_fa_frequencies, pysra_h, c=cbox('mid grey'))

    # Normal 1d analysis with tied boundaries
    outputs = site_response(soil_profile, in_sig, freqs=(0.5, 10), xi=0.02, analysis_dt=0.005, x_eles=1, ff_bc=False)
    resp_dt = outputs['time'][2] - outputs['time'][1]
    surf_sig = eqsig.AccSignal(outputs['ACCX'][0], resp_dt)
    sps[0].plot(surf_sig.time, surf_sig.values, c=cbox(0), label='o3 - 1D')
    sps[1].plot(surf_sig.fa_frequencies, abs(surf_sig.fa_spectrum), c=cbox(0))
    surf_sig.smooth_fa_frequencies = in_sig.fa_frequencies[1:]
    h = surf_sig.smooth_fa_spectrum / in_sig.smooth_fa_spectrum
    sps[2].plot(surf_sig.smooth_fa_frequencies, h, c=cbox(0))

    # 2d analysis with tied boundaries
    outputs = site_response(soil_profile, in_sig, freqs=(0.5, 10), xi=0.02, analysis_dt=0.005, x_eles=20, ff_bc=False)
    resp_dt = outputs['time'][2] - outputs['time'][1]
    surf_sig = eqsig.AccSignal(outputs['ACCX'][0], resp_dt)
    sps[0].plot(surf_sig.time, surf_sig.values, c=cbox(1), label='o3 - 2D tied', ls='--')
    sps[1].plot(surf_sig.fa_frequencies, abs(surf_sig.fa_spectrum), c=cbox(1), ls='--')
    surf_sig.smooth_fa_frequencies = in_sig.fa_frequencies
    h = surf_sig.smooth_fa_spectrum / in_sig.smooth_fa_spectrum
    sps[2].plot(surf_sig.smooth_fa_frequencies, h, c=cbox(1), ls='--')

    # 2d analysis with free-field columns
    outputs = site_response(soil_profile, in_sig, freqs=(0.5, 10), xi=0.02, analysis_dt=0.005, x_eles=20, ff_bc=True)
    resp_dt = outputs['time'][2] - outputs['time'][1]
    surf_sig = eqsig.AccSignal(outputs['ACCX'][0], resp_dt)
    sps[0].plot(surf_sig.time, surf_sig.values, c=cbox(2), label='o3 - 2D FF columns', ls=':')
    sps[1].plot(surf_sig.fa_frequencies, abs(surf_sig.fa_spectrum), c=cbox(2), ls=':')
    surf_sig.smooth_fa_frequencies = in_sig.fa_frequencies
    h = surf_sig.smooth_fa_spectrum / in_sig.smooth_fa_spectrum
    sps[2].plot(surf_sig.smooth_fa_frequencies, h, c=cbox(2), ls=':')

    sps[0].set_ylabel('Acc [m/s2]')
    sps[1].set_ylabel('FA')

    sps[2].set_ylabel('H')
    sps[2].set_xlabel('Freq [Hz]')

    sps[2].axhline(1, c='k', ls='--')
    sps[0].legend()
    plt.show()


if __name__ == '__main__':
    run()
