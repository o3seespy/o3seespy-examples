import numpy as np
import eqsig
import sfsimodels as sm
import o3seespy as o3
import matplotlib.pyplot as plt
from bwplot import cbox
import engformat as ef
import all_paths as ap

import os


from matplotlib import rc
rc('font', family='Helvetica', size=9, weight='light')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42  # To avoid type 3 fonts



def get_response(bd, asig, dtype, l_ph):
    """
    Compute the response of a nonlinear lollipop on a foundation with linear/nonlinear soil
    Units are N, m, s

    :param bd:
        SDOF building object
    :param asig:
        Acceleration signal object
    :return:
    """
    osi = o3.OpenSeesInstance(ndm=2, state=3)

    # Establish nodes
    top_ss_node = o3.node.Node(osi, 0, bd.h_eff)
    bot_ss_node = o3.node.Node(osi, 0, 0)

    # Fix bottom node
    o3.Fix3DOF(osi, bot_ss_node, o3.cc.FIXED, o3.cc.FIXED, o3.cc.FIXED)

    # nodal mass (weight / g):
    o3.Mass(osi, top_ss_node, bd.mass_eff, 0.0, 0)

    # Define a column element with a plastic hinge at base
    transf = o3.geom_transf.Linear2D(osi, [])  # can change for P-delta effects
    area = 1.0
    e_mod = 200.0e9
    iz = bd.k_eff * bd.h_eff ** 3 / (3 * e_mod)
    ele_nodes = [bot_ss_node, top_ss_node]

    # Superstructure element
    elastic_sect = o3.section.Elastic2D(osi, e_mod, area, iz)
    integ = o3.beam_integration.HingeMidpoint(osi, elastic_sect, l_ph, elastic_sect, l_ph, elastic_sect)
    vert_ele = o3.element.ForceBeamColumn(osi, ele_nodes, transf, integ)
    # vert_ele = o3.element.ElasticBeamColumn2D(osi, ele_nodes, area=area, e_mod=e_mod, iz=iz, transf=transf)

    # angular_freq = o3.get_eigen(osi, n=1)[0] ** 0.5
    # response_period = 2 * np.pi / angular_freq
    # print('response_period: ', response_period)
    omega = 2 * np.pi / bd.t_fixed

    # define superstructure damping using rotational spring approach from Millen et al. (2017) to avoid double damping
    if dtype == 'rot_dashpot':
        cxx = bd.xi * 2 * np.sqrt(bd.mass_eff * bd.k_eff)
        equiv_c_rot = cxx * (2.0 / 3) ** 2 * bd.h_eff ** 2
        ss_rot_dashpot_mat = o3.uniaxial_material.Viscous(osi, equiv_c_rot, alpha=1.)
        sfi_dashpot_ele = o3.element.TwoNodeLink(osi, [bot_ss_node, top_ss_node], mats=[ss_rot_dashpot_mat],
                                            dirs=[o3.cc.DOF2D_ROTZ])
    elif dtype == 'horz_dashpot':
        cxx = bd.xi * 2 * np.sqrt(bd.mass_eff * bd.k_eff)
        ss_rot_dashpot_mat = o3.uniaxial_material.Viscous(osi, cxx, alpha=1.)
        sfi_dashpot_ele = o3.element.ZeroLength(osi, [bot_ss_node, top_ss_node], mats=[ss_rot_dashpot_mat],
                                            dirs=[o3.cc.X])
    else:
        beta_k = 2 * bd.xi / omega
        o3.rayleigh.Rayleigh(osi, 0, 0, beta_k_init=beta_k, beta_k_comm=0.0)


    # Define the input motion for the dynamic analysis
    acc_series = o3.time_series.Path(osi, dt=asig.dt, values=-asig.values)  # should be negative
    o3.pattern.UniformExcitation(osi, dir=o3.cc.X, accel_series=acc_series)
    print('loaded gm')
    o3.wipe_analysis(osi)

    o3.algorithm.Newton(osi)
    o3.system.SparseGeneral(osi)
    o3.numberer.RCM(osi)
    o3.constraints.Transformation(osi)
    o3.integrator.Newmark(osi, 0.5, 0.25)
    o3.analysis.Transient(osi)

    o3.test_check.NormDispIncr(osi, tol=1.0e-6, max_iter=10)
    analysis_time = asig.time[-1]
    analysis_dt = 0.001

    # define outputs of analysis
    od = {
        "time": o3.recorder.TimeToArrayCache(osi),
        "rel_deck_disp": o3.recorder.NodeToArrayCache(osi, top_ss_node, [o3.cc.DOF2D_X], 'disp'),
        "deck_accel": o3.recorder.NodeToArrayCache(osi, top_ss_node, [o3.cc.DOF2D_X], 'accel'),
        "deck_rot": o3.recorder.NodeToArrayCache(osi, top_ss_node, [o3.cc.DOF2D_ROTZ], 'disp'),
        "chord_rots": o3.recorder.ElementToArrayCache(osi, vert_ele, arg_vals=['chordRotation']),
        "col_forces": o3.recorder.ElementToArrayCache(osi, vert_ele, arg_vals=['force']),
    }
    if dtype in ['rot_dashpot', 'horz_dashpot']:
        od['dashpot_force'] = o3.recorder.ElementToArrayCache(osi, sfi_dashpot_ele, arg_vals=['force'])

    o3.analyze(osi, int(analysis_time / analysis_dt), analysis_dt)

    o3.wipe(osi)

    for item in od:
        od[item] = od[item].collect()
    od['col_shear'] = -od['col_forces'][:, 0]
    od['col_moment'] = od['col_forces'][:, 2]
    od['hinge_rotation'] = od['chord_rots'][:, 1]
    od['hinge_rotation1'] = -od['chord_rots'][:, 2]
    del od['col_forces']
    del od['chord_rots']

    return od



def create():
    """
    Run an SDOF using three different damping options

    """
    dtypes = ['rot_dashpot', 'horz_dashpot', 'rayleigh']
    # dtypes = ['rayleigh']
    lss = ['-', '--', ':']

    bd = sm.SDOFBuilding()
    bd.mass_eff = 120.0e3
    bd.h_eff = 10.0
    bd.t_fixed = 0.7
    bd.xi = 0.2  # use high damping to evaluate performance of diff damping options
    bd.inputs += ['xi']
    l_ph = 0.2

    record_filename = 'test_motion_dt0p01.txt'
    asig = eqsig.load_asig(ap.MODULE_DATA_PATH + 'gms/' + record_filename, m=0.5)

    bf, sps = plt.subplots(nrows=3, figsize=(5, 8))

    for c, dtype in enumerate(dtypes):

        outputs = get_response(bd, asig, dtype=dtype, l_ph=l_ph)
        # Time series plots
        sps[0].plot(outputs['time'], outputs['deck_accel'] / 9.8, c=cbox(c), lw=0.7, label=dtype, ls=lss[c])
        # Hysteretic plots
        sps[1].plot(outputs['hinge_rotation'] * 1e3, outputs['col_moment'] / 1e3, c=cbox(c), label=dtype, ls=lss[c])
        sps[2].plot(outputs['rel_deck_disp'] * 1e3, outputs['col_shear'] / 1e3, c=cbox(c), label=dtype, ls=lss[c])
        if dtype == 'rot_dashpot':
            sps[1].plot(outputs['hinge_rotation'] * 1e3, (outputs['col_moment'] - outputs['dashpot_force'][:, 2] / 2) / 1e3, c='r', label=dtype)
            # sps[1].plot(outputs['hinge_rotation'] * 1e3, (outputs['dashpot_force'][:, 2]) / 1e3, c='orange', label=dtype)
            sps[2].plot(outputs['rel_deck_disp'] * 1e3, (outputs['col_shear'] - 3. / 2. * outputs['dashpot_force'][:, 2] / bd.h_eff) / 1e3, c='r', label=dtype)

    ei = bd.k_eff * bd.h_eff ** 3 / 3

    # moms = np.array([-1000e3, 1000e3])
    # k = ei / (l_ph * (1 - l_ph / bd.h_eff / 2))
    # sps[1].plot(moms / k * 1e3, moms / 1e3, c='k')
    # k_drift = ei / (bd.h_eff * 1. / 3)
    # sps[1].plot(moms / k * 1e3, moms / 1e3, c='r')
    sps[0].set_ylabel('Deck accel. [g]', fontsize=7)
    sps[0].set_xlabel('Time [s]')
    ef.text_at_rel_pos(sps[1], 0.1, 0.9, 'Column hinge')
    sps[1].set_xlabel('Rotation [mrad]')
    sps[1].set_ylabel('Moment [kNm]')
    sps[1].set_ylabel('Shear [kN]')
    sps[1].set_xlabel('Disp. [mm]')

    ef.revamp_legend(sps[0])

    ef.xy(sps[0], x_origin=True)
    ef.xy(sps[0], x_axis=True, y_axis=True)

    bf.tight_layout()
    name = __file__.replace('.py', '')
    name = name.split("fig_")[-1]
    bf.savefig(f'figures/{name}.png', dpi=90)
    plt.show()


if __name__ == '__main__':
    create()



