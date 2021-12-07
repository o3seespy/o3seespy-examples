import o3seespy as o3
from inspect import signature

import numpy as np
import math
import matplotlib.pyplot as plt


def run_pm4sand_et(sl, csr, esig_v0=101.0e3, static_bias=0.0, n_lim=100, k0=0.5, runfile=None, water_bulk_mod=2.2e6, strain_limit=0.03, strain_inc=5.0e-6, cached=0):

    nu_init = k0 / (1 + k0)

    # Initialise OpenSees instance
    osi = o3.OpenSeesInstance(ndm=2, ndf=3, state=3)

    # Establish nodes
    h_ele = 1.
    bl_node = o3.node.Node(osi, 0, 0)
    br_node = o3.node.Node(osi, h_ele, 0)
    tr_node = o3.node.Node(osi, h_ele, h_ele)
    tl_node = o3.node.Node(osi, 0, h_ele)
    all_nodes = [bl_node, br_node, tr_node, tl_node]

    # Fix bottom node
    o3.Fix3DOF(osi, bl_node, o3.cc.FIXED, o3.cc.FIXED, o3.cc.FIXED)
    o3.Fix3DOF(osi, br_node, o3.cc.FIXED, o3.cc.FIXED, o3.cc.FIXED)
    o3.Fix3DOF(osi, tr_node, o3.cc.FREE, o3.cc.FREE, o3.cc.FIXED)
    o3.Fix3DOF(osi, tl_node, o3.cc.FREE, o3.cc.FREE, o3.cc.FIXED)
    # Set out-of-plane DOFs to be slaved
    o3.EqualDOF(osi, tr_node, tl_node, [o3.cc.X, o3.cc.Y])

    # Define material
    pm4sand = o3.nd_material.PM4Sand(osi, sl.relative_density, sl.g0_mod, sl.h_po, sl.unit_sat_mass, 101.3, nu=nu_init)

    # Note water bulk modulus is irrelevant since constant volume test - so as soil skeleton contracts
    # the bulk modulus of the soil skeleton controls the change in effective stress
    ele = o3.element.SSPquadUP(osi, all_nodes, pm4sand, 1.0, water_bulk_mod, 1.,
                                sl.permeability, sl.permeability, sl.e_curr, alpha=1.0e-5, b1=0.0, b2=0.0)

    n = 1
    omegas = np.array(o3.get_eigen(osi, n=n)) ** 0.5
    periods = 2 * np.pi / omegas
    periods = [0.001]
    print('response_periods: ', periods)

    o3.constraints.Transformation(osi)
    o3.test_check.NormDispIncr(osi, tol=1.0e-6, max_iter=35, p_flag=0)
    o3.numberer.RCM(osi)
    etype = 'implicit'
    etype = 'explicit_difference'  # explicit_difference is working!
    if etype == 'implicit':
        o3.algorithm.Newton(osi)
        o3.system.FullGeneral(osi)
        o3.integrator.Newmark(osi, gamma=0.5, beta=0.25)
        dt = 0.01
    else:
        o3.algorithm.Linear(osi, factor_once=True)

        if etype == 'newmark_explicit':
            o3.system.FullGeneral(osi)
            o3.integrator.NewmarkExplicit(osi, gamma=0.5)
            explicit_dt = periods[0] / np.pi / 8
        elif etype == 'central_difference':
            o3.system.FullGeneral(osi)
            o3.integrator.CentralDifference(osi)
            explicit_dt = periods[0] / np.pi / 32  # 0.5 is a factor of safety
        elif etype == 'explicit_difference':
            o3.system.Diagonal(osi)
            o3.integrator.ExplicitDifference(osi)
            explicit_dt = periods[0] / np.pi / 8
        else:
            raise ValueError(etype)
        print('explicit_dt: ', explicit_dt)
        dt = explicit_dt
    o3.analysis.Transient(osi)
    freqs = [0.05, 3]
    xi = 0.02
    if etype in ['newmark_explicit', 'central_difference', 'implicit']:  # Does not support modal damping
        omega_1 = 2 * np.pi * freqs[0]
        omega_2 = 2 * np.pi * freqs[1]
        a0 = 2 * xi * omega_1 * omega_2 / (omega_1 + omega_2)
        a1 = 2 * xi / (omega_1 + omega_2)
        o3.rayleigh.Rayleigh(osi, a0, 0, a1, 0)
    else:
        omega_1 = 2 * np.pi * freqs[0]
        omega_2 = 2 * np.pi * freqs[1]
        a0 = 2 * xi * omega_1 * omega_2 / (omega_1 + omega_2)
        a1 = 2 * xi / (omega_1 + omega_2)
        o3.rayleigh.Rayleigh(osi, a0, 0, 0, 0)
        # o3.ModalDamping(osi, [xi, xi, xi])

    o3.update_material_stage(osi, pm4sand, stage=0)
    # print('here1: ', o3.get_ele_response(osi, ele, 'stress'), esig_v0, csr)

    all_stresses_cache = o3.recorder.ElementToArrayCache(osi, ele, arg_vals=['stress'], dt=1)
    all_strains_cache = o3.recorder.ElementToArrayCache(osi, ele, arg_vals=['strain'], dt=1)
    nodes_cache = o3.recorder.NodesToArrayCache(osi, all_nodes, dofs=[1, 2, 3], res_type='disp', dt=1)
    o3.recorder.NodesToFile(osi, 'node_disp.txt', all_nodes, dofs=[1, 2, 3], res_type='disp', dt=1)

    # Add static vertical pressure and stress bias
    ttot = 10
    time_series = o3.time_series.Path(osi, time=[0, ttot, 1e10], values=[0, 1, 1])
    o3.pattern.Plain(osi, time_series)
    o3.Load(osi, tl_node, [0, -esig_v0 / 2, 0])
    o3.Load(osi, tr_node, [0, -esig_v0 / 2, 0])

    o3.analyze(osi, num_inc=int(ttot / dt), dt=dt)

    ts2 = o3.time_series.Path(osi, time=[ttot, 1e10, 1e10], values=[1., 1., 1.], factor=1)
    o3.pattern.Plain(osi, ts2, fact=1.)
    y_vert = o3.get_node_disp(osi, tr_node, o3.cc.Y)
    o3.SP(osi, tl_node, dof=o3.cc.Y, dof_values=[y_vert])
    o3.SP(osi, tr_node, dof=o3.cc.Y, dof_values=[y_vert])

    # Close the drainage valves
    delta_time = 10
    for node in all_nodes:
        o3.remove_sp(osi, node, dof=3)
    o3.analyze(osi, int(delta_time / dt), dt=dt)
    print('here3: ', o3.get_ele_response(osi, ele, 'stress'), esig_v0, csr)

    o3.update_material_stage(osi, pm4sand, stage=1)
    o3.set_parameter(osi, value=0, eles=[ele], args=['FirstCall', pm4sand.tag])
    o3.analyze(osi, int(delta_time / dt), dt=dt)
    o3.set_parameter(osi, value=sl.poissons_ratio, eles=[ele], args=['poissonRatio', pm4sand.tag])

    o3.extensions.to_py_file(osi)

    n_cyc = 0.0
    target_strain = 1.1 * strain_limit
    target_disp = target_strain * h_ele
    limit_reached = 0
    export = 1
    adj_strain_inc = strain_inc / dt * 0.01
    print('adj_strain_inc: ', adj_strain_inc)
    while n_cyc < n_lim:
        print('n_cyc: ', n_cyc)
        h_disp = o3.get_node_disp(osi, tr_node, o3.cc.X)
        curr_time = o3.get_time(osi)
        steps = target_strain / adj_strain_inc
        ts0 = o3.time_series.Path(osi, time=[curr_time, curr_time + steps, 1e10], values=[h_disp, target_disp, target_disp], factor=1)
        pat0 = o3.pattern.Plain(osi, ts0)
        o3.SP(osi, tr_node, dof=o3.cc.X, dof_values=[1.0])
        curr_stress = o3.get_ele_response(osi, ele, 'stress')[2]
        if math.isnan(curr_stress):
            raise ValueError

        if export:
            o3.extensions.to_py_file(osi)
            export = 0
        t_inc = 0.001
        while curr_stress < (csr - static_bias) * esig_v0:
            o3.analyze(osi, min(int(t_inc / dt), 10), dt=dt)
            curr_stress = o3.get_ele_response(osi, ele, 'stress')[2]
            h_disp = o3.get_node_disp(osi, tr_node, o3.cc.X)
            print('h_disp: ', h_disp, curr_stress)
            if h_disp >= target_disp:
                print(f'STRAIN LIMIT REACHED - on load ({h_disp} >= {target_disp})')
                limit_reached = 1
                break
        if limit_reached:
            break
        n_cyc += 0.25
        print('load reversal, n_cyc: ', n_cyc)
        curr_time = o3.get_time(osi)
        o3.remove_load_pattern(osi, pat0)
        o3.remove(osi, ts0)
        o3.remove_sp(osi, tr_node, dof=o3.cc.X)
        # Reverse cycle
        steps = (h_disp + target_disp) / (adj_strain_inc * h_ele)
        rev_time = curr_time + steps
        ts0 = o3.time_series.Path(osi, time=[curr_time, rev_time, 1e10],
                                   values=[h_disp, -target_disp, -target_disp], factor=1)
        print(f'time=[{curr_time, rev_time, 1e10}], values=[{h_disp, -target_disp, -target_disp}]')
        pat0 = o3.pattern.Plain(osi, ts0)
        o3.SP(osi, tr_node, dof=o3.cc.X, dof_values=[1.0])
        i = 0
        while curr_stress > -(csr + static_bias) * esig_v0:
            o3.analyze(osi, min(int(t_inc / dt), 10), dt=dt)
            curr_stress = o3.get_ele_response(osi, ele, 'stress')[2]
            h_disp = o3.get_node_disp(osi, tr_node, o3.cc.X)
            curr_time = o3.get_time(osi)
            print('h_disp: ', h_disp, curr_stress, curr_time)

            if -h_disp >= target_disp:
                print(f'STRAIN LIMIT REACHED - on reverse (-{h_disp} >= {target_disp})')
                limit_reached = 1
                break
            i += 1
            if curr_time > rev_time:
                break
        if limit_reached:
            break
        n_cyc += 0.5
        print('reload, n_cyc: ', n_cyc)
        curr_time = o3.get_time(osi)
        o3.remove_load_pattern(osi, pat0)
        o3.remove(osi, ts0)
        o3.remove_sp(osi, tr_node, dof=o3.cc.X)
        # reload cycle
        steps = (-h_disp + target_disp) / (adj_strain_inc * h_ele)
        rev_time = curr_time + steps
        ts0 = o3.time_series.Path(osi, time=[curr_time, rev_time, 1e10],
                                   values=[h_disp, target_disp, target_disp], factor=1)
        pat0 = o3.pattern.Plain(osi, ts0)
        o3.SP(osi, tr_node, dof=o3.cc.X, dof_values=[1.0])
        while curr_stress < static_bias * esig_v0:
            o3.analyze(osi, min(int(t_inc / dt), 10), dt=dt)
            curr_stress = o3.get_ele_response(osi, ele, 'stress')[2]
            h_disp = o3.get_node_disp(osi, tr_node, o3.cc.X)
            curr_time = o3.get_time(osi)
            print('h_disp: ', h_disp, curr_stress, curr_time)

            if h_disp >= target_disp:
                print(f'STRAIN LIMIT REACHED - on reload ({h_disp} >= {target_disp})')
                limit_reached = 1
                break
            # if curr_time > rev_time:
            #     break
        if limit_reached:
            break
        o3.remove_load_pattern(osi, pat0)
        o3.remove(osi, ts0)
        o3.remove_sp(osi, tr_node, dof=o3.cc.X)
        n_cyc += 0.25

    o3.wipe(osi)
    all_stresses = all_stresses_cache.collect()
    all_strains = all_strains_cache.collect()
    disps = nodes_cache.collect()
    stress = all_stresses[:, 2]
    strain = all_strains[:, 2]
    ppt = all_stresses[:, 1]

    return stress, strain, ppt, disps

    pass


if __name__ == '__main__':
    import o3seespy as o3

    import liquepy as lq

    esig_v0 = 101.3
    gravity = 9.8

    sl = lq.num.o3.PM4Sand(liq_mass_density=1.0)
    sl.relative_density = 0.35
    sl.g0_mod = 476.0
    sl.h_po = 0.53
    crr_n15 = 0.13
    sl.unit_sat_weight = 1.42 * gravity

    sl.e_min = 0.5
    sl.e_max = 0.8
    k0 = 0.5
    sl.poissons_ratio = 0.3
    sl.phi = 33.

    sl.permeability = 1.0e-9
    sl.p_atm = 101.0e3
    strain_inc = 5.e-6
    csr = 0.16

    stress, strain, ppt, disps = run_pm4sand_et(sl, csr=csr, n_lim=20, strain_limit=0.03,
                                                esig_v0=esig_v0, strain_inc=strain_inc, k0=k0)

    bf, sps = plt.subplots(nrows=2)
    sps[0].plot(stress, label='shear')
    sps[0].plot(ppt, label='PPT')
    sps[1].plot(strain, stress, label='o3seespy')

    sps[0].set_xlabel('Time [s]')
    sps[0].set_ylabel('Stress [kPa]')
    sps[1].set_xlabel('Strain')
    sps[1].set_ylabel('Stress [kPa]')
    sps[0].legend()
    sps[1].legend()
    plt.show()
