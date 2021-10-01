import o3seespy as o3

import numpy as np
import matplotlib.pyplot as plt
import time


def run_mz_triaxial():
    """
    This function runs an o3seespy equivalent of the ManzariDafalias triaxial compression
    example from https://opensees.berkeley.edu/wiki/index.php/Manzari_Dafalias_Material

    The intention is to demonstrate the compatibility between o3seespy and the Tcl version
    of OpenSees.
    """

    damp = 0.1
    omega0 = 0.0157
    omega1 = 64.123
    a1 = 2. * damp / (omega0 + omega1)
    a0 = a1 * omega0 * omega1

    # Initialise OpenSees instance
    osi = o3.OpenSeesInstance(ndm=3, ndf=4, state=3)

    # Establish nodes
    n_coords = [
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0]
    ]
    nm = []
    for nc in n_coords:
        nm.append(o3.node.Node(osi, *nc))

    o3.Fix4DOF(osi, nm[0], o3.cc.FREE, o3.cc.FIXED, o3.cc.FIXED, o3.cc.FIXED)
    o3.Fix4DOF(osi, nm[1], o3.cc.FREE, o3.cc.FREE, o3.cc.FIXED, o3.cc.FIXED)
    o3.Fix4DOF(osi, nm[2], o3.cc.FIXED, o3.cc.FREE, o3.cc.FIXED, o3.cc.FIXED)
    o3.Fix4DOF(osi, nm[3], o3.cc.FIXED, o3.cc.FIXED, o3.cc.FIXED, o3.cc.FIXED)
    o3.Fix4DOF(osi, nm[4], o3.cc.FREE, o3.cc.FIXED, o3.cc.FREE, o3.cc.FIXED)
    o3.Fix4DOF(osi, nm[5], o3.cc.FREE, o3.cc.FREE, o3.cc.FREE, o3.cc.FIXED)
    o3.Fix4DOF(osi, nm[6], o3.cc.FIXED, o3.cc.FREE, o3.cc.FREE, o3.cc.FIXED)
    o3.Fix4DOF(osi, nm[7], o3.cc.FIXED, o3.cc.FIXED, o3.cc.FREE, o3.cc.FIXED)

    # Define material
    p_conf = -300.0  # confinement stress
    dev_disp = -0.3  # deviatoric strain
    perm = 1.0e-10  # permeability
    e_curr = 0.8  # void ratio

    mzmod = o3.nd_material.ManzariDafalias(osi, g0=125, nu=0.05, e_init=0.8, m_c=1.25, c_c=0.712, lambda_c=0.019,
                                              e_0=0.934, ksi=0.7, p_atm=100, m_yield=0.01, h_0=7.05, c_h=0.968, n_b=1.1,
                                              a_0=0.704, n_d=3.5, z_max=4, c_z=600, den=1.42)

    water_bulk_mod = 2.2e6
    f_den = 1.0
    ele = o3.element.SSPbrickUP(osi, nm, mzmod, water_bulk_mod, f_den, perm,
                                perm, perm, void=e_curr, alpha=1.5e-9, b1=0.0, b2=0.0, b3=0.0)

    all_stresses_cache = o3.recorder.ElementToArrayCache(osi, ele, arg_vals=['stress'], fname='stresses_03.txt')
    all_strains_cache = o3.recorder.ElementToArrayCache(osi, ele, arg_vals=['strain'], fname='strains_03.txt')
    nodes_cache = o3.recorder.NodesToArrayCache(osi, nm, dofs=[1, 2, 3], res_type='disp')

    o3.constraints.Penalty(osi, 1.0e18, 1.0e18)
    o3.test_check.NormDispIncr(osi, tol=1.0e-5, max_iter=20, p_flag=0)
    o3.algorithm.Newton(osi)
    o3.numberer.RCM(osi)
    o3.system.BandGeneral(osi)
    o3.integrator.Newmark(osi, gamma=0.5, beta=0.25)
    o3.rayleigh.Rayleigh(osi, a0, 0.0, a1, 0.0)
    o3.analysis.Transient(osi)

    # Add static vertical pressure and stress bias
    p_node = p_conf / 4.0
    time_series = o3.time_series.Path(osi, time=[0, 10000, 1e10], values=[0, 1, 1])
    o3.pattern.Plain(osi, time_series)
    o3.Load(osi, nm[0], [p_node, 0, 0, 0])
    o3.Load(osi, nm[1], [p_node, p_node, 0, 0])
    o3.Load(osi, nm[2], [0, p_node, 0, 0])
    o3.Load(osi, nm[3], [0, 0, 0, 0])
    o3.Load(osi, nm[4], [p_node, 0, p_node, 0])
    o3.Load(osi, nm[5], [p_node, p_node, p_node, 0])
    o3.Load(osi, nm[6], [0, p_node, p_node, 0])
    o3.Load(osi, nm[7], [0, 0, p_node, 0])

    o3.analyze(osi, num_inc=100, dt=100)
    o3.analyze(osi, 50, 100)

    # Close the drainage valves
    for node in nm:
        o3.remove_sp(osi, node, dof=4)
    o3.analyze(osi, 50, dt=100)

    z_vert = o3.get_node_disp(osi, nm[4], o3.cc.DOF3D_Z)
    l_values = [1, 1 + dev_disp / z_vert, 1 + dev_disp / z_vert]
    ts2 = o3.time_series.Path(osi, time=[20000, 1020000, 10020000], values=l_values, factor=1)
    o3.pattern.Plain(osi, ts2, fact=1.)
    o3.SP(osi, nm[4], dof=o3.cc.DOF3D_Z, dof_values=[z_vert])
    o3.SP(osi, nm[5], dof=o3.cc.DOF3D_Z, dof_values=[z_vert])
    o3.SP(osi, nm[6], dof=o3.cc.DOF3D_Z, dof_values=[z_vert])
    o3.SP(osi, nm[7], dof=o3.cc.DOF3D_Z, dof_values=[z_vert])

    o3.extensions.to_py_file(osi)

    dt = 100
    num_step = 10000

    rem_step = num_step

    def sub_step_analyze(dt, sub_step):
        loc_success = 0
        if sub_step > 10:
            return -10
        for i in range(1, 3):
            print(f'try dt = {dt}')
            loc_success = o3.analyze(osi, 1, dt)
            if loc_success != 0:
                loc_success = sub_step_analyze(dt / 2., sub_step + 1)
                if success == -1:
                    print('Did not converge.')
                    return loc_success
            else:
                if i == 1:
                    print(f'Substep {sub_step}: Left side converged with dt = {dt}')
                else:
                    print(f'Substep {sub_step}: Right side converged with dt = {dt}')
        return loc_success

    print('Start analysis')
    start_t = time.process_time()
    success = 0
    while success != -10:
        sub_step = 0
        success = o3.analyze(osi, rem_step, dt)
        if success == 0:
            print('Analysis Finished')
            break
        else:
            cur_time = o3.get_time(osi)
            print(f'Analysis failed at {cur_time} . Try substepping.')
            success = sub_step_analyze(dt / 2, sub_step + 1)
            cur_step = int((cur_time - 20000) / dt + 1)
            rem_step = int(num_step - cur_step)
            print(f'Current step: {cur_step}, Remaining steps: {rem_step}')

    end_t = time.process_time()
    print(f'loading analysis execution time: {end_t - start_t:.2f} seconds.')

    o3.wipe(osi)
    all_stresses = all_stresses_cache.collect()
    all_strains = all_strains_cache.collect()
    disps = nodes_cache.collect()

    return all_stresses, all_strains, disps

    pass


if __name__ == '__main__':

    stresses, strains, disps = run_mz_triaxial()

    bf, sps = plt.subplots(nrows=2)
    sps[0].plot(stresses[:, 0], label='stress0', c='r')
    sps[0].plot(stresses[:, 1], label='stress1', c='b')
    sps[0].plot(stresses[:, 2], label='stress2', c='g')
    # sps[0].plot(ppt, label='PPT')
    sps[1].plot(strains[:, 2], stresses[:, 2], label='o3seespy')

    stresses_tcl = np.loadtxt('stress.out')
    strains_tcl = np.loadtxt('strain.out')
    sps[0].plot(stresses_tcl[:, 0], label='stress0-TCL', c='r', ls='--', lw=3)
    sps[0].plot(stresses_tcl[:, 1], label='stress1-TCL', c='b', ls=':', lw=3)
    sps[0].plot(stresses_tcl[:, 2], label='stress2-TCL', c='g', ls='--', lw=3)
    sps[1].plot(strains_tcl[:, 2], stresses_tcl[:, 2], label='Tcl', c='r', ls='--')

    sps[0].set_xlabel('Time [s]')
    sps[0].set_ylabel('Stress [kPa]')
    sps[0].legend()
    sps[1].legend()
    bf.savefig('Comparison-ManzariDafalias-tcl-vs-o3.png')
    plt.show()
