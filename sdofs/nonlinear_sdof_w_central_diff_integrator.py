import numpy as np
import o3seespy as o3
import eqsig
import all_paths as ap

def gen_response(period, xi, asig, etype, fos_for_dt=None):

    # Define inelastic SDOF
    mass = 1.0
    f_yield = 1.5  # Reduce this to make it nonlinear
    r_post = 0.0

    # Initialise OpenSees instance
    osi = o3.OpenSeesInstance(ndm=2, state=0)

    # Establish nodes
    bot_node = o3.node.Node(osi, 0, 0)
    top_node = o3.node.Node(osi, 0, 0)

    # Fix bottom node
    o3.Fix3DOF(osi, top_node, o3.cc.FREE, o3.cc.FIXED, o3.cc.FIXED)
    o3.Fix3DOF(osi, bot_node, o3.cc.FIXED, o3.cc.FIXED, o3.cc.FIXED)
    # Set out-of-plane DOFs to be slaved
    o3.EqualDOF(osi, top_node, bot_node, [o3.cc.Y, o3.cc.ROTZ])

    # nodal mass (weight / g):
    o3.Mass(osi, top_node, mass, 0., 0.)

    # Define material
    k_spring = 4 * np.pi ** 2 * mass / period ** 2
    # bilinear_mat = o3.uniaxial_material.Steel01(osi, fy=f_yield, e0=k_spring, b=r_post)
    mat = o3.uniaxial_material.Elastic(osi, e_mod=k_spring)

    # Assign zero length element, # Note: pass actual node and material objects into element
    o3.element.ZeroLength(osi, [bot_node, top_node], mats=[mat], dirs=[o3.cc.DOF2D_X], r_flag=1)

    # Define the dynamic analysis

    # Define the dynamic analysis
    acc_series = o3.time_series.Path(osi, dt=asig.dt, values=-1 * asig.values)  # should be negative
    o3.pattern.UniformExcitation(osi, dir=o3.cc.X, accel_series=acc_series)

    # set damping based on first eigen mode
    angular_freq = o3.get_eigen(osi, solver='fullGenLapack', n=1)[0] ** 0.5
    period = 2 * np.pi / angular_freq
    beta_k = 2 * xi / angular_freq
    o3.rayleigh.Rayleigh(osi, alpha_m=0.0, beta_k=beta_k, beta_k_init=0.0, beta_k_comm=0.0)

    o3.set_time(osi, 0.0)
    # Run the dynamic analysis
    o3.wipe_analysis(osi)

    # Run the dynamic analysis
    o3.numberer.RCM(osi)
    o3.system.FullGeneral(osi)
    if etype == 'central_difference':
        o3.algorithm.Linear(osi, factor_once=True)
        o3.integrator.CentralDifference(osi)
        explicit_dt = 2 / angular_freq / fos_for_dt
        analysis_dt = explicit_dt
    elif etype == 'implicit':
        o3.algorithm.Newton(osi)
        o3.integrator.Newmark(osi, gamma=0.5, beta=0.25)
        analysis_dt = 0.001
    else:
        raise ValueError()
    o3.constraints.Transformation(osi)
    o3.analysis.Transient(osi)

    o3.test_check.EnergyIncr(osi, tol=1.0e-10, max_iter=10)
    analysis_time = asig.time[-1]

    outputs = {
        "time": [],
        "rel_disp": [],
        "rel_accel": [],
        "rel_vel": [],
        "force": []
    }
    rec_dt = 0.002
    n_incs = int(analysis_dt / rec_dt)
    n_incs = 1
    while o3.get_time(osi) < analysis_time:
        o3.analyze(osi, n_incs, analysis_dt)
        curr_time = o3.get_time(osi)
        outputs["time"].append(curr_time)
        outputs["rel_disp"].append(o3.get_node_disp(osi, top_node, o3.cc.X))
        outputs["rel_vel"].append(o3.get_node_vel(osi, top_node, o3.cc.X))
        outputs["rel_accel"].append(o3.get_node_accel(osi, top_node, o3.cc.X))
        o3.gen_reactions(osi)
        outputs["force"].append(-o3.get_node_reaction(osi, bot_node, o3.cc.X))  # Negative since diff node
    o3.wipe(osi)
    for item in outputs:
        outputs[item] = np.array(outputs[item])
    return outputs

def run(show=0):
    record_filename = 'test_motion_dt0p01.txt'
    asig = eqsig.load_asig(ap.MODULE_DATA_PATH + 'gms/' + record_filename, m=0.5)

    period = 1.0

    from eqsig import sdof
    import matplotlib.pyplot as plt
    import engformat as ef
    bf, ax = plt.subplots(nrows=2)
    xi = 0.05
    outputs = gen_response(period, xi, asig, etype='implicit')
    ax[0].plot(outputs['time'], outputs['rel_disp'], label='implicit', ls='--')
    outputs = gen_response(period, xi, asig, etype='central_difference', fos_for_dt=5)
    ax[0].plot(outputs['time'], outputs['rel_disp'], label='central_difference', ls='--')
    ef.text_at_rel_pos(ax[0], 0.05, 0.95, f'$\\xi$ = {xi*100:.0f}%')
    periods = np.array([period])

    # Compare closed form elastic solution
    resp_u, resp_v, resp_a = sdof.response_series(motion=asig.values, dt=asig.dt, periods=periods, xi=xi)
    ax[0].plot(asig.time, resp_u[0], ls='-', label='Elastic', zorder=0)

    xi = 0.0
    outputs = gen_response(period, xi, asig, etype='implicit')
    ax[1].plot(outputs['time'], outputs['rel_disp'], label='implicit', ls='--')
    outputs = gen_response(period, xi, asig, etype='central_difference', fos_for_dt=5)
    ax[1].plot(outputs['time'], outputs['rel_disp'], label='central_difference', ls='--')
    ef.text_at_rel_pos(ax[1], 0.05, 0.95, f'$\\xi$ = {xi*100:.0f}%')
    periods = np.array([period])

    # Compare closed form elastic solution
    resp_u, resp_v, resp_a = sdof.response_series(motion=asig.values, dt=asig.dt, periods=periods, xi=xi)
    ax[1].plot(asig.time, resp_u[0], ls='-', label='Elastic', zorder=0)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    run(show=1)
