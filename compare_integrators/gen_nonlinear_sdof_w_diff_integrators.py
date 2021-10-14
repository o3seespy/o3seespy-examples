import numpy as np
import o3seespy as o3
import eqsig
import all_paths as ap


def run_analysis(asig, period, xi, f_yield, etype):
    # Load a ground motion

    # Define inelastic SDOF
    mass = 1.0

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
    bilinear_mat = o3.uniaxial_material.Steel01(osi, fy=f_yield, e0=k_spring, b=r_post)

    # Assign zero length element, # Note: pass actual node and material objects into element
    o3.element.ZeroLength(osi, [bot_node, top_node], mats=[bilinear_mat], dirs=[o3.cc.DOF2D_X], r_flag=1)

    # Define the dynamic analysis

    # Define the dynamic analysis
    acc_series = o3.time_series.Path(osi, dt=asig.dt, values=-1 * asig.values)  # should be negative
    o3.pattern.UniformExcitation(osi, dir=o3.cc.X, accel_series=acc_series)

    # set damping based on first eigen mode
    angular_freqs = np.array(o3.get_eigen(osi, solver='fullGenLapack', n=1)) ** 0.5
    beta_k = 2 * xi / angular_freqs[0]
    print('angular_freqs: ', angular_freqs)
    periods = 2 * np.pi / angular_freqs

    o3.rayleigh.Rayleigh(osi, alpha_m=0.0, beta_k=beta_k, beta_k_init=0.0, beta_k_comm=0.0)

    # Run the dynamic analysis
    o3.wipe_analysis(osi)

    # Run the dynamic analysis
    o3.constraints.Transformation(osi)
    o3.test_check.NormDispIncr(osi, tol=1.0e-6, max_iter=35, p_flag=0)
    o3.numberer.RCM(osi)
    if etype == 'implicit':
        o3.algorithm.Newton(osi)
        o3.system.SparseGeneral(osi)
        o3.integrator.Newmark(osi, gamma=0.5, beta=0.25)
        analysis_dt = 0.01
    else:
        o3.algorithm.Linear(osi, factor_once=True)
        o3.system.FullGeneral(osi)
        if etype == 'newmark_explicit':
            o3.integrator.NewmarkExplicit(osi, gamma=0.6)
            explicit_dt = periods[0] / np.pi / 32
        elif etype == 'central_difference':
            o3.integrator.CentralDifference(osi)
            o3.opy.integrator('HHTExplicit')
            explicit_dt = periods[0] / np.pi / 16  # 0.5 is a factor of safety
        elif etype == 'explicit_difference':
            o3.integrator.ExplicitDifference(osi)
            explicit_dt = periods[0] / np.pi / 32
        else:
            raise ValueError(etype)
        print('explicit_dt: ', explicit_dt)
        analysis_dt = explicit_dt
    o3.analysis.Transient(osi)

    analysis_time = asig.time[-1]

    outputs = {
        "time": [],
        "rel_disp": [],
        "rel_accel": [],
        "rel_vel": [],
        "force": []
    }

    while o3.get_time(osi) < analysis_time:
        o3.analyze(osi, 1, analysis_dt)
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


def run(show):
    period = 0.5
    xi = 0.05
    f_yield = 5.5  # Reduce this to make it nonlinear
    f_yield = 1.5  # Reduce this to make it nonlinear
    record_filename = 'test_motion_dt0p01.txt'
    asig = eqsig.load_asig(ap.MODULE_DATA_PATH + 'gms/' + record_filename, m=0.5)
    etypes = ['implicit', 'newmark_explicit', 'explicit_difference', 'central_difference']
    cs = ['k', 'b', 'g', 'orange', 'm']
    ls = ['-', '--', '-.', ':', '--']
    for i in range(len(etypes)):
        etype = etypes[i]
        od = run_analysis(asig, period, xi, f_yield, etype=etype)

        if show:
            import matplotlib.pyplot as plt
            plt.plot(od['time'], od['rel_disp'], label=etype, c=cs[i], ls=ls[i])
            periods = np.array([period])
    if show:
        # Compare closed form elastic solution
        from eqsig import sdof
        resp_u, resp_v, resp_a = sdof.response_series(motion=asig.values, dt=asig.dt, periods=periods, xi=xi)
        plt.plot(asig.time, resp_u[0], ls='--', label='Elastic', c='r', lw=1)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    run(show=1)
