import eqsig
from eqsig import sdof
import numpy as np

import o3seespy as o3
import all_paths as ap


def get_elastic_response(mass, k_spring, motion, dt, xi=0.05, r_post=0.0):
    """
    Run seismic analysis of a nonlinear SDOF

    :param mass: SDOF mass
    :param k_spring: spring stiffness
    :param motion: array_like,
        acceleration values
    :param dt: float, time step of acceleration values
    :param xi: damping ratio
    :param r_post: post-yield stiffness
    :return:
    """
    osi = o3.OpenSeesInstance(ndm=2, state=0)

    height = 5.
    # Establish nodes
    bot_node = o3.node.Node(osi, 0, 0)
    top_node = o3.node.Node(osi, 0, height)

    # Fix bottom node
    o3.Fix3DOF(osi, top_node, o3.cc.FREE, o3.cc.FIXED, o3.cc.FREE)
    o3.Fix3DOF(osi, bot_node, o3.cc.FIXED, o3.cc.FIXED, o3.cc.FIXED)
    # Set out-of-plane DOFs to be slaved
    o3.EqualDOF(osi, top_node, bot_node, [o3.cc.Y])

    # nodal mass (weight / g):
    o3.Mass(osi, top_node, mass, 0., 0.)

    # Define material
    transf = o3.geom_transf.Linear2D(osi, [])
    area = 1.0
    e_mod = 1.0e6
    iz = k_spring * height ** 3 / (3 * e_mod)
    ele_nodes = [bot_node, top_node]

    ele = o3.element.ElasticBeamColumn2D(osi, ele_nodes, area=area, e_mod=e_mod, iz=iz, transf=transf)
    # Define the dynamic analysis
    acc_series = o3.time_series.Path(osi, dt=dt, values=-motion)  # should be negative
    o3.pattern.UniformExcitation(osi, dir=o3.cc.X, accel_series=acc_series)

    # set damping based on first eigen mode
    angular_freq = o3.get_eigen(osi, solver='fullGenLapack', n=1)[0] ** 0.5
    response_period = 2 * np.pi / angular_freq
    print('response_period: ', response_period)
    beta_k = 2 * xi / angular_freq
    o3.rayleigh.Rayleigh(osi, alpha_m=0.0, beta_k=beta_k, beta_k_init=0.0, beta_k_comm=0.0)

    # Run the dynamic analysis

    o3.wipe_analysis(osi)

    o3.algorithm.Newton(osi)
    o3.system.SparseGeneral(osi)
    o3.numberer.RCM(osi)
    o3.constraints.Transformation(osi)
    o3.integrator.Newmark(osi, 0.5, 0.25)
    o3.analysis.Transient(osi)

    o3.test_check.EnergyIncr(osi, tol=1.0e-10, max_iter=10)
    analysis_time = (len(motion) - 1) * dt
    analysis_dt = 0.001
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
        outputs["force"].append(o3.get_ele_response(osi, ele, 'force'))
    o3.wipe(osi)
    for item in outputs:
        outputs[item] = np.array(outputs[item])

    return outputs


def test_sdof():
    """
    Create a plot of an elastic analysis, nonlinear analysis and closed form elastic

    :return:
    """

    record_filename = 'short_motion_dt0p01.txt'
    asig = eqsig.load_asig(ap.MODULE_DATA_PATH + 'gms/' + record_filename, m=0.5)
    period = 1.0
    xi = 0.05
    mass = 1.0
    f_yield = 1.5  # Reduce this to make it nonlinear
    r_post = 0.0

    periods = np.array([period])
    resp_u, resp_v, resp_a = sdof.response_series(motion=asig.values, dt=asig.dt, periods=periods, xi=xi)

    k_spring = 4 * np.pi ** 2 * mass / period ** 2
    outputs = get_elastic_response(mass, k_spring, asig.values, asig.dt, xi=xi, r_post=r_post)
    acc_opensees_elastic = np.interp(asig.time, outputs["time"], outputs["rel_accel"]) - asig.values
    time = asig.time

    run = 1
    if run:
        import matplotlib.pyplot as plt
        bf, sps = plt.subplots(nrows=3)
        sps[0].plot(time, resp_u[0], lw=0.7, c='r')
        sps[0].plot(outputs["time"], outputs["rel_disp"], ls='--')
        sps[1].plot(outputs['rel_disp'], outputs['force'][:, 2])
        sps[2].plot(time, resp_a[0], lw=0.7, c='r')
        sps[2].plot(outputs["time"], outputs["rel_accel"], ls='--')
        sps[2].plot(time, acc_opensees_elastic, ls='--', c='g')
        plt.show()


if __name__ == '__main__':
    test_sdof()
