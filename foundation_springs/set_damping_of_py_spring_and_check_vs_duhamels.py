import numpy as np
import o3seespy as o3
import eqsig
import all_paths as ap


def run_analysis(mat, xi, asig):

    # Define inelastic SDOF
    mass = 1.0

    # Initialise OpenSees instance
    osi = o3.OpenSeesInstance(ndm=2, state=3)

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
    mat.build(osi)
    # Assign zero length element, # Note: pass actual node and material objects into element
    o3.element.ZeroLength(osi, [bot_node, top_node], mats=[mat], dirs=[o3.cc.DOF2D_X], r_flag=1)

    # Define the dynamic analysis

    # Define the dynamic analysis
    acc_series = o3.time_series.Path(osi, dt=asig.dt, values=-1 * asig.values)  # should be negative
    o3.pattern.UniformExcitation(osi, dir=o3.cc.X, accel_series=acc_series)

    # set damping based on first eigen mode
    angular_freq = o3.get_eigen(osi, solver='fullGenLapack', n=1)[0] ** 0.5
    beta_k = 2 * xi / angular_freq
    o3.rayleigh.Rayleigh(osi, alpha_m=0.0, beta_k=beta_k, beta_k_init=0.0, beta_k_comm=0.0)

    # Run the dynamic analysis
    o3.wipe_analysis(osi)

    # Run the dynamic analysis
    o3.algorithm.Newton(osi)
    o3.system.SparseGeneral(osi)
    o3.numberer.RCM(osi)
    o3.constraints.Transformation(osi)
    o3.integrator.Newmark(osi, gamma=0.5, beta=0.25)
    o3.analysis.Transient(osi)
    o3.extensions.to_py_file(osi)

    o3.test_check.EnergyIncr(osi, tol=1.0e-10, max_iter=10)
    analysis_time = asig.time[-1]
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
        outputs["force"].append(-o3.get_node_reaction(osi, bot_node, o3.cc.X))  # Negative since diff node
    o3.wipe(osi)
    for item in outputs:
        outputs[item] = np.array(outputs[item])

    return outputs


def run(show=0):
    period = 1.0
    mass = 1.0
    xi = 0.05
    k_spring = 4 * np.pi ** 2 * mass / period ** 2
    f_yield = 1.5  # Reduce this to make it nonlinear
    r_post = 0.0

    # Load a ground motion
    record_filename = 'test_motion_dt0p01.txt'
    asig = eqsig.load_asig(ap.MODULE_DATA_PATH + 'gms/' + record_filename, m=0.5)

    smat = o3.uniaxial_material.Steel01(None, fy=f_yield, e0=k_spring, b=r_post)
    soil_type = 2
    kvfactor = 1.39  # for QzSimple2 material (soil_type=2)
    kxpfactor = 0.542  # for PySimple2 material (soil_type=2)
    ele_damping = 1
    if ele_damping:
        # xi_h_sys = omega_sys * c_h / (2 * k_h)
        crad = xi * 2 * k_spring / (2 * np.pi / period)
        # crad = xi * 100
        xi_a = 0.0
    else:
        crad = 0.0
        xi_a = xi
    f_ult = 10.0
    z50 = kvfactor * f_ult / k_spring
    z50p = kxpfactor * f_ult / k_spring
    # smat = o3.uniaxial_material.QzSimple2(None, soil_type, f_ult, z50, 0.0, crad_soil['vert'])
    smat = o3.uniaxial_material.PySimple2(None, soil_type, f_ult, z50p, cd=0.1, c=crad)
    od = run_analysis(smat, xi_a, asig)
    if show:
        import matplotlib.pyplot as plt
        plt.plot(od['time'], od['rel_disp'], label='o3seespy')
        periods = np.array([period])

        # Compare closed form elastic solution
        from eqsig import sdof
        resp_u, resp_v, resp_a = sdof.response_series(motion=asig.values, dt=asig.dt, periods=periods, xi=xi)
        plt.plot(asig.time, resp_u[0], ls='--', label='Elastic')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    run(show=1)
