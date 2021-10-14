from collections import OrderedDict

import sfsimodels
import eqsig

import numpy as np

from openseespy import opensees as opy
import o3seespy as o3


def calc_yield_curvature(depth, eps_yield):
    """
    The yield curvature of a section from Priestley (Fig 4.15)

    :param depth:
    :param eps_yield:
    :return:
    """
    # TODO: get full validation of equation
    return 2.1 * eps_yield / depth


def get_inelastic_response(fb, asig, etype, extra_time=0.0, xi=0.05, analysis_dt=0.001):
    """
    Run seismic analysis of a nonlinear FrameBuilding

    Parameters
    ----------
    fb: sfsimodels.Frame2DBuilding object
    asig: eqsig.AccSignal object
    extra_time
    xi
    analysis_dt

    Returns
    -------

    """
    osi = o3.OpenSeesInstance(ndm=2)

    q_floor = 6000.  # kPa
    trib_width = fb.floor_length / 3
    trib_mass_per_length = q_floor * trib_width / 9.8

    # Establish nodes and set mass based on trib area
    # Nodes named as: C<column-number>-S<storey-number>, first column starts at C1-S0 = ground level left
    nd = OrderedDict()
    col_xs = np.cumsum(fb.bay_lengths)
    col_xs = np.insert(col_xs, 0, 0)
    n_cols = len(col_xs)
    sto_ys = fb.heights
    sto_ys = np.insert(sto_ys, 0, 0)
    for cc in range(1, n_cols + 1):
        for ss in range(fb.n_storeys + 1):
            nd[f"C{cc}-S{ss}"] = o3.node.Node(osi, col_xs[cc - 1], sto_ys[ss])

            if ss != 0:
                if cc == 1:
                    node_mass = trib_mass_per_length * fb.bay_lengths[0] / 2
                elif cc == n_cols:
                    node_mass = trib_mass_per_length * fb.bay_lengths[-1] / 2
                else:
                    node_mass = trib_mass_per_length * (fb.bay_lengths[cc - 2] + fb.bay_lengths[cc - 1] / 2)
                o3.set_node_mass(osi, nd[f"C{cc}-S{ss}"], node_mass, node_mass, node_mass * 1.5 ** 2)

    # Set all nodes on a storey to have the same displacement
    for ss in range(0, fb.n_storeys + 1):
        for cc in range(1, n_cols + 1):
            o3.set_equal_dof(osi, nd[f"C1-S{ss}"], nd[f"C{cc}-S{ss}"], o3.cc.X)

    # Fix all base nodes
    for cc in range(1, n_cols + 1):
        o3.Fix3DOF(osi, nd[f"C{cc}-S0"], o3.cc.FIXED, o3.cc.FIXED, o3.cc.FIXED)

    # Coordinate transformation
    transf = o3.geom_transf.Linear2D(osi, [])

    l_hinge = fb.bay_lengths[0] * 0.1

    # Define material
    e_conc = 30.0e6
    i_beams = 0.4 * fb.beam_widths * fb.beam_depths ** 3 / 12
    i_columns = 0.5 * fb.column_widths * fb.column_depths ** 3 / 12
    a_beams = fb.beam_widths * fb.beam_depths
    a_columns = fb.column_widths * fb.column_depths
    ei_beams = e_conc * i_beams
    ei_columns = e_conc * i_columns
    eps_yield = 300.0e6 / 200e9
    phi_y_col = calc_yield_curvature(fb.column_depths, eps_yield)
    phi_y_beam = calc_yield_curvature(fb.beam_depths, eps_yield) * 10  # TODO: re-evaluate

    # Define beams and columns
    # Columns named as: C<column-number>-S<storey-number>, first column starts at C1-S0 = ground floor left
    # Beams named as: B<bay-number>-S<storey-number>, first beam starts at B1-S1 = first storey left (foundation at S0)

    md = OrderedDict()  # material dict
    sd = OrderedDict()  # section dict
    ed = OrderedDict()  # element dict

    for ss in range(fb.n_storeys):

        # set columns
        for cc in range(1, fb.n_cols + 1):
            lp_i = 0.4
            lp_j = 0.4  # plastic hinge length
            ele_str = f"C{cc}-S{ss}S{ss+1}"

            top_sect = o3.section.Elastic2D(osi, e_conc, a_columns[ss][cc - 1], i_columns[ss][cc - 1])
            bot_sect = o3.section.Elastic2D(osi, e_conc, a_columns[ss][cc - 1], i_columns[ss][cc - 1])
            centre_sect = o3.section.Elastic2D(osi, e_conc, a_columns[ss][cc - 1], i_columns[ss][cc - 1])
            sd[ele_str + "T"] = top_sect
            sd[ele_str + "B"] = bot_sect
            sd[ele_str + "C"] = centre_sect

            integ = o3.beam_integration.HingeMidpoint(osi, bot_sect, lp_i, top_sect, lp_j, centre_sect)

            bot_node = nd[f"C{cc}-S{ss}"]
            top_node = nd[f"C{cc}-S{ss+1}"]
            ed[ele_str] = o3.element.ForceBeamColumn(osi, [bot_node, top_node], transf, integ)

        # Set beams
        for bb in range(1, fb.n_bays + 1):
            lp_i = 0.5
            lp_j = 0.5
            ele_str = f"C{bb-1}C{bb}-S{ss}"

            mat = o3.uniaxial_material.ElasticBilin(osi, ei_beams[ss][bb - 1], 0.05 * ei_beams[ss][bb - 1], phi_y_beam[ss][bb - 1])
            md[ele_str] = mat
            left_sect = o3.section.Uniaxial(osi, mat, quantity=o3.cc.M_Z)
            right_sect = o3.section.Uniaxial(osi, mat, quantity=o3.cc.M_Z)
            centre_sect = o3.section.Elastic2D(osi, e_conc, a_beams[ss][bb - 1], i_beams[ss][bb - 1])
            integ = o3.beam_integration.HingeMidpoint(osi, left_sect, lp_i, right_sect, lp_j, centre_sect)

            left_node = nd[f"C{bb}-S{ss+1}"]
            right_node = nd[f"C{bb+1}-S{ss+1}"]
            ed[ele_str] = o3.element.ForceBeamColumn(osi, [left_node, right_node], transf, integ)

    # Define the dynamic analysis
    a_series = o3.time_series.Path(osi, dt=asig.dt, values=-1 * asig.values)  # should be negative
    o3.pattern.UniformExcitation(osi, dir=o3.cc.X, accel_series=a_series)

    # set damping based on first eigen mode
    # angular_freq_sqrd = o3.get_eigen(osi, solver='fullGenLapack', n=1)
    # if hasattr(angular_freq_sqrd, '__len__'):
    #     angular_freq = angular_freq_sqrd[0] ** 0.5
    # else:
    #     angular_freq = angular_freq_sqrd ** 0.5
    # if isinstance(angular_freq, complex):
    #     raise ValueError("Angular frequency is complex, issue with stiffness or mass")

    angular_freqs = np.array(o3.get_eigen(osi, solver='fullGenLapack', n=2)) ** 0.5
    print('angular_freqs: ', angular_freqs)
    periods = 2 * np.pi / angular_freqs
    print('periods: ', periods)

    omega0 = 0.2
    omega1 = 10.0
    a1 = 2. * xi / (omega0 + omega1)
    a0 = a1 * omega0 * omega1

    beta_k = 2 * xi / angular_freqs[0]
    # o3.rayleigh.Rayleigh(osi, alpha_m=a0, beta_k=a1, beta_k_init=0.0, beta_k_comm=0.0)

    o3.ModalDamping(osi, [xi, xi])

    # Run the dynamic analysis

    o3.wipe_analysis(osi)

    o3.constraints.Transformation(osi)
    o3.test_check.NormDispIncr(osi, tol=1.0e-5, max_iter=35, p_flag=0)
    o3.numberer.RCM(osi)
    if etype == 'implicit':
        o3.algorithm.Newton(osi)
        o3.system.FullGeneral(osi)
        o3.integrator.Newmark(osi, gamma=0.5, beta=0.25)
        dt = 0.01
    else:
        o3.algorithm.Linear(osi)

        if etype == 'newmark_explicit':
            o3.system.FullGeneral(osi)
            o3.integrator.NewmarkExplicit(osi, gamma=0.6)
            explicit_dt = periods[0] / np.pi / 8
        elif etype == 'central_difference':
            o3.system.FullGeneral(osi)
            o3.integrator.CentralDifference(osi)
            explicit_dt = periods[0] / np.pi / 16  # 0.5 is a factor of safety
        elif etype == 'hht_explicit':
            o3.integrator.HHTExplicit(osi, alpha=0.5)
            explicit_dt = periods[0] / np.pi / 8
        elif etype == 'explicit_difference':
            # o3.opy.system('Diagonal')
            o3.system.Diagonal(osi)
            o3.integrator.ExplicitDifference(osi)
            explicit_dt = periods[0] / np.pi / 4
        else:
            raise ValueError(etype)
        print('explicit_dt: ', explicit_dt)
        dt = explicit_dt
    o3.analysis.Transient(osi)

    analysis_time = (len(asig.values) - 1) * asig.dt + extra_time
    outputs = {
        "time": [],
        "rel_disp": [],
        "rel_accel": [],
        "rel_vel": [],
        "force": [],
        "ele_mom": [],
        "ele_curve": [],
    }
    print("Analysis starting")
    while o3.get_time(osi) < analysis_time:
        curr_time = opy.getTime()
        o3.analyze(osi, 1, analysis_dt)
        outputs["time"].append(curr_time)
        outputs["rel_disp"].append(o3.get_node_disp(osi, nd["C%i-S%i" % (1, fb.n_storeys)], o3.cc.X))
        outputs["rel_vel"].append(o3.get_node_vel(osi, nd["C%i-S%i" % (1, fb.n_storeys)], o3.cc.X))
        outputs["rel_accel"].append(o3.get_node_accel(osi, nd["C%i-S%i" % (1, fb.n_storeys)], o3.cc.X))
        # outputs['ele_mom'].append(opy.eleResponse('-ele', [ed['B%i-S%i' % (1, 0)], 'basicForce']))
        o3.gen_reactions(osi)
        react = 0
        for cc in range(1, fb.n_cols):
            react += -o3.get_node_reaction(osi, nd["C%i-S%i" % (cc, 0)], o3.cc.X)
        outputs["force"].append(react)  # Should be negative since diff node
    o3.wipe(osi)
    for item in outputs:
        outputs[item] = np.array(outputs[item])

    return outputs


def load1_small_frame_building_sample_data():
    """
    Sample data for the FrameBuilding object

    :param fb:
    :return:
    """
    number_of_storeys = 1
    interstorey_height = 3.4  # m
    masses = 40.0e3  # kg
    n_bays = 1

    fb = sfsimodels.FrameBuilding2D(number_of_storeys, n_bays)
    fb.interstorey_heights = interstorey_height * np.ones(number_of_storeys)
    fb.floor_length = 18.0  # m
    fb.floor_width = 16.0  # m
    fb.storey_masses = masses * np.ones(number_of_storeys)  # kg

    fb.bay_lengths = [6.]
    fb.set_beam_prop("depth", [0.5], repeat="up")
    fb.set_beam_prop("width", [0.4], repeat="up")
    fb.set_column_prop("width", [0.5, 0.5], repeat="up")
    fb.set_column_prop("depth", [0.5, 0.5], repeat="up")
    return fb


def load_small_frame_building_sample_data():
    """
    Sample data for the FrameBuilding object

    :param fb:
    :return:
    """

    interstorey_heights = np.array([4.9, 3.4, 3.4, 3.4])  # m
    bay_lengths = np.array([4.5, 4.5, 3.2, 3.2])
    number_of_storeys = len(interstorey_heights)
    storey_pressure = 8.0e3  # Pa
    n_bays = len(bay_lengths)

    fb = sfsimodels.FrameBuilding2D(number_of_storeys, n_bays)
    fb.interstorey_heights = interstorey_heights
    fb.floor_length = 15.4  # m
    fb.floor_width = 5.0  # m
    fb.storey_masses = storey_pressure / 9.8 * fb.floor_length * fb.floor_width * np.ones(number_of_storeys)  # kg

    fb.bay_lengths = bay_lengths
    fb.set_beam_prop("depth", 0.6, repeat="all")
    fb.set_beam_prop("width", 0.4, repeat="all")
    fb.set_column_prop("width", 0.5, repeat="all")
    fb.set_column_prop("depth", 0.5, repeat="all")
    return fb


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import all_paths as ap
    record_filename = 'short_motion_dt0p01.txt'
    motion_step = 0.01
    rec = np.loadtxt(ap.MODULE_DATA_PATH + f'gms/{record_filename}', skiprows=2)

    xi = 0.05

    acc_signal = eqsig.AccSignal(rec, motion_step)
    time = acc_signal.time

    # frame = load_frame_building_sample_data()
    frame = load_small_frame_building_sample_data()
    print("Building loaded")

    etypes = ['implicit']  # , 'central_difference'
    # etypes =['central_difference']
    for i, etype in enumerate(etypes):
        outputs = get_inelastic_response(frame, acc_signal, etype, xi=xi, extra_time=0)
        print("Analysis complete")
        acc_opensees = np.interp(time, outputs["time"], outputs["rel_accel"]) - rec
        ux_opensees = np.interp(time, outputs["time"], outputs["rel_disp"])
        plt.plot(ux_opensees)
    plt.show()
    print("Complete")
