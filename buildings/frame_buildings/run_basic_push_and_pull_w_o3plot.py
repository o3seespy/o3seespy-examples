from collections import OrderedDict

import matplotlib.pyplot as plt
import sfsimodels
import eqsig

import numpy as np
import o3plot
from openseespy import opensees as opy
import o3seespy as o3
import all_paths as ap
import os


def calc_yield_curvature(depth, eps_yield):
    """
    The yield curvature of a section from Priestley (Fig 4.15)

    :param depth:
    :param eps_yield:
    :return:
    """
    # TODO: get full validation of equation
    return 2.1 * eps_yield / depth


def get_inelastic_response(fb, roof_drift_ratio=0.05, elastic=False, w_sfsi=False, out_folder=''):
    """
    Run seismic analysis of a nonlinear FrameBuilding

    Units: Pa, N, m, s

    Parameters
    ----------
    fb: sfsimodels.Frame2DBuilding object
    xi

    Returns
    -------

    """
    osi = o3.OpenSeesInstance(ndm=2, state=3)

    q_floor = 7.0e3  # Pa
    trib_width = fb.floor_length
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
                o3.set_node_mass(osi, nd[f"C{cc}-S{ss}"], node_mass, 0., 0.)

    # Set all nodes on a storey to have the same displacement
    for ss in range(0, fb.n_storeys + 1):
        for cc in range(2, n_cols + 1):
            o3.set_equal_dof(osi, nd[f"C1-S{ss}"], nd[f"C{cc}-S{ss}"], o3.cc.X)

    # Fix all base nodes
    for cc in range(1, n_cols + 1):
        o3.Fix3DOF(osi, nd[f"C{cc}-S0"], o3.cc.FIXED, o3.cc.FIXED, o3.cc.FIXED)

    # Define material
    e_conc = 30.0e9  # kPa
    i_beams = 0.4 * fb.beam_widths * fb.beam_depths ** 3 / 12
    i_columns = 0.5 * fb.column_widths * fb.column_depths ** 3 / 12
    a_beams = fb.beam_widths * fb.beam_depths
    a_columns = fb.column_widths * fb.column_depths
    ei_beams = e_conc * i_beams
    ei_columns = e_conc * i_columns
    eps_yield = 300.0e6 / 200e9
    phi_y_col = calc_yield_curvature(fb.column_depths, eps_yield)
    phi_y_beam = calc_yield_curvature(fb.beam_depths, eps_yield)

    # Define beams and columns
    # Columns named as: C<column-number>-S<storey-number>, first column starts at C1-S0 = ground floor left
    # Beams named as: B<bay-number>-S<storey-number>, first beam starts at B1-S1 = first storey left (foundation at S0)

    md = OrderedDict()  # material dict
    sd = OrderedDict()  # section dict
    ed = OrderedDict()  # element dict

    for ss in range(fb.n_storeys):

        # set columns
        lp_i = 0.4
        lp_j = 0.4  # plastic hinge length
        col_transf = o3.geom_transf.Linear2D(osi, )  # d_i=[0.0, lp_i], d_j=[0.0, -lp_j]
        for cc in range(1, fb.n_cols + 1):

            ele_str = f"C{cc}-S{ss}S{ss + 1}"

            if elastic:
                top_sect = o3.section.Elastic2D(osi, e_conc, a_columns[ss][cc - 1], i_columns[ss][cc - 1])
                bot_sect = o3.section.Elastic2D(osi, e_conc, a_columns[ss][cc - 1], i_columns[ss][cc - 1])
            else:
                m_cap = ei_columns[ss][cc - 1] * phi_y_col[ss][cc - 1]
                mat = o3.uniaxial_material.ElasticBilin(osi, ei_columns[ss][cc - 1], 0.05 * ei_columns[ss][cc - 1],
                                                        1 * phi_y_col[ss][cc - 1])
                mat_axial = o3.uniaxial_material.Elastic(osi, e_conc * a_columns[ss][cc - 1])
                top_sect = o3.section.Aggregator(osi, mats=[[mat_axial, o3.cc.P], [mat, o3.cc.M_Z]])
                bot_sect = o3.section.Aggregator(osi, mats=[[mat_axial, o3.cc.P], [mat, o3.cc.M_Z]])

            centre_sect = o3.section.Elastic2D(osi, e_conc, a_columns[ss][cc - 1], i_columns[ss][cc - 1])
            sd[ele_str + "T"] = top_sect
            sd[ele_str + "B"] = bot_sect
            sd[ele_str + "C"] = centre_sect

            integ = o3.beam_integration.HingeMidpoint(osi, bot_sect, lp_i, top_sect, lp_j, centre_sect)

            bot_node = nd[f"C{cc}-S{ss}"]
            top_node = nd[f"C{cc}-S{ss + 1}"]
            ed[ele_str] = o3.element.ForceBeamColumn(osi, [bot_node, top_node], col_transf, integ)
            print('mc: ', ei_columns[ss][cc - 1] * phi_y_col[ss][cc - 1])
        # Set beams
        lp_i = 0.4
        lp_j = 0.4
        beam_transf = o3.geom_transf.Linear2D(osi, )
        for bb in range(1, fb.n_bays + 1):

            ele_str = f"C{bb}C{bb + 1}-S{ss + 1}"

            print('mb: ', ei_beams[ss][bb - 1] * phi_y_beam[ss][bb - 1])
            print('phi_b: ', phi_y_beam[ss][bb-1])

            if elastic:
                left_sect = o3.section.Elastic2D(osi, e_conc, a_beams[ss][bb - 1], i_beams[ss][bb - 1])
                right_sect = o3.section.Elastic2D(osi, e_conc, a_beams[ss][bb - 1], i_beams[ss][bb - 1])
            else:
                m_cap = ei_beams[ss][bb - 1] * phi_y_beam[ss][bb - 1]
                # mat_flex = o3.uniaxial_material.ElasticBilin(osi, ei_beams[ss][bb - 1], 0.05 * ei_beams[ss][bb - 1], phi_y_beam[ss][bb - 1])
                mat_flex = o3.uniaxial_material.Steel01(osi, m_cap, e0=ei_beams[ss][bb-1], b=0.05)
                mat_axial = o3.uniaxial_material.Elastic(osi, e_conc * a_beams[ss][bb - 1])
                left_sect = o3.section.Aggregator(osi, mats=[[mat_axial, o3.cc.P], [mat_flex, o3.cc.M_Z], [mat_flex, o3.cc.M_Y]])
                right_sect = o3.section.Aggregator(osi, mats=[[mat_axial, o3.cc.P], [mat_flex, o3.cc.M_Z], [mat_flex, o3.cc.M_Y]])

            centre_sect = o3.section.Elastic2D(osi, e_conc, a_beams[ss][bb - 1], i_beams[ss][bb - 1])
            integ = o3.beam_integration.HingeMidpoint(osi, left_sect, lp_i, right_sect, lp_j, centre_sect)

            left_node = nd[f"C{bb}-S{ss + 1}"]
            right_node = nd[f"C{bb + 1}-S{ss + 1}"]
            ed[ele_str] = o3.element.ForceBeamColumn(osi, [left_node, right_node], beam_transf, integ)

    # Apply gravity loads
    gravity = 9.8 * 1e-2
    # If true then load applied along beam
    g_beams = 0  # TODO: when this is true and analysis is inelastic then failure
    ts_po = o3.time_series.Linear(osi, factor=1)
    o3.pattern.Plain(osi, ts_po)

    for ss in range(1, fb.n_storeys + 1):
        print('ss:', ss)
        if g_beams:
            for bb in range(1, fb.n_bays + 1):
                ele_str = f"C{bb}C{bb + 1}-S{ss}"
                o3.EleLoad2DUniform(osi, ed[ele_str], -trib_mass_per_length * gravity)
        else:
            for cc in range(1, fb.n_cols + 1):
                if cc == 1 or cc == n_cols:
                    node_mass = trib_mass_per_length * fb.bay_lengths[0] / 2
                elif cc == n_cols:
                    node_mass = trib_mass_per_length * fb.bay_lengths[-1] / 2
                else:
                    node_mass = trib_mass_per_length * (fb.bay_lengths[cc - 2] + fb.bay_lengths[cc - 1] / 2)
                # This works
                o3.Load(osi, nd[f"C{cc}-S{ss}"], [0, -node_mass * gravity, 0])

    tol = 1.0e-3
    o3.constraints.Plain(osi)
    o3.numberer.RCM(osi)
    o3.system.BandGeneral(osi)
    o3.test_check.NormDispIncr(osi, tol, 10)
    o3.algorithm.Newton(osi)
    n_steps_gravity = 10
    d_gravity = 1. / n_steps_gravity
    o3.integrator.LoadControl(osi, d_gravity, num_iter=10)
    o3.analysis.Static(osi)
    o3.analyze(osi, n_steps_gravity)
    opy.reactions()
    print('b1_int: ', o3.get_ele_response(osi, ed['C1C2-S1'], 'force'))
    print('c1_int: ', o3.get_ele_response(osi, ed['C1-S0S1'], 'force'))

    # o3.extensions.to_py_file(osi, 'po.py')
    o3.load_constant(osi, time=0.0)

    # Define the analysis

    # set damping based on first eigen mode
    angular_freq = opy.eigen('-fullGenLapack', 1) ** 0.5
    if isinstance(angular_freq, complex):
        raise ValueError("Angular frequency is complex, issue with stiffness or mass")
    print('angular_freq: ', angular_freq)
    response_period = 2 * np.pi / angular_freq
    print('response period: ', response_period)

    # Run the analysis
    o3r = o3.results.Results2D()
    o3r.cache_path = out_folder
    o3r.dynamic = True
    o3r.start_recorders(osi)
    d_inc = 0.00001

    o3.numberer.RCM(osi)
    o3.system.BandGeneral(osi)
    # o3.test_check.NormUnbalance(osi, 2, max_iter=10, p_flag=2)
    # o3.test_check.FixedNumIter(osi, max_iter=10)
    o3.test_check.NormDispIncr(osi, 0.002, 10, p_flag=0)
    o3.algorithm.Newton(osi)
    o3.integrator.DisplacementControl(osi, nd[f"C1-S{fb.n_storeys}"], o3.cc.X, d_inc)
    o3.analysis.Static(osi)

    d_max = 0.05 * fb.max_height  # TODO: set to 5%
    print('d_max: ', d_max)
    # n_steps = int(d_max / d_inc)

    print("Analysis starting")
    print('int_disp: ', o3.get_node_disp(osi, nd[f"C1-S{fb.n_storeys}"], o3.cc.X))
    # opy.recorder('Element', '-file', 'ele_out.txt', '-time', '-ele', 1, 'force')
    tt = 0
    outputs = {
        'h_disp': [],
        'vb': [],
        'REACT-C1C2-S1': [],
        'REACT-C1-S0S1': [],
    }
    hd = 0
    n_max = 2
    n_cycs = 0
    xs = fb.heights / fb.max_height  # TODO: more sophisticated displacement profile
    ts_po = o3.time_series.Linear(osi, factor=1)
    o3.pattern.Plain(osi, ts_po)
    for i, xp in enumerate(xs):
        o3.Load(osi, nd[f"C1-S{i + 1}"], [xp, 0.0, 0])

    # o3.analyze(osi, 2)
    # n_max = 0

    while n_cycs < n_max:
        print('n_cycles: ', n_cycs)
        for i in range(2):
            if i == 0:
                o3.integrator.DisplacementControl(osi, nd[f"C1-S{fb.n_storeys}"], o3.cc.X, d_inc)
            else:
                o3.integrator.DisplacementControl(osi, nd[f"C1-S{fb.n_storeys}"], o3.cc.X, -d_inc)
            while hd * (-1) ** i < d_max:
                ok = o3.analyze(osi, 1)
                hd = o3.get_node_disp(osi, nd[f"C1-S{fb.n_storeys}"], o3.cc.X)
                outputs['h_disp'].append(hd)
                opy.reactions()
                vb = 0
                for cc in range(1, fb.n_cols + 1):
                    vb += o3.get_node_reaction(osi, nd[f"C{cc}-S0"], o3.cc.X)
                outputs['vb'].append(-vb)
                outputs['REACT-C1C2-S1'].append(o3.get_ele_response(osi, ed['C1C2-S1'], 'force'))
                outputs['REACT-C1-S0S1'].append(o3.get_ele_response(osi, ed['C1-S0S1'], 'force'))

        n_cycs += 1

    opy.wipe()
    # o3r.save_to_cache()
    for item in outputs:
        outputs[item] = np.array(outputs[item])
    print('complete')
    return outputs


def load_2storey_frame_building_sample_data():
    """
    Sample data for the FrameBuilding object

    :param fb:
    :return:
    """
    number_of_storeys = 2
    interstorey_height = 3.4  # m
    masses = 40.0e3  # kg
    n_bays = 2

    fb = sfsimodels.FrameBuilding2D(number_of_storeys, n_bays)
    fb.interstorey_heights = interstorey_height * np.ones(number_of_storeys)
    fb.floor_length = 18.0  # m
    fb.floor_width = 16.0  # m
    fb.storey_masses = masses * np.ones(number_of_storeys)  # kg

    fb.bay_lengths = [3., 3.0]
    fb.set_beam_prop("depth", 0.5, repeat="all")
    fb.set_beam_prop("width", 0.4, repeat="all")
    fb.set_column_prop("width", 0.5, repeat="all")
    fb.set_column_prop("depth", 0.5, repeat="all")
    return fb


if __name__ == '__main__':
    name = __file__.replace('.py', '')
    name = name.split("run_")[-1]
    out_folder = ap.OP_PATH + name + '/'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    frame = load_2storey_frame_building_sample_data()
    print("Building loaded")
    rerun = 0
    if rerun:
        outs = get_inelastic_response(frame, elastic=0, out_folder=out_folder)

        bf, sps = plt.subplots(nrows=2)
        sps[0].plot(outs["h_disp"], outs["REACT-C1C2-S1"][:, :3])
        # sps[0].plot(outs["h_disp"], outs["REACT-C1-S0S1"][:, :3])
        import engformat as ef
        ef.xy(sps[0], x_origin=True)
        sps[1].plot(outs["h_disp"], outs["vb"])
        plt.show()
    print("Complete")
    w_playback = 1  # TODO: not working properly
    if w_playback:
        o3res = o3.results.Results2D(cache_path=out_folder, dynamic=True)
        o3res.load_from_cache()
        o3plot.replot(o3res, xmag=5.5, t_scale=10)
    # o3plot.replot(out_folder, dynamic=1, xmag=10, dt=0.001, t_scale=1)

