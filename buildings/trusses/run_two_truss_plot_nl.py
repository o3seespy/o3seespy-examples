import o3seespy as o3
import numpy as np
import os
import o3plot
import matplotlib.pyplot as plt


def run(out_folder):

    osi = o3.OpenSeesInstance(ndm=2, ndf=2, state=3)
    x_centre = 0.0
    y_centre = 0.0
    top_node = o3.node.Node(osi, x_centre, y_centre)
    fd_area = 1
    fd_e_mod = 1e9
    fd_iz = 1e6
    top_nodes = []
    bot_nodes = []
    sf_eles = []
    fd_eles = []

    o3.Mass(osi, top_node, 10, 10)

    fy = 500
    k = 1.0e4
    b = 0.1
    pro_params = [5, 0.925, 0.15]
    sf_mat = o3.uniaxial_material.SteelMPF(osi, fy, fy, k, b, b, params=pro_params)

    diff_pos = 0.5
    depth = 1
    bot_nodes.append(o3.node.Node(osi, x_centre - diff_pos, y_centre - depth))
    o3.Fix2DOF(osi, bot_nodes[0], o3.cc.FIXED, o3.cc.FIXED)
    bot_nodes.append(o3.node.Node(osi, x_centre + diff_pos, y_centre - depth))
    o3.Fix2DOF(osi, bot_nodes[1], o3.cc.FIXED, o3.cc.FIXED)

    top_nodes.append(o3.node.Node(osi, x_centre, y_centre))

    sf_eles.append(o3.element.Truss(osi, [top_nodes[0], bot_nodes[0]], big_a=1.0, mat=sf_mat))
    sf_eles.append(o3.element.Truss(osi, [top_nodes[0], bot_nodes[1]], big_a=1.0, mat=sf_mat))

    o3.EqualDOF(osi, top_node, top_nodes[0], dofs=[o3.cc.DOF2D_X, o3.cc.DOF2D_Y])

    ts0 = o3.time_series.Linear(osi, factor=1)
    o3.pattern.Plain(osi, ts0)
    o3.Load(osi, top_node, [100, -500])

    o3.constraints.Transformation(osi)
    o3.test_check.NormDispIncr(osi, tol=1.0e-6, max_iter=35, p_flag=0)
    o3.algorithm.Newton(osi)
    o3.numberer.RCM(osi)
    o3.system.FullGeneral(osi)
    n_steps_gravity = 15
    d_gravity = 1. / n_steps_gravity
    o3.integrator.LoadControl(osi, d_gravity, num_iter=10)
    # o3.rayleigh.Rayleigh(osi, a0, a1, 0.0, 0.0)
    o3.analysis.Static(osi)
    o3r = o3.results.Results2D(cache_path=out_folder, dynamic=True)
    o3r.pseudo_dt = 0.1
    o3r.start_recorders(osi, dt=0.1)
    nr = o3.recorder.NodeToArrayCache(osi, top_node, [o3.cc.DOF2D_X, o3.cc.DOF2D_Y], 'disp')
    er = o3.recorder.ElementToArrayCache(osi, sf_eles[0], arg_vals=['force'])
    for i in range(n_steps_gravity):
        o3.analyze(osi, num_inc=1)
    o3.load_constant(osi, time=0.0)
    import o3seespy.extensions
    o3.extensions.to_py_file(osi, 'ofile.py')
    print('init_disp: ', o3.get_node_disp(osi, top_node, o3.cc.DOF2D_Y))
    print('init_disp: ', o3.get_node_disp(osi, top_nodes[0], o3.cc.DOF2D_Y))
    print('init_disp: ', o3.get_node_disp(osi, top_nodes[-1], o3.cc.DOF2D_Y))
    o3.wipe(osi)
    o3r.save_to_cache()
    # o3r.coords = o3.get_all_node_coords(osi)
    # o3r.ele2node_tags = o3.get_all_ele_node_tags_as_dict(osi)
    data = nr.collect()
    edata = er.collect()
    # bf, sps = plt.subplots(nrows=2)
    # sps[0].plot(data[:, 0])
    # sps[0].plot(data[:, 1])
    # sps[1].plot(edata[:, 0])
    # # sps[0].plot(data[1])
    # plt.show()
    o3plot.replot(o3r)



if __name__ == '__main__':
    name = __file__.replace('.py', '')
    name = name.split("run_")[-1]
    import all_paths as ap

    out_folder = ap.OP_PATH + name + '/'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    run(out_folder)