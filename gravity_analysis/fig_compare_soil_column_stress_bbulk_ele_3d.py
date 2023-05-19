import o3seespy as o3
import copy
import sfsimodels as sm
import numpy as np
# import eqsig
# import all_paths as ap
# # for linear analysis comparison
# import liquepy as lq


def site_response(sp, dy=0.5, forder=1.0e3, static=0):
    """
    Run gravity analysis of a soil profile

    Parameters
    ----------
    sp: sfsimodels.SoilProfile object
        A soil profile

    Returns
    -------

    """

    osi = o3.OpenSeesInstance(ndm=3, ndf=3, state=3)
    assert isinstance(sp, sm.SoilProfile)
    sp.gen_split(props=['shear_vel', 'unit_mass', 'g_mod', 'poissons_ratio'], target=dy)
    thicknesses = sp.split["thickness"]
    n_node_rows = len(thicknesses) + 1
    node_depths = np.cumsum(sp.split["thickness"])
    node_depths = np.insert(node_depths, 0, 0)
    ele_depths = (node_depths[1:] + node_depths[:-1]) / 2
    rho = sp.split['unit_mass']
    g_mod = sp.split['g_mod']
    poi = sp.split['poissons_ratio']
    lam = 2 * g_mod * poi / (1 - 2 * poi)
    mu = g_mod
    v_dil = np.sqrt((lam + 2 * mu) / rho)
    ele_h = sp.split['thickness']
    dts = ele_h / v_dil
    min_dt = min(dts)
    print('min_dt: ', min_dt)
    grav = 9.81

    ele_width = min(thicknesses)

    # Define nodes and set boundary conditions for simple shear deformation
    # Start at top and build down
    # [z-oop][y-down][x-horz]
    sn = [
        [[o3.node.Node(osi, 0, 0, 0), o3.node.Node(osi, ele_width, 0, 0)]],
        [[o3.node.Node(osi, 0, 0, 1), o3.node.Node(osi, ele_width, 0, 1)]]
    ]
    zvals = [0, 1]

    for i in range(1, n_node_rows):
        for zz in range(2):
            zval = zvals[zz]
            # Establish left and right nodes
            sn[zz].append([o3.node.Node(osi, 0, -node_depths[i], zval), o3.node.Node(osi, ele_width, -node_depths[i], zval)])
        # set x and y dofs equal for left and right nodes
        o3.EqualDOF(osi, sn[0][i][0], sn[0][i][1], [o3.cc.X, o3.cc.Y, o3.cc.DOF3D_Z])
        o3.EqualDOF(osi, sn[0][i][0], sn[1][i][0], [o3.cc.X, o3.cc.Y, o3.cc.DOF3D_Z])
        o3.EqualDOF(osi, sn[0][i][0], sn[1][i][1], [o3.cc.X, o3.cc.Y, o3.cc.DOF3D_Z])
    # Fix base nodes
    for zz in range(2):
        o3.Fix3DOF(osi, sn[zz][-1][0], o3.cc.FIXED, o3.cc.FIXED, o3.cc.FIXED)
        o3.Fix3DOF(osi, sn[zz][-1][1], o3.cc.FIXED, o3.cc.FIXED, o3.cc.FIXED)

    # define materials
    ele_thick = 1.0  # m
    soil_mats = []
    prev_args = []
    prev_kwargs = {}
    prev_sl_type = None
    eles = []
    for i in range(len(thicknesses)):
        y_depth = ele_depths[i]

        sl_id = sp.get_layer_index_by_depth(y_depth)
        sl = sp.layer(sl_id)
        app2mod = {}
        if y_depth > sp.gwl:
            umass = sl.unit_sat_mass / forder
        else:
            umass = sl.unit_dry_mass / forder
        # Define material
        sl_class = o3.nd_material.ElasticIsotropic
        sl.e_mod = 2 * sl.g_mod * (1 + sl.poissons_ratio) / forder
        app2mod['rho'] = 'unit_moist_mass'
        overrides = {'nu': sl.poissons_ratio, 'unit_moist_mass': umass}

        args, kwargs = o3.extensions.get_o3_kwargs_from_obj(sl, sl_class, custom=app2mod, overrides=overrides)
        changed = 0
        if sl.type != prev_sl_type or len(args) != len(prev_args) or len(kwargs) != len(prev_kwargs):
            changed = 1
        else:
            for j, arg in enumerate(args):
                if not np.isclose(arg, prev_args[j]):
                    changed = 1
            for pm in kwargs:
                if pm not in prev_kwargs or not np.isclose(kwargs[pm], prev_kwargs[pm]):
                    changed = 1

        if changed:
            mat = sl_class(osi, *args, **kwargs)
            prev_sl_type = sl.type
            prev_args = copy.deepcopy(args)
            prev_kwargs = copy.deepcopy(kwargs)

            soil_mats.append(mat)

        # def element
        xx = 0
        yy = i
        zz = 0
        nodes = [sn[zz+1][yy+1][xx], sn[zz+1][yy+1][xx+1],  # left-bot-front -> right-bot-front
                 sn[zz][yy + 1][xx + 1], sn[zz][yy + 1][xx],  # right-bot-back -> left-bot-back
                 sn[zz + 1][yy][xx], sn[zz + 1][yy][xx + 1],  # left-top-front -> right-top-front
                 sn[zz][yy][xx + 1], sn[zz][yy][xx]  # right-top-back -> left-top-back
                 ]
        # eles.append(o3.element.SSPquad(osi, nodes, mat, o3.cc.PLANE_STRAIN, ele_thick, 0.0, -grav))
        eles.append(o3.element.BbarBrick(osi, nodes, mat, 0.0, -grav * mat.den, 0.0))

    # Gravity analysis
    if not static:
        o3.constraints.Transformation(osi)
        o3.test_check.NormDispIncr(osi, tol=1.0e-5, max_iter=30, p_flag=0)
        o3.algorithm.Newton(osi)
        o3.numberer.RCM(osi)
        o3.system.ProfileSPD(osi)
        o3.integrator.Newmark(osi, 5./6, 4./9)  # include numerical damping
        o3.analysis.Transient(osi)
        o3.analyze(osi, 40, 1.)

        # reset time and analysis
        o3.set_time(osi, 0.0)
        o3.wipe_analysis(osi)
    else:
        o3.domain_change(osi)
        o3.constraints.Transformation(osi)
        o3.test_check.NormDispIncr(osi, tol=1.0e-6, max_iter=10, p_flag=0)
        o3.algorithm.Newton(osi)
        o3.numberer.apply_rcm(osi)
        o3.system.Mumps(osi)
        o3.integrator.LoadControl(osi, 0.1)
        o3.analysis.Static(osi)
        o3.analyze(osi, 10)

    sigys = []
    for yy in range(len(eles)):
        ele = eles[yy]
        sigys.append(o3.get_ele_response(osi, ele, 'stresses')[1])

    return np.array(sigys), ele_depths




def run():
    xi = 0.03

    sl = sm.Soil()
    vs = 250.
    unit_mass = 1700.0
    sl.g_mod = vs ** 2 * unit_mass
    sl.poissons_ratio = 0.3
    sl.unit_dry_weight = unit_mass * 9.8
    sl.xi = xi  # for linear analysis

    sl_base = sm.Soil()
    vs = 350.
    unit_mass = 1700.0
    sl_base.g_mod = vs ** 2 * unit_mass
    sl_base.poissons_ratio = 0.3
    sl_base.unit_dry_weight = unit_mass * 9.8
    sl_base.xi = xi  # for linear analysis
    soil_profile = sm.SoilProfile()
    soil_profile.add_layer(0, sl)
    soil_profile.add_layer(10., sl_base)
    soil_profile.height = 20.0
    gm_scale_factor = 1.0

    import matplotlib.pyplot as plt
    from bwplot import cbox
    bf, sps = plt.subplots(ncols=3, figsize=(6, 8))

    # ys = site_response(soil_profile, static=0)
    # sps[0].plot(ys, c=cbox(0))
    forder = 1.0e3
    sigys, ys = site_response(soil_profile, static=1, forder=forder)
    sps[0].plot(-sigys, ys, c=cbox(1))

    sigys_expected = soil_profile.get_v_total_stress_at_depth(ys)
    sps[0].plot(sigys_expected / forder, ys, c=cbox(0), ls='--')

    sps[0].legend(prop={'size': 6})
    name = __file__.replace('.py', '')
    name = name.split("fig_")[-1]
    bf.suptitle(name)
    bf.savefig(f'figs/{name}.png', dpi=90)
    plt.show()


if __name__ == '__main__':
    run()
