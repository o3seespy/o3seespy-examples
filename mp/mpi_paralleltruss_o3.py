# Built based off the example from OpenSeesPy Docs
# run with `mpiexec -np 2 python <file-name>`

import o3seespy as o3


def run():
    np = o3.mp.get_np()
    pid = o3.mp.get_pid()
    print(np)
    osi = o3.OpenSeesInstance(ndm=2, ndf=2, mp=True)
    mat = o3.uniaxial_material.Elastic(osi, 3000.0)
    nds = []
    if pid == 0:
        nds.append(o3.node.Node(osi, 0.0, 0.0, tag=0))
        nds.append(None)
        nds.append(None)
        nds.append(o3.node.Node(osi, 72.0, 96.0, tag=3))
        o3.Fix2DOF(osi, nds[0], o3.cc.FIXED, o3.cc.FIXED)
        ele = o3.element.Truss(osi, [nds[0], nds[3]], 10.0, mat)
        ts = o3.time_series.Linear(osi, 1)
        o3.pattern.Plain(osi, ts)
        o3.Load(osi, nds[3], [100.0, -50.0])

    else:
        nds.append(None)
        nds.append(o3.node.Node(osi, 144.0, 0.0, tag=1))
        nds.append(o3.node.Node(osi, 168.0, 0.0, tag=2))
        nds.append(o3.node.Node(osi, 72.0, 96.0, tag=3))
        o3.Fix2DOF(osi, nds[1], o3.cc.FIXED, o3.cc.FIXED)
        o3.Fix2DOF(osi, nds[2], o3.cc.FIXED, o3.cc.FIXED)
        o3.element.Truss(osi, [nds[1], nds[3]], 5.0, mat)
        o3.element.Truss(osi, [nds[2], nds[3]], 5.0, mat)

    o3.domain_change(osi)
    o3.constraints.Transformation(osi)
    o3.test_check.NormDispIncr(osi, tol=1.0e-6, max_iter=10, p_flag=0)
    o3.algorithm.Newton(osi)
    o3.numberer.apply_rcm(osi)
    o3.system.Mumps(osi)
    o3.integrator.LoadControl(osi, 0.1)
    o3.analysis.Static(osi)
    o3.analyze(osi, 10)

    print('Node 4: ', [o3.get_node_coords(osi, nds[3]), o3.get_node_disp(osi, nds[3])])

    o3.load_constant(osi)

    if pid == 0:
        o3.pattern.Plain(osi, ts)
        o3.Load(osi, nds[3], [1.0, 0.0])
    o3.domain_change(osi)
    o3.integrator.ParallelDisplacementControl(osi, nds[3], o3.cc.X, 0.1)
    o3.analyze(osi, 10)

    print('Node 4: ', [o3.get_node_coords(osi, nds[3]), o3.get_node_disp(osi, nds[3])])
    o3.stop(osi)


if __name__ == '__main__':
    run()