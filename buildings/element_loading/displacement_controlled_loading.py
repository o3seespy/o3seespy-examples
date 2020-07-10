import o3seespy as o3
import matplotlib.pyplot as plt


def run():
    l = 100
    osi = o3.OpenSeesInstance(ndm=1, ndf=1)
    nodes = [o3.node.Node(osi, 0.0), o3.node.Node(osi, l)]
    o3.Fix(osi, nodes[0], [o3.cc.X])
    fy = 0.3
    e0 = 200.
    eps_y = fy / e0
    b = -0.01
    mat = o3.uniaxial_material.Steel01(osi, fy=fy, e0=e0, b=b)
    area = 1000.0
    truss = o3.element.Truss(osi, nodes, area, mat=mat)
    ts = o3.time_series.Linear(osi)
    pat = o3.pattern.Plain(osi, ts)
    o3.Load(osi, nodes[1], [1.0])

    step_size = 0.01
    dy = eps_y * l
    max_disp = 5 * dy
    n_steps = int(max_disp / step_size)

    ndr = o3.recorder.NodeToArrayCache(osi, nodes[1], dofs=[o3.cc.X], res_type='disp')
    efr = o3.recorder.ElementToArrayCache(osi, truss, arg_vals=['localForce'])
    o3.constraints.Plain(osi)
    o3.numberer.Plain(osi)
    o3.test.NormDispIncr(osi, tol=1.0e-6, max_iter=100, p_flag=0)
    o3.algorithm.Newton(osi)
    o3.system.BandGeneral(osi)
    o3.integrator.DisplacementControl(osi, nodes[1], o3.cc.X, step_size)
    o3.analysis.Static(osi)

    o3.analyze(osi, n_steps)
    o3.wipe(osi)
    disp = ndr.collect()
    force = efr.collect()
    plt.plot(disp, force)
    plt.show()


if __name__ == '__main__':
    run()
