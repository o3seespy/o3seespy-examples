import o3seespy as o3


def run():
    osi = o3.OpenSeesInstance(ndm=1, ndf=1)
    mat = o3.uniaxial_material.Elastic(osi, 3000.0)

    n1 = o3.node.Node(osi, 0.0)
    n2 = o3.node.Node(osi, 72.0)

    o3.Fix1DOF(osi, n1, o3.cc.FIXED)

    o3.element.Truss(osi, [n1, n2], 10.0, mat)
    ts0 = o3.time_series.Linear(osi)
    o3.pattern.Plain(osi, ts0)
    o3.Load(osi, n1, [10])

    o3.constraints.Transformation(osi)
    o3.test_check.NormDispIncr(osi, tol=1.0e-6, max_iter=10)
    o3.numberer.RCM(osi)
    o3.system.ProfileSPD(osi)
    o3.algorithm.NewtonLineSearch(osi, 0.75)
    o3.integrator.Newmark(osi, 0.5, 0.25)

    o3.analysis.VariableTransient(osi)

    o3.analyze(osi, 5, 0.0001, 0.00001, 0.001, 10)
    time = o3.get_time(osi)
    print(f'time: ', o3.get_time(osi))
    approx_vtime = 0.0001 + 0.001  # One step at target, then one step at maximum
    assert 0.99 < time / approx_vtime < 1.01,  (time,  approx_vtime)
    o3.set_time(osi, 0.0)
    # Can still run a non-variable analysis - since analyze function has multiple dispatch.
    o3.analyze(osi, 5, 0.0001)
    time = o3.get_time(osi)
    print(f'time: ', o3.get_time(osi))
    approx_vtime = 0.0001 * 5  # variable transient is not active so time should be dt * 5
    # If variable transient is not active then time would be 0.0005
    assert 0.99 < time / approx_vtime < 1.01,  (time,  approx_vtime)

if __name__ == '__main__':
    run()
