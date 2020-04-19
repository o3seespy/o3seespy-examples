import numpy as np
import sfsimodels as sm
import sfsimodels.num.mesh


def run():
    rho = 1.8
    sl1 = sm.Soil(g_mod=50, unit_dry_weight=rho * 9.8, poissons_ratio=0.3)
    sl2 = sm.Soil(g_mod=100, unit_dry_weight=rho * 9.8, poissons_ratio=0.3)
    sl3 = sm.Soil(g_mod=400, unit_dry_weight=rho * 9.8, poissons_ratio=0.3)
    sl4 = sm.Soil(g_mod=600, unit_dry_weight=rho * 9.8, poissons_ratio=0.3)
    sp = sm.SoilProfile()
    sp.add_layer(0, sl1)
    sp.add_layer(5, sl2)
    sp.add_layer(12, sl3)
    sp.height = 18
    sp.x = 0

    sp.x_angles = [0.0, 0.0, 0.0]

    fd = sm.RaftFoundation()
    fd.width = 4
    fd.height = 1.0
    fd.depth = 0.5
    fd.length = 100
    tds = sm.TwoDSystem(width=10, height=15)
    tds.add_sp(sp, x=0)
    tds.x_surf = np.array([0, 40])
    tds.y_surf = np.array([0, 0])
    bd = sm.NullBuilding()
    bd.set_foundation(fd, x=0)
    tds.add_bd(bd, x=5)

    x_scale_pos = np.array([0, 5, 15, 30])
    x_scale_vals = np.array([2., 1.0, 2.0, 3.0])
    femesh = sm.num.mesh.FiniteElement2DMesh(tds, 0.3, x_scale_pos=x_scale_pos, x_scale_vals=x_scale_vals)

    import sys
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtGui, QtCore
    import o3plot


    win = pg.plot()
    win.setWindowTitle('ECP definition')
    win.setXRange(0, femesh.tds.width)
    win.setYRange(-femesh.tds.height, max(femesh.tds.y_surf))

    o3plot.plot_finite_element_mesh(win, femesh)
    # o3_plot.plot_two_d_system(win, tds)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()


if __name__ == '__main__':
    # plot2()
    run()
    # replot()

