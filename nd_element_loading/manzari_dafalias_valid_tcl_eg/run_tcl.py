import subprocess
import matplotlib.pyplot as plt
import numpy as np


def run_pm4sand_et(windows=0):

    runfile = 'mz_triaxial_comp.tcl'
    name = __file__.replace('.py', '')
    path = name.split("run_")[0]

    stress_tmpfname = 'stress.out'
    strain_tmpfname = 'strain.out'

    if windows:
        p = subprocess.Popen('%sOpenSees.exe %s%s' % (path, path, runfile), shell=True, stdout=subprocess.PIPE)
    else:
        print('%sOpenSees %s' % (path, runfile))
        p = subprocess.Popen('%sOpenSees %s%s' % (path, path, runfile), shell=True, stdout=subprocess.PIPE)
    stdout, stderr = p.communicate()
    print(p.returncode)  # is 0 if success
    stresses = np.loadtxt(stress_tmpfname)
    stress = stresses
    strain = np.loadtxt(strain_tmpfname)
    # ppt = esig_v0 + stresses[149:, 2] * 1e3
    bf, sps = plt.subplots(nrows=3)
    sps[0].plot(stress[:, 1], label='horizontal')
    sps[0].plot(stress[:, 2], label='vertical')
    sps[0].plot(stress[:, 3], label='shear')
    sps[1].plot(strain[:, 3], stress[:, 3])
    # sps[2].plot(disps[:, 7:10])
    sps[0].set_xlabel('Time [s]')
    sps[0].set_ylabel('Stress [kPa]')
    sps[1].set_xlabel('Strain')
    sps[1].set_ylabel('Stress [kPa]')
    sps[0].legend()
    plt.show()


run_pm4sand_et()

