import all_paths as ap
import eqsig
import matplotlib.pyplot as plt

record_filename = 'short_motion_dt0p01.txt'
in_sig = eqsig.load_asig(ap.MODULE_DATA_PATH + 'gms/' + record_filename, m=1.0)

bf, ax = plt.subplots(nrows=3)
ax[0].plot(in_sig.time, in_sig.values)
ax[1].plot(in_sig.time, in_sig.velocity)
ax[2].plot(in_sig.time, in_sig.displacement)
print(in_sig.velocity[-1])
plt.show()

