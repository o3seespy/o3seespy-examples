import o3soil.sra

import sfsimodels as sm

import numpy as np
import eqsig

import all_paths as ap
import liquepy as lq
import pysra

import json


def run():

    vs = 250.
    sp = sm.SoilProfile()
    sp.height = 30.
    sl = sm.Soil()

    sl.type = 'elastic'
    sl.poissons_ratio = 0.
    sl.unit_dry_weight = 2.202e3 * 9.8
    sl.g_mod = vs ** 2 * sl.unit_dry_mass
    sl.xi = 0.01  # for linear analysis
    sp.add_layer(0, sl)

    rock = sm.Soil()
    vs = 450.
    unit_mass = 1700.0
    rock.g_mod = vs ** 2 * unit_mass
    rock.unit_dry_weight = unit_mass * 9.8
    rock.poissons_ratio = 0.
    rock.xi = 0.01
    sp.add_layer(sp.height, rock)
    sp.height = sp.height + 1

    in_sig = eqsig.load_asig(ap.MODULE_DATA_PATH + 'gms/short_motion_dt0p01.txt', m=0.05)

    opfile = __file__[:-3] + 'op.py'
    fixed_base = 0
    if fixed_base:
        base_imp = -1
        wave_field = 'within'
    else:
        base_imp = 0
        wave_field = 'outcrop'

    outputs = o3soil.sra.site_response(sp, in_sig, outs={'ACCX': 'all'}, dy=0.5, opfile=opfile,
                                       base_imp=base_imp, freqs=(0.5, 20), xi=0.02, analysis_dt=0.005)
    resp_dt = outputs['time'][2] - outputs['time'][1]
    o3_surf_sig = eqsig.AccSignal(outputs['ACCX'][0], resp_dt)

    pysra_outs_full = lq.sra.run_pysra(sp, in_sig, [0], wave_field=wave_field)
    pysra_surf_sig = eqsig.AccSignal(pysra_outs_full['ACCX'][0], in_sig.dt)

    show = 1

    if show:
        import matplotlib.pyplot as plt
        from bwplot import cbox

        bf, sps = plt.subplots(nrows=3)

        sps[0].plot(in_sig.time, in_sig.values, c='k', label='Input', lw=1)
        sps[0].plot(o3_surf_sig.time, o3_surf_sig.values, c=cbox(0), label='o3-surf-direct', lw=1)
        sps[0].plot(pysra_surf_sig.time, pysra_surf_sig.values, c=cbox(4), label='pysra-surf-direct', lw=1, ls='--')

        sps[1].loglog(o3_surf_sig.fa_frequencies, abs(o3_surf_sig.fa_spectrum), c=cbox(0), lw=1)

        sps[1].loglog(pysra_surf_sig.fa_frequencies, abs(pysra_surf_sig.fa_spectrum), c=cbox(4), lw=1)
        in_sig.smooth_fa_frequencies = in_sig.fa_frequencies
        pysra_surf_sig.smooth_fa_frequencies = in_sig.fa_frequencies
        o3_surf_sig.smooth_fa_frequencies = in_sig.fa_frequencies
        in_sig.generate_smooth_fa_spectrum(band=80)
        pysra_surf_sig.generate_smooth_fa_spectrum(band=80)
        o3_surf_sig.generate_smooth_fa_spectrum(band=80)

        sps[2].plot(in_sig.smooth_fa_frequencies, o3_surf_sig.smooth_fa_spectrum / in_sig.smooth_fa_spectrum,
                    c=cbox(0), label='O3 - smoothed')
        sps[2].plot(in_sig.smooth_fa_frequencies, pysra_surf_sig.smooth_fa_spectrum / in_sig.smooth_fa_spectrum,
                    c=cbox(4), label='Pysra - smoothed')
        pysra_sp = lq.sra.sm_profile_to_pysra(sp)
        freqs, tfs = lq.sra.calc_pysra_tf(pysra_sp, in_sig.fa_frequencies, absolute=True)
        sps[2].plot(freqs, tfs,  c='k', label='Pysra - exact')
        sps[0].legend(loc='lower right')

        vs_damped = lq.sra.theory.calc_damped_vs_dormieux_1990(sl.get_shear_vel(saturated=False), sl.xi)
        vs_br_damped = lq.sra.theory.calc_damped_vs_dormieux_1990(rock.get_shear_vel(saturated=False), rock.xi)
        h = sp.height
        omega = in_sig.fa_frequencies * np.pi * 2
        imp = lq.sra.theory.calc_impedance(sl.unit_dry_mass, vs_damped, rock.unit_dry_mass, vs_br_damped)
        tfs = lq.sra.theory.calc_tf_elastic_br(h, vs_damped, omega, imp, absolute=True)
        sps[2].semilogx(in_sig.fa_frequencies, tfs, label='Theoretical Elastic BR', c='r', ls='--')

        sps[2].set_xlim([0.1, 30])
        sps[0].set_ylabel('Acc [m/s2]')
        sps[1].set_ylabel('FAS')
        sps[2].set_ylabel('H')
        sps[0].set_xlabel('Time [s]')
        sps[1].set_xlabel('Freq [Hz]')
        sps[2].set_xlabel('Freq [Hz]')
        sps[2].legend()
        plt.show()


if __name__ == '__main__':
    run()

