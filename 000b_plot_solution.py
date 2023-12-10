import time
from functools import partial

import xtrack as xt
import xdeps as xd
from var_limits import set_var_limits_and_steps
import numpy as np

def get_phase(lhc):

    tw = lhc.twiss()

    # # Original (to be re-enabled when loop-around is deployed)
    # twiss_init = tw.lhcb1.get_twiss_init("mkd.h5l6.b1")
    # twiss_init.mux = 0
    # mux = lhc.lhcb1.twiss(
    #     ele_start="mkd.h5l6.b1", ele_stop="tctpxh.4l5.b1", twiss_init=twiss_init
    # ).mux[-1]
    # dmux1 = (mux * 2 - np.round(mux * 2)) * 180

    # Temporary (to be removed when loop-around is deployed)
    twiss_init = tw.lhcb1.get_twiss_init("tctpxh.4l5.b1")
    twiss_init.mux = 0
    mux = tw.lhcb1.qx - lhc.lhcb1.twiss(
        ele_start="tctpxh.4l5.b1", ele_stop="mkd.h5l6.b1", twiss_init=twiss_init
    ).mux[-1]
    dmux1 = (mux * 2 - np.round(mux * 2)) * 180

    twiss_init = tw.lhcb2.get_twiss_init("mkd.h5r6.b2").reverse() # To go to b4
    twiss_init.mux = 0
    mux = lhc.lhcb2.twiss(
        ele_start="mkd.h5r6.b2",
        ele_stop="tctpxh.4r5.b2",
        reverse=False,
        twiss_init=twiss_init,
    ).mux[-1]
    dmux2 = (mux * 2 - np.round(mux * 2)) * 180
    print()

    return dmux1, dmux2


lhc = xt.Multiline.from_json("hllhc_optimized_mkdtct.json")

knobs_optimized = lhc.metadata["knobs_optimized"]
knobs_initial = lhc.metadata["knobs_initial"]

lhc.vars.update(knobs_initial)
tw0 = lhc.lhcb1.twiss()

lhc.vars.update(knobs_optimized)
tw1 = lhc.lhcb1.twiss()
degx, degy = get_phase(lhc)

s_elem = lhc.lhcb1.get_s_position()
k1 = lhc.lhcb1.attr["k1"]
l = lhc.lhcb1.attr["length"]

import matplotlib.pyplot as plt
plt.close("all")


nemitt_x = 2.5e-6
gemitt_x = nemitt_x / tw0.beta0 / tw0.gamma0
dp_p = 2e-4

plt.figure(2, figsize=(6.4, 4.8*1.5))
ax1 = plt.subplot(5, 1, 1)
mask= np.abs(k1*l) > 0
ax1.bar(np.array(s_elem)[mask], (k1*l)[mask], width=l[mask], align='edge')
ax1.axhline(0, color='k')
ax1.set_ylabel('k1l')
plt.legend()
ax2 = plt.subplot(5, 1, 2, sharex=ax1)
ax2.plot(tw0.s, tw0.betx, label='initial')
ax2.plot(tw1.s, tw1.betx, label='optimized')
ax2.set_ylabel(r"$\beta_x$ [m]")
ax3 = plt.subplot(5, 1, 3, sharex=ax1, sharey=ax2)
ax3.plot(tw0.s, tw0.bety, label='initial')
ax3.plot(tw1.s, tw1.bety, label='optimized')
ax3.set_ylabel(r"$\beta_y$ [m]")
ax4 = plt.subplot(5, 1, 4, sharex=ax1)
ax4.plot(tw0.s, tw0.dx, label='initial')
ax4.plot(tw1.s, tw1.dx, label='optimized')
ax4.set_ylabel(r"$D_x$ [m]")
ax5 = plt.subplot(5, 1, 5, sharex=ax1)
plt.plot(tw0.s, np.sqrt(tw0.betx * gemitt_x + (tw0.dx * dp_p)**2), label='initial')
plt.plot(tw1.s, np.sqrt(tw1.betx * gemitt_x + (tw1.dx * dp_p)**2), label='optimized')
ax5.set_ylabel(r"$\sigma_x$ [m]")
plt.suptitle(f"Emittance = {nemitt_x*1e6:.2f} um - RMS momentum spread = {dp_p:.2e}"
             f'\n MKD-TCT phase = {degx:.2f} deg')

for ax in [ax2, ax3, ax4, ax5]:
    ax.grid(True)
ax2.legend()

plt.show()

