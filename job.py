import time
from functools import partial

import xtrack as xt
import xdeps as xd
from var_limits import set_var_limits_and_steps
import numpy as np


lhc = xt.Multiline.from_json("hllhc16.json")

set_var_limits_and_steps(lhc)

tw = lhc.twiss()

tw.lhcb1.rows["ip.*"].cols["betx bety"]

t1 = lhc.lhcb1.twiss(ele_start="ip5", ele_stop="e.ds.r6.b1", betx=0.075, bety=0.18)
t2 = lhc.lhcb2.twiss(ele_start="ip5", ele_stop="e.ds.r6.b2", betx=0.075, bety=0.18)

# lhc.vars.load_madx_optics_file("acc-models-lhc/strengths/round/opt_round_150_1500.madx")

# lhc.vars.load_madx_optics_file("acc-models-lhc/strengths/flat/opt_flathv_75_180_1500.madx")


def get_phase(lhc):
    tw = lhc.twiss()
    twiss_init = tw.lhcb1.get_twiss_init("mkd.h5l6.b1")
    twiss_init.mux = 0
    mux = lhc.lhcb1.twiss(
        ele_start="mkd.h5l6.b1", ele_stop="tctpxh.4l5.b1", twiss_init=twiss_init
    ).mux[-1]
    dmux1 = (mux * 2 - np.round(mux * 2)) * 180

    twiss_init = tw.lhcb2.get_twiss_init("mkd.h5r6.b2").reverse()
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


vir5rb1 = [
    "kqt13.r5b1",
    "kq8.r5b1",
    "kq7.r5b1",
    "kq5.r5b1",
    "kqtl11.r5b1",
    "kq10.r5b1",
    "kq9.r5b1",
    "kq6.r5b1",
    "kqt12.r5b1",
    "kq4.r5b1",
]

vir5rb2 = [
    "kq6.r5b2",
    "kq10.r5b2",
    "kq4.r5b2",
    "kq7.r5b2",
    "kq5.r5b2",
    "kq9.r5b2",
    "kqtl11.r5b2",
    "kqt13.r5b2",
    "kqt12.r5b2",
    "kq8.r5b2",
]

vir6b1 = [
    "kqtl11.r6b1",
    "kqt13.r6b1",
    "kq9.r6b1",
    "kq5.r6b1",
    "kqt12.r6b1",
    # "kq4.l6b1",
    "kqt13.l6b1",
    "kq8.r6b1",
    "kq10.r6b1",
    "kqtl11.l6b1",
    "kqt12.l6b1",
    "kq8.l6b1",
    "kq10.l6b1",
    "kq5.l6b1",
    "kq9.l6b1",
    "kq4.r6b1",
]

vir6b2 = [
    "kq10.l6b2",
    "kq8.r6b2",
    "kq8.l6b2",
    "kqtl11.l6b2",
    "kq9.l6b2",
    "kqt13.l6b2",
    "kqt12.r6b2",
    "kqt13.r6b2",
    "kq5.l6b2",
    "kq9.r6b2",
    "kqt12.l6b2",
    "kq4.l6b2",
    "kq4.r6b2",
    "kq5.r6b2",
    "kqtl11.r6b2",
    "kq10.r6b2",
]


opt1 = lhc.lhcb1.match(
    solve=False,
    ele_start="ip5",
    ele_stop="e.ds.r5.b1",
    default_tol={None: 1e-8, "betx": 1e-6, "bety": 1e-6},
    twiss_init=xt.TwissInit(betx=0.5, bety=0.5),
    targets=[
        xt.TargetSet(
            "betx bety alfx alfy mux muy".split(),
            value="preserve",
            at=xt.END,
            tag="pre",
        ),
    ],
    vary=[],
)

TPhase = xt.TargetRelPhaseAdvance
TIneq = xt.TargetInequality
TSet = xt.TargetSet
Target = xt.Target


def bxdump(tw):
    ld = 761
    bx = tw["betx", "ip6"]
    ax = tw["alfx", "ip6"]
    bxdump = bx - 2 * ld * ax + ld**2 * (1 + ax**2) / bx
    return bxdump


def bydump(tw):
    ld = 761
    by = tw["bety", "ip6"]
    ay = tw["alfy", "ip6"]
    bydump = by - 2 * ld * ay + ld**2 * (1 + ay**2) / by
    return bydump


def bdump(tw):
    return np.sqrt(bxdump(tw) * bydump(tw))

GreaterThan = xt.GreaterThan
LessThan = xt.LessThan
Range = xt.Range

# GreaterThan = partial(xt.GreaterThan, mode='smooth', sigma_rel=0.001)
# LessThan = partial(xt.LessThan, mode='smooth', sigma_rel=0.001)
# GreaterThan = xt.GreaterThanAux
# LessThan = xt.LessThanAux

opt2 = lhc.lhcb1.match(
    solve=False,
    ele_start="ip5",
    ele_stop="e.ds.r6.b1",
    default_tol={None: 1e-8, "betx": 1e-6, "bety": 1e-6},
    twiss_init=xt.TwissInit(betx=0.075, bety=0.18),
    targets=[
        TSet("betx bety alfx alfy mux muy dx dpx".split(), value=t1, at=xt.END, tag="sq"),
        TPhase("mux", 7.44496, ele_stop="mkd.h5l6.b1", ele_start="tclpx.4r5.b1", tag="mkdtct"),
        Target("betx", GreaterThan(430, mode='smooth', sigma_rel=0.01), at="tcdqa.a4r6.b1",tag="tcdq"),
        Target("bety", GreaterThan(145, mode='smooth', sigma_rel=0.01), at="tcdqa.a4r6.b1",tag="tcdq"),
        Target("bety", GreaterThan(170, mode='smooth', sigma_rel=0.01), at="tcdsa.4l6.b1",tag="tcdq"),
        Target("dx",  Range(-0.7, 0.7, mode='smooth', sigma_rel=0.01), at="mqy.5l6.b1",tag="disp"),
        Target("dx",  Range(-0.7, 0.7, mode='smooth', sigma_rel=0.01), at="mqy.4r6.b1", tag="disp"),
        # TPhase("mux", Range(0.25 - 4 / 360.0, 0.25 + 4 / 360.0, mode='smooth', sigma_rel=0.01),
        #        ele_stop="tcsp.a4r6.b1", ele_start="mkd.h5l6.b1", tag="mkdtcdq"),
        TPhase("mux", GreaterThan(0.25 - 4 / 360.0, mode='smooth', sigma_rel=0.01),
                ele_stop="tcsp.a4r6.b1", ele_start="mkd.h5l6.b1", tag="mkdtcdq"),
        TPhase("mux", LessThan(0.25 + 4 / 360.0, mode='smooth', sigma_rel=0.01),
                ele_stop="tcsp.a4r6.b1", ele_start="mkd.h5l6.b1", tag="mkdtcdq"),
        # TPhase("mux", Range(0.25 - 4 / 360.0, 0.25 + 4 / 360.0, mode='smooth', sigma_rel=0.01),
        #        ele_stop="tcdqa.a4r6.b1", ele_start="mkd.h5l6.b1", tag="mkdtcdq"),
        TPhase("mux", GreaterThan(0.25 - 4 / 360.0, mode='smooth', sigma_rel=0.01),
                ele_stop="tcdqa.a4r6.b1", ele_start="mkd.h5l6.b1", tag="mkdtcdq"),
        TPhase("mux", LessThan(0.25 + 4 / 360.0, mode='smooth', sigma_rel=0.01),
                ele_stop="tcdqa.a4r6.b1", ele_start="mkd.h5l6.b1", tag="mkdtcdq"),
        # TPhase("mux", Range(0.25 - 4 / 360.0, 0.25 + 4 / 360.0, mode='smooth', sigma_rel=0.01),
        #        ele_stop="tcdqa.b4r6.b1", ele_start="mkd.h5l6.b1", tag="mkdtcdq"),
        TPhase("mux", GreaterThan(0.25 - 4 / 360.0, mode='smooth', sigma_rel=0.01),
                ele_stop="tcdqa.b4r6.b1", ele_start="mkd.h5l6.b1", tag="mkdtcdq"),
        TPhase("mux", LessThan(0.25 + 4 / 360.0, mode='smooth', sigma_rel=0.01),
                ele_stop="tcdqa.b4r6.b1", ele_start="mkd.h5l6.b1", tag="mkdtcdq"),
        # TPhase("mux", Range(0.25 - 4 / 360.0, 0.25 + 4 / 360.0, mode='smooth', sigma_rel=0.01),
        #        ele_stop="tcdqa.c4r6.b1", ele_start="mkd.h5l6.b1", tag="mkdtcdq"),
        TPhase("mux", GreaterThan(0.25 - 4 / 360.0, mode='smooth', sigma_rel=0.01),
                ele_stop="tcdqa.c4r6.b1", ele_start="mkd.h5l6.b1", tag="mkdtcdq"),
        TPhase("mux", LessThan(0.25 + 4 / 360.0, mode='smooth', sigma_rel=0.01),
                ele_stop="tcdqa.c4r6.b1", ele_start="mkd.h5l6.b1", tag="mkdtcdq"),
        Target(bdump, GreaterThan(4500, mode='smooth', sigma_rel=0.01), tag="dump"),
        Target(bxdump,GreaterThan(4000, mode='smooth', sigma_rel=0.01), tag="dump"),
        Target(bydump,GreaterThan(3200, mode='smooth', sigma_rel=0.01), tag="dump"),
    ],
    vary=[],
)

opt = lhc.lhcb1.match(
    solve=False,assert_within_tol=False,
    targets=opt1.targets + opt2.targets,
    vary=xt.VaryList(vir5rb1 + vir6b1),
)


kvals0 = opt.get_knob_values(0)
for vv in opt.vary:
    if vv.name in kvals0:
        step_guess = kvals0[vv.name] * 0.01
        if vv.step < step_guess:
            vv.step = step_guess


degx, degy = get_phase(lhc)

t1 = time.time()
while degx < -20:
    opt.targets[14].value -= 0.002; opt.step(20); degx, degy = get_phase(lhc)
    print(f'phix = {degx:.2f} deg, penalty = {opt.log().penalty[-1]}')
t2 = time.time()

print('Refining solution')
t3 = time.time()
pen = opt.log().penalty[-1]
while pen>1e-9:
    opt.step(20); degx, degy = get_phase(lhc)
    pen = opt.log().penalty[-1]
    tol_met = opt.log().tol_met[-1]
    print(f'phix = {degx:.2f} deg, penalty = {pen}')
    if np.isnan(pen):
        break

t4 = time.time()
print(f'Initial solution took {t2-t1:.2f} s')
print(f'Refining solution took {t4-t3:.2f} s')

knobs_initial = opt.get_knob_values(0)
knobs_optimized = opt.get_knob_values()
lhc.metadata["knobs_optimized"] = knobs_optimized
lhc.metadata["knobs_initial"] = knobs_initial
lhc.to_json("hllhc_optimized_mkdtct.json")

lhc.vars.update(knobs_initial)
tw0 = lhc.lhcb1.twiss()

lhc.vars.update(knobs_optimized)
tw1 = lhc.lhcb1.twiss()

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

plt.figure(101)
x = np.linspace(-20, 20, 10000)
plt.plot(x, [opt.targets[18].value.auxtarget(xx) for xx in x])

plt.figure(102)
x = np.linspace(-1000, 1000, 10000)
plt.plot(x, [opt.targets[15].value.auxtarget(xx) for xx in x])




plt.show()

