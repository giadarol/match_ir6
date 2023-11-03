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

GreaterThan = partial(xt.GreaterThan, mode='smooth')
LessThan = partial(xt.LessThan, mode='smooth')

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
        Target("betx", GreaterThan(430), at="tcdqa.a4r6.b1", tag="tcdq"),
        Target("bety", GreaterThan(145), at="tcdqa.a4r6.b1", tag="tcdq"),
        Target("bety", GreaterThan(170), at="tcdsa.4l6.b1",  tag="tcdq"),
        Target("dx",   GreaterThan(-0.7),  at="mqy.5l6.b1",  tag="disp"),
        Target("dx",   LessThan(    0.7),  at="mqy.5l6.b1",  tag="disp"),
        Target("dx",   GreaterThan(-0.7),  at="mqy.4r6.b1",  tag="disp"),
        Target("dx",   LessThan(    0.7),  at="mqy.4r6.b1",  tag="disp"),
        TPhase("mux",  GreaterThan(0.25 - 4 / 360.0), ele_stop="tcsp.a4r6.b1",  ele_start="mkd.h5l6.b1", tag="mkdtcdq"),
        TPhase("mux",  LessThan(   0.25 + 4 / 360.0), ele_stop="tcsp.a4r6.b1",  ele_start="mkd.h5l6.b1", tag="mkdtcdq"),
        TPhase("mux",  GreaterThan(0.25 - 4 / 360.0), ele_stop="tcdqa.a4r6.b1", ele_start="mkd.h5l6.b1", tag="mkdtcdq"),
        TPhase("mux",  LessThan(   0.25 + 4 / 360.0), ele_stop="tcdqa.a4r6.b1", ele_start="mkd.h5l6.b1", tag="mkdtcdq"),
        TPhase("mux",  GreaterThan(0.25 - 4 / 360.0), ele_stop="tcdqa.b4r6.b1", ele_start="mkd.h5l6.b1", tag="mkdtcdq"),
        TPhase("mux",  LessThan(   0.25 + 4 / 360.0), ele_stop="tcdqa.b4r6.b1", ele_start="mkd.h5l6.b1", tag="mkdtcdq"),
        TPhase("mux",  GreaterThan(0.25 - 4 / 360.0), ele_stop="tcdqa.c4r6.b1", ele_start="mkd.h5l6.b1", tag="mkdtcdq"),
        TPhase("mux",  LessThan(   0.25 + 4 / 360.0), ele_stop="tcdqa.c4r6.b1", ele_start="mkd.h5l6.b1", tag="mkdtcdq"),
        Target(bdump, GreaterThan(4500), tag="dump"),
        Target(bxdump,GreaterThan(4000), tag="dump"),
        Target(bydump,GreaterThan(3200), tag="dump"),
    ],
    vary=[],
)

opt = lhc.lhcb1.match(
    solve=False,assert_within_tol=False,
    targets=opt1.targets + opt2.targets,
    vary=xt.VaryList(vir5rb1 + vir6b1),
)


# Reproduce issue
opt.targets[14].value = 7.362960000000009
lhc.vars.update({'kqt13.r5b1': -0.0022877224551007738,
 'kq8.r5b1': -0.007138655310595913,
 'kq7.r5b1': 0.008493633025800633,
 'kq5.r5b1': 0.0009858720588893752,
 'kqtl11.r5b1': -0.0010331018026608473,
 'kq10.r5b1': -0.007397085034797899,
 'kq9.r5b1': 0.006703539090152135,
 'kq6.r5b1': -0.0024068911419632827,
 'kqt12.r5b1': -0.004488122378270206,
 'kq4.r5b1': -0.000942533595881166,
 'kqtl11.r6b1': 0.002577921086915818,
 'kqt13.r6b1': 0.0055704986478204935,
 'kq9.r6b1': -0.006665349152169336,
 'kq5.r6b1': -0.006595215337611925,
 'kqt12.r6b1': 0.0009355611978655791,
 'kqt13.l6b1': -0.0023683793151235083,
 'kq8.r6b1': 0.00916371233983537,
 'kq10.r6b1': 0.006998652558807122,
 'kqtl11.l6b1': -0.001992994717659459,
 'kqt12.l6b1': -0.005710414671951595,
 'kq8.l6b1': -0.007149828831124582,
 'kq10.l6b1': -0.007560215100255281,
 'kq5.l6b1': 0.007613380060698968,
 'kq9.l6b1': 0.0066723409649584045,
 'kq4.r6b1': 0.0056397007801390715})
opt._add_point_to_log()
opt.solver.max_rel_penaly_increase = np.inf
opt.step(1)


prrrrr

for vv in opt.vary:
    new_val = lhc.vv[vv.name] + vv.step
    if new_val < vv.limits[1]:
        lhc.vv[vv.name] = new_val

upper_0 = opt.targets[-8].value.upper
jac_zero = opt.solver._last_jac[-8, :].copy()

# incriminated jacobian
opt._err.get_jacobian(opt.solver._last_jac_x)[-8, :]

du_vect = np.linspace(-1e-2, 1e-2, 101)
jac_lines = np.zeros((len(du_vect), len(opt.solver._last_jac[-8, :])))
for ii, du in enumerate(du_vect):
    print(ii)
    opt.targets[-8].value.upper = upper_0 + du
    jac_lines[ii, :] = opt._err.get_jacobian(opt.solver._last_jac_x)[-8, :]
    
prrrr

du_vect = np.logspace(-20, -3, 100)

jac_lines = np.zeros((len(du_vect), len(opt.solver._last_jac[-8, :])))
for ii, du in enumerate(du_vect):
    print(ii)
    opt.reload(0)
    opt.targets[-8].value.upper = upper_0 + du
    try:
        opt.step(1)
    except ValueError as err:
        print(err)

    jac_lines[ii, :] = opt.solver._last_jac[-8, :]


# This crushes
opt.targets[-4].value.upper *= 1.01
opt.targets[-6].value.upper *= 1.01
opt.targets[-8].value.upper *= 1.01

opt.step()


# opt.targets[-4].freeze()
# opt.targets[-6].freeze()
# opt.targets[-8].freeze()

# opt.step(20)



