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
    "kq4.l6b1",
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

GreaterThan = xt.GreaterThanAux
LessThan = xt.LessThanAux
opt2 = lhc.lhcb1.match(
    solve=False,
    ele_start="ip5",
    ele_stop="e.ds.r6.b1",
    twiss_init=xt.TwissInit(betx=0.075, bety=0.18),
    targets=[
        TSet(
            "betx bety alfx alfy mux muy dx dpx".split(), value=t1, at=xt.END, tag="sq"
        ),
        TPhase("mux", 7.44496, "mkd.h5l6.b1", "tclpx.4r5.b1", tag="mkdtct"),
        Target("betx", GreaterThan(430), at="tcdqa.a4r6.b1",tag="tcdq"),
        Target("bety", GreaterThan(145), at="tcdqa.a4r6.b1",tag="tcdq"),
        Target("bety", GreaterThan(170), at="tcdsa.4l6.b1",tag="tcdq"),
        Target("dx", GreaterThan(-0.7), at="mqy.5l6.b1",tag="disp"),
        Target("dx", LessThan(0.7), at="mqy.5l6.b1", tag="disp"),
        Target("dx", GreaterThan(-0.7), at="mqy.4r6.b1", tol=0.7, tag="disp"),
        Target("dx", LessThan(-0.7), at="mqy.4r6.b1", tol=0.7, tag="disp"),
        TPhase("mux", LessThan(   0.25 + 4 / 360.0), "tcsp.a4r6.b1", "mkd.h5l6.b1", tag="mkdtcdq"),
        TPhase("mux", GreaterThan(0.25 - 4 / 360.0), "tcsp.a4r6.b1", "mkd.h5l6.b1", tag="mkdtcdq"),
        TPhase("mux", LessThan(   0.25 + 4 / 360.0), "tcdqa.b4r6.b1", "mkd.h5l6.b1", tag="mkdtcdq"),
        TPhase("mux", GreaterThan(0.25 - 4 / 360.0), "tcdqa.b4r6.b1", "mkd.h5l6.b1", tag="mkdtcdq"),
        TPhase("mux", LessThan(   0.25 + 4 / 360.0), "tcdqa.c4r6.b1", "mkd.h5l6.b1", tag="mkdtcdq"),
        TPhase("mux", GreaterThan(0.25 - 4 / 360.0), "tcdqa.c4r6.b1", "mkd.h5l6.b1", tag="mkdtcdq"),
        TPhase("mux", LessThan(   0.25 + 4 / 360.0), "tcdqa.a4r6.b1", "mkd.h5l6.b1", tag="mkdtcdq"),
        TPhase("mux", GreaterThan(0.25 - 4 / 360.0), "tcdqa.a4r6.b1", "mkd.h5l6.b1", tag="mkdtcdq"),
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

prrrr

for tt in opt.targets:
    if hasattr(tt, "freeze"):
        tt.freeze()

for tt in opt.targets:
    if hasattr(tt, "freeze"):
        tt.unfreeze()

opt.target_status()
opt.disable_targets(tag='mkdtct')
opt.disable_targets(tag='tcdq')
opt.disable_targets(tag='dump')
opt.disable_targets(tag='mkdtcdq')
opt.target_status()
opt.targets[20].active=True

opt.targets[14].active=False
opt.targets[18].active=False
opt.targets[24].active=False


(-48.99239596993709, -51.28015004637074)
opt.targets[20].active=True

for tt in opt.targets:
    tt.zero_on_tol = True

opt.solve();opt.log()