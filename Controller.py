#############################################################################################
# DON'T CHANGE ANY OF THE BELOW. NECESSARY FOR CORRECTLY JOINING THE COMPETITION SIMULATION #
#############################################################################################

import os, sys, time, datetime, traceback
import spaceteams as st
def custom_exception_handler(exctype, value, tb):
    error_message = "".join(traceback.format_exception(exctype, value, tb))
    st.logger_fatal(error_message)
sys.excepthook = custom_exception_handler
st.connect_to_sim(sys.argv)

import numpy as np
import numpy.typing as npt


this = st.GetThisSystem()

doSpawning = st.SimGlobals.GetSimEntity().GetParam(st.VarType.bool, "DoSpawning")
if doSpawning:
    while True:
        time.sleep(1.0 / this.GetParam(st.VarType.double, "LoopFreqHz"))

phobos: st.Entity = this.GetParam(st.VarType.entityRef, "Phobos")
phobosFrame = phobos.GetBodyFixedFrame()
shuttle: st.Entity = this.GetParam(st.VarType.entityRef, "Shuttle")
shuttleFrame = shuttle.GetBodyFixedFrame()
target: st.Entity = this.GetParam(st.VarType.entityRef, "Target")

topCamera: st.Entity = this.GetParam(st.VarType.entityRef, "TopCamera")
topCameraFrame = topCamera.GetBodyFixedFrame()
bottomCamera: st.Entity = this.GetParam(st.VarType.entityRef, "BottomCamera")
bottomCameraFrame = bottomCamera.GetBodyFixedFrame()


def GetNearestDebrisLocations(N: int) -> list[npt.NDArray]:
    allDebrisInfo: list[npt.NDArray] = shuttle.GetParamArray(st.VarType.doubleM3x3,
                                                            ["Radar", "SortedDebrisData"])
    nearest = allDebrisInfo[:min(N, len(allDebrisInfo))]
    return nearest


def PanTiltCamera(camera: st.Entity, yaw_deg: float, pitch_deg: float) -> None:
    if camera == topCamera:
        yaw_deg = np.clip(yaw_deg, -89.0, 89.0)
        pitch_deg = np.clip(pitch_deg, -50.0, 0.0)
    elif camera == bottomCamera:
        yaw_deg = np.clip(yaw_deg, -89.0, 89.0)
        pitch_deg = np.clip(pitch_deg, 0.0, 50.0)
    camera.setRotation_DCM(st.math.Euler321_to_DCM([np.deg2rad(yaw_deg), np.deg2rad(pitch_deg), 0.0]), shuttleFrame)


def SetCommand(accelCommand: npt.NDArray, angAccelCommand: npt.NDArray) -> None:
    shuttle.SetParam(st.VarType.doubleV3, ["JetSelection", "CommandedAccel"], accelCommand)
    shuttle.SetParam(st.VarType.doubleV3, ["JetSelection", "CommandedAngAccel"], angAccelCommand)


#############################################################################################
# DON'T CHANGE ANY OF THE ABOVE. NECESSARY FOR CORRECTLY JOINING THE COMPETITION SIMULATION #
#############################################################################################


#########################################################################
# Mission controller for Escape from Phobos (competition version, no measurement modes).
# Point the AXIS body axis at the target, burn along it with a fixed command until the speed toward the target
# reaches CRUISE, coast, remove cross-track drift with the lateral axes, coast in from 300 m, zero command inside
# 50 m (the score is frozen there). No debris avoidance yet.
# Logs every loop to %LOCALAPPDATA%\SimDynamX\STPro\Output\phobos_logs\state_*.csv
#########################################################################

import csv

AXIS = "+z"     # body axis used for the main burn: "+z" (no large turn needed) or "+x" (strongest thrust, needs a 90 deg turn)
CRUISE = 10.0   # cruise speed m/s (scored optimum with the F/75 fuel term)

t0_sim: st.timestamp = st.SimGlobals.GetSimEntity().GetParam(st.VarType.timestamp, ["Time", "StartDateTAI"])
t0: datetime.datetime = t0_sim.as_datetime()
loop_hz: float = this.GetParam(st.VarType.double, "LoopFreqHz")

EX, EY, EZ = np.eye(3)
ZERO = np.zeros(3)


# Command to send for a burn along each body axis (found with the offline simulator, verified in the real simulation
# on Oct 31). The LP that allocates the jets has non-unique optima for pure axis commands; a tiny extra component makes
# it pick a solution with more thrust. Only linear components are perturbed, so no torque is produced.
BURN_CMDS = {
    "+x": (np.array([0.3, 0.0, 0.0]), 0.175),      # the pure command is best
    "-x": (np.array([-0.3, -0.001, 0.0]), 0.128),  # pure -x gives only 0.097
    "+y": (np.array([0.0, 0.3, 0.0]), 0.143),      # the pure command is best
    "-y": (np.array([0.001, -0.3, 0.0]), 0.174),   # pure -y gives only 0.096
    "+z": (np.array([0.001, 0.0, 0.3]), 0.135),    # pure +z gives only 0.096
    "-z": (np.array([0.0, 0.0, -0.3]), 0.117),     # the pure command is best
}
AXIS_INDEX = {"+x": (0, 1), "-x": (0, -1), "+y": (1, 1), "-y": (1, -1), "+z": (2, 1), "-z": (2, -1)}


class TurnAndBurn:
    """Mission controller: fly to the target at a cruise speed with fixed-thrust axis burns.
    - Attitude: keep the main thrust axis (BURN_CMDS[axis], default +z) pointed at the target. Turns use tiny angular
      commands (proportional region, clipped to +-cmd_max); large rate errors (> w_bang) use full angular thrust.
    - Along-track speed: burn the main axis while the speed toward the target is below cruise, coast above it
      (hysteresis dv_on / dv_off).
    - Cross-track velocity (sideways drift from gravity or a debris hit): fire one of the four lateral body axes
      (+x, -x, +y, -y for a +z main axis) with its BURN_CMDS command until the drift is below lat_tol / 3. No turning
      needed, so a hit of 1 m/s costs about 6 s and 4 kg to clean up.
    - Inside no_maneuver_within of the target the shuttle just coasts in; inside keepout (50 m) the score is frozen
      and the command is zero (thrusting on would ram the target ship).
    Lessons from the real simulation that shaped this: axis commands produce no torque; a residual rate after a turn
    drifts the pointing, so turns end only when the rate error is below w_tol and after a short settle; angular commands
    above ~0.003 rad/s^2 are full thrust and give 0.003 rad/s jumps, so near alignment the command stays tiny; a
    velocity-to-be-gained direction becomes ill-defined near cruise speed, so along-track and cross-track are
    controlled separately. Debris avoidance will set lateral_target() instead of zero."""

    def __init__(self, axis="+z", cruise=10.0, gain=36.0, rot_accel=0.05, w_max=0.25, k_lin=1.0, cmd_max=0.002, w_bang=0.05,
                 align_tol=0.015, w_tol=0.001, settle_s=0.3, trim_tol=0.05, lat_tol=0.6, dv_on=1.0, dv_off=0.3,
                 no_maneuver_within=300.0, keepout=50.0, turn_timeout=15.0, gravity=(0.0031, -0.0011, 0.0118)):
        self.axis = axis; self.burn_cmd, self.burn_accel = BURN_CMDS[axis]
        self.cruise = cruise if cruise is not None else 1000.0
        self.gain = gain; self.rot_accel = rot_accel; self.w_max = w_max; self.k_lin = k_lin; self.cmd_max = cmd_max; self.w_bang = w_bang
        self.align_tol = align_tol; self.w_tol = w_tol; self.settle_s = settle_s; self.trim_tol = trim_tol; self.lat_tol = lat_tol
        self.dv_on = dv_on; self.dv_off = dv_off; self.no_maneuver_within = no_maneuver_within; self.keepout = keepout
        self.turn_timeout = turn_timeout
        self.g = np.asarray(gravity, float)
        k, _ = AXIS_INDEX[axis]
        self.lateral_axes = [a for a in BURN_CMDS if AXIS_INDEX[a][0] != k]   # the four axes perpendicular to the main one
        self.state = "turn"; self.settle_until = None; self.turn_since = None; self.turns = 0; self.laterals = 0
        self.angle = 0.0; self.lat_axis = None

    def lateral_target(self, api, u_t, dist):
        """Avoidance goes here later: desired cross-track velocity (Phobos frame). Zero = fly straight at the target."""
        return np.zeros(3)

    def __call__(self, api):
        to_t = api.target - api.loc; dist = np.linalg.norm(to_t); u_t = to_t / dist
        if dist <= self.keepout or self.state == "done":
            self.state = "done"; api.set_command(np.zeros(3), np.zeros(3)); return
        v = api.vel; w = api.ang_vel_body
        v_along = float(v @ u_t); v_perp = v - v_along * u_t
        lat_err = v_perp - self.lateral_target(api, u_t, dist)        # cross-track velocity to remove (Phobos frame)
        k, sign = AXIS_INDEX[self.axis]
        axis_now = sign * api.R[:, k]                                   # main thrust axis in the Phobos frame
        rot = np.cross(axis_now, u_t); s_ = np.linalg.norm(rot)
        angle = float(np.arctan2(s_, axis_now @ u_t)); self.angle = angle
        rot_b = api.phobos_to_body(rot / s_) if s_ > 1e-9 else np.zeros(3)
        near = dist < self.no_maneuver_within

        if self.state not in ("turn", "settle"):
            if near:                                   # close to the target: coast in, no more maneuvers
                self.state = "coast"; api.set_command(np.zeros(3), np.zeros(3)); return
            if angle > self.trim_tol or np.linalg.norm(w) > 0.02:
                self.state = "turn"; self.turn_since = api.t; self.turns += 1
                api.set_command(np.zeros(3), np.zeros(3)); return
            lat = np.linalg.norm(lat_err)
            if self.state == "lateral" and lat > self.lat_tol / 3.0 and self.lat_axis is not None:
                ka, sa = AXIS_INDEX[self.lat_axis]
                if (sa * api.R[:, ka]) @ (-lat_err) > 0.3 * lat:   # the chosen axis still pushes against the drift
                    api.set_command(BURN_CMDS[self.lat_axis][0].copy(), np.zeros(3)); return
            if lat > self.lat_tol:
                best, best_val = None, 0.0
                for a in self.lateral_axes:
                    ka, sa = AXIS_INDEX[a]
                    val = float((sa * api.R[:, ka]) @ (-lat_err)) / lat * BURN_CMDS[a][1]
                    if val > best_val: best, best_val = a, val
                if best is not None:
                    if self.state != "lateral": self.laterals += 1
                    self.state = "lateral"; self.lat_axis = best
                    api.set_command(BURN_CMDS[best][0].copy(), np.zeros(3)); return
            short = self.cruise - v_along
            if self.state == "burn":
                burn = short > self.dv_off
            else:
                burn = short > self.dv_on
            self.state = "burn" if burn else "coast"; self.lat_axis = None
            api.set_command(self.burn_cmd.copy() if burn else np.zeros(3), np.zeros(3)); return

        # attitude control: desired rate = error-reducing rate (sqrt braking curve, linear near zero) + line-of-sight rotation feedforward
        w_ff = api.phobos_to_body(np.cross(u_t, -v_perp / dist))
        if self.state == "settle":
            api.set_command(np.zeros(3), np.zeros(3))
            if api.t >= self.settle_until:
                ok = angle < self.align_tol * 1.5 and np.linalg.norm(w - w_ff) < self.w_tol * 1.5
                self.state = "burn" if ok else "turn"
                if not ok: self.turn_since = api.t
            return
        if self.turn_since is None:
            self.turn_since = api.t
        w_mag = min(np.sqrt(2.0 * self.rot_accel * angle), self.k_lin * angle, self.w_max)
        w_des = w_mag * rot_b + w_ff
        w_err = w_des - w; n_err = np.linalg.norm(w_err)
        if n_err > self.w_bang:                       # large rate error (e.g. after a hit): full angular thrust
            ang_cmd = 0.03 * w_err / n_err
        else:                                         # tiny-command region: proportional
            ang_cmd = np.clip(w_err / self.gain * 5.0, -self.cmd_max, self.cmd_max)
        aligned = angle < self.align_tol and n_err < self.w_tol
        timed_out = api.t - self.turn_since > self.turn_timeout and angle < self.trim_tol
        if aligned or timed_out:
            self.state = "settle"; self.settle_until = api.t + self.settle_s; ang_cmd = np.zeros(3)
        api.set_command(np.zeros(3), ang_cmd)


class RealAPI:
    """Wraps the real simulation state to look like the offline simulator API. Call update() every loop."""

    def __init__(self):
        self.target = np.asarray(target.getLocVelAcc().loc_WRT_ExprIn(phobosFrame), float)
        self.lin_cmd = np.zeros(3); self.ang_cmd = np.zeros(3)

    def update(self, t, loc, vel, bx, by, bz, angvel_phobos):
        self.t = t; self.loc = np.asarray(loc, float); self.vel = np.asarray(vel, float)
        self.R = np.column_stack((bx, by, bz))      # columns = body axes in the Phobos frame (body -> Phobos)
        self.ang_vel_body = self.R.T @ np.asarray(angvel_phobos, float)

    def phobos_to_body(self, v): return self.R.T @ np.asarray(v, float)
    def body_to_phobos(self, v): return self.R @ np.asarray(v, float)
    def set_command(self, lin_body, ang_body):
        self.lin_cmd = np.asarray(lin_body, float).copy(); self.ang_cmd = np.asarray(ang_body, float).copy()


def open_logs():
    for base in (os.environ.get("LOCALAPPDATA", ""), os.path.expanduser("~")):
        try:
            log_dir = os.path.join(base, "SimDynamX", "STPro", "Output", "phobos_logs") if base else "phobos_logs"
            os.makedirs(log_dir, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            return log_dir, stamp
        except OSError:
            continue
    raise OSError("cannot create the log directory")


def vec_or_nan(fn, n):
    try:
        return np.asarray(fn(), dtype=float).ravel()
    except Exception:
        return np.full(n, np.nan)


# ---------------------------------------------------------------------------
log_dir, stamp = open_logs()
state_file = open(os.path.join(log_dir, f"state_{stamp}.csv"), "w", newline="")
state_csv = csv.writer(state_file)
state_csv.writerow(["t", "t_sim", "phase", "loop_dt", "sent", "cmd_ax", "cmd_ay", "cmd_az", "cmd_wx", "cmd_wy", "cmd_wz",
                    "px", "py", "pz", "vx", "vy", "vz", "q0", "q1", "q2", "q3",
                    "bx_x", "bx_y", "bx_z", "by_x", "by_y", "by_z", "bz_x", "bz_y", "bz_z",
                    "wx", "wy", "wz", "fuel_kg", "damage_J", "dist_target_m"])
st.OnScreenLogMessage(f"turnburn: logging to {log_dir}", "MyController", st.Severity.Info)

api = RealAPI()
mission = TurnAndBurn(AXIS, cruise=CRUISE)

t_start = None
t_prev_loop = None
last_phase = ""
last_flush_t = -1e9
iteration = 0
exit_flag: bool = False
while not exit_flag:
    time.sleep(1.0 / loop_hz)

    t_sim = (st.SimGlobals.SimClock.GetTimeNow().as_datetime() - t0).total_seconds()
    if t_start is None:
        t_start = t_sim
    t = t_sim - t_start
    loop_dt = 0.0 if t_prev_loop is None else t - t_prev_loop
    t_prev_loop = t

    state = shuttle.getLocVelAcc()
    loc = state.loc_WRT_ExprIn(phobosFrame)
    vel = state.vel_WRT_ExprIn(phobosFrame)
    to_target = target.getLocVelAcc().loc_WRT_ExprIn(phobosFrame) - loc
    distance = np.linalg.norm(to_target)
    fuel = shuttle.GetParam(st.VarType.double, ["Stats", "FuelUsed"])
    damage = shuttle.GetParam(st.VarType.double, ["Stats", "Damage"])
    angvel = shuttle.getAngVelocity().WRT_ExprIn(phobosFrame)
    quat = vec_or_nan(lambda: shuttle.getRotation().Quat_WRT(phobosFrame), 4)
    bx = vec_or_nan(lambda: st.frames.FramedVector(EX, shuttleFrame).ExprIn(phobosFrame), 3)
    by = vec_or_nan(lambda: st.frames.FramedVector(EY, shuttleFrame).ExprIn(phobosFrame), 3)
    bz = vec_or_nan(lambda: st.frames.FramedVector(EZ, shuttleFrame).ExprIn(phobosFrame), 3)

    api.update(t, loc, vel, bx, by, bz, angvel)
    mission(api)
    lin_cmd, ang_cmd = api.lin_cmd, api.ang_cmd
    phase = mission.state
    SetCommand(lin_cmd, ang_cmd)
    if iteration % 60 == 0 or phase != last_phase:
        last_phase = phase
        st.OnScreenLogMessage(f"{phase}  dist {distance:.0f} m  speed {np.linalg.norm(vel):.1f} m/s  fuel {fuel:.1f} kg  "
                              f"point err {np.degrees(mission.angle):.2f} deg  turns {mission.turns}  laterals {mission.laterals}",
                              "MyController", st.Severity.Info)

    try:
        state_csv.writerow([f"{t:.4f}", f"{t_sim:.4f}", phase, f"{loop_dt:.4f}", 1, *lin_cmd, *ang_cmd, *loc, *vel, *quat,
                            *bx, *by, *bz, *angvel, fuel, damage, distance])
        if t - last_flush_t >= 1.0:
            last_flush_t = t
            state_file.flush()
    except Exception as e:
        st.OnScreenLogMessage(f"logging error: {e}", "MyController", st.Severity.Warning)

    iteration += 1

st.leave_sim()
