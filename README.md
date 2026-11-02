# Escape from Phobos — Controller

Our entry for the Fall 2026 Space Teams University competition
([rules and info](https://github.com/SimDynamX/SpaceTeamsPRO/discussions/218)).
The whole solution is one file, `Controller.py`, which replaces the starter `Controller.py`
in [STU4-Solution](https://github.com/SimDynamX/STU4-Solution). Nothing in the simulation
(`STU4-Sim`) is modified.

## What the controller does

The shuttle has 38 fixed jets and the simulation allocates them from two commands we send every
loop: a linear acceleration and an angular acceleration, both in the body frame. The controller
is a small state machine that runs at the controller's loop rate (about 57 Hz):

1. **turn** — point the body +z axis at the target ship. Attitude is controlled with small angular
   commands; the turn ends when the pointing error is below 0.015 rad and the body rate below
   0.001 rad/s, followed by a 0.3 s settle so the burn starts with no residual rotation.
2. **burn** — fire along +z with a fixed command until the speed toward the target reaches the
   cruise speed (10 m/s). Axis-aligned commands produce no torque, so no attitude control is
   needed during the burn.
3. **coast** — zero command. If gravity brings the speed more than 1 m/s below cruise, burn again;
   if the pointing drifts more than 0.05 rad, pause and re-point.
4. **lateral** — cross-track velocity (from gravity or a debris hit) above 0.6 m/s is removed with
   one of the four side axes, picked by how well it opposes the drift, without turning the ship.
5. From 300 m out the shuttle coasts in; inside the 50 m keep-out sphere the score is frozen and the
   command is set to zero.

If a collision leaves the shuttle spinning, angular rates above 0.05 rad/s are removed with full
angular thrust before the small-command pointing control takes over again.

The burn commands (`BURN_CMDS` in the file) are axis vectors with a small extra component. The jet
allocation in the simulation is a linear program whose optimum is not unique for a pure axis
command; the extra component steers it to the solution with the most thrust (for example 0.135 m/s²
along +z instead of 0.096 m/s²), at the same fuel per m/s.

Debris avoidance is not implemented in this version: the shuttle flies a straight line and relies on
the recovery logic if it is hit.

## Settings

At the top of the user section of `Controller.py`:

- `AXIS` — body axis used for the main burn, `"+z"` (default, the shuttle starts pointing at the
  target) or `"+x"` (strongest thrust, needs a 90° turn first).
- `CRUISE` — cruise speed in m/s, default 10.

## Running

1. Install Space Teams PRO and clone `STU4-Sim` and `STU4-Solution` as described in the
   [STU4-Solution wiki](https://github.com/SimDynamX/STU4-Solution/wiki).
2. Copy this `Controller.py` over `STU4-Solution/Controller.py`.
3. Run `COMPETITION_SIM`. The controller prints its state, distance, speed, fuel and pointing error
   to the on-screen log, and writes a per-loop state log to
   `%LOCALAPPDATA%\SimDynamX\STPro\Output\phobos_logs\state_<timestamp>.csv`.

Dependencies: only `numpy`, which the Space Teams PRO Python already has.

## Results

Qualification run on October 31, 2026 (previous version of the controller): reached the target in
184.4 s using 49.3 kg of fuel with no damage.

## Files

- `Controller.py` — the controller (the starter boilerplate at the top is unchanged).
- `README.md` — this file.
