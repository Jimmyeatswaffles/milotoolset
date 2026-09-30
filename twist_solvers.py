"""
Arm twist solvers - ports of CharForeTwist and CharUpperTwist.

A performance never animates the eight arm twist bones (foreTwist1/2 and upperTwist1/2 on
each side). In the game, two solver objects on the character set them every frame from the
arm's pose. Without them the twist bones stay at rest, and because they are SIBLINGS of the
animated chain rather than part of it - foreTwist1 hangs off the upper arm beside the
forearm, upperTwist1 off the clavicle beside the upper arm - they don't even bend with the
elbow. Every sleeve and forearm vertex weighted to them gets stretched between the moving
arm and the stationary twist bone, which is the pinched, collapsed look seen in imported
performances.

This module reproduces the solvers so their output can be keyed like any other channel.

===========================================================================================
WHAT EACH SOLVER DOES - from the RB3 decompilation (system/char)
===========================================================================================
CharForeTwist::Poll measures how far the hand has twisted about the forearm's long axis,
then sets foreTwist1 to the forearm's full transform plus ONE THIRD of that twist, and
foreTwist2 to the forearm's orientation plus TWO THIRDS, placed partway from the forearm
towards the hand. So it carries the elbow bend as well as the twist.

CharUpperTwist::Poll reads the upper arm, builds a twist-free frame by swinging the
clavicle's axes onto the upper arm's direction, and sets upperTwist1 and upperTwist2 to that
direction with one third and two thirds of the upper arm's twist.

The math is ported literally, in the engine's own row-vector convention (rows are basis
vectors, world = local * parent), so no sign has to be translated by hand. Only the results
are converted to Blender. Three helpers aren't decompiled: MakeRotQuat is used only to swing
one axis onto another, which is computed directly as a shortest-arc rotation; LimitAng is
assumed to wrap into +/-180 degrees; Interp is a plain linear blend.

===========================================================================================
WIRING AND SETTINGS - read from the character milo
===========================================================================================
Which bones each solver uses, and the forearm offset and bias, are stored in the character
milo (billiejoe.milo_xbox holds foreTwist_L/R.ik and upperTwist_L/R.ik). The source's field
names are misleading for the upper arm: the field called mTwist2 is wired to the animated
upper arm, which is the INPUT, and the fields called mUpperArm and mTwist1 are wired to
upperTwist1 and upperTwist2, the outputs. Nothing overwrites the motion capture.

===========================================================================================
GDRB'S FOREARM OFFSET IS 90 DEGREES OFF RB3'S
===========================================================================================
Billie Joe stores offsets of +90 (left) and -90 (right). Fed his rest pose, the RB3 formula
with those offsets misses the stored twist bones by exactly 30 degrees - 90 divided by the
solver's 3. His stored rest pose settles what the offset has to be: the left hand's measured
twist at rest is +6.089 degrees and each twist bone sits 2.030 degrees further round, exactly
one third per step, so the effective offset is 0; the right comes out at 180. That's what
the RB3 header documents ("usually 180 for right hand, 0 for left hand"), and both stored
values are exactly 90 below it - consistent with GDRB measuring the wrist from an axis turned
90 degrees relative to RB3's. With the stored offset reduced by 90, the port reproduces the
stored forearm twist bones to 0.000 degrees on the left and 0.009 on the right. This is
inferred from one character's data rather than seen in GDRB code.

The upper arm needs no correction, but its bind pose isn't a resting point of the solver:
Billie Joe's upper twist bones are bound identical to the upper arm (full twist), where the
solver places them at a third and two thirds. So at the bind pose the solver turns them by
6.77 and 3.38 degrees - exactly what the code predicts, and what the game does too.
"""

import math
import struct

from .utilities import _log


# See the module docstring: GDRB forearm offsets are stored 90 degrees below the values the
# RB3-era formula expects.
GDRB_FORE_OFFSET_CORRECTION = -90.0
# Effective values if no character milo is supplied - RB3's documented defaults, which are
# also what Billie Joe's stored values work out to after the correction.
DEFAULT_FORE_OFFSET = {'L': 0.0, 'R': 180.0}
DEFAULT_FORE_BIAS = 0.0


# ---------------------------------------------------------------------------------------
# Row-vector 3x3 helpers (engine convention: rows are the basis vectors)
# ---------------------------------------------------------------------------------------

def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(a):
    n = math.sqrt(_dot(a, a))
    return (a[0] / n, a[1] / n, a[2] / n) if n > 1e-12 else a


def _vm(v, M):                       # row vector times matrix
    return tuple(v[0] * M[0][j] + v[1] * M[1][j] + v[2] * M[2][j] for j in range(3))


def _mm(A, B):                       # row-major product A * B
    return tuple(_vm(A[i], B) for i in range(3))


def _transpose(A):
    return tuple(tuple(A[j][i] for j in range(3)) for i in range(3))


def _lerp(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def _compose(Rl, tl, Rp, tp):        # world = local * parent
    return _mm(Rl, Rp), tuple(x + y for x, y in zip(_vm(tl, Rp), tp))


def _relative(Rc, tc, Rp, tp):       # local = child * parent^-1 (orthonormal parent)
    RpT = _transpose(Rp)
    return _mm(Rc, RpT), _vm(tuple(c - p for c, p in zip(tc, tp)), RpT)


def _limit_ang(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _rotate_about_x(a):              # Matrix3::RotateAboutX: Set(1,0,0, 0,c,s, 0,-s,c)
    c, s = math.cos(a), math.sin(a)
    return ((1.0, 0.0, 0.0), (0.0, c, s), (0.0, -s, c))


def _look_at(x, y):                  # z = normalize(x cross y); y = z cross x
    z = _norm(_cross(x, y))
    return (x, _cross(z, x), z)


def _swing(v, a, b):
    """v rotated by the shortest rotation taking a onto b."""
    a, b = _norm(a), _norm(b)
    ax = _cross(a, b)
    s = math.sqrt(_dot(ax, ax))
    if s < 1e-9:
        return v
    k = (ax[0] / s, ax[1] / s, ax[2] / s)
    ang = math.atan2(s, _dot(a, b))
    c, sn = math.cos(ang), math.sin(ang)
    kxv = _cross(k, v)
    kv = _dot(k, v)
    return tuple(v[i] * c + kxv[i] * sn + k[i] * kv * (1.0 - c) for i in range(3))


# ---------------------------------------------------------------------------------------
# The solvers
# ---------------------------------------------------------------------------------------

def fore_twist(Rf, tf, Rhl, thl, offset_deg, bias_deg, ratio):
    """CharForeTwist::Poll, in the upper arm's frame (its world = identity).

    Rf, tf: forearm local transform. Rhl, thl: hand local transform (relative to forearm).
    Returns ((R1, t1), (R2, t2)): foreTwist1's transform relative to the upper arm, and
    foreTwist2's relative to foreTwist1."""
    Rh, th = _compose(Rhl, thl, Rf, tf)
    c1 = max(-1.0, min(1.0, _dot(Rh[2], Rf[1])))
    c2 = max(-1.0, min(1.0, _dot(Rf[0], _cross(Rf[1], Rh[2]))))
    bias = math.radians(bias_deg)
    angle = _limit_ang(math.radians(offset_deg) + math.atan2(c2, c1) + bias) - bias
    m = _rotate_about_x(angle * 0.33333)
    R1 = _mm(m, Rf)
    t1 = tf
    t2w = _lerp(t1, th, ratio)
    R2w = _mm(m, R1)
    return (R1, t1), _relative(R2w, t2w, R1, t1)


def upper_twist(Ru, t1_rest, t2_rest_local):
    """CharUpperTwist::Poll, in the clavicle's frame (its world = identity).

    Ru: upper arm local rotation. t1_rest: upperTwist1's local position; t2_rest_local:
    upperTwist2's local position relative to upperTwist1. Returns the two twist bones'
    transforms, relative to the clavicle and to upperTwist1 respectively."""
    v68 = _swing((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), Ru[0])
    M1 = _look_at(Ru[0], _lerp(v68, Ru[1], 0.333))
    tw2 = tuple(x + y for x, y in zip(_vm(t2_rest_local, M1), t1_rest))
    M2 = _look_at(Ru[0], _lerp(v68, Ru[1], 0.666))
    return (M1, t1_rest), _relative(M2, tw2, M1, t1_rest)


# ---------------------------------------------------------------------------------------
# Reading solver settings from a character milo
# ---------------------------------------------------------------------------------------

def _syms(body, p, count):
    out = []
    for _ in range(count):
        if p + 4 > len(body):
            return None
        n = struct.unpack_from('>I', body, p)[0]
        if n == 0 or n > 80 or p + 4 + n > len(body):
            return None
        raw = body[p + 4:p + 4 + n]
        if not all(32 <= c < 127 for c in raw):
            return None
        out.append(raw.decode('latin-1'))
        p += 4 + n
    return out, p


def read_twist_configs(body):
    """Finds CharForeTwist and CharUpperTwist settings in a character milo body.

    Character milos nest subdirectories, so their entries aren't reliably delimited by end
    marker. The solver objects have distinctive layouts instead - CharForeTwist stores its
    offset, then the hand and twist2 bone names, then its bias; CharUpperTwist stores three
    bone names - so they're located by those layouts and validated by bone naming.

    Returns {'fore': {'L': (hand, twist2, offset, bias), ...},
             'upper': {'L': (input_upperArm, twist1, twist2), ...}}."""
    fore, upper = {}, {}
    for p in range(0, len(body) - 16):
        if len(fore) < 2:
            r = _syms(body, p + 4, 2)
            if r and r[0][0].startswith('bone_') and '-hand' in r[0][0] \
                    and 'foreTwist' in r[0][1] and r[1] + 4 <= len(body):
                side = 'L' if '_L-' in r[0][0] else ('R' if '_R-' in r[0][0] else None)
                if side and side not in fore:
                    off = struct.unpack_from('>f', body, p)[0]
                    bias = struct.unpack_from('>f', body, r[1])[0]
                    if abs(off) <= 360.0 and abs(bias) <= 360.0:
                        fore[side] = (r[0][0], r[0][1], off, bias)
        if len(upper) < 2:
            r = _syms(body, p, 3)
            if r and 'upperArm' in r[0][0] and 'upperTwist' in r[0][1] \
                    and 'upperTwist' in r[0][2]:
                side = 'L' if '_L-' in r[0][0] else ('R' if '_R-' in r[0][0] else None)
                if side and side not in upper:
                    upper[side] = tuple(r[0])
        if len(fore) == 2 and len(upper) == 2:
            break
    return {'fore': fore, 'upper': upper}


# ---------------------------------------------------------------------------------------
# Rigs: one per arm, resolved against an armature
# ---------------------------------------------------------------------------------------

class ArmTwistRig:
    """Everything needed to solve one arm's twist bones, resolved to armature bone names.

    rest(bone) must return that bone's rest transform relative to its parent in engine
    convention: (rows 3x3, translation)."""

    def __init__(self, side, names, fore_offset, fore_bias, rest):
        self.side = side
        self.names = names
        self.fore_offset = fore_offset
        self.fore_bias = fore_bias
        self.rest = {k: rest(v) for k, v in names.items()}
        hand_x = self.rest['hand'][1][0]
        twist2_x = self.rest['foreTwist2'][1][0]
        self.ratio = (twist2_x / hand_x) if abs(hand_x) > 1e-9 else 0.5

    def solve(self, upperarm_rot, forearm_rot, hand_rot):
        """Inputs are the three animated bones' local rotations in engine convention.
        Returns {role: (rows, translation)} for the four twist bones, each relative to its
        own parent."""
        Rf, tf = forearm_rot, self.rest['foreArm'][1]
        Rhl, thl = hand_rot, self.rest['hand'][1]
        (A1, a1), (A2, a2) = fore_twist(Rf, tf, Rhl, thl, self.fore_offset,
                                        self.fore_bias, self.ratio)
        (B1, b1), (B2, b2) = upper_twist(upperarm_rot, self.rest['upperTwist1'][1],
                                         self.rest['upperTwist2'][1])
        return {'foreTwist1': (A1, a1), 'foreTwist2': (A2, a2),
                'upperTwist1': (B1, b1), 'upperTwist2': (B2, b2)}


_ROLES = ('clavicle', 'upperArm', 'foreArm', 'hand',
          'upperTwist1', 'upperTwist2', 'foreTwist1', 'foreTwist2')


def build_rigs(lookup, parents, rest, configs=None):
    """Builds an ArmTwistRig per side whose bones all exist and whose hierarchy matches
    what the solvers assume. `lookup` maps a stem like 'bone_L-hand' to the armature bone
    name; `parents` maps armature bone name -> parent name; `rest(name)` as for the rig.

    Returns (rigs, notes) - notes explain any arm that was skipped."""
    configs = configs or {'fore': {}, 'upper': {}}
    rigs, notes = [], []
    for side in ('L', 'R'):
        names = {}
        for role in _ROLES:
            bn = lookup.get(f'bone_{side}-{role}')
            if not bn:
                names = None
                notes.append(f"{side} arm skipped: no bone for {role}")
                break
            names[role] = bn
        if names is None:
            continue
        expect = {'upperTwist1': 'clavicle', 'upperTwist2': 'upperTwist1',
                  'upperArm': 'clavicle', 'foreTwist1': 'upperArm',
                  'foreTwist2': 'foreTwist1', 'foreArm': 'upperArm', 'hand': 'foreArm'}
        wrong = [r for r, par in expect.items() if parents.get(names[r]) != names[par]]
        if wrong:
            notes.append(f"{side} arm skipped: hierarchy doesn't match the solvers "
                         f"({', '.join(wrong)} not parented as expected)")
            continue
        fc = configs['fore'].get(side)
        if fc:
            offset = fc[2] + GDRB_FORE_OFFSET_CORRECTION
            bias = fc[3]
            source = f"character milo (stored {fc[2]:+g}, effective {offset:+g})"
        else:
            offset, bias = DEFAULT_FORE_OFFSET[side], DEFAULT_FORE_BIAS
            source = f"defaults (effective {offset:+g})"
        rigs.append(ArmTwistRig(side, names, offset, bias, rest))
        notes.append(f"{side} arm twist solver: forearm offset from {source}, bias {bias:g}")
    return rigs, notes
