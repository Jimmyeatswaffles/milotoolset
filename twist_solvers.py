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
        # Dance Central rigs have no upperTwist bones (they use a shoulderTwist chain
        # instead), so the upper-arm solver only runs where its bones exist.
        self.has_upper = all(r in names for r in _UPPER_ROLES)

    def solve(self, upperarm_rot, forearm_rot, hand_rot):
        """Inputs are the three animated bones' local rotations in engine convention.
        Returns {role: (rows, translation)} for the four twist bones, each relative to its
        own parent."""
        Rf, tf = forearm_rot, self.rest['foreArm'][1]
        Rhl, thl = hand_rot, self.rest['hand'][1]
        (A1, a1), (A2, a2) = fore_twist(Rf, tf, Rhl, thl, self.fore_offset,
                                        self.fore_bias, self.ratio)
        out = {'foreTwist1': (A1, a1), 'foreTwist2': (A2, a2)}
        if self.has_upper:
            (B1, b1), (B2, b2) = upper_twist(upperarm_rot, self.rest['upperTwist1'][1],
                                             self.rest['upperTwist2'][1])
            out['upperTwist1'] = (B1, b1)
            out['upperTwist2'] = (B2, b2)
        return out


_FORE_ROLES = ('upperArm', 'foreArm', 'hand', 'foreTwist1', 'foreTwist2')
_UPPER_ROLES = ('clavicle', 'upperTwist1', 'upperTwist2')
_EXPECTED_PARENT = {'upperTwist1': 'clavicle', 'upperTwist2': 'upperTwist1',
                    'upperArm': 'clavicle', 'foreTwist1': 'upperArm',
                    'foreTwist2': 'foreTwist1', 'foreArm': 'upperArm', 'hand': 'foreArm'}


def build_rigs(lookup, parents, rest, configs=None,
               offset_correction=GDRB_FORE_OFFSET_CORRECTION):
    """Builds an ArmTwistRig per side whose bones exist and whose hierarchy matches what
    the solvers assume. `lookup` maps a stem like 'bone_L-hand' to the armature bone name;
    `parents` maps armature bone name -> parent name; `rest(name)` as for the rig.

    The forearm bones are required; the upper-arm twist bones are optional, because Dance
    Central rigs don't have them (a five-bone shoulderTwist chain sits there instead, driven
    by a different solver whose settings aren't in the skeleton). `offset_correction` is
    added to forearm offsets read from a character milo: -90 for GDRB (see the module
    docstring), 0 for Dance Central, whose skeletons reproduce their stored forearm twist
    rest pose with RB3's documented offsets (0 left, 180 right) to within 0.05 degrees.

    Returns (rigs, notes) - notes explain what was and wasn't solved."""
    configs = configs or {'fore': {}, 'upper': {}}
    rigs, notes = [], []
    for side in ('L', 'R'):
        missing = [r for r in _FORE_ROLES if not lookup.get(f'bone_{side}-{r}')]
        if missing:
            notes.append(f"{side} arm skipped: no bone for {', '.join(missing)}")
            continue
        names = {r: lookup[f'bone_{side}-{r}'] for r in _FORE_ROLES}
        if all(lookup.get(f'bone_{side}-{r}') for r in _UPPER_ROLES):
            names.update({r: lookup[f'bone_{side}-{r}'] for r in _UPPER_ROLES})
        wrong = [r for r, par in _EXPECTED_PARENT.items()
                 if r in names and par in names and parents.get(names[r]) != names[par]]
        if wrong:
            notes.append(f"{side} arm skipped: hierarchy doesn't match the solvers "
                         f"({', '.join(wrong)} not parented as expected)")
            continue
        fc = configs['fore'].get(side)
        if fc:
            offset = fc[2] + offset_correction
            bias = fc[3]
            source = f"character milo (stored {fc[2]:+g}, effective {offset:+g})"
        else:
            offset, bias = DEFAULT_FORE_OFFSET[side], DEFAULT_FORE_BIAS
            source = f"defaults (effective {offset:+g})"
        rig = ArmTwistRig(side, names, offset, bias, rest)
        rigs.append(rig)
        notes.append(f"{side} arm twist solver: forearm offset from {source}, bias {bias:g}"
                     + ("" if rig.has_upper else
                        "; upper arm not solved (no upperTwist bones on this rig)"))
    return rigs, notes


# ---------------------------------------------------------------------------------------
# CharBlendBone - Dance Central's shoulder, thigh and spine twist bones
# ---------------------------------------------------------------------------------------
#
# Dance Central rigs drive their extra twist bones with CharBlendBone (DC3 decompilation,
# system/char/CharBlendBone.cpp). Each one names two source bones and a list of target
# bones with a weight each, plus flags for which components to blend. Every frame it takes
# the sources' world transforms, blends each enabled position axis linearly and the rotation
# by Nlerp (math/Rot.cpp: quaternions, normalised lerp, back to a matrix), and writes the
# result to the target's world transform.
#
# A retail DC character (main.milo_xbox) carries seven: shoulder_L/R-pos and -twist spread
# shoulderTwist2-4 between shoulderTwist1 (on the clavicle) and shoulderTwist5 (on the upper
# arm); thigh_L/R-twist set thighTwist01 halfway between the thigh and thigh_dummy; and
# spine_twist2 sets spineTwist3 a tenth of the way from spine1 to spine3. Fed DC1's rest
# pose, they reproduce the stored shoulder bones exactly in position and within 0.3 degrees
# in rotation (exact at weight 0.5, slightly off at 0.25/0.75 - the signature of Nlerp
# against a constant-speed blend), and the thigh bones within 0.05 degrees (2.5 on one
# female side). spineTwist3's bind pose sits near spine3 rather than where the solver puts
# it, so at rest the solver turns it about 6 degrees - as the game does.
#
# Blending is linear in position and Nlerp in rotation, both unchanged by a rigid change of
# frame, so it's evaluated in the armature's own space rather than true world space.

_BLEND_FLAGS_REV3 = 4      # trans x, y, z, rotation
_BLEND_FLAGS_REV4 = 5      # ... plus set-local


def _quat_from_rows(R):
    """Quaternion (w, x, y, z) for engine-convention rows. Used only to Nlerp between two
    rotations and convert back, so the result doesn't depend on the row/column convention
    (the conjugate of an Nlerp is the Nlerp of the conjugates)."""
    m = R
    t = m[0][0] + m[1][1] + m[2][2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2.0
        return (0.25 * s, (m[1][2] - m[2][1]) / s, (m[2][0] - m[0][2]) / s,
                (m[0][1] - m[1][0]) / s)
    i = max(range(3), key=lambda k: m[k][k])
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(1e-12, 1.0 + m[i][i] - m[j][j] - m[k][k])) * 2.0
    q = [0.0, 0.0, 0.0, 0.0]
    q[1 + i] = 0.25 * s
    q[1 + j] = (m[i][j] + m[j][i]) / s
    q[1 + k] = (m[i][k] + m[k][i]) / s
    q[0] = (m[j][k] - m[k][j]) / s
    return tuple(q)


def _rows_from_quat(q):
    w, x, y, z = q
    return ((1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)),
            (2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)),
            (2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)))


def _nlerp(a, b, t):
    if sum(x * y for x, y in zip(a, b)) < 0.0:
        b = tuple(-x for x in b)
    q = [x + (y - x) * t for x, y in zip(a, b)]
    n = math.sqrt(sum(x * x for x in q)) or 1.0
    return tuple(x / n for x in q)


def _try_blend_bone(c):
    """Parses one CharBlendBone object, or returns None. The object header's length isn't
    fixed, so the start of the target list is searched for, and a parse is only accepted if
    every target and source is a bone name, every weight lies in 0-1, the flags are 0/1, and
    the object ends exactly where its data does."""
    if len(c) < 8:
        return None
    rev = struct.unpack_from('>I', c, 0)[0]
    if rev not in (3, 4):
        return None
    nflags = _BLEND_FLAGS_REV4 if rev > 3 else _BLEND_FLAGS_REV3
    for start in range(4, min(80, len(c) - 8)):
        try:
            p = start
            n = struct.unpack_from('>I', c, p)[0]; p += 4
            if not 1 <= n <= 64:
                continue
            targets = []
            for _ in range(n):
                L = struct.unpack_from('>I', c, p)[0]
                name = c[p + 4:p + 4 + L].decode('latin-1'); p += 4 + L
                w = struct.unpack_from('>f', c, p)[0]; p += 4
                if not name.startswith('bone_') or not -1e-6 <= w <= 1.0 + 1e-6:
                    raise ValueError
                targets.append((name, w))
            srcs = []
            for _ in range(2):
                L = struct.unpack_from('>I', c, p)[0]
                name = c[p + 4:p + 4 + L].decode('latin-1'); p += 4 + L
                if not name.startswith('bone_'):
                    raise ValueError
                srcs.append(name)
            flags = list(c[p:p + nflags]); p += nflags
            if p != len(c) or any(f not in (0, 1) for f in flags):
                continue
            return dict(targets=targets, src1=srcs[0], src2=srcs[1],
                        trans=tuple(bool(f) for f in flags[:3]), rot=bool(flags[3]))
        except (struct.error, ValueError, UnicodeDecodeError):
            continue
    return None


def read_blend_bones(body):
    """Every CharBlendBone in a character milo body. Character milos nest subdirectories,
    which the generic directory walker can't always follow, so objects are delimited by end
    marker and recognised by their layout."""
    marker = b'\xAD\xDE\xAD\xDE'
    marks, i = [], 0
    while True:
        j = body.find(marker, i)
        if j < 0:
            break
        marks.append(j)
        i = j + 4
    out = []
    for k in range(len(marks) - 1):
        bb = _try_blend_bone(body[marks[k] + 4:marks[k + 1]])
        if bb is not None:
            out.append(bb)
    return out


class BlendBoneSolver:
    """The CharBlendBones of one character, resolved against an armature."""

    def __init__(self, configs, lookup, parents):
        self.solvers = []
        self.notes = []
        self.pos_targets = set()
        for cfg in configs:
            try:
                s1 = lookup[_bone_key(cfg['src1'])]
                s2 = lookup[_bone_key(cfg['src2'])]
                tg = [(lookup[_bone_key(n)], w) for n, w in cfg['targets']]
            except KeyError as e:
                self.notes.append(f"blend bone skipped: {e.args[0]} isn't on this armature")
                continue
            self.solvers.append((s1, s2, tg, cfg['trans'], cfg['rot']))
            if any(cfg['trans']):
                self.pos_targets.update(n for n, _w in tg)
        self.parents = parents
        self.targets = sorted({n for s in self.solvers for n, _w in s[2]})

    def solve(self, local):
        """`local(name)` -> (rows, translation) for any bone at this frame. Returns
        {target bone: (rows, translation) relative to its parent}."""
        world_cache = {}

        def world(n):
            if n in world_cache:
                return world_cache[n]
            R, t = local(n)
            p = self.parents.get(n)
            if p is not None:
                R, t = _compose(R, t, *world(p))
            world_cache[n] = (R, t)
            return world_cache[n]

        solved = {}
        for s1, s2, targets, trans, rot in self.solvers:
            Ra, ta = world(s1)
            Rb, tb = world(s2)
            qa, qb = _quat_from_rows(Ra), _quat_from_rows(Rb)
            for name, w in targets:
                R, t = solved.get(name) or world(name)
                if any(trans):
                    t = tuple((ta[i] + (tb[i] - ta[i]) * w) if trans[i] else t[i]
                              for i in range(3))
                if rot:
                    R = _rows_from_quat(_nlerp(qa, qb, w))
                solved[name] = (R, t)
        out = {}
        for name, (R, t) in solved.items():
            p = self.parents.get(name)
            out[name] = _relative(R, t, *world(p)) if p is not None else (R, t)
        return out


def _bone_key(name):
    """'bone_L-shoulderTwist2.mesh' -> 'bone_L-shoulderTwist2', the form armature lookups
    are keyed by."""
    for ext in ('.mesh', '.trans'):
        if name.endswith(ext):
            return name[:-len(ext)]
    return name
