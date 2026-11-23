"""
Rock Band 3 body proportions (height, weight and muscle).

RB3's character creator has three body sliders (0..1 in steps of 0.1). The body shapes come
from one CharClip per gender, 'male' and 'female', in the CharClipSet 'deform_clips'
(char/main/deform.milo, the BandCharDesc 'deform_path'): 18 samples of bone-local channels,
plus a single-sample block of channels that don't change with the body (fingers and such).

===========================================================================================
WEIGHTS - recovered from the retail Xbox 360 executable
===========================================================================================
BandCharDesc::ComputeDeformWeights (0x82322ee0) isn't in the decompilation; this is a
straight reading of the PowerPC code. The 18 samples are 6 body groups x 3 heights, index
group * 3 + height step:

    height    0 -> short, 0.5 -> normal, 1 -> tall, linear in between:
              h < 0.5: (normal 2h, tall 0, short 1-2h); else (normal 2-2h, tall 2h-1, short 0)

    groups    the corners of a weight (x) by muscle (y) grid:
                 group 0  average, muscle 1      group 3  skinny,  muscle 1
                 group 1  heavy,   muscle 0      group 4  average, muscle 0
                 group 2  heavy,   muscle 1      group 5  skinny,  muscle 0
              The muscle 0.5 row is the even mix of the two rows. The 3x3 grid is split
              into 8 triangles and the slider point is interpolated barycentrically inside
              the one it falls in (helper 0x82322e50 does the point-in-triangle test).

The skin normal maps follow the same groups (0x823232e0): the group whose three samples
add up to the most picks <gender>_<part>_norm_<suffix>.tex - '' (the plain _norm.tex),
heavy, athletic, skinny, soft (female: hourglass), weak - first maximum wins, so the
default 0.5/0.5/0.5 body, an exact tie between groups 0 and 4, gets the plain map.

===========================================================================================
CLIP FORMAT - verified on deform.milo_xbox (male 138 + 231 channels, female 171 + 198)
===========================================================================================
Each block: u32 channel count, (symbol name, float weight) per channel, seven u32 type
offsets (pos, scale, quat, rotx, roty, rotz, end), u32 compression, u32 sample count, u32
frame count (+ that many floats), then the samples, each padded to 16 bytes. Compression 2
(what RB3 uses): positions and scales as three int16 x 1300/32767, quaternions as four
int16 / 32767, single-axis rotations as int16 x 0.00061035156 rad.

Applying is CharBonesMeshes::PoseMeshes: a .pos replaces the bone's local position, a
.quat or .rotx/.roty/.rotz replaces its local rotation outright (RotateAboutZ sets the
matrix; scale goes back to 1), then a .scale sets the local scale. RB3's deform milo has no
bone hierarchy of its own, so forward kinematics runs over skeleton.milo's Trans
hierarchy. At 0.5/0.5/0.5 that puts the bone_* bones on skeleton.milo's rest pose, the
pose the outfits are bound in.

Girth works as in Rock Band 2 (see rb2_proportions): the meshes are skinned only to bone_*
bones, and each mesh's MeshDeform re-skins it through the exo_* bones the sliders scale.
RB3 keeps the MeshDeforms inside each item's inline _resource subdirectory, so they are
read with the nested walker. The result goes into an 'RB3 Proportions' shape key.
"""

import json
import math
import struct

import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, PointerProperty,
                       StringProperty)
from bpy_extras.io_utils import ImportHelper

from .io import read_milo_container_body, parse_milo_skeleton
from .utilities import _log
from . import rb2_proportions as rb2p


# ---------------------------------------------------------------------------------------
# Weights (ComputeDeformWeights, 360 @0x82322ee0)
# ---------------------------------------------------------------------------------------

_GRID_VERTS = (
    ((0.0, 0.0), (0, 0, 0, 0, 0, 1)),
    ((0.5, 0.0), (0, 0, 0, 0, 1, 0)),
    ((1.0, 0.0), (0, 1, 0, 0, 0, 0)),
    ((0.0, 0.5), (0, 0, 0, .5, 0, .5)),
    ((0.5, 0.5), (.5, 0, 0, 0, .5, 0)),
    ((1.0, 0.5), (0, .5, .5, 0, 0, 0)),
    ((0.0, 1.0), (0, 0, 0, 1, 0, 0)),
    ((0.5, 1.0), (1, 0, 0, 0, 0, 0)),
    ((1.0, 1.0), (0, 0, 1, 0, 0, 0)),
)
_GRID_TRIS = ((0, 1, 3), (1, 4, 3), (1, 5, 4), (1, 2, 5),
              (3, 7, 6), (3, 4, 7), (4, 5, 7), (5, 8, 7))
GROUP_LABELS = ("Average, Muscular", "Heavy", "Heavy, Muscular", "Skinny, Muscular",
                "Average", "Skinny")
NORMAL_SUFFIX = {'male': ('', 'heavy', 'athletic', 'skinny', 'soft', 'weak'),
                 'female': ('', 'heavy', 'athletic', 'skinny', 'hourglass', 'weak')}
N_SAMPLES = 18


def _in_tri(tri, p):
    sign = 0.0
    for i in range(3):
        (ax, ay), (bx, by) = _GRID_VERTS[tri[i]][0], _GRID_VERTS[tri[(i + 1) % 3]][0]
        c = (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax)
        if sign == 0.0:
            sign = c
        elif c * sign < 0.0:
            return False
    return True


def height_weights(h):
    """[normal, tall, short]"""
    h = min(max(h, 0.0), 1.0)
    if h < 0.5:
        return [2.0 * h, 0.0, 1.0 - 2.0 * h]
    return [2.0 - 2.0 * h, 2.0 * h - 1.0, 0.0]


def group_weights(weight, muscle):
    w = min(max(weight, 0.0), 1.0)
    m = min(max(muscle, 0.0), 1.0)
    tri = next(t for t in _GRID_TRIS if _in_tri(t, (w, m)))
    (x0, y0), (x1, y1), (x2, y2) = (_GRID_VERTS[i][0] for i in tri)
    det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    b1 = ((w - x0) * (y2 - y0) - (x2 - x0) * (m - y0)) / det
    b2 = ((x1 - x0) * (m - y0) - (w - x0) * (y1 - y0)) / det
    bary = (1.0 - b1 - b2, b1, b2)
    return [sum(bary[k] * _GRID_VERTS[tri[k]][1][g] for k in range(3)) for g in range(6)]


def compute_deform_weights(height, weight, muscle):
    """The game's 18 sample weights, index group * 3 + height step."""
    g = group_weights(weight, muscle)
    hw = height_weights(height)
    return [g[i] * hw[k] for i in range(6) for k in range(3)]


def body_type(gender, height, weight, muscle):
    """(group index, normal-map suffix) the game picks for the skin normal maps."""
    ws = compute_deform_weights(height, weight, muscle)
    best, best_v = 0, 0.0
    for g in range(6):
        s = ws[g * 3] + ws[g * 3 + 1] + ws[g * 3 + 2]
        if best_v < s:
            best, best_v = g, s
    suffixes = NORMAL_SUFFIX['female' if gender == 'female' else 'male']
    return best, suffixes[best]


# ---------------------------------------------------------------------------------------
# Reading the deform clips
# ---------------------------------------------------------------------------------------

class DeformError(Exception):
    pass


POS_SCALE = 1300.0 / 32767.0
ROT_SCALE = 0.00061035156
_KIND_OF = ('pos', 'scale', 'quat', 'rotx', 'roty', 'rotz')


def _u32(c, o):
    return struct.unpack_from('>I', c, o)[0]


def _read_block(c, o):
    n = _u32(c, o); o += 4
    names = []
    for _ in range(n):
        ln = _u32(c, o)
        if ln > 256:
            raise DeformError(f"implausible channel name length {ln}")
        names.append(c[o + 4:o + 4 + ln].decode('latin-1'))
        o += 4 + ln + 4                                   # name, then its weight
    offs = struct.unpack_from('>7I', c, o); o += 28
    comp, n_samples, n_frames = struct.unpack_from('>3I', c, o); o += 12
    if comp > 2:
        raise DeformError(f"compression {comp} isn't handled (RB3 deform clips use 2)")
    if offs[6] != n or offs[0] != 0:
        raise DeformError(f"channel offsets {offs} don't match {n} channels")
    o += 4 * n_frames
    counts = [offs[i + 1] - offs[i] for i in range(6)]
    vec = 6 if comp >= 2 else 16
    quat = 8 if comp >= 1 else 16
    rot = 2 if comp >= 1 else 4
    raw = counts[0] * vec + counts[1] * vec + counts[2] * quat + sum(counts[3:]) * rot
    stride = (raw + 15) & ~15
    if o + stride * n_samples > len(c):
        raise DeformError("samples run past the end of the clip")
    kinds = []
    for k, cnt in enumerate(counts):
        kinds += [_KIND_OF[k]] * cnt
    samples = []
    for s in range(n_samples):
        q = o + s * stride
        vals = []
        for kind in kinds:
            if kind in ('pos', 'scale'):
                if comp >= 2:
                    vals.append([v * POS_SCALE for v in struct.unpack_from('>3h', c, q)])
                else:
                    vals.append(list(struct.unpack_from('>3f', c, q)))
                q += vec
            elif kind == 'quat':
                if comp >= 1:
                    vals.append([max(v / 32767.0, -1.0) for v in struct.unpack_from('>4h', c, q)])
                else:
                    vals.append(list(struct.unpack_from('>4f', c, q)))
                q += quat
            else:
                if comp >= 1:
                    vals.append([struct.unpack_from('>h', c, q)[0] * ROT_SCALE])
                else:
                    vals.append([struct.unpack_from('>f', c, q)[0]])
                q += rot
        samples.append(vals)
    return dict(names=names, kinds=kinds, samples=samples), o + stride * n_samples


def _clip_blocks(clip):
    blocks, o = [], 0
    while True:
        p = rb2p._find_block(clip, o)
        if p is None:
            break
        blk, o = _read_block(clip, p)
        blocks.append(blk)
    return blocks


def _walk(body):
    from .mesh_importer import walk_milo_entries
    return walk_milo_entries(body)


def read_deform_milo(filepath):
    """{gender: {'full': 18-sample block, 'one': 1-sample block or None}}."""
    with open(filepath, 'rb') as f:
        body = read_milo_container_body(f.read())
    out = {}
    for _depth, _path, etype, ename, s, e in _walk(body):
        if etype != 'CharClip' or ename not in ('male', 'female'):
            continue
        blocks = _clip_blocks(body[s:e])
        full = next((b for b in blocks if len(b['samples']) == N_SAMPLES), None)
        if full is None:
            raise DeformError(f"'{ename}': expected an {N_SAMPLES}-sample block, found "
                              f"{[len(b['samples']) for b in blocks]}")
        one = next((b for b in blocks if len(b['samples']) == 1), None)
        out[ename] = dict(full=full, one=one)
    if not out:
        raise DeformError("no 'male' / 'female' CharClip in this milo - is it RB3's "
                          "char/main/deform.milo?")
    return out


def _stem(name):
    return name[:-5] if name.endswith('.mesh') else name


def _decode_trans(chunk):
    """(local 12 floats, parent name) of a revision-9 Trans object, or None."""
    try:
        if len(chunk) < 120 or (_u32(chunk, 0) & 0xFFFF) != 9:
            return None
        q = 8
        tl = _u32(chunk, q)
        if tl > 64:
            return None
        q += 4 + tl
        if chunk[q] not in (0, 1):
            return None
        q += 1
        nl = _u32(chunk, q)
        if nl > 256:
            return None
        q += 4 + nl
        local = struct.unpack_from('>12f', chunk, q); q += 96   # local, then world
        q += 4
        tl2 = _u32(chunk, q)
        if tl2 > 256:
            return None
        q += 4 + tl2 + 1
        pl = _u32(chunk, q)
        if pl > 256:
            return None
        return list(local), chunk[q + 4:q + 4 + pl].decode('latin-1')
    except (struct.error, IndexError, UnicodeDecodeError):
        return None


def hierarchy_from_skeleton(filepath):
    """{bone stem: [parent stem, 12 floats (row-vector 3x3 + position)]} from a milo's
    Trans entries, nested directories included - skeleton.milo keeps its bones in the inline
    'skeleton_unshared' directory, and a character milo (main.milo) carries a copy."""
    with open(filepath, 'rb') as f:
        body = read_milo_container_body(f.read())
    hier = {}
    try:
        entries = _walk(body)
    except Exception:
        entries = []
    for _depth, _path, etype, ename, s, e in entries:
        if etype == 'Trans' and (ename.startswith(('bone_', 'exo_', 'spot_'))):
            got = _decode_trans(body[s:e])
            if got:
                local, parent = got
                hier[_stem(ename)] = [_stem(parent) if parent else '', local]
    if not hier:
        _dir, bones, _cols = parse_milo_skeleton(filepath)
        for name, m12, parent in bones:
            hier[_stem(name)] = [_stem(parent) if parent else '', list(m12)]
    if not hier:
        raise DeformError(f"no bones read from {filepath}")
    return hier


# ---------------------------------------------------------------------------------------
# Blending and forward kinematics
# ---------------------------------------------------------------------------------------

def blend_block(block, weights):
    """{channel: (kind, value)}; quaternions sign-aligned and renormalised."""
    out = {}
    samples = block['samples']
    if len(samples) == 1:
        for i, (name, kind) in enumerate(zip(block['names'], block['kinds'])):
            out[name] = (kind, list(samples[0][i]))
        return out
    for i, (name, kind) in enumerate(zip(block['names'], block['kinds'])):
        if kind == 'quat':
            ref = samples[0][i]
            acc = [0.0, 0.0, 0.0, 0.0]
            for k, w in enumerate(weights):
                if w:
                    q = samples[k][i]
                    sign = 1.0 if sum(a * b for a, b in zip(q, ref)) >= 0 else -1.0
                    for c in range(4):
                        acc[c] += w * sign * q[c]
            ln = math.sqrt(sum(a * a for a in acc)) or 1.0
            out[name] = (kind, [a / ln for a in acc])
        else:
            dim = len(samples[0][i])
            out[name] = (kind, [sum(w * samples[k][i][c] for k, w in enumerate(weights) if w)
                                for c in range(dim)])
    return out


def channels_for(clip, height, weight, muscle, include_constant=True):
    weights = compute_deform_weights(height, weight, muscle)
    channels = {}
    if include_constant and clip.get('one'):
        channels.update(blend_block(clip['one'], weights))
    channels.update(blend_block(clip['full'], weights))
    return channels


def _quat_rows(x, y, z, w):
    """Milo row-vector rotation rows for a quaternion (the transpose of the column form,
    as rb2_proportions established)."""
    col = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
           [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
           [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    return [[col[c][r] for c in range(3)] for r in range(3)]


def _axis_rows(kind, a):
    """Hmx::Matrix3::RotateAboutX/Y/Z - each sets the whole matrix. Like the quaternions,
    the stored rows are the transpose of the decomp's Set(c, -s, 0, s, c, 0, ...) as
    written: the skeleton's rest knee (rows (0.995, -0.095, 0), (0.095, 0.995, 0)) is what
    the deform clip's normal-height knee.rotz of -0.095 has to reproduce. Read the other
    way round, the 'short' sample (-0.309) bent the knees backwards and threw the feet
    forward."""
    c, s = math.cos(a), math.sin(a)
    if kind == 'rotz':
        return [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]
    if kind == 'roty':
        return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
    return [[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]]


def _local(hier, stem, channels):
    m = hier[stem][1]
    rows = [list(m[0:3]), list(m[3:6]), list(m[6:9])]
    pos = list(m[9:12])
    p = channels.get(stem + '.pos')
    if p is not None:
        pos = list(p[1])
    q = channels.get(stem + '.quat')
    if q is not None:
        rows = _quat_rows(*q[1])
    for kind in ('rotx', 'roty', 'rotz'):
        r = channels.get(stem + '.' + kind)
        if r is not None:
            rows = _axis_rows(kind, r[1][0])
    sc = channels.get(stem + '.scale')
    if sc is not None:
        norms = [math.sqrt(sum(v * v for v in row)) or 1.0 for row in rows]
        rows = [[v * sc[1][i] / norms[i] for v in rows[i]] for i in range(3)]
    # column form (T @ M): column c of the 3x3 is row c of the milo matrix
    return [[rows[0][r], rows[1][r], rows[2][r], pos[r]] for r in range(3)] + \
        [[0.0, 0.0, 0.0, 1.0]]


def game_world_rb3(hier, channels):
    """{bone stem: 4x4 world matrix (column form, rows of lists)} for every bone of the
    hierarchy, with the channels applied as PoseMeshes does."""
    world = {}

    def resolve(stem, depth=0):
        hit = world.get(stem)
        if hit is not None:
            return hit
        parent = hier[stem][0]
        loc = _local(hier, stem, channels)
        if parent and parent in hier and parent != stem and depth < 256:
            world[stem] = rb2p._mat_mul(resolve(parent, depth + 1), loc)
        else:
            world[stem] = loc
        return world[stem]

    for stem in hier:
        resolve(stem)
    return world


# ---------------------------------------------------------------------------------------
# Applying to an armature
# ---------------------------------------------------------------------------------------

_DATA_KEY = "rb3_deform_data"
BAKED_KEY = rb2p.BAKED_KEY
SHAPE_KEY = "RB3 Proportions"
_CACHE = {}


def store_deform(arm_obj, gender, clip, hierarchy, deform_path, skeleton_path):
    arm_obj[_DATA_KEY] = json.dumps(dict(gender=gender, clip=clip, hierarchy=hierarchy))
    pg = arm_obj.rb3_proportions
    pg.source = deform_path
    pg.skeleton_source = skeleton_path
    pg["gender"] = 1 if gender == 'female' else 0


def _deform_of(arm_obj):
    raw = arm_obj.get(_DATA_KEY)
    if not raw:
        return None
    hit = _CACHE.get(arm_obj.name)
    if hit and hit[0] == raw:
        return hit[1]
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    _CACHE[arm_obj.name] = (raw, data)
    return data


def apply_proportions(arm_obj, report=False):
    """Poses the armature for its rb3_proportions sliders. Every bone of the game's
    skeleton is placed where forward kinematics over skeleton.milo's hierarchy puts it, as
    a change from the neutral body (0.5/0.5/0.5), which is the pose the outfits are bound
    in. Returns (bones placed, [deform bones the armature lacks])."""
    from mathutils import Matrix, Vector
    from .viseme_importer import build_bone_lookup
    deform = _deform_of(arm_obj)
    if deform is None or arm_obj.type != 'ARMATURE' or arm_obj.get(BAKED_KEY):
        return 0, []
    pg = arm_obj.rb3_proportions
    hier, clip = deform['hierarchy'], deform['clip']
    world = game_world_rb3(hier, channels_for(clip, pg.height, pg.weight, pg.muscle,
                                              pg.include_constant))
    neutral = game_world_rb3(hier, channels_for(clip, 0.5, 0.5, 0.5, pg.include_constant))
    driven = {n.rsplit('.', 1)[0] for blk in (clip['full'], clip.get('one')) if blk
              for n in blk['names']}

    lookup = build_bone_lookup(arm_obj)
    stem_of = {}
    for stem in world:
        name = lookup.get(stem)
        if name:
            stem_of[name] = stem
    missing = sorted(s for s in driven if s in world and lookup.get(s) is None)

    def col_norms(m):
        return [math.sqrt(sum(m[r][c] ** 2 for r in range(3))) or 1.0 for c in range(3)]

    placed = 0
    pose_m = {}
    for bone in rb2p._bone_order(arm_obj):
        pb = arm_obj.pose.bones[bone.name]
        rest = bone.matrix_local
        if bone.parent is not None:
            base = pose_m[bone.parent.name] @ (bone.parent.matrix_local.inverted() @ rest)
        else:
            base = rest.copy()
        stem = stem_of.get(bone.name)
        if stem is None:
            pose_m[bone.name] = base @ pb.matrix_basis
            continue
        g, n = world[stem], neutral[stem]
        # the change from neutral, carried onto this armature's rest
        pos = rest.to_translation() + Vector([g[r][3] - n[r][3] for r in range(3)])
        gs, ns = col_norms(g), col_norms(n)
        rel = Vector([gs[i] / ns[i] for i in range(3)])
        gq = Matrix([[g[r][c] / gs[c] for c in range(3)] for r in range(3)])
        nq = Matrix([[n[r][c] / ns[c] for c in range(3)] for r in range(3)])
        rot = gq @ nq.inverted() @ rest.to_3x3().normalized()
        target = Matrix.Translation(pos) @ rot.to_4x4() @ Matrix.Diagonal(rel.to_4d())
        basis = base.inverted() @ target
        loc, q, sca = basis.decompose()
        pb.location = loc
        pb.scale = sca
        if pb.rotation_mode == 'QUATERNION':
            pb.rotation_quaternion = q
        elif pb.rotation_mode == 'AXIS_ANGLE':
            axis, ang = q.to_axis_angle()
            pb.rotation_axis_angle = (ang, *axis)
        else:
            pb.rotation_euler = q.to_euler(pb.rotation_mode)
        pose_m[bone.name] = target
        placed += 1

    n_girth = rb2p.apply_girth(arm_obj, deform, world, neutral, pose_m, pg.apply_girth,
                               shape_key=SHAPE_KEY, relative=True)
    try:
        from . import rb3_textures
        _g, suffix = body_type(pg.gender, pg.height, pg.weight, pg.muscle)
        rb3_textures.update_body_normals(arm_obj, pg.gender, suffix)
    except Exception as e:
        _log(f"RB3 proportions: skin normal maps not updated ({e})")
    if report:
        _log(f"RB3 proportions: '{arm_obj.name}' height {pg.height:g}, weight "
             f"{pg.weight:g}, muscle {pg.muscle:g}: {placed} bone(s) placed"
             + (f", girth on {n_girth} mesh(es)" if n_girth else "")
             + (f"; {len(missing)} deform bone(s) not in the armature "
                f"({', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''})"
                if missing else ""))
    return placed, missing


def conform_rest(context, arm_obj, hier, clip_to, clip_from):
    """Moves the armature's rest pose by the change from clip_from's neutral body to
    clip_to's (each at 0.5 / 0.5 / 0.5), position and orientation; bones outside the
    hierarchy keep their offset from their parent. Used for female characters: RB3 has one
    shared skeleton (skeleton.milo, and main.milo's copy, both male-proportioned - checked
    bone for bone), the male outfits are bound exactly on it, and the female clip turns it
    female, so female meshes are bound on that female neutral body. Returns (bones moved,
    median distance moved)."""
    from mathutils import Matrix, Vector
    from .viseme_importer import build_bone_lookup
    to_w = game_world_rb3(hier, channels_for(clip_to, 0.5, 0.5, 0.5))
    from_w = game_world_rb3(hier, channels_for(clip_from, 0.5, 0.5, 0.5))
    lookup = build_bone_lookup(arm_obj)
    by_name = {lookup.get(s): s for s in to_w if lookup.get(s)}

    def rot_of(m):
        norms = [math.sqrt(sum(m[r][c] ** 2 for r in range(3))) or 1.0 for c in range(3)]
        return Matrix([[m[r][c] / norms[c] for c in range(3)] for r in range(3)])

    prev_active = context.view_layer.objects.active
    prev_mode = arm_obj.mode
    context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')
    moved = []
    try:
        ebs = arm_obj.data.edit_bones
        old = {eb.name: eb.matrix.copy() for eb in ebs}
        new = {}

        def visit(eb):
            m = old[eb.name]
            stem = by_name.get(eb.name)
            if stem is not None:
                t, f = to_w[stem], from_w[stem]
                pos = m.to_translation() + Vector([t[r][3] - f[r][3] for r in range(3)])
                rot = rot_of(t) @ rot_of(f).inverted() @ m.to_3x3().normalized()
                nm = Matrix.Translation(pos) @ rot.to_4x4()
                moved.append((pos - m.to_translation()).length)
            elif eb.parent is not None:
                nm = new[eb.parent.name] @ (old[eb.parent.name].inverted() @ m)
            else:
                nm = m
            new[eb.name] = nm
            for c in eb.children:
                visit(c)

        for eb in [e for e in ebs if e.parent is None]:
            visit(eb)
        for eb in ebs:
            length = eb.length
            eb.matrix = new[eb.name]
            eb.length = length
    finally:
        bpy.ops.object.mode_set(mode=prev_mode if prev_mode != 'EDIT' else 'OBJECT')
        context.view_layer.objects.active = prev_active
    moved.sort()
    return len(moved), (moved[len(moved) // 2] if moved else 0.0)


def clear_proportions(arm_obj):
    from .viseme_importer import build_bone_lookup
    deform = _deform_of(arm_obj)
    if deform is None:
        return 0
    lookup = build_bone_lookup(arm_obj)
    n = 0
    for stem in deform['hierarchy']:
        name = lookup.get(stem)
        pb = arm_obj.pose.bones.get(name) if name else None
        if pb is None:
            continue
        pb.location = (0.0, 0.0, 0.0)
        pb.scale = (1.0, 1.0, 1.0)
        pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        pb.rotation_euler = (0.0, 0.0, 0.0)
        n += 1
    for obj in rb2p._armature_users(arm_obj):
        keys = obj.data.shape_keys
        if keys and SHAPE_KEY in keys.key_blocks:
            keys.key_blocks[SHAPE_KEY].value = 0.0
    return n


def load_and_apply(arm_obj, deform_path, skeleton_path, gender, height=None, weight=None,
                   muscle=None):
    clips = read_deform_milo(deform_path)
    if gender not in clips:
        raise DeformError(f"no '{gender}' clip in {deform_path} (has {', '.join(clips)})")
    hier = hierarchy_from_skeleton(skeleton_path)
    store_deform(arm_obj, gender, clips[gender], hier, deform_path, skeleton_path)
    pg = arm_obj.rb3_proportions
    for key, v in (("height", height), ("weight", weight), ("muscle", muscle)):
        if v is not None:
            pg[key] = float(v)
    return apply_proportions(arm_obj, report=True)


# ---------------------------------------------------------------------------------------
# Girth: MeshDeform (nested inside RB3 items' _resource subdirectories)
# ---------------------------------------------------------------------------------------

def parse_meshdeform_rb3(data):
    """(mesh name, {'n', 'bones', 'inv' [12 floats each], 'vi', 'bk', 'w'}) from an RB3
    MeshDeform (revision 1) - the same dict rb2_proportions.parse_meshdeform returns.

    Layout (decomp rndobj/MeshDeform.cpp, the bone descriptors read from retail files):
    17-byte object header, symbol mesh, u32 bone count, per bone {symbol name, inverse
    bind (12 floats), bind (12 floats)}, then the weights as a byte array: u32 size, then
    per vertex u8 count n and n x (u8 bone index, u8 weight / 255), then the mesh inverse
    transform. RB2 stored a dense float table instead."""
    ln = _u32(data, 17)
    mesh = data[21:21 + ln].decode('latin-1')
    o = 21 + ln
    k = _u32(data, o); o += 4
    if k > 1024:
        raise DeformError(f"'{mesh}': implausible bone count {k}")
    bones, inv = [], []
    for _ in range(k):
        ln = _u32(data, o)
        if ln > 256:
            raise DeformError(f"'{mesh}': bad bone name length {ln}")
        bones.append(_stem(data[o + 4:o + 4 + ln].decode('latin-1')))
        o += 4 + ln
        inv.append(list(struct.unpack_from('>12f', data, o)))
        o += 96
    size = _u32(data, o); o += 4
    end = o + size
    if end > len(data):
        raise DeformError(f"'{mesh}': weight array runs past the end")
    vi, bk, w = [], [], []
    n = 0
    while o < end:
        cnt = data[o]; o += 1
        for j in range(cnt):
            b, x = data[o + 2 * j], data[o + 2 * j + 1]
            if x and b < k:
                vi.append(n); bk.append(b); w.append(x / 255.0)
        o += 2 * cnt
        n += 1
    return mesh, dict(n=n, bones=bones, inv=inv, vi=vi, bk=bk, w=w)


def read_meshdeforms(filepath):
    """{mesh name: MeshDeform dict} for every MeshDeform anywhere in a milo."""
    with open(filepath, 'rb') as f:
        body = read_milo_container_body(f.read())
    out = {}
    for _depth, _path, etype, ename, s, e in _walk(body):
        if etype != 'MeshDeform':
            continue
        try:
            mesh, md = parse_meshdeform_rb3(body[s:e])
            out[mesh] = md
        except (rb2p.DeformError, struct.error, UnicodeDecodeError) as ex:
            _log(f"RB3 proportions: MeshDeform '{ename}' skipped: {ex}")
    return out


def attach_meshdeforms(objects, filepath):
    """Stores a milo's MeshDeforms on the mesh objects named after their meshes (the same
    storage rb2_proportions' girth uses). Returns the names that got one."""
    mds = read_meshdeforms(filepath)
    done = []
    for obj in objects:
        if obj.type != 'MESH':
            continue
        md = (mds.get(rb2p._BLENDER_DUP.sub('', obj.name)) or mds.get(obj.data.name)
              or mds.get(obj.get("milo_mesh", "")))
        if md is None or md['n'] != len(obj.data.vertices):
            continue
        obj[rb2p._MD_KEY] = json.dumps(dict(
            n=md['n'], bones=md['bones'], inv=md['inv'], vi=rb2p._pack(md['vi'], 'i'),
            bk=rb2p._pack(md['bk'], 'i'), w=rb2p._pack(md['w'], 'f')))
        rb2p._GROUP_CACHE.pop(obj.name, None)
        done.append(obj.name)
    return done


# ---------------------------------------------------------------------------------------
# Properties, operators, panel
# ---------------------------------------------------------------------------------------

def _update(self, context):
    obj = self.id_data
    if obj is not None and obj.type == 'ARMATURE':
        apply_proportions(obj)


class RB3ProportionSettings(bpy.types.PropertyGroup):
    height: FloatProperty(name="Height", min=0.0, max=1.0, default=0.5, update=_update,
                          description="The character creator's height slider (0 = shortest, "
                                      "1 = tallest)")
    weight: FloatProperty(name="Weight", min=0.0, max=1.0, default=0.5, update=_update,
                          description="The character creator's weight slider (0 = "
                                      "skinniest, 1 = heaviest)")
    muscle: FloatProperty(name="Muscle", min=0.0, max=1.0, default=0.5, update=_update,
                          description="The character creator's muscle (build) slider")
    gender: EnumProperty(name="Gender", items=(('male', "Male", ""),
                                               ('female', "Female", "")), default='male')
    include_constant: BoolProperty(
        name="Include Constant Channels", default=True, update=_update,
        description="Also apply the deform clip's single-sample channels (fingers and other "
                    "bones that don't change with body shape)")
    apply_girth: BoolProperty(
        name="Apply Girth", default=True, update=_update,
        description="Widen and narrow the meshes through their MeshDeform data (the exo "
                    "bones), written to an 'RB3 Proportions' shape key")
    source: StringProperty(name="Deform Milo")
    skeleton_source: StringProperty(name="Skeleton Milo")


class POSE_OT_rb3_load_deform(bpy.types.Operator, ImportHelper):
    """Load Rock Band 3's deform milo (char/main/deform) onto this armature and pose it
    for its height, weight and muscle. The bone hierarchy comes from the skeleton milo
    beside it (char/main/skeleton)"""
    bl_idname = "pose.rb3_load_deform"
    bl_label = "Load RB3 Deform Milo"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".milo_xbox"
    filter_glob: StringProperty(default="*.milo_xbox;*.milo_ps3;*.milo", options={'HIDDEN'})
    skeleton_path: StringProperty(
        name="Skeleton Milo", subtype='FILE_PATH',
        description="The skeleton milo for the bone hierarchy. Empty: skeleton.milo_* beside "
                    "the deform milo")
    gender: EnumProperty(name="Gender", items=(('male', "Male", ""),
                                               ('female', "Female", "")), default='male')

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == 'ARMATURE'

    def invoke(self, context, event):
        self.gender = context.object.rb3_proportions.gender
        return super().invoke(context, event)

    def execute(self, context):
        import os
        arm = context.object
        skel = bpy.path.abspath(self.skeleton_path) if self.skeleton_path else ""
        if not skel:
            d = os.path.dirname(self.filepath)
            for f in sorted(os.listdir(d)):
                if f.lower().startswith('skeleton.milo'):
                    skel = os.path.join(d, f)
                    break
        if not skel or not os.path.isfile(skel):
            self.report({'ERROR'}, "No skeleton milo found beside the deform milo - set "
                                   "Skeleton Milo")
            return {'CANCELLED'}
        try:
            arm.rb3_proportions.gender = self.gender
            applied, unmatched = load_and_apply(arm, self.filepath, skel, self.gender)
        except Exception as e:
            self.report({'ERROR'}, f"Couldn't load deform: {e}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"{applied} bone(s) placed"
                    + (f"; {len(unmatched)} deform bone(s) not in this armature"
                       if unmatched else ""))
        return {'FINISHED'}


class POSE_OT_rb3_load_girth(bpy.types.Operator, ImportHelper):
    """Read the MeshDeform (girth) data from an RB3 item milo onto this armature's meshes of
    the same names"""
    bl_idname = "pose.rb3_load_girth"
    bl_label = "Load Girth From Milo"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".milo_xbox"
    filter_glob: StringProperty(default="*.milo_xbox;*.milo_ps3;*.milo", options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == 'ARMATURE'

    def execute(self, context):
        arm = context.object
        meshes = [o for o in bpy.data.objects if o.type == 'MESH' and any(
            m.type == 'ARMATURE' and m.object is arm for m in o.modifiers)]
        try:
            done = attach_meshdeforms(meshes, self.filepath)
        except Exception as e:
            self.report({'ERROR'}, f"Couldn't read MeshDeforms: {e}")
            return {'CANCELLED'}
        if not done:
            self.report({'WARNING'}, "No MeshDeform in this milo matches a mesh of this "
                                     "armature (by name and vertex count)")
            return {'CANCELLED'}
        apply_proportions(arm)
        self.report({'INFO'}, f"Girth data loaded for {len(done)} mesh(es)")
        return {'FINISHED'}


BODY_PRESETS = (                     # (label, weight, muscle) - the six sample groups
    ("Skinny", 0.0, 0.0), ("Skinny, Muscular", 0.0, 1.0),
    ("Average", 0.5, 0.0), ("Average, Muscular", 0.5, 1.0),
    ("Heavy", 1.0, 0.0), ("Heavy, Muscular", 1.0, 1.0),
)


class POSE_OT_rb3_proportion_preset(bpy.types.Operator):
    """Set weight and muscle to one of the six body shapes the deform clip stores"""
    bl_idname = "pose.rb3_proportion_preset"
    bl_label = "RB3 Body Preset"
    bl_options = {'REGISTER', 'UNDO'}

    weight: FloatProperty()
    muscle: FloatProperty()

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == 'ARMATURE'

    def execute(self, context):
        pg = context.object.rb3_proportions
        pg["weight"] = self.weight
        pg["muscle"] = self.muscle
        apply_proportions(context.object)
        return {'FINISHED'}


class POSE_OT_rb3_clear_proportions(bpy.types.Operator):
    """Return the bones to the skeleton's rest pose (the loaded deform is kept)"""
    bl_idname = "pose.rb3_clear_proportions"
    bl_label = "Clear Proportions"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == 'ARMATURE'

    def execute(self, context):
        n = clear_proportions(context.object)
        self.report({'INFO'}, f"{n} bone(s) reset to rest")
        return {'FINISHED'}


class DATA_PT_rb3_proportions(bpy.types.Panel):
    bl_label = "Rock Band 3 Proportions"
    bl_idname = "DATA_PT_rb3_proportions"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "data"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == 'ARMATURE'

    def draw(self, context):
        layout = self.layout
        arm = context.object
        pg = arm.rb3_proportions
        loaded = _deform_of(arm) is not None
        row = layout.row(align=True)
        row.prop(pg, "gender", text="")
        row.operator(POSE_OT_rb3_load_deform.bl_idname,
                     text="Load Deform Milo" if not loaded else "Reload Deform Milo",
                     icon='ARMATURE_DATA')
        if loaded:
            layout.label(text=bpy.path.basename(pg.source) or "(deform loaded)", icon='FILE')
        else:
            layout.label(text="Not in use - load char/main/deform", icon='INFO')

        col = layout.column()
        col.enabled = loaded
        col.prop(pg, "height", slider=True)
        col.prop(pg, "weight", slider=True)
        col.prop(pg, "muscle", slider=True)
        g, suffix = body_type(pg.gender, pg.height, pg.weight, pg.muscle)
        col.label(text=f"Body type: {GROUP_LABELS[g]}  (skin normal map "
                       f"'{'_norm_' + suffix if suffix else '_norm'}')", icon='INFO')
        col.prop(pg, "include_constant")
        row = col.row(align=True)
        row.prop(pg, "apply_girth")
        row.operator(POSE_OT_rb3_load_girth.bl_idname, text="", icon='MOD_MESHDEFORM')
        n_md = len(rb2p._armature_users(arm))
        col.label(text=f"Girth data on {n_md} mesh(es)" if n_md else
                  "No girth data yet - use the button beside Apply Girth",
                  icon='CHECKMARK' if n_md else 'INFO')

        box = col.box()
        box.label(text="Body Shapes (weight / muscle)")
        grid = box.grid_flow(columns=2, even_columns=True, align=True)
        for label, w, m in BODY_PRESETS:
            op = grid.operator(POSE_OT_rb3_proportion_preset.bl_idname, text=label)
            op.weight, op.muscle = w, m
        col.operator(POSE_OT_rb3_clear_proportions.bl_idname, icon='LOOP_BACK')


_classes = (
    RB3ProportionSettings,
    POSE_OT_rb3_load_deform,
    POSE_OT_rb3_load_girth,
    POSE_OT_rb3_proportion_preset,
    POSE_OT_rb3_clear_proportions,
    DATA_PT_rb3_proportions,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.rb3_proportions = PointerProperty(type=RB3ProportionSettings)


def unregister():
    del bpy.types.Object.rb3_proportions
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
