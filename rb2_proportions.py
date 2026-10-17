"""
Rock Band 2 body proportions (height and weight).

RB2's character creator has two body sliders, each with five steps (0, 0.25, 0.5, 0.75,
1). The body shapes come from a CharClipSamples clip named 'deform' in
<gender>_<instrument>_deform.milo (male_vocal_deform, female_vocal_deform): five samples
of bone-local channels, in the same order as the five body presets the game's own editor
scripts define (char_objects.dta, CHAR_COMMON):

    sample 0  neutral           height 0.5  weight 0.5
    sample 1  tall_and_fat      height 1    weight 1
    sample 2  short_and_fat     height 0    weight 1
    sample 3  tall_and_skinny   height 1    weight 0
    sample 4  short_and_skinny  height 0    weight 0

Length comes from bone_* positions (thigh, shin, spine segments), girth from scaling the
exo_* helper bones, and bone_head-scale scales the head. A second, single-sample block
holds channels that don't change with body shape (fingers and the like).

Applying: the samples are each bone's LOCAL transform relative to its parent in the
GAME's hierarchy, which the deform milo carries itself as CharBone entries (a local
matrix and a parent per bone; 137 male, 144 female). Forward kinematics over that
hierarchy with the neutral sample lands within 0.24 units of the bind pose the outfit
meshes are skinned at (checked on strapjacket_resource), so the positions are computed
that way and each Blender bone is placed there in armature space - independent of how
the Blender armature happens to be parented. An armature whose parenting differs from the
game's (a part's file bones, a different skeleton) would otherwise scatter the body: an
offset meant for bone_spine3 applied under another parent.

Girth: the outfit meshes are skinned only to bone_* bones; the exo_* bones the deform
scales reach them through each mesh's MeshDeform (torso_naked.2.deform,
strapjacket_resource.deform, ...): a second, full skinning of every vertex to the exo
bones (N x K float weights, each row summing to 1), then per bone its name, inverse bind
matrix and bind matrix. Re-skinned through the deformed exo bones the strap jacket's
chest goes from 43.0 x 10.3 (width x depth) at neutral to 50.7 x 14.5 tall-and-fat and
38.2 x 8.7 short-and-skinny; at neutral it moves 0.3 units at most. Without it a heavy
body is lengthened but not widened and looks stretched. Here the change from neutral is
written into an 'RB2 Proportions' shape key, pre-divided by what the posed armature will
do to each vertex, so that the armature lands every vertex where the game puts it.
Only MeshDeform bones that are part of the body deform contribute girth. The heads'
MeshDeforms also list exo_head and every face bone (head2.mesh: brows, lids, cheeks, lips,
jaw; head2_chops: only those), which RB2's body sliders don't reshape - the face changes
only by bone_head-scale's uniform scale. That share of each vertex follows the armature
unchanged, so the face keeps its shape and a beard or eyebrow mesh gets no shape key.

Blending: the scripts only set [height] and [weight] and call deform; the blend itself is
in the game's code. Here the five samples sit on a 3x3 height x weight grid - corners and
centre as above, each edge midpoint the average of its two corners - interpolated
bilinearly in each quarter. All five presets come out exactly; values in between are a
smooth estimate. Prefabs don't always sit on the slider steps (guitar9 is height 0.8,
vocals6 height 0.15 / weight 0.75), so any value is accepted, with optional snapping.

FORMAT (verified on male_vocal_deform: 90 + 79 channels, female_vocal_deform: 127 + 51):
each block is a count, then (symbol channel, float weight) per channel, then seven u32
running channel counts by kind (pos, scale, quat, rotx, roty, rotz, end), u32 compression
(1 here: positions and scales as float x,y,z + pad, quaternions as four int16 / 32767),
u32 sample count, u32 frame count (+ that many floats), then the samples - each padded
to 16 bytes (the female's 2024-byte samples sit 2032 apart).
"""

import base64
import json
import math
import struct

import bpy
from bpy.props import BoolProperty, FloatProperty, PointerProperty, StringProperty
from bpy_extras.io_utils import ImportHelper
from mathutils import Matrix, Quaternion, Vector

from .io import read_milo_container_body
from .utilities import _log
from .viseme_importer import build_bone_lookup


SLIDER_STEPS = (0.0, 0.25, 0.5, 0.75, 1.0)
PRESETS = (                      # (label, height, weight) - char_objects.dta's editor list
    ("Neutral", 0.5, 0.5),
    ("Tall & Fat", 1.0, 1.0),
    ("Short & Fat", 0.0, 1.0),
    ("Tall & Skinny", 1.0, 0.0),
    ("Short & Skinny", 0.0, 0.0),
)
_KINDS = ('pos', 'scale', 'quat')
_BLENDER_DUP = __import__('re').compile(r'\.\d{3}$')


class DeformError(Exception):
    pass


# ---------------------------------------------------------------------------------------
# Reading the deform clip
# ---------------------------------------------------------------------------------------

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
    counts = struct.unpack_from('>7I', c, o); o += 28
    comp, n_samples, n_frames = struct.unpack_from('>3I', c, o); o += 12
    o += 4 * n_frames
    if counts[0] != 0 or counts[3] != n or counts[6] != n:
        raise DeformError(f"only position, scale and quaternion channels are handled "
                          f"(counts {counts})")
    if comp != 1:
        raise DeformError(f"compression {comp} isn't handled (deform clips use 1)")
    n_pos, n_scale, n_quat = counts[1], counts[2] - counts[1], counts[3] - counts[2]
    stride = (n_pos * 16 + n_scale * 16 + n_quat * 8 + 15) & ~15
    if o + stride * n_samples > len(c):
        raise DeformError("samples run past the end of the clip")
    samples = []
    for k in range(n_samples):
        q = o + k * stride
        vals = []
        for i in range(n):
            if i < counts[2]:
                vals.append(list(struct.unpack_from('>3f', c, q))); q += 16
            else:
                vals.append([max(v / 32767.0, -1.0) for v in struct.unpack_from('>4h', c, q)])
                q += 8
        samples.append(vals)
    kinds = ['pos'] * n_pos + ['scale'] * n_scale + ['quat'] * n_quat
    return dict(names=names, kinds=kinds, samples=samples), o + stride * n_samples


def _find_block(c, start):
    o = start
    while o + 8 < len(c):
        n = _u32(c, o)
        if 1 <= n <= 1024:
            ln = _u32(c, o + 4)
            if 4 < ln < 64 and o + 8 + ln <= len(c):
                s = c[o + 8:o + 8 + ln]
                if s.endswith((b'.pos', b'.scale', b'.quat', b'.rotz')) \
                        and all(32 < x < 127 for x in s):
                    return o
        o += 1
    return None


def _parse_charbone(data):
    """(parent stem, 12 floats: row-vector 3x3 rotation (scale baked in) + position).
    CharBone: 17-byte object header, u32 Trans revision, the local matrix, the world
    matrix, then constraint data and the parent's name."""
    m = list(struct.unpack_from('>12f', data, 21))
    parent = ''
    for k in range(117, len(data) - 4):
        ln = _u32(data, k)
        if 0 < ln < 96 and k + 4 + ln <= len(data):
            raw = data[k + 4:k + 4 + ln]
            if all(32 < x < 127 for x in raw):
                parent = raw.decode('latin-1')
                break
    return _stem(parent), m


def _stem(name):
    for ext in ('.mesh', '.trans'):
        if name.endswith(ext):
            return name[:-len(ext)]
    return name


def read_deform_milo(filepath):
    """{'full': block with 5 samples, 'one': block with 1 sample or None,
    'hierarchy': {bone stem: [parent stem, 12 floats]}}."""
    from .rb2_creator_textures import walk_milo
    with open(filepath, 'rb') as f:
        body = read_milo_container_body(f.read())
    objs, _metas = walk_milo(body)
    clip = None
    hierarchy = {}
    for _d, t, n, a, z in objs:
        if t == 'CharClipSamples' and (clip is None or n == 'deform'):
            clip = body[a:z]
        elif t == 'CharBone':
            try:
                parent, m = _parse_charbone(body[a:z])
                hierarchy[_stem(n)] = [parent, m]
            except struct.error:
                pass
    if clip is None:
        raise DeformError("no CharClipSamples 'deform' clip in this milo")
    blocks, o = [], 0
    while True:
        p = _find_block(clip, o)
        if p is None:
            break
        blk, o = _read_block(clip, p)
        blocks.append(blk)
    full = next((b for b in blocks if len(b['samples']) == 5), None)
    if full is None:
        raise DeformError(f"expected a 5-sample block (the five body presets), found "
                          f"{[len(b['samples']) for b in blocks]}")
    one = next((b for b in blocks if len(b['samples']) == 1), None)
    if not hierarchy:
        raise DeformError("this deform milo has no CharBone entries (the game's skeleton "
                          "hierarchy), which placing the bones needs")
    return dict(full=full, one=one, hierarchy=hierarchy)


# ---------------------------------------------------------------------------------------
# Blending
# ---------------------------------------------------------------------------------------

# Grid node (height row 0..2, weight column 0..2) -> weights over the 5 samples.
_N, _TF, _SF, _TS, _SS = range(5)


def _node(*pairs):
    w = [0.0] * 5
    for idx, f in pairs:
        w[idx] += f
    return w


_GRID = (
    (_node((_SS, 1)), _node((_SS, .5), (_SF, .5)), _node((_SF, 1))),         # short
    (_node((_SS, .5), (_TS, .5)), _node((_N, 1)), _node((_SF, .5), (_TF, .5))),  # mid
    (_node((_TS, 1)), _node((_TS, .5), (_TF, .5)), _node((_TF, 1))),         # tall
)


def snap(v):
    return min(SLIDER_STEPS, key=lambda s: abs(s - v))


def sample_weights(height, weight):
    """Weights over the five samples for a height/weight in 0..1."""
    gy = max(0.0, min(1.0, height)) * 2.0
    gx = max(0.0, min(1.0, weight)) * 2.0
    i0, j0 = min(int(gy), 1), min(int(gx), 1)
    fy, fx = gy - i0, gx - j0
    out = [0.0] * 5
    for di, wy in ((0, 1 - fy), (1, fy)):
        for dj, wx in ((0, 1 - fx), (1, fx)):
            f = wy * wx
            if f:
                for k, v in enumerate(_GRID[i0 + di][j0 + dj]):
                    out[k] += f * v
    return out


def blend_block(block, weights):
    """{channel: (kind, value)} for one block at the given sample weights."""
    out = {}
    samples = block['samples']
    single = len(samples) == 1
    for i, (name, kind) in enumerate(zip(block['names'], block['kinds'])):
        if single:
            out[name] = (kind, samples[0][i])
            continue
        if kind == 'quat':
            ref = samples[0][i]
            acc = [0.0, 0.0, 0.0, 0.0]
            for k, w in enumerate(weights):
                if not w:
                    continue
                q = samples[k][i]
                sign = 1.0 if sum(a * b for a, b in zip(q, ref)) >= 0 else -1.0
                for c in range(4):
                    acc[c] += w * sign * q[c]
            ln = math.sqrt(sum(a * a for a in acc)) or 1.0
            out[name] = (kind, [a / ln for a in acc])
        else:
            out[name] = (kind, [sum(w * samples[k][i][c] for k, w in enumerate(weights))
                                for c in range(3)])
    return out


# ---------------------------------------------------------------------------------------
# Applying to an armature
# ---------------------------------------------------------------------------------------

_DATA_KEY = "rb2_deform_data"
_GIRTH_ON_KEY = "rb2_girth_applied"
_CACHE = {}


def store_deform(arm_obj, deform, source):
    arm_obj[_DATA_KEY] = json.dumps(deform)
    arm_obj.rb2_proportions["source"] = source


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


def _mat_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def game_world(deform, channels):
    """{bone stem: 4x4 column-major-in-rows world matrix} by forward kinematics over the
    game's hierarchy: each bone's CharBone local matrix, with its position replaced by a
    .pos channel, its rotation by a .quat channel and its scale by a .scale channel."""
    hier = deform['hierarchy']
    world = {}

    def local(stem):
        _parent, m = hier[stem]
        rows = [m[0:3], m[3:6], m[6:9]]
        pos = list(m[9:12])
        p = channels.get(stem + '.pos')
        if p is not None:
            pos = list(p[1])
        q = channels.get(stem + '.quat')
        if q is not None:
            x, y, z, w = q[1]
            # A milo quaternion (x, y, z, w) is the transpose of the row-vector rows - checked
            # against exo_stomach / exo_L-breast's CharBone matrices.
            rot_col = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                       [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                       [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
            scale = [math.sqrt(sum(v * v for v in r)) for r in rows]
        else:
            norms = [math.sqrt(sum(v * v for v in r)) or 1.0 for r in rows]
            rot_col = [[rows[c][r] / norms[c] for c in range(3)] for r in range(3)]
            scale = norms                  # whatever scale the bone carries at rest
        sc = channels.get(stem + '.scale')
        if sc is not None:
            scale = list(sc[1])
        # column form: T @ R @ S
        return [[rot_col[r][0] * scale[0], rot_col[r][1] * scale[1], rot_col[r][2] * scale[2],
                 pos[r]] for r in range(3)] + [[0.0, 0.0, 0.0, 1.0]]

    def resolve(stem, depth=0):
        if stem in world:
            return world[stem]
        parent = hier[stem][0]
        loc = local(stem)
        if parent and parent in hier and parent != stem and depth < 256:
            world[stem] = _mat_mul(resolve(parent, depth + 1), loc)
        else:
            world[stem] = loc
        return world[stem]

    for stem in hier:
        resolve(stem)
    return world


HEAD_SCALE_CHANNEL = 'bone_head-scale.scale'


def _channels_for(deform, weights, include_constant=True, head_scale=True):
    channels = {}
    if deform.get('one') and include_constant:
        channels.update(blend_block(deform['one'], weights))
    channels.update(blend_block(deform['full'], weights))
    if not head_scale:
        channels.pop(HEAD_SCALE_CHANNEL, None)
    return channels


def _bone_order(arm_obj):
    out = []

    def visit(b):
        out.append(b)
        for c in b.children:
            visit(c)
    for b in arm_obj.data.bones:
        if b.parent is None:
            visit(b)
    return out


def apply_proportions(arm_obj, report=False):
    """Poses the armature for its current height/weight: every bone the game's hierarchy
    covers is placed where forward kinematics over that hierarchy puts it; the rest follow
    their Blender parents. Returns (bones placed, [deform bones the armature lacks])."""
    from mathutils import Matrix
    deform = _deform_of(arm_obj)
    if deform is None or arm_obj.type != 'ARMATURE' or arm_obj.get(BAKED_KEY):
        return 0, []
    pg = arm_obj.rb2_proportions
    h, w = pg.height, pg.weight
    if pg.snap_to_steps:
        h, w = snap(h), snap(w)
    world = game_world(deform, _channels_for(deform, sample_weights(h, w),
                                             pg.include_constant, pg.scale_head))

    lookup = build_bone_lookup(arm_obj)
    stem_of = {}
    for stem in world:
        name = lookup.get(stem)
        if name:
            stem_of[name] = stem
    missing = sorted(s for s in world if lookup.get(s) is None)

    # Where the game puts each bone at neutral, against this armature's rest pose. They
    # match (meshes are bound at neutral) unless the armature sits elsewhere - then the
    # body is moved by the change from neutral instead of placed absolutely.
    neutral = game_world(deform, _channels_for(deform, sample_weights(0.5, 0.5),
                                               pg.include_constant, pg.scale_head))
    dists = sorted((Vector([neutral[s][r][3] for r in range(3)])
                    - arm_obj.data.bones[n].matrix_local.to_translation()).length
                   for n, s in stem_of.items() if s.startswith('bone_'))
    relative = bool(dists) and dists[len(dists) // 2] > 1.0

    placed = 0
    pose_m = {}
    for bone in _bone_order(arm_obj):
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
        g = world[stem]
        pos = Vector([g[r][3] for r in range(3)])
        if relative:
            pos = rest.to_translation() + pos - Vector([neutral[stem][r][3] for r in range(3)])
        scale = Vector([math.sqrt(sum(g[r][c] ** 2 for r in range(3))) for c in range(3)])
        rest_scale = Vector([math.sqrt(sum(neutral[stem][r][c] ** 2 for r in range(3)))
                             or 1.0 for c in range(3)])
        # The Blender rest has no scale; the meshes are bound at neutral, so scale is
        # relative to the neutral body's.
        rel = Vector([scale[i] / rest_scale[i] for i in range(3)])
        rot = rest.to_3x3().normalized()
        if stem + '.quat' in deform['full']['names'] or \
                (deform.get('one') and stem + '.quat' in deform['one']['names']):
            gq = Matrix([[g[r][c] / (scale[c] or 1.0) for c in range(3)] for r in range(3)])
            nq = Matrix([[neutral[stem][r][c] / rest_scale[c] for c in range(3)]
                         for r in range(3)])
            rot = gq @ nq.inverted() @ rot
        target = Matrix.Translation(pos) @ rot.to_4x4() @ Matrix.Diagonal(rel.to_4d())
        basis = base.inverted() @ target
        loc, q, sca = basis.decompose()
        pb.location = loc
        pb.scale = sca
        if q.angle > 1e-4:
            if pb.rotation_mode == 'QUATERNION':
                pb.rotation_quaternion = q
            elif pb.rotation_mode == 'AXIS_ANGLE':
                axis, ang = q.to_axis_angle()
                pb.rotation_axis_angle = (ang, *axis)
            else:
                pb.rotation_euler = q.to_euler(pb.rotation_mode)
        pose_m[bone.name] = target
        placed += 1

    if pg.apply_girth or arm_obj.get(_GIRTH_ON_KEY):
        n_girth = apply_girth(arm_obj, deform, world, neutral, pose_m, pg.apply_girth)
        arm_obj[_GIRTH_ON_KEY] = bool(pg.apply_girth)
        if report and n_girth:
            _log(f"RB2 proportions: girth written to {n_girth} mesh(es)")

    if report:
        hier = deform['hierarchy']
        diffs = []
        for name, stem in stem_of.items():
            b = arm_obj.data.bones[name]
            bp = stem_of.get(b.parent.name, _stem(b.parent.name)) if b.parent else ''
            gp = hier[stem][0]
            if gp != bp:
                diffs.append(f"{stem} (game parent '{gp or '-'}', armature '{bp or '-'}')")
        if diffs:
            _log(f"RB2 proportions: {len(diffs)} bone(s) are parented differently in this "
                 f"armature than in the game (handled - positions come from the game's "
                 f"hierarchy): {'; '.join(diffs[:10])}{' ...' if len(diffs) > 10 else ''}")
        if relative:
            _log(f"RB2 proportions: this armature's rest pose isn't the game's neutral body "
                 f"(median {dists[len(dists) // 2]:.2f} units apart) - applying the change "
                 f"from neutral instead of absolute positions")
    return placed, missing


def conform_rest_to_neutral(context, arm_obj, deform):
    """Moves an armature's rest pose onto the game's neutral body: every bone of the
    deform's hierarchy is put where forward kinematics with the neutral sample places it
    (orientation kept), and every other bone keeps its offset from its parent.

    The outfit milos are all bound at the neutral body (strapjacket_resource's bind matrices
    sit within 0.24 units of it), and the mesh importer places a part's own bones - hair
    physics bones - relative to the armature's rest pose. A skeleton milo whose rest isn't
    the neutral body puts those bones, and the hair fitted to them, off the head. Returns
    (bones moved, median distance they moved)."""
    neutral = game_world(deform, _channels_for(deform, sample_weights(0.5, 0.5)))
    lookup = build_bone_lookup(arm_obj)
    stem_to_name = {s: lookup.get(s) for s in neutral if lookup.get(s)}
    name_to_pos = {n: Vector([neutral[s][r][3] for r in range(3)])
                   for s, n in stem_to_name.items()}

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
            if eb.name in name_to_pos:
                rot = m.to_3x3().normalized().to_4x4()
                nm = Matrix.Translation(name_to_pos[eb.name]) @ rot
                moved.append((nm.to_translation() - m.to_translation()).length)
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
    """Puts every bone the deform places back to its rest pose."""
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
    for obj in _armature_users(arm_obj):
        keys = obj.data.shape_keys
        if keys and SHAPE_KEY in keys.key_blocks:
            keys.key_blocks[SHAPE_KEY].value = 0.0
    return n


def load_and_apply(arm_obj, filepath, height=None, weight=None):
    """Reads a deform milo onto an armature and applies it. Returns (applied, unmatched)."""
    deform = read_deform_milo(filepath)
    store_deform(arm_obj, deform, filepath)
    pg = arm_obj.rb2_proportions
    if height is not None:
        pg["height"] = float(height)
    if weight is not None:
        pg["weight"] = float(weight)
    applied, unmatched = apply_proportions(arm_obj, report=True)
    _log(f"RB2 proportions: '{arm_obj.name}' <- {filepath}: height {pg.height:g}, weight "
         f"{pg.weight:g}; {applied} bone(s) placed"
         + (f", {len(unmatched)} with no matching bone ({', '.join(sorted(unmatched)[:8])}"
            f"{', ...' if len(unmatched) > 8 else ''})" if unmatched else ""))
    return applied, unmatched


# ---------------------------------------------------------------------------------------
# Girth: MeshDeform
# ---------------------------------------------------------------------------------------

_MD_KEY = "rb2_meshdeform"
SHAPE_KEY = "RB2 Proportions"
_GROUP_CACHE = {}


def parse_meshdeform(data):
    """(mesh name, {'n', 'bones', 'inv' [12 floats each], 'vi', 'bk', 'w'}) - weights kept
    sparse: vertex index, bone index, weight."""
    ln = _u32(data, 17)
    mesh = data[21:21 + ln].decode('latin-1')
    o = 21 + ln
    n, k = struct.unpack_from('>2I', data, o); o += 8
    if n > 1_000_000 or k > 1024 or o + n * k * 4 > len(data):
        raise DeformError(f"'{mesh}': implausible MeshDeform size {n} x {k}")
    weights = struct.unpack_from(f'>{n * k}f', data, o); o += n * k * 4
    vi, bk, w = [], [], []
    for i in range(n):
        row = weights[i * k:(i + 1) * k]
        for j, x in enumerate(row):
            if x > 1e-6:
                vi.append(i); bk.append(j); w.append(x)
    bones, inv = [], []
    for _ in range(k):
        ln = _u32(data, o)
        bones.append(_stem(data[o + 4:o + 4 + ln].decode('latin-1')))
        o += 4 + ln
        inv.append(list(struct.unpack_from('>12f', data, o)))
        o += 96                                   # inverse bind, then bind (unused)
    return mesh, dict(n=n, bones=bones, inv=inv, vi=vi, bk=bk, w=w)


def read_meshdeforms(filepath):
    """{mesh name: MeshDeform dict} for every MeshDeform in a milo."""
    from .rb2_creator_textures import walk_milo
    with open(filepath, 'rb') as f:
        body = read_milo_container_body(f.read())
    objs, _metas = walk_milo(body)
    out = {}
    for _d, t, n, a, z in objs:
        if t == 'MeshDeform':
            try:
                mesh, md = parse_meshdeform(body[a:z])
                out[mesh] = md
            except (DeformError, struct.error) as e:
                _log(f"RB2 proportions: MeshDeform '{n}' skipped: {e}")
    return out


def _pack(values, fmt):
    return base64.b64encode(struct.pack(f'<{len(values)}{fmt}', *values)).decode('ascii')


def _unpack(text, fmt):
    raw = base64.b64decode(text)
    return struct.unpack(f'<{len(raw) // 4}{fmt}', raw)


def attach_meshdeforms(objects, filepath):
    """Stores a milo's MeshDeforms on the mesh objects named after their meshes. Returns
    the names of the objects that got one."""
    mds = read_meshdeforms(filepath)
    done = []
    for obj in objects:
        if obj.type != 'MESH':
            continue
        md = mds.get(_BLENDER_DUP.sub('', obj.name)) or mds.get(obj.data.name)
        if md is None or md['n'] != len(obj.data.vertices):
            continue
        obj[_MD_KEY] = json.dumps(dict(n=md['n'], bones=md['bones'], inv=md['inv'],
                                       vi=_pack(md['vi'], 'i'), bk=_pack(md['bk'], 'i'),
                                       w=_pack(md['w'], 'f')))
        _GROUP_CACHE.pop(obj.name, None)
        done.append(obj.name)
    return done


def _meshdeform_of(obj):
    raw = obj.get(_MD_KEY)
    if not raw:
        return None
    hit = _CACHE.get(('md', obj.name))
    if hit and hit[0] == raw:
        return hit[1]
    import numpy as np
    d = json.loads(raw)
    md = dict(n=d['n'], bones=d['bones'],
              inv=np.array([[[m[0], m[3], m[6], m[9]], [m[1], m[4], m[7], m[10]],
                             [m[2], m[5], m[8], m[11]], [0.0, 0.0, 0.0, 1.0]]
                            for m in d['inv']]),
              vi=np.array(_unpack(d['vi'], 'i'), dtype=np.int64),
              bk=np.array(_unpack(d['bk'], 'i'), dtype=np.int64),
              w=np.array(_unpack(d['w'], 'f')))
    _CACHE[('md', obj.name)] = (raw, md)
    return md


def _group_weights(obj, bone_names):
    """(N x J weights, [bone names]) of the object's vertex groups that name bones."""
    import numpy as np
    me = obj.data
    key = (obj.name, len(me.vertices), len(obj.vertex_groups))
    hit = _GROUP_CACHE.get(obj.name)
    if hit and hit[0] == key:
        return hit[1], hit[2]
    gi = {vg.index: vg.name for vg in obj.vertex_groups if vg.name in bone_names}
    cols = sorted(set(gi.values()))
    col_of = {n: i for i, n in enumerate(cols)}
    u = np.zeros((len(me.vertices), len(cols)))
    for v in me.vertices:
        for g in v.groups:
            name = gi.get(g.group)
            if name is not None:
                u[v.index, col_of[name]] += g.weight
    _GROUP_CACHE[obj.name] = (key, u, cols)
    return u, cols


def _armature_users(arm_obj):
    out = []
    for obj in bpy.data.objects:
        if obj.type != 'MESH' or not obj.get(_MD_KEY):
            continue
        if any(m.type == 'ARMATURE' and m.object is arm_obj for m in obj.modifiers):
            out.append(obj)
    return out


def apply_girth(arm_obj, deform, world, neutral, pose_m, enabled=True):
    """Writes each MeshDeform mesh's 'RB2 Proportions' shape key. Returns the count."""
    import numpy as np
    from mathutils import Matrix
    users = _armature_users(arm_obj)
    if not users:
        return 0
    bones = arm_obj.data.bones
    blender_def = {}
    for b in bones:
        pm = pose_m.get(b.name)
        if pm is not None:
            blender_def[b.name] = np.array(pm @ b.matrix_local.inverted())

    done = 0
    for obj in users:
        md = _meshdeform_of(obj)
        me = obj.data
        if md is None or md['n'] != len(me.vertices):
            continue
        keys = me.shape_keys
        if not enabled:
            if keys and SHAPE_KEY in keys.key_blocks:
                keys.key_blocks[SHAPE_KEY].value = 0.0
            continue
        n = md['n']
        to_arm = np.array(arm_obj.matrix_world.inverted() @ obj.matrix_world)
        basis = np.empty(n * 3)
        src = keys.reference_key.data if keys else me.vertices
        src.foreach_get('co', basis)
        v = np.c_[basis.reshape(n, 3), np.ones(n)] @ to_arm.T          # armature space

        # The game's re-skinning through the deformed exo bones that are part of the body
        # deform. The rest of each vertex's MeshDeform weight (exo_head, face bones) isn't
        # reshaped by the body sliders, so that share follows the armature as it is.
        body = np.zeros(n)
        moved = np.zeros((n, 4))
        for k, stem in enumerate(md['bones']):
            if stem not in world or stem not in neutral:
                continue
            sel = md['bk'] == k
            if not sel.any():
                continue
            vi, w = md['vi'][sel], md['w'][sel]
            m = np.eye(4) + (np.array(world[stem]) - np.array(neutral[stem])) @ md['inv'][k]
            moved[vi] += w[:, None] * (v[vi] @ m.T)
            body[vi] += w
        if not (body > 1e-6).any():
            if keys and SHAPE_KEY in keys.key_blocks:
                keys.key_blocks[SHAPE_KEY].value = 0.0
            continue

        # Displayed = armature(key). Wanted = moved (the body share, absolute) + the rest of
        # the weight carried by the armature: key = blend^-1 @ moved + (1 - body) * v.
        u, cols = _group_weights(obj, blender_def)
        if cols:
            mats = np.stack([blender_def[c] for c in cols])               # J x 4 x 4
            tot = u.sum(1)
            blend = np.einsum('nj,jab->nab', u, mats)
            free = tot <= 1e-8
            blend[~free] /= tot[~free, None, None]
            blend[free] = np.eye(4)
            key_co = np.einsum('nab,nb->na', np.linalg.inv(blend), moved)
        else:
            key_co = moved.copy()
        key_co += (1.0 - body)[:, None] * v
        key_co[:, 3] = 1.0
        key_co = key_co @ np.linalg.inv(to_arm).T

        if keys is None:
            obj.shape_key_add(name="Basis", from_mix=False)
            keys = me.shape_keys
        kb = keys.key_blocks.get(SHAPE_KEY) or obj.shape_key_add(name=SHAPE_KEY,
                                                                 from_mix=False)
        kb.data.foreach_set('co', key_co[:, :3].ravel())
        kb.value = 1.0
        me.update()
        done += 1
    return done


# ---------------------------------------------------------------------------------------
# Baking: proportions into the rest pose and the meshes, for export
# ---------------------------------------------------------------------------------------
#
# A proportioned character is four live layers on the imported model: the bones posed by
# the sliders, the 'RB2 Proportions' (girth) and 'RB2 Cuff' shape keys, and the Armature
# modifier putting them together. Exporters understand none of it, so the bake makes the
# current look the plain rest state:
#
#   1. each mesh deformed by the armature is evaluated - pose, shape keys, Armature
#      modifier; anything after the Armature modifier (shell fur) switched off for it - and
#      that shape, with its normals, becomes the mesh's base geometry; the RB2 shape keys go
#      (any others are carried over by their offsets);
#   2. the pose becomes the armature's rest pose ("Apply Pose as Rest Pose"); bone scale
#      (girth helpers, the head scale) ends up in positions and lengths, which matches the
#      captured meshes;
#   3. objects hanging from bones keep their place;
#   4. the armature is marked baked, so the sliders can't stack proportions on top.
#
# Vertex groups and the Armature modifier stay, so the character still poses and animates
# from its new rest - with one caveat: RB2 animations that set bone POSITIONS are authored
# for the neutral body and pull those bones back toward it.

BAKED_KEY = "rb2_proportions_baked"
RB2_SHAPE_KEYS = (SHAPE_KEY, "RB2 Cuff")


def _armature_meshes(arm_obj):
    return [o for o in bpy.data.objects if o.type == 'MESH' and any(
        m.type == 'ARMATURE' and m.object is arm_obj for m in o.modifiers)]


def _duplicate_character(context, arm_obj, meshes, name):
    """A hidden, unbaked copy of the armature and its meshes, in its own collection."""
    coll = bpy.data.collections.new(name)
    context.scene.collection.children.link(coll)
    arm2 = arm_obj.copy()
    arm2.data = arm_obj.data.copy()
    coll.objects.link(arm2)
    copies = {arm_obj: arm2}
    for o in meshes + [c for c in arm_obj.children if c not in meshes]:
        o2 = o.copy()
        if o.data is not None:
            o2.data = o.data.copy()
        for m in o2.modifiers:
            if m.type == 'ARMATURE' and m.object is arm_obj:
                m.object = arm2
        coll.objects.link(o2)
        copies[o] = o2
    for o, o2 in copies.items():
        if o.parent in copies:
            o2.parent = copies[o.parent]
    layer = context.view_layer.layer_collection.children.get(coll.name)
    if layer is not None:
        layer.exclude = True                   # out of the way, and out of exports
    return coll


def _capture_mesh(context, obj):
    """Writes the object's evaluated (posed, shape-keyed, armature-deformed) shape and
    normals into its base mesh and drops the RB2 shape keys. Returns True if captured."""
    import numpy as np
    me = obj.data
    arm_index = max((i for i, m in enumerate(obj.modifiers) if m.type == 'ARMATURE'),
                    default=-1)
    toggled = []
    for i, m in enumerate(obj.modifiers):
        if i > arm_index and m.show_viewport:
            m.show_viewport = False            # e.g. shell fur: not part of the body
            toggled.append(m)
    # Hidden pieces (the outfit hide flags) aren't evaluated while hidden.
    was_hidden, was_vp_hidden = obj.hide_get(), obj.hide_viewport
    obj.hide_viewport = False
    try:
        obj.hide_set(False)
    except RuntimeError:
        pass
    try:
        dg = context.evaluated_depsgraph_get()
        dg.update()
        ev = obj.evaluated_get(dg)
        em = ev.to_mesh()
        try:
            n = len(me.vertices)
            if len(em.vertices) != n or len(em.loops) != len(me.loops):
                return False
            co = np.empty(n * 3, dtype=np.float64)
            em.vertices.foreach_get('co', co)
            normals = None
            try:
                normals = [tuple(cn.vector) for cn in em.corner_normals]
            except AttributeError:
                try:
                    em.calc_normals_split()
                    normals = [tuple(l.normal) for l in em.loops]
                except AttributeError:
                    normals = None
        finally:
            ev.to_mesh_clear()
    finally:
        for m in toggled:
            m.show_viewport = True
        obj.hide_viewport = was_vp_hidden
        try:
            obj.hide_set(was_hidden)
        except RuntimeError:
            pass

    # other shape keys keep their offsets from the old basis
    keep = []
    keys = me.shape_keys
    if keys is not None:
        basis = np.empty(len(me.vertices) * 3)
        keys.reference_key.data.foreach_get('co', basis)
        for kb in keys.key_blocks:
            if kb == keys.reference_key or kb.name in RB2_SHAPE_KEYS:
                continue
            d = np.empty(len(me.vertices) * 3)
            kb.data.foreach_get('co', d)
            keep.append((kb.name, kb.value, d - basis))
        obj.shape_key_clear()
    me.vertices.foreach_set('co', co)
    if keep:
        obj.shape_key_add(name="Basis", from_mix=False)
        for name, value, delta in keep:
            kb = obj.shape_key_add(name=name, from_mix=False)
            kb.data.foreach_set('co', co + delta)
            kb.value = value
    me.update()
    if normals is not None and len(normals) == len(me.loops):
        try:
            me.normals_split_custom_set(normals)
        except (RuntimeError, TypeError, ValueError):
            pass
    for k in ("rb2_meshdeform",):
        if k in obj:
            del obj[k]                         # girth is in the mesh now
    return True


def bake_proportions(context, arm_obj, keep_copy=True, remove_hidden=False):
    """Bakes the armature's current proportions into its rest pose and its meshes.
    Returns (meshes baked, [skipped names], copy collection or None)."""
    meshes = _armature_meshes(arm_obj)
    copy = None
    if keep_copy:
        copy = _duplicate_character(context, arm_obj, meshes,
                                    f"{arm_obj.name} (unbaked)")
    if remove_hidden:
        for o in [o for o in meshes if o.get("rb2_hidden_by_flags")]:
            meshes.remove(o)
            bpy.data.objects.remove(o, do_unlink=True)

    baked, skipped = 0, []
    for o in meshes:
        if _capture_mesh(context, o):
            baked += 1
        else:
            skipped.append(o.name)

    # objects hanging from bones (no Armature modifier) keep their place
    bone_children = [(o, o.matrix_world.copy()) for o in bpy.data.objects
                     if o.parent is arm_obj and o.parent_type == 'BONE']

    prev_active = context.view_layer.objects.active
    prev_mode = arm_obj.mode
    context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='POSE')
    try:
        bpy.ops.pose.armature_apply(selected=False)
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')
        context.view_layer.objects.active = prev_active
    for o, mw in bone_children:
        o.matrix_world = mw

    pg = arm_obj.rb2_proportions
    arm_obj[BAKED_KEY] = f"height {pg.height:g}, weight {pg.weight:g}"
    _log(f"RB2 proportions: baked '{arm_obj.name}' at {arm_obj[BAKED_KEY]} - {baked} mesh(es)"
         + (f"; skipped {', '.join(skipped)}" if skipped else "")
         + (f"; unbaked copy in collection '{copy.name}'" if copy else ""))
    return baked, skipped, copy


# ---------------------------------------------------------------------------------------
# Properties, operators, panel
# ---------------------------------------------------------------------------------------

def _update(self, context):
    obj = self.id_data
    if obj is not None and obj.type == 'ARMATURE':
        apply_proportions(obj)


class RB2ProportionSettings(bpy.types.PropertyGroup):
    height: FloatProperty(name="Height", min=0.0, max=1.0, default=0.5, update=_update,
                          description="The character creator's height slider (0 = "
                                      "shortest, 1 = tallest)")
    weight: FloatProperty(name="Weight", min=0.0, max=1.0, default=0.5, update=_update,
                          description="The character creator's weight slider (0 = "
                                      "skinniest, 1 = heaviest)")
    snap_to_steps: BoolProperty(
        name="Snap to Slider Steps", default=False, update=_update,
        description="Round to the creator's five slider steps (0, 0.25, 0.5, 0.75, 1). "
                    "Off by default: some prefab characters use values in between")
    include_constant: BoolProperty(
        name="Include Constant Channels", default=True, update=_update,
        description="Also apply the deform clip's single-sample channels (fingers and "
                    "other bones that don't change with body shape)")
    scale_head: BoolProperty(
        name="Scale Head", default=True, update=_update,
        description="Apply the deform's head scale (bone_head-scale: 0.92 skinny to 1.08 "
                    "heavy for the male). The face's shape is never changed - only RB3 "
                    "added face sculpting")
    apply_girth: BoolProperty(
        name="Apply Girth", default=True, update=_update,
        description="Widen and narrow the meshes through their MeshDeform data (the exo "
                    "bones), written to an 'RB2 Proportions' shape key. Meshes need their "
                    "MeshDeform loaded (the character importer does this; otherwise use "
                    "Load Girth From Milo)")
    source: StringProperty(name="Deform Milo")


class POSE_OT_rb2_load_deform(bpy.types.Operator, ImportHelper):
    """Load a Rock Band 2 deform milo (male_vocal_deform, female_vocal_deform) onto this
    armature and pose it for its height and weight"""
    bl_idname = "pose.rb2_load_deform"
    bl_label = "Load RB2 Deform Milo"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".milo_xbox"
    filter_glob: StringProperty(default="*.milo_xbox;*.milo_ps3;*.milo", options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == 'ARMATURE'

    def execute(self, context):
        arm = context.object
        try:
            applied, unmatched = load_and_apply(arm, self.filepath)
        except Exception as e:
            self.report({'ERROR'}, f"Couldn't load deform: {e}")
            return {'CANCELLED'}
        msg = f"{applied} bone(s) placed"
        if unmatched:
            msg += f"; {len(unmatched)} deform bone(s) not in this armature (see the log)"
        self.report({'WARNING'} if unmatched and not applied else {'INFO'}, msg)
        return {'FINISHED'}


class POSE_OT_rb2_load_girth(bpy.types.Operator, ImportHelper):
    """Read the MeshDeform (girth) data from an outfit milo - the one holding the meshes,
    e.g. strapjacket_resource or torso naked - onto this armature's meshes of the same
    names"""
    bl_idname = "pose.rb2_load_girth"
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


class POSE_OT_rb2_bake_proportions(bpy.types.Operator):
    """Bake the current height, weight and girth into the armature's rest pose and the
    meshes, so the character exports as a plain rigged model (FBX, glTF ...)"""
    bl_idname = "pose.rb2_bake_proportions"
    bl_label = "Bake Body and Armature Locations"
    bl_options = {'REGISTER', 'UNDO'}

    keep_copy: BoolProperty(
        name="Keep an Unbaked Copy", default=True,
        description="Duplicate the armature and its meshes first, into a collection left "
                    "out of the view layer, so the sliders can still be used on the copy")
    remove_hidden: BoolProperty(
        name="Remove Hidden Pieces", default=False,
        description="Delete meshes hidden by the outfit's hide flags (covered by another "
                    "part), so they aren't exported")
    bake_textures: BoolProperty(
        name="Also Bake Diffuse Textures", default=False,
        description="Run 'Bake All Character Diffuse Textures' too, for a character ready to "
                    "export in one step")

    @classmethod
    def poll(cls, context):
        o = context.object
        return o is not None and o.type == 'ARMATURE' and not o.get(BAKED_KEY)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=380)

    def draw(self, context):
        col = self.layout.column()
        pg = context.object.rb2_proportions
        col.label(text=f"Baking height {pg.height:g}, weight {pg.weight:g}", icon='INFO')
        col.prop(self, "keep_copy")
        col.prop(self, "remove_hidden")
        col.prop(self, "bake_textures")
        if not self.keep_copy:
            col.label(text="Without a copy the bake can only be undone with Ctrl+Z",
                      icon='ERROR')

    def execute(self, context):
        arm = context.object
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        try:
            baked, skipped, copy = bake_proportions(context, arm, self.keep_copy,
                                                    self.remove_hidden)
        except Exception as e:
            self.report({'ERROR'}, f"Bake failed: {e}")
            return {'CANCELLED'}
        if self.bake_textures:
            try:
                bpy.ops.material.rb2cc_bake_all('EXEC_DEFAULT')
            except Exception as e:
                self.report({'WARNING'}, f"Proportions baked; texture bake failed: {e}")
                return {'FINISHED'}
        msg = f"Baked {baked} mesh(es) and the rest pose"
        if skipped:
            msg += f"; {len(skipped)} skipped (see the log)"
        if copy is not None:
            msg += f"; unbaked copy in '{copy.name}'"
        self.report({'WARNING'} if skipped else {'INFO'}, msg)
        return {'FINISHED'}


class POSE_OT_rb2_proportion_preset(bpy.types.Operator):
    """Set height and weight to one of the character creator's body presets"""
    bl_idname = "pose.rb2_proportion_preset"
    bl_label = "RB2 Body Preset"
    bl_options = {'REGISTER', 'UNDO'}

    height: FloatProperty()
    weight: FloatProperty()

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == 'ARMATURE'

    def execute(self, context):
        pg = context.object.rb2_proportions
        pg["height"] = self.height
        pg["weight"] = self.weight
        apply_proportions(context.object)
        return {'FINISHED'}


class POSE_OT_rb2_clear_proportions(bpy.types.Operator):
    """Return the deform bones to the skeleton's rest pose (the loaded deform is kept)"""
    bl_idname = "pose.rb2_clear_proportions"
    bl_label = "Clear Proportions"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == 'ARMATURE'

    def execute(self, context):
        n = clear_proportions(context.object)
        self.report({'INFO'}, f"{n} bone channel(s) reset to rest")
        return {'FINISHED'}


class DATA_PT_rb2_proportions(bpy.types.Panel):
    bl_label = "Rock Band 2 Proportions"
    bl_idname = "DATA_PT_rb2_proportions"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "data"

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == 'ARMATURE'

    def draw(self, context):
        layout = self.layout
        arm = context.object
        pg = arm.rb2_proportions
        if arm.get(BAKED_KEY):
            box = layout.box()
            box.label(text=f"Baked at {arm[BAKED_KEY]}", icon='CHECKMARK')
            box.label(text="The proportions are the rest pose now - the sliders are off. "
                           "Use the unbaked copy to change them.")
            return
        loaded = _deform_of(arm) is not None
        layout.operator(POSE_OT_rb2_load_deform.bl_idname,
                        text="Load Deform Milo" if not loaded else "Load Another Deform Milo",
                        icon='ARMATURE_DATA')
        if loaded:
            layout.label(text=bpy.path.basename(pg.source) or "(deform loaded)", icon='FILE')
        else:
            layout.label(text="Not in use - load a <gender>_<instrument>_deform milo",
                         icon='INFO')

        col = layout.column()
        col.enabled = loaded
        col.prop(pg, "weight", slider=True)
        col.prop(pg, "height", slider=True)
        if pg.snap_to_steps:
            col.label(text=f"Using height {snap(pg.height):g}, weight {snap(pg.weight):g}")
        col.prop(pg, "snap_to_steps")
        col.prop(pg, "include_constant")
        col.prop(pg, "scale_head")
        row = col.row(align=True)
        row.prop(pg, "apply_girth")
        row.operator(POSE_OT_rb2_load_girth.bl_idname, text="", icon='MOD_MESHDEFORM')
        n_md = len(_armature_users(arm))
        col.label(text=f"Girth data on {n_md} mesh(es)" if n_md else
                  "No girth data yet - use the button beside Apply Girth",
                  icon='INFO' if not n_md else 'CHECKMARK')

        box = col.box()
        box.label(text="Creator Presets")
        grid = box.grid_flow(columns=2, even_columns=True, align=True)
        for label, h, w in PRESETS:
            op = grid.operator(POSE_OT_rb2_proportion_preset.bl_idname, text=label)
            op.height, op.weight = h, w
        col.operator(POSE_OT_rb2_clear_proportions.bl_idname, icon='LOOP_BACK')
        layout.separator()
        layout.operator(POSE_OT_rb2_bake_proportions.bl_idname, icon='EXPORT')


_classes = (
    RB2ProportionSettings,
    POSE_OT_rb2_load_deform,
    POSE_OT_rb2_load_girth,
    POSE_OT_rb2_proportion_preset,
    POSE_OT_rb2_clear_proportions,
    POSE_OT_rb2_bake_proportions,
    DATA_PT_rb2_proportions,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.rb2_proportions = PointerProperty(type=RB2ProportionSettings)


def unregister():
    del bpy.types.Object.rb2_proportions
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
