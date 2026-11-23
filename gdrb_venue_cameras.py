"""
Green Day: Rock Band venue cameras - the camera director's shots.

Used by gdrb_venue_importer when "Import Cameras" is on. Every BandCamShot in the venue's milos
becomes a Blender camera, keyed through its keyframes; the venue's mocap camera tracks
(americanidiot_mocams) become animated cameras through cam_importer.

===========================================================================================
BANDCAMSHOT - verified on all 1328 shots of americanidiot, dookie and twentyfirst
===========================================================================================
BandCamShot revision 28 over CamShot revision 49, then the Object header (a properties tree,
e.g. crowd_density, of varying length), a few shot fields, and the keyframe count. The count
is found structurally - the first offset from which that many keyframes parse end to end -
rather than by walking the properties tree. Each keyframe:

    f32 duration, f32 blend, f32 blend ease     frames at 30 fps
    f32 field of view                           vertical, radians
    4x3 matrix                                  the camera pose (engine: looks down +Y, +Z up)
    f32 x2 screen offset, f32 x4 focus/blur
    u32 n, n x look-at target                   (tag, character symbol, object symbol)
    focal target, PARENT                        same form; character '' = none
    u8, f32 x5, u8

Durations are frames at 30 fps: Mocap_BJCatGuitarsolo1's 387 is exactly its camera track's
length. Blend leads into the NEXT keyframe: earlier keyframes mostly blend (652 of 1011),
final ones mostly don't (499 of 585).

After the keyframes each shot places the band for itself: (character symbol, a byte, 4x3
matrix) per member. Billie Joe's usual place is his mic, (-77, -304, 119), facing the crowd
(rotated 180 degrees about Z). Placements shared by two members in one shot ((-10, -218, 119)
for Billie Joe and Mike at once) or at the origin are stand-ins, not real positions.

===========================================================================================
PLACEMENT
===========================================================================================
A keyframe with no parent is in venue world space: BJ_Back's three keyframes aim at Billie
Joe's neck to within 3 degrees.

A parented keyframe (40% of them) is relative to a band member's bone or collision volume -
'billiejoe' / 'chest.coll', 'mikedirnt' / 'bone_guitar.mesh'. Its axes are world-aligned, not
turned with the character: read that way, 46 of 50 'Back' keyframes sit behind the performer
and 140 of 147 'Face'/'Front' ones in front; turned with him, the reverse. Those cameras are
parented to an Empty per bone, at the member's stage position plus the bone's height. The
heights are anatomical estimates anchored on the one the data pins down - bone_neck at 60
units, measured from 22 centred world-space shots (interquartile 57-62) - and the Empties
can be moved or snapped to an imported skeleton; the cameras follow.

A shot whose keyframes have different parents (70 of 1328) is baked to world space.

Shots store band positions in one shared stage space, identical in every venue. Only
americanidiot's stage actually sits there; dookie's and twentyfirst's band stands about 360-400
units further along Y and 120 lower. That offset is solved from each venue's own world-space
cameras (estimate_band_offset) and put on a 'Band Origin' Empty that every member hangs from.

Mocap shots carry no pose of their own (identity matrix): they play a camera track, and only
the tracks some shot plays are imported (51 of americanidiot's 60).

Mocap tracks are relative to the performer, not the venue (mk_bj_cat_guitarsolo1_camA stays
within 40 units of the origin at waist height), and the performer moves through them, so
they're imported under one Empty at the origin to be moved into place.
"""

import math
import re
import struct
from collections import Counter

import bpy

from . import mesh_importer as mi
from .utilities import _log


FRAME_RATE = 30.0
_SENSOR_HEIGHT = 24.0
_MAX_FOCAL = 5000.0
_DISPLAY_SIZE = 10.0

# Height of each parent bone/collision volume above the character's feet, in game units.
# bone_neck is measured (see the module notes); the rest follow GDRB proportions - the head
# mesh spans Z 61-71 (mesh_importer) - and are estimates.
BONE_HEIGHTS = {
    '': 0.0,
    'bone_pelvis.mesh': 38.0,
    'bone_guitar.mesh': 38.0,
    'instrument0': 38.0,
    'bone_R-hand.mesh': 38.0,
    'bone_L-hand.mesh': 38.0,
    'bone_nut.mesh': 45.0,
    'chest.coll': 50.0,
    'back.coll': 50.0,
    'bone_spine3.mesh': 50.0,
    'bone_neck.mesh': 60.0,
    'neck.coll': 60.0,
    'mic.mesh': 60.0,
    'bone_head.mesh': 63.0,
    'face.coll': 65.0,
    'ear.coll': 65.0,
    'head.coll': 66.0,
    'forehead.coll': 68.0,
    'bone_drumbase.mesh': 0.0,
}
DEFAULT_BONE_HEIGHT = 50.0

BAND_NAMES = {'billiejoe': "Billie Joe", 'mikedirnt': "Mike Dirnt", 'trecool': "Tre Cool"}


class ShotParseError(Exception):
    pass


# ---------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------

def _u32(b, o):
    return struct.unpack_from('>I', b, o)[0]


def _sym(b, o, max_len=256):
    n = _u32(b, o)
    if n > max_len or o + 4 + n > len(b):
        raise ShotParseError(f"bad symbol at {o}")
    s = b[o + 4:o + 4 + n]
    if not all(32 <= c < 127 for c in s):
        raise ShotParseError(f"bad symbol at {o}")
    return s.decode('latin-1'), o + 4 + n


def _ref(b, o):
    if _u32(b, o) > 255:
        raise ShotParseError(f"bad reference at {o}")
    char, o = _sym(b, o + 4)
    obj, o = _sym(b, o)
    return (char, obj), o


def _matrix_ok(m):
    if not all(math.isfinite(x) and abs(x) < 1e5 for x in m):
        return False
    return all(1e-3 < math.sqrt(m[r * 3] ** 2 + m[r * 3 + 1] ** 2 + m[r * 3 + 2] ** 2) < 1e3
               for r in range(3))


def _keyframe(b, o):
    pre = struct.unpack_from('>3f', b, o)
    fov = struct.unpack_from('>f', b, o + 12)[0]
    m = struct.unpack_from('>12f', b, o + 16)
    if not (0.0 <= fov < 3.15 and _matrix_ok(m)
            and all(math.isfinite(x) and abs(x) < 1e7 for x in pre)):
        raise ShotParseError(f"no keyframe at {o}")
    q = o + 64
    screen = struct.unpack_from('>2f', b, q)
    q += 24                                         # screen offset, focus/blur
    n = _u32(b, q)
    q += 4
    if n > 16:
        raise ShotParseError("implausible target count")
    targets = []
    for _ in range(n):
        t, q = _ref(b, q)
        targets.append(t)
    focal, q = _ref(b, q)
    parent, q = _ref(b, q)
    if b[q] not in (0, 1) or b[q + 21] not in (0, 1):
        raise ShotParseError("keyframe tail")
    q += 22
    return dict(duration=pre[0], blend=pre[1], ease=pre[2], fov=fov, m=m, screen=screen,
                targets=targets, focal=focal, parent=parent), q


def _keyframes_at(b, c):
    n = _u32(b, c)
    if not 1 <= n <= 64:
        return None
    q = c + 4
    frames = []
    try:
        for _ in range(n):
            f, q = _keyframe(b, q)
            frames.append(f)
    except (ShotParseError, struct.error, IndexError):
        return None
    return frames, q


def _band_placements(b, o):
    """(character symbol, byte, 4x3 matrix) records after the keyframes."""
    out = {}
    for p in range(o, len(b) - 57):
        try:
            name, q = _sym(b, p, 32)
        except (ShotParseError, struct.error):
            continue
        if not name or not name.isidentifier() or name in out or b[q] not in (0, 1):
            continue
        m = struct.unpack_from('>12f', b, q + 1)
        if _matrix_ok(m):
            out[name] = m
    return out


def parse_band_cam_shot(chunk, name):
    """dict(name, frames, band) for one BandCamShot, or raises ShotParseError."""
    for c in range(8, min(len(chunk) - 4, 4096)):
        r = _keyframes_at(chunk, c)
        if r:
            frames, end = r
            return dict(name=name, frames=frames, band=_band_placements(chunk, end),
                        tracks=sorted({m.decode('latin-1') for m in
                                       re.findall(rb'[A-Za-z0-9_]+\.(?:tnm|cnm)', chunk)}))
    raise ShotParseError("no keyframes found")


# ---------------------------------------------------------------------------------------
# Band positions
# ---------------------------------------------------------------------------------------

def _is_standin(shot, char):
    m = shot['band'].get(char)
    if m is None:
        return True
    pos = tuple(round(x) for x in m[9:])
    if abs(pos[0]) + abs(pos[1]) + abs(pos[2]) < 60:
        return True
    others = [tuple(round(x) for x in mm[9:]) for c, mm in shot['band'].items() if c != char]
    return pos in others


def usual_positions(shots):
    """Each band member's most common real stage position across the venue's shots."""
    counts = {}
    exact = {}
    for s in shots:
        for char, m in s['band'].items():
            if _is_standin(s, char):
                continue
            key = tuple(round(x) for x in m[9:])
            counts.setdefault(char, Counter())[key] += 1
            exact[(char, key)] = tuple(m[9:])
    return {c: exact[(c, cnt.most_common(1)[0][0])] for c, cnt in counts.items()}


def member_position(shot, char, usual):
    if not _is_standin(shot, char):
        return tuple(shot['band'][char][9:])
    return usual.get(char)


# Shots place the band in a shared stage space: every venue stores Billie Joe at
# (-77, -303, 119), but dookie's and twentyfirst's stages aren't there - twentyfirst's mic
# stand is at (-78, 76, 0). The venue's world-space cameras show where the band really is:
# each one aims at a member's bone, so the translation that best puts the stored positions on
# their sight lines is the band's offset in that venue. Solved by least squares over centred,
# single-target keyframes. americanidiot comes out at about zero (its stage IS that space;
# applying the noise would worsen its 1.5-degree median aim, so it's left alone); dookie
# (-6, 359, -124) takes the median aim error from 42 to 8 degrees, twentyfirst (4, 397, -135)
# from 63 to 7 - consistent with its mic stand to within about 20 units, the rest being
# the shots' deliberately off-centre framing.
_MIN_OFFSET_SAMPLES = 5


def estimate_band_offset(shots, usual):
    """(offset or None, samples, median aim error before, after) in degrees."""
    import numpy as np
    rows = []
    for s in shots:
        if s['tracks']:
            continue
        for f in s['frames']:
            if f['parent'][0] or len(f['targets']) != 1:
                continue
            if abs(f['screen'][0]) > 0.2 or abs(f['screen'][1]) > 0.2:
                continue
            char, obj = f['targets'][0]
            pos = member_position(s, char, usual)
            if pos is None or obj not in BONE_HEIGHTS:
                continue
            target = np.array(pos) + np.array([0.0, 0.0, BONE_HEIGHTS[obj]])
            m = f['m']
            d = np.array(m[3:6], dtype=float)
            ln = np.linalg.norm(d)
            if ln < 1e-6:
                continue
            d /= ln
            rows.append((np.eye(3) - np.outer(d, d), np.array(m[9:], dtype=float), target))
    if len(rows) < _MIN_OFFSET_SAMPLES:
        return None, len(rows), None, None

    def median_miss(t):
        out = []
        for proj, cam, target in rows:
            to = target + t - cam
            n = np.linalg.norm(to)
            if n > 1e-6:
                out.append(math.degrees(math.asin(min(1.0, np.linalg.norm(proj @ to) / n))))
        out.sort()
        return out[len(out) // 2] if out else 0.0

    a = sum(r[0] for r in rows)
    b = sum(r[0] @ (r[1] - r[2]) for r in rows)
    try:
        t = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return None, len(rows), None, None
    before, after = median_miss(np.zeros(3)), median_miss(t)
    if np.linalg.norm(t) < 50.0 or not (before > 10.0 and after < before * 0.5):
        return None, len(rows), before, after
    return tuple(float(x) for x in t), len(rows), before, after


# ---------------------------------------------------------------------------------------
# Blender construction
# ---------------------------------------------------------------------------------------

def _rx90():
    from mathutils import Matrix
    return Matrix.Rotation(math.pi / 2, 4, 'X')


def _camera_matrix(m):
    """A keyframe's 4x3 matrix as a Blender camera's matrix (engine +Y forward, +Z up ->
    Blender -Z forward, +Y up)."""
    from .io import _milo_to_blender_matrix
    return _milo_to_blender_matrix(m) @ _rx90()


def _focal(fov):
    if not (0.0 < fov < math.pi):
        return None
    return min(_MAX_FOCAL, max(1.0, (_SENSOR_HEIGHT * 0.5) / math.tan(fov * 0.5)))


class _Anchors:
    """Empties standing in for the band: one per member position, one per bone under it."""

    def __init__(self, collection, usual, offset=None):
        from mathutils import Vector
        self.col = collection
        self.usual = usual
        self.members = {}
        self.bones = {}
        # Every member hangs from this, at the band's offset in the venue (see
        # estimate_band_offset) - moving it moves the whole band and every camera framed on
        # a member.
        self.origin = bpy.data.objects.new("Band Origin", None)
        self.origin.empty_display_type = 'CUBE'
        self.origin.empty_display_size = 10.0
        self.origin.location = Vector(offset or (0.0, 0.0, 0.0))
        self.origin["milo_note"] = ("Where the venue puts the band. Shots store band positions "
                                    "in a shared stage space; this offset was solved from the "
                                    "venue's own cameras. Move it to realign the band")
        self.origin["milo_offset_solved"] = offset is not None
        collection.objects.link(self.origin)

    def bone(self, char, obj, pos):
        from mathutils import Vector
        key = (char, tuple(round(x) for x in pos))
        member = self.members.get(key)
        if member is None:
            usual = self.usual.get(char)
            label = BAND_NAMES.get(char, char)
            if usual is None or tuple(round(x) for x in usual) != key[1]:
                label = f"{label} @ {pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f}"
            member = bpy.data.objects.new(label, None)
            member.empty_display_type = 'SINGLE_ARROW'
            member.empty_display_size = 20.0
            member.parent = self.origin
            member.location = Vector(pos)
            member["milo_character"] = char
            self.col.objects.link(member)
            self.members[key] = member
        bkey = (key, obj)
        b = self.bones.get(bkey)
        if b is None:
            h = BONE_HEIGHTS.get(obj, DEFAULT_BONE_HEIGHT)
            b = bpy.data.objects.new(f"{member.name}: {obj or 'root'}", None)
            b.empty_display_type = 'SPHERE'
            b.empty_display_size = 3.0
            b.parent = member
            b.location = Vector((0.0, 0.0, h))
            b["milo_character"] = char
            b["milo_object"] = obj
            b["milo_height_estimated"] = obj != 'bone_neck.mesh'
            self.col.objects.link(b)
            self.bones[bkey] = b
        return b


def _key_shot(obj, cam_data, shot, mats, scene_fps):
    """Keys a multi-keyframe shot: each keyframe held for its duration, then blended into the
    next over its blend time (eased when it has an ease)."""
    from .cam_importer import _new_action, _curve, _assign
    action, slot, cb = _new_action(f"SHOT_{obj.name}", 'OBJECT', obj.name)
    loc = [_curve(cb, 'location', i, "Transform") for i in range(3)]
    rot = [_curve(cb, 'rotation_quaternion', i, "Transform") for i in range(4)]
    scl = [_curve(cb, 'scale', i, "Transform") for i in range(3)]
    laction, lslot, lcb = _new_action(f"SHOTLENS_{obj.name}", 'CAMERA', cam_data.name)
    lens = _curve(lcb, 'lens', 0, "Lens")

    to_frame = lambda t: 1.0 + t / FRAME_RATE * scene_fps
    t = 0.0
    prev_q = None
    frames = shot['frames']
    for i, (f, mat) in enumerate(zip(frames, mats)):
        loc_v, q, sc = mat.decompose()
        if prev_q is not None and prev_q.dot(q) < 0.0:
            q.negate()
        prev_q = q
        last = i == len(frames) - 1
        blend = max(0.0, f['blend']) if not last else 0.0
        interp = ('CONSTANT' if blend <= 0.0 else ('SINE' if f['ease'] > 0 else 'LINEAR'))
        stops = [t]
        if f['duration'] > 0:
            stops.append(t + f['duration'])
        for j, tt in enumerate(stops):
            bf = to_frame(tt)
            ip = interp if j == len(stops) - 1 else 'CONSTANT'
            for fc, vals in ((loc, loc_v), (rot, q), (scl, sc)):
                for k, v in enumerate(vals):
                    kp = fc[k].keyframe_points.insert(bf, v, options={'FAST'})
                    kp.interpolation = ip
                    if ip == 'SINE':
                        kp.easing = 'EASE_IN_OUT'
            fl = _focal(f['fov'])
            if fl is not None:
                kp = lens.keyframe_points.insert(bf, fl, options={'FAST'})
                kp.interpolation = ip
                if ip == 'SINE':
                    kp.easing = 'EASE_IN_OUT'
        t = stops[-1] + (blend if blend > 0 else (1.0 if not last else 0.0))
    for fc in loc + rot + scl + [lens]:
        fc.update()
    _assign(obj, action, slot)
    _assign(cam_data, laction, lslot)
    return to_frame(t)


def _shot_group(shot):
    for f in shot['frames']:
        for char, _o in f['targets']:
            if char:
                return BAND_NAMES.get(char, "Other")
    return "Other"


def build_shot_cameras(context, root, shots, usual, scene_fps, cols, offset=None):
    """One camera per shot. Returns (cameras made, parented, baked, notes, last frame)."""
    from mathutils import Matrix
    anchors_col = bpy.data.collections.new("Band Anchors")
    root.children.link(anchors_col)
    anchors = _Anchors(anchors_col, usual, offset)
    made = parented = baked = 0
    notes = []
    end = 1.0
    for shot in shots:
        frames = shot['frames']
        parents = {f['parent'] for f in frames}
        single_parent = next(iter(parents)) if len(parents) == 1 else None

        def bone_for(f):
            char, obj = f['parent']
            pos = member_position(shot, char, usual)
            if pos is None:
                notes.append(f"'{shot['name']}': no stage position for '{char}', placed at "
                             f"the origin")
                pos = (0.0, 0.0, 0.0)
            return anchors.bone(char, obj, pos)

        parent_obj = None
        if single_parent and single_parent[0]:
            parent_obj = bone_for(frames[0])
            mats = [_camera_matrix(f['m']) for f in frames]
            parented += 1
        else:
            mats = []
            for f in frames:
                mat = _camera_matrix(f['m'])
                if f['parent'][0]:
                    b = bone_for(f)
                    context.view_layer.update()
                    mat = b.matrix_world @ mat
                mats.append(mat)
            if len(parents) > 1:
                baked += 1

        name = shot['name'][:-5] if shot['name'].lower().endswith('.shot') else shot['name']
        data = bpy.data.cameras.new(name)
        data.sensor_fit = 'VERTICAL'
        data.sensor_height = _SENSOR_HEIGHT
        data.display_size = _DISPLAY_SIZE
        data.clip_start = 1.0
        data.clip_end = 100000.0
        obj = bpy.data.objects.new(name, data)
        obj.rotation_mode = 'QUATERNION'
        group = _shot_group(shot)
        _collection_under(root, f"Shots/{group}", cols).objects.link(obj)
        if parent_obj is not None:
            obj.parent = parent_obj
            obj.matrix_parent_inverse = Matrix.Identity(4)
        obj.matrix_basis = mats[0]
        fl = _focal(frames[0]['fov'])
        if fl is not None:
            data.lens = fl
        if len(frames) > 1:
            end = max(end, _key_shot(obj, data, shot, mats, scene_fps))
        obj["milo_shot"] = shot['name']
        obj["milo_keyframes"] = len(frames)
        obj["milo_targets"] = ", ".join(sorted({f"{c}/{o}" if o else c
                                                 for f in frames for c, o in f['targets']
                                                 if c}))
        obj["milo_parent"] = (f"{single_parent[0]}/{single_parent[1]}"
                              if single_parent and single_parent[0] else
                              ("mixed (baked)" if len(parents) > 1 else ""))
        obj["milo_duration_frames"] = sum(max(0.0, f['duration']) + max(0.0, f['blend'])
                                          for f in frames)
        made += 1
    return made, parented, baked, notes, end


def _collection_under(root, path, cache):
    parent = root
    built = ''
    for part in path.split('/'):
        built = f"{built}/{part}" if built else part
        if built not in cache:
            col = bpy.data.collections.new(part)
            parent.children.link(col)
            cache[built] = col
        parent = cache[built]
    return parent


def build_mocap_cameras(context, root, milo, scene_fps, cols, wanted):
    """The mocap camera tracks (TransAnim + CamAnim pairs) of one milo that the venue's shots
    use (wanted: track stems), animated, under one Empty. Returns (cameras made, notes)."""
    from .cam_importer import (parse_trans_anim, parse_cam_anim, build_camera,
                               CamImportError)
    trans, fovs, notes = {}, {}, []
    for etype, ename, s, e in milo.root.entries:
        stem = ename.rsplit('.', 1)[0]
        if stem not in wanted:
            continue
        try:
            if etype == 'TransAnim':
                trans[stem] = parse_trans_anim(milo.body[s:e], ename)
            elif etype == 'CamAnim':
                fovs[stem] = parse_cam_anim(milo.body[s:e], ename)
        except (CamImportError, struct.error, IndexError) as ex:
            notes.append(f"{milo.basename}: skipped {ename}: {ex}")
    stems = sorted(set(trans) | set(fovs))
    if not stems:
        return 0, notes
    col = _collection_under(root, "Mocap", cols)
    rig = bpy.data.objects.new(f"{milo.dir_name} (move onto the performer)", None)
    rig.empty_display_type = 'PLAIN_AXES'
    rig.empty_display_size = 30.0
    rig["milo_note"] = ("Mocap cameras are relative to the performer, not the venue - move "
                        "this Empty to where he stands")
    col.objects.link(rig)
    made = 0
    for stem in stems:
        try:
            obj, _c = build_camera(context, dict(name=stem, trans=trans.get(stem),
                                                 fov=fovs.get(stem)), rig, None, scene_fps)
        except CamImportError as ex:
            notes.append(f"'{stem}': {ex}")
            continue
        for c in list(obj.users_collection):
            c.objects.unlink(obj)
        col.objects.link(obj)
        made += 1
    return made, notes


def collect_shots(milos):
    """Every BandCamShot in the given milos' root directories. Returns (shots, failures)."""
    shots, failed, seen = [], [], set()
    for milo in milos:
        for etype, ename, s, e in milo.root.entries:
            if etype != 'BandCamShot' or ename in seen:
                continue
            seen.add(ename)
            try:
                shot = parse_band_cam_shot(milo.body[s:e], ename)
            except (ShotParseError, struct.error, IndexError) as ex:
                failed.append((ename, str(ex)))
                continue
            shot['source'] = milo.basename
            shots.append(shot)
    return shots, failed


def import_venue_cameras(context, root, milos, include_mocap=True, set_scene_range=False):
    """Builds every camera of a venue under a 'Cameras' collection in root. Returns a summary
    string."""
    scene = context.scene
    scene_fps = scene.render.fps / max(scene.render.fps_base, 1e-6)
    cam_root = bpy.data.collections.new("Cameras")
    root.children.link(cam_root)
    cols = {}
    shots, failed = collect_shots(milos)
    # Mocap shots point at the camera tracks; the tracks are the cameras.
    static = [s for s in shots if not s['tracks']]
    usual = usual_positions(shots)
    for c, p in sorted(usual.items()):
        _log(f"  stage position of {BAND_NAMES.get(c, c)}: "
             f"({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})")
    offset, n, before, after = estimate_band_offset(shots, usual)
    if offset is not None:
        _log(f"  band offset in this venue: ({offset[0]:.1f}, {offset[1]:.1f}, "
             f"{offset[2]:.1f}) from {n} camera(s) - median aim error {before:.1f} -> "
             f"{after:.1f} degrees")
    elif before is not None:
        _log(f"  band stands where the shots put it (median aim error {before:.1f} degrees "
             f"over {n} camera(s))")
    made, parented, baked, notes, end = build_shot_cameras(context, cam_root, static, usual,
                                                           scene_fps, cols, offset)
    mocap = 0
    # Only tracks a shot plays: the shared milos the chain passes through carry unrelated
    # TransAnims (dookie has no mocap cameras, but world/shared/gen has a track).
    wanted = {t.rsplit('.', 1)[0] for s in shots for t in s['tracks']}
    if include_mocap and wanted:
        for milo in milos:
            if milo.has_type('TransAnim'):
                n, mnotes = build_mocap_cameras(context, cam_root, milo, scene_fps, cols,
                                                wanted)
                mocap += n
                notes += mnotes
    for n in notes[:40]:
        _log(f"  {n}")
    for nm, why in failed:
        _log(f"  SKIPPED shot {nm}: {why}")
    if set_scene_range and end > scene.frame_end:
        scene.frame_end = int(math.ceil(end))
    _log(f"  cameras: {made} shot(s) ({parented} parented to a band member, {baked} baked "
         f"to world space), {mocap} mocap track(s)"
         + (f", {len(failed)} shot(s) unreadable" if failed else ""))
    return (f"{made} shot camera(s)" + (f", {mocap} mocap" if mocap else "")
            + (f", {len(failed)} unreadable" if failed else ""))
