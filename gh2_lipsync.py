"""
Guitar Hero 2 (Xbox 360) viseme and .voc lipsync import.

GH2 is a generation apart from Rock Band: its data keeps the PS2-era little-endian byte order
even on 360, its viseme clips are packed differently, and its lipsync is a keyframed curve
file (.voc) rather than RB's 30 Hz .lipsync. It gets its own module so none of that can leak
into the RB-era importers - but it hands off to the same machinery at the end: viseme poses
become the same tagged VISEME_ Actions, and .voc weights become the same LIPSYNC_ weight
channels, so Object > Rock Band Lipsync > Bake Lipsync Preview works on GH2 unchanged.

===========================================================================================
VISEME MILO - verified on metal_viseme_xbox.milo_xbox and metal_viseme.milo_xbox
===========================================================================================
A little-endian CharClipSet, milo revision 25, holding CharBone transforms, a CharClipFilter
and two clips:

  visemes   ONE clip with 16 samples, one viseme per sample. Sample 0 is rest (jaw closed,
            lips untouched). Values are OFFSETS: positions are lip-bone displacements,
            rotations are near-identity quaternions (the jaw only hinges, 0 to 13.7 deg).
  neutral   one sample of ABSOLUTE values, the role RB3's Base clip plays. Used here to
            measure the rotation convention, exactly as Base is for RB3.

The 360 file drives the jaw plus eight lip bones; the older file (metal_viseme) only the
jaw, with near-identical jaw motion. Both read through the same code.

Each clip is a 66-byte header, then THREE CharBonesSamples block headers, then each block's
sample data in turn. A block header is: channel count, that many (name, weight) pairs, ten
cumulative per-type channel counts (positions, scales, quaternions, single-axis rotations),
compression, sample count. With compression 1, positions are three floats (12 bytes - RB3
uses 16) and quaternions four int16 over 32767, stored x, y, z, w. An empty block still
takes its full 52-byte header. Checked on all four clips: every byte consumed, every
quaternion unit length. The header length is located rather than assumed, in case other
singers' clips carry a longer one.

===========================================================================================
WHICH SAMPLE IS WHICH VISEME - from the singer's FaceFX actor (.fac)
===========================================================================================
The milo's 16 samples carry no names, and the .voc refers to visemes only by name. The link
is in the singer's FaceFX actor file, which the character's FaceFxLipSyncServo references
(metal_singer.milo -> '../metal_singer.fac'). Its header names Harmonix and "Karaoke
Revolution Vol 4" - GH2 reused the karaoke games' setup - and it holds one FxBonePoseNode per
viseme, each with a named pose for bone_jaw. Their order is the sample order:

  0 Neutral, 1 Eat, 2 Earth, 3 If, 4 Ox, 5 Oat, 6 Wet, 7 Size, 8 Church, 9 Fave,
  10 Though, 11 Told, 12 Bump, 13 New, 14 Roar, 15 Cage

Verified rather than assumed: each node's jaw pose, taken relative to the .fac's own Neutral,
was compared with the sample at the same index. The total mismatch over samples 1-15 is 8.35
degrees against the older jaw-only rig (21.2 against the 360 rig, which retunes a few jaw
values now that the lips share the work); not one of 100,000 random orderings came within
reach - their best 0.1% start at 39.4 degrees. An earlier provisional default, matched to
Rock Band 3's jaw openings, scored no better than random: GH2's poses were authored
differently, so RB3 isn't a usable yardstick for them.

The order is still a setting ("Sample Order"), recorded on every Action, and it can be read
straight from a .fac file instead - other singers have their own actor files, and only
metal_singer's has been checked.

===========================================================================================
.VOC LIPSYNC - verified on sept.voc
===========================================================================================
A little-endian keyframe file from a Harmonix PC tool ('FACE' header). It holds named
tracks: 15 visemes (Eat, If, Ox, ... - RB3's phoneme names without the _hi/_lo suffix) plus
head orientation, eye gaze, head emphasis, eyebrow raise and blink. Each track is its name,
8 zero bytes, a key count, the keys, and 8 more bytes. Each key is 18 bytes: a 2-byte field
(always 0), then time, value, and two slopes (always 0). Times are SECONDS, so no tempo is
involved. Viseme values are 0-1 weights that pulse once per syllable.

Known gaps:
  1. Interpolation between keys isn't recorded - with every slope zero it's either straight
     lines or an ease. It's an import option, linear by default.
  2. The ten non-viseme tracks are imported as voc_* channels on the armature but drive
     nothing: which GH2 bones they move, and in what units, isn't in these files.
  3. The neutral clip is used only to measure the rotation convention; visemes are applied
     to the rig's rest pose, as the RB-era importers do by default.
"""

import math
import os
import struct

import bpy
from bpy.props import StringProperty, BoolProperty, EnumProperty
from bpy_extras.io_utils import ImportHelper

from .io import read_milo_container_body
from .utilities import _log
from .viseme_importer import (
    build_bone_lookup, detect_rotation_convention, _build_viseme_action, _channel_stem,
    _get_or_create_channelbag, _MILO_VISEME_TAG, _MILO_VISEME_NAME,
)
from .lipsync_importer import (
    build_lipsync_action, ensure_weight_props, LIPSYNC_FPS,
)


GH2_VISEME_NAMES = ('Eat', 'If', 'Ox', 'Oat', 'Earth', 'Size', 'Church', 'Fave', 'Though',
                    'Bump', 'New', 'Told', 'Roar', 'Wet', 'Cage')
# Samples 1-15, from metal_singer.fac's pose-node order (see the module docstring).
DEFAULT_SAMPLE_ORDER = "Eat,Earth,If,Ox,Oat,Wet,Size,Church,Fave,Though,Told,Bump,New,Roar,Cage"

_GH2_SAMPLE_PROP = "milo_gh2_sample"
_GH2_ORDER_PROP = "milo_gh2_sample_order"
_VOC_EXTRA_PREFIX = "voc_"

_MARKER = b'\xAD\xDE\xAD\xDE'
_BLOCK_COUNTS = 10            # cumulative per-type counts in a block header


class GH2ImportError(Exception):
    pass


# ---------------------------------------------------------------------------------------
# Little-endian readers
# ---------------------------------------------------------------------------------------

class _LE:
    def __init__(self, data, p=0):
        self.d = data
        self.p = p

    def u32(self):
        v = struct.unpack_from('<I', self.d, self.p)[0]; self.p += 4; return v

    def f32(self):
        v = struct.unpack_from('<f', self.d, self.p)[0]; self.p += 4; return v

    def sym(self, limit=256):
        n = self.u32()
        if n > limit or self.p + n > len(self.d):
            raise GH2ImportError(f"implausible string length {n} at offset {self.p - 4}")
        s = self.d[self.p:self.p + n].decode('latin-1'); self.p += n
        return s


def _gh2_entries(body):
    """{name: (type, bytes)} for a little-endian revision-25 directory. Entries are
    delimited by end marker; the first marker closes the directory itself."""
    r = _LE(body)
    rev = r.u32()
    if rev != 25:
        raise GH2ImportError(f"milo revision {rev}, expected 25 for Guitar Hero 2 "
                             f"(is this a little-endian GH2 file?)")
    r.sym(); dir_name = r.sym(); r.u32(); r.u32()
    count = r.u32()
    if count > 4096:
        raise GH2ImportError("implausible entry count - this doesn't look like a GH2 milo")
    ents = [(r.sym(), r.sym()) for _ in range(count)]
    marks, i = [], 0
    while True:
        j = body.find(_MARKER, i)
        if j < 0:
            break
        marks.append(j); i = j + 4
    if len(marks) != count + 1:
        raise GH2ImportError(f"found {len(marks)} end markers, expected {count + 1}")
    return dir_name, {n: (t, body[marks[k] + 4:marks[k + 1]]) for k, (t, n) in enumerate(ents)}


def _parse_blocks(c, start):
    """Parses the three CharBonesSamples blocks starting at `start`. Returns (blocks, end)."""
    r = _LE(c, start)
    blocks = []
    for _ in range(3):
        n = r.u32()
        if n > 512:
            raise GH2ImportError("implausible channel count")
        names = []
        for _ in range(n):
            names.append(r.sym(128)); r.f32()
        counts = [r.u32() for _ in range(_BLOCK_COUNTS)]
        comp = r.u32()
        ns = r.u32()
        if comp != 1 or ns > 100000:
            raise GH2ImportError("unsupported block layout")
        blocks.append(dict(names=names, counts=counts, ns=ns))
    for b in blocks:
        npos = b['counts'][1]
        nq = b['counts'][3] - b['counts'][2]
        if npos + nq != len(b['names']):
            raise GH2ImportError("channel counts don't match the channel list "
                                 "(scales or single-axis rotations aren't supported)")
        b['samples'] = []
        for _ in range(b['ns']):
            s = {}
            for i in range(npos):
                s[b['names'][i]] = ('pos', struct.unpack_from('<3f', c, r.p)); r.p += 12
            for i in range(nq):
                s[b['names'][npos + i]] = ('quat', tuple(
                    max(v / 32767.0, -1.0) for v in struct.unpack_from('<4h', c, r.p)))
                r.p += 8
            b['samples'].append(s)
    return blocks, r.p


def parse_gh2_clip(c):
    """Decodes one GH2 CharClipSamples into its three blocks. The 66-byte header both
    retail files use is tried first; otherwise the start is searched for, accepting only a
    parse that consumes the entry exactly with every quaternion unit length."""
    for start in [66] + [s for s in range(0, min(len(c), 512)) if s != 66]:
        try:
            blocks, end = _parse_blocks(c, start)
        except (GH2ImportError, struct.error, IndexError, UnicodeDecodeError):
            continue
        if end != len(c):
            continue
        quats = [v for b in blocks for s in b['samples'] for (k, v) in s.values() if k == 'quat']
        if all(abs(math.sqrt(sum(x * x for x in q)) - 1.0) < 0.01 for q in quats):
            return blocks
    raise GH2ImportError("couldn't decode this clip's sample data")


def _sample_pose(blocks, index):
    """The merged channels at one sample: the per-frame block's sample `index` plus the
    constant blocks' single samples."""
    pose = {}
    for b in blocks:
        if not b['samples']:
            continue
        if len(b['samples']) == 1:
            pose.update(b['samples'][0])
        elif index < len(b['samples']):
            pose.update(b['samples'][index])
    return pose


def read_gh2_viseme_milo(filepath):
    """Returns (dir_name, [16 sample poses], neutral_rest_rotations)."""
    with open(filepath, 'rb') as f:
        body = read_milo_container_body(f.read())
    dir_name, E = _gh2_entries(body)
    if 'visemes' not in E:
        raise GH2ImportError("no 'visemes' clip - is this a GH2 viseme milo?")
    vis = parse_gh2_clip(E['visemes'][1])
    count = max(b['ns'] for b in vis)
    poses = [_sample_pose(vis, i) for i in range(count)]
    rest = {}
    if 'neutral' in E:
        neu = parse_gh2_clip(E['neutral'][1])
        for chan, (kind, v) in _sample_pose(neu, 0).items():
            if kind == 'quat':
                x, y, z, w = v
                rest[_channel_stem(chan)] = (w, x, y, z)
    return dir_name, poses, rest


def read_fac_order(filepath):
    """The viseme names for samples 1-15, in the order a FaceFX actor (.fac) stores its
    FxBonePoseNodes - the first node is Neutral (sample 0) and is dropped. Each node is
    located by its class name, and its own name is the first short length-prefixed string
    after it."""
    with open(filepath, 'rb') as f:
        d = f.read()
    if d[:4] != b'FACE':
        raise GH2ImportError("not a FaceFX actor file (missing FACE header)")
    tag = b'FxBonePoseNode\\'
    names, p = [], 0
    while True:
        i = d.find(tag, p)
        if i < 0:
            break
        name = None
        for k in range(i + len(tag), min(len(d) - 4, i + len(tag) + 64)):
            n = struct.unpack_from('<I', d, k)[0]
            if 2 <= n <= 32 and k + 4 + n <= len(d) \
                    and all(48 <= c < 123 for c in d[k + 4:k + 4 + n]):
                name = d[k + 4:k + 4 + n].decode('latin-1')
                break
        if name is None:
            raise GH2ImportError("couldn't read a pose node's name in the .fac")
        names.append(name)
        p = i + len(tag)
    if len(names) != 16 or names[0].lower() != 'neutral':
        raise GH2ImportError(f"expected 16 pose nodes starting with Neutral, found "
                             f"{len(names)}: {', '.join(names[:4])}...")
    return names[1:]


def parse_sample_order(text):
    """'Cage,Fave,...' -> ['Cage', 'Fave', ...] for samples 1-15, validated."""
    names = [n.strip() for n in text.split(',') if n.strip()]
    if len(names) != 15:
        raise GH2ImportError(f"Sample Order needs 15 names (samples 1-15), got {len(names)}")
    if len(set(names)) != 15:
        raise GH2ImportError("Sample Order names each viseme more than once")
    return names


# ---------------------------------------------------------------------------------------
# .voc
# ---------------------------------------------------------------------------------------

def read_voc(filepath):
    """Returns {track name: [(seconds, value), ...]} for every track in a .voc."""
    with open(filepath, 'rb') as f:
        d = f.read()
    if d[:4] != b'FACE':
        raise GH2ImportError("not a .voc file (missing FACE header)")
    tracks = {}
    # Track names are located by their length-prefixed strings, then each track's layout
    # (8 bytes, key count, 18-byte keys, 8 bytes) is validated: times must increase.
    known = list(GH2_VISEME_NAMES) + [
        'Orientation Head Pitch', 'Orientation Head Roll', 'Orientation Head Yaw',
        'Gaze Eye Pitch', 'Gaze Eye Yaw', 'Emphasis Head Pitch', 'Emphasis Head Roll',
        'Emphasis Head Yaw', 'Eyebrow Raise', 'Blink']
    for name in known:
        tag = struct.pack('<I', len(name)) + name.encode('latin-1')
        i = d.find(tag)
        if i < 0:
            continue
        p = i + len(tag) + 8
        count = struct.unpack_from('<I', d, p)[0]; p += 4
        if p + count * 18 > len(d):
            raise GH2ImportError(f"track '{name}' runs past the end of the file")
        keys = []
        for _ in range(count):
            t, v = struct.unpack_from('<2f', d, p + 2)
            keys.append((t, v)); p += 18
        if any(b[0] < a[0] for a, b in zip(keys, keys[1:])):
            raise GH2ImportError(f"track '{name}' doesn't decode cleanly (times go backwards)")
        tracks[name] = keys
    if not any(n in tracks for n in GH2_VISEME_NAMES):
        raise GH2ImportError("no viseme tracks found in this .voc")
    return tracks


# ---------------------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------------------

class IMPORT_OT_gh2_viseme_set(bpy.types.Operator, ImportHelper):
    """Import a Guitar Hero 2 singer viseme milo as one Action per viseme on the active
    armature, named so a GH2 .voc lipsync can bake against them"""
    bl_idname = "import_scene.gh2_viseme_set"
    bl_label = "Import GH2 Viseme Set"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".milo_xbox"
    filter_glob: StringProperty(default="*.milo_xbox", options={'HIDDEN'})

    sample_order: StringProperty(
        name="Sample Order",
        description="The viseme name for each of samples 1-15, comma-separated (sample 0 "
                     "is Neutral). The viseme milo doesn't name its samples; the default is "
                     "the order from metal_singer.fac, verified against the jaw poses. "
                     "Ignored if a FaceFX Actor file is given below",
        default=DEFAULT_SAMPLE_ORDER,
    )
    # Plain text on purpose: a file-path field's folder button can't open a second file
    # browser from inside this import dialog. Paste the path.
    fac_path: StringProperty(
        name="FaceFX Actor (.fac)",
        description="Optional: the singer's .fac file (e.g. metal_singer.fac, one folder above "
                     "the character milo). If given, the sample order is read from it "
                     "instead of the Sample Order field - other singers have their own",
        default="",
    )
    pos_space: EnumProperty(
        name="Position Offsets",
        items=[('PARENT', "Parent Frame", "Offsets are in the bone's parent frame"),
               ('LOCAL', "Bone Frame", "Offsets are in the bone's own rest frame")],
        default='PARENT',
    )
    rot_space: EnumProperty(
        name="Rotation Offsets",
        items=[('AUTO', "Auto (measure)", "Measure the convention against the 'neutral' clip"),
               ('LOCAL', "Bone Frame", "Rotation offsets are in each bone's own frame"),
               ('PARENT', "Parent Frame", "Rotation offsets are in the parent's frame")],
        default='AUTO',
    )

    def execute(self, context):
        arm = context.active_object
        if arm is None or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Select the GH2 singer's armature first.")
            return {'CANCELLED'}
        try:
            if self.fac_path:
                order = read_fac_order(bpy.path.abspath(self.fac_path))
                order_source = f"from {os.path.basename(self.fac_path)}"
            else:
                order = parse_sample_order(self.sample_order)
                order_source = ("from metal_singer.fac (default)"
                                if self.sample_order == DEFAULT_SAMPLE_ORDER else "custom")
            dir_name, poses, rest = read_gh2_viseme_milo(self.filepath)
        except Exception as e:
            _log(f"GH2 VISEME IMPORT FAILED: {e}")
            self.report({'ERROR'}, f"Could not import GH2 visemes: {e}")
            return {'CANCELLED'}
        if len(poses) < 16:
            self.report({'ERROR'}, f"Expected 16 viseme samples, found {len(poses)}.")
            return {'CANCELLED'}

        _log(f"===== Importing GH2 viseme set '{dir_name}' from {self.filepath} =====")
        lookup = build_bone_lookup(arm)
        channels = {_channel_stem(c) for p in poses for c in p}
        found = [s for s in channels if s in lookup]
        _log(f"  {len(poses)} samples; {len(found)} of {len(channels)} driven bone(s) "
             f"found on '{arm.name}'")
        if not found:
            self.report({'ERROR'}, "None of the viseme bones are on this armature - is it "
                                   "the GH2 singer's skeleton?")
            return {'CANCELLED'}

        convention = 'ROW'
        if self.rot_space == 'AUTO':
            if rest:
                convention, _r, _c = detect_rotation_convention(arm, lookup, rest)
            else:
                _log("  No 'neutral' clip to measure against - assuming row-vector.")
        _log(f"  Sample order (samples 1-15): {', '.join(order)}  [{order_source}]")

        made = 0
        for i, name in enumerate(order, start=1):
            action, matched, _un = _build_viseme_action(
                arm, name, dir_name, poses[i], lookup,
                pos_space=self.pos_space, rot_space=self.rot_space,
                conjugate_rotations=True, milo_rest_rot=rest, convention=convention)
            action[_GH2_SAMPLE_PROP] = i
            action[_GH2_ORDER_PROP] = ",".join(order)
            made += 1
        summary = f"Imported {made} GH2 viseme Action(s) from '{dir_name}' onto '{arm.name}'"
        _log(f"===== {summary} =====")
        self.report({'INFO'}, summary)
        return {'FINISHED'}


class IMPORT_OT_gh2_voc(bpy.types.Operator, ImportHelper):
    """Import a Guitar Hero 2 .voc lipsync file as viseme weight channels on the active
    armature, ready for Bake Lipsync Preview"""
    bl_idname = "import_scene.gh2_voc"
    bl_label = "Import GH2 Lipsync (.voc)"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".voc"
    filter_glob: StringProperty(default="*.voc", options={'HIDDEN'})

    interpolation: EnumProperty(
        name="Interpolation",
        description="How to blend between keys. The .voc doesn't record it - every slope is "
                     "stored as zero - so this is a choice",
        items=[('LINEAR', "Linear", "Straight lines between keys"),
               ('BEZIER', "Smooth", "Ease in and out of each key"),
               ('CONSTANT', "Hold", "Step from key to key")],
        default='LINEAR',
    )
    import_extra_tracks: BoolProperty(
        name="Import Head/Gaze/Blink Tracks",
        description="Also import the non-viseme tracks (head orientation, eye gaze, head "
                     "emphasis, eyebrow raise, blink) as voc_* channels on the armature. "
                     "They drive nothing yet - which bones they move isn't in the file",
        default=True,
    )
    set_scene_range: BoolProperty(
        name="Set Scene Frame Range",
        description="Set the scene's frame range to cover the whole lipsync",
        default=True,
    )

    def execute(self, context):
        arm = context.active_object
        if arm is None or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Select the singer's armature first.")
            return {'CANCELLED'}
        try:
            tracks = read_voc(self.filepath)
        except Exception as e:
            _log(f"GH2 VOC IMPORT FAILED: {e}")
            self.report({'ERROR'}, f"Could not read .voc: {e}")
            return {'CANCELLED'}

        scene = context.scene
        fps = scene.render.fps / max(scene.render.fps_base, 1e-6)
        song = os.path.splitext(os.path.basename(self.filepath))[0]
        _log(f"===== Importing GH2 lipsync '{song}' from {self.filepath} =====")

        # Seconds -> frames at the scene's own rate. build_lipsync_action adds 1, so frame 0
        # lands on Blender frame 1, as the RB lipsync import does.
        weights = {n: [(t * fps, v) for t, v in keys]
                   for n, keys in tracks.items() if n in GH2_VISEME_NAMES}
        available = {a.get(_MILO_VISEME_NAME) for a in bpy.data.actions
                     if a.get(_MILO_VISEME_TAG)}
        used = [n for n, k in weights.items() if k]
        missing = [n for n in used if n not in available]
        ensure_weight_props(arm, sorted(weights))
        action, total = build_lipsync_action(arm, song, weights, self.interpolation)

        extra_keys = 0
        if self.import_extra_tracks:
            cb = _get_or_create_channelbag(action, action.slots[0])
            grp = cb.groups.get("VOC Head/Eyes") or cb.groups.new("VOC Head/Eyes")
            for name, keys in tracks.items():
                if name in GH2_VISEME_NAMES or not keys:
                    continue
                prop = _VOC_EXTRA_PREFIX + name.replace(' ', '_')
                if prop not in arm:
                    arm[prop] = 0.0
                fc = cb.fcurves.new(f'["{prop}"]', index=0)
                fc.group = grp
                for t, v in keys:
                    fc.keyframe_points.insert(1 + t * fps, v, options={'FAST'}
                                              ).interpolation = self.interpolation
                fc.update()
                extra_keys += len(keys)

        if arm.animation_data is None:
            arm.animation_data_create()
        arm.animation_data.action = action
        try:
            arm.animation_data.action_slot = action.slots[0]
        except (AttributeError, IndexError):
            pass

        last = max((k[-1][0] for k in tracks.values() if k), default=0.0)
        if self.set_scene_range:
            scene.frame_start = 1
            scene.frame_end = max(scene.frame_start, int(math.ceil(last * fps)) + 1)

        _log(f"  {len(used)} viseme track(s), {total} weight key(s)"
             + (f"; {extra_keys} head/gaze/blink key(s) as voc_* channels (not baked)"
                if extra_keys else "") + f"; {last:.1f}s at {fps:g} fps")
        if missing:
            _log(f"  WARNING: no imported pose for {', '.join(missing)} - import the singer's "
                 f"GH2 viseme set first, or those visemes will contribute nothing")
        summary = (f"Imported GH2 lipsync '{song}': {len(used)} viseme channel(s), {total} "
                   f"key(s)" + (f"; {len(missing)} without a pose (see log)" if missing else ""))
        _log(f"===== {summary} =====")
        _log("  NEXT STEP: run Object > Rock Band Lipsync > Bake Lipsync Preview.")
        self.report({'WARNING' if missing else 'INFO'},
                    summary + " - now run 'Bake Lipsync Preview'")
        return {'FINISHED'}
