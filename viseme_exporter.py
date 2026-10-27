"""
Dance Central 1 viseme milo export (Xbox 360).

Builds a COMPLETE viseme CharClipSet milo - the 'viseme' directory a DC1 character's face
plays from - out of Blender Actions on the active armature, the way the GDRB raw CharClip
export writes clips from Actions. Nothing is patched into a vanilla file: Base and every
viseme clip are written fresh from the armature, so the whole set agrees with the custom
character's face.

This replaces the earlier "Inject New Base Viseme" operator, which rewrote only the Base clip
inside a vanilla milo. Testing showed that isn't enough: Harmonix's face animation is tied
to the viseme clips themselves, so a custom face only stops deforming when the full set is
rebuilt for it.

===========================================================================================
FORMAT - reproduced byte for byte against a retail DC1 viseme.milo_xbox
===========================================================================================
An uncompressed milo (one block at 0x810), revision 28, directory type CharClipSet named
'viseme', holding 30 CharClips at version 19 - RB3's clip version, read and written here
with rb3_clip_exporter's parse_rb3_clip / encode_rb3_clip, which reproduce all 30 retail
clips exactly. The directory table's two string-table hints are 2 x (entries + 1) and the
summed name lengths (each + 1) of the entries and the directory. The CharClipSet's own
object body (416 bytes: viewports and defaults, no names or character data in it) is
copied verbatim from retail.

Every clip has the same object header (revision 2, type 'viseme'), 30 fps, an empty 'full'
sample block (no channels, one sample), and its pose in the 'one' block as a single sample
at compression 1 (positions as floats, quaternions as int16). After the samples come
mZeros - every Base channel the clip does NOT carry, weight 1.0 - so a clip's 'one'
channels plus its zeros are always exactly Base's channel list. Then a one-entry beat track
and an empty sync-anim symbol.

  Base   flags 0xf00, play flags 0x220, no relative clip, all 58 channels as ABSOLUTE local
         transforms (35 positions, 23 quaternions), no zeros.
  Blink  flags 0xf00, play flags 0x200, relative to Base.
  others flags 0,     play flags 0x220, relative to Base.

The non-Base clips are OFFSETS from Base, holding only the channels that move. Channels in
both blocks are sorted by name within their type, positions before quaternions.

===========================================================================================
FROM ACTIONS
===========================================================================================
Base is the armature's REST pose: each Base channel is the bone's local rest transform
(relative to its parent), the same matrix the DC1 character export writes for that bone, so
Base and the exported skeleton agree exactly.

Each viseme is one Action, read at its first frame, and stored as its offset from rest -
the exact inverse of the viseme importer, so importing a set and exporting it gives the
same offsets back:

  position   rest_rotation x pose location        (offset in the parent's frame)
  rotation   the pose quaternion as it is          (as-stored convention)
             or conj(rest x pose x rest^-1)        (conjugated convention)

DC1 rotations are as-stored - measured on DC1's own skeleton by the clip importer (19.0
against 43.6 degrees). Actions made by the viseme importer record the convention they were
imported with, and Auto uses it.

Actions are matched to clips by the importer's clip name, by a VISEME_ prefix, or by being
named after a DC1 clip. A clip with no Action is written with no motion (all zeros), and the
log says which.

===========================================================================================
FACE ONLY
===========================================================================================
A viseme can only move the face. Any Action that animates a bone outside the face - a body
bone, the neck or bone_head itself, or the armature object - is refused as a whole, and the
export stops without writing anything. Face bones are DC1's viseme bones plus everything
beneath bone_head. Body bones that are keyed but hold their rest pose don't count. Face
bones that DC1's viseme set has no channel for (eyes, for example) can't be stored and are
reported.
"""

import base64
import os
import struct

import bpy
from bpy.props import StringProperty, BoolProperty, EnumProperty
from bpy_extras.io_utils import ExportHelper
from mathutils import Euler, Quaternion, Vector

from .utilities import _log
from .viseme_importer import _channel_stem, build_bone_lookup
from .gdrb_clip_exporter import _action_fcurves
from .rb3_clip_exporter import encode_rb3_clip, wrap_uncompressed

# __init__ star-imports this module; only the operator should land there, not constants
# like END_MARKER or CLIP_FPS that other modules also define.
__all__ = ['EXPORT_OT_dc1_viseme_milo']

MILO_REVISION = 28
DIR_TYPE = "CharClipSet"
DIR_NAME = "viseme"
CLIP_TYPE = "CharClip"
END_MARKER = b'\xAD\xDE\xAD\xDE'
CLIP_FPS = 30.0
COMPRESSION = 1
BASE_CLIP = "Base"
RELATIVE_CLIP = "Base"

# Hmx::Object header of every retail clip: revision 2, type 'viseme', no data tree, no note.
OBJECT_HEADER = bytes.fromhex("0000000200000006766973656d650000000000")

# The CharClipSet's own object body, verbatim from retail (see the module docstring).
DIR_BODY = base64.b64decode(
    "AAAAGAAAABsAAAACAAAAAAAAAAAAAAAAAAAABz81BPO/NQTzAAAAAD8TzTo/E806vxPNOj7RBes+"
    "0QXrP1EF68Pds9fD3bPXQ92z1wAAAAC/gAAAAAAAAD+AAAAAAAAAAAAAAAAAAAAAAAAAP4AAAMRA"
    "AAAAAAAAAAAAAAAAAAA/gAAAAAAAAL+AAAAAAAAAAAAAAAAAAAAAAAAAP4AAAERAAAAAAAAAAAAA"
    "AD+AAAAAAAAAAAAAAAAAAAAAAAAAv4AAAAAAAAA/gAAAAAAAAAAAAAAAAAAAREAAAD+AAAAAAAAA"
    "AAAAAAAAAAAAAAAAP4AAAAAAAAC/gAAAAAAAAAAAAAAAAAAAxEAAAD+AAAAAAAAAAAAAAAAAAAA/"
    "gAAAAAAAAAAAAAAAAAAAP4AAAAAAAADEQAAAAAAAAL+AAAAAAAAAAAAAAAAAAAC/gAAAAAAAAAAA"
    "AAAAAAAAP4AAAAAAAABEQAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAABaAAAAAAA=")

# Base's channels, in retail order (positions, then quaternions, each sorted by name).
BASE_POS_STEMS = (
    'bone_L-brow1', 'bone_L-brow2', 'bone_L-brow3', 'bone_L-cheek', 'bone_L-cheek2',
    'bone_L-crease', 'bone_L-eyelid-low', 'bone_L-lid', 'bone_L-lipcorner', 'bone_L-nose',
    'bone_R-brow1', 'bone_R-brow2', 'bone_R-brow3', 'bone_R-cheek', 'bone_R-cheek2',
    'bone_R-crease', 'bone_R-eyelid-low', 'bone_R-lid', 'bone_R-lipcorner', 'bone_R-nose',
    'bone_brow-low', 'bone_brow-mid', 'bone_chin', 'bone_forehead', 'bone_jaw',
    'bone_liptop_left', 'bone_liptop_mid', 'bone_liptop_right', 'bone_lowlip_left',
    'bone_lowlip_mid', 'bone_lowlip_right', 'bone_nose', 'bone_tongue1', 'bone_tongue2',
    'bone_tongue3',
)
BASE_QUAT_STEMS = (
    'bone_L-brow1', 'bone_L-brow2', 'bone_L-brow3', 'bone_L-eyelid-low', 'bone_L-lid',
    'bone_R-brow1', 'bone_R-brow2', 'bone_R-brow3', 'bone_R-eyelid-low', 'bone_R-lid',
    'bone_chin', 'bone_forehead', 'bone_jaw', 'bone_liptop_left', 'bone_liptop_mid',
    'bone_liptop_right', 'bone_lowlip_left', 'bone_lowlip_mid', 'bone_lowlip_right',
    'bone_nose', 'bone_tongue1', 'bone_tongue2', 'bone_tongue3',
)

# Every retail clip: (flags, play flags, beat-track beat). The beat is the clip's one beat
# track entry, paired with frame 0; retail values are kept for the retail names.
DC1_CLIPS = {
    'Angry': (0x0, 0x220, 7.666666507720947),
    'Base': (0xf00, 0x220, 0.06666667014360428),
    'Blink': (0xf00, 0x200, 0.13333334028720856),
    'Concentrate': (0x0, 0x220, 6.333333492279053),
    'Excited': (0x0, 0x220, 5.666666507720947),
    'Grin': (0x0, 0x220, 3.6666667461395264),
    'Mouth_Open': (0x0, 0x220, 1.0),
    'O_Face': (0x0, 0x220, 4.333333492279053),
    'Open_Smile': (0x0, 0x220, 3.0),
    'Open_Smile_02': (0x0, 0x220, 8.800000190734863),
    'Pursed': (0x0, 0x220, 2.3333332538604736),
    'Sexy': (0x0, 0x220, 5.0),
    'Smile': (0x0, 0x220, 7.0),
    'Smirk': (0x0, 0x220, 1.6666666269302368),
    'Squash_High': (0x0, 0x220, 0.8666666746139526),
    'Squash_Low': (0x0, 0x220, 4.199999809265137),
    'phnm_A_I': (0x0, 0x220, 14.333333015441895),
    'phnm_BROWS_DOWN': (0x0, 0x220, 23.0),
    'phnm_BROWS_UP': (0x0, 0x220, 22.33333396911621),
    'phnm_CLOSED': (0x0, 0x220, 21.66666603088379),
    'phnm_C_D_S': (0x0, 0x220, 17.0),
    'phnm_E': (0x0, 0x220, 15.0),
    'phnm_F_V': (0x0, 0x220, 19.0),
    'phnm_LOUD': (0x0, 0x220, 21.0),
    'phnm_L_TH': (0x0, 0x220, 17.66666603088379),
    'phnm_M_B_P': (0x0, 0x220, 18.33333396911621),
    'phnm_O': (0x0, 0x220, 15.666666984558105),
    'phnm_SH': (0x0, 0x220, 20.33333396911621),
    'phnm_U': (0x0, 0x220, 16.33333396911621),
    'phnm_W_OO': (0x0, 0x220, 19.66666603088379),
}
DEFAULT_CLIP_FLAGS = (0x0, 0x220, 0.0)
_CLIP_BY_LOWER = {k.lower(): k for k in DC1_CLIPS}

# Tags the viseme importer puts on its Actions.
_TAG_VISEME = "milo_viseme"
_TAG_NAME = "milo_viseme_name"
_TAG_NEUTRAL = "milo_viseme_neutral"
_TAG_CONVENTION = "milo_viseme_convention"

# Below this a position offset is nothing, so the channel goes to the zeros list (positions
# are floats at compression 1). A rotation is nothing when it quantises to the identity.
POS_ZERO = 1e-5
# A body bone counts as moving if any key strays this far from its rest value.
BODY_TOL = 1e-4

ROOT_HEAD = 'bone_head'


# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------

def _q16(v):
    return max(-32768, min(32767, int(round(v * 32767.0))))


def _canonical(q):
    """(x, y, z, w) with w >= 0 - q and -q are the same rotation."""
    return tuple(-c for c in q) if q[3] < 0.0 else tuple(q)


def _quat_is_identity(q_xyzw):
    x, y, z, _w = _canonical(q_xyzw)
    return _q16(x) == 0 and _q16(y) == 0 and _q16(z) == 0


def _local_rest(bone):
    """The bone's rest transform relative to its parent - what the character export writes
    for it (skeleton_exporter.build_all_bone_trans_entries)."""
    return (bone.parent.matrix_local.inverted() @ bone.matrix_local) if bone.parent \
        else bone.matrix_local


def _xyzw(q):
    """Blender's (w, x, y, z) as the file's (x, y, z, w)."""
    return (q[1], q[2], q[3], q[0])


def _in_subtree(bone, root_name):
    b = bone
    while b is not None:
        if b.name == root_name:
            return True
        b = b.parent
    return False


def clip_name_for_action(action):
    """The clip an Action stands for, or None if it isn't a viseme Action. Retail names
    match whatever their case (VISEME_angry -> Angry)."""
    if action.get(_TAG_NEUTRAL):
        return None
    name = action.get(_TAG_NAME) if action.get(_TAG_VISEME) else None
    if not name and action.name.startswith("VISEME_"):
        name = action.name[len("VISEME_"):]
    if not name:
        return _CLIP_BY_LOWER.get(action.name.lower())
    return _CLIP_BY_LOWER.get(name.lower(), name)


# ---------------------------------------------------------------------------------------
# Reading an Action
# ---------------------------------------------------------------------------------------

_POSE_PREFIX = 'pose.bones["'
_OBJECT_TRANSFORM = ('location', 'rotation_quaternion', 'rotation_euler',
                     'rotation_axis_angle', 'scale', 'delta_location',
                     'delta_rotation_quaternion', 'delta_rotation_euler', 'delta_scale')
# Each transform property's value at rest, per array index.
_REST_VALUE = {
    'location': lambda i: 0.0,
    'rotation_quaternion': lambda i: 1.0 if i == 0 else 0.0,
    'rotation_euler': lambda i: 0.0,
    'rotation_axis_angle': lambda i: 1.0 if i == 2 else 0.0,
    'scale': lambda i: 1.0,
}
_EULER_ORDERS = ('XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX')


def collect_curves(action):
    """({bone: {prop: {index: fcurve}}}, [object-level transform curves])."""
    bones, obj = {}, []
    for fc in _action_fcurves(action):
        path = fc.data_path
        if path.startswith(_POSE_PREFIX):
            end = path.find('"]', len(_POSE_PREFIX))
            if end < 0:
                continue
            bone = path[len(_POSE_PREFIX):end]
            prop = path[end + 3:] if path[end + 2:end + 3] == '.' else ''
            bones.setdefault(bone, {}).setdefault(prop, {})[fc.array_index] = fc
        elif path in _OBJECT_TRANSFORM:
            obj.append(fc)
    return bones, obj


def _moves(rest, fc):
    return any(abs(kp.co[1] - rest) > BODY_TOL for kp in fc.keyframe_points)


def _bone_curve_moves(prop, index, fc):
    """Whether a pose bone's transform curve ever leaves its rest value (checked at every
    key). Non-transform properties (custom props, constraints) don't move the bone."""
    rest = _REST_VALUE.get(prop)
    return rest is not None and _moves(rest(index), fc)


def _object_curve_moves(fc):
    prop = fc.data_path.replace('delta_', '')
    return _moves(_REST_VALUE.get(prop, _REST_VALUE['location'])(fc.array_index), fc)


def evaluate_bone(pose_bone, props, frame):
    """(location Vector, rotation Quaternion, scale tuple) of a bone's pose basis from its
    curves at `frame`. Rotation follows whichever rotation curves the Action keys."""
    def comp(prop, n):
        c = props.get(prop, {})
        rest = _REST_VALUE[prop]
        return [c[i].evaluate(frame) if i in c else rest(i) for i in range(n)]

    loc = Vector(comp('location', 3))
    if 'rotation_quaternion' in props:
        q = Quaternion(comp('rotation_quaternion', 4))
        if q.magnitude < 1e-8:
            q = Quaternion((1.0, 0.0, 0.0, 0.0))
        q.normalize()
    elif 'rotation_euler' in props:
        order = pose_bone.rotation_mode if pose_bone.rotation_mode in _EULER_ORDERS else 'XYZ'
        q = Euler(comp('rotation_euler', 3), order).to_quaternion()
    elif 'rotation_axis_angle' in props:
        a = comp('rotation_axis_angle', 4)
        axis = Vector(a[1:])
        q = Quaternion(axis.normalized(), a[0]) if axis.length > 1e-8 else \
            Quaternion((1.0, 0.0, 0.0, 0.0))
    else:
        q = Quaternion((1.0, 0.0, 0.0, 0.0))
    return loc, q, tuple(comp('scale', 3))


def _pose_varies(props, frame0):
    """Whether any of a bone's curves holds a key that differs from its value at frame0 -
    a viseme is one pose, so only the first frame is written."""
    for comps in props.values():
        for fc in comps.values():
            if _moves(fc.evaluate(frame0), fc):
                return True
    return False


# ---------------------------------------------------------------------------------------
# The face rig
# ---------------------------------------------------------------------------------------

class FaceRig:
    """Which of the armature's bones carry DC1 viseme channels, and which count as face."""

    def __init__(self, arm_obj):
        self.arm = arm_obj
        lookup = build_bone_lookup(arm_obj)
        pbs = arm_obj.pose.bones
        self.pos, self.quat, self.missing = {}, {}, []      # stem -> bone name
        for stems, table, suffix in ((BASE_POS_STEMS, self.pos, '.pos'),
                                     (BASE_QUAT_STEMS, self.quat, '.quat')):
            for stem in stems:
                name = lookup.get(stem)
                if name and name in pbs:
                    table[stem] = name
                else:
                    self.missing.append(stem + suffix)
        self.stem_of = {name: stem for table in (self.pos, self.quat)
                        for stem, name in table.items()}
        head = lookup.get(ROOT_HEAD)
        self.head = head if head in pbs else None
        self.rest = {}
        for name in self.stem_of:
            rl = _local_rest(pbs[name].bone)
            self.rest[name] = (rl.to_translation(), rl.to_quaternion())

    def is_face(self, bone_name):
        if bone_name in self.stem_of:
            return True
        if self.head is None or bone_name == self.head:
            return False
        pb = self.arm.pose.bones.get(bone_name)
        return pb is not None and _in_subtree(pb.bone, self.head)

    def channels(self):
        """Base's channel names this rig supports, in file order."""
        return ([s + '.pos' for s in BASE_POS_STEMS if s in self.pos]
                + [s + '.quat' for s in BASE_QUAT_STEMS if s in self.quat])


def encode_rotation(rest_q, basis_q, convention):
    """A rotation offset as stored, from a pose basis rotation - the inverse of the viseme
    importer: as stored, the basis IS the offset; conjugated, the importer's
    rest^-1 x conj(q) x rest is undone."""
    if convention == 'ROW':
        q = (rest_q @ basis_q @ rest_q.inverted()).conjugated()
    else:
        q = basis_q.copy()
    q.normalize()
    return _canonical(_xyzw(q))


def base_pose(rig, convention):
    """Base: every supported channel's absolute local rest transform."""
    pose = {}
    for stem, name in rig.pos.items():
        t = rig.rest[name][0]
        pose[stem + '.pos'] = (t.x, t.y, t.z)
    for stem, name in rig.quat.items():
        q = rig.rest[name][1].normalized()
        if convention == 'ROW':
            q = q.conjugated()
        pose[stem + '.quat'] = _canonical(_xyzw(q))
    return pose


def viseme_offsets(rig, curves, frame, convention):
    """{channel: offset} for the channels the Action moves, plus notes on what it moves
    that a DC1 viseme can't hold."""
    pbs = rig.arm.pose.bones
    out = {}
    notes = dict(no_rotation=[], no_position=[], scale=[], varies=[], unstored=[])
    for bone in sorted(b for b in curves if b in pbs):
        if bone not in rig.stem_of:
            if rig.is_face(bone):
                notes['unstored'].append(bone)
            continue
        props = curves[bone]
        loc, q, scale = evaluate_bone(pbs[bone], props, frame)
        if _pose_varies(props, frame):
            notes['varies'].append(bone)
        if max(abs(s - 1.0) for s in scale) > BODY_TOL:
            notes['scale'].append(bone)
        rest_q = rig.rest[bone][1]
        stem = rig.stem_of[bone]
        if stem in rig.pos:
            d = rest_q @ loc
            if max(abs(c) for c in d) > POS_ZERO:
                out[stem + '.pos'] = (d.x, d.y, d.z)
        elif loc.length > POS_ZERO:
            notes['no_position'].append(bone)
        if stem in rig.quat:
            dq = encode_rotation(rest_q, q, convention)
            if not _quat_is_identity(dq):
                out[stem + '.quat'] = dq
        elif not _quat_is_identity(_xyzw(q)):
            notes['no_rotation'].append(bone)
    return out, notes


def check_face_only(rig, curves, obj_curves):
    """(bones moved outside the face, body bones keyed but held at rest). Only the first
    blocks the export."""
    moving, still = [], []
    pbs = rig.arm.pose.bones
    for bone, props in curves.items():
        if bone not in pbs or rig.is_face(bone):
            continue
        moves = any(_bone_curve_moves(prop, i, fc)
                    for prop, comps in props.items() for i, fc in comps.items())
        (moving if moves else still).append(bone)
    moving.sort()
    if any(_object_curve_moves(fc) for fc in obj_curves):
        moving.append(f"the armature object '{rig.arm.name}'")
    return moving, sorted(still)


# ---------------------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------------------

def _sorted_channels(names):
    return (sorted(n for n in names if n.endswith('.pos'))
            + sorted(n for n in names if n.endswith('.quat')))


def encode_viseme_clip(name, pose, base_channels, is_base=False):
    """A CharClip v19 body (no end marker) for one viseme: `pose` in the 'one' block, every
    other Base channel in the zeros list."""
    flags, play_flags, beat = DC1_CLIPS.get(name, DEFAULT_CLIP_FLAGS)
    one_ch = _sorted_channels(pose)
    zeros = [] if is_base else [c for c in base_channels if c not in pose]
    d = dict(
        object_raw=OBJECT_HEADER, fps=CLIP_FPS, flags=flags, play_flags=play_flags,
        range_=0.0, relative='' if is_base else RELATIVE_CLIP, old_version=-1,
        do_not_compress=0, node_size=0, transitions=[], beat_events=[],
        full=([], [{}], [], COMPRESSION),
        one=([(c, 1.0) for c in one_ch], [{c: pose[c] for c in one_ch}], [], COMPRESSION),
        zeros=[(c, 1.0) for c in _sorted_channels(zeros)],
        beat_track=[(beat, 0.0)], sync_anim='',
    )
    return encode_rb3_clip(d)


def build_milo_body(clips):
    """The milo object stream for {clip name: CharClip bytes}, entries sorted by name as
    retail's are."""
    names = sorted(clips)
    w = bytearray()

    def u32(v):
        w.extend(struct.pack('>I', v))

    def sym(s):
        raw = s.encode('latin-1')
        u32(len(raw))
        w.extend(raw)

    u32(MILO_REVISION)
    sym(DIR_TYPE)
    sym(DIR_NAME)
    u32(2 * (len(names) + 1))
    u32(sum(len(n) + 1 for n in names) + len(DIR_NAME) + 1)
    u32(len(names))
    for n in names:
        sym(CLIP_TYPE)
        sym(n)
    w.extend(DIR_BODY)
    w.extend(END_MARKER)
    for n in names:
        w.extend(clips[n])
        w.extend(END_MARKER)
    return bytes(w)


# ---------------------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------------------

class EXPORT_OT_dc1_viseme_milo(bpy.types.Operator, ExportHelper):
    """Build a complete Dance Central 1 viseme milo from the active armature: Base from its
    rest pose, every viseme from a face-only Action"""
    bl_idname = "export_scene.dc1_viseme_milo"
    bl_label = "Export DC1 Viseme Milo"
    bl_options = {'REGISTER'}

    filename_ext = ".milo_xbox"
    filter_glob: StringProperty(default="*.milo_xbox", options={'HIDDEN'})

    convention: EnumProperty(
        name="Rotation Convention",
        description="How rotations are stored. Must match how the viseme Actions were made",
        items=[('AUTO', "Auto", "Use the convention recorded on Actions made by the viseme "
                                "importer; as stored when there is none"),
               ('COLUMN', "As Stored (DC1)", "Rotations written as Blender has them - "
                                             "measured on DC1's own skeleton"),
               ('ROW', "Conjugated (RB3-style)", "Rotations conjugated, for Actions imported "
                                                  "with the RB3-style convention")],
        default='AUTO',
    )
    missing_clips: EnumProperty(
        name="Missing Visemes",
        description="What to do when a DC1 viseme has no Action",
        items=[('NEUTRAL', "Write With No Motion", "Write the clip with every channel at "
                                                    "Base, so the face holds still for it"),
               ('ERROR', "Stop", "Refuse to export until every DC1 viseme has an Action")],
        default='NEUTRAL',
    )
    include_extra: BoolProperty(
        name="Include Extra Visemes",
        description="Also write viseme Actions whose names aren't one of DC1's 29 visemes",
        default=False,
    )

    def invoke(self, context, event):
        if not self.filepath:
            blend = bpy.data.filepath
            folder = os.path.dirname(blend) if blend else ""
            self.filepath = os.path.join(folder, DIR_NAME + self.filename_ext)
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    @staticmethod
    def _gather(rig):
        """{clip: Action} for this armature, duplicates resolved and logged."""
        found = {}
        for act in bpy.data.actions:
            clip = clip_name_for_action(act)
            if clip is None or clip == BASE_CLIP:
                continue
            curves, _obj = collect_curves(act)
            if curves and not any(b in rig.arm.pose.bones for b in curves):
                continue                    # another character's Action
            found.setdefault(clip, []).append(act)
        chosen = {}
        for clip, acts in found.items():
            acts.sort(key=lambda a: (a.name != f"VISEME_{clip}", a.name))
            chosen[clip] = acts[0]
            if len(acts) > 1:
                _log(f"  '{clip}': several Actions ({', '.join(a.name for a in acts)}) - "
                     f"using '{acts[0].name}'")
        return chosen

    def _convention(self, actions):
        if self.convention != 'AUTO':
            return self.convention
        tags = [a.get(_TAG_CONVENTION) for a in actions if a.get(_TAG_CONVENTION)]
        if not tags:
            return 'COLUMN'
        conv = max(set(tags), key=tags.count)
        if len(set(tags)) > 1:
            _log(f"  WARNING: the viseme Actions were imported with different rotation "
                 f"conventions; using {conv}, the most common")
        return conv

    def execute(self, context):
        arm = context.active_object
        if arm is None or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Select the character's armature.")
            return {'CANCELLED'}
        rig = FaceRig(arm)
        base_channels = rig.channels()
        if not base_channels:
            self.report({'ERROR'}, f"'{arm.name}' has none of DC1's face bones "
                                   f"(bone_jaw, bone_L-brow1, ...) - nothing to export.")
            return {'CANCELLED'}

        _log(f"===== Exporting DC1 viseme milo from '{arm.name}' to {self.filepath} =====")
        if rig.missing:
            _log(f"  {len(rig.missing)} of {len(BASE_POS_STEMS) + len(BASE_QUAT_STEMS)} "
                 f"DC1 face channel(s) have no bone on this rig and are left out of every "
                 f"clip: {', '.join(rig.missing)}")
        if rig.head is None:
            _log("  NOTE: no bone_head on this rig - only DC1's viseme bones count as face")

        chosen = self._gather(rig)
        convention = self._convention(list(chosen.values()))
        _log(f"  Rotation convention: "
             f"{'as stored' if convention == 'COLUMN' else 'conjugated'}")

        wanted = [n for n in DC1_CLIPS if n != BASE_CLIP]
        extra = sorted(n for n in chosen if n not in DC1_CLIPS)
        if extra and not self.include_extra:
            _log(f"  Not DC1 viseme names, left out (Include Extra Visemes is off): "
                 f"{', '.join(extra)}")
            extra = []
        missing = [n for n in wanted if n not in chosen]
        if missing and self.missing_clips == 'ERROR':
            msg = f"No Action for {len(missing)} DC1 viseme(s): {', '.join(missing)}"
            _log(f"  EXPORT STOPPED: {msg}")
            self.report({'ERROR'}, msg)
            return {'CANCELLED'}

        clips = {BASE_CLIP: encode_viseme_clip(BASE_CLIP, base_pose(rig, convention),
                                               base_channels, is_base=True)}
        refused, neutral, written = [], [], 0
        for name in wanted + extra:
            act = chosen.get(name)
            if act is None:
                neutral.append(name)
                clips[name] = encode_viseme_clip(name, {}, base_channels)
                continue
            curves, obj_curves = collect_curves(act)
            moving, still = check_face_only(rig, curves, obj_curves)
            if moving:
                refused.append((act.name, moving))
                continue
            frame = act.frame_range[0]
            pose, notes = viseme_offsets(rig, curves, frame, convention)
            clips[name] = encode_viseme_clip(name, pose, base_channels)
            written += 1
            _log(f"  '{name}' <- '{act.name}': {len(pose)} channel(s) move")
            if still:
                _log(f"    body bone(s) keyed at rest, ignored: {', '.join(still[:10])}"
                     + (" ..." if len(still) > 10 else ""))
            if notes['varies']:
                _log(f"    NOTE: the Action changes over time; only frame {frame:g} is "
                     f"written (a viseme is one pose)")
            if notes['unstored']:
                _log(f"    face bone(s) DC1's viseme set has no channel for, left out: "
                     f"{', '.join(notes['unstored'])}")
            if notes['no_rotation']:
                _log(f"    rotation can't be stored (DC1 keeps only position for): "
                     f"{', '.join(notes['no_rotation'])}")
            if notes['no_position']:
                _log(f"    position can't be stored (no position channel): "
                     f"{', '.join(notes['no_position'])}")
            if notes['scale']:
                _log(f"    scale isn't stored in visemes, ignored on: "
                     f"{', '.join(notes['scale'])}")

        if refused:
            for act_name, moving in refused:
                _log(f"  REFUSED '{act_name}': it animates more than the face - "
                     f"{', '.join(moving[:12])}" + (" ..." if len(moving) > 12 else ""))
            names = ', '.join(f"'{a}'" for a, _m in refused)
            msg = (f"{len(refused)} Action(s) animate body bones, so they can't be visemes: "
                   f"{names}. Only face bones may move - remove the body keys and export "
                   f"again. Nothing was written (see the log for the bones).")
            self.report({'ERROR'}, msg)
            return {'CANCELLED'}

        if neutral:
            _log(f"  No Action, written with no motion: {', '.join(neutral)}")

        body = build_milo_body(clips)
        try:
            with open(self.filepath, 'wb') as f:
                f.write(wrap_uncompressed(body))
        except OSError as e:
            self.report({'ERROR'}, f"Could not write {self.filepath}: {e}")
            return {'CANCELLED'}

        summary = (f"Wrote {len(clips)} clip(s) to {os.path.basename(self.filepath)}: Base from "
                   f"the rest pose ({len(base_channels)} channels), {written} from Actions"
                   + (f", {len(neutral)} with no motion" if neutral else ""))
        _log(f"===== {summary} =====")
        self.report({'WARNING' if neutral else 'INFO'}, summary)
        return {'FINISHED'}
