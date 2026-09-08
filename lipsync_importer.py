"""
Rock Band .lipsync importer.

Reads a raw .lipsync file (the per-frame viseme weight stream the game plays back) and
brings it into Blender as EDITABLE weight channels, plus a separate operator that bakes
those weights into a normal pose Action for previewing.

===========================================================================================
WHY WEIGHTS, NOT A BAKED POSE
===========================================================================================
A .lipsync file holds no pose data at all - only viseme names and per-frame weights. The
poses live in the viseme CharClipSet, imported separately by viseme_importer.py as
VISEME_<name> Actions. This importer deliberately keeps those two things apart:

  .lipsync  ->  one animated custom property per viseme (the source of truth, editable)
  weights + VISEME_* Actions  ->  baked pose Action (a disposable preview, regenerate at will)

Blending N weighted poses down into resolved bone transforms is lossy and NOT invertible -
many different weight combinations land on visually similar bone poses. If the baked pose
were the source of truth, a hand-edited result could never be written back out to a valid
.lipsync. Keeping the weight curves as the real data means editing and exporting are
symmetric operations, which is what makes the intended workflow possible: auto-generate
lipsync from karaoke phonemes elsewhere, import it here, then hand-key the eyelid and
expression channels that phoneme-based tools never touch, and export the combined result.

===========================================================================================
FORMAT NOTES (verified against a retail RB3 song.lipsync)
===========================================================================================
Big-endian throughout. Header, then a viseme name table, then a keyframe stream in which
each frame lists only the visemes whose weight CHANGED on that frame, as (index, weight)
byte pairs. Weights are 0-255 and normalise to 0-1.

Weight persistence is HOLD-UNTIL-CHANGED, not re-declared-every-frame. That's directly
observable in the data: a viseme's weight ramps smoothly on consecutive frames and then
writes an explicit 0 before dropping out of the stream entirely. If an absent viseme
already meant zero, writing that 0 would be pointless. Long holds confirm it from the
other direction - Brow_down sets 77 and then isn't mentioned again for 683 frames (23
seconds), which only makes sense as a held expression. Hence CONSTANT interpolation on
the imported curves by default: the value persists until the next explicit change,
exactly as the game plays it.

The stream is a fixed 30 Hz. Frame N of the file maps to Blender frame N + 1 (Blender
timelines conventionally start at 1). If the scene isn't at 30 fps the mapping drifts, so
the operator warns rather than silently resampling.

===========================================================================================
KNOWN GAPS
===========================================================================================
1. Export isn't implemented yet. The weight curves are stored in a form designed to make
   it straightforward - sample each channel at 30 Hz, diff against the previous frame,
   emit (index, weight) pairs for whatever changed - but that's still to be written.
2. The bake blends rotations by weighted-averaging quaternions and renormalising, which
   is an approximation of true spherical blending. At the small per-viseme rotations in
   this data (mostly under 30 degrees) the error is negligible, and it's what makes N-way
   weighted blending tractable in one pass. It is NOT the same as Blender's NLA stacking,
   which composes sequentially and would diverge on overlapping channels.
3. Baking keys every bone on every frame in the range, so a full-length song is a very
   large amount of keyframe data. The operator defaults to the scene's frame range and
   reports an estimate rather than trying to bake 400 seconds by default.
"""

import struct

import bpy
from bpy.props import StringProperty, BoolProperty, IntProperty, EnumProperty
from bpy_extras.io_utils import ImportHelper
from mathutils import Quaternion

from .utilities import _log
from .viseme_importer import (
    _MILO_VISEME_TAG, _MILO_VISEME_NAME, _get_or_create_channelbag,
)


LIPSYNC_FPS = 30.0
WEIGHT_PROP_PREFIX = "viseme_"
_LIPSYNC_SONG_PROP = "milo_lipsync_song"
_LIPSYNC_VISEMES_PROP = "milo_lipsync_visemes"


class LipsyncImportError(Exception):
    """Raised when a .lipsync file can't be parsed."""
    pass


# ---------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------

def parse_lipsync(filepath):
    """Parses a .lipsync file into
    {version, subversion, dta_import, visemes, frames} where `frames` is a list (one entry
    per 30Hz frame) of [(viseme_index, weight_byte), ...] holding only that frame's CHANGES.

    Raises LipsyncImportError with a specific reason if anything doesn't add up - including
    a trailing-bytes check, since a clean parse should consume the file exactly."""
    with open(filepath, 'rb') as f:
        data = f.read()

    if len(data) < 16:
        raise LipsyncImportError("file is too small to be a .lipsync")

    p = 0

    def u32():
        nonlocal p
        v = struct.unpack_from('>I', data, p)[0]; p += 4; return v

    def f32():
        nonlocal p
        v = struct.unpack_from('>f', data, p)[0]; p += 4; return v

    def numstring(max_len=4096):
        nonlocal p
        n = u32()
        if n > max_len or p + n > len(data):
            raise LipsyncImportError(f"implausible string length {n} at offset {p-4:#x}")
        s = data[p:p + n].decode('latin-1'); p += n
        return s

    version = u32()
    subversion = u32()

    dta_import = ''
    if subversion == 2:
        dta_import = numstring()
        p += 1          # embedded-DTB flag
        u32()           # unknown
    else:
        f32()           # fps, on the older layout

    viseme_count = u32()
    if viseme_count > 4096:
        raise LipsyncImportError(f"implausible viseme count {viseme_count}")
    visemes = [numstring(256) for _ in range(viseme_count)]

    frame_count = u32()
    byte_count = u32()
    if p + byte_count > len(data):
        raise LipsyncImportError(
            f"keyframe block claims {byte_count} bytes but only {len(data)-p} remain")

    start = p
    frames = []
    for i in range(frame_count):
        if p >= len(data):
            raise LipsyncImportError(f"ran out of data at frame {i} of {frame_count}")
        change_count = data[p]; p += 1
        changes = []
        for _ in range(change_count):
            idx, weight = data[p], data[p + 1]; p += 2
            if idx >= viseme_count:
                raise LipsyncImportError(
                    f"frame {i} references viseme index {idx}, but the table only has "
                    f"{viseme_count} entries")
            changes.append((idx, weight))
        frames.append(changes)

    consumed = p - start
    if consumed != byte_count:
        raise LipsyncImportError(
            f"keyframe stream consumed {consumed} bytes but the header declared "
            f"{byte_count} - the file may be a different .lipsync variant")
    if p != len(data):
        raise LipsyncImportError(f"{len(data)-p} unexpected trailing byte(s)")

    return dict(version=version, subversion=subversion, dta_import=dta_import,
                visemes=visemes, frames=frames)


def expand_weight_tracks(parsed):
    """Turns the sparse change-stream into {viseme_name: [(frame_index, weight_0_to_1)]},
    keeping ONLY the frames where that viseme's weight actually changed.

    Because playback holds the last value until the next change (see module docstring),
    the sparse points plus CONSTANT interpolation reproduce the stream exactly, with no
    need to materialise a value per viseme per frame. For a real song that's the
    difference between a few thousand keyframes and several hundred thousand."""
    visemes = parsed['visemes']
    tracks = {name: [] for name in visemes}
    current = {}
    for frame_idx, changes in enumerate(parsed['frames']):
        for idx, weight in changes:
            name = visemes[idx]
            value = weight / 255.0
            if current.get(name) != value:
                tracks[name].append((frame_idx, value))
                current[name] = value
    return tracks


# ---------------------------------------------------------------------------------------
# Blender: weight channels
# ---------------------------------------------------------------------------------------

def _weight_path(viseme_name):
    return f'["{WEIGHT_PROP_PREFIX}{viseme_name}"]'


def ensure_weight_props(obj, viseme_names):
    """Creates a 0-1 custom property per viseme on the object, so the channels show up as
    real sliders in the sidebar and are keyframeable/editable like anything else."""
    for name in viseme_names:
        key = WEIGHT_PROP_PREFIX + name
        if key not in obj:
            obj[key] = 0.0
        try:
            ui = obj.id_properties_ui(key)
            ui.update(min=0.0, max=1.0, soft_min=0.0, soft_max=1.0,
                      description=f"Blend weight for viseme '{name}'")
        except (AttributeError, TypeError):
            pass    # older API; the property still works, just without UI clamping


def build_lipsync_action(obj, song_name, tracks, interpolation='CONSTANT'):
    """Writes the weight tracks into a new Action as one F-Curve per viseme."""
    action_name = f"LIPSYNC_{song_name}"
    existing = bpy.data.actions.get(action_name)
    if existing is not None:
        bpy.data.actions.remove(existing)
    action = bpy.data.actions.new(action_name)
    action[_LIPSYNC_SONG_PROP] = song_name
    action[_LIPSYNC_VISEMES_PROP] = sorted(tracks.keys())

    slot = action.slots.new(id_type='OBJECT', name=obj.name)
    channelbag = _get_or_create_channelbag(action, slot)

    total_keys = 0
    for name, points in tracks.items():
        if not points:
            continue
        fc = channelbag.fcurves.new(_weight_path(name), index=0)
        grp = channelbag.groups.get("Visemes") or channelbag.groups.new("Visemes")
        fc.group = grp
        for frame_idx, value in points:
            # file frame 0 -> Blender frame 1
            kp = fc.keyframe_points.insert(frame_idx + 1, value, options={'FAST'})
            kp.interpolation = interpolation
        total_keys += len(points)

    return action, total_keys


# ---------------------------------------------------------------------------------------
# Blender: baking weights + viseme poses into a preview Action
# ---------------------------------------------------------------------------------------

def _action_channelbag(action):
    if not len(action.layers):
        return None
    layer = action.layers[0]
    if not len(layer.strips) or not len(action.slots):
        return None
    return layer.strips[0].channelbag(action.slots[0], ensure=False)


def collect_viseme_poses():
    """Gathers every imported viseme Action into
    {viseme_name: {'loc': {bone: [x,y,z]}, 'quat': {bone: [w,x,y,z]}, 'euler': {bone: z}}}.

    Reads the stored keyframe values directly rather than evaluating the Actions, which is
    valid because viseme_importer writes each as a single static pose on frame 1."""
    poses = {}
    for action in bpy.data.actions:
        if not action.get(_MILO_VISEME_TAG):
            continue
        name = action.get(_MILO_VISEME_NAME)
        if not name:
            continue
        cb = _action_channelbag(action)
        if cb is None:
            continue
        entry = {'loc': {}, 'quat': {}, 'euler': {}}
        for fc in cb.fcurves:
            if not len(fc.keyframe_points):
                continue
            path = fc.data_path
            if not path.startswith('pose.bones["'):
                continue
            bone = path[len('pose.bones["'):path.index('"]')]
            value = fc.keyframe_points[0].co[1]
            if path.endswith('.location'):
                entry['loc'].setdefault(bone, [0.0, 0.0, 0.0])[fc.array_index] = value
            elif path.endswith('.rotation_quaternion'):
                entry['quat'].setdefault(bone, [1.0, 0.0, 0.0, 0.0])[fc.array_index] = value
            elif path.endswith('.rotation_euler') and fc.array_index == 2:
                entry['euler'][bone] = value
        poses[name] = entry
    return poses


def blend_pose(active):
    """Blends weighted viseme poses into a single pose.

    `active` is [(weight, pose_entry), ...] for the visemes with non-zero weight.

    Positions sum linearly - they're offsets from rest, so a weighted sum is exactly the
    intended "N shapes applied at N strengths". Rotations are weighted-averaged as
    quaternions and renormalised, with sign alignment against identity so that q and -q
    (the same rotation) can't cancel each other out. Any weight not used up by the active
    visemes is left on identity, so partial weights ease out of the rest pose rather than
    snapping.

    This blends all N at once against the rest pose, which is how the game combines
    visemes - deliberately NOT the sequential composition Blender's NLA stacking would
    apply, since that diverges once two visemes drive the same bone."""
    loc = {}
    euler = {}
    quat_acc = {}
    quat_w = {}

    for weight, entry in active:
        for bone, v in entry['loc'].items():
            acc = loc.setdefault(bone, [0.0, 0.0, 0.0])
            acc[0] += v[0] * weight
            acc[1] += v[1] * weight
            acc[2] += v[2] * weight
        for bone, v in entry['euler'].items():
            euler[bone] = euler.get(bone, 0.0) + v * weight
        for bone, q in entry['quat'].items():
            acc = quat_acc.setdefault(bone, [0.0, 0.0, 0.0, 0.0])
            # align against identity (w>0) so opposite-sign quaternions don't cancel
            sign = -1.0 if q[0] < 0.0 else 1.0
            for i in range(4):
                acc[i] += q[i] * weight * sign
            quat_w[bone] = quat_w.get(bone, 0.0) + weight

    quat = {}
    for bone, acc in quat_acc.items():
        leftover = 1.0 - min(quat_w.get(bone, 0.0), 1.0)
        acc[0] += leftover      # remaining weight rests on identity (1,0,0,0)
        q = Quaternion(acc)
        if q.magnitude < 1e-8:
            q = Quaternion((1.0, 0.0, 0.0, 0.0))
        else:
            q.normalize()
        quat[bone] = q

    return loc, quat, euler


# ---------------------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------------------

class IMPORT_OT_rb3_lipsync(bpy.types.Operator, ImportHelper):
    """Import a Rock Band .lipsync file as editable per-viseme weight channels on the
    active armature. Import the matching viseme set first, then bake a preview to see it"""
    bl_idname = "import_scene.rb3_lipsync"
    bl_label = "Import RB3 Lipsync (.lipsync)"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".lipsync"
    filter_glob: StringProperty(default="*.lipsync", options={'HIDDEN'})

    interpolation: EnumProperty(
        name="Interpolation",
        description="How weights behave between keyframes",
        items=[
            ('CONSTANT', "Constant (accurate)",
             "Hold each weight until the next change, exactly as the game plays the "
             "stream back"),
            ('LINEAR', "Linear (smoother)",
             "Ramp between changes. Easier to hand-edit, but not what the game does"),
        ],
        default='CONSTANT',
    )

    set_scene_range: BoolProperty(
        name="Set Scene Frame Range",
        description="Set the scene's start/end frames to cover the imported lipsync",
        default=True,
    )

    def execute(self, context):
        arm_obj = context.active_object
        if arm_obj is None or arm_obj.type != 'ARMATURE':
            self.report({'ERROR'},
                        "Select the target armature first - weight channels are stored "
                        "on it, alongside its viseme Actions.")
            return {'CANCELLED'}

        try:
            parsed = parse_lipsync(self.filepath)
        except Exception as e:
            _log(f"LIPSYNC IMPORT FAILED: {e}")
            self.report({'ERROR'}, f"Could not parse .lipsync: {e}")
            return {'CANCELLED'}

        import os
        song_name = os.path.splitext(os.path.basename(self.filepath))[0]
        frames = parsed['frames']
        visemes = parsed['visemes']

        _log(f"===== Importing lipsync '{song_name}' from {self.filepath} =====")
        _log(f"  version {parsed['version']}, {len(visemes)} viseme(s), "
             f"{len(frames)} frames ({len(frames)/LIPSYNC_FPS:.1f}s at {LIPSYNC_FPS:g}Hz)")

        # Which of these visemes actually have poses imported?
        available = {a.get(_MILO_VISEME_NAME) for a in bpy.data.actions
                     if a.get(_MILO_VISEME_TAG)}
        missing = [v for v in visemes if v not in available]
        if not available:
            self.report({'ERROR'},
                        "No viseme Actions found. Import the character's viseme set "
                        "(File > Import > Rock Band 3 Viseme Set) before importing "
                        "lipsync - a .lipsync file contains no pose data of its own.")
            return {'CANCELLED'}
        if missing:
            _log(f"  WARNING: {len(missing)} viseme(s) referenced by this song have no "
                 f"imported pose and will contribute nothing: {', '.join(missing[:10])}"
                 + (" ..." if len(missing) > 10 else ""))

        tracks = expand_weight_tracks(parsed)
        ensure_weight_props(arm_obj, visemes)
        action, total_keys = build_lipsync_action(
            arm_obj, song_name, tracks, self.interpolation)

        if arm_obj.animation_data is None:
            arm_obj.animation_data_create()
        arm_obj.animation_data.action = action
        try:
            arm_obj.animation_data.action_slot = action.slots[0]
        except (AttributeError, IndexError):
            pass

        used = sum(1 for t in tracks.values() if t)
        _log(f"  {used} viseme(s) used by this song, {total_keys} weight keyframe(s) "
             f"written (sparse - only where a weight changes)")

        scene = context.scene
        if self.set_scene_range:
            scene.frame_start = 1
            scene.frame_end = len(frames)
        if abs(scene.render.fps / max(scene.render.fps_base, 1e-6) - LIPSYNC_FPS) > 0.01:
            _log(f"  NOTE: scene is {scene.render.fps/scene.render.fps_base:g} fps but "
                 f"lipsync data is {LIPSYNC_FPS:g} Hz - set the scene to 30 fps or "
                 f"playback timing will drift.")

        summary = (f"Imported '{song_name}': {used} viseme channel(s), {total_keys} "
                   f"keyframe(s), {len(frames)} frames")
        if missing:
            summary += f"; {len(missing)} viseme(s) have no pose (see log)"
        _log(f"===== {summary} =====")
        # The weight channels are custom properties - on their own they animate NOTHING
        # visible, because nothing is driving bones off them yet. Say so explicitly:
        # otherwise the obvious next move is to press play, see a motionless face, and
        # reasonably conclude the import failed.
        _log("  NEXT STEP: run Object > Rock Band Lipsync > Bake Lipsync Preview to turn "
             "these weights into bone motion. Until then the face will not move on "
             "playback - the weight channels are data, not deformation.")
        self.report({'WARNING' if missing else 'INFO'},
                    summary + " - now run 'Bake Lipsync Preview' to see it move")
        return {'FINISHED'}


class POSE_OT_bake_lipsync_preview(bpy.types.Operator):
    """Bake the armature's viseme weight channels into a normal pose Action so the
    lipsync can be played back. Re-run this any time the weights are edited"""
    bl_idname = "pose.bake_lipsync_preview"
    bl_label = "Bake Lipsync Preview"
    bl_options = {'REGISTER', 'UNDO'}

    frame_start: IntProperty(name="Start Frame", default=1, min=0)
    frame_end: IntProperty(name="End Frame", default=250, min=0)

    separate_action: BoolProperty(
        name="Bake To Separate Action",
        description="Write the baked bone curves into their own Action instead of "
                     "alongside the weight channels. An object can only have ONE Action "
                     "assigned at a time, so a separate Action has to be swapped in "
                     "manually to preview - and swapping it in detaches the weight "
                     "channels. Leave this off unless you specifically want them split",
        default=False,
    )

    def invoke(self, context, event):
        self.frame_start = context.scene.frame_start
        self.frame_end = context.scene.frame_end
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        arm_obj = context.active_object
        if arm_obj is None or arm_obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Select the armature holding the lipsync weights.")
            return {'CANCELLED'}

        anim = arm_obj.animation_data
        weight_action = anim.action if anim else None
        if weight_action is None:
            self.report({'ERROR'},
                        "No Action assigned - import a .lipsync onto this armature first.")
            return {'CANCELLED'}
        cb = _action_channelbag(weight_action)
        if cb is None:
            self.report({'ERROR'}, "The assigned Action has no animation channels.")
            return {'CANCELLED'}

        # weight fcurves, keyed by viseme name
        weight_fcurves = {}
        prefix = f'["{WEIGHT_PROP_PREFIX}'
        for fc in cb.fcurves:
            if fc.data_path.startswith(prefix) and fc.data_path.endswith('"]'):
                weight_fcurves[fc.data_path[len(prefix):-2]] = fc
        if not weight_fcurves:
            self.report({'ERROR'},
                        "The assigned Action has no viseme weight channels - is this a "
                        "lipsync Action?")
            return {'CANCELLED'}

        poses = collect_viseme_poses()
        if not poses:
            self.report({'ERROR'}, "No viseme Actions found - import the viseme set first.")
            return {'CANCELLED'}

        usable = {n: fc for n, fc in weight_fcurves.items() if n in poses}
        if not usable:
            self.report({'ERROR'},
                        "None of this song's visemes have matching imported poses.")
            return {'CANCELLED'}

        start, end = self.frame_start, self.frame_end
        if end < start:
            self.report({'ERROR'}, "End frame is before start frame.")
            return {'CANCELLED'}

        song = weight_action.get(_LIPSYNC_SONG_PROP, weight_action.name)

        if self.separate_action:
            action_name = f"LIPSYNCBAKE_{song}"
            existing = bpy.data.actions.get(action_name)
            if existing is not None:
                bpy.data.actions.remove(existing)
            bake = bpy.data.actions.new(action_name)
            slot = bake.slots.new(id_type='OBJECT', name=arm_obj.name)
            bake_cb = _get_or_create_channelbag(bake, slot)
        else:
            # Bake INTO the weight Action. A single Action can animate custom properties
            # and pose bones at once, and an object can only have one Action assigned -
            # so keeping both here means playback just works after baking, with the
            # weight channels still sitting right there as the editable source of truth.
            # Baking into a separate Action would force the user to swap Actions to
            # preview, which detaches the very weights they'd want to keep editing.
            action_name = weight_action.name
            bake = weight_action
            bake_cb = cb
            stale = [fc for fc in bake_cb.fcurves
                     if fc.data_path.startswith('pose.bones[')]
            for fc in stale:
                bake_cb.fcurves.remove(fc)
            if stale:
                _log(f"  Cleared {len(stale)} bone curve(s) from a previous bake.")

        pose_bones = arm_obj.pose.bones
        curves = {}

        def curve(path, index, group):
            key = (path, index)
            fc = curves.get(key)
            if fc is None:
                fc = bake_cb.fcurves.new(path, index=index)
                grp = bake_cb.groups.get(group) or bake_cb.groups.new(group)
                fc.group = grp
                curves[key] = fc
            return fc

        _log(f"===== Baking lipsync preview '{song}' frames {start}-{end} =====")
        keys = 0
        for frame in range(start, end + 1):
            active = []
            for name, fc in usable.items():
                w = fc.evaluate(frame)
                if w > 1e-4:
                    active.append((w, poses[name]))
            if not active:
                continue
            loc, quat, euler = blend_pose(active)

            for bone, v in loc.items():
                if bone not in pose_bones:
                    continue
                path = f'pose.bones["{bone}"].location'
                for i in range(3):
                    curve(path, i, bone).keyframe_points.insert(
                        frame, v[i], options={'FAST'})
                keys += 3
            for bone, q in quat.items():
                if bone not in pose_bones:
                    continue
                path = f'pose.bones["{bone}"].rotation_quaternion'
                for i in range(4):
                    curve(path, i, bone).keyframe_points.insert(
                        frame, q[i], options={'FAST'})
                keys += 4
            for bone, z in euler.items():
                if bone not in pose_bones:
                    continue
                curve(f'pose.bones["{bone}"].rotation_euler', 2, bone
                      ).keyframe_points.insert(frame, z, options={'FAST'})
                keys += 1

        for fc in bake_cb.fcurves:
            fc.update()

        summary = (f"Baked into '{action_name}': {end-start+1} frame(s), "
                   f"{len(curves)} curve(s), {keys} keyframe(s)")
        _log(f"===== {summary} =====")
        if self.separate_action:
            _log("  Assign this Action to the armature to preview it. Note that doing so "
                 "detaches the weight channels, which live on the other Action.")
        else:
            _log("  Press play - the bone curves are in the same Action as the weights, "
                 "so no swapping needed. Re-run this after editing any weight.")
        self.report({'INFO'}, summary)
        return {'FINISHED'}


# ---------------------------------------------------------------------------------------
# Menu entries - the bake operator is a required step, so it needs to be findable without
# resorting to F3 search.
# ---------------------------------------------------------------------------------------

class VIEW3D_MT_milo_lipsync(bpy.types.Menu):
    bl_idname = "VIEW3D_MT_milo_lipsync"
    bl_label = "Rock Band Lipsync"

    def draw(self, context):
        self.layout.operator(POSE_OT_bake_lipsync_preview.bl_idname,
                             text="Bake Lipsync Preview", icon='ACTION')


def menu_func_lipsync(self, context):
    obj = context.active_object
    if obj is not None and obj.type == 'ARMATURE':
        self.layout.menu(VIEW3D_MT_milo_lipsync.bl_idname)
