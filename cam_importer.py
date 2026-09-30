"""
Green Day: Rock Band motion-capture camera importer.

Reads a song's camera milo (e.g. 21guns_cams.milo_xbox) and creates one Blender camera per
mocap camera, keyed with its movement, rotation and per-frame field of view.

===========================================================================================
WHAT'S IN THE FILE - verified against a retail 21guns_cams.milo_xbox
===========================================================================================
A revision-25 WorldDir. Cameras are NOT CharClips - they use the engine's general render
animation objects:

  cam_a.tnm / cam_b.tnm / cam_c.tnm   RndTransAnim - position and rotation keys
  cam_a.cnm / cam_b.cnm / cam_c.cnm   RndCamAnim   - field-of-view keys
  MocapCam1_DoF.anim ...              PropAnim     - depth of field (not imported yet)
  MocapCam1.shot ... (15 of them)     BandCamShot  - shot definitions for the game's camera
                                                     director; not needed to play a camera
  mocap.cam                           RndCam       - the one camera the shots drive

Each TransAnim/CamAnim pair shares a stem (cam_a.tnm + cam_a.cnm) and makes one camera.
MocapCam1.shot confirms the pairing: it references cam_a.tnm and cam_a.cnm together.

Both formats decode to the last byte against the RB3 decompilation's Load() functions
(RndTransAnim revision 7, RndCamAnim revision 2), and a clean parse is asserted rather
than assumed. Keys are stored value-then-frame (math/Key.h), quaternions x,y,z,w
(math/Mtx.h).

===========================================================================================
TIMING
===========================================================================================
Every track in the sampled file has RndAnimatable rate 0 - k30_fps, i.e. real seconds at 30
frames per second, NOT beats (rate 1 would be 480 frames per beat). So no tempo handling is
needed: a key at frame N lands at N/30 seconds, converted to the scene's own frame rate.
The tracks run to frame 9879 (329.3 s), essentially the whole song.

The TransAnim's spline, slerp and follow-path flags are all off, so keys are meant to be
interpolated linearly. That's what the Blender curves use. Quaternion keys are sign-aligned
with their predecessor first, since linear interpolation between q and a nearby -q would
swing the long way round.

===========================================================================================
ORIENTATION - from the engine's projection code, confirmed against the data
===========================================================================================
RndCam::UpdateLocal builds the projection so screen depth comes from local +Y, horizontal
from +X, and vertical from -Z: an engine camera looks down +Y with +Z up. A Blender camera
looks down -Z with +Y up, so each rotation is followed by a fixed +90 degree turn about X.

The stored quaternions are used without conjugation, the same as the body performance. That
was tested rather than assumed: with rotations as stored, each camera's sight lines converge
on a point that lies in front of the camera for 99.5-100% of its keys; conjugated, the same
point is in front for only 14-66%. Up vectors come out 0.92-0.98 toward world +Z either way.

===========================================================================================
FIELD OF VIEW
===========================================================================================
CamAnim keys are the camera's mYFov: a VERTICAL angle in radians (the projection applies it
to the vertical axis and scales horizontal by aspect). Blender cameras are set to vertical
sensor fit and keyed on focal length, which is the animatable property.

The tracks contain real zooms, not just a fixed lens. cam_a and cam_c stay near 57-60
degrees, but cam_b eases from 42 degrees down to a telephoto hold around 7-8, then further
through 1 degree, briefly touching or crossing zero before easing back out; it makes two
more zooms later (to about 23 and 10 degrees). So every positive angle is kept, focal
lengths beyond Blender's 5000 mm limit are held there, and only zero or negative angles are
skipped - those, plus a single 0.0 key on cam_b's final frame that looks like an
end-of-track marker.

===========================================================================================
PLACEMENT
===========================================================================================
The cameras are free objects in the venue's world space, not tied to a bone: every
TransAnim's target field is empty, and mocap.cam is parented to the directory root
('21guns_cams') with an identity rest transform. They're recreated under an Empty of that
name. The band's placement on stage lives in the venue's files, not this one, and the
cameras' sight lines don't converge on the origin, so the rig may need moving to line up
with an imported character - moving the one Empty does that for all three.

===========================================================================================
KNOWN GAPS
===========================================================================================
1. Depth of field (the PropAnim tracks) isn't imported.
2. Beat-rate tracks (rate 1) are rejected rather than converted; none appear in the sample.
3. Each TransAnim carries one scale key; cameras don't scale, so it's read but not applied.
"""

import math
import os
import struct

import bpy
from bpy.props import StringProperty, BoolProperty
from bpy_extras.io_utils import ImportHelper
from mathutils import Quaternion

from .io import read_milo_container_body
from .utilities import _log
from .mesh_importer import _read_dir_entries, _entry_spans
from .viseme_importer import _get_or_create_channelbag


REV25 = 25
TRANSANIM_REV = 7
CAMANIM_REV = 2
# RndAnimatable::Rate values that count real time at 30 fps (k30_fps, k30_fps_ui,
# k30_fps_tutorial). Rate 1 (k480_fpb) and 3 (k1_fpb) count beats instead.
_SECONDS_RATES = {0: 30.0, 2: 30.0, 4: 30.0}
# Blender's focal length tops out at 5000 mm (a vertical FOV of about 0.27 degrees on the
# sensor used here). Extreme-telephoto keys are held at that limit rather than rejected.
_MAX_FOCAL = 5000.0
# Blender sensor height used with vertical sensor fit; any value works, since the focal
# length is derived from it.
_SENSOR_HEIGHT = 24.0
# Viewport display size for the camera objects. Blender's default of 1 unit is tiny at game
# scale - Billie Joe stands about 70 units tall - so the cameras drew as specks. This only
# changes how the camera is drawn, not what it sees.
_DISPLAY_SIZE = 10.0
# Engine camera looks down +Y with +Z up; Blender's looks down -Z with +Y up.
_ENGINE_TO_BLENDER_CAM = Quaternion((math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0))


class CamImportError(Exception):
    """Raised when a camera milo can't be read."""
    pass


class _Reader:
    def __init__(self, data):
        self.d = data
        self.p = 0

    def u8(self):
        v = self.d[self.p]; self.p += 1; return v

    def u32(self):
        v = struct.unpack_from('>I', self.d, self.p)[0]; self.p += 4; return v

    def f32(self):
        v = struct.unpack_from('>f', self.d, self.p)[0]; self.p += 4; return v

    def sym(self):
        n = self.u32()
        if n > 1024 or self.p + n > len(self.d):
            raise CamImportError(f"implausible string length {n} at offset {self.p - 4}")
        s = self.d[self.p:self.p + n].decode('latin-1'); self.p += n
        return s

    def floats(self, count, per):
        need = count * per * 4
        if count > 1_000_000 or self.p + need > len(self.d):
            raise CamImportError(f"key array of {count} runs past the end of the object")
        fmt = '>' + 'f' * per
        out = [struct.unpack_from(fmt, self.d, self.p + i * per * 4) for i in range(count)]
        self.p += need
        return out


def _read_object_header(r):
    """Hmx::Object metadata as saved in revision-25 files."""
    rev = r.u32()
    r.sym()                      # type
    if rev > 0:
        r.sym()                  # note
    if r.u8():
        raise CamImportError("object embeds a DTB tree, which isn't parsed")


def _read_animatable(r):
    """RndAnimatable header. Returns (frame, rate)."""
    r.u32()                      # revision
    frame = r.f32()
    rate = r.u32()
    return frame, rate


def parse_trans_anim(data, name):
    """RndTransAnim revision 7. Returns dict(rot=[(x,y,z,w,frame)], trans=[(x,y,z,frame)],
    scale=[...], rate, trans_spline, rot_slerp, rot_spline)."""
    r = _Reader(data)
    rev = r.u32()
    if rev != TRANSANIM_REV:
        raise CamImportError(f"'{name}' is TransAnim revision {rev}; only "
                             f"{TRANSANIM_REV} is supported")
    _read_object_header(r)
    _frame, rate = _read_animatable(r)
    r.sym()                              # mTrans - the driven object; empty here
    rot = r.floats(r.u32(), 5)           # Key<Quat>: x, y, z, w, frame
    trans = r.floats(r.u32(), 4)         # Key<Vector3>: x, y, z, frame
    r.sym()                              # mKeysOwner
    trans_spline = r.u8()
    r.u8()                               # mRepeatTrans
    scale = r.floats(r.u32(), 4)
    r.u8()                               # mScaleSpline
    r.u8()                               # mFollowPath
    rot_slerp = r.u8()
    rot_spline = r.u8()
    if r.p != len(data):
        raise CamImportError(f"'{name}' has {len(data) - r.p} unexpected trailing byte(s)")
    return dict(rot=rot, trans=trans, scale=scale, rate=rate,
                trans_spline=bool(trans_spline), rot_slerp=bool(rot_slerp),
                rot_spline=bool(rot_spline))


def parse_cam_anim(data, name):
    """RndCamAnim revision 2. Returns dict(fov=[(fov_radians, frame)], rate)."""
    r = _Reader(data)
    rev = r.u32()
    if rev != CAMANIM_REV:
        raise CamImportError(f"'{name}' is CamAnim revision {rev}; only "
                             f"{CAMANIM_REV} is supported")
    _read_object_header(r)
    _frame, rate = _read_animatable(r)
    r.sym()                              # mCam - empty here
    fov = r.floats(r.u32(), 2)           # Key<float>: value, frame
    r.sym()                              # mKeysOwner
    if r.p != len(data):
        raise CamImportError(f"'{name}' has {len(data) - r.p} unexpected trailing byte(s)")
    return dict(fov=fov, rate=rate)


def _camera_clip(data, dir_name):
    """Near/far clip from the RndCam, located just after its parent reference (the
    directory root). Returns (near, far) or None if they can't be found plausibly."""
    tag = struct.pack('>I', len(dir_name)) + dir_name.encode('latin-1')
    i = data.find(tag)
    if i < 0 or i + len(tag) + 12 > len(data):
        return None
    near, far, _yfov = struct.unpack_from('>3f', data, i + len(tag))
    if 0.0 < near < far < 1e7:
        return near, far
    return None


def parse_camera_milo(filepath):
    """Returns (dir_name, [camera dicts], clip or None, [(name, reason)] failures)."""
    with open(filepath, 'rb') as f:
        body = read_milo_container_body(f.read())
    revision = struct.unpack_from('>I', body, 0)[0]
    if revision != REV25:
        raise CamImportError(f"this milo is revision {revision}, not {REV25} (GDRB/TBRB)")
    _rev, _dtype, dir_name, entries = _read_dir_entries(body)
    spans = _entry_spans(body, len(entries))

    trans, fovs, failed = {}, {}, []
    clip = None
    for (etype, ename), (s, e) in zip(entries, spans):
        stem = ename.rsplit('.', 1)[0]
        try:
            if etype == 'TransAnim':
                trans[stem] = parse_trans_anim(body[s:e], ename)
            elif etype == 'CamAnim':
                fovs[stem] = parse_cam_anim(body[s:e], ename)
            elif etype == 'Cam' and clip is None:
                clip = _camera_clip(body[s:e], dir_name)
        except (CamImportError, struct.error, IndexError) as ex:
            failed.append((ename, str(ex)))

    cameras = []
    for stem in sorted(set(trans) | set(fovs)):
        cameras.append(dict(name=stem, trans=trans.get(stem), fov=fovs.get(stem)))
    return dir_name, cameras, clip, failed


# ---------------------------------------------------------------------------------------
# Blender construction
# ---------------------------------------------------------------------------------------

def _new_action(name, id_type, owner_name):
    existing = bpy.data.actions.get(name)
    if existing is not None:
        bpy.data.actions.remove(existing)
    action = bpy.data.actions.new(name)
    slot = action.slots.new(id_type=id_type, name=owner_name)
    return action, slot, _get_or_create_channelbag(action, slot)


def _assign(id_block, action, slot):
    if id_block.animation_data is None:
        id_block.animation_data_create()
    id_block.animation_data.action = action
    try:
        id_block.animation_data.action_slot = slot
    except (AttributeError, TypeError):
        pass


def _curve(channelbag, path, index, group):
    fc = channelbag.fcurves.new(path, index=index)
    grp = channelbag.groups.get(group) or channelbag.groups.new(group)
    fc.group = grp
    return fc


def _frame_map(rate, name, scene_fps):
    fpu = _SECONDS_RATES.get(rate)
    if fpu is None:
        raise CamImportError(
            f"'{name}' uses animation rate {rate}, which counts beats rather than seconds "
            f"and needs a tempo to convert - not supported yet")
    return lambda f: 1.0 + f / fpu * scene_fps


def build_camera(context, cam, parent, clip, scene_fps):
    """Creates one camera object with its movement and lens Actions. Returns
    (object, counts dict)."""
    name = cam['name']
    data = bpy.data.cameras.new(name)
    data.sensor_fit = 'VERTICAL'
    data.sensor_height = _SENSOR_HEIGHT
    data.display_size = _DISPLAY_SIZE
    if clip:
        data.clip_start, data.clip_end = clip
    obj = bpy.data.objects.new(name, data)
    context.collection.objects.link(obj)
    obj.parent = parent
    obj.rotation_mode = 'QUATERNION'
    counts = {'rot': 0, 'trans': 0, 'fov': 0, 'fov_skipped': 0, 'fov_clamped': 0}

    ta = cam['trans']
    if ta:
        to_frame = _frame_map(ta['rate'], name, scene_fps)
        action, slot, cb = _new_action(f"CAM_{name}", 'OBJECT', obj.name)
        interp = 'BEZIER' if ta['trans_spline'] else 'LINEAR'
        loc = [_curve(cb, 'location', i, "Transform") for i in range(3)]
        for x, y, z, f in ta['trans']:
            bf = to_frame(f)
            for i, v in enumerate((x, y, z)):
                loc[i].keyframe_points.insert(bf, v, options={'FAST'}).interpolation = interp
        counts['trans'] = len(ta['trans'])

        rinterp = 'BEZIER' if ta['rot_spline'] else 'LINEAR'
        rq = [_curve(cb, 'rotation_quaternion', i, "Transform") for i in range(4)]
        prev = None
        for x, y, z, w, f in ta['rot']:
            q = Quaternion((w, x, y, z)) @ _ENGINE_TO_BLENDER_CAM
            if prev is not None and sum(a * b for a, b in zip(q, prev)) < 0.0:
                q = Quaternion((-q[0], -q[1], -q[2], -q[3]))
            prev = q
            bf = to_frame(f)
            for i in range(4):
                rq[i].keyframe_points.insert(bf, q[i], options={'FAST'}).interpolation = rinterp
        counts['rot'] = len(ta['rot'])
        for fc in loc + rq:
            fc.update()
        _assign(obj, action, slot)

    ca = cam['fov']
    if ca:
        to_frame = _frame_map(ca['rate'], name, scene_fps)
        action, slot, cb = _new_action(f"CAMLENS_{name}", 'CAMERA', data.name)
        lens = _curve(cb, 'lens', 0, "Lens")
        for fov, f in ca['fov']:
            # Only a zero or negative angle is unusable. Small positive angles are real
            # extreme-telephoto work: cam_b zooms smoothly from 42 degrees down through
            # 8, then 1, touching zero at the far end before easing back out.
            if not (0.0 < fov < math.pi):
                counts['fov_skipped'] += 1
                continue
            focal = (_SENSOR_HEIGHT * 0.5) / math.tan(fov * 0.5)
            if focal > _MAX_FOCAL:
                focal = _MAX_FOCAL
                counts['fov_clamped'] += 1
            lens.keyframe_points.insert(to_frame(f), focal,
                                        options={'FAST'}).interpolation = 'LINEAR'
            counts['fov'] += 1
        lens.update()
        _assign(data, action, slot)

    return obj, counts


class IMPORT_OT_gdrb_cameras(bpy.types.Operator, ImportHelper):
    """Import the motion-captured cameras from a Green Day: Rock Band song camera milo, with
    their movement, rotation and zoom. Each camera becomes a Blender camera under one Empty,
    so the whole rig can be moved into line with the band"""
    bl_idname = "import_scene.gdrb_cameras"
    bl_label = "Import GDRB Cameras Milo"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".milo_xbox"
    filter_glob: StringProperty(default="*.milo_xbox;*.milo_ps3", options={'HIDDEN'})

    set_scene_range: BoolProperty(
        name="Extend Scene Frame Range",
        description="Extend the scene's end frame to cover the longest camera track",
        default=True,
    )

    def execute(self, context):
        try:
            dir_name, cameras, clip, failed = parse_camera_milo(self.filepath)
        except Exception as e:
            _log(f"GDRB CAMERA IMPORT FAILED: {e}")
            self.report({'ERROR'}, f"Could not parse camera milo: {e}")
            return {'CANCELLED'}
        if not cameras:
            self.report({'ERROR'}, "No camera animation (TransAnim/CamAnim) found in this milo.")
            return {'CANCELLED'}

        scene = context.scene
        scene_fps = scene.render.fps / max(scene.render.fps_base, 1e-6)
        _log(f"===== Importing GDRB cameras '{dir_name}' from {self.filepath} =====")

        root = bpy.data.objects.new(dir_name, None)
        root.empty_display_type = 'PLAIN_AXES'
        context.collection.objects.link(root)

        made = 0
        last = 0.0
        for cam in cameras:
            try:
                _obj, c = build_camera(context, cam, root, clip, scene_fps)
            except CamImportError as e:
                failed.append((cam['name'], str(e)))
                continue
            made += 1
            for key in ('trans', 'fov'):
                tr = cam[key]
                if tr:
                    arr = tr['trans'] if key == 'trans' else tr['fov']
                    if arr:
                        last = max(last, 1.0 + arr[-1][-1] / 30.0 * scene_fps)
            note = ""
            if c['fov_skipped']:
                note += f", {c['fov_skipped']} zero/negative lens key(s) skipped"
            if c['fov_clamped']:
                note += (f", {c['fov_clamped']} extreme-zoom key(s) held at Blender's "
                         f"{_MAX_FOCAL:g} mm limit")
            _log(f"  '{cam['name']}': {c['trans']} position, {c['rot']} rotation, "
                 f"{c['fov']} lens key(s){note}")

        for nm, reason in failed:
            _log(f"  SKIPPED {nm}: {reason}")
        if self.set_scene_range and last > scene.frame_end:
            scene.frame_end = int(math.ceil(last))

        summary = (f"Imported {made} camera(s) from '{dir_name}' under Empty '{root.name}'"
                   + (f"; {len(failed)} failed (see log)" if failed else ""))
        _log(f"===== {summary} =====")
        _log("  Cameras are in the venue's world space. If they don't frame the band, move "
             "the Empty to line them up - the band's stage position isn't in this file.")
        self.report({'WARNING' if failed else 'INFO'}, summary)
        return {'FINISHED'}
