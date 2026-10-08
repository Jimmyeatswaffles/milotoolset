"""
Raw Green Day: Rock Band CharClip export.

Writes Blender Actions as standalone GDRB animation clips - CharClipSamples objects - for
compiling into a GDRB milo by hand, the way the TBRB 'Custom Song Asset' export writes loose
.mesh/.mat/.tex files. Each clip is its own file: the object's bytes (revision, body) followed
by the 0xADDEADDE end marker, named after the clip with no extension (GDRB clip objects carry
none - main_anim, j_e_idle_01) and placed in a CharClipSamples/ folder, the layout milo
extraction tools use. A clip object doesn't store its own name - the file name supplies it - so
characters Windows forbids (Dance Central names contain '->') are replaced in the file name, and
names.txt beside the files maps each file back to the clip's exact name.

===========================================================================================
FORMAT - reproduced against two retail GDRB clips
===========================================================================================
A CharClipSamples object (revision 16) wrapping a CharClip (version 12):

  16, 12, object revision 2, clip type symbol, note symbol, has-tree byte,
  start beat, end beat, beats per second,
  flags, play flags, blend width, range, relative-clip symbol, old version, do-not-compress,
  transitions: node size, vector count, then per vector a clip name and (beat, beat) pairs,
  beat events: count, then (symbol, beat) per event,
  one unidentified byte (1 in every retail clip),
  the 'full' CharBonesSamples (channels that change), the 'one' block (constant channels),
  a 4-byte zero.

The fields after the tempo follow the order of CharClip::Load in the RB3 decompilation and
match both retail clips: main_anim (21 Guns, no transitions) and j_e_idle_01 (a self-loop
transition with 17 beat pairs).

A CharBonesSamples block (version 16): channel count, (name, weight) per channel - positions,
then quaternions, then single-axis Z rotations - seven cumulative per-type counts, compression,
sample count, the frame table (count, then floats), then the samples. Retail clips use two
compressions. At 2 (main_anim): positions as three int16 over a 1300-unit range. At 1
(j_e_idle_01): positions as four floats, the fourth always 0.0. At both: quaternions as four
int16 over 1.0, and Z rotations as one int16 times 0.00061035156 radians. Each sample is padded to a multiple
of 16 bytes. The retail padding holds leftover memory from Harmonix's tools, which the engine
skips by size; this writer pads with zeros.

Verified by round trip: decoding main_anim and j_e_idle_01 and re-encoding them reproduces
every byte except that padding.

===========================================================================================
TIMING
===========================================================================================
Exported clips hold one sample per frame at 30 fps, and the frame table - 30 entries per second
of clip time, each a sample index - is the identity. End beat = seconds x beats per second.

===========================================================================================
CHANNELS
===========================================================================================
Every bone the Action animates is written as its local transform (rest pose x the Action's
pose), using the GDRB convention measured on import - stored quaternions are Blender's own,
unconjugated. Rotations are written as quaternions throughout, including hinge joints that
retail clips store as single-axis rotations. Positions are written only for bones whose local
translation actually moves or differs from rest. Channels that never change go in the 'one'
block, as the game does. Twist bones are left out by default: the game's own solvers drive
them.
"""

import math
import os
import re
import struct

import bpy
from bpy.props import StringProperty, BoolProperty, EnumProperty, FloatProperty
from bpy_extras.io_utils import ExportHelper
from mathutils import Matrix, Quaternion, Vector

from .utilities import _log


CLIP_SAMPLES_REV = 16
CLIP_REV = 12
OBJECT_REV = 2
BONES_SAMPLES_REV = 16
COMPRESSION = 2
POS_SCALE = 1300.0
ROT_CHANNEL_SCALE = 0.00061035156
CLIP_FPS = 30.0
END_MARKER = b'\xAD\xDE\xAD\xDE'
# Characters Windows forbids in file names. Dance Central clip names contain '->'
# ("Chump_L->Chump_R_..."), so those can't be written as-is; see _safe_filename.
_FORBIDDEN = re.compile(r'[<>:"/\\|?*]')

# Header field values and sample compression copied from the two retail clips, for the two
# kinds of clip they are: j_e_idle_01 (a clip-set clip) stores positions as floats
# (compression 1), main_anim (a song performance) as int16 (compression 2).
HEADER_PRESETS = {
    'clip': dict(flags=0x0007f302, play_flags=0x00000200, blend_width=2.0, range_=0.0,
                 relative='', old_version=2, do_not_compress=0, tail_byte=1),
    'performance': dict(flags=0x0007f303, play_flags=0x00020200, blend_width=2.0,
                        range_=0.0, relative='', old_version=2, do_not_compress=0,
                        tail_byte=1),
}
PRESET_COMPRESSION = {'clip': 1, 'performance': 2}


class _W:
    def __init__(self):
        self.b = bytearray()

    def u8(self, v): self.b += struct.pack('>B', v)
    def u32(self, v): self.b += struct.pack('>I', v)
    def i32(self, v): self.b += struct.pack('>i', v)
    def f32(self, v): self.b += struct.pack('>f', v)
    def i16(self, v): self.b += struct.pack('>h', v)

    def sym(self, s):
        raw = s.encode('latin-1')
        self.u32(len(raw))
        self.b += raw


def _q16(v):
    """Inverse of the decoder's max(i / 32767, -1)."""
    return max(-32768, min(32767, int(round(v * 32767.0))))


def encode_bones_samples(channels, samples, frame_times, compression=COMPRESSION):
    """One CharBonesSamples block. `channels`: [(name, weight)] already ordered positions,
    then quaternions, then rotz. `samples`: [{name: value}] (pos (x,y,z), quat (x,y,z,w),
    rotz angle)."""
    pos = [n for n, _ in channels if n.endswith('.pos')]
    quat = [n for n, _ in channels if n.endswith('.quat')]
    rotz = [n for n, _ in channels if n.endswith('.rotz')]
    if [n for n, _ in channels] != pos + quat + rotz:
        raise ValueError("channels must be ordered positions, quaternions, rotz")
    w = _W()
    w.i32(BONES_SAMPLES_REV)
    w.i32(len(channels))
    for name, weight in channels:
        w.sym(name)
        w.f32(weight)
    np_, nq, nz = len(pos), len(quat), len(rotz)
    for c in (0, np_, np_, np_ + nq, np_ + nq, np_ + nq, np_ + nq + nz):
        w.u32(c)
    w.u32(compression)
    w.u32(len(samples))
    w.u32(len(frame_times))
    for f in frame_times:
        w.f32(f)
    if compression not in (1, 2):
        raise ValueError(f"compression {compression} isn't written (retail GDRB uses 1 and 2)")
    pos_size = 16 if compression == 1 else 6
    raw = np_ * pos_size + nq * 8 + nz * 2
    pad = ((raw + 15) & ~15) - raw
    for s in samples:
        for n in pos:
            if compression == 1:
                # Four floats: x, y, z and a fourth that's 0.0 in every retail clip.
                for c in s[n]:
                    w.f32(c)
                w.f32(0.0)
                continue
            for c in s[n]:
                w.i16(_q16(c / POS_SCALE))
        for n in quat:
            for c in s[n]:
                w.i16(_q16(c))
        for n in rotz:
            w.i16(max(-32768, min(32767, int(round(s[n] / ROT_CHANNEL_SCALE)))))
        w.b += b'\x00' * pad
    return bytes(w.b)


def encode_clip(header, full, one):
    """A CharClipSamples object body (no end marker). `header`: clip_type, note,
    start_beat, end_beat, beats_per_sec, the preset fields, transitions [(clip, [(a, b)])]
    and beat_events [(symbol, beat)]. `full`/`one`: (channels, samples, frame_times)."""
    w = _W()
    w.i32(CLIP_SAMPLES_REV)
    w.i32(CLIP_REV)
    w.i32(OBJECT_REV)
    w.sym(header['clip_type'])
    w.sym(header.get('note', ''))
    w.u8(0)                                     # no data tree
    w.f32(header['start_beat'])
    w.f32(header['end_beat'])
    w.f32(header['beats_per_sec'])
    w.u32(header['flags'])
    w.u32(header['play_flags'])
    w.f32(header['blend_width'])
    w.f32(header['range_'])
    w.sym(header['relative'])
    w.i32(header['old_version'])
    w.u8(header['do_not_compress'])
    trans = header.get('transitions', [])
    w.u32(header.get('node_size', 0))
    w.u32(len(trans))
    for clip, pairs in trans:
        w.sym(clip)
        w.u32(len(pairs))
        for a, b in pairs:
            w.f32(a)
            w.f32(b)
    events = header.get('beat_events', [])
    w.u32(len(events))
    for sym, beat in events:
        w.sym(sym)
        w.f32(beat)
    w.u8(header['tail_byte'])
    w.b += encode_bones_samples(*full)
    w.b += encode_bones_samples(*one)
    w.u32(0)
    return bytes(w.b)


def parse_clip_header(c):
    """The header fields encode_clip writes, read back from an existing GDRB clip, and the
    offset where its sample blocks start. Used to verify the writer against retail clips."""
    p = [0]

    def u8():
        v = c[p[0]]; p[0] += 1; return v

    def u32():
        v = struct.unpack_from('>I', c, p[0])[0]; p[0] += 4; return v

    def i32():
        v = struct.unpack_from('>i', c, p[0])[0]; p[0] += 4; return v

    def f32():
        v = struct.unpack_from('>f', c, p[0])[0]; p[0] += 4; return v

    def sym():
        n = u32(); v = c[p[0]:p[0] + n].decode('latin-1'); p[0] += n; return v

    if (i32(), i32(), i32()) != (CLIP_SAMPLES_REV, CLIP_REV, OBJECT_REV):
        raise ValueError("not a GDRB CharClipSamples (16 / 12 / 2)")
    h = dict(clip_type=sym(), note=sym())
    if u8():
        raise ValueError("clip has a data tree")
    h.update(start_beat=f32(), end_beat=f32(), beats_per_sec=f32(), flags=u32(),
             play_flags=u32(), blend_width=f32(), range_=f32(), relative=sym(),
             old_version=i32(), do_not_compress=u8())
    h['node_size'] = u32()
    nv = u32()
    h['transitions'] = []
    for _ in range(nv):
        name = sym()
        n = u32()
        h['transitions'].append((name, [(f32(), f32()) for _ in range(n)]))
    ne = u32()
    h['beat_events'] = [(sym(), f32()) for _ in range(ne)]
    h['tail_byte'] = u8()
    return h, p[0]


# ---------------------------------------------------------------------------------------
# Blender Actions -> clip data
# ---------------------------------------------------------------------------------------

def _action_bones(action):
    """Bone names the Action keys, from its F-curve paths."""
    bones = set()
    for fc in _action_fcurves(action):
        if fc.data_path.startswith('pose.bones["'):
            bones.add(fc.data_path[len('pose.bones["'):fc.data_path.index('"]')])
    return bones


def _action_fcurves(action):
    try:
        for layer in action.layers:
            for strip in layer.strips:
                for slot in action.slots:
                    cb = strip.channelbag(slot, ensure=False)
                    if cb is not None:
                        yield from cb.fcurves
        return
    except AttributeError:
        pass
    yield from action.fcurves


def _local_rest(bone):
    return (bone.parent.matrix_local.inverted() @ bone.matrix_local) if bone.parent \
        else bone.matrix_local


def sample_action(arm_obj, action, scene_fps, include_twist=False):
    """Samples an Action at 30 fps into (channel names, per-frame {channel: value}, seconds).
    Each bone's local transform is its rest pose times the Action's pose basis."""
    curves = {}
    for fc in _action_fcurves(action):
        if not fc.data_path.startswith('pose.bones["'):
            continue
        bone = fc.data_path[len('pose.bones["'):fc.data_path.index('"]')]
        prop = fc.data_path.rsplit('.', 1)[-1]
        curves.setdefault(bone, {}).setdefault(prop, {})[fc.array_index] = fc
    names = sorted(b for b in curves
                   if b in arm_obj.pose.bones and (include_twist or 'Twist' not in b))
    f0, f1 = action.frame_range
    seconds = max(0.0, (f1 - f0) / scene_fps)
    n = int(math.floor(seconds * CLIP_FPS + 1e-6)) + 1
    frames = []
    for k in range(n):
        bf = f0 + k / CLIP_FPS * scene_fps
        sample = {}
        for b in names:
            c = curves[b]
            bone = arm_obj.pose.bones[b].bone
            rl = _local_rest(bone)
            loc = Vector([c['location'][i].evaluate(bf) if 'location' in c and i in c['location']
                          else 0.0 for i in range(3)])
            if 'rotation_quaternion' in c:
                q = Quaternion([c['rotation_quaternion'][i].evaluate(bf)
                                if i in c['rotation_quaternion'] else (1.0 if i == 0 else 0.0)
                                for i in range(4)])
                q.normalize()
            else:
                q = Quaternion((1.0, 0.0, 0.0, 0.0))
            rest_q = rl.to_quaternion()
            local_q = rest_q @ q
            local_q.normalize()
            local_t = rl.to_translation() + rest_q @ loc
            stem = b.rsplit('.', 1)[0] if b.endswith(('.mesh', '.trans')) else b
            sample[stem + '.quat'] = (local_q[1], local_q[2], local_q[3], local_q[0])
            sample[stem + '.pos'] = (local_t.x, local_t.y, local_t.z)
        frames.append(sample)
    return names, frames, seconds


def split_channels(frames, rest_pos, pos_tol=1e-3, rot_tol=1e-4):
    """Chooses channels and splits them between the 'full' (changing) and 'one' (constant)
    blocks, as retail clips do. Positions are kept only where they move or differ from rest;
    quaternion signs are made continuous so int16 storage doesn't flip between samples."""
    if not frames:
        return [], []
    keys = list(frames[0])
    keep = []
    for k in keys:
        vals = [f[k] for f in frames]
        if k.endswith('.pos'):
            r = rest_pos.get(k[:-4])
            moves = any(max(abs(a - b) for a, b in zip(v, vals[0])) > pos_tol for v in vals)
            off_rest = r is not None and max(abs(a - b) for a, b in zip(vals[0], r)) > pos_tol
            if not (moves or off_rest):
                continue
        keep.append(k)
    for k in keep:
        if k.endswith('.quat'):
            prev = None
            for f in frames:
                q = f[k]
                if prev is not None and sum(a * b for a, b in zip(q, prev)) < 0.0:
                    q = tuple(-x for x in q)
                    f[k] = q
                prev = q
    full, one = [], []
    for k in keep:
        vals = [f[k] for f in frames]
        tol = pos_tol if k.endswith('.pos') else rot_tol
        const = all(max(abs(a - b) for a, b in zip(v, vals[0])) <= tol for v in vals)
        (one if const else full).append(k)
    order = lambda ks: sorted([k for k in ks if k.endswith('.pos')]) + \
        sorted([k for k in ks if k.endswith('.quat')])
    return order(full), order(one)


def build_clip_bytes(arm_obj, action, scene_fps, clip_type, preset, bpm, include_twist):
    names, frames, seconds = sample_action(arm_obj, action, scene_fps, include_twist)
    rest_pos = {}
    for b in names:
        t = _local_rest(arm_obj.pose.bones[b].bone).to_translation()
        stem = b.rsplit('.', 1)[0] if b.endswith(('.mesh', '.trans')) else b
        rest_pos[stem] = (t.x, t.y, t.z)
    full_k, one_k = split_channels(frames, rest_pos)
    n = len(frames)
    frame_table = [float(i) for i in range(n)]
    comp = PRESET_COMPRESSION[preset]
    full = ([(k, 1.0) for k in full_k], [{k: f[k] for k in full_k} for f in frames],
            frame_table, comp)
    one = ([(k, 1.0) for k in one_k], [{k: frames[0][k] for k in one_k}] if one_k else [],
           [], comp)
    bps = bpm / 60.0
    header = dict(HEADER_PRESETS[preset], clip_type=clip_type, note='', start_beat=0.0,
                  end_beat=seconds * bps, beats_per_sec=bps, transitions=[],
                  beat_events=[], node_size=0)
    return encode_clip(header, full, one), dict(samples=n, seconds=seconds,
                                                 full=len(full_k), one=len(one_k))


def _safe_filename(name):
    """A clip name made safe as a file name on every platform. The object's real name is
    kept in names.txt beside the files."""
    return _FORBIDDEN.sub('_', name).rstrip('. ') or "clip"


class EXPORT_OT_gdrb_raw_charclips(bpy.types.Operator, ExportHelper):
    """Export Actions as raw Green Day: Rock Band CharClip objects (CharClipSamples) - loose
    files to compile into a GDRB milo yourself, like the TBRB custom song asset export"""
    bl_idname = "export_scene.gdrb_raw_charclips"
    bl_label = "Export GDRB Raw CharClips"
    bl_options = {'REGISTER'}

    filename_ext = ""
    filter_glob: StringProperty(default="*", options={'HIDDEN'})

    which: EnumProperty(
        name="Actions",
        items=[('ACTIVE', "Active Action", "The armature's active Action only"),
               ('CLIPS', "All Imported Clips", "Every CLIP_ Action imported onto this "
                                               "armature")],
        default='CLIPS',
    )
    preset: EnumProperty(
        name="Clip Kind",
        description="Header flags copied from a retail GDRB clip of this kind",
        items=[('clip', "Clip (like j_e_idle_01)", "Flags from a retail clip-set clip"),
               ('performance', "Performance (like main_anim)",
                "Flags from a retail song performance")],
        default='clip',
    )
    clip_type: StringProperty(
        name="Clip Type",
        description="The clip's type symbol. Retail GDRB clips use the CharClipSet's type, "
                     "e.g. guitar_body_right",
        default="guitar_body_right",
    )
    bpm: FloatProperty(
        name="Tempo (BPM)",
        description="Written as the clip's beats per second (bpm / 60); the end beat is the "
                     "clip's length at this tempo",
        default=120.0, min=1.0, max=400.0,
    )
    include_twist: BoolProperty(
        name="Include Twist Bones",
        description="Also write twist bones. Off by default: GDRB's own solvers drive them",
        default=False,
    )

    def execute(self, context):
        arm = context.active_object
        if arm is None or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Select the armature the Actions were imported onto.")
            return {'CANCELLED'}
        if self.which == 'ACTIVE':
            act = arm.animation_data.action if arm.animation_data else None
            actions = [act] if act else []
        else:
            actions = [a for a in bpy.data.actions if a.get("milo_clip")]
        if not actions:
            self.report({'ERROR'}, "No Actions to export.")
            return {'CANCELLED'}
        out_dir = os.path.join(os.path.dirname(bpy.path.abspath(self.filepath)),
                               "CharClipSamples")
        os.makedirs(out_dir, exist_ok=True)
        scene = context.scene
        fps = scene.render.fps / max(scene.render.fps_base, 1e-6)
        _log(f"===== Exporting {len(actions)} GDRB raw CharClip(s) to {out_dir} =====")
        written = 0
        manifest = []
        for act in actions:
            name = act.get("milo_clip_name") or (act.name[5:] if act.name.startswith("CLIP_")
                                                 else act.name)
            try:
                data, info = build_clip_bytes(arm, act, fps, self.clip_type, self.preset,
                                              self.bpm, self.include_twist)
            except Exception as e:
                _log(f"  SKIPPED {act.name}: {e}")
                continue
            fname = _safe_filename(name)
            with open(os.path.join(out_dir, fname), 'wb') as f:
                f.write(data + END_MARKER)
            manifest.append((fname, name))
            if fname != name:
                _log(f"  '{name}' written as '{fname}' (characters Windows forbids replaced)")
            written += 1
            _log(f"  '{name}': {info['samples']} sample(s), {info['seconds']:.2f}s, "
                 f"{info['full']} animated + {info['one']} constant channel(s), "
                 f"{len(data) + 4} bytes")
        # The clip object doesn't store its own name - it comes from the file - so record
        # each file's original clip name for whatever compiles these into a milo.
        with open(os.path.join(out_dir, "names.txt"), 'w', encoding='utf-8') as f:
            f.write("# file name<TAB>CharClipSamples object name\n")
            for fname, name in manifest:
                f.write(f"{fname}\t{name}\n")
        summary = f"Wrote {written} raw CharClip(s) to {out_dir}"
        _log(f"===== {summary} =====")
        self.report({'INFO'}, summary)
        return {'FINISHED'}
