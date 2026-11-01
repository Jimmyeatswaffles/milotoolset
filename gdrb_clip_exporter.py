"""
Green Day: Rock Band character body performance export.

Writes a Blender Action as a complete GDRB song body milo - the per-song mocap file each band
member's 'song' outfit category loads (char/<member>/.../song/<song>.milo, or a custom path
through outfitloader.dta's create_path). The file matches the retail layout of
americanidiot.milo_xbox:

  uncompressed milo, revision 25, a CharClipSet named 'body' (its own object body is the same
  429 bytes in every retail and working file, kept verbatim below), holding in this order
    CharClipGroup   song_<song>        -> main_anim   (group field 'which' = -1)
    CharClipGroup   song_<song>_ramp   -> ramp        (group field 'which' = 0)
    CharClipSamples main_anim          the whole-song performance (play flags 0x20200)
    CharClipSamples ramp               the alternate clip retail uses with the stage-ramp
                                       waypoint configs (play flags 0x200)
  stored as two blocks: everything up to main_anim's end marker, then ramp. Without a ramp
  clip the file is the single-group layout of the Good Riddance export that played in game.

The game finds the performance by listing the clip set's CharClipGroups whose names start with
'song_' (the director's get_song_clips), so the group names follow song_<short name>.

The clips themselves are encoded exactly as the earlier raw CharClip export did, template mode
included; only the container is new.

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

# Header field values and sample compression copied from the two retail clips, for the two
# kinds of clip they are: j_e_idle_01 (a clip-set clip) stores positions as floats
# (compression 1), main_anim (a song performance) as int16 (compression 2).
HEADER_PRESETS = {
    'clip': dict(flags=0x0007f302, play_flags=0x00000200, blend_width=2.0, range_=0.0,
                 relative='', old_version=2, do_not_compress=0, tail_byte=1),
    'performance': dict(flags=0x0007f303, play_flags=0x00020200, blend_width=2.0,
                        range_=0.0, relative='', old_version=2, do_not_compress=0,
                        tail_byte=1),
    # americanidiot.milo_xbox's 'ramp': the performance's flags without play flag 0x20000.
    'ramp': dict(flags=0x0007f303, play_flags=0x00000200, blend_width=2.0, range_=0.0,
                 relative='', old_version=2, do_not_compress=0, tail_byte=1),
}
PRESET_COMPRESSION = {'clip': 1, 'performance': 2, 'ramp': 2}
# Retail song performances all run their clips at 1.5 beats per second (90 BPM),
# whatever the song's tempo.
PERFORMANCE_BPM = 90.0


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


# ---------------------------------------------------------------------------------------
# Template mode: match the clip being replaced
# ---------------------------------------------------------------------------------------
#
# A character doesn't take its animation channels from each clip. When a clip type is set
# (guitar_body_right ...), CharServoBone::SetClipType calls CharBoneDir::StuffBones, which
# fills the character's channel list from that clip type's CharBoneDir resource - so every
# clip of the type is expected to fit one fixed layout of channel names AND types (RB3
# decompilation, system/char). Retail clips are exported against it. The default export
# above isn't: it writes every rotation as a quaternion, so a hinge the layout declares as
# rotz (knees, a forearm, every finger joint) arrives as .quat, and it carries only the bones
# the Action animates.
#
# That was the difference between Good Riddance's own performance and an exported
# replacement that crashed the game the moment it began to play: the original fits the
# layout exactly (12 pos, 36 quat, 25 rotz channels); the export had 26 of its 54 shared bones
# in a different channel type and lacked the guitar, mic, mic-stand, hand-target and facing
# bones. The routine that copies a clip's channels into the layout isn't decompiled, so the
# exact failing line can't be shown, but a clip that doesn't fit the layout is the strongest
# explanation for a crash on the first frame of playback.
#
# So with a template - the original clip being replaced - the export writes exactly the
# template's channels (names, types, order, weights), its header settings (clip type, flags,
# play flags, tempo, compression) and, optionally, its length. Hinge channels are converted
# from the Action's rotation; bones the Action doesn't animate hold the template's own first
# pose. The end marker follows the template file, so the output matches whatever extraction
# tool produced it.

HINGE_OFF_AXIS_WARN = 0.05      # x/y quaternion magnitude (about 5.7 degrees) worth reporting


def read_template_clip(path, clip_name='main_anim'):
    """The pieces of an existing GDRB clip a template export copies. `path` is either a raw
    CharClipSamples file or a GDRB body milo, from which the clip named `clip_name` is used."""
    from .viseme_importer import _Reader, _read_char_bones_samples
    with open(path, 'rb') as f:
        raw = f.read()
    if len(raw) >= 4 and struct.unpack_from('<I', raw, 0)[0] in _MILO_MAGICS:
        clips = read_body_milo_clips(raw)
        if clip_name not in clips:
            raise ValueError(f"no CharClipSamples '{clip_name}' in {os.path.basename(path)} "
                             f"(it has: {', '.join(clips) or 'none'})")
        raw = clips[clip_name] + END_MARKER
    has_marker = raw.endswith(END_MARKER)
    c = raw[:-4] if has_marker else raw
    header, off = parse_clip_header(c)
    r = _Reader(c, off)
    full = _read_char_bones_samples(r)
    one = _read_char_bones_samples(r)
    pose = {}
    for blk in (one, full):
        if blk['samples']:
            pose.update({k: v for k, (_kind, v) in blk['samples'][0].items()})
    channels = [(n, w) for n, w in full['bones'] + one['bones']]
    return dict(header=header, channels=channels, pose=pose, compression=full['compression'],
                has_marker=has_marker, seconds=(header['end_beat'] - header['start_beat'])
                / max(header['beats_per_sec'], 1e-6))


def _hinge_angle(q_xyzw):
    """The hinge angle whose pure Z rotation is closest to a local rotation - the exact
    inverse of how the importer builds a hinge (Quaternion(cos a/2, 0, 0, sin a/2)) - and
    how far the rotation strays off the Z axis."""
    x, y, z, w = q_xyzw
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    return 2.0 * math.atan2(z, w), math.sqrt(x * x + y * y)


def build_clip_from_template(arm_obj, action, scene_fps, template, include_twist=False,
                             pad=True):
    names, frames, seconds = sample_action(arm_obj, action, scene_fps, include_twist)
    if pad:
        want = int(round(template['seconds'] * CLIP_FPS)) + 1
        while frames and len(frames) < want:
            frames.append(dict(frames[-1]))
    channels = template['channels']
    off_axis = {}
    samples = []
    prev_q = {}
    for f in frames:
        s = {}
        for ch, _w in channels:
            stem, kind = ch.rsplit('.', 1)
            if kind == 'pos':
                s[ch] = f.get(ch, template['pose'][ch])
            elif kind == 'quat':
                q = f.get(ch)
                if q is None:
                    q = template['pose'][ch]
                last = prev_q.get(ch)
                if last is not None and sum(a * b for a, b in zip(q, last)) < 0.0:
                    q = tuple(-c for c in q)
                prev_q[ch] = q
                s[ch] = q
            else:                                   # rotz (rotx/roty would go the same way)
                q = f.get(stem + '.quat')
                if q is None:
                    s[ch] = template['pose'][ch]
                else:
                    ang, stray = _hinge_angle(q)
                    off_axis[stem] = max(off_axis.get(stem, 0.0), stray)
                    s[ch] = ang
        samples.append(s)

    def constant(ch):
        v0 = samples[0][ch]
        if isinstance(v0, float):
            return all(abs(x[ch] - v0) <= 1e-5 for x in samples)
        return all(max(abs(a - b) for a, b in zip(x[ch], v0)) <= 1e-5 for x in samples)

    order = {'pos': 0, 'quat': 1, 'rotz': 2}
    full_ch = sorted([c for c in channels if not constant(c[0])],
                     key=lambda c: (order.get(c[0].rsplit('.', 1)[1], 3), c[0]))
    one_ch = sorted([c for c in channels if constant(c[0])],
                    key=lambda c: (order.get(c[0].rsplit('.', 1)[1], 3), c[0]))
    n = len(samples)
    comp = template['compression']
    full = (full_ch, [{c: x[c] for c, _ in full_ch} for x in samples],
            [float(i) for i in range(n)], comp)
    one = (one_ch, [{c: samples[0][c] for c, _ in one_ch}] if one_ch else [], [], comp)
    th = template['header']
    clip_seconds = (n - 1) / CLIP_FPS
    header = dict(th, start_beat=0.0, end_beat=clip_seconds * th['beats_per_sec'])
    data = encode_clip(header, full, one)
    used = {c.rsplit('.', 1)[0] for c, _ in channels}
    animated = {b.rsplit('.', 1)[0] if b.endswith(('.mesh', '.trans')) else b for b in names}
    return data, dict(samples=n, seconds=clip_seconds, full=len(full_ch), one=len(one_ch),
                      held=sorted(used - animated),
                      off_axis={k: v for k, v in off_axis.items() if v > HINGE_OFF_AXIS_WARN},
                      dropped=sorted(animated - used),
                      marker=template['has_marker'])


# ---------------------------------------------------------------------------------------
# Character body performance milo
# ---------------------------------------------------------------------------------------

MILO_REVISION = 25
BODY_DIR_TYPE = "CharClipSet"
BODY_DIR_NAME = "body"
GROUP_REV = 4
MAIN_CLIP = "main_anim"
RAMP_CLIP = "ramp"
_MILO_MAGICS = (0xCABEDEAF, 0xCBBEDEAF, 0xCCBEDEAF, 0xCDBEDEAF)
MILO_START = 0x810

# The CharClipSet 'body' object's own body, verbatim: identical in americanidiot.milo_xbox and
# the Good Riddance export that played in game (CharClipSet header, the clip type
# guitar_body_right, the preview transforms and settings).
BODY_DIR_BODY = bytes.fromhex(
    "000000190000001600000002000000116775697461725f626f64795f7269676874000000"
    "073f3504f3bf3504f3000000003f13cd3a3f13cd3abf13cd3a3ed105eb3ed105eb3f5105"
    "ebc3ddb3d7c3ddb3d743ddb3d700000000bf800000000000003f80000000000000000000"
    "0000000000000000003f800000c44000000000000000000000000000003f800000000000"
    "00bf800000000000000000000000000000000000003f8000004440000000000000000000"
    "003f80000000000000000000000000000000000000bf800000000000003f800000000000"
    "000000000000000000444000003f800000000000000000000000000000000000003f8000"
    "0000000000bf800000000000000000000000000000c44000003f80000000000000000000"
    "00000000003f8000000000000000000000000000003f80000000000000c4400000000000"
    "00bf800000000000000000000000000000bf8000000000000000000000000000003f8000"
    "000000000044400000000000000000000001000000000000000000000000000000000000"
    "00000000000000000000000000000000000000000000005a000000000000000000")


def encode_clip_group(clips, which):
    """A CharClipGroup (revision 4) body, no end marker: object header with an empty type,
    note and no data tree, the clip names, then 'which' (-1 on retail main groups, 0 on ramp
    groups), two zero fields, weight 1.0 and a zero byte - as in americanidiot.milo_xbox."""
    w = _W()
    w.i32(GROUP_REV)
    w.i32(OBJECT_REV)
    w.sym('')
    w.sym('')
    w.u8(0)
    w.u32(len(clips))
    for c in clips:
        w.sym(c)
    w.i32(which)
    w.u32(0)
    w.u32(0)
    w.f32(1.0)
    w.u8(0)
    return bytes(w.b)


def build_body_milo(song, main_clip, ramp_clip=None):
    """The complete milo file for a song body performance. `main_clip`/`ramp_clip` are
    CharClipSamples bodies without end markers. Returns the file bytes."""
    ents = [('CharClipGroup', f"song_{song}", encode_clip_group([MAIN_CLIP], -1))]
    if ramp_clip is not None:
        ents.append(('CharClipGroup', f"song_{song}_ramp", encode_clip_group([RAMP_CLIP], 0)))
    ents.append(('CharClipSamples', MAIN_CLIP, main_clip))
    if ramp_clip is not None:
        ents.append(('CharClipSamples', RAMP_CLIP, ramp_clip))
    w = _W()
    w.u32(MILO_REVISION)
    w.sym(BODY_DIR_TYPE)
    w.sym(BODY_DIR_NAME)
    w.u32(2 * (len(ents) + 1))
    w.u32(sum(len(n) + 1 for _t, n, _b in ents) + len(BODY_DIR_NAME) + 1)
    w.u32(len(ents))
    for t, n, _b in ents:
        w.sym(t)
        w.sym(n)
    w.b += BODY_DIR_BODY + END_MARKER
    blocks = []
    for t, n, b in ents:
        w.b += b + END_MARKER
        if n == MAIN_CLIP and ramp_clip is not None:
            # Retail stores the ramp clip as a second block.
            blocks.append(len(w.b))
    body = bytes(w.b)
    sizes, start = [], 0
    for end in blocks + [len(body)]:
        sizes.append(end - start)
        start = end
    head = struct.pack('<4I', 0xCABEDEAF, MILO_START, len(sizes), max(sizes))
    head += b''.join(struct.pack('<I', s) for s in sizes)
    return head.ljust(MILO_START, b'\x00') + body


def _clip_samples_length(body, p):
    """Byte length of the CharClipSamples object starting at body[p] (no end marker)."""
    from .viseme_importer import _Reader, _read_char_bones_samples
    _h, off = parse_clip_header(body[p:])
    r = _Reader(body, p + off)
    _read_char_bones_samples(r)
    _read_char_bones_samples(r)
    return r.p + 4 - p


def _clip_group_length(body, p):
    """Byte length of the CharClipGroup object starting at body[p] (no end marker)."""
    q = p + 8
    for _ in range(2):
        q += 4 + struct.unpack_from('>I', body, q)[0]
    if body[q]:
        raise ValueError("CharClipGroup with a data tree isn't supported")
    q += 1
    n = struct.unpack_from('>I', body, q)[0]
    q += 4
    for _ in range(n):
        q += 4 + struct.unpack_from('>I', body, q)[0]
    return q + 4 + 4 + 4 + 4 + 1 - p


def read_body_milo_clips(data):
    """{clip name: CharClipSamples body} from a GDRB body milo (any container type). Objects
    are walked by their parsed lengths, not by searching for end markers, since sample data
    can contain the marker bytes."""
    from .io import read_milo_container_body
    body = read_milo_container_body(data)
    p = [0]

    def u32():
        v = struct.unpack_from('>I', body, p[0])[0]; p[0] += 4; return v

    def sym():
        n = u32(); v = body[p[0]:p[0] + n].decode('latin-1'); p[0] += n; return v

    u32()
    if sym() != BODY_DIR_TYPE:
        raise ValueError("not a CharClipSet milo")
    sym()
    u32(); u32()
    ents = [(sym(), sym()) for _ in range(u32())]
    m = body.find(END_MARKER, p[0])
    if m < 0:
        raise ValueError("no end marker after the CharClipSet body")
    q = m + 4
    clips = {}
    for t, n in ents:
        if t == 'CharClipSamples':
            ln = _clip_samples_length(body, q)
            clips[n] = body[q:q + ln]
        elif t == 'CharClipGroup':
            ln = _clip_group_length(body, q)
        else:
            raise ValueError(f"unexpected {t} '{n}' in a body milo")
        if body[q + ln:q + ln + 4] != END_MARKER:
            raise ValueError(f"{t} '{n}' doesn't end where expected")
        q += ln + 4
    return clips


def _is_milo(path):
    with open(path, 'rb') as f:
        head = f.read(4)
    return len(head) == 4 and struct.unpack('<I', head)[0] in _MILO_MAGICS


def _ramp_template(template):
    """A copy of a performance template with the retail ramp clip's play flags."""
    t = dict(template)
    t['header'] = dict(template['header'], play_flags=HEADER_PRESETS['ramp']['play_flags'])
    return t


class EXPORT_OT_gdrb_body_performance_milo(bpy.types.Operator, ExportHelper):
    """Create a Green Day: Rock Band character body performance milo (like
    americanidiot.milo_xbox) from an Action: a CharClipSet 'body' holding song_<song> with
    main_anim, and optionally song_<song>_ramp with ramp"""
    bl_idname = "export_scene.gdrb_body_performance_milo"
    bl_label = "Create Character Body Performance Milo"
    bl_options = {'REGISTER'}

    filename_ext = ".milo_xbox"
    filter_glob: StringProperty(default="*.milo_xbox;*.milo_ps3", options={'HIDDEN'})

    song_name: StringProperty(
        name="Song Short Name",
        description="The song's short name, used for the clip groups song_<name> and "
                     "song_<name>_ramp (e.g. forgetyou). Empty: the file name",
        default="",
    )
    performance_action: StringProperty(
        name="Performance Action",
        description="The Action written as main_anim. Empty: the armature's active Action",
        default="",
    )
    ramp_mode: EnumProperty(
        name="Ramp Clip",
        description="What to write as the ramp clip (retail uses it with the stage-ramp "
                     "waypoint configs)",
        items=[('COPY', "Copy of Performance",
                "Write the performance again as ramp, so the file has the full retail layout"),
               ('ACTION', "Separate Action", "Write another Action as ramp"),
               ('NONE', "None", "Only song_<name> and main_anim, like the Good Riddance "
                                "export that played in game")],
        default='COPY',
    )
    ramp_action: StringProperty(
        name="Ramp Action",
        description="The Action written as ramp when Ramp Clip is Separate Action",
        default="",
    )
    bpm: FloatProperty(
        name="Tempo (BPM)",
        description="Written as the clips' beats per second (bpm / 60). Retail performances "
                     "all use 90. Ignored with a template",
        default=PERFORMANCE_BPM, min=1.0, max=400.0,
    )
    clip_type: StringProperty(
        name="Clip Type",
        description="The clips' type symbol (retail: guitar_body_right). Ignored with a "
                     "template",
        default="guitar_body_right",
    )
    # Plain text on purpose: a file-path field's folder button can't open a second file
    # browser from inside this export dialog. Paste the path.
    template_path: StringProperty(
        name="Template",
        description="Optional, recommended: the clip you're replacing - a retail body milo "
                     "(e.g. americanidiot.milo_xbox; its main_anim and ramp are used) or a raw "
                     "CharClipSamples file. The export then copies its channel layout, header "
                     "settings and length, so the clip fits what the game expects",
        default="",
    )
    pad_to_template: BoolProperty(
        name="Pad To Template Length",
        description="With a template: hold the last pose until the template's length",
        default=True,
    )
    include_twist: BoolProperty(
        name="Include Twist Bones",
        description="Also write twist bones. Off by default: GDRB's own solvers drive them",
        default=False,
    )

    def draw(self, context):
        col = self.layout.column()
        col.prop(self, "song_name")
        col.prop_search(self, "performance_action", bpy.data, "actions")
        col.prop(self, "ramp_mode")
        if self.ramp_mode == 'ACTION':
            col.prop_search(self, "ramp_action", bpy.data, "actions")
        col.separator()
        col.prop(self, "template_path")
        sub = col.column()
        sub.enabled = bool(self.template_path)
        sub.prop(self, "pad_to_template")
        sub = col.column()
        sub.enabled = not self.template_path
        sub.prop(self, "clip_type")
        sub.prop(self, "bpm")
        col.separator()
        col.prop(self, "include_twist")
        col.label(text="For PS3, rename the file to .milo_ps3 (same bytes).", icon='INFO')

    def _build(self, arm, action, fps, template, preset):
        if template is not None:
            data, info = build_clip_from_template(arm, action, fps, template,
                                                  self.include_twist, self.pad_to_template)
        else:
            data, info = build_clip_bytes(arm, action, fps, self.clip_type, preset, self.bpm,
                                          self.include_twist)
        return data, info

    def _log_info(self, label, data, info, template):
        _log(f"  {label}: {info['samples']} sample(s), {info['seconds']:.2f}s, "
             f"{info['full']} animated + {info['one']} constant channel(s), {len(data)} bytes")
        if template is None:
            return
        if info['held']:
            _log(f"    held at the template's first pose (not in the Action): "
                 f"{', '.join(info['held'])}")
        if info['dropped']:
            _log(f"    animated but not part of the template's layout, so left out: "
                 f"{', '.join(info['dropped'][:12])}")
        for bone, stray in sorted(info['off_axis'].items()):
            _log(f"    WARNING: {bone} is a hinge (rotz) in the template but bends up to "
                 f"{math.degrees(2 * math.asin(min(1.0, stray))):.1f} deg off its axis "
                 f"in the Action - that part can't be stored")

    def execute(self, context):
        arm = context.active_object
        if arm is None or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Select the armature the Action animates.")
            return {'CANCELLED'}
        if self.performance_action:
            action = bpy.data.actions.get(self.performance_action)
            if action is None:
                self.report({'ERROR'}, f"No Action named '{self.performance_action}'.")
                return {'CANCELLED'}
        else:
            action = arm.animation_data.action if arm.animation_data else None
            if action is None:
                self.report({'ERROR'}, f"'{arm.name}' has no active Action - pick a "
                                       f"Performance Action.")
                return {'CANCELLED'}
        ramp_action = None
        if self.ramp_mode == 'ACTION':
            ramp_action = bpy.data.actions.get(self.ramp_action)
            if ramp_action is None:
                self.report({'ERROR'}, "Pick the Ramp Action (or choose another Ramp Clip "
                                       "option).")
                return {'CANCELLED'}

        out_path = bpy.path.abspath(self.filepath)
        song = self.song_name.strip() or os.path.basename(out_path).split('.', 1)[0]
        if not song:
            self.report({'ERROR'}, "Set the Song Short Name.")
            return {'CANCELLED'}
        scene = context.scene
        fps = scene.render.fps / max(scene.render.fps_base, 1e-6)

        template = ramp_template = None
        if self.template_path:
            tpath = bpy.path.abspath(self.template_path)
            try:
                template = read_template_clip(tpath, MAIN_CLIP)
            except Exception as e:
                self.report({'ERROR'}, f"Couldn't read the template: {e}")
                return {'CANCELLED'}
            ramp_template = _ramp_template(template)
            if self.ramp_mode == 'ACTION' and _is_milo(tpath):
                # A body milo template with its own ramp clip supplies the ramp's layout and
                # length; otherwise the performance template is used with ramp play flags.
                try:
                    ramp_template = read_template_clip(tpath, RAMP_CLIP)
                except Exception:
                    pass

        _log(f"===== Creating GDRB body performance milo for '{song}' -> {out_path} =====")
        if template is not None:
            th = template['header']
            _log(f"  template: {len(template['channels'])} channels, clip type "
                 f"'{th['clip_type']}', flags {th['flags']:#x}/{th['play_flags']:#x}, "
                 f"{th['beats_per_sec'] * 60:g} BPM, compression {template['compression']}, "
                 f"{template['seconds']:.1f}s")
        try:
            main, info = self._build(arm, action, fps, template, 'performance')
            self._log_info(f"main_anim from '{action.name}'", main, info, template)
            ramp = None
            if self.ramp_mode == 'COPY':
                if template is not None:
                    ramp, rinfo = self._build(arm, action, fps, ramp_template, 'ramp')
                else:
                    ramp, rinfo = self._build(arm, action, fps, None, 'ramp')
                self._log_info("ramp (copy of the performance)", ramp, rinfo, None)
            elif self.ramp_mode == 'ACTION':
                ramp, rinfo = self._build(arm, ramp_action, fps, ramp_template, 'ramp')
                self._log_info(f"ramp from '{ramp_action.name}'", ramp, rinfo, ramp_template)
            data = build_body_milo(song, main, ramp)
        except Exception as e:
            self.report({'ERROR'}, f"Export failed: {e}")
            return {'CANCELLED'}
        with open(out_path, 'wb') as f:
            f.write(data)
        groups = f"song_{song}" + (f" + song_{song}_ramp" if ramp is not None else "")
        summary = (f"Wrote body performance milo ({groups}, {info['seconds']:.1f}s) to "
                   f"{os.path.basename(out_path)}")
        _log(f"===== {summary} ({len(data)} bytes) =====")
        self.report({'INFO'}, summary)
        return {'FINISHED'}
