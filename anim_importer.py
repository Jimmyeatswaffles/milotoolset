"""
Green Day: Rock Band performance-animation importer.

Reads the full motion-capture performance out of a song's animation milo and brings it
into Blender in two stages, mirroring how lipsync is handled: an import that decodes the
clip at its own native sample resolution, and a separate bake that retimes it onto the
scene's timeline using the song's tempo map.

===========================================================================================
WHY TWO STAGES
===========================================================================================
The animation is stored BEAT-INDEXED, not time-indexed, so "which pose plays when" can't
be answered by the milo alone - it needs the song's tempo map. Keeping the two apart means
the decoded performance is imported once and stays the editable source, while the timed
result is a disposable bake you regenerate whenever the tempo source or frame range
changes. Same reasoning as the lipsync weight channels: the thing that can't be recovered
from the baked output is what gets preserved.

===========================================================================================
FORMAT - verified against a retail 21guns.milo_xbox
===========================================================================================
A CharClipSet holding CharClipSamples clips (plus CharClipGroups, which this ignores).
'main_anim' decodes as CharClipSamples version 16 wrapping a CharClip version 12, with:

    start_beat 0.0   end_beat 491.3   beats_per_sec 1.5

and a bone-samples block at compression 2 - the same int16-quantised encoding the viseme
data uses, so it shares that decoder. 73 animated channels over 63 bones (36 quaternion,
25 rotz, 12 position), 3130 unique pose samples, plus 5 constant channels in the 'one'
block (mic, mic stand, foot IK).

The frame table is a BEAT-TO-SAMPLE lookup at 20 subdivisions per beat. Its length is
exactly end_beat * 20 + 1 (491.3 * 20 + 1 == 9827, matching the file), its last entry is
exactly num_samples - 1, and beat 1 reads 4.600 while beat 2 reads 8.800 - the spacing
varies because the mocap is adaptively decimated. Playback therefore means: take a beat,
index the table at beat*20, read a FRACTIONAL sample index, and interpolate between the
two neighbouring samples.

Positions use the same 1300.0 quantisation scale as the viseme data. Confirmed here
independently: bone_footik.pos decodes to a height of 37.41 against the skeleton's pelvis
at 37.07, and bone_mic_stand_top to 57.1.

Unlike the viseme clips, these samples are ABSOLUTE local transforms rather than deltas
relative to a Base pose - which is why bone_footik reads as a real world-height figure.
Each sample is therefore composed against the armature's rest pose directly.

===========================================================================================
TEMPO - the clip's own beats_per_sec is correct
===========================================================================================
Each clip carries beats_per_sec (1.5, i.e. 90 BPM, for 21guns main_anim), and that is the
rate the game plays it at. Tested in Blender against the song audio: baking at the clip's
own tempo stays in sync for the whole performance, with no drift.

An earlier version of this importer assumed the opposite and required the song's MIDI,
because 21guns.mid's tempo map varies (30 events, 61-86 BPM) and integrating it gave a
different duration. That comparison was flawed: it treated the clip's beat axis as though
it were the song's musical beat axis, and they are separate timelines. The clip's "beats"
are its own units, converted to seconds by its own beats_per_sec - the MIDI's tempo changes
don't apply to them. So the clip's tempo is the default, and a MIDI is only an optional
override that is not normally needed.

The MIDI parser is kept for that override. It reads the header's ticks-per-quarter and the
meta events of type 0x51, stepping over everything else by length, so no external module
has to be installed into Blender's Python. Verified against a retail 8-track game MIDI,
including running-status shorthand.

===========================================================================================
KNOWN GAPS
===========================================================================================
1. Rotations are written WITHOUT conjugation by default. This was settled by testing
   rather than measurement: with conjugation enabled the retail 21guns performance
   deforms the character and lifts it off the ground; with it disabled the performance
   plays correctly. That's the opposite of the viseme importer, where conjugation is
   required - the two paths store rotations differently, and the viseme path measures
   its convention against a Base clip that an animation milo doesn't have. The toggle
   remains for other titles.
2. .rotz channels are hinge joints - knees, elbows and finger segments, 25 of them in a
   performance. RESOLVED: they were decoding 20x too small (see viseme_importer.read_rotz),
   which left knees and feet straight while quaternion-driven thighs moved normally. They
   are now composed as the bone's full local Z rotation, matching the engine's use of
   Matrix3::RotateAboutZ. .rotx/.roty channels exist in the engine too but don't appear in
   the performances checked, so they remain unsupported.
3. Five animated bones (bone_facing, bone_L/R-hand_fret, bone_L/R-hand_pegs) are not in
   billiejoe_skeleton. They look like guitar-attachment targets rather than body bones;
   they're reported and skipped rather than treated as errors.
4. SMPTE-format MIDI files (negative division) aren't handled - only ticks-per-quarter.
"""

import math
import os
import struct

import bpy
from bpy.props import StringProperty, BoolProperty, FloatProperty, IntProperty, EnumProperty
from bpy_extras.io_utils import ImportHelper
from mathutils import Vector, Quaternion, Matrix

from .twist_solvers import read_twist_configs, build_rigs

from .io import read_milo_container_body
from .utilities import _log
from .mesh_importer import _read_dir_entries, _entry_spans, MeshImportError
from .viseme_importer import (
    _Reader, _locate_bone_samples, _channel_stem, build_bone_lookup,
    _read_char_bones_samples, VisemeImportError,
)


# The frame table is indexed at this many subdivisions per beat - see the module docstring.
# The frame table maps clip time to a fractional sample index at 30 entries per SECOND of
# clip time (clip time = beat / the clip's own beats_per_sec). Checked exactly on two clips:
# 21 Guns' main_anim (491.3 beats at 1.5 beats/s = 327.53 s -> 9826 + 1 entries) and the
# j_e_idle_01 idle (26.929 beats at 2.143 beats/s = 12.567 s -> 377 + 1 entries).
# An earlier version used "20 entries per beat"; that was a coincidence of main_anim's
# 1.5 beats/s (20 x 1.5 = 30), gives identical results for any clip at that tempo, and the
# wrong speed for any other.
FRAME_TABLE_RATE = 30.0
DEFAULT_FALLBACK_BPM = 120.0
MIDI_TEMPO_META = 0x51

_ANIM_TAG = "milo_anim"
_ANIM_CLIP = "milo_anim_clip"
_ANIM_END_BEAT = "milo_anim_end_beat"
_ANIM_BEAT_TABLE = "milo_anim_beat_table"
_ANIM_SAMPLES = "milo_anim_samples"
_ANIM_BPS = "milo_anim_beats_per_sec"
# Custom property on the armature remembering which character milo supplies its twist
# solver settings, so the animation import can fill it in.
CHARACTER_PATH_PROP = "milo_character_path"


class AnimImportError(Exception):
    """Raised when an animation milo or MIDI can't be read."""
    pass


# ---------------------------------------------------------------------------------------
# MIDI tempo map - no external dependency, see module docstring
# ---------------------------------------------------------------------------------------

def _varlen(data, p):
    """MIDI variable-length quantity: 7 bits per byte, high bit means 'continues'."""
    value = 0
    while True:
        c = data[p]; p += 1
        value = (value << 7) | (c & 0x7F)
        if not c & 0x80:
            return value, p


def parse_midi_tempo_map(filepath):
    """Returns (division_ticks_per_quarter, [(tick, microseconds_per_quarter), ...]).

    Walks every track and keeps only meta event 0x51. Everything else is skipped by
    length, which is why this needs no knowledge of the actual musical content. Handles
    running status (a status byte omitted because it repeats the previous one), which is
    common in real files and the one genuinely easy thing to get wrong here."""
    with open(filepath, 'rb') as f:
        data = f.read()

    if data[:4] != b'MThd':
        raise AnimImportError("not a MIDI file (missing MThd header)")
    header_len = struct.unpack_from('>I', data, 4)[0]
    _fmt, n_tracks, division = struct.unpack_from('>HHH', data, 8)
    if division & 0x8000:
        raise AnimImportError(
            "this MIDI uses SMPTE timecode division, which isn't supported - only "
            "ticks-per-quarter files are handled")
    if division <= 0:
        raise AnimImportError(f"implausible MIDI division {division}")

    tempos = []
    p = 8 + header_len
    for _ in range(n_tracks):
        if data[p:p + 4] != b'MTrk':
            raise AnimImportError(f"expected a track chunk at offset {p}")
        track_len = struct.unpack_from('>I', data, p + 4)[0]
        q = p + 8
        end = q + track_len
        tick = 0
        running = None
        while q < end:
            delta, q = _varlen(data, q)
            tick += delta
            status = data[q]
            if status < 0x80:
                st = running          # running status: reuse the previous status byte
            else:
                st = status
                q += 1
                if st < 0xF0:
                    running = st
            if st == 0xFF:
                meta = data[q]; q += 1
                ln, q = _varlen(data, q)
                if meta == MIDI_TEMPO_META and ln == 3:
                    tempos.append((tick, (data[q] << 16) | (data[q+1] << 8) | data[q+2]))
                q += ln
            elif st in (0xF0, 0xF7):
                ln, q = _varlen(data, q)
                q += ln
            elif st is None:
                raise AnimImportError(f"running status used before any status byte "
                                      f"at offset {q}")
            else:
                q += 1 if (st & 0xF0) in (0xC0, 0xD0) else 2
        p = end

    if not tempos:
        raise AnimImportError("this MIDI contains no tempo events")
    tempos.sort()
    if tempos[0][0] != 0:
        # An implicit 120 BPM applies before the first explicit tempo event.
        tempos.insert(0, (0, 500000))
    return division, tempos


class TempoMap:
    """Converts between musical beats and seconds.

    Built either from a MIDI tempo map or from a single fallback BPM. Integrating the
    piecewise-constant tempo is what makes beat-indexed animation land on the right
    seconds; a single average BPM does not (see the module docstring's drift figures)."""

    def __init__(self, division=None, tempos=None, bpm=None):
        if bpm is not None:
            self.division = 480
            self.tempos = [(0, int(round(60_000_000.0 / bpm)))]
            self.source = f"{bpm:g} BPM (fixed)"
        else:
            self.division = division
            self.tempos = tempos
            bpms = [60_000_000.0 / us for _t, us in tempos]
            self.source = (f"MIDI tempo map, {len(tempos)} event(s), "
                           f"{min(bpms):.1f}-{max(bpms):.1f} BPM")

    def beat_to_seconds(self, beat):
        target = beat * self.division
        sec = 0.0
        prev_tick, prev_us = 0, self.tempos[0][1]
        for tick, us in self.tempos:
            if tick >= target:
                break
            sec += (tick - prev_tick) / self.division * (prev_us / 1e6)
            prev_tick, prev_us = tick, us
        sec += (target - prev_tick) / self.division * (prev_us / 1e6)
        return sec

    def seconds_to_beat(self, seconds):
        """Inverse of beat_to_seconds. Walks the same segments forward, accumulating
        elapsed time, then solves the remainder inside whichever segment contains it."""
        elapsed = 0.0
        prev_tick, prev_us = 0, self.tempos[0][1]
        for tick, us in self.tempos:
            seg = (tick - prev_tick) / self.division * (prev_us / 1e6)
            if elapsed + seg >= seconds:
                break
            elapsed += seg
            prev_tick, prev_us = tick, us
        remain = seconds - elapsed
        return (prev_tick + remain / (prev_us / 1e6) * self.division) / self.division

    def average_bpm(self, end_beat):
        total = self.beat_to_seconds(end_beat)
        return (end_beat / total * 60.0) if total > 0 else 0.0


# ---------------------------------------------------------------------------------------
# Animation milo decoding
# ---------------------------------------------------------------------------------------

def _parse_clip_header(chunk):
    """Reads as far as beats_per_sec, which is deterministic. Everything after that is
    version-conditional (flags, play flags, blend width, range, a relative-clip name,
    node and event lists), so the sample block is located by signature scan instead -
    the same resync-and-validate approach used for the viseme clips."""
    r = _Reader(chunk, 0)
    samples_version = r.i32()
    clip_version = r.i32()
    meta_revision = r.i32()
    clip_type = r.numstring()
    note = r.numstring() if meta_revision > 0 else ''
    has_tree = r.u8()
    if has_tree == 1:
        raise AnimImportError("clip embeds a DTB tree, which isn't parsed")
    start_beat = r.f32()
    end_beat = r.f32()
    beats_per_sec = r.f32()
    return dict(samples_version=samples_version, clip_version=clip_version,
                type=clip_type, note=note, start_beat=start_beat,
                end_beat=end_beat, beats_per_sec=beats_per_sec)


# How far into a clip the wide search looks for its sample block. GDRB's j_e_idle_01 puts it
# at byte 243 behind a 17-entry transition table; a clip with many more transitions would
# sit further in, so this leaves plenty of room.
_WIDE_SEARCH_LIMIT = 65536
# Bytes a clip may hold after its sample blocks. Both the 21 Guns performance and
# j_e_idle_01 end exactly 4 bytes after them.
_MAX_TRAILING = 16


def _locate_bone_samples_wide(chunk):
    """Fallback for clips whose header is longer than a performance's.

    A song performance's samples start right after a short header, which is all
    _locate_bone_samples searches. GDRB's idle clips add a transition table after the clip
    name - j_e_idle_01 carries 17 pairs of beat values naming where it can loop or hand off -
    so their samples start further in (byte 243 there). Scanning further raises the chance of
    a false match, so this demands more: the two blocks must share a version, as every real
    clip's do, and they must end within a few bytes of the end of the entry."""
    for off in range(0, min(len(chunk), _WIDE_SEARCH_LIMIT)):
        try:
            r = _Reader(chunk, off)
            full = _read_char_bones_samples(r)
            one = _read_char_bones_samples(r)
        except Exception:
            continue
        if full['version'] != one['version']:
            continue
        if not (0 <= len(chunk) - r.p <= _MAX_TRAILING):
            continue
        if not full['samples'] and not one['samples']:
            continue
        return off, full, one, r.p
    raise AnimImportError("couldn't locate this clip's sample data, even with a wide search")


def parse_animation_clip(chunk, name):
    """Decodes one CharClipSamples entry into a clip dict."""
    header = _parse_clip_header(chunk)
    try:
        _off, full, one, _end = _locate_bone_samples(chunk)
    except VisemeImportError:
        # Clips with a transition table in their header (idles, for instance) push the
        # sample block past the window _locate_bone_samples searches. See
        # _locate_bone_samples_wide.
        _off, full, one, _end = _locate_bone_samples_wide(chunk)

    if not full['samples']:
        raise AnimImportError("clip has no animated samples")

    table = list(full['frame_times'])
    num_samples = full['num_samples']

    # The table maps clip time to sample index at FRAME_TABLE_RATE entries per second, so its
    # length should be (end_beat / beats_per_sec) * 30 + 1 and its final entry the last sample
    # index. If either fails, the interpretation is wrong for this clip and the timing would
    # be silently bogus.
    bps = header['beats_per_sec'] or 1.0
    expected = int(round(header['end_beat'] / bps * FRAME_TABLE_RATE)) + 1
    if table and abs(len(table) - expected) > 1:
        _log(f"    WARNING: '{name}' frame table has {len(table)} entries but end_beat "
             f"{header['end_beat']:.2f} predicts {expected} at {FRAME_TABLE_RATE:g} "
             f"entries per second of clip time - timing for this clip is unverified.")
    if table and abs(table[-1] - (num_samples - 1)) > 1.0:
        _log(f"    WARNING: '{name}' frame table ends at {table[-1]:.1f} but the clip has "
             f"{num_samples} samples - timing for this clip is unverified.")

    header.update(name=name, full=full, one=one, beat_table=table,
                  num_samples=num_samples)
    return header


def parse_animation_milo(filepath):
    """Returns (dir_name, [clip dicts], [(name, reason)])."""
    with open(filepath, 'rb') as f:
        body = read_milo_container_body(f.read())
    _rev, _dtype, dir_name, entries = _read_dir_entries(body)
    spans = _entry_spans(body, len(entries))

    clips, failed = [], []
    for (etype, ename), (s, e) in zip(entries, spans):
        if etype not in ('CharClipSamples', 'CharClip'):
            continue
        try:
            clips.append(parse_animation_clip(body[s:e], ename))
        except (AnimImportError, MeshImportError, VisemeImportError, ValueError,
                struct.error) as ex:
            failed.append((ename, str(ex)))
    return dir_name, clips, failed


# ---------------------------------------------------------------------------------------
# Sample -> pose
# ---------------------------------------------------------------------------------------

def sample_channels(clip, sample_index):
    """The clip's channel values at one sample index, merging the constant 'one' block
    with the animated 'full' block (their bone sets are disjoint)."""
    out = {}
    one = clip['one']
    if one['samples']:
        out.update(one['samples'][0])
    full = clip['full']
    i = max(0, min(int(sample_index), len(full['samples']) - 1))
    out.update(full['samples'][i])
    return out


def _decode_channel(kind, value, conjugate):
    """Turns a raw stored channel into Blender-space terms. Positions carry the 1300
    quantisation scale; rotations follow the convention established by the viseme work."""
    if kind == 'pos':
        # Already in real units - the shared decoder applies the 1300 scale now.
        return Vector((value[0], value[1], value[2]))
    if kind == 'quat':
        x, y, z, w = value
        q = Quaternion((w, x, y, z))
        return q.conjugated() if conjugate else q
    # rotz: already an angle in radians (see viseme_importer.read_rotz). The optional
    # negation mirrors the quaternion toggle and is off by default.
    return -value if conjugate else value


def pose_basis_for_bone(pose_bone, kind, decoded):
    """Converts an ABSOLUTE local transform into Blender's pose basis.

    Animation samples are the bone's own local transform, not an offset from a reference
    pose the way viseme samples are, so the rest pose has to be divided out:
    basis = rest_local^-1 * posed_local."""
    bone = pose_bone.bone
    if bone.parent is not None:
        rest_local = bone.parent.matrix_local.inverted() @ bone.matrix_local
    else:
        rest_local = bone.matrix_local
    rest_rot = rest_local.to_quaternion()

    if kind == 'pos':
        rest_t = rest_local.to_translation()
        return rest_rot.inverted() @ (decoded - rest_t)
    if kind == 'quat':
        return rest_rot.inverted() @ decoded

    # rotz: a hinge joint (knee, elbow, finger segments). The engine applies it with
    # Matrix3::RotateAboutZ, which REPLACES the bone's local rotation with a pure Z
    # rotation rather than rotating the rest orientation further. So the angle is the
    # bone's entire local rotation, and like the other channels it has to be divided by
    # the rest rotation to become a pose basis.
    #
    # Built as a standard Z rotation of +angle. That's the form the skeleton's own rest
    # matrices take once io.py's _milo_to_blender_matrix has converted them - checked on
    # both knees and both forearms, whose rest angles decode to -6.4 and -13.6 degrees and
    # sit at the nearly-straight end of their animated ranges (-76 to -16, -118 to -21).
    # The literal argument order of RotateAboutZ's Set() gives the opposite sign, which
    # would put every rest pose on the far side of zero from its own animation - not
    # something a real joint does.
    half = decoded * 0.5
    hinge = Quaternion((math.cos(half), 0.0, 0.0, math.sin(half)))
    return rest_rot.inverted() @ hinge


# ---------------------------------------------------------------------------------------
# Blender action building
# ---------------------------------------------------------------------------------------

def _get_channelbag(action, slot):
    layer = action.layers[0] if len(action.layers) else action.layers.new("Layer")
    strip = layer.strips[0] if len(layer.strips) else layer.strips.new(type='KEYFRAME')
    return strip.channelbag(slot, ensure=True)


class _CurveCache:
    """Creates each F-Curve once and keeps it, so a long bake doesn't re-scan the
    channelbag for every key it writes."""

    def __init__(self, channelbag):
        self.cb = channelbag
        self.curves = {}

    def key(self, path, index, group, frame, value, interp='BEZIER'):
        fc = self.curves.get((path, index))
        if fc is None:
            fc = self.cb.fcurves.new(path, index=index)
            grp = self.cb.groups.get(group) or self.cb.groups.new(group)
            fc.group = grp
            self.curves[(path, index)] = fc
        kp = fc.keyframe_points.insert(frame, value, options={'FAST'})
        kp.interpolation = interp

    def finish(self):
        for fc in self.curves.values():
            fc.update()


def write_pose_keys(cache, pose_bones, lookup, channels, frame, conjugate, counters):
    """Writes one frame's worth of keys for every channel that resolves to a bone."""
    for chan, (kind, value) in channels.items():
        bone_name = lookup.get(_channel_stem(chan))
        pb = pose_bones.get(bone_name) if bone_name else None
        if pb is None:
            counters['unmatched'].add(chan)
            continue
        decoded = _decode_channel(kind, value, conjugate)
        basis = pose_basis_for_bone(pb, kind, decoded)
        path = f'pose.bones["{bone_name}"]'
        if kind == 'pos':
            for i in range(3):
                cache.key(f'{path}.location', i, bone_name, frame, basis[i])
            counters['keys'] += 3
        elif kind == 'quat':
            if pb.rotation_mode != 'QUATERNION':
                pb.rotation_mode = 'QUATERNION'
            for i in range(4):
                cache.key(f'{path}.rotation_quaternion', i, bone_name, frame, basis[i])
            counters['keys'] += 4
        else:
            # Hinge channels resolve to a quaternion (see pose_basis_for_bone), so they're
            # keyed the same way - no Euler axis-order ambiguity to get wrong.
            if pb.rotation_mode != 'QUATERNION':
                pb.rotation_mode = 'QUATERNION'
            for i in range(4):
                cache.key(f'{path}.rotation_quaternion', i, bone_name, frame, basis[i])
            counters['keys'] += 4


def _engine_rows_from_quat(q):
    """A Blender-convention local rotation as engine rows (Blender columns = rows)."""
    m = q.to_matrix()
    return tuple(tuple(m[j][i] for j in range(3)) for i in range(3))


def _engine_rows_from_hinge(angle):
    """A hinge channel's full local rotation (a standard Z rotation of +angle, as
    pose_basis_for_bone builds it) as engine rows."""
    c, s = math.cos(angle), math.sin(angle)
    return ((c, s, 0.0), (-s, c, 0.0), (0.0, 0.0, 1.0))


def _quat_from_engine_rows(R):
    """Engine rows back to a Blender quaternion (rows become columns)."""
    return Matrix(((R[0][0], R[1][0], R[2][0], 0.0),
                   (R[0][1], R[1][1], R[2][1], 0.0),
                   (R[0][2], R[1][2], R[2][2], 0.0),
                   (0.0, 0.0, 0.0, 1.0))).to_quaternion()


def _input_rows(rig, role, suffix, channels, conjugate):
    """The engine-convention local rotation of one of the solver's input bones, from the
    sample if it's animated there, otherwise its rest rotation."""
    stem = rig.names[role].rsplit('.', 1)[0]
    entry = channels.get(f"{stem}.{suffix}")
    if entry is None:
        return rig.rest[role][0]
    kind, value = entry
    decoded = _decode_channel(kind, value, conjugate)
    if kind == 'quat':
        # Stored quaternions are int16-quantised and come out fractionally off unit length;
        # the solvers build frames directly from these axes, so normalise first, as the
        # engine's own rotation matrices are.
        return _engine_rows_from_quat(decoded.normalized())
    if kind == 'rotz':
        return _engine_rows_from_hinge(decoded)
    return rig.rest[role][0]


def write_twist_keys(cache, pose_bones, rig, channels, frame, conjugate, prev, counters):
    """Solves one arm's twist bones for this sample and keys their rotations.

    Only rotation is keyed: both solvers leave the twist bones' positions where the
    skeleton puts them (checked across a full performance), so a location channel would
    add nothing."""
    out = rig.solve(_input_rows(rig, 'upperArm', 'quat', channels, conjugate),
                    _input_rows(rig, 'foreArm', 'rotz', channels, conjugate),
                    _input_rows(rig, 'hand', 'quat', channels, conjugate))
    for role, (R, _t) in out.items():
        name = rig.names[role]
        pb = pose_bones.get(name)
        if pb is None:
            continue
        if pb.rotation_mode != 'QUATERNION':
            pb.rotation_mode = 'QUATERNION'
        basis = pose_basis_for_bone(pb, 'quat', _quat_from_engine_rows(R))
        last = prev.get(name)
        if last is not None and sum(a * b for a, b in zip(basis, last)) < 0.0:
            basis = Quaternion((-basis[0], -basis[1], -basis[2], -basis[3]))
        prev[name] = basis
        path = f'pose.bones["{name}"].rotation_quaternion'
        for i in range(4):
            cache.key(path, i, name, frame, basis[i])
        counters['keys'] += 4
        counters['twist_keys'] += 4


def build_twist_rigs(arm_obj, lookup, solve, character_path):
    """Builds the arm twist rigs for an armature, or returns ([], notes). Shared by the
    performance and clip-set importers. A character milo path that reads successfully is
    remembered on the armature."""
    if not solve:
        return [], []
    configs = None
    notes = []
    if character_path:
        try:
            with open(bpy.path.abspath(character_path), 'rb') as f:
                configs = read_twist_configs(read_milo_container_body(f.read()))
            notes.append(f"read {len(configs['fore'])} forearm and "
                         f"{len(configs['upper'])} upper-arm solver(s) from the "
                         f"character milo")
        except Exception as e:
            configs = None
            notes.append(f"could not read the character milo ({e}); using defaults")
        # Remembering the path is separate from reading it, so a failure here can't be
        # reported as a failure to read the file.
        if configs and (configs['fore'] or configs['upper']):
            try:
                arm_obj[CHARACTER_PATH_PROP] = character_path
            except (TypeError, AttributeError):
                pass
    parents = {}
    for pb in arm_obj.pose.bones:
        parents[pb.bone.name] = pb.bone.parent.name if pb.bone.parent else None

    def rest(name):
        bone = arm_obj.pose.bones[name].bone
        if bone.parent is not None:
            rl = bone.parent.matrix_local.inverted() @ bone.matrix_local
        else:
            rl = bone.matrix_local
        # Blender's columns are the basis vectors; the solvers want them as rows.
        rows = tuple(tuple(rl[j][i] for j in range(3)) for i in range(3))
        t = rl.to_translation()
        return rows, (t.x, t.y, t.z)

    rigs, more = build_rigs(lookup, parents, rest, configs)
    return rigs, notes + more


def _interp_channels(clip, sample_pos):
    """The clip's channels at a FRACTIONAL sample index, blending the two neighbouring
    samples: positions and hinge angles linearly, quaternions by sign-aligned normalised
    blend. The frame table points between samples because the motion capture was decimated
    unevenly, so this is what playback between stored samples looks like."""
    n = clip['num_samples']
    sample_pos = max(0.0, min(float(sample_pos), n - 1.0))
    i0 = int(sample_pos)
    i1 = min(i0 + 1, n - 1)
    t = sample_pos - i0
    a = sample_channels(clip, i0)
    if t <= 1e-6 or i1 == i0:
        return a
    b = sample_channels(clip, i1)
    out = {}
    for chan, (kind, v0) in a.items():
        v1 = b.get(chan, (kind, v0))[1]
        if kind == 'quat':
            if sum(x * y for x, y in zip(v0, v1)) < 0.0:
                v1 = tuple(-x for x in v1)
            q = [x + (y - x) * t for x, y in zip(v0, v1)]
            mag = math.sqrt(sum(x * x for x in q)) or 1.0
            out[chan] = (kind, tuple(x / mag for x in q))
        elif kind == 'pos':
            out[chan] = (kind, tuple(x + (y - x) * t for x, y in zip(v0, v1)))
        else:
            out[chan] = (kind, v0 + (v1 - v0) * t)
    return out


def _clip_groups(body, entries, spans, clip_names):
    """{clip name: [group names]} from the milo's CharClipGroups. A group lists its member
    clips by name (GDRB's 'stand' holds just j_e_idle_01), so member names are matched
    against the file's own clips."""
    groups = {}
    for (etype, ename), (s, e) in zip(entries, spans):
        if etype != 'CharClipGroup':
            continue
        data = body[s:e]
        for clip in clip_names:
            tag = struct.pack('>I', len(clip)) + clip.encode('latin-1')
            if tag in data:
                groups.setdefault(clip, []).append(ename)
    return groups


# ---------------------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------------------

class IMPORT_OT_gdrb_animation(bpy.types.Operator, ImportHelper):
    """Import a Green Day: Rock Band performance animation milo. Each clip becomes an
    Action keyed at the clip's own sample resolution; use Bake Performance afterwards to
    retime it onto the scene timeline with the song's tempo"""
    bl_idname = "import_scene.gdrb_animation"
    bl_label = "Import GDRB CharClip Milo"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".milo_xbox"
    filter_glob: StringProperty(
        default="*.milo_xbox;*.milo_ps3;*.milo", options={'HIDDEN'})

    conjugate_rotations: BoolProperty(
        name="Convert Rotation Convention",
        description="Conjugate rotations on import. Performance animation needs this "
                     "OFF - testing on retail GDRB data showed enabling it deforms the "
                     "character and lifts it off the ground. Note this differs from the "
                     "viseme importer, where the conversion IS required; the two store "
                     "rotations differently, so don't carry the setting across",
        default=False,
    )

    solve_arm_twist: BoolProperty(
        name="Solve Arm Twist Bones",
        description="Compute the eight arm twist bones the way the game's CharForeTwist and "
                     "CharUpperTwist solvers do. A performance never animates them, and "
                     "without this they stay at rest while the arm moves, pinching the "
                     "forearms and sleeves",
        default=True,
    )

    # A plain text field on purpose. This dialog is itself a file browser, and Blender can't
    # open a second one from inside it, so a file-path field's folder button here only ever
    # produced "Cannot activate a file selector dialog, one already open". Use Object >
    # Rock Band Animation > Set Character Milo to pick the file instead; it's remembered on
    # the armature and filled in here automatically.
    character_path: StringProperty(
        name="Character Milo",
        description="Optional: the character's own milo (e.g. billiejoe.milo_xbox), to "
                     "read its twist solvers' settings. Filled in automatically from Object "
                     "> Rock Band Animation > Set Character Milo, or paste a path. Without "
                     "it, the standard settings are used, which match Billie Joe's",
        default="",
    )

    def invoke(self, context, event):
        obj = context.active_object
        if obj is not None and obj.type == 'ARMATURE':
            stored = obj.get(CHARACTER_PATH_PROP)
            if stored:
                self.character_path = stored
        return ImportHelper.invoke(self, context, event)

    def _twist_rigs(self, arm_obj, lookup):
        return build_twist_rigs(arm_obj, lookup, self.solve_arm_twist, self.character_path)

    def execute(self, context):
        arm_obj = context.active_object
        if arm_obj is None or arm_obj.type != 'ARMATURE':
            self.report({'ERROR'},
                        "Select the target armature first - animation is keyed onto its "
                        "pose bones by name.")
            return {'CANCELLED'}

        try:
            dir_name, clips, failed = parse_animation_milo(self.filepath)
        except Exception as e:
            _log(f"GDRB ANIMATION IMPORT FAILED: {e}")
            self.report({'ERROR'}, f"Could not parse animation milo: {e}")
            return {'CANCELLED'}

        if not clips:
            self.report({'ERROR'},
                        "No CharClipSamples entries found - is this a song animation "
                        "milo? (A CharClipGroup on its own holds no samples.)")
            return {'CANCELLED'}

        _log(f"===== Importing GDRB animation '{dir_name}' from {self.filepath} =====")
        lookup = build_bone_lookup(arm_obj)
        pose_bones = arm_obj.pose.bones
        rigs, twist_notes = self._twist_rigs(arm_obj, lookup)
        for n in twist_notes:
            _log(f"  {n}")
        made = 0
        all_unmatched = set()

        for clip in clips:
            name = clip['name']
            action_name = f"ANIM_{name}"
            existing = bpy.data.actions.get(action_name)
            if existing is not None:
                bpy.data.actions.remove(existing)
            action = bpy.data.actions.new(action_name)
            action[_ANIM_TAG] = True
            action[_ANIM_CLIP] = name
            action[_ANIM_END_BEAT] = float(clip['end_beat'])
            action[_ANIM_SAMPLES] = int(clip['num_samples'])
            action[_ANIM_BPS] = float(clip['beats_per_sec'])
            action[_ANIM_BEAT_TABLE] = clip['beat_table']

            slot = action.slots.new(id_type='OBJECT', name=arm_obj.name)
            cache = _CurveCache(_get_channelbag(action, slot))
            counters = {'keys': 0, 'unmatched': set(), 'twist_keys': 0}

            # One key per stored sample, at frame = sample index + 1. This is the clip's
            # own resolution - no tempo involved - so it stays a faithful record of what
            # the file contains and can be re-timed any number of ways afterwards.
            twist_prev = {}
            for i in range(clip['num_samples']):
                channels = sample_channels(clip, i)
                write_pose_keys(cache, pose_bones, lookup, channels,
                                i + 1, self.conjugate_rotations, counters)
                for rig in rigs:
                    write_twist_keys(cache, pose_bones, rig, channels, i + 1,
                                     self.conjugate_rotations, twist_prev, counters)
            cache.finish()
            all_unmatched |= counters['unmatched']
            made += 1
            _log(f"  '{name}': {clip['num_samples']} sample(s), "
                 f"{len(clip['beat_table'])} beat-table entries, end_beat "
                 f"{clip['end_beat']:.2f}, {counters['keys']} keyframe(s)"
                 + (f", {counters['twist_keys']} of them solved arm-twist keys"
                    if counters['twist_keys'] else ""))

        for nm, reason in failed:
            _log(f"  SKIPPED {nm}: {reason}")
        if all_unmatched:
            stems = sorted({_channel_stem(c) for c in all_unmatched})
            _log(f"  {len(stems)} animated bone(s) are not on '{arm_obj.name}' and were "
                 f"skipped: {', '.join(stems)}")
            _log("    (bone_facing and the hand_fret/hand_pegs pairs are guitar "
                 "attachment targets, not body bones - expected to be absent.)")

        total_keys = sum(c['num_samples'] for c in clips)
        if total_keys > 2000:
            _log(f"  NOTE: importing at native sample resolution produces a large Action "
                 f"({clips[0]['num_samples']} samples per clip). That's the clip's own "
                 f"data rather than a choice, but expect the Action to be heavy to scrub.")
        summary = (f"Imported {made} clip(s) from '{dir_name}' at sample resolution"
                   + (f", {len(failed)} failed" if failed else ""))
        _log(f"===== {summary} =====")
        _log("  NEXT STEP: Object > Rock Band Animation > Bake Performance, supplying the "
             "song's .mid so the beat-indexed animation lands on the right seconds.")
        self.report({'WARNING' if failed else 'INFO'},
                    summary + " - now run 'Bake Performance' to retime it")
        return {'FINISHED'}


class POSE_OT_bake_gdrb_animation(bpy.types.Operator):
    """Retime an imported performance onto the scene timeline using the song's tempo map.
    Supply the song's MIDI for correct timing - a fixed BPM will drift"""
    bl_idname = "pose.bake_gdrb_animation"
    bl_label = "Bake Performance"
    bl_options = {'REGISTER', 'UNDO'}

    midi_path: StringProperty(
        name="Song MIDI (optional)",
        description="Optional override. Leave empty to use the clip's own tempo, which "
                     "testing showed plays in sync with the audio. Supplying a MIDI "
                     "retimes against its tempo map instead",
        subtype='FILE_PATH',
        default="",
    )

    clip_bpm: FloatProperty(
        name="Clip BPM",
        description="Tempo the clip plays at, pre-filled from the clip's own "
                     "beats_per_sec when the dialog opens. That value is what the game "
                     "uses and stays in sync with the song audio. Only used when no MIDI "
                     "is given",
        default=DEFAULT_FALLBACK_BPM, min=1.0, max=1000.0,
    )

    frame_start: IntProperty(name="Start Frame", default=1, min=0)
    frame_end: IntProperty(name="End Frame", default=250, min=0)

    def invoke(self, context, event):
        self.frame_start = context.scene.frame_start
        self.frame_end = context.scene.frame_end
        # Pre-fill the tempo from the clip itself rather than a guess. The value is only
        # a default in the dialog, so it can still be overridden before baking.
        obj = context.active_object
        anim = obj.animation_data if obj is not None else None
        act = anim.action if anim else None
        if act is not None and act.get(_ANIM_BPS):
            self.clip_bpm = float(act[_ANIM_BPS]) * 60.0
        return context.window_manager.invoke_props_dialog(self, width=420)

    def execute(self, context):
        arm_obj = context.active_object
        if arm_obj is None or arm_obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Select the armature holding the imported animation.")
            return {'CANCELLED'}

        anim = arm_obj.animation_data
        source = anim.action if anim else None
        if source is None or not source.get(_ANIM_TAG):
            self.report({'ERROR'},
                        "Assign an imported ANIM_* Action to this armature first - the "
                        "bake reads the clip's beat table from it.")
            return {'CANCELLED'}

        table = list(source[_ANIM_BEAT_TABLE])
        end_beat = float(source[_ANIM_END_BEAT])
        # The clip's own tempo converts its beats to the clip time the frame table is indexed
        # by. Actions imported before this was stored are from 1.5 beats/s clips - the only
        # tempo the old 20-per-beat rule was right for - so that's the fallback.
        clip_bps = float(source.get(_ANIM_BPS) or 1.5)
        if not table:
            self.report({'ERROR'}, "This Action has no beat table stored.")
            return {'CANCELLED'}

        if self.midi_path:
            try:
                division, tempos = parse_midi_tempo_map(bpy.path.abspath(self.midi_path))
                tempo = TempoMap(division=division, tempos=tempos)
            except Exception as e:
                _log(f"BAKE FAILED: could not read MIDI: {e}")
                self.report({'ERROR'}, f"Could not read the MIDI: {e}")
                return {'CANCELLED'}
        else:
            bpm = self.clip_bpm
            # If the dialog was bypassed (script / redo panel), fall back to the value
            # stored on the Action instead of the operator's generic default.
            if source.get(_ANIM_BPS) and abs(bpm - DEFAULT_FALLBACK_BPM) < 1e-6:
                bpm = float(source[_ANIM_BPS]) * 60.0
            tempo = TempoMap(bpm=bpm)

        scene = context.scene
        fps = scene.render.fps / max(scene.render.fps_base, 1e-6)
        total_sec = tempo.beat_to_seconds(end_beat)

        _log(f"===== Baking performance '{source.get(_ANIM_CLIP, source.name)}' =====")
        _log(f"  tempo source: {tempo.source}")
        _log(f"  clip is {end_beat:.2f} beats -> {total_sec:.2f}s "
             f"({int(total_sec//60)}m{total_sec%60:04.1f}s) at {fps:g} fps")
        if self.midi_path:
            _log(f"  average tempo {tempo.average_bpm(end_beat):.1f} BPM (MIDI override)")
        stored = source.get(_ANIM_BPS)
        if stored and not self.midi_path:
            _log(f"  clip's own tempo: {float(stored)*60.0:g} BPM "
                 f"(beats_per_sec {float(stored):g})")

        start, end = self.frame_start, self.frame_end
        if end < start:
            self.report({'ERROR'}, "End frame is before start frame.")
            return {'CANCELLED'}


        bake_name = f"PERF_{source.get(_ANIM_CLIP, source.name)}"
        existing = bpy.data.actions.get(bake_name)
        if existing is not None:
            bpy.data.actions.remove(existing)
        bake = bpy.data.actions.new(bake_name)
        slot = bake.slots.new(id_type='OBJECT', name=arm_obj.name)
        cache = _CurveCache(_get_channelbag(bake, slot))

        cb = _source_channelbag(source)
        if cb is None:
            self.report({'ERROR'}, "The source Action has no animation channels.")
            return {'CANCELLED'}
        curves = [fc for fc in cb.fcurves if fc.data_path.startswith('pose.bones[')]
        full_frames = int(total_sec * fps)
        if end - start + 1 >= full_frames:
            _log(f"  NOTE: baking the full {full_frames} frames writes roughly "
                 f"{full_frames * len(curves):,} keyframes. Bake a shorter range first to "
                 f"check the result before committing to the whole performance.")

        written = 0
        past_end = 0
        for frame in range(start, end + 1):
            seconds = (frame - 1) / fps
            beat = tempo.seconds_to_beat(seconds)
            if beat > end_beat:
                past_end += 1
                continue
            idx = beat / clip_bps * FRAME_TABLE_RATE
            # Interpolate the beat table, then interpolate between the two samples it
            # points at - the table stores a FRACTIONAL sample index, and sample spacing
            # is uneven because the mocap was adaptively decimated.
            i0 = max(0, min(int(idx), len(table) - 1))
            i1 = min(i0 + 1, len(table) - 1)
            t = idx - i0
            sample_pos = table[i0] * (1.0 - t) + table[i1] * t
            src_frame = sample_pos + 1.0     # import keyed sample i at frame i+1
            for fc in curves:
                cache.key(fc.data_path, fc.array_index, _group_of(fc),
                          frame, fc.evaluate(src_frame))
                written += 1
        cache.finish()

        summary = (f"Baked '{bake_name}': frames {start}-{end}, {len(curves)} curve(s), "
                   f"{written} keyframe(s)")
        if past_end:
            summary += f"; {past_end} frame(s) past the end of the clip were skipped"
        _log(f"===== {summary} =====")
        _log("  Assign this Action to play the performance in real time.")
        self.report({'INFO'}, summary)
        return {'FINISHED'}


def _source_channelbag(action):
    if not len(action.layers) or not len(action.slots):
        return None
    layer = action.layers[0]
    if not len(layer.strips):
        return None
    return layer.strips[0].channelbag(action.slots[0], ensure=False)


def _group_of(fcurve):
    grp = getattr(fcurve, 'group', None)
    if grp is not None:
        return grp.name
    path = fcurve.data_path
    if path.startswith('pose.bones["'):
        return path[len('pose.bones["'):path.index('"]')]
    return "Animation"


class POSE_OT_set_gdrb_character_milo(bpy.types.Operator, ImportHelper):
    """Choose the character milo whose arm twist solver settings the animation import
    should use. It's remembered on the selected armature and filled in automatically"""
    bl_idname = "pose.set_gdrb_character_milo"
    bl_label = "Set Character Milo"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".milo_xbox"
    filter_glob: StringProperty(default="*.milo_xbox;*.milo_ps3;*.milo", options={'HIDDEN'})

    def execute(self, context):
        arm_obj = context.active_object
        if arm_obj is None or arm_obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Select the character's armature first.")
            return {'CANCELLED'}
        try:
            with open(self.filepath, 'rb') as f:
                configs = read_twist_configs(read_milo_container_body(f.read()))
        except Exception as e:
            self.report({'ERROR'}, f"Could not read that milo: {e}")
            return {'CANCELLED'}
        found = len(configs['fore']) + len(configs['upper'])
        if not found:
            # Most likely the skeleton, outfit or head milo rather than the character's own.
            self.report({'ERROR'},
                        "No arm twist solvers in that milo. Pick the character's own milo "
                        "(e.g. billiejoe.milo_xbox), not its skeleton, outfit or head milo.")
            return {'CANCELLED'}
        arm_obj[CHARACTER_PATH_PROP] = self.filepath
        msg = (f"Character milo set on '{arm_obj.name}': {os.path.basename(self.filepath)} "
               f"({len(configs['fore'])} forearm, {len(configs['upper'])} upper-arm solver(s))")
        _log(msg)
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class VIEW3D_MT_milo_animation(bpy.types.Menu):
    bl_idname = "VIEW3D_MT_milo_animation"
    bl_label = "Rock Band Animation"

    def draw(self, context):
        self.layout.operator(POSE_OT_bake_gdrb_animation.bl_idname,
                             text="Bake Performance", icon='ACTION')
        self.layout.operator(POSE_OT_set_gdrb_character_milo.bl_idname,
                             text="Set Character Milo...", icon='FILEBROWSER')


def menu_func_gdrb_animation(self, context):
    obj = context.active_object
    if obj is not None and obj.type == 'ARMATURE':
        self.layout.menu(VIEW3D_MT_milo_animation.bl_idname)


class IMPORT_OT_gdrb_clip_set(bpy.types.Operator, ImportHelper):
    """Import every animation clip in a Green Day: Rock Band clip-set milo (idles and other
    short clips) as one Action each on the active armature, already playing at the right
    speed - no bake step. For full-song performances use the CharClip importer above"""
    bl_idname = "import_scene.gdrb_clip_set"
    bl_label = "Import GDRB Clip Set Milo"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".milo_xbox"
    filter_glob: StringProperty(default="*.milo_xbox;*.milo_ps3;*.milo", options={'HIDDEN'})

    conjugate_rotations: BoolProperty(
        name="Convert Rotation Convention",
        description="Leave off for GDRB, as for performances: conjugating rotations "
                     "deforms the character",
        default=False,
    )
    solve_arm_twist: BoolProperty(
        name="Solve Arm Twist Bones",
        description="Compute the arm twist bones the way the game's twist solvers do",
        default=True,
    )
    character_path: StringProperty(
        name="Character Milo",
        description="Optional: the character's own milo, for its twist solver settings. "
                     "Filled in from Object > Rock Band Animation > Set Character Milo",
        default="",
    )

    def invoke(self, context, event):
        obj = context.active_object
        if obj is not None and obj.type == 'ARMATURE' and obj.get(CHARACTER_PATH_PROP):
            self.character_path = obj[CHARACTER_PATH_PROP]
        return ImportHelper.invoke(self, context, event)

    def execute(self, context):
        arm_obj = context.active_object
        if arm_obj is None or arm_obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Select the target armature first.")
            return {'CANCELLED'}
        try:
            with open(self.filepath, 'rb') as f:
                body = read_milo_container_body(f.read())
            _rev, _dt, dir_name, entries = _read_dir_entries(body)
            spans = _entry_spans(body, len(entries))
            dir_name_, clips, failed = parse_animation_milo(self.filepath)
        except Exception as e:
            _log(f"GDRB CLIP SET IMPORT FAILED: {e}")
            self.report({'ERROR'}, f"Could not parse clip-set milo: {e}")
            return {'CANCELLED'}
        if not clips:
            self.report({'ERROR'}, "No animation clips (CharClipSamples) in this milo.")
            return {'CANCELLED'}

        scene = context.scene
        fps = scene.render.fps / max(scene.render.fps_base, 1e-6)
        groups = _clip_groups(body, entries, spans, [c['name'] for c in clips])
        lookup = build_bone_lookup(arm_obj)
        pose_bones = arm_obj.pose.bones
        rigs, notes = build_twist_rigs(arm_obj, lookup, self.solve_arm_twist,
                                       self.character_path)
        _log(f"===== Importing GDRB clip set '{dir_name}' from {self.filepath} =====")
        for n in notes:
            _log(f"  {n}")

        made = 0
        unmatched = set()
        for clip in clips:
            name = clip['name']
            bps = clip['beats_per_sec'] or 1.0
            seconds = clip['end_beat'] / bps
            table = clip['beat_table']
            frames = int(math.ceil(seconds * fps)) + 1

            action_name = f"CLIP_{name}"
            existing = bpy.data.actions.get(action_name)
            if existing is not None:
                bpy.data.actions.remove(existing)
            action = bpy.data.actions.new(action_name)
            action["milo_clip"] = True
            action["milo_clip_name"] = name
            action["milo_clip_set"] = dir_name
            action["milo_clip_groups"] = groups.get(name, [])
            action["milo_clip_seconds"] = seconds
            # A library of clips nothing is assigned to yet would otherwise have zero users
            # and be discarded when the file is saved and reopened.
            action.use_fake_user = True
            slot = action.slots.new(id_type='OBJECT', name=arm_obj.name)
            cache = _CurveCache(_get_channelbag(action, slot))
            counters = {'keys': 0, 'unmatched': set(), 'twist_keys': 0}
            twist_prev = {}

            # Keyed straight onto the scene timeline: frame k is k/fps seconds into the
            # clip, the frame table turns that into a fractional sample index at
            # FRAME_TABLE_RATE entries per second, and the two neighbouring samples are
            # blended.
            for k in range(frames):
                idx = min(k / fps, seconds) * FRAME_TABLE_RATE
                if table:
                    i0 = max(0, min(int(idx), len(table) - 1))
                    i1 = min(i0 + 1, len(table) - 1)
                    t = idx - i0
                    sample_pos = table[i0] * (1.0 - t) + table[i1] * t
                else:
                    sample_pos = idx / max(seconds * FRAME_TABLE_RATE, 1e-6) \
                        * (clip['num_samples'] - 1)
                channels = _interp_channels(clip, sample_pos)
                write_pose_keys(cache, pose_bones, lookup, channels, k + 1,
                                self.conjugate_rotations, counters)
                for rig in rigs:
                    write_twist_keys(cache, pose_bones, rig, channels, k + 1,
                                     self.conjugate_rotations, twist_prev, counters)
            cache.finish()
            unmatched |= counters['unmatched']
            made += 1
            grp = f", group {', '.join(groups[name])}" if name in groups else ""
            _log(f"  '{name}': {seconds:.2f}s at {bps * 60:.1f} BPM -> {frames} frame(s), "
                 f"{counters['keys']} keyframe(s){grp}")

        for nm, reason in failed:
            _log(f"  SKIPPED {nm}: {reason}")
        if unmatched:
            stems = sorted({_channel_stem(c) for c in unmatched})
            _log(f"  {len(stems)} animated bone(s) aren't on '{arm_obj.name}' and were "
                 f"skipped: {', '.join(stems)}")
        summary = (f"Imported {made} clip(s) from '{dir_name}' as CLIP_ Actions"
                   + (f", {len(failed)} failed" if failed else ""))
        _log(f"===== {summary} =====")
        _log("  Each Action plays at the clip's own speed - assign one to preview it.")
        self.report({'WARNING' if failed else 'INFO'}, summary)
        return {'FINISHED'}
