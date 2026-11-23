"""
Dance Central 3 CharClip export for Boomy: writes Blender Actions as DC3 move clips (CharClip
version 22) into a Boomy move library, each as a new clip variant of an existing library move.

===========================================================================================
WHAT BOOMY EXPECTS - from BoomyExporter/ExportOperator.cs and BoomyBuilder/AssetsImporter.cs
===========================================================================================
A move library is <library>/<category>/<song>/<move>/, the category one of dc1, dc2, dc3,
dc1_dlc, dc2_dlc, dc3_dlc or dcs (the editor's GAME_CATEGORIES). Any folder at that depth is a
move. Each holds:
  move.json            name, difficulty, display_name, ham_move_name, milo_name, song_name,
                       and clips: {clip key: avg_beats_per_second, genre, era, flags,
                       linked_to, linked_from}
  <ham_move_name>      the HamMove (scoring data), raw object bytes, e.g. Shake_It_x.move
  <milo_name>.seq      its DancerSequence (the reference skeleton the player is scored against)
  <milo_name>.tex / <milo_name>_sm.tex   the move card icons
  <clip key>           one CharClip per variant, raw object bytes, NO extension
  move.png, .boomy     editor preview and the 'move1' marker
Object files are the bytes between a milo's end markers, without the marker; Boomy adds it
when it builds the song milo.

The builder imports the files as they are. The game keeps the visual and the scoring apart,
as retail DC3 data already does (gangnamstyle's 'wednesday_night' clips score against
Cowboy_Call_gangnamstyle.move): the move graph groups clip variants under a parent named by
milo_name, the expert song.anim plays the clip key, and the 'move' keys name the HamMove.
So a new clip can stand as one more variant of an existing move: this exporter copies a
template move's HamMove, DancerSequence and icons unchanged and adds the Action as the clip.
THE PLAYER IS SCORED AGAINST THE TEMPLATE MOVE, not the new animation - pick a template whose
movement is close, or treat the result as a visual test.

===========================================================================================
CHARCLIP VERSION 22 - CharClip::Save in the DC3 decompilation (system/char/CharClip.cpp)
===========================================================================================
  22, the Hmx::Object header (revision, type symbol, has-tree flag and data tree, note),
  frames per second, flags, play flags, range, relative-clip symbol, old version (int),
  do-not-compress (byte), transitions (total node count, vector count, then per vector a clip
  name and (cur beat, next beat) pairs), beat events (count, then (symbol, beat)), the 'full'
  and 'one' CharBonesSamples (version 16, as RB3), mZeros (count, then (bone, weight)), the
  beat track (count, then (beat, frame)), mSyncAnim (a symbol), then two fields RB3 lacks:
  mSkeletonSamples (count, then per sample a map of (int, float)) and mBitmask (int, one bit
  per difficulty).
Verified on gangnamstyle.milo_xbox: all 181 retail clips parse to their last byte and
re-encode byte for byte. No retail clip has skeleton samples. Rebuilt through
build_dc3_clip from their own decoded poses, its 121 move clips come back within int16
quantisation (worst 4.8e-5), with the same header, beat length and self-transition offsets.

What the 143 move clips in its move_data share, and what this copies from the template clip:
an object header of type 'main' with a data tree (clip_skeleton 'devin',
clip_skeleton_index 0), flags 0, play flags 0x1000 (a beat-phase period of 1: ClipPlayer's
ClipStart keeps the clip's beat phase), old version 4, compression 1 (positions as floats),
no zeros, no beat events, and a single self-transition node about two beats in. mBitmask is 0
on move clips; only the A_B transition clips set it (ClipPlayer only tests it on those).

===========================================================================================
TIMING
===========================================================================================
Retail move clips run about six beats: one beat of lead-in, the four-beat measure, about one
beat of tail for blending. Boomy keys the expert clip one beat before the measure
(Animator.CreateExpertClips, beatOffset -1) and ClipStart places the clip's first sample on
the key, so the measure starts exactly one beat into the clip. The Action is sampled at 30 fps
and its beat track runs from beat 0 at the tempo given; 'Lead-in Beats' says how far into the
Action the measure starts, and the start is trimmed or padded so that lands on beat 1.

===========================================================================================
SPECIAL CHANNELS
===========================================================================================
bone_facing - root travel. The game never poses it as a bone: CharServoBone adds its position
to the pelvis (and other root-level bones) after posing, which is where a dancer's travel
lives. The DC clip importer folds it into the pelvis for that reason, so by default the
pelvis's horizontal travel is split back out into bone_facing and the root bones are written
relative to it, as retail clips are. If the Action animates a bone named bone_facing, that is
used as it stands.
bone_footik - not a pose: CharIKFoot reads its x and y as the left and right foot's
'planted' flags (planted at >= 0.98). A held template value would lock the feet at the wrong
moments, so it's written as 0 (feet follow the animation) unless 'Keep Foot-Plant Keys' is on
and the Action keys that bone.
Rotations are written as stored by the DC clip importer's 'as stored' convention (the one it
measures on Dance Central rigs), so an imported clip re-exports to its own values; hinge
channels (rotz) are converted from the Action's rotation as in the RB3 exporter.
"""

import json
import math
import os
import re
import shutil
import struct

import bpy
from bpy.props import StringProperty, BoolProperty, EnumProperty, FloatProperty

from .utilities import _log

DC3_CLIP_REV = 22
END_MARKER = b'\xAD\xDE\xAD\xDE'
CLIP_FPS = 30.0
BOOMY_CATEGORIES = ('dc1', 'dc2', 'dc3', 'dc1_dlc', 'dc2_dlc', 'dc3_dlc', 'dcs')
MOVE_JSON_KEYS = ('name', 'difficulty', 'display_name', 'ham_move_name', 'milo_name',
                  'song_name', 'clips')
PLANTED = 0.98


class DC3BoomyError(Exception):
    pass


class _W:
    def __init__(self):
        self.b = bytearray()

    def u8(self, v): self.b += struct.pack('>B', v)
    def u32(self, v): self.b += struct.pack('>I', v)
    def i32(self, v): self.b += struct.pack('>i', v)
    def f32(self, v): self.b += struct.pack('>f', v)

    def sym(self, s):
        raw = s.encode('latin-1')
        self.u32(len(raw))
        self.b += raw


# ---------------------------------------------------------------------------------------
# CharClip v22
# ---------------------------------------------------------------------------------------

def parse_dc3_clip(c):
    """Every field of a DC3 CharClip, in order. The object header (with its data tree) is
    kept as raw bytes; the sample blocks are decoded with the shared reader."""
    from .dc_clip_importer import _clip_fps
    from .rb3_clip_exporter import _R
    from .viseme_importer import _Reader, _read_char_bones_samples
    if len(c) < 8 or struct.unpack_from('>I', c, 0)[0] != DC3_CLIP_REV:
        raise DC3BoomyError("not a version-22 (Dance Central 3) CharClip")
    fps, after = _clip_fps(c, versions=(DC3_CLIP_REV,))
    d = dict(object_raw=bytes(c[4:after - 4]), fps=fps)
    r = _R(c, after)
    d['flags'] = r.u32()
    d['play_flags'] = r.u32()
    d['range_'] = r.f32()
    d['relative'] = r.sym()
    d['old_version'] = r.i32()
    d['do_not_compress'] = r.u8()
    d['node_size'] = r.u32()
    trans = []
    for _ in range(r.u32()):
        name = r.sym()
        n = r.u32()
        trans.append((name, [(r.f32(), r.f32()) for _ in range(n)]))
    d['transitions'] = trans
    d['beat_events'] = [(r.sym(), r.f32()) for _ in range(r.u32())]
    rr = _Reader(c, r.p)
    d['full'] = _read_char_bones_samples(rr)
    d['one'] = _read_char_bones_samples(rr)
    r.p = rr.p
    d['zeros'] = [(r.sym(), r.f32()) for _ in range(r.u32())]
    d['beat_track'] = [(r.f32(), r.f32()) for _ in range(r.u32())]
    d['sync_anim'] = r.sym()
    skel = []
    for _ in range(r.u32()):
        skel.append([(r.i32(), r.f32()) for _ in range(r.u32())])
    d['skeleton_samples'] = skel
    d['bitmask'] = r.i32()
    d['end'] = r.p
    return d


def encode_dc3_clip(d):
    """A CharClip v22 body (no end marker) from parse_dc3_clip's fields. 'full' and 'one'
    are (channels, samples, frame_times, compression) tuples."""
    from .gdrb_clip_exporter import encode_bones_samples
    w = _W()
    w.u32(DC3_CLIP_REV)
    w.b += d['object_raw']
    w.f32(d['fps'])
    w.u32(d['flags'])
    w.u32(d['play_flags'])
    w.f32(d['range_'])
    w.sym(d['relative'])
    w.i32(d['old_version'])
    w.u8(d['do_not_compress'])
    w.u32(sum(len(pairs) for _n, pairs in d['transitions']))
    w.u32(len(d['transitions']))
    for name, pairs in d['transitions']:
        w.sym(name)
        w.u32(len(pairs))
        for a, b in pairs:
            w.f32(a)
            w.f32(b)
    w.u32(len(d['beat_events']))
    for s, beat in d['beat_events']:
        w.sym(s)
        w.f32(beat)
    w.b += encode_bones_samples(*d['full'])
    w.b += encode_bones_samples(*d['one'])
    w.u32(len(d['zeros']))
    for s, wt in d['zeros']:
        w.sym(s)
        w.f32(wt)
    w.u32(len(d['beat_track']))
    for beat, frame in d['beat_track']:
        w.f32(beat)
        w.f32(frame)
    w.sym(d['sync_anim'])
    w.u32(len(d['skeleton_samples']))
    for smp in d['skeleton_samples']:
        w.u32(len(smp))
        for k, v in smp:
            w.i32(k)
            w.f32(v)
    w.i32(d['bitmask'])
    return bytes(w.b)


def block_args(blk):
    """A decoded sample block in the form encode_bones_samples takes."""
    return (blk['bones'],
            [{k: v for k, (_kind, v) in s.items()} for s in blk['samples']],
            blk['frame_times'], blk['compression'])


# ---------------------------------------------------------------------------------------
# A clip from sampled frames, shaped by a template clip
# ---------------------------------------------------------------------------------------

def _stem(ch):
    return ch.rsplit('.', 1)[0]


def _adjust_lead_in(frames, bpm, lead_in):
    """Trims or pads the start so the measure lands one beat into the clip. Returns
    (frames, samples dropped (+) or added (-))."""
    spb = CLIP_FPS * 60.0 / bpm
    shift = int(round((lead_in - 1.0) * spb))
    if shift > 0:
        if shift >= len(frames):
            raise DC3BoomyError(f"the lead-in ({lead_in:g} beats) is longer than the Action")
        return frames[shift:], shift
    if shift < 0:
        return [dict(frames[0]) for _ in range(-shift)] + frames, shift
    return frames, 0


def _split_travel(frames, root_stems):
    """Moves the pelvis's horizontal travel into bone_facing.pos, writing the root-level
    bones relative to it - the inverse of CharServoBone's MoveToFacing (no rotation)."""
    for f in frames:
        p = f.get('bone_pelvis.pos')
        if p is None:
            continue
        fx, fy = p[0], p[1]
        f['bone_facing.pos'] = (fx, fy, 0.0)
        for stem in root_stems:
            k = stem + '.pos'
            if k in f:
                x, y, z = f[k]
                f[k] = (x - fx, y - fy, z)


def build_dc3_clip(frames, root_stems, template, template_name, clip_name, bpm,
                   lead_in=1.0, split_travel=True, keep_foot_plants=False):
    """CharClip v22 bytes (no end marker) from sampled frames - [{channel: value}], '.pos'
    (x, y, z) and '.quat' (x, y, z, w) per bone stem, 30 per second - plus a summary."""
    from .gdrb_clip_exporter import _hinge_angle, HINGE_OFF_AXIS_WARN
    if not frames:
        raise DC3BoomyError("the Action has no frames")
    if bpm <= 0:
        raise DC3BoomyError("the tempo must be above 0 BPM")
    frames = [dict(f) for f in frames]
    animated = {_stem(k) for f in frames[:1] for k in f}
    frames, shifted = _adjust_lead_in(frames, bpm, lead_in)
    keyed_facing = 'bone_facing' in animated
    travel = False
    if split_travel and not keyed_facing and 'bone_pelvis' in animated:
        _split_travel(frames, root_stems)
        travel = True

    pose = {}
    for blk in (template['one'], template['full']):
        if blk['samples']:
            pose.update({k: v for k, (_kind, v) in blk['samples'][0].items()})
    channels = [(n, w) for n, w in template['full']['bones'] + template['one']['bones']]
    kinds = {n.rsplit('.', 1)[1] for n, _w in channels}
    if not kinds <= {'pos', 'quat', 'rotz'}:
        raise DC3BoomyError(f"the template clip has channel types this can't write: "
                            f"{', '.join(sorted(kinds - {'pos', 'quat', 'rotz'}))}")
    held, off_axis, prev_q, samples = set(), {}, {}, []
    for f in frames:
        s = {}
        for ch, _w in channels:
            stem, kind = ch.rsplit('.', 1)
            facing = stem == 'bone_facing'
            if stem == 'bone_footik' and kind == 'pos':
                v = f.get(ch) if keep_foot_plants else None
                s[ch] = tuple(v) if v is not None else (0.0, 0.0, 0.0)
                continue
            if kind == 'pos':
                if ch in f:
                    s[ch] = tuple(f[ch])
                else:
                    s[ch] = (0.0, 0.0, 0.0) if facing else pose[ch]
                    if not facing:
                        held.add(stem)
            elif kind == 'quat':
                q = f.get(ch)
                if q is None:
                    q = (0.0, 0.0, 0.0, 1.0) if facing else pose[ch]
                    if not facing:
                        held.add(stem)
                last = prev_q.get(ch)
                if last is not None and sum(a * b for a, b in zip(q, last)) < 0.0:
                    q = tuple(-c for c in q)
                prev_q[ch] = q
                s[ch] = tuple(q)
            else:                                           # rotz hinge
                q = f.get(stem + '.quat')
                if q is None:
                    s[ch] = 0.0 if facing else pose[ch]
                    if not facing:
                        held.add(stem)
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
    key = lambda c: (order[c[0].rsplit('.', 1)[1]], c[0])
    full_ch = sorted([c for c in channels if not constant(c[0])], key=key)
    one_ch = sorted([c for c in channels if constant(c[0])], key=key)
    comp = template['full']['compression']
    n = len(samples)
    full = (full_ch, [{c: x[c] for c, _ in full_ch} for x in samples], [], comp)
    one = (one_ch, [{c: samples[0][c] for c, _ in one_ch}] if one_ch else [], [], comp)

    bpf = bpm / 60.0 / CLIP_FPS
    last = float(n - 1)
    length = last * bpf
    beat_track = [(0.0, 0.0), (length, last)] if n > 1 else [(0.0, 0.0)]
    # The template's self-transition, kept at the same beat offsets from the clip's start.
    t_start = template['beat_track'][0][0] if template['beat_track'] else 0.0
    pairs = [(a - t_start, b - t_start)
             for name, ps in template['transitions'] if name == template_name
             for a, b in ps if 0.0 <= a - t_start <= length and 0.0 <= b - t_start <= length]
    transitions = [(clip_name, pairs)] if pairs else []
    d = dict(object_raw=template['object_raw'], fps=CLIP_FPS, flags=template['flags'],
             play_flags=template['play_flags'], range_=template['range_'],
             relative='', old_version=template['old_version'],
             do_not_compress=template['do_not_compress'], transitions=transitions,
             beat_events=[], full=full, one=one, zeros=list(template['zeros']),
             beat_track=beat_track, sync_anim='', skeleton_samples=[], bitmask=0)
    data = encode_dc3_clip(d)
    used = {_stem(c) for c, _ in channels}
    return data, dict(samples=n, seconds=last / CLIP_FPS, beats=length,
                      channels=len(channels), shifted=shifted, travel=travel,
                      keyed_facing=keyed_facing, held=sorted(held),
                      dropped=sorted(animated - used),
                      off_axis={k: v for k, v in off_axis.items() if v > HINGE_OFF_AXIS_WARN})


# ---------------------------------------------------------------------------------------
# Boomy move library folders
# ---------------------------------------------------------------------------------------

def _read_object_file(path):
    with open(path, 'rb') as f:
        data = f.read()
    return data[:-4] if data.endswith(END_MARKER) else data


def read_boomy_template(folder, clip_key=''):
    """A Boomy library move folder as a template: its move.json, the chosen clip parsed,
    and the files a new variant shares with it."""
    folder = os.path.abspath(folder)
    jpath = os.path.join(folder, 'move.json')
    if not os.path.isfile(jpath):
        raise DC3BoomyError(f"no move.json in {folder} - point Template Move at one move's "
                            f"folder in a Boomy move library")
    with open(jpath, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    missing = [k for k in MOVE_JSON_KEYS if k not in meta]
    if missing:
        raise DC3BoomyError(f"move.json is missing {', '.join(missing)}")
    clips = meta['clips'] or {}
    have = [k for k in clips if os.path.isfile(os.path.join(folder, k))]
    if not have:
        raise DC3BoomyError(f"none of the clips move.json lists are in {folder}")
    key = clip_key or have[0]
    if key not in have:
        raise DC3BoomyError(f"no clip '{key}' in the template (it has {', '.join(have)})")
    raw = _read_object_file(os.path.join(folder, key))
    clip = parse_dc3_clip(raw)
    if clip['end'] != len(raw):
        raise DC3BoomyError(f"template clip '{key}' doesn't end where its fields do")
    files = []
    ham = os.path.join(folder, meta['ham_move_name'])
    if not os.path.isfile(ham):
        raise DC3BoomyError(f"the template's HamMove {meta['ham_move_name']} is missing")
    files.append(ham)
    notes = []
    for name in (meta['milo_name'] + '.seq', meta['milo_name'] + '.tex',
                 meta['milo_name'] + '_sm.tex', 'move.png'):
        p = os.path.join(folder, name)
        if os.path.isfile(p):
            files.append(p)
        else:
            notes.append(f"template has no {name}")
    return dict(folder=folder, meta=meta, clip_key=key, clip=clip, clip_meta=clips[key],
                files=files, notes=notes)


def slug(name):
    s = re.sub(r'[^0-9a-zA-Z]+', '_', name).strip('_').lower()
    return s or 'custom_move'


def write_boomy_move(dest, template, move_name, display_name, clips, bpm, overwrite=False):
    """Writes one move folder: the template's HamMove, sequence and icons, the new clips
    (raw, no extension) and a move.json listing only them. `clips`: {clip key: bytes}."""
    if os.path.exists(dest):
        if not overwrite:
            raise DC3BoomyError(f"{dest} already exists (turn on Overwrite to replace it)")
        if not os.path.isfile(os.path.join(dest, '.boomy')):
            raise DC3BoomyError(f"{dest} exists and isn't a Boomy move folder - not replacing")
        shutil.rmtree(dest)
    os.makedirs(dest)
    for p in template['files']:
        shutil.copyfile(p, os.path.join(dest, os.path.basename(p)))
    tm, cm = template['meta'], template['clip_meta']
    entries = {}
    for key, data in clips.items():
        with open(os.path.join(dest, key), 'wb') as f:
            f.write(data)
        entries[key] = dict(avg_beats_per_second=round(bpm / 60.0, 6),
                            genre=cm.get('genre', ''), era=cm.get('era', ''),
                            flags=int(cm.get('flags', 0)), linked_to='', linked_from='')
    meta = dict(name=move_name, difficulty=tm['difficulty'], display_name=display_name,
                ham_move_name=tm['ham_move_name'], milo_name=tm['milo_name'],
                song_name=tm['song_name'], clips=entries)
    with open(os.path.join(dest, 'move.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(dest, '.boomy'), 'w', encoding='utf-8') as f:
        f.write('move1')
    return meta


# ---------------------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------------------

def _euler_only_bones(action):
    from .gdrb_clip_exporter import _action_fcurves
    rot = {}
    for fc in _action_fcurves(action):
        if not fc.data_path.startswith('pose.bones["'):
            continue
        bone = fc.data_path[len('pose.bones["'):fc.data_path.index('"]')]
        prop = fc.data_path.rsplit('.', 1)[-1]
        rot.setdefault(bone, set()).add(prop)
    return sorted(b for b, props in rot.items()
                  if 'rotation_euler' in props and 'rotation_quaternion' not in props)


def _action_targets_armature(action, arm):
    from .gdrb_clip_exporter import _action_fcurves
    for fc in _action_fcurves(action):
        if fc.data_path.startswith('pose.bones["'):
            bone = fc.data_path[len('pose.bones["'):fc.data_path.index('"]')]
            if bone in arm.pose.bones:
                return True
    return False


class EXPORT_OT_dc3_boomy_clips(bpy.types.Operator):
    """Write the armature's Actions as Dance Central 3 CharClips into a Boomy move library,
    each a new clip of a template move (which supplies the scoring data and icon)"""
    bl_idname = "export_scene.dc3_boomy_clips"
    bl_label = "Export Actions to CharClips for Boomy"
    bl_options = {'REGISTER'}

    directory: StringProperty(
        name="Move Library",
        description="The Boomy move library folder (the one holding dc1, dc2, dc3, ... dcs)",
        subtype='DIR_PATH',
    )
    filter_folder: BoolProperty(default=True, options={'HIDDEN'})

    # Plain text field on purpose: a path field's folder button can't open a second file
    # browser from inside this dialog. Paste the path.
    template_move: StringProperty(
        name="Template Move",
        description="A move folder in the library (with move.json) to base each new move on. "
                    "Its HamMove and sequence are copied unchanged, so the player is scored "
                    "against THIS move; its clip supplies the clip header and channel layout",
        default="",
    )
    template_clip: StringProperty(
        name="Template Clip",
        description="Which of the template's clips to copy the layout from (empty: the first)",
        default="",
    )
    category: EnumProperty(
        name="Category",
        description="Library category folder the moves go in (Boomy only lists these)",
        items=[(c, c, "") for c in BOOMY_CATEGORIES],
        default='dc3_dlc',
    )
    song_folder: StringProperty(
        name="Song Folder",
        description="Folder under the category grouping these moves in Boomy's browser",
        default="custom",
    )
    which: EnumProperty(
        name="Actions",
        items=[('ACTIVE', "Active Action", "Only the armature's active Action"),
               ('ALL', "All Matching Actions",
                "Every Action that animates this armature's bones (filtered by name below)")],
        default='ACTIVE',
    )
    name_filter: StringProperty(
        name="Name Contains",
        description="With All Matching Actions: only Actions whose names contain this",
        default="",
    )
    bpm: FloatProperty(
        name="Tempo (BPM)",
        description="The tempo the Action was animated to - it sets the clip's beat track. "
                    "0 uses the template clip's (from its move.json)",
        default=0.0, min=0.0, max=400.0,
    )
    lead_in: FloatProperty(
        name="Lead-in Beats",
        description="Beats of the Action before the measure's downbeat. Retail clips have 1 "
                    "(then 4 beats of move and about 1 of tail); the start is trimmed or "
                    "padded to match",
        default=1.0, min=0.0, max=8.0,
    )
    split_travel: BoolProperty(
        name="Root Travel to bone_facing",
        description="Move the pelvis's horizontal travel into bone_facing, as retail clips "
                    "store it (the DC importer folds it into the pelvis). Ignored if the "
                    "Action animates a bone_facing bone",
        default=True,
    )
    keep_foot_plants: BoolProperty(
        name="Keep Foot-Plant Keys",
        description="Write the Action's bone_footik keys (the game's per-foot 'planted' "
                    "flags). Off writes 'not planted', so the feet follow the animation",
        default=False,
    )
    include_twist: BoolProperty(
        name="Include Twist Bones",
        description="Also sample twist bones the Action keys. Normally off: the game solves "
                    "them, and retail move clips carry none",
        default=False,
    )
    overwrite: BoolProperty(
        name="Overwrite",
        description="Replace move folders this exporter wrote before",
        default=False,
    )

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def draw(self, context):
        col = self.layout.column()
        col.prop(self, "template_move")
        col.prop(self, "template_clip")
        col.separator()
        col.prop(self, "category")
        col.prop(self, "song_folder")
        col.prop(self, "which")
        if self.which == 'ALL':
            col.prop(self, "name_filter")
        col.separator()
        col.prop(self, "bpm")
        col.prop(self, "lead_in")
        col.prop(self, "split_travel")
        col.prop(self, "keep_foot_plants")
        col.prop(self, "include_twist")
        col.prop(self, "overwrite")

    def _actions(self, arm):
        if self.which == 'ACTIVE':
            action = arm.animation_data.action if arm.animation_data else None
            return [action] if action else []
        return [a for a in bpy.data.actions
                if self.name_filter.lower() in a.name.lower() and _action_targets_armature(a, arm)]

    def execute(self, context):
        from .gdrb_clip_exporter import sample_action
        arm = context.active_object
        if arm is None or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Select the Dance Central armature whose Actions to export")
            return {'CANCELLED'}
        if not self.template_move:
            self.report({'ERROR'}, "Set Template Move to a move folder in the Boomy library")
            return {'CANCELLED'}
        library = bpy.path.abspath(self.directory)
        if not os.path.isdir(library):
            self.report({'ERROR'}, f"Move library folder not found: {library}")
            return {'CANCELLED'}
        missing_cats = [c for c in BOOMY_CATEGORIES if not os.path.isdir(os.path.join(library, c))]
        if len(missing_cats) == len(BOOMY_CATEGORIES):
            _log(f"  note: {library} has none of Boomy's category folders - is it the "
                 f"library root?")
        actions = self._actions(arm)
        if not actions:
            self.report({'ERROR'}, "No Actions to export" if self.which == 'ALL'
                        else f"'{arm.name}' has no active Action")
            return {'CANCELLED'}
        try:
            tmpl = read_boomy_template(bpy.path.abspath(self.template_move), self.template_clip)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        bpm = self.bpm or float(tmpl['clip_meta'].get('avg_beats_per_second', 0) or 0) * 60.0
        if bpm <= 0:
            self.report({'ERROR'}, "The template has no tempo - set Tempo (BPM)")
            return {'CANCELLED'}
        fps = context.scene.render.fps / context.scene.render.fps_base
        root_stems = {(b.name.rsplit('.', 1)[0] if b.name.endswith(('.mesh', '.trans'))
                       else b.name) for b in arm.pose.bones if b.parent is None}
        song_dir = os.path.join(library, self.category, slug(self.song_folder) or 'custom')
        tm = tmpl['meta']
        _log(f"===== DC3 clips for Boomy: template '{tm['name']}' "
             f"({tm['ham_move_name']}, clip {tmpl['clip_key']}) at {bpm:g} BPM =====")
        for n in tmpl['notes']:
            _log(f"  note: {n}")
        done, failed = [], []
        for action in actions:
            name = slug(action.name)
            clip_key = f"{name}_v0"
            dest = os.path.join(song_dir, name)
            try:
                euler = _euler_only_bones(action)
                if euler:
                    raise DC3BoomyError(
                        f"{', '.join(euler[:6])}{' ...' if len(euler) > 6 else ''} "
                        f"rotate in Euler only - set those bones to Quaternion and re-key")
                _names, frames, _secs = sample_action(arm, action, fps, self.include_twist)
                data, info = build_dc3_clip(frames, root_stems, tmpl['clip'],
                                            tmpl['clip_key'], clip_key, bpm, self.lead_in,
                                            self.split_travel, self.keep_foot_plants)
                if END_MARKER in data:
                    # The game reads objects by structure, but MiloLib (Boomy) and other
                    # tools that split a milo at end markers would cut this clip short.
                    _log(f"  '{action.name}': WARNING - the clip's sample data happens to "
                         f"contain the end-marker bytes AD DE AD DE; nudge a key slightly "
                         f"and re-export if Boomy or MiloEditor misreads it")
                write_boomy_move(dest, tmpl, name, action.name, {clip_key: data}, bpm,
                                 self.overwrite)
            except Exception as e:
                failed.append(action.name)
                _log(f"  '{action.name}': FAILED - {e}")
                continue
            done.append(action.name)
            _log(f"  '{action.name}' -> {dest}")
            _log(f"    clip {clip_key}: {info['samples']} samples, {info['seconds']:.2f}s, "
                 f"{info['beats']:.2f} beats, {info['channels']} channels")
            if info['shifted'] > 0:
                _log(f"    trimmed {info['shifted']} samples of lead-in")
            elif info['shifted'] < 0:
                _log(f"    held the first pose for {-info['shifted']} samples of lead-in")
            if info['beats'] < 5.0:
                _log(f"    WARNING: shorter than lead-in + a 4-beat measure (5 beats); the "
                     f"measure won't be fully covered")
            if info['travel']:
                _log("    pelvis travel written to bone_facing")
            if info['held']:
                _log(f"    not in the Action, held at the template's first pose: "
                     f"{', '.join(info['held'][:12])}{' ...' if len(info['held']) > 12 else ''}")
            if info['dropped']:
                _log(f"    in the Action but not the template's layout, left out: "
                     f"{', '.join(info['dropped'][:12])}"
                     f"{' ...' if len(info['dropped']) > 12 else ''}")
            for bone, stray in sorted(info['off_axis'].items()):
                _log(f"    WARNING: {bone} is a hinge but bends up to "
                     f"{math.degrees(2 * math.asin(min(1.0, stray))):.1f} deg off its axis")
        if not done:
            self.report({'ERROR'}, f"Nothing exported - see the log ({failed[0]} failed)")
            return {'CANCELLED'}
        msg = f"Wrote {len(done)} move(s) to {song_dir}"
        if failed:
            msg += f"; {len(failed)} failed (see the log)"
        self.report({'WARNING'} if failed else {'INFO'}, msg)
        return {'FINISHED'}
