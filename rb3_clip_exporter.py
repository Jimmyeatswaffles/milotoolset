"""
Rock Band 3 CharClip export: writes Blender Actions as RB3 animation clips (CharClip version
19) and INJECTS them into a song's milo, alongside a CharClipGroup for each - so a song.anim
(made with Milo Mod, say) can call them.

===========================================================================================
CHARCLIP VERSION 19 - from CharClip::Load in the RB3 decompilation (system/char)
===========================================================================================
  19, the Hmx::Object header (revision, type symbol, has-tree flag and data tree, note),
  frames per second, flags, play flags, range, relative-clip symbol, old version (int),
  do-not-compress (byte), transitions (node size, vector count, then per vector a clip name
  and (beat, beat) pairs), beat events (count, then (symbol, beat)),
  the 'full' and 'one' CharBonesSamples (version 16, as GDRB and Dance Central),
  mZeros (count, then (bone symbol, weight)), the beat track (count, then (beat, frame)),
  and mSyncAnim (a symbol).

For version 19 the loader takes no old-revision branches: gOldRev is fixed at 13. Verified
by round trip - every retail RB3 clip checked re-encodes byte for byte, except the per-sample
padding, which Harmonix's tools fill with leftover memory and the engine skips by size.

A CharClipGroup (version 2) is the Hmx::Object header, the list of clip names it holds, an
int (mWhich) and flags; a directed cut's camera shot names the GROUP to play.
"""

import math
import os
import struct

import bpy
from bpy.props import StringProperty, BoolProperty, EnumProperty, FloatProperty
from bpy_extras.io_utils import ExportHelper

from .utilities import _log

RB3_CLIP_REV = 19
RB3_GROUP_REV = 2
END_MARKER = b'\xAD\xDE\xAD\xDE'


class RB3ClipError(Exception):
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


class _R:
    def __init__(self, c, p=0):
        self.c, self.p = c, p

    def u8(self):
        v = self.c[self.p]; self.p += 1; return v

    def u32(self):
        v = struct.unpack_from('>I', self.c, self.p)[0]; self.p += 4; return v

    def i32(self):
        v = struct.unpack_from('>i', self.c, self.p)[0]; self.p += 4; return v

    def f32(self):
        v = struct.unpack_from('>f', self.c, self.p)[0]; self.p += 4; return v

    def sym(self):
        n = self.u32()
        if n > 4096:
            raise RB3ClipError("implausible string length")
        v = self.c[self.p:self.p + n].decode('latin-1'); self.p += n; return v


# ---------------------------------------------------------------------------------------
# CharClip v19
# ---------------------------------------------------------------------------------------

def parse_rb3_clip(c):
    """Every field of an RB3 CharClip, in order. The object header (with its data tree)
    is kept as raw bytes; the sample blocks are decoded with the shared reader."""
    from .dc_clip_importer import _clip_fps
    from .viseme_importer import _Reader, _read_char_bones_samples
    if struct.unpack_from('>I', c, 0)[0] != RB3_CLIP_REV:
        raise RB3ClipError("not a version-19 CharClip")
    fps, after = _clip_fps(c, versions=(RB3_CLIP_REV,))
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
    d['end'] = r.p
    return d


def _block_args(blk):
    """A decoded sample block in the form encode_bones_samples takes."""
    return (blk['bones'],
            [{k: v for k, (_kind, v) in s.items()} for s in blk['samples']],
            blk['frame_times'], blk['compression'])


def encode_rb3_clip(d):
    """A CharClip v19 body (no end marker) from parse_rb3_clip's fields. 'full' and 'one'
    are (channels, samples, frame_times, compression) tuples."""
    from .gdrb_clip_exporter import encode_bones_samples
    w = _W()
    w.u32(RB3_CLIP_REV)
    w.b += d['object_raw']
    w.f32(d['fps'])
    w.u32(d['flags'])
    w.u32(d['play_flags'])
    w.f32(d['range_'])
    w.sym(d['relative'])
    w.i32(d['old_version'])
    w.u8(d['do_not_compress'])
    w.u32(d['node_size'])
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
    return bytes(w.b)


def encode_clip_group(clip_names, which=0, flags=0):
    """A CharClipGroup v2 body (no end marker): an empty object header, then the clips."""
    w = _W()
    w.u32(RB3_GROUP_REV)
    w.u32(2)            # Hmx::Object revision
    w.sym('')           # type
    w.u8(0)             # no data tree
    w.sym('')           # note
    w.u32(len(clip_names))
    for n in clip_names:
        w.sym(n)
    w.i32(which)
    w.i32(flags)
    return bytes(w.b)


def parse_clip_group(c):
    r = _R(c, 0)
    if r.u32() != RB3_GROUP_REV:
        raise RB3ClipError("not a version-2 CharClipGroup")
    r.u32(); r.sym()
    if r.u8():
        raise RB3ClipError("clip group with a data tree isn't handled")
    r.sym()
    clips = [r.sym() for _ in range(r.u32())]
    return dict(clips=clips, which=r.i32(), flags=r.i32(), end=r.p)


# ---------------------------------------------------------------------------------------
# A clip from an Action, shaped by a template
# ---------------------------------------------------------------------------------------
#
# As with GDRB (see gdrb_clip_exporter's template mode), a character takes its channel
# layout from its clip type's CharBoneDir resource, so a written clip copies a retail clip
# of the kind it stands in for: its object header (which holds the clip type - 'mic_body'
# for the singer's directed cut cam_ms_stagedive), flags, play flags, channel layout
# (names, types, order, weights), compression and beat rate. The Action supplies the poses:
# one sample per frame at 30 fps with no frame table (as RB3's idle clips are stored).
# Hinge channels (rotz) are converted from the Action's rotation. Bones the Action doesn't
# animate hold the template's first pose - except bone_facing, which stays at zero by
# default: the template's facing places ITS character (the stage dive starts 114 units
# back), while an imported Action carries its travel in the pelvis already.

def read_template_clip(milo_path, clip_name=''):
    """(clip name, parse_rb3_clip fields) for a CharClip inside an RB3 milo."""
    from .io import read_milo_container_body
    from .mesh_importer import _read_dir_entries
    from .viseme_importer import find_viseme_clips
    with open(milo_path, 'rb') as f:
        body = read_milo_container_body(f.read())
    _rev, _dt, _dn, ents = _read_dir_entries(body)
    names = [n for t, n in ents if t == 'CharClip']
    if not names:
        raise RB3ClipError(f"{os.path.basename(milo_path)} has no CharClip to use as a template")
    want = clip_name or names[0]
    if want not in names:
        raise RB3ClipError(f"no clip '{want}' in {os.path.basename(milo_path)} "
                           f"(it has {', '.join(names[:8])}{' ...' if len(names) > 8 else ''})")
    _d, chunks = find_viseme_clips(milo_path)
    chunk = chunks.get(want)
    if chunk is None:
        raise RB3ClipError(f"couldn't locate clip '{want}' in {os.path.basename(milo_path)}")
    return want, parse_rb3_clip(chunk)


def build_rb3_clip(arm_obj, action, scene_fps, template, template_name, clip_name,
                   include_twist=False, zero_facing=True):
    """CharClip v19 bytes (no end marker) for an Action, plus a summary."""
    from .gdrb_clip_exporter import sample_action, _hinge_angle, HINGE_OFF_AXIS_WARN, CLIP_FPS
    names, frames, seconds = sample_action(arm_obj, action, scene_fps, include_twist)
    if not frames:
        raise RB3ClipError("the Action has no frames")
    pose = {}
    for blk in (template['one'], template['full']):
        if blk['samples']:
            pose.update({k: v for k, (_kind, v) in blk['samples'][0].items()})
    channels = [(n, w) for n, w in template['full']['bones'] + template['one']['bones']]
    held, off_axis, prev_q, samples = set(), {}, {}, []
    for f in frames:
        s = {}
        for ch, _w in channels:
            stem, kind = ch.rsplit('.', 1)
            facing = stem == 'bone_facing' and zero_facing
            if kind == 'pos':
                if ch in f:
                    s[ch] = f[ch]
                else:
                    s[ch] = (0.0, 0.0, 0.0) if facing else pose[ch]
                    held.add(stem)
            elif kind == 'quat':
                q = f.get(ch)
                if q is None:
                    q = (0.0, 0.0, 0.0, 1.0) if facing else pose[ch]
                    held.add(stem)
                last = prev_q.get(ch)
                if last is not None and sum(a * b for a, b in zip(q, last)) < 0.0:
                    q = tuple(-c for c in q)
                prev_q[ch] = q
                s[ch] = q
            else:                                           # rotz hinge
                q = f.get(stem + '.quat')
                if q is None:
                    s[ch] = 0.0 if facing else pose[ch]
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
    key = lambda c: (order.get(c[0].rsplit('.', 1)[1], 3), c[0])
    full_ch = sorted([c for c in channels if not constant(c[0])], key=key)
    one_ch = sorted([c for c in channels if constant(c[0])], key=key)
    comp = template['full']['compression']
    n = len(samples)
    full = (full_ch, [{c: x[c] for c, _ in full_ch} for x in samples], [], comp)
    one = (one_ch, [{c: samples[0][c] for c, _ in one_ch}] if one_ch else [], [], comp)

    # Beat rate as the template's (beats per frame from its beat track).
    bt = template['beat_track']
    bpf = ((bt[-1][0] - bt[0][0]) / (bt[-1][1] - bt[0][1])) if len(bt) > 1 and bt[-1][1] > bt[0][1] else 0.1
    last = float(n - 1)
    beat_track = [(0.0, 0.0), (last * bpf, last)] if n > 1 else [(0.0, 0.0)]
    # Transitions that point back at the template clip itself become self-transitions of
    # the new clip, scaled into its length; transitions to other clips are dropped.
    t_len = (bt[-1][0] - bt[0][0]) if len(bt) > 1 else 0.0
    scale = (last * bpf / t_len) if t_len > 0 else 1.0
    transitions = [(clip_name, [(a * scale, b * scale) for a, b in pairs])
                   for name, pairs in template['transitions'] if name == template_name]
    d = dict(template, fps=CLIP_FPS, transitions=transitions,
             node_size=template['node_size'] if transitions else 0,
             beat_events=[], beat_track=beat_track, full=full, one=one)
    data = encode_rb3_clip(d)
    animated = {b.rsplit('.', 1)[0] if b.endswith(('.mesh', '.trans')) else b for b in names}
    used = {c.rsplit('.', 1)[0] for c, _ in channels}
    return data, dict(samples=n, seconds=(n - 1) / CLIP_FPS, channels=len(channels),
                      held=sorted(held), dropped=sorted(animated - used),
                      off_axis={k: v for k, v in off_axis.items() if v > HINGE_OFF_AXIS_WARN})


# ---------------------------------------------------------------------------------------
# Injecting into a song milo
# ---------------------------------------------------------------------------------------
#
# Song milos come in three layouts (see the investigation of okt_itwillrain, blacksheep,
# justthewayyouare and hellagood): a revision-28 'lipsync' directory (Magma/RBN customs),
# a revision-28 RB3 DLC directory (a reference to world/shared/director.milo, the lipsync
# INLINED as a subdirectory, then song.anim), and revision-25 legacy DLC. New objects go
# into the TOP-LEVEL table and their data after the last top-level object, so an inlined
# directory is never touched. Revision 25 is refused for now: RB3 clips in it are untested.
#
# Re-exporting replaces the previously injected objects of the same names. Those always sit
# at the end of the file (that's where this puts them), so they're found from the end
# markers - and each is parsed back to confirm the cut is exact before anything is removed.

INJECT_TYPES = ('CharClip', 'CharClipGroup')


def read_song_milo(path):
    from .io import read_milo_container_body
    with open(path, 'rb') as f:
        body = read_milo_container_body(f.read())
    r = _R(body, 0)
    rev = r.u32()
    if rev != 28:
        raise RB3ClipError(
            f"{os.path.basename(path)} is a revision-{rev} milo; injection supports the "
            f"revision-28 song milos RB3 and Magma use (revision 25 is legacy DLC)")
    dir_type, dir_name = r.sym(), r.sym()
    stc, sts = r.i32(), r.u32()
    entries = [(r.sym(), r.sym()) for _ in range(r.u32())]
    return dict(body=body, dir_type=dir_type, dir_name=dir_name, stc=stc, sts=sts,
                entries=entries, table_end=r.p)


def _tail_chunks(body, k):
    """The last k marker-delimited object chunks, as (start, end) excluding markers."""
    marks = []
    p = len(body)
    if not body.endswith(END_MARKER):
        raise RB3ClipError("the milo body doesn't end with an object end marker")
    for _ in range(k + 1):
        q = body.rfind(END_MARKER, 0, p)
        if q < 0:
            raise RB3ClipError("ran out of end markers looking for injected objects")
        marks.append(q)
        p = q
    return [(marks[i + 1] + 4, marks[i]) for i in range(k)][::-1]


def inject_objects(milo, new_objects):
    """New milo body: the song milo with new_objects [(type, name, data)] added, replacing
    earlier injected objects of the same names. Returns (body, notes)."""
    entries = list(milo['entries'])
    body = milo['body']
    new_names = {n for _t, n, _d in new_objects}
    notes = []
    # Previously injected objects to replace must be in the trailing run of clip objects.
    tail = 0
    while tail < len(entries) and entries[-1 - tail][0] in INJECT_TYPES:
        tail += 1
    clash = [(t, n) for t, n in entries if n in new_names]
    cut = len(body)
    if clash:
        tail_entries = entries[len(entries) - tail:]
        if any(c not in tail_entries for c in clash):
            raise RB3ClipError(
                f"{', '.join(n for _t, n in clash)} already exist in this milo but weren't "
                f"added by this exporter, so they won't be replaced")
        spans = _tail_chunks(body, tail)
        for (t, n), (s, e) in zip(tail_entries, spans):
            c = body[s:e]
            try:
                end = (parse_rb3_clip(c)['end'] if t == 'CharClip' else parse_clip_group(c)['end'])
            except Exception as ex:
                raise RB3ClipError(f"couldn't confirm the existing '{n}' ({ex}) - not replacing")
            if end != len(c):
                raise RB3ClipError(f"existing '{n}' doesn't end where expected - not replacing")
        # Remove the whole trailing run, then re-add the ones that aren't being replaced.
        keep = [(t, n, body[s:e]) for (t, n), (s, e) in zip(tail_entries, spans)
                if n not in new_names]
        cut = spans[0][0]           # keeps the marker that ends the object before them
        entries = entries[:len(entries) - tail]
        new_objects = keep + list(new_objects)
        notes.append(f"replaced {', '.join(n for _t, n in clash)}")
    old_syms = {milo['dir_type'], milo['dir_name']} | {s for e in milo['entries'] for s in e}
    added = {s for t, n, _d in new_objects for s in (t, n)} - old_syms
    stc = milo['stc'] + 2 * len(added)
    sts = milo['sts'] + sum(len(s) + 1 for s in added) + (64 if milo['sts'] == 0 else 0)
    w = _W()
    w.u32(28)
    w.sym(milo['dir_type'])
    w.sym(milo['dir_name'])
    w.i32(stc)
    w.u32(sts)
    entries += [(t, n) for t, n, _d in new_objects]
    w.u32(len(entries))
    for t, n in entries:
        w.sym(t)
        w.sym(n)
    w.b += body[milo['table_end']:cut]
    for _t, _n, data in new_objects:
        w.b += data + END_MARKER
    return bytes(w.b), notes


def wrap_uncompressed(body):
    """An uncompressed (CABEDEAF) single-block container, laid out as retail and Magma."""
    hdr = bytearray(0x810)
    struct.pack_into('<4I', hdr, 0, 0xCABEDEAF, 0x810, 1, len(body))
    struct.pack_into('<I', hdr, 16, len(body))
    return bytes(hdr) + body


# ---------------------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------------------

class EXPORT_OT_rb3_clip_inject(bpy.types.Operator, ExportHelper):
    """Write the active armature's Action as a Rock Band 3 animation clip (with a clip group)
    into a copy of a song's milo, so a song.anim can call it"""
    bl_idname = "export_scene.rb3_clip_inject"
    bl_label = "Inject RB3 CharClip Into Song Milo"
    bl_options = {'REGISTER'}

    filename_ext = ".milo_xbox"
    filter_glob: StringProperty(default="*.milo_xbox", options={'HIDDEN'})

    # Plain text fields on purpose: a path field's folder button can't open a second file
    # browser from inside this export dialog. Paste the paths.
    song_milo: StringProperty(
        name="Song Milo",
        description="The song's milo to add the clip to (revision 28: Magma/RBN customs or "
                     "RB3 DLC). It's read, not changed - the result is saved where you choose",
        default="",
    )
    template_milo: StringProperty(
        name="Template Milo",
        description="An RB3 milo holding a clip of the kind this replaces - e.g. a directed "
                     "cut such as cam_ms_stagedive_01.milo_xbox. Its clip type, flags, "
                     "channel layout and beat rate are copied",
        default="",
    )
    template_clip: StringProperty(
        name="Template Clip",
        description="Which clip in the template milo (empty: its first clip)",
        default="",
    )
    clip_name: StringProperty(
        name="Clip Name",
        description="Name for the clip (empty: the Action's name)",
        default="",
    )
    group_name: StringProperty(
        name="Clip Group",
        description="Name for the clip group holding it - what a camera shot or song.anim "
                     "calls. Empty: the clip name plus _01, as retail directed cuts are named",
        default="",
    )
    zero_facing: BoolProperty(
        name="Zero bone_facing",
        description="Keep bone_facing at zero when the Action doesn't animate it, rather "
                     "than holding the template's (which places the template's character)",
        default=True,
    )
    include_twist: BoolProperty(
        name="Include Twist Bones",
        description="Also write twist bones the Action keys. Normally off: the game solves "
                     "them itself",
        default=False,
    )

    def draw(self, context):
        col = self.layout.column()
        col.prop(self, "song_milo")
        col.prop(self, "template_milo")
        col.prop(self, "template_clip")
        col.separator()
        col.prop(self, "clip_name")
        col.prop(self, "group_name")
        col.prop(self, "zero_facing")
        col.prop(self, "include_twist")

    def execute(self, context):
        arm = context.active_object
        if arm is None or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Select the armature whose Action to export")
            return {'CANCELLED'}
        action = arm.animation_data.action if arm.animation_data else None
        if action is None:
            self.report({'ERROR'}, f"'{arm.name}' has no active Action")
            return {'CANCELLED'}
        if not self.song_milo or not self.template_milo:
            self.report({'ERROR'}, "Set both the Song Milo and the Template Milo paths")
            return {'CANCELLED'}
        song_path = bpy.path.abspath(self.song_milo)
        out_path = bpy.path.abspath(self.filepath)
        if os.path.normcase(os.path.abspath(song_path)) == os.path.normcase(os.path.abspath(out_path)):
            self.report({'ERROR'}, "Save to a new file, not over the song milo it reads")
            return {'CANCELLED'}
        clip_name = self.clip_name or action.name
        group_name = self.group_name or f"{clip_name}_01"
        if group_name == clip_name:
            self.report({'ERROR'}, "The clip and its group need different names")
            return {'CANCELLED'}
        try:
            milo = read_song_milo(song_path)
            t_name, template = read_template_clip(bpy.path.abspath(self.template_milo),
                                                  self.template_clip)
            fps = context.scene.render.fps / context.scene.render.fps_base
            clip, info = build_rb3_clip(arm, action, fps, template, t_name, clip_name,
                                        self.include_twist, self.zero_facing)
            group = encode_clip_group([clip_name])
            body, notes = inject_objects(milo, [('CharClip', clip_name, clip),
                                                ('CharClipGroup', group_name, group)])
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        with open(out_path, 'wb') as f:
            f.write(wrap_uncompressed(body))
        _log(f"===== RB3 clip '{clip_name}' (group '{group_name}') injected into "
             f"{os.path.basename(song_path)} -> {out_path} =====")
        _log(f"  template '{t_name}': {info['channels']} channels; {info['samples']} samples, "
             f"{info['seconds']:.2f}s")
        for n in notes:
            _log(f"  {n}")
        if info['held']:
            _log(f"  not in the Action, held: {', '.join(info['held'])}")
        if info['dropped']:
            _log(f"  in the Action but not the template's layout, left out: "
                 f"{', '.join(info['dropped'][:12])}")
        for bone, stray in sorted(info['off_axis'].items()):
            _log(f"  WARNING: {bone} is a hinge in the template but bends up to "
                 f"{math.degrees(2 * math.asin(min(1.0, stray))):.1f} deg off its axis")
        self.report({'INFO'}, f"Injected '{clip_name}' and group '{group_name}' "
                              f"({info['seconds']:.2f}s) into {os.path.basename(out_path)}")
        return {'FINISHED'}
