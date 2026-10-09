"""
Dance Central 2 and 3 routine import: lays a song's CharClip Actions out on the NLA in the
order its song.anim plays them.

Run it after importing the song's clips (Milo CharClip Import > Dance Central 2 / 3), with
the dancer's armature selected and pointed at the same song milo.

===========================================================================================
SONG.ANIM - verified on hello / gangnamstyle (DC3) and forgetyou (DC2)
===========================================================================================
Each song milo has easy/, medium/ and expert/ directories, each holding a PropAnim called
song.anim (identified by which directory's byte range contains it). Its five tracks all
animate the HamDirector: clip (which clip starts when), move (the move being scored),
practice (practice-mode sections), shot (camera) and postproc. All are symbol tracks except
postproc. DC3 writes PropAnim revision 15 and DC2 revision 14, but every per-track field that
depends on the revision appears from 13 on, so both read identically - one importer.

A track is: key type (6 = symbol), target object, property path (a small data tree), an
interpolation, an interpolation-handler symbol, an exception id, one flag byte, then the
keys - for symbol keys, (value, frame). Frames are at 30 fps, and every key lands on a beat.

===========================================================================================
HOW A ROUTINE IS PLAYED
===========================================================================================
Expert routines name the song's long expert_* clips directly. Easy and medium routines name
MOVES, one per bar (Low_Flow at 225, 281.2, 337.5 ... - 4 beats apart at 128 BPM), and the
clip files are named in pairs, A_B: move A leading into move B. So a move key followed by
move B plays clip A_B; a key that names a clip outright (expert_01a, rest) plays that clip.
Routines begin with 'groove' (a pre-dance idle with no clip in the song file) and expert
routines end with an empty key.

No clip in the song file carries a separate crop. Each plays from its first frame and is cut
off where the next key starts - often heavily: Gangnam Style uses only 327 of expert_02's
1,658 frames. Every clip plays in beat time aligned to whole beats (play flags 0x1000), and
each one's beat track is in the song's own beats at the song's tempo - Hello's
Low_Flow_Pretty_Dress runs beats 37.93 to 41.05 at exactly 128 BPM - so beat-time playback is
the clip's native 30 fps speed and nothing is stretched. A move clip covers ~3.1 of its 4
beats; the rest of the bar is where the game blends into the next move (each clip's
transition list gives those blend beats). Here the last pose is simply held until the next
strip.

Choices made here, not read from the data:
  * The final move of an easy/medium routine has no next move to pick a transition from, so
    it uses the move's own repeat clip, A_A.
  * Crossfades (on by default) use the NLA's own blend in/out over a fixed overlap - 16
    frames, the overlap the retail section clips carry past their cut point - rather than
    each clip's transition beats.
"""

import math
import struct

import bpy
from bpy.props import StringProperty, BoolProperty, EnumProperty
from bpy_extras.io_utils import ImportHelper

from .io import read_milo_container_body, _mw_collect_directory_meta
from .utilities import _log


_KEY_TYPES = {0: 'float', 1: 'color', 2: 'object', 3: 'bool', 4: 'quat', 5: 'vector3',
              6: 'symbol'}
_DIFFICULTIES = ('easy', 'medium', 'expert')
_SONG_FPS = 30.0


class DCRoutineError(Exception):
    pass


# ---------------------------------------------------------------------------------------
# song.anim
# ---------------------------------------------------------------------------------------

def _sym(c, p):
    n = struct.unpack_from('>I', c, p)[0]
    if n > 512:
        raise ValueError("string too long")
    return c[p + 4:p + 4 + n].decode('latin-1'), p + 4 + n


def _prop_path(c, p):
    """A property path: has-tree byte, then a data tree whose header is a 16-bit node count
    and a 32-bit line (6 bytes - the form these PropAnims use), then the nodes."""
    if c[p] != 1:
        return [], p + 1
    p += 1
    count = struct.unpack_from('>h', c, p)[0]
    if not 0 <= count <= 64:
        raise ValueError("bad property path")
    p += 6
    out = []
    for _ in range(count):
        t = struct.unpack_from('>I', c, p)[0]; p += 4
        if t == 0:
            out.append(struct.unpack_from('>i', c, p)[0]); p += 4
        elif t == 1:
            out.append(struct.unpack_from('>f', c, p)[0]); p += 4
        elif t in (2, 5, 18):
            s, p = _sym(c, p)
            out.append(s)
        else:
            raise ValueError(f"unexpected property path node {t}")
    return out, p


def _parse_track(c, p):
    """One PropKeys track (PropKeys::Load), returning (track, end)."""
    kt = struct.unpack_from('>I', c, p)[0]
    if kt not in _KEY_TYPES:
        raise ValueError
    target, q = _sym(c, p + 4)
    prop, q = _prop_path(c, q)
    interp = struct.unpack_from('>I', c, q)[0]
    q += 4                                    # interpolation
    _handler, q = _sym(c, q)
    q += 4                                    # exception id
    q += 1                                    # flag (revision 13+)
    n = struct.unpack_from('>I', c, q)[0]; q += 4
    if n > 100000:
        raise ValueError
    keys = []
    for _ in range(n):
        if kt in (6, 2):
            v, q = _sym(c, q)
        elif kt == 0:
            v = struct.unpack_from('>f', c, q)[0]; q += 4
        elif kt == 3:
            v = c[q]; q += 1
        elif kt == 5:
            v = struct.unpack_from('>3f', c, q); q += 12
        else:
            v = struct.unpack_from('>4f', c, q); q += 16
        f = struct.unpack_from('>f', c, q)[0]; q += 4
        keys.append((v, f))
    return dict(type=_KEY_TYPES[kt], target=target, prop=prop, interp=interp, keys=keys), q


def parse_song_anim(c):
    """{property name: track} for a song.anim's tracks. Tracks are found in order and each
    one is only accepted if it parses into a named target with non-decreasing key frames."""
    tracks, p = {}, 0
    while p < len(c) - 12:
        try:
            tr, q = _parse_track(c, p)
            frames = [k[1] for k in tr['keys']]
            if tr['target'] and tr['prop'] and all(b >= a for a, b in zip(frames, frames[1:])):
                tracks[str(tr['prop'][0])] = tr
                p = q
                continue
        except (ValueError, struct.error, IndexError, UnicodeDecodeError):
            pass
        p += 1
    return tracks


def read_routines(filepath):
    """(dir_name, {difficulty: {property: track}}) from a DC2/DC3 song milo. Each song.anim
    is assigned to the difficulty directory whose byte range contains it."""
    with open(filepath, 'rb') as f:
        body = read_milo_container_body(f.read())
    _, entries = _mw_collect_directory_meta(body, 0)
    # The directory's own name, from its header (revision, type, name) - the same name the
    # CharClip importer records on each Action as milo_clip_set.
    _rev = struct.unpack_from('>I', body, 0)[0]
    _dtype, q = _sym(body, 4)
    dir_name, _q = _sym(body, q)
    dirs = [(n, s, e) for t, n, s, e in entries if t == 'ObjectDir' and n in _DIFFICULTIES]
    out = {}
    for t, n, s, e in entries:
        if t != 'PropAnim' or n != 'song.anim':
            continue
        owner = next((d for d, ds, de in dirs if ds <= s and e <= de), None)
        if owner:
            out[owner] = parse_song_anim(body[s:e])
    return dir_name, out


def plan_routine(clip_keys, available):
    """Turns a clip track's keys into slots: [(start_frame, end_frame or None, clip or None,
    note)]. `available` is the set of clip names that have an imported Action."""
    keys = [(str(v), f) for v, f in clip_keys]
    slots = []
    for i, (name, start) in enumerate(keys):
        end = keys[i + 1][1] if i + 1 < len(keys) else None
        if name == '':
            continue                                   # end-of-routine marker
        nxt = next((k[0] for k in keys[i + 1:i + 2] if k[0] != ''), None)
        if name in available:
            slots.append((start, end, name, None))
            continue
        if name == 'groove':
            slots.append((start, end, None, "'groove' (the pre-dance idle) has no clip in "
                                             "the song file"))
            continue
        if nxt is not None:
            want = f"{name}_{nxt}"
            note = None
        else:
            want = f"{name}_{name}"
            note = f"final move '{name}': using its repeat clip '{want}'"
        if want in available:
            slots.append((start, end, want, note))
        else:
            slots.append((start, end, None, f"move '{name}'"
                          + (f" into '{nxt}'" if nxt else "")
                          + f": no clip '{want}'"))
    return slots


# ---------------------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------------------

class IMPORT_OT_dc_routine(bpy.types.Operator, ImportHelper):
    """Lay a Dance Central 2/3 song's imported CharClip Actions out on the NLA in the order
    its song.anim plays them. Import the song's CharClips onto the armature first"""
    bl_idname = "import_scene.dc_routine"
    bl_label = "Import DC Routine To NLA"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".milo_xbox"
    filter_glob: StringProperty(default="*.milo_xbox", options={'HIDDEN'})

    difficulty: EnumProperty(
        name="Difficulty",
        description="Which routine to lay out. Expert plays the song's long expert_ "
                     "sections; Easy and Medium are built move by move from transition clips",
        items=[('expert', "Expert", ""), ('medium', "Medium", ""), ('easy', "Easy", "")],
        default='expert',
    )
    crossfade: BoolProperty(
        name="Crossfade Between Clips",
        description="Blend each clip into the next over a short overlap, using the NLA's own "
                     "blend in/out, as the game blends between clips. Clips alternate between "
                     "two tracks so they can overlap. Off: clips butt end to end on one track",
        default=True,
    )
    crossfade_frames: bpy.props.FloatProperty(
        name="Crossfade Length",
        description="Overlap between consecutive clips, in the game's 30 fps frames. 16 is "
                     "the overlap the retail section clips carry past their cut point",
        default=16.0, min=1.0, max=60.0,
    )
    hold_gaps: BoolProperty(
        name="Hold Between Clips",
        description="Without crossfading: hold each clip's last pose until the next one "
                     "starts. (Crossfading always holds through the gap.) A move clip covers "
                     "about 3 of its 4 beats; in the game the rest is a blend into the next "
                     "move. Slots with no clip are left empty either way",
        default=True,
    )

    def execute(self, context):
        arm = context.active_object
        if arm is None or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Select the dancer's armature first.")
            return {'CANCELLED'}
        try:
            dir_name, routines = read_routines(self.filepath)
        except Exception as e:
            _log(f"DC ROUTINE IMPORT FAILED: {e}")
            self.report({'ERROR'}, f"Could not read the song milo: {e}")
            return {'CANCELLED'}
        tracks = routines.get(self.difficulty)
        if not tracks or 'clip' not in tracks:
            self.report({'ERROR'}, f"No {self.difficulty} routine (song.anim clip track) in "
                                   f"this milo.")
            return {'CANCELLED'}

        actions = {}
        for a in bpy.data.actions:
            if a.get("milo_clip") and a.get("milo_clip_set", dir_name) == dir_name:
                actions[a.get("milo_clip_name")] = a
        if not actions:
            self.report({'ERROR'}, f"No imported CharClips from '{dir_name}' - import the "
                                   f"song's clips first (Milo CharClip Import > Dance "
                                   f"Central 2 or 3).")
            return {'CANCELLED'}

        slots = plan_routine(tracks['clip']['keys'], set(actions))
        scene = context.scene
        scale = (scene.render.fps / max(scene.render.fps_base, 1e-6)) / _SONG_FPS
        _log(f"===== Laying out the {self.difficulty} routine of '{dir_name}' "
             f"({len(slots)} slots) =====")
        if abs(scale - 1.0) > 1e-6:
            _log(f"  scene is at {scale * _SONG_FPS:g} fps; routine frames are scaled from "
                 f"the game's 30")

        ad = arm.animation_data or arm.animation_data_create()
        if ad.action is not None:
            _log(f"  note: the armature's active Action ('{ad.action.name}') plays on top of "
                 f"the NLA - clear it to see the routine")
        base_name = f"DC Routine ({self.difficulty})"
        for old in [t for t in ad.nla_tracks
                    if t.name in (base_name, base_name + " A", base_name + " B")]:
            ad.nla_tracks.remove(old)
        if self.crossfade:
            # Two tracks so neighbouring clips can overlap; B sits above A.
            tracks = [ad.nla_tracks.new(), ad.nla_tracks.new()]
            tracks[0].name, tracks[1].name = base_name + " A", base_name + " B"
        else:
            tracks = [ad.nla_tracks.new()]
            tracks[0].name = base_name

        placed, empty, strips = 0, [], []
        filled = [s[2] is not None for s in slots]
        last_frame = 1.0
        fade = self.crossfade_frames * scale
        for i, (start, end, clip, note) in enumerate(slots):
            if note:
                if clip is None:
                    empty.append(f"frame {start:.1f}: {note}")
                else:
                    _log(f"  frame {start:.1f}: {note}")
            if clip is None:
                continue
            action = actions[clip]
            a0, a1 = action.frame_range
            s0 = 1.0 + start * scale
            nxt_filled = i + 1 < len(slots) and filled[i + 1]
            overlap = 0.0
            if end is None:
                length = a1 - a0
            elif self.crossfade and nxt_filled:
                # Run past the next key by the crossfade, holding the last pose if the clip
                # ends first (an F-curve holds its last value beyond its final key). Capped
                # below the next slot so clip i never reaches clip i+2 on the same track.
                nxt_end = slots[i + 2][0] if i + 2 < len(slots) else None
                room = ((nxt_end - end) * scale * 0.9) if nxt_end is not None else fade
                overlap = max(0.0, min(fade, room))
                length = (end - start) * scale + overlap
            else:
                length = min(a1 - a0, (end - start) * scale - 1e-3)
            track = tracks[placed % len(tracks)]
            strip = track.strips.new(f"{clip} @{start:.0f}", int(math.ceil(s0)), action)
            try:
                strip.action_slot = action.slots[0]
            except (AttributeError, IndexError, TypeError):
                pass
            strip.action_frame_start = a0
            strip.action_frame_end = a0 + max(length, 1e-3)
            try:
                strip.frame_start_ui = s0          # the key's exact (fractional) frame
            except (AttributeError, TypeError):
                pass
            strip.blend_type = 'REPLACE'
            try:
                strip.use_auto_blend = False
            except AttributeError:
                pass
            if self.crossfade:
                strip.extrapolation = 'NOTHING'
            else:
                strip.extrapolation = 'HOLD_FORWARD' if (self.hold_gaps and nxt_filled) \
                    else 'NOTHING'
            strips.append((strip, placed % len(tracks), overlap, nxt_filled))
            placed += 1
            last_frame = max(last_frame, s0 + length)

        # Crossfades: whichever of the two strips is on the upper track ramps its influence -
        # the incoming one blends in, or the outgoing one blends out over the one beneath.
        fades = 0
        if self.crossfade:
            for (sa, ta, ov, nxt), (sb, tb, _o, _n) in zip(strips, strips[1:]):
                if not nxt or ov <= 0.0:
                    continue
                if tb == 1:
                    sb.blend_in = ov
                else:
                    sa.blend_out = ov
                fades += 1
            _log(f"  crossfades: {fades}, {self.crossfade_frames:g} game frame(s) each "
                 f"(shorter where the next slot is too short)")

        for e in empty:
            _log(f"  EMPTY {e}")
        scene.frame_start = 1
        scene.frame_end = max(scene.frame_end, int(math.ceil(last_frame)))
        summary = (f"Placed {placed} clip(s) on {len(tracks)} NLA track(s) '{base_name}'"
                   + (f"; {len(empty)} slot(s) left empty (see log)" if empty else ""))
        _log(f"===== {summary} =====")
        self.report({'WARNING' if empty else 'INFO'}, summary)
        return {'FINISHED'}


# ---------------------------------------------------------------------------------------
# Face animation ("lipsync")
# ---------------------------------------------------------------------------------------
#
# Each difficulty directory holds a CharLipSync, dancer_face.lipsync, but it carries no
# timeline - only the list of expressions and a starting preset, about 700 bytes. The song's
# face is animated by the PropAnim beside it, dancer_face.anim: one float track per
# expression (Angry, Grin, Smirk ... and brow_* tracks), each keying that expression's weight
# (0-1) on dancer_face.lipsync at the game's 30 fps. Verified on all nine (three songs x three
# difficulties): DC2 writes PropAnim revision 14 and DC3 revision 15, same layout. The tracks
# use linear (kLinear) or smooth (kHermite) interpolation, never stepped.
#
# The expression names are the DC viseme set's own clip names, so the weights land on the
# same viseme_<name> channels a .lipsync import creates, and Bake Lipsync Preview drives the
# DC visemes (imported with the Rock Band 3 viseme option) from them unchanged.

_INTERP_TO_BLENDER = {0: 'CONSTANT', 1: 'LINEAR', 2: 'BEZIER', 3: 'BEZIER', 4: 'BEZIER',
                      5: 'BEZIER', 6: 'BEZIER'}


def read_face_anims(filepath):
    """(dir_name, {difficulty: [(expression, interpolation, [(weight, frame)])]})."""
    with open(filepath, 'rb') as f:
        body = read_milo_container_body(f.read())
    _, entries = _mw_collect_directory_meta(body, 0)
    _dtype, q = _sym(body, 4)
    dir_name, _q = _sym(body, q)
    dirs = [(n, s, e) for t, n, s, e in entries if t == 'ObjectDir' and n in _DIFFICULTIES]
    out = {}
    for t, n, s, e in entries:
        if t != 'PropAnim' or n != 'dancer_face.anim':
            continue
        owner = next((d for d, ds, de in dirs if ds <= s and e <= de), None)
        if owner is None:
            continue
        c = body[s:e]
        p, tracks = 0, []
        while p < len(c) - 12:
            try:
                tr, q2 = _parse_track(c, p)
                frames = [k[1] for k in tr['keys']]
                if (tr['type'] == 'float' and tr['prop']
                        and all(b >= a for a, b in zip(frames, frames[1:]))):
                    tracks.append((str(tr['prop'][-1]), tr['interp'], tr['keys']))
                    p = q2
                    continue
            except (ValueError, struct.error, IndexError, UnicodeDecodeError):
                pass
            p += 1
        out[owner] = tracks
    return dir_name, out


class IMPORT_OT_dc_lipsync(bpy.types.Operator, ImportHelper):
    """Import a Dance Central 2/3 song's facial animation from its milo as viseme weight
    channels - the same kind a .lipsync import creates - for Bake Lipsync Preview to drive
    the DC visemes with"""
    bl_idname = "import_scene.dc_lipsync"
    bl_label = "Import DC Face Animation"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".milo_xbox"
    filter_glob: StringProperty(default="*.milo_xbox", options={'HIDDEN'})

    difficulty: EnumProperty(
        name="Difficulty",
        description="Each difficulty has its own facial animation",
        items=[('expert', "Expert", ""), ('medium', "Medium", ""), ('easy', "Easy", "")],
        default='expert',
    )
    set_scene_range: BoolProperty(
        name="Set Scene Frame Range",
        description="Set the scene's frame range to cover the facial animation",
        default=True,
    )

    def execute(self, context):
        from .lipsync_importer import (ensure_weight_props, build_lipsync_action,
                                       _action_channelbag, LIPSYNC_FPS)
        obj = context.active_object
        if obj is None or obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Select the dancer's armature first.")
            return {'CANCELLED'}
        try:
            dir_name, anims = read_face_anims(self.filepath)
        except Exception as e:
            _log(f"DC FACE ANIMATION IMPORT FAILED: {e}")
            self.report({'ERROR'}, f"Could not read the song milo: {e}")
            return {'CANCELLED'}
        tracks_in = anims.get(self.difficulty)
        if not tracks_in:
            self.report({'ERROR'}, f"No {self.difficulty} facial animation "
                                   f"(dancer_face.anim) in this milo.")
            return {'CANCELLED'}

        tracks = {name: [(frame, weight) for weight, frame in keys]
                  for name, _interp, keys in tracks_in}
        names = sorted(tracks)
        ensure_weight_props(obj, names)
        song = f"{dir_name}_{self.difficulty}"
        action, total = build_lipsync_action(obj, song, tracks, 'LINEAR')
        modes = {name: _INTERP_TO_BLENDER.get(interp, 'BEZIER')
                 for name, interp, _keys in tracks_in}
        cb = _action_channelbag(action)
        if cb is not None:
            for fc in cb.fcurves:
                name = fc.data_path.split('viseme_', 1)[-1].rstrip('"]')
                mode = modes.get(name, 'LINEAR')
                for kp in fc.keyframe_points:
                    kp.interpolation = mode
                    if mode == 'BEZIER':
                        kp.handle_left_type = kp.handle_right_type = 'AUTO_CLAMPED'
                fc.update()
        ad = obj.animation_data or obj.animation_data_create()
        ad.action = action
        try:
            ad.action_slot = action.slots[0]
        except (AttributeError, IndexError, TypeError):
            pass

        last = max((f for keys in tracks.values() for f, _w in keys), default=0.0)
        if self.set_scene_range:
            context.scene.frame_start = 1
            context.scene.frame_end = int(math.ceil(last)) + 1
        used = [n for n in names if any(w > 0.0 for _f, w in tracks[n])]
        _log(f"===== DC face animation '{song}': {len(names)} expression track(s), "
             f"{total} key(s), {last / LIPSYNC_FPS:.1f}s =====")
        _log(f"  used: {', '.join(used)}")
        _log("  Import the song's DC visemes (Milo Viseme Import > Rock Band 3) onto the same "
             "armature, then Bake Lipsync Preview.")
        self.report({'INFO'}, f"Imported {len(names)} DC face track(s) as '{action.name}'")
        return {'FINISHED'}
