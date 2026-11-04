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

import bisect
import math
import os
import struct

import bpy
from bpy.props import StringProperty, BoolProperty, EnumProperty
from bpy_extras.io_utils import ImportHelper

from .io import read_milo_container_body, _mw_collect_directory_meta
from .utilities import _log
from .anim_importer import _source_channelbag


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


def _quat_curves(action):
    """{data_path: [w, x, y, z F-Curves]} for every bone rotation_quaternion in `action`."""
    cb = _source_channelbag(action)
    out = {}
    for fc in (cb.fcurves if cb else ()):
        dp = fc.data_path
        if dp.startswith('pose.bones[') and dp.endswith('.rotation_quaternion'):
            out.setdefault(dp, [None] * 4)[fc.array_index] = fc
    return {k: v for k, v in out.items() if None not in v}


def _midi_tempos(data):
    """[(tick, microseconds per quarter note)] from a Standard MIDI File's tempo events."""
    if data[:4] != b'MThd':
        return []
    ntrk = struct.unpack('>H', data[10:12])[0]
    pos, out = 8 + struct.unpack('>I', data[4:8])[0], []
    for _ in range(ntrk):
        if data[pos:pos + 4] != b'MTrk':
            break
        end = pos + 8 + struct.unpack('>I', data[pos + 4:pos + 8])[0]
        p, tick, status = pos + 8, 0, 0

        def vlq(p):
            v = 0
            while True:
                b = data[p]
                p += 1
                v = (v << 7) | (b & 0x7F)
                if not b & 0x80:
                    return v, p
        while p < end:
            dt, p = vlq(p)
            tick += dt
            b = data[p]
            if b >= 0x80:
                status, p = b, p + 1
            if status == 0xFF:
                typ = data[p]
                ln, q = vlq(p + 1)
                if typ == 0x51 and ln == 3:
                    out.append((tick, int.from_bytes(data[q:q + 3], 'big')))
                p = q + ln
            elif status in (0xF0, 0xF7):
                ln, p = vlq(p)
                p += ln
            elif (status & 0xF0) in (0xC0, 0xD0):
                p += 1
            else:
                p += 2
        pos = end
    return sorted(out)


def find_song_bpm(filepath):
    """(bpm, note) from the song's <song>.mid, looked for in the folders above the milo
    (songs/<song>/gen/x.milo_xbox -> songs/<song>/<song>.mid). bpm is None if not found."""
    d = os.path.dirname(os.path.abspath(filepath))
    for _ in range(4):
        want = os.path.basename(d).lower() + ".mid"
        try:
            for name in os.listdir(d):
                if name.lower() == want:
                    tempos = _midi_tempos(open(os.path.join(d, name), 'rb').read())
                    if tempos:
                        distinct = sorted({round(60e6 / t[1], 2) for t in tempos})
                        note = f"{name}: {distinct[0]:g} BPM"
                        if len(distinct) > 1:
                            note += f" (the song changes tempo - {len(distinct)} different values; using the first)"
                        return 60e6 / tempos[0][1], note
        except (OSError, IndexError, struct.error):
            pass
        d = os.path.dirname(d)
    return None, "no <song>.mid found"


def read_beat_tracks(filepath, frame_counts):
    """{clip: [(beat, frame), ...]} from the CharClip chunks in `filepath`. Every DC3 clip stores
    a beat track (CharClip::mBeatTrack: a beat value for each frame) at the end of its chunk, and
    the game advances a clip in song beats, not seconds - unless its play flags have kPlayRealTime
    (0x200). `frame_counts` = {clip: sample count}. Returns ({clip: beat track}, {clip: playFlags})."""
    from .viseme_importer import find_viseme_clips
    from .dc_clip_importer import _clip_fps
    try:
        _dn, raw = find_viseme_clips(filepath)
    except Exception as e:
        _log(f"  beat tracks: could not read clips from this milo ({e})")
        return {}, {}
    out, flags = {}, {}
    for name, n in frame_counts.items():
        ch = raw.get(name)
        if ch is None:
            continue
        try:                              # CharClip::Load: fps, flags, playFlags
            flags[name] = struct.unpack_from('>I', ch, _clip_fps(ch, versions=(22,))[1] + 4)[0]
        except Exception:
            pass
        if n < 2:
            continue
        i = ch.rfind(struct.pack('>f', float(n - 1)))
        keys, p = [], i - 4               # the last (beat, frame) pair, walking back
        while i >= 4 and p >= 4:
            beat, fr = struct.unpack_from('>ff', ch, p)
            if fr != float(n - 1 - len(keys)):
                break
            keys.append((beat, fr))
            p -= 8
        keys.reverse()
        if len(keys) == n and keys[0][1] == 0.0 and keys[-1][0] > keys[0][0]:
            out[name] = keys
    return out, flags


def bpm_from_beat_tracks(beat_tracks):
    """(bpm, note) from the clips' own beat tracks. A clip's beat track gives the song beat of
    each of its frames, at the game's 30 fps, so its slope is the song's tempo - Hello's
    Low_Flow_Pretty_Dress runs beats 37.93 to 41.05 over its frames at exactly 128 BPM. The
    median over all clips, so one odd clip can't skew it. (None, note) if there are none."""
    rates = []
    for keys in beat_tracks.values():
        beats = keys[-1][0] - keys[0][0]
        frames = keys[-1][1] - keys[0][1]
        if beats > 1e-6 and frames > 0:
            rates.append(_SONG_FPS * 60.0 * beats / frames)
    if not rates:
        return None, "no clip beat tracks"
    rates.sort()
    bpm = rates[len(rates) // 2]
    return bpm, f"from {len(rates)} clip beat track(s)"


def beat_warp(action, keys, frames_per_beat):
    """Returns a copy ('<clip>.beat') of `action` re-timed the way the game plays it: the key for
    clip frame k moves to start + (beat_k - beat_0) * frames_per_beat. Clips whose beat track is
    not a straight line (slow or fast parts inside the move) come out exact."""
    new = action.copy()
    new.name = action.name + ".beat"
    new["milo_clip"] = 0
    new["milo_flip"] = 1
    a0 = action.frame_range[0]
    b0 = keys[0][0]
    cb = _source_channelbag(new)
    for fc in (cb.fcurves if cb else ()):
        for kp in fc.keyframe_points:
            k = min(max(int(round(kp.co.x - a0)), 0), len(keys) - 1)
            dx = a0 + (keys[k][0] - b0) * frames_per_beat - kp.co.x
            kp.co.x += dx
            kp.handle_left.x += dx
            kp.handle_right.x += dx
        fc.update()
    return new


def unroll_quats(action):
    """Custom DLC: some DLC clips store a bone's quaternion as -q for the rest of the clip
    from one frame on (same rotation, opposite sign: expert_02h/02i/02j clavicles, 03g/03h
    right thigh). Blender interpolates q -> -q straight through zero, so that frame twitches.
    Returns `action`, or a copy ('<clip>.fix') where every bone's quaternion keys are flipped
    to stay on one side (dot with the previous key >= 0)."""
    curves = _quat_curves(action)
    def bad(cs):
        pts = [fc.keyframe_points for fc in cs]
        return any(sum(pts[c][j].co.y * pts[c][j - 1].co.y for c in range(4)) < 0.0
                   for j in range(1, len(pts[0])))
    if not any(bad(cs) for cs in curves.values()):
        return action
    new = action.copy()
    new.name = action.name + ".fix"
    new["milo_clip"] = 0           # keep it out of the clip lookup in execute()
    new["milo_flip"] = 1
    for cs in _quat_curves(new).values():
        pts = [fc.keyframe_points for fc in cs]
        for j in range(1, len(pts[0])):
            if sum(pts[c][j].co.y * pts[c][j - 1].co.y for c in range(4)) < 0.0:
                for c in range(4):
                    kp = pts[c][j]
                    kp.co.y, kp.handle_left.y, kp.handle_right.y = (
                        -kp.co.y, -kp.handle_left.y, -kp.handle_right.y)
        for fc in cs:
            fc.update()
    return new


def _read_clips_any(filepath):
    """The song's clips, read as DC3 (version 22) and failing that DC2 (version 20)."""
    from .dc_clip_importer import read_dc_clips
    for game in ('DC3', 'DC2'):
        try:
            _dn, clips, _groups, _failed = read_dc_clips(filepath, game)
            if clips:
                return clips
        except Exception:
            continue
    return []


def read_facing_samples(filepath, names):
    """{clip: ([(x, y, z)], [rotz])} - each clip's bone_facing channels, one entry per sample."""
    out = {}
    clips = _read_clips_any(filepath)
    for c in clips:
        if c['name'] not in names:
            continue
        const = dict(c['one']['samples'][0]) if c['one']['samples'] else {}
        pos, rot = [], []
        for smp in c['full']['samples']:
            ch = dict(const)
            ch.update(smp)
            fp, fr = ch.get('bone_facing.pos'), ch.get('bone_facing.rotz')
            pos.append(tuple(fp[1]) if fp else (0.0, 0.0, 0.0))
            rot.append(float(fr[1]) if fr else 0.0)
        out[c['name']] = (pos, rot)
    return out


def read_plant_samples(filepath, names):
    """{clip: [(left, right)]} - each clip's bone_footik channel, the game's per-foot "planted"
    signal (CharIKFoot reads it as mData.v[mDataIndex]: index 0 = left, 1 = right; 1.0 = planted)."""
    out = {}
    clips = _read_clips_any(filepath)
    for c in clips:
        if c['name'] not in names:
            continue
        const = dict(c['one']['samples'][0]) if c['one']['samples'] else {}
        vals = []
        for smp in c['full']['samples']:
            ch = dict(const)
            ch.update(smp)
            d = ch.get('bone_footik.pos')
            vals.append((float(d[1][0]), float(d[1][1])) if d else (0.0, 0.0))
        out[c['name']] = vals
    return out


def plant_channel(infos, plants, fpb, n_frames):
    """Per timeline frame (left, right) planted values, blended across overlapping clips with the
    same weights as the crossfade (the game blends the channel, then tests it against 0.98)."""
    out = [None] * n_frames
    for info in infos:
        vals = plants.get(info['clip'])
        if not vals:
            continue
        last = len(vals) - 1
        keys, sc = info['keys'], info['sc']
        times = [(b - keys[0][0]) * fpb for b, _f in keys] if keys else None
        s0, ov = info['s0'], info['ov_prev']
        for F in range(int(math.ceil(s0)), min(int(math.floor(s0 + info['length'])), n_frames - 1) + 1):
            rel = (F - s0) + info['skip']
            if times is None:
                k = rel / sc
            else:
                j = min(max(bisect.bisect_right(times, rel) - 1, 0), last - 1)
                span = times[j + 1] - times[j]
                k = j + ((rel - times[j]) / span if span > 1e-9 else 0.0)
            k = min(max(k, 0.0), float(last))
            i0 = int(k)
            i1 = min(i0 + 1, last)
            f = k - i0
            v = (vals[i0][0] + (vals[i1][0] - vals[i0][0]) * f,
                 vals[i0][1] + (vals[i1][1] - vals[i0][1]) * f)
            if ov > 0.0 and F < s0 + ov and out[F] is not None:
                u = (F - s0) / ov
                w = u * u * (3.0 - 2.0 * u)
                v = (out[F][0] * (1.0 - w) + v[0] * w, out[F][1] * (1.0 - w) + v[1] * w)
            out[F] = v
    return out


def facing_path(infos, samples, fpb, n_frames, fps=30.0, regulate=True):
    """The dancer's floor path, built the way the game builds it. CharClip::FacingSet turns each
    clip's bone_facing into a per-frame delta (the move since the previous frame, turned back by
    the previous heading, plus the change in heading); blended clips mix their deltas by weight;
    CharServoBone::MoveToDeltaFacing then adds each delta to the character's own transform.
    On top of that HamRegulate steers the dancer toward where the clips' absolute facing puts
    them relative to their waypoint (HamRegulate::Regulate, mode 0: error / 8 beats, capped at
    4 units a second) and Waypoint::Constrain keeps them inside a 36-unit box around the waypoint.
    Waypoint = the armature's own starting spot, so the path is relative to it. Not ported: the
    foot-planted gating (it needs the IK foot states). infos = one dict per placed strip: clip,
    s0, length, ov_prev (frames it blends in over the strip before it), skip, sc, keys (beat
    track or None). Returns [(x, y, z, heading)] for timeline frames 0..n_frames-1."""
    dpos = [(0.0, 0.0, 0.0)] * n_frames
    dang = [0.0] * n_frames
    apos = [None] * n_frames          # blended absolute facing (what regulation steers toward)
    aang = [0.0] * n_frames
    for info in infos:
        got = samples.get(info['clip'])
        if got is None:
            continue
        pos, rot = got
        last = len(pos) - 1
        keys, sc = info['keys'], info['sc']
        times = [(b - keys[0][0]) * fpb for b, _f in keys] if keys else None

        def kidx(rel):
            if times is None:
                k = rel / sc
            else:
                j = min(max(bisect.bisect_right(times, rel) - 1, 0), last - 1)
                span = times[j + 1] - times[j]
                k = j + ((rel - times[j]) / span if span > 1e-9 else 0.0)
            return min(max(k, 0.0), float(last))

        def at(k):
            i0 = int(k)
            i1 = min(i0 + 1, last)
            f = k - i0
            return (tuple(a + (b - a) * f for a, b in zip(pos[i0], pos[i1])),
                    rot[i0] + (rot[i1] - rot[i0]) * f)

        s0, ov = info['s0'], info['ov_prev']
        for F in range(int(math.ceil(s0)), min(int(math.floor(s0 + info['length'])), n_frames - 1) + 1):
            rel = (F - s0) + info['skip']
            (p1, a1), (p0, a0_) = at(kidx(rel)), at(kidx(rel - 1.0))
            c, s_ = math.cos(-a0_), math.sin(-a0_)
            dx, dy, dz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
            d = (dx * c - dy * s_, dx * s_ + dy * c, dz)
            da = (a1 - a0_ + math.pi) % (2.0 * math.pi) - math.pi      # LimitAng
            ap, aa = p1, a1
            if ov > 0.0 and F < s0 + ov:
                u = (F - s0) / ov
                w = u * u * (3.0 - 2.0 * u)
                q, qa = dpos[F], dang[F]
                d = tuple(q[i] * (1.0 - w) + d[i] * w for i in range(3))
                da = qa * (1.0 - w) + da * w
                if apos[F] is not None:
                    ap = tuple(apos[F][i] * (1.0 - w) + ap[i] * w for i in range(3))
                    aa = aang[F] * (1.0 - w) + aa * w
            dpos[F], dang[F], apos[F], aang[F] = d, da, ap, aa
    path, x, y, z, th = [], 0.0, 0.0, 0.0, 0.0
    held, held_a = (0.0, 0.0, 0.0), 0.0
    arrive, max_speed, box = 8.0, 4.0, 36.0 + 0.01      # HamRegulate / HamCharacter's waypoint
    beat_per_frame, sec_per_frame = 1.0 / fpb, 1.0 / fps
    for F in range(n_frames):
        d = dpos[F]
        c, s_ = math.cos(th), math.sin(th)
        x += d[0] * c - d[1] * s_
        y += d[0] * s_ + d[1] * c
        z += d[2]
        th += dang[F]
        if apos[F] is not None:
            held, held_a = apos[F], aang[F]
        if regulate and apos[F] is not None:
            rel = th - held_a
            c, s_ = math.cos(rel), math.sin(rel)
            ex = -(x - (held[0] * c - held[1] * s_))        # waypoint (0, 0) - facing.v
            ey = -(y - (held[0] * s_ + held[1] * c))
            er = -math.sin(rel)                              # waypoint heading 0
            k = min(1.0, max(0.0, beat_per_frame / arrive))
            ex, ey, er = ex * k, ey * k, er * k
            mag = math.hypot(ex, ey)
            cap = max_speed * sec_per_frame
            if mag > 0.0 and mag > cap:
                f = cap / mag
                ex, ey, er = ex * f, ey * f, er * f
            th += er
            x = min(max(x + ex, -box), box)
            y = min(max(y + ey, -box), box)
        path.append((x, y, z, th))
    return path


def floor_offsets(heights, floor, snap_range, smooth=4):
    """Floor lock (the same idea as the stitch app's): `heights[F]` is the height of the planted
    foot's ankle at timeline frame F, or None when no foot is planted (a step or a jump). Frames
    with a planted foot get the offset that puts that ankle at `floor`; the frames in between
    keep the offset of their neighbours (blended across the gap) so the dancer doesn't move up
    and down every time a foot lifts. The result is smoothed twice over +-`smooth` frames.
    Returns one offset per frame."""
    raw = [(floor - h) if h is not None and h < floor + snap_range else None for h in heights]
    known = [i for i, v in enumerate(raw) if v is not None]
    if not known:
        return [0.0] * len(raw)
    out, k = [0.0] * len(raw), 0
    for i in range(len(raw)):
        if raw[i] is not None:
            out[i] = raw[i]
            continue
        while k + 1 < len(known) and known[k + 1] < i:
            k += 1
        a = known[k] if known[k] < i else None
        b = next((x for x in known[k:] if x > i), None)
        if a is None:
            out[i] = raw[b]
        elif b is None:
            out[i] = raw[a]
        else:
            out[i] = raw[a] + (raw[b] - raw[a]) * (i - a) / (b - a)
    for _ in range(2):
        pad = [sum(out[max(0, i - smooth):i + smooth + 1]) / (min(len(out), i + smooth + 1) - max(0, i - smooth))
               for i in range(len(out))]
        out = pad
    return out


def align_skip(song_beat, start_beat, period):
    """ClipPlayer::ClipStart: a clip with beat-align play flags (period = flags >> 12 & 15) is
    entered `Mod(key beat - clip start beat, period)` beats into the clip, so its beats stay in
    step with the song's whole beats. Returns that offset in beats (0 <= offset < period)."""
    if math.fmod(song_beat, 1.0) != 0.0 and math.ceil(song_beat) - song_beat < 0.0001:
        song_beat = math.ceil(song_beat)
    return (song_beat - start_beat) % period


def match_quat_signs(action, prev, start, scale, sc=1.0, skip=0.0):
    """Custom DLC: Blender's NLA blends quaternion components linearly, so when the outgoing
    clip stores a bone's rotation as q and the incoming clip as -q (same rotation) the
    crossfade swings through zero and the bone whips around. Returns `action`, or a copy
    ('<clip>.flip') with those bones' quaternion curves negated to agree with `prev`'s.
    prev = (outgoing action, its start frame, its crossfade overlap, its strip scale, its
    first action frame offset); sc / skip = the incoming strip's scale and offset."""
    p_action, p_start, p_ov, p_sc, p_skip = prev
    theirs = _quat_curves(p_action)
    mine = _quat_curves(action)
    pa0, a0 = p_action.frame_range[0], action.frame_range[0]
    flip = []
    for path, cs in mine.items():
        ts = theirs.get(path)
        if ts is None:
            continue
        d = 0.0
        # ponytail: sampled at 9 points, weighted toward the middle of the overlap where a
        # sign mismatch hurts most (the blend there is smallest), not every frame
        for n in range(1, 10):
            u = n / 10.0
            ta = pa0 + p_skip + ((start - p_start) * scale + p_ov * u) / p_sc
            tb = a0 + skip + p_ov * u / sc
            d += u * (1.0 - u) * sum(x.evaluate(ta) * y.evaluate(tb) for x, y in zip(ts, cs))
        if d < 0.0:
            flip.append(path)
    if not flip:
        return action
    new = action.copy()
    new.name = action.name + ".flip"
    new["milo_clip"] = 0           # keep it out of the clip lookup in execute()
    new["milo_flip"] = 1
    for path, cs in _quat_curves(new).items():
        if path in flip:
            for fc in cs:
                for kp in fc.keyframe_points:
                    kp.co.y = -kp.co.y
                    kp.handle_left.y = -kp.handle_left.y
                    kp.handle_right.y = -kp.handle_right.y
                fc.update()
    return new


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
    dlc_bpm: bpy.props.FloatProperty(
        name="DLC Song BPM",
        description="Custom DLC only. The song's tempo; the game plays clips in song beats, so "
                     "this sets how many frames one beat lasts. 0 reads it from the song's "
                     ".mid (<song>/<song>.mid next to the gen folder); type a value to "
                     "override that",
        default=0.0, min=0.0, max=300.0,
    )
    blend_beats: bpy.props.FloatProperty(
        name="Blend Width (beats)",
        description="How many song beats each clip takes to blend into the next. DC3's dancer "
                     "driver (HamDriver::LayerClip::Eval) fades every clip in over exactly 1 "
                     "beat along the EaseSigmoid curve (3t^2 - 2t^3), so 1 is the game's value. "
                     "Falls back to Crossfade Length when the song's tempo can't be found",
        default=1.0, min=0.0, max=4.0,
    )
    dlc_beat_align: BoolProperty(
        name="Align Clip Starts To Beat",
        description="Custom DLC only. The game enters each clip a fraction of a beat in so the "
                     "clip's beats line up with the song's whole beats (ClipPlayer::ClipStart, "
                     "for clips with beat-align play flags). Off: every clip starts at its first "
                     "frame",
        default=True,
    )
    dlc_facing: BoolProperty(
        name="Move Dancer By Facing Deltas",
        description="Custom DLC only. The game never uses a clip's own floor position: it adds up "
                     "each clip's per-frame movement and turning (bone_facing deltas) into the "
                     "dancer's position, so moves chain without jumping. This builds that path "
                     "and keys it on the armature object in an extra NLA track. Import the "
                     "charclips with 'Apply Root Motion (bone_facing)' UNTICKED first, or the "
                     "travel is applied twice",
        default=False,
    )
    dlc_regulate: BoolProperty(
        name="Keep Dancer Near Their Spot",
        description="Needs Move Dancer By Facing Deltas. The game also steers the dancer toward "
                     "where the clips' own floor positions put them (HamRegulate) and keeps them "
                     "inside a 36 unit box around their starting spot (Waypoint). Without it the "
                     "small errors in the movement add up and the dancer drifts away",
        default=True,
    )
    lock_to_ground: BoolProperty(
        name="Lock To Ground",
        description="Keeps the dancer's feet on the floor. Clips can stand at slightly different "
                     "heights, so the dancer floats or sinks from clip to clip. On every frame "
                     "this measures the foot the clip marks as planted (its bone_footik channel; "
                     "the lowest ankle when a clip has none) and moves the dancer up or down so "
                     "that foot sits at its rest height, keyed on the armature object in a "
                     "separate NLA track. Jumps are left alone. Steps through the whole routine, "
                     "so it takes a minute or so on a full song",
        default=True,
    )
    dlc_floor_range: bpy.props.FloatProperty(
        name="Ground Snap Range",
        description="Safety limit: ignore a planted foot whose ankle is further than this many "
                     "units above the floor height. Frames with no planted foot (jumps) are "
                     "always left alone",
        default=15.0, min=1.0, max=60.0,
    )
    swap_tracks: BoolProperty(
        name="Swap A / B Tracks",
        description="Put the first clip on track B instead of A, so the clips alternate B, A, "
                     "B, A... The upper track blends over the lower one, so this flips which "
                     "clips sit on top. Off: original layout",
        default=False,
    )
    hold_gaps: BoolProperty(
        name="Hold Between Clips",
        description="Without crossfading: hold each clip's last pose until the next one "
                     "starts. (Crossfading always holds through the gap.) A move clip covers "
                     "about 3 of its 4 beats; in the game the rest is a blend into the next "
                     "move. Slots with no clip are left empty either way",
        default=True,
    )

    custom_dlc: BoolProperty(
        name="Custom DLC",
        description="DC3 custom songs. Plays each clip in song beats by its beat track (the game "
                     "does; custom clips aren't always authored at the song's tempo), aligns clip "
                     "starts to whole beats, and enables the options below. Official songs don't "
                     "need it",
        default=False,
    )

    def _key_facing(self, arm, scene, ad, base_name, infos, fpb, last_frame):
        """Keys the integrated facing path on the armature object (location and Z rotation)."""
        from mathutils import Matrix, Vector
        from .anim_importer import _CurveCache, _get_channelbag
        n_frames = int(math.ceil(last_frame)) + 2
        if self.custom_dlc and self.dlc_facing:
            samples = read_facing_samples(self.filepath, {i['clip'] for i in infos})
            path = facing_path(infos, samples, fpb, n_frames,
                               scene.render.fps / scene.render.fps_base, self.dlc_regulate)
        else:
            path = [(0.0, 0.0, 0.0, 0.0)] * n_frames
        if self.lock_to_ground:
            offs = self._floor_lock(arm, scene, n_frames, infos, fpb)
            if offs:
                path = [(x, y, z + offs[F], th) for F, (x, y, z, th) in enumerate(path)]
        if arm.rotation_mode in ('QUATERNION', 'AXIS_ANGLE'):
            arm.rotation_mode = 'XYZ'
        s = abs(arm.scale.x) or 1.0
        base_loc, base_rot = arm.location.copy(), arm.rotation_euler.z
        rz = Matrix.Rotation(base_rot, 3, 'Z')
        act = bpy.data.actions.new(f"{base_name} facing")
        slot = act.slots.new(id_type='OBJECT', name=arm.name)
        cache = _CurveCache(_get_channelbag(act, slot))
        for F in range(1, n_frames):
            x, y, z, th = path[F]
            loc = base_loc + rz @ Vector((x * s, y * s, z * s))
            for i in range(3):
                cache.key('location', i, 'Facing', F, loc[i], interp='LINEAR')
            cache.key('rotation_euler', 2, 'Facing', F, base_rot + th, interp='LINEAR')
        cache.finish()
        trk = ad.nla_tracks.new()
        trk.name = base_name + " facing"
        st = trk.strips.new(act.name, 1, act)
        try:
            st.action_slot = act.slots[0]
        except (AttributeError, IndexError, TypeError):
            pass
        st.extrapolation = 'HOLD'
        end = path[-1]
        _log(f"  object path keyed on '{arm.name}' (NLA track '{trk.name}'): ends "
             f"{math.hypot(end[0], end[1]):.0f} units from the start, farthest point "
             f"{max(math.hypot(p[0], p[1]) for p in path):.0f}")

    def _floor_lock(self, arm, scene, n_frames, infos=None, fpb=1.0):
        """Steps through the routine and returns the height offset (armature units) per frame that
        puts the planted foot on the floor, or [] if the ankles can't be found. A foot counts as
        planted when the clip's bone_footik channel says so (>= 0.98, as CharIKFoot tests it);
        frames with no planted foot (a jump) are left alone. Without that channel it falls back
        to the lowest ankle."""
        from .viseme_importer import build_bone_lookup   # bones can carry a .mesh / .trans suffix
        lookup = build_bone_lookup(arm)
        def find(side):                    # 'bone_L-ankle' first, else any name ending in L-ankle
            if f'bone_{side}-ankle' in lookup:
                return [lookup[f'bone_{side}-ankle']]
            return [lookup[k] for k in lookup if k.lower().endswith(f'{side.lower()}-ankle')][:1]
        left, right = find('L'), find('R')
        ankles = left + right
        if not ankles:
            _log("  floor lock: no ankle bones found (looked for L-ankle / R-ankle, ignoring a "
                 f".mesh/.trans suffix). The armature has {len(arm.pose.bones)} bones, e.g. "
                 f"{', '.join(b.name for b in list(arm.pose.bones)[:6])}")
            return []
        plant = None
        if infos and left and right:
            try:
                plant = plant_channel(infos, read_plant_samples(
                    self.filepath, {i['clip'] for i in infos}), fpb, n_frames)
            except Exception as e:
                _log(f"  floor lock: could not read the foot-plant channel ({e}); using the lowest ankle")
        floor = min(arm.data.bones[n].head_local.z for n in ankles)     # rest-pose standing height
        children = [o for o in arm.children_recursive if not o.hide_viewport]
        wm, keep = bpy.context.window_manager, scene.frame_current
        heights = [None] * n_frames
        wm.progress_begin(0, n_frames)
        for o in children:                # meshes aren't needed to read bone heights
            o.hide_viewport = True
        try:
            for F in range(1, n_frames):
                scene.frame_set(F)
                ev = arm.evaluated_get(bpy.context.evaluated_depsgraph_get())
                zl = min((ev.pose.bones[n].head.z for n in left), default=None)
                zr = min((ev.pose.bones[n].head.z for n in right), default=None)
                if plant is not None and plant[F] is not None:
                    cand = [z for z, v in ((zl, plant[F][0]), (zr, plant[F][1])) if v >= 0.98]
                    heights[F] = min(cand, default=None)
                elif plant is None:
                    heights[F] = min(ev.pose.bones[n].head.z for n in ankles)
                if F % 100 == 0:
                    wm.progress_update(F)
        finally:
            for o in children:
                o.hide_viewport = False
            scene.frame_set(keep)
            wm.progress_end()
        offs = floor_offsets(heights, floor, self.dlc_floor_range)
        moved = [abs(o) for o in offs if abs(o) > 0.01]
        planted = sum(1 for h in heights if h is not None)
        _log(f"  floor lock: rest ankle height {floor:.2f}, "
             f"{'planted feet on ' + str(planted) + ' frames (foot-plant data)' if plant is not None else 'lowest ankle'}, "
             f"corrected {len(moved)} of {n_frames} frames (largest {max(moved, default=0.0):.1f} units)")
        return offs

    def draw(self, context):
        layout = self.layout
        # the original options, in their original order
        layout.prop(self, "difficulty")
        layout.prop(self, "crossfade")
        col = layout.column()
        col.enabled = self.crossfade
        col.prop(self, "blend_beats")
        col.prop(self, "crossfade_frames")
        layout.prop(self, "hold_gaps")
        layout.prop(self, "lock_to_ground")
        row = layout.row()
        row.enabled = self.lock_to_ground
        row.prop(self, "dlc_floor_range")
        # custom DLC lives in its own section
        layout.separator()
        box = layout.box()
        box.label(text="Custom DLC (DC3)")
        box.prop(self, "custom_dlc")
        col = box.column()
        col.enabled = self.custom_dlc
        col.prop(self, "dlc_bpm")
        col.prop(self, "dlc_beat_align")
        col.prop(self, "dlc_facing")
        sub = col.column()
        sub.enabled = self.dlc_facing
        sub.prop(self, "dlc_regulate")
        col.prop(self, "swap_tracks")

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
                    if t.name in (base_name, base_name + " A", base_name + " B",
                                  base_name + " facing")]:
            ad.nla_tracks.remove(old)
        # Derived Actions (.fix / .flip / .beat) left over from an earlier layout.
        for a in [a for a in bpy.data.actions if a.get("milo_flip") and a.users == 0]:
            bpy.data.actions.remove(a)
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
        prev = None                       # (action, start, overlap, scale, skip) of the strip just placed
        infos = []                        # per placed strip, for the facing path
        fixed = {}                        # clip name -> sign-continuous Action (Custom DLC)
        beats = {}                        # clip name -> (action to use, 'beat' | 'real')
        # The song's tempo - needed for the one-beat crossfade on every song, and for beat
        # timing on custom ones. Typed BPM (custom DLC) > the song's .mid > the clips' own
        # beat tracks, which are in the song's beats at its tempo.
        tracks_, pflags = read_beat_tracks(self.filepath, {
            c: int(round(a.frame_range[1] - a.frame_range[0])) + 1
            for c, a in actions.items()})
        scene_fps = scene.render.fps / scene.render.fps_base
        bpm, note = (self.dlc_bpm, "typed in") if self.custom_dlc and self.dlc_bpm > 0.0 \
            else find_song_bpm(self.filepath)
        if bpm is None:
            bpm, note = bpm_from_beat_tracks(tracks_)
        fpb = scene_fps * 60.0 / bpm if bpm else None
        if fpb:
            _log(f"  tempo: {bpm:g} BPM ({note}) - {fpb:.3f} frames per beat")
        else:
            _log(f"  tempo: not found ({note}); crossfades use Crossfade Length instead of "
                 f"{self.blend_beats:g} beat(s)")
            if self.custom_dlc:
                bpm, fpb = 130.0, scene_fps * 60.0 / 130.0
                self.report({'WARNING'}, f"Custom DLC: no tempo found; using 130 BPM. "
                                         f"Set DLC Song BPM to override.")
        if self.custom_dlc:
            for c, a in actions.items():
                if pflags.get(c, 0) & 0x200:          # kPlayRealTime: plays in seconds
                    beats[c] = (a, 'real')
                elif c in tracks_:
                    beats[c] = (beat_warp(a, tracks_[c], fpb), 'beat')
                else:
                    beats[c] = (a, 'real')       # no beat track: plays at its own speed
            modes = [m for _a, m in beats.values()]
            _log(f"  Custom DLC: {modes.count('beat')} clip(s) timed by their beat track "
                 f"({fpb:.3f} frames per beat at {bpm:g} BPM), {modes.count('real')} at their "
                 f"own speed")
        for i, (start, end, clip, note) in enumerate(slots):
            if note:
                if clip is None:
                    empty.append(f"frame {start:.1f}: {note}")
                else:
                    _log(f"  frame {start:.1f}: {note}")
            if clip is None:
                prev = None
                continue
            action, mode = beats.get(clip, (actions[clip], 'real'))
            a0, a1 = action.frame_range
            s0 = 1.0 + start * scale
            nxt_filled = i + 1 < len(slots) and filled[i + 1]
            sc = 1.0                          # strip.scale: timeline frames per action frame
            skip = 0.0                        # action frames skipped at the clip's start
            if self.custom_dlc and self.dlc_beat_align and mode == 'beat':
                align = (pflags.get(clip, 0) >> 12) & 0xF
                if align:
                    skip = align_skip(start * scale / fpb, tracks_[clip][0][0], float(align)) * fpb
            clip_len = (a1 - a0) * sc - skip  # the clip's length on the timeline
            # Quaternion sign fixes, on every song: a bone stored as q in one clip and -q in
            # the next (the same rotation) makes the NLA's straight blend swing through zero.
            if clip not in fixed:
                fixed[clip] = unroll_quats(action)
            action = fixed[clip]
            if self.crossfade and prev is not None and prev[2] > 0.0:
                action = match_quat_signs(action, prev, start, scale, sc, skip)
            overlap = 0.0
            if end is None:
                length = a1 - a0
            elif self.crossfade and nxt_filled:
                # Run past the next key by the crossfade, holding the last pose if the clip
                # ends first (an F-curve holds its last value beyond its final key). Capped
                # below the next slot so clip i never reaches clip i+2 on the same track.
                nxt_end = slots[i + 2][0] if i + 2 < len(slots) else None
                room = ((nxt_end - end) * scale * 0.9) if nxt_end is not None else fade
                width = self.blend_beats * fpb if fpb else fade
                overlap = max(0.0, min(width, room))
                length = (end - start) * scale + overlap
            else:
                length = min(clip_len, (end - start) * scale - 1e-3)
            lane = (placed + (1 if self.swap_tracks and len(tracks) > 1 else 0)) % len(tracks)
            track = tracks[lane]
            strip = track.strips.new(f"{clip} @{start:.0f}", int(math.ceil(s0)), action)
            try:
                strip.action_slot = action.slots[0]
            except (AttributeError, IndexError, TypeError):
                pass
            strip.action_frame_start = a0 + skip
            strip.action_frame_end = a0 + skip + max(length / sc, 1e-3)
            if sc != 1.0:
                strip.scale = sc              # strip still spans `length` timeline frames
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
            infos.append(dict(clip=clip, s0=s0, length=length, skip=skip, sc=sc,
                              ov_prev=(strips[-1][2] if strips and strips[-1][3] else 0.0),
                              keys=tracks_.get(clip) if mode == 'beat' else None))
            strips.append((strip, lane, overlap, nxt_filled))
            placed += 1
            prev = (action, start, overlap, sc, skip)
            last_frame = max(last_frame, s0 + length)

        # Crossfades: whichever of the two strips is on the upper track ramps its influence -
        # the incoming one blends in, or the outgoing one blends out over the one beneath.
        fades = 0
        if self.crossfade:
            ramps = {}                    # strip -> [(frame, influence)]
            for (sa, ta, ov, nxt), (sb, tb, _o, _n) in zip(strips, strips[1:]):
                if not nxt or ov <= 0.0:
                    continue
                # The game fades the new clip in with EaseSigmoid and gives the old clip the
                # rest (HamDriver::LayerClip::Eval), so the upper strip's influence follows
                # that curve. EaseSigmoid(t) = 3t^2 - 2t^3 (math/Easing.h), i.e. smoothstep.
                up, rising = (sb, True) if tb == 1 else (sa, False)
                t0 = sb.frame_start
                pts = ramps.setdefault(up, [])
                for k in range(9):
                    u = k / 8.0
                    w = u * u * (3.0 - 2.0 * u)
                    pts.append((t0 + u * ov, w if rising else 1.0 - w))
                fades += 1
            for st, pts in ramps.items():
                try:
                    st.use_animated_influence = True
                    for f, w in sorted(pts):
                        st.influence = w
                        st.keyframe_insert("influence", frame=f)
                    for fc in st.fcurves:
                        for kp in fc.keyframe_points:
                            kp.interpolation = 'LINEAR'
                except Exception as e:   # fall back to the plain linear NLA blend
                    _log(f"  animated influence failed on {st.name} ({e}); using linear blend")
                    st.use_animated_influence = False
                    ov_ = max(f for f, _w in pts) - min(f for f, _w in pts)
                    if pts[0][1] < pts[-1][1]:
                        st.blend_in = ov_
                    else:
                        st.blend_out = ov_
            width_note = (f"{self.blend_beats:g} beat(s)" if fpb
                          else f"{self.crossfade_frames:g} game frame(s)")
            _log(f"  crossfades: {fades}, {width_note} each along the game's EaseSigmoid "
                 f"curve (shorter where the next slot is too short)")

        if infos and (self.lock_to_ground or (self.custom_dlc and self.dlc_facing)):
            self._key_facing(arm, scene, ad, base_name, infos, fpb or 1.0, last_frame)

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
# DC visemes (imported with Milo Viseme Import > Dance Central, Import Base Viseme on - DC's
# visemes are offsets from Base, which isn't the rest pose) from them unchanged.

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
        _log("  Import the character's DC visemes (Milo Viseme Import > Dance Central, with "
             "Import Base Viseme on) onto the same armature, then Bake Lipsync Preview.")
        self.report({'INFO'}, f"Imported {len(names)} DC face track(s) as '{action.name}'")
        return {'FINISHED'}
