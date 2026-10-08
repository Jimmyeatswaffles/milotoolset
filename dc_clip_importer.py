"""
Dance Central 1, 2 and 3, and Rock Band 2, CharClip import.

Imports every clip in a Dance Central clip milo (e.g. clips.milo_xbox: intros, idles, win
poses) as its own Action on the active armature, keyed in real time - no bake step. It sits
apart from the GDRB/TBRB animation code because DC1 is a different engine generation: its
clips are RB3-era CharClips rather than revision-25 CharClipSamples.

===========================================================================================
FORMAT - verified on a retail clips.milo_xbox
===========================================================================================
A big-endian CharClipSet at milo revision 28 - the same revision as Rock Band 3 - holding
CharClip entries at version 19, exactly like RB3's viseme clips, plus CharClipGroups. So it's
read with the RB3 directory walker (viseme_importer.find_viseme_clips) and the same sample
locator; all 17 retail clips decode.

Timing comes from the clip itself. CharClip version 19 stores mFramesPerSec (CharClip::Load
in the RB3 decompilation) rather than beats and a tempo, and the sample blocks carry no time
table, so sample N plays at N / mFramesPerSec seconds. Every retail clip is at 30 fps
(realtime_idle_01: 304 samples, 10.1 s). No song tempo is involved.

The samples are ABSOLUTE local transforms, not offsets: bone_pelvis.pos sits 38.6 units up,
in line with GDRB's 37. Compression 1 - positions as floats, rotations as int16 - so the
shared decoder applies; quaternions come out unit length and hinge joints land in natural
ranges (elbow -100 to -14 deg, knee -33 to -18, finger joints 0 to 27).

No clip animates a twist bone (59 distinct channels across all 17), so as in GDRB the arm
twist bones are solved from the arm pose (twist_solvers), here with the standard settings.

===========================================================================================
ROCK BAND 2 - verified on the retail medium_dramatic.milo_ps3
===========================================================================================
A revision-25 milo, the TBRB/GDRB generation, with 72 CharClipSamples at version 14 wrapping
a version-9 CharClip (type 'musician'). The sample blocks are GDRB's own - version 16,
compression 1, a frame table - and only the clip header differs: a data tree pairing a facial
expression with the clip, then start beat, end beat and tempo. Clips start part-way in, so
the length is (end - start) / tempo, which matches each frame table exactly.

Every clip carries bone_facing.pos and bone_facing.rotz, and unlike Dance Central's they
TURN - all 72 rotate their facing, some through nearly 190 degrees - so the turning half of
the root motion is exercised here for the first time on real data.

===========================================================================================
DANCE CENTRAL 2 - verified on the retail forgetyou.milo_xbox (DLC)
===========================================================================================
The same layout as DC3 below - revision 32, a nested 'clips' directory, a data tree in each
clip header - with CharClip version 20 instead of 22, which is all that kept the DC3 reader
from accepting it. All 31 clips read through the DC3 path: 30 fps throughout, all 159,106
quaternions unit length, identical value for value to an independent parse anchored on the
first channel name.

===========================================================================================
DANCE CENTRAL 3 - verified on the retail gangnamstyle.milo_xbox (DLC)
===========================================================================================
Milo revision 32, compressed, with the clips in a nested 'clips' directory: 38 CharClips at
version 22 - routine sections (expert_02 ... expert_06), move-to-move clips and rest clips.
The sample blocks are the same version-16 layout as DC1's. What differs is the header: an
Hmx::Object data tree (clip_skeleton 'devin', clip_skeleton_index) sits before the frame
rate, and a list of related clips sits before the samples. So the tree is walked node by
node to reach the frame rate, and the samples are located by structure. All 38 decode, at
30 fps, identical value for value to an independent parse anchored on the first channel name.

===========================================================================================
ARM TWIST
===========================================================================================
Dance Central rigs (DC1's male/female_skeleton_shared, 93 bones) have the same forearm twist
bones as GDRB - foreTwist1/2, parented the way CharForeTwist expects - but no upperTwist
bones: a five-bone shoulderTwist chain sits there instead, plus thighTwist01. The forearm
solver's standard offsets (RB3's documented 0 left, 180 right) reproduce both skeletons'
stored forearm twist rest pose to within 0.05 degrees, so they're used as they stand, with
no GDRB-style correction. The shoulder chain is most likely driven by CharBlendBone (blends
target bones between two sources by per-bone weights), whose settings live in each dancer's
character milo, not the skeleton. With that milo set (Set Character Milo), they're read and
solved too - see twist_solvers.BlendBoneSolver.

===========================================================================================
ROTATION CONVENTION - measured on the armature at import time
===========================================================================================
Whether stored rotations need conjugating can't be settled from the file alone, and the
evidence so far points both ways. By analogy, RB3's version-19 viseme clips - the same format
as these - measured as conjugated against an RB3 rig (0.00 against 156 degrees). But
measured directly, these DC1 clips favour as-stored: on a stand-in Harmonix rig (Billie Joe's
GDRB skeleton, since no DC1 skeleton was available), their first frames sit a median 18.4
degrees from rest as stored against 41.3 conjugated, over 476 rotations. Both skeleton
parsers read bone matrices identically and share one armature builder, so it isn't an
importer difference.

Since confirmed on a real Dance Central rig: on DC1's male_skeleton_shared, as-stored wins
by 19.0 against 43.6 degrees for the DC1 clips and 19.4 against 46.5 for the DC3 ones.

So the importer measures on the real armature every time: each clip's first frame is compared
with the rig's rest pose under both conventions, and the closer median wins. A mid-dance pose
stays near rest for most bones under the right convention and is scrambled under the wrong
one. If the two are within a few degrees, the tiebreak is as-stored - the only measurement
made on DC1's own data. Both numbers are logged, and the choice can be overridden.
"""

import math
import struct

import bpy
from bpy.props import StringProperty, BoolProperty, EnumProperty
from bpy_extras.io_utils import ImportHelper
from mathutils import Quaternion

from .io import read_milo_container_body, _mw_collect_directory_meta
from .utilities import _log
from .viseme_importer import (
    find_viseme_clips, _locate_bone_samples, _channel_stem, _Reader,
    _read_char_bones_samples,
)
from .anim_importer import (
    build_bone_lookup, build_twist_rigs, write_pose_keys, write_twist_keys,
    _interp_channels, _CurveCache, _get_channelbag, CHARACTER_PATH_PROP,
    _decode_channel, _engine_rows_from_quat, _engine_rows_from_hinge,
    _quat_from_engine_rows, pose_basis_for_bone,
)
from .twist_solvers import read_blend_bones, BlendBoneSolver


# Offset of mFramesPerSec in a version-19 CharClip: version, then the Hmx::Object header
# (revision, empty type, empty note, no tree) and the clip's type symbol. Read by walking
# that header rather than at a fixed offset, since the type name's length varies.
_MIN_CONVENTION_GAP_DEG = 5.0


class DCClipImportError(Exception):
    pass


# DTB node types (system/obj/Data.h in the DC3 decompilation) by how their value is stored.
_DTB_WORD = {0, 1, 6, 8, 9, 36}                     # int, float, unhandled, else, endif, autorun
_DTB_TEXT = {2, 3, 4, 5, 7, 18, 32, 33, 34, 35, 37}  # var, func, object, symbol, ifdef, string,
                                                    # define, include, merge, ifndef, undef
_DTB_TREE = {16, 17, 19}                            # array, command, property
_DTB_GLOB = 20


def _skip_dtb_array(c, p, depth=0, header=6):
    """Steps over one binary DataArray and returns the offset just past it: a 16-bit node
    count, a 32-bit line number, optionally a 16-bit field (`header` 6 or 8 bytes in all),
    then the nodes.

    DC3's DataArray::Load reads the 16-bit field, making 8 bytes, but the trees embedded in
    the retail gangnamstyle clips have only 6 - the first node's type follows the line
    number directly. Callers try both and keep the one that parses into a sane header."""
    if depth > 64:
        raise DCClipImportError("data tree nested implausibly deep")
    count = struct.unpack_from('>h', c, p)[0]
    if not 0 <= count <= 4096:
        raise DCClipImportError(f"implausible data tree node count {count}")
    p += header
    for _ in range(count):
        t = struct.unpack_from('>I', c, p)[0]; p += 4
        if t in _DTB_WORD:
            p += 4
        elif t in _DTB_TEXT:
            n = struct.unpack_from('>I', c, p)[0]
            if n > 65536:
                raise DCClipImportError("implausible string in data tree")
            p += 4 + n
        elif t in _DTB_TREE:
            p = _skip_dtb_array(c, p, depth + 1, header)
        elif t == _DTB_GLOB:
            p += 4 + abs(struct.unpack_from('>i', c, p)[0])
        else:
            raise DCClipImportError(f"unknown data tree node type {t}")
        if p > len(c):
            raise DCClipImportError("data tree runs past the end of the clip")
    return p


def _clip_fps(chunk, versions=(19, 22)):
    """(mFramesPerSec, offset just past it) from a CharClip.

    The Hmx::Object header is: revision, type symbol, has-tree flag, the data tree if
    present, then the note. DC3's version-22 clips carry a tree there (clip_skeleton and
    clip_skeleton_index), which pushes the frame rate from byte 21 to byte 96 in the retail
    gangnamstyle clips; DC1's carry none. The tree is walked node by node, and a walk is
    only accepted if it lands on a valid note followed by a plausible frame rate. An earlier version of this reader read the note
    before the tree flag - harmless for DC1, where both are zero bytes, but wrong."""
    ver = struct.unpack_from('>I', chunk, 0)[0]
    if ver not in versions:
        raise DCClipImportError(f"CharClip version {ver}; expected {versions}")
    p = 4
    meta_rev = struct.unpack_from('>I', chunk, p)[0]; p += 4
    n = struct.unpack_from('>I', chunk, p)[0]; p += 4 + n          # type symbol
    has_tree = chunk[p]; p += 1
    starts = [p]
    if has_tree:
        starts = []
        for header in (6, 8):           # see _skip_dtb_array
            try:
                starts.append(_skip_dtb_array(chunk, p, header=header))
            except (DCClipImportError, struct.error, IndexError):
                pass
    for q in starts:
        try:
            if meta_rev > 0:
                n = struct.unpack_from('>I', chunk, q)[0]
                if n > 4096:
                    continue
                q += 4 + n                                          # note symbol
            fps = struct.unpack_from('>f', chunk, q)[0]
        except struct.error:
            continue
        if 1.0 <= fps <= 240.0:
            return fps, q + 4
    raise DCClipImportError("couldn't read past this clip's header to its frame rate")


def _scan_samples(chunk, start):
    """Finds a clip's two sample blocks by structure rather than position: a version-16
    block that parses, followed by a second at the same version, every quaternion unit
    length. DC3 clips put a transition list and other fields between the frame rate and
    the samples, so their offset varies clip to clip."""
    sig = struct.pack('>I', 16)
    p = start
    while True:
        i = chunk.find(sig, p)
        if i < 0:
            raise DCClipImportError("couldn't locate this clip's sample data")
        p = i + 1
        try:
            r = _Reader(chunk, i)
            full = _read_char_bones_samples(r)
            one = _read_char_bones_samples(r)
        except Exception:
            continue
        if full['version'] != one['version'] or not full['bones']:
            continue
        quats = [v for b in (full, one) for s in b['samples'] for (k, v) in s.values()
                 if k == 'quat']
        if all(abs(math.sqrt(sum(x * x for x in q)) - 1.0) < 0.01 for q in quats):
            return full, one


# CharClip versions read with the DC3-style header walk and structural sample scan.
_HEADER_TREE_VERSIONS = {'DC2': (20,), 'DC3': (22,)}


# ---------------------------------------------------------------------------------------
# Rock Band 2
# ---------------------------------------------------------------------------------------

RB2_SAMPLES_REV = 14
TABLE_RATE = 30.0       # frame-table entries per second of clip time (as in GDRB)


def _rb2_clip_header(c):
    """(start_beat, end_beat, beats_per_sec) from an RB2 CharClipSamples (version 14
    wrapping a version-9 CharClip). After the object header - revision, type ('musician'),
    has-tree flag, the data tree if present, note - come the start beat, end beat and
    tempo. The tree pairs a facial expression with the clip (viseme_group /
    exp_rocker_teethgrit_happy) and is walked node by node; both tree-header sizes are
    tried and a walk is only accepted if it lands on a sane tempo."""
    if struct.unpack_from('>I', c, 0)[0] != RB2_SAMPLES_REV:
        raise DCClipImportError("not an RB2 CharClipSamples (version 14)")
    meta_rev = struct.unpack_from('>I', c, 8)[0]
    p = 12
    n = struct.unpack_from('>I', c, p)[0]; p += 4 + n            # type symbol
    has_tree = c[p]; p += 1
    starts = [p]
    if has_tree:
        starts = []
        for header in (6, 8):
            try:
                starts.append(_skip_dtb_array(c, p, header=header))
            except (DCClipImportError, struct.error, IndexError):
                pass
    for q in starts:
        try:
            if meta_rev > 0:
                n = struct.unpack_from('>I', c, q)[0]
                if n > 4096:
                    continue
                q += 4 + n                                         # note symbol
            start, end, bps = struct.unpack_from('>3f', c, q)
        except struct.error:
            continue
        if 0.05 <= bps <= 20.0 and end >= start and abs(start) < 1e6:
            return start, end, bps
    raise DCClipImportError("couldn't read this RB2 clip's header")


def read_rb2_clips(filepath):
    """Rock Band 2 clip milo -> same tuple as read_dc_clips. RB2 is a revision-25 milo
    (the TBRB/GDRB generation) holding CharClipSamples whose sample blocks are GDRB's own
    layout (version 16, a frame table); only the clip header differs. Verified on the retail
    medium_dramatic.milo_ps3: all 72 clips."""
    from .viseme_importer import find_viseme_clips_rev25
    from .anim_importer import _locate_bone_samples_wide
    from .mesh_importer import _read_dir_entries, _entry_spans
    dir_name, raw = find_viseme_clips_rev25(filepath)
    clips, failed = [], []
    for name, chunk in raw.items():
        try:
            start, end, bps = _rb2_clip_header(chunk)
            _off, full, one, _end = _locate_bone_samples_wide(chunk)
        except Exception as e:
            failed.append((name, str(e)))
            continue
        n = max(full['num_samples'], 1 if one['samples'] else 0)
        if n == 0:
            failed.append((name, "no samples"))
            continue
        # Length comes from the beat range, not the end beat alone: RB2 clips start part-way
        # in (stand_idle_ext_c_med_05: beats 2.66 to 15.46 at 2.132/s = 6.0 s), and the
        # frame table - 30 entries per second - has exactly 6.0 x 30 + 1 entries.
        seconds = (end - start) / bps
        clips.append(dict(name=name, fps=TABLE_RATE, full=full, one=one, num_samples=n,
                          seconds=seconds, table=list(full['frame_times'])))
    with open(filepath, 'rb') as f:
        body = read_milo_container_body(f.read())
    _rev, _dt, _dn, entries = _read_dir_entries(body)
    spans = _entry_spans(body, len(entries))
    groups = {}
    for (etype, ename), (s, e) in zip(entries, spans):
        if etype != 'CharClipGroup':
            continue
        data = body[s:e]
        for c in clips:
            tag = struct.pack('>I', len(c['name'])) + c['name'].encode('latin-1')
            if tag in data:
                groups.setdefault(c['name'], []).append(ename)
    return dir_name, clips, groups, failed


def read_dc_clips(filepath, game='DC1'):
    """Returns (dir_name, [clip dicts], {clip: [groups]}, [(name, reason)] failures).
    `game` is 'DC1' (version-19 clips), 'DC2' (version 20) or 'DC3' (version 22)."""
    if game == 'RB2':
        return read_rb2_clips(filepath)
    dir_name, raw = find_viseme_clips(filepath)
    clips, failed = [], []
    for name, chunk in raw.items():
        try:
            if game in _HEADER_TREE_VERSIONS:
                # DC2 and DC3 share the layout: a data tree in the header, and the samples
                # behind a list of related clips - only the CharClip version differs.
                fps, after = _clip_fps(chunk, versions=_HEADER_TREE_VERSIONS[game])
                full, one = _scan_samples(chunk, after)
            else:
                fps, _after = _clip_fps(chunk, versions=(19,))
                _off, full, one, _end = _locate_bone_samples(chunk)
        except Exception as e:
            failed.append((name, str(e)))
            continue
        n = max(full['num_samples'], 1 if one['samples'] else 0)
        if n == 0:
            failed.append((name, "no samples"))
            continue
        clips.append(dict(name=name, fps=fps, full=full, one=one, num_samples=n,
                          seconds=(n - 1) / fps))
    # Group membership: a CharClipGroup lists its clips by name.
    with open(filepath, 'rb') as f:
        body = read_milo_container_body(f.read())
    _, entries = _mw_collect_directory_meta(body, 0)
    groups = {}
    for etype, ename, s, e in entries:
        if etype != 'CharClipGroup':
            continue
        data = body[s:e]
        for c in clips:
            tag = struct.pack('>I', len(c['name'])) + c['name'].encode('latin-1')
            if tag in data:
                groups.setdefault(c['name'], []).append(ename)
    return dir_name, clips, groups, failed


def measure_convention(arm_obj, lookup, clips):
    """('ROW' or 'COLUMN', row_median_deg, column_median_deg, bones measured).

    Compares each clip's first-frame rotations with the armature's rest pose. ROW means the
    stored quaternions need conjugating, as RB3's version-19 clips do."""
    errs = {'ROW': [], 'COLUMN': []}
    for clip in clips:
        pose = dict(clip['one']['samples'][0]) if clip['one']['samples'] else {}
        if clip['full']['samples']:
            pose.update(clip['full']['samples'][0])
        for chan, (kind, value) in pose.items():
            if kind != 'quat':
                continue
            bone_name = lookup.get(_channel_stem(chan))
            pb = arm_obj.pose.bones.get(bone_name) if bone_name else None
            if pb is None:
                continue
            b = pb.bone
            rl = (b.parent.matrix_local.inverted() @ b.matrix_local) if b.parent \
                else b.matrix_local
            rest = rl.to_quaternion()
            x, y, z, w = value
            q = Quaternion((w, x, y, z))
            for conv, cand in (('COLUMN', q), ('ROW', q.conjugated())):
                d = abs(sum(a * c for a, c in zip(rest, cand)))
                errs[conv].append(math.degrees(2.0 * math.acos(min(1.0, d))))
    if not errs['ROW']:
        return 'COLUMN', None, None, 0
    row = sorted(errs['ROW'])[len(errs['ROW']) // 2]
    col = sorted(errs['COLUMN'])[len(errs['COLUMN']) // 2]
    if abs(row - col) < _MIN_CONVENTION_GAP_DEG:
        return 'COLUMN', row, col, len(errs['ROW'])     # too close: see the module docstring
    return ('ROW' if row < col else 'COLUMN'), row, col, len(errs['ROW'])


# ---------------------------------------------------------------------------------------
# Root motion: bone_facing
# ---------------------------------------------------------------------------------------

_FACING_POS = 'bone_facing.pos'
_FACING_ROT = 'bone_facing.rotz'


def facing_targets(arm_obj, lookup, clip):
    """Channel stems of the bones bone_facing is applied to: those the clip animates that sit
    at the top of the armature's hierarchy (the pelvis, plus any other root-level bones such
    as props). Every clip channel is expressed relative to the facing point, so these are the
    ones that need moving onto the floor; everything below the pelvis follows it."""
    stems = set()
    for b, _w in clip['full']['bones'] + clip['one']['bones']:
        stem = _channel_stem(b)
        if stem == 'bone_facing':
            continue
        bone_name = lookup.get(stem)
        pb = arm_obj.pose.bones.get(bone_name) if bone_name else None
        if pb is not None and pb.bone.parent is None:
            stems.add(stem)
    return stems


def apply_facing(channels, targets, conjugate):
    """Applies a frame's bone_facing to the target bones, as CharServoBone::Poll does with
    MoveToFacing (DC3 decompilation, system/char/CharServoBone.cpp).

    The game never poses bone_facing as a bone - no Dance Central rig has one, and
    CharBonesMeshes suppresses the missing-bone warning for that name. Instead, after posing
    the skeleton it takes the pelvis (and any top-level bones), rotates its position and its
    axes about +Z by bone_facing.rotz, then adds bone_facing.pos. That's where all of a
    dancer's travel lives: in rest_step_fwd, bone_facing walks 19.5 units forward while the
    pelvis moves under 2. Without this the body animates in place.

    The rotation matches the engine's RotateAboutZ (math/Rot.h, Rot.cpp): the standard
    counter-clockwise turn about +Z, applied to the position and to each of the bone's axes.
    Positions share one space in the game and Blender, so that's a plain world-Z rotation,
    whatever rotation convention the clip's own quaternions use. Every retail clip checked
    has zero facing rotation, so only the translation has been seen in real data.

    Returns a new channel dict; channels are in stored (pre-decode) form."""
    fpos = channels.get(_FACING_POS)
    frot = channels.get(_FACING_ROT)
    if fpos is None and frot is None:
        return channels
    fx, fy, fz = fpos[1] if fpos else (0.0, 0.0, 0.0)
    r = frot[1] if frot else 0.0
    c, s = math.cos(r), math.sin(r)
    qz = Quaternion((math.cos(r * 0.5), 0.0, 0.0, math.sin(r * 0.5)))
    out = dict(channels)
    for stem in targets:
        key = f"{stem}.pos"
        if key in out:
            x, y, z = out[key][1]
            out[key] = ('pos', (x * c - y * s + fx, x * s + y * c + fy, z + fz))
        if r == 0.0:
            continue
        key = f"{stem}.quat"
        if key in out:
            x, y, z, w = out[key][1]
            q = Quaternion((w, x, y, z))
            # The decoded (Blender-side) rotation is q, or its conjugate for row-convention
            # clips; turn that about world Z, then store it back the same way.
            if conjugate:
                q = (qz @ q.conjugated()).conjugated()
            else:
                q = qz @ q
            out[key] = ('quat', (q[1], q[2], q[3], q[0]))
        key = f"{stem}.rotz"
        if key in out:
            out[key] = ('rotz', out[key][1] + (-r if conjugate else r))
    return out


def _rest_rows(arm_obj, name):
    """A bone's rest transform relative to its parent, in engine rows (Blender's matrix
    columns are the basis vectors; the solvers want them as rows)."""
    bone = arm_obj.pose.bones[name].bone
    rl = (bone.parent.matrix_local.inverted() @ bone.matrix_local) if bone.parent \
        else bone.matrix_local
    t = rl.to_translation()
    return tuple(tuple(rl[j][i] for j in range(3)) for i in range(3)), (t.x, t.y, t.z)


def _frame_locals(arm_obj, channels, conjugate, rest_cache):
    """local(name) -> (rows, translation) for any bone at one frame: the clip's own value
    where it animates the bone, its rest pose otherwise."""
    def local(name):
        if name not in rest_cache:
            rest_cache[name] = _rest_rows(arm_obj, name)
        R, t = rest_cache[name]
        stem = name.rsplit('.', 1)[0] if name.endswith(('.mesh', '.trans')) else name
        e = channels.get(f"{stem}.quat")
        if e is not None:
            R = _engine_rows_from_quat(_decode_channel('quat', e[1], conjugate).normalized())
        else:
            e = channels.get(f"{stem}.rotz")
            if e is not None:
                R = _engine_rows_from_hinge(_decode_channel('rotz', e[1], conjugate))
        e = channels.get(f"{stem}.pos")
        if e is not None:
            v = _decode_channel('pos', e[1], conjugate)
            t = (v[0], v[1], v[2])
        return R, t
    return local


def write_blend_keys(cache, arm_obj, solver, channels, frame, conjugate, prev, rest_cache):
    """Runs the character's CharBlendBones for one frame and keys their target bones:
    rotation always, location only where a solver blends position."""
    out = solver.solve(_frame_locals(arm_obj, channels, conjugate, rest_cache))
    for name, (R, t) in out.items():
        pb = arm_obj.pose.bones.get(name)
        if pb is None:
            continue
        if pb.rotation_mode != 'QUATERNION':
            pb.rotation_mode = 'QUATERNION'
        q = pose_basis_for_bone(pb, 'quat', _quat_from_engine_rows(R))
        last = prev.get(name)
        if last is not None and sum(a * b for a, b in zip(q, last)) < 0.0:
            q = Quaternion((-q[0], -q[1], -q[2], -q[3]))
        prev[name] = q
        path = f'pose.bones["{name}"]'
        for i in range(4):
            cache.key(f'{path}.rotation_quaternion', i, name, frame, q[i])
        if name in solver.pos_targets:
            from mathutils import Vector
            loc = pose_basis_for_bone(pb, 'pos', Vector(t))
            for i in range(3):
                cache.key(f'{path}.location', i, name, frame, loc[i])


class _IMPORT_OT_dc_clip_base(bpy.types.Operator, ImportHelper):
    """Shared Dance Central clip import. Not registered itself - each game subclasses it and
    sets _game, which picks the clip reader (read_dc_clips). Once the clips are read,
    everything is shared: real-time Actions, the convention measurement, arm twist."""
    bl_options = {'REGISTER', 'UNDO'}
    _game = "DC1"

    filename_ext = ".milo_xbox"
    filter_glob: StringProperty(default="*.milo_xbox", options={'HIDDEN'})

    rotation_convention: EnumProperty(
        name="Rotation Convention",
        items=[('AUTO', "Auto (measure)", "Compare each clip's first frame with the "
                                          "armature's rest pose and use the closer "
                                          "convention. Falls back to RB3's if too close"),
               ('ROW', "Conjugate (RB3-style)", "Conjugate stored rotations, as RB3's "
                                                "version-19 clips need"),
               ('COLUMN', "As stored (GDRB-style)", "Use stored rotations unchanged")],
        default='AUTO',
    )
    apply_facing: BoolProperty(
        name="Apply Root Motion (bone_facing)",
        description="Move the dancer around the floor the way the game does: each frame, "
                     "the pelvis (and any other top-level bones) is turned about the vertical "
                     "axis by bone_facing.rotz and offset by bone_facing.pos. All of a "
                     "dancer's travel is in bone_facing, and no rig has a bone of that name, "
                     "so without this the dancer animates in place",
        default=True,
    )
    solve_arm_twist: BoolProperty(
        name="Solve Arm Twist Bones",
        description="Compute the twist bones the game's solvers drive. Forearms use the "
                     "forearm twist solver (its standard settings match Dance Central's "
                     "skeletons). Shoulder, thigh and spine twist bones use the CharBlendBone "
                     "solvers in the dancer's character milo, set with Set Character Milo. "
                     "Dance Central clips don't animate them, so without this they stay at "
                     "rest while "
                     "the arms move",
        default=True,
    )

    def execute(self, context):
        arm = context.active_object
        if arm is None or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Select the target armature first.")
            return {'CANCELLED'}
        try:
            dir_name, clips, groups, failed = read_dc_clips(self.filepath, self._game)
        except Exception as e:
            _log(f"{self._game} CLIP IMPORT FAILED: {e}")
            self.report({'ERROR'}, f"Could not read the clip milo: {e}")
            return {'CANCELLED'}
        if not clips:
            self.report({'ERROR'}, "No readable CharClips in this milo.")
            return {'CANCELLED'}

        _log(f"===== Importing {self._game} clips '{dir_name}' from {self.filepath} =====")
        lookup = build_bone_lookup(arm)
        if self.rotation_convention == 'AUTO':
            conv, row, col, n = measure_convention(arm, lookup, clips)
            if n:
                _log(f"  Rotation convention: first frames vs rest pose over {n} rotation(s) "
                     f"- conjugated median {row:.1f} deg, as stored {col:.1f} deg -> "
                     f"{'conjugating (RB3-style)' if conv == 'ROW' else 'as stored'}"
                     + (" [too close to call; using as stored]"
                        if abs(row - col) < _MIN_CONVENTION_GAP_DEG else ""))
            else:
                _log("  Rotation convention: no clip bones on this armature to measure - "
                     "using as stored")
        else:
            conv = self.rotation_convention
            _log(f"  Rotation convention: {conv} (set manually)")
        conjugate = (conv == 'ROW')

        # The forearm solver's standard offsets are right for Dance Central as they stand -
        # both DC1 skeletons reproduce their stored forearm twist rest pose with them to
        # within 0.05 degrees - so no GDRB-style correction is applied. A character milo set
        # with Set Character Milo is still used if it carries forearm solver settings.
        char_path = arm.get(CHARACTER_PATH_PROP, "") if self.solve_arm_twist else ""
        rigs, notes = build_twist_rigs(arm, lookup, self.solve_arm_twist, char_path,
                                       offset_correction=0.0)
        for n in notes:
            _log(f"  {n}")

        # Shoulder, thigh and spine twist bones: CharBlendBones from the character milo.
        blend = None
        if self.solve_arm_twist:
            if char_path:
                try:
                    with open(bpy.path.abspath(char_path), 'rb') as f:
                        configs = read_blend_bones(read_milo_container_body(f.read()))
                    parents = {pb.bone.name: (pb.bone.parent.name if pb.bone.parent else None)
                               for pb in arm.pose.bones}
                    blend = BlendBoneSolver(configs, lookup, parents)
                    for n in blend.notes:
                        _log(f"  {n}")
                    if blend.solvers:
                        _log(f"  blend bones: {len(blend.solvers)} solver(s) from the character "
                             f"milo drive {len(blend.targets)} bone(s): "
                             f"{', '.join(b.rsplit('.', 1)[0] for b in blend.targets)}")
                    else:
                        _log("  blend bones: none found in the character milo")
                        blend = None
                except Exception as e:
                    _log(f"  blend bones: could not read the character milo ({e})")
                    blend = None
            else:
                _log("  blend bones: no character milo set, so the shoulder, thigh and spine "
                     "twist bones aren't solved - use Object > Rock Band Animation > Set "
                     "Character Milo with the dancer's character milo (e.g. main.milo_xbox)")

        scene = context.scene
        fps = scene.render.fps / max(scene.render.fps_base, 1e-6)
        pose_bones = arm.pose.bones
        made, unmatched = 0, set()
        for clip in clips:
            name = clip['name']
            frames = int(math.ceil(clip['seconds'] * fps)) + 1
            action_name = f"CLIP_{name}"
            old = bpy.data.actions.get(action_name)
            if old is not None:
                bpy.data.actions.remove(old)
            action = bpy.data.actions.new(action_name)
            action["milo_clip"] = True
            action["milo_clip_name"] = name
            action["milo_clip_set"] = dir_name
            action["milo_clip_groups"] = groups.get(name, [])
            action["milo_clip_seconds"] = clip['seconds']
            action["milo_clip_game"] = self._game
            action.use_fake_user = True
            slot = action.slots.new(id_type='OBJECT', name=arm.name)
            cache = _CurveCache(_get_channelbag(action, slot))
            counters = {'keys': 0, 'unmatched': set(), 'twist_keys': 0}
            twist_prev = {}
            blend_prev = {}
            rest_cache = {}
            targets = facing_targets(arm, lookup, clip) if self.apply_facing else set()
            for k in range(frames):
                t = min(k / fps, clip['seconds'])
                table = clip.get('table')
                if table:
                    # Frame table: clip time (30 entries per second) -> fractional sample.
                    idx = t * TABLE_RATE
                    i0 = max(0, min(int(idx), len(table) - 1))
                    i1 = min(i0 + 1, len(table) - 1)
                    sample_pos = table[i0] + (table[i1] - table[i0]) * (idx - i0)
                else:
                    sample_pos = t * clip['fps']
                channels = _interp_channels(clip, sample_pos)
                if targets:
                    channels = apply_facing(channels, targets, conjugate)
                write_pose_keys(cache, pose_bones, lookup, channels, k + 1, conjugate, counters)
                for rig in rigs:
                    write_twist_keys(cache, pose_bones, rig, channels, k + 1, conjugate,
                                     twist_prev, counters)
                if blend is not None:
                    write_blend_keys(cache, arm, blend, channels, k + 1, conjugate,
                                     blend_prev, rest_cache)
            cache.finish()
            unmatched |= counters['unmatched']
            made += 1
            grp = f", group {', '.join(groups[name])}" if name in groups else ""
            if self.apply_facing and not targets:
                _log(f"  '{name}': no top-level animated bone found to carry bone_facing - "
                     f"root motion not applied")
            _log(f"  '{name}': {clip['num_samples']} sample(s) at {clip['fps']:g} fps = "
                 f"{clip['seconds']:.2f}s -> {frames} frame(s), {counters['keys']} key(s){grp}")

        for n, r in failed:
            _log(f"  SKIPPED {n}: {r}")
        if unmatched:
            stems = sorted({_channel_stem(c) for c in unmatched})
            _log(f"  {len(stems)} animated bone(s) aren't on '{arm.name}': {', '.join(stems)}")
        summary = (f"Imported {made} {self._game} clip(s) from '{dir_name}' as CLIP_ Actions"
                   + (f", {len(failed)} failed" if failed else ""))
        _log(f"===== {summary} =====")
        _log("  Each Action plays at the clip's own speed - assign one to preview it.")
        self.report({'WARNING' if failed else 'INFO'}, summary)
        return {'FINISHED'}


class IMPORT_OT_dc1_clip_set(_IMPORT_OT_dc_clip_base):
    """Import every animation clip in a Dance Central 1 clip milo as its own Action on the
    active armature, keyed in real time at the clip's own frame rate - no bake step"""
    bl_idname = "import_scene.dc1_clip_set"
    bl_label = "Import DC1 CharClip Milo"
    _game = "DC1"


class IMPORT_OT_dc3_clip_set(_IMPORT_OT_dc_clip_base):
    """Import every animation clip in a Dance Central 3 milo - a song's routine sections,
    move transitions and rest clips - as its own Action on the active armature, keyed in
    real time at the clip's own frame rate - no bake step"""
    bl_idname = "import_scene.dc3_clip_set"
    bl_label = "Import DC3 CharClip Milo"
    _game = "DC3"


class IMPORT_OT_dc2_clip_set(_IMPORT_OT_dc_clip_base):
    """Import every animation clip in a Dance Central 2 milo - a song's routine sections,
    move transitions and rest clips - as its own Action on the active armature, keyed in
    real time at the clip's own frame rate - no bake step"""
    bl_idname = "import_scene.dc2_clip_set"
    bl_label = "Import DC2 CharClip Milo"
    _game = "DC2"


class IMPORT_OT_rb2_clip_set(_IMPORT_OT_dc_clip_base):
    """Import every animation clip in a Rock Band 2 clip milo (a musician's idles, rhythm
    and solo moves) as its own Action on the active armature, keyed in real time at the
    clip's own tempo, with the musician's movement and turning around the stage applied"""
    bl_idname = "import_scene.rb2_clip_set"
    bl_label = "Import RB2 CharClip Milo"
    _game = "RB2"
    filename_ext = ".milo_xbox"
    filter_glob: StringProperty(default="*.milo_xbox;*.milo_ps3", options={'HIDDEN'})

    # No RB2 skeleton or character milo has been available to check the twist solvers'
    # settings against, as was done for GDRB and Dance Central, so this starts off.
    solve_arm_twist: BoolProperty(
        name="Solve Twist Bones",
        description="Compute twist bones with the standard forearm solver settings. Off by "
                     "default for Rock Band 2: its settings haven't been verified",
        default=False,
    )
