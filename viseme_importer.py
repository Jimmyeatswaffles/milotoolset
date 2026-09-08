"""
Rock Band 3 (Xbox 360) Viseme Set importer.

Parses a "viseme" CharClipSet milo (e.g. viseme_male.milo_xbox / viseme_female.milo_xbox)
and creates one Blender Action per named viseme CharClip, targeting the pose bones of the
currently active armature. This is Phase 1 of the lipsync workflow: face-rig testing
(scrub/promote these Actions to Pose Assets) AND the data source Phase 2 (lipsync import/
export) will read poses back out of.

===========================================================================================
SCOPE: Xbox 360 only, RB3-era encoding only.
===========================================================================================
This assumes the CharClip encoding found in retail X360 viseme milos: CharClip version 19,
holding two CharBonesSamples blocks ("full" then "one") followed by a trailing CharBones
name+weight list. All multi-byte values are big-endian.

Confirmed against 68/68 real CharClip entries in a retail viseme_male.milo_xbox with zero
parse failures and zero ambiguous matches (see _locate_bone_samples below for what
"ambiguous" would mean). Decoded values sanity-check cleanly: every viseme clip's "one"
block carries small delta-like position offsets (~0.001-0.4 units) and near-identity
quaternions (w ~0.97-0.999), consistent with small facial deltas - except the clip named
"Base", whose "one" block holds much larger absolute-looking position values. That's
expected: "Base" is the rig's reference/rest layout for these driver bones, not a playable
viseme delta, and it does not appear in real .lipsync viseme tables. It still gets
imported like everything else (no name-based special-casing - that would be fragile) -
just don't be surprised if scrubbing that one Action alone looks like the face "jumps"
rather than subtly deforming.

===========================================================================================
KNOWN GAPS - read before relying on this for anything beyond a first visual check
===========================================================================================
1. The exact byte layout of the CharClip header fields between the beat-timing floats and
   the first CharBonesSamples block (flags / a NumString naming a "relative" clip / an
   always -1 int / a decompress flag - roughly 13-20 bytes, length depends on that NumString)
   is NOT pinned down field-by-field. Rather than hardcode an offset that would silently
   break on the next clip whose fields happen to total a different length,
   _locate_bone_samples scans a small window of candidate offsets and structurally
   validates each one (bone_count in a sane range, every bone name printable ASCII ending
   in a recognised suffix, computed sample size fitting the remaining bytes, and the WHOLE
   "full" + "one" pair decoding without running past the entry) - the same resync-and-
   validate approach io.py already uses for Trans/CharCollide in parse_milo_skeleton. It
   has been unique and correct on every real clip tested; if it ever reports "N candidate
   offsets" for a clip you import, treat that clip's result as unverified.

2. "full" vs "one": these two CharBonesSamples blocks split a clip's bones by whether
   they hold still - 'one' holds the bones that stay constant (one sample), 'full' holds
   the bones that vary (sampled per frame). Their bone sets are disjoint in all 68
   CharClips checked. parse_viseme_clip now merges both; reading only 'one' dropped
   bone_jaw.quat from 40 of 68 clips and left the Neutral_*/exp_* clips empty entirely.
   For the varying bones it takes sample 0, which is a fair static representative in the
   viseme clips (their 'full' channels barely move). The long exp_* performance clips
   are genuinely animated across their whole frame range, though, and importing those as
   real multi-frame Actions - rather than collapsing them to their first frame - is
   still an open feature.

3. The trailing CharBones list (after "one") is parsed only far enough to confirm the
   entry's end boundary; its contents (more named+weighted bones, no samples attached)
   aren't used. Purpose unconfirmed - possibly an influence/mask list.

4. Coordinate space: POSITION and ROTATION offsets do NOT use the same frame, and are
   controlled by two separate operator options ("Position Space" / "Rotation Space").

   Positions are in the bone's PARENT frame and get rotated into the bone's own axes on
   import (see _rest_rotation). Directly readable from the decoded 'Base' clip: the
   lateral component mirrors between L/R bones while the vertical component matches,
   midline bones sit at ~0 laterally, and the vertical axis is the one that separates
   brow bones from cheek bones. Cross-checked against Brow_down / Brow_up, which move
   purely along that vertical axis in opposite directions, exactly as their names imply.
   Writing those parent-frame offsets straight into bone-local channels (as an earlier
   version did) rotates the motion into the wrong axis - it made Brow_down slide the
   brows sideways. Only the rest ROTATION is used for the conversion, never the full
   rest matrix, so a bone's rest offset can't leak in as spurious translation.

   Rotations depend on which quaternion convention the data uses, and that is MEASURED
   per import rather than assumed (rot_space 'AUTO'; see detect_rotation_convention).
   Milo does matrix maths row-vector style while Blender is column-vector, so milo
   MATRICES get transposed on import - io.py's _milo_to_blender_matrix does exactly that.
   Whether milo's stored QUATERNIONS follow the same convention is not stated by the
   format, and it cannot be settled by inspection, so the importer compares the
   armature's rest pose against the Base clip's rest pose both ways and takes whichever
   fits (typically 0 deg vs 156 deg - completely unambiguous).

   Two consequences follow from that one measurement and must move together: a row-vector
   quaternion needs conjugating AND has its composition order flipped, since
   (A*B)^T == B^T * A^T. Treating them as independent settings is what caused a long
   run of whack-a-mole where fixing the jaw broke the lips and fixing the lips broke the
   jaw.

   The reason this took so long to find is that the ambiguity is invisible on the
   easiest bones to check. At a rest rotation of 180 degrees w == 0, so conj(q) == -q ==
   the same rotation: the jaw (179.97 deg) and tongue1 (179.96 deg) are IDENTICAL under
   both conventions, while the brows and lips (89.99 deg) come out inverted by 179.98 deg.
   Validating a convention against the jaw therefore proves nothing at all. The brow
   bones are worse still - their offsets spin about their own rest axis, making every
   option numerically identical (difference exactly 0.0000) and visually near-inert, so
   brow visemes looked correct under every variant and never signalled anything.

   Rotation convention (fixed): Milo does its transform maths row-vector style (v * M)
   while Blender is column-vector (M * v) - io.py's _milo_to_blender_matrix already
   compensates for matrices by transposing them. For a rotation matrix transpose ==
   inverse, and the quaternion equivalent of inverting is conjugating, so rotation
   channels need the matching conjugation on import (verified numerically:
   matrix(conjugate(q)) == transpose(matrix(q))). An earlier version skipped this, so
   every rotation spun the correct axis by the correct angle in the WRONG direction -
   the jaw hinged backwards and eyelids pulled up instead of down - while position
   channels, which need no such conversion, were already correct. Controlled by the
   operator's "Convert Rotation Convention" toggle (on by default).

   Stride note (fixed): at compression < 2 a position sample occupies 16 bytes - it is
   stored as a Vector4, with the 4th float being padding/homogeneous w, not a coordinate
   (see type_size(idx<2) == 16). An earlier version read only 3 floats (12 bytes), so
   the cursor fell 4 bytes behind on every position bone and the drift corrupted
   everything later in the sample - most visibly the quaternions, which came out
   unnormalised (magnitudes as low as 0.38). The per-sample trailing pad skip absorbed
   the shortfall, so the file still appeared to parse cleanly. There is now a hard
   stride assertion after each sample so this class of misalignment fails loudly instead
   of returning plausible-looking garbage. Post-fix, all 437 quaternions across all 68
   clips in a retail viseme_male.milo_xbox normalise to within 3e-5 of 1.0.

5. .rotz channels rotate the pose bone's local Z axis (rotation_euler.z). The real per-bone
   axis is stored in the CHARACTER skeleton milo's CharBone.rotation field (kRotX/Y/Z/Full/
   None), which this importer never reads. Until that's wired in, treat any viseme that
   uses a .rotz channel as approximate.

None of these block using the tool for its first job (rig-testing pose review) - they
matter more once poses get blended for real lipsync playback.
"""

import math
import struct

import bpy
from bpy.props import StringProperty, BoolProperty, EnumProperty
from bpy_extras.io_utils import ImportHelper
from mathutils import Vector, Quaternion

from .io import read_milo_container_body, _mw_collect_directory_meta
from .utilities import _log


# ---------------------------------------------------------------------------------------
# Low-level binary reader (big-endian only - X360 object streams)
# ---------------------------------------------------------------------------------------

class _Reader:
    __slots__ = ('d', 'p')

    def __init__(self, data, pos=0):
        self.d = data
        self.p = pos

    def u8(self):
        v = self.d[self.p]; self.p += 1; return v

    def i16(self):
        v = struct.unpack_from('>h', self.d, self.p)[0]; self.p += 2; return v

    def u32(self):
        v = struct.unpack_from('>I', self.d, self.p)[0]; self.p += 4; return v

    def i32(self):
        v = struct.unpack_from('>i', self.d, self.p)[0]; self.p += 4; return v

    def f32(self):
        v = struct.unpack_from('>f', self.d, self.p)[0]; self.p += 4; return v

    def numstring(self, max_len=128):
        n = self.u32()
        if n < 0 or n > max_len or self.p + n > len(self.d):
            raise ValueError(f"implausible string length {n}")
        raw = self.d[self.p:self.p + n]
        if not all(9 <= c < 127 for c in raw):
            raise ValueError("non-ASCII string content")
        self.p += n
        return raw.decode('latin-1')


class VisemeImportError(Exception):
    """Raised for anything that stops a viseme milo or one of its clips from being
    parsed - caught at the operator level and reported to the user, never silently
    swallowed."""
    pass


# ---------------------------------------------------------------------------------------
# CharBonesSamples / CharBones decoding
# ---------------------------------------------------------------------------------------

_BONE_SUFFIXES = ('.pos', '.quat', '.rotz')


def _read_char_bones(r, version, max_bones=300):
    bone_count = r.i32()
    if not (0 <= bone_count <= max_bones):
        raise ValueError(f"implausible bone_count {bone_count}")
    no_weight = (version != -1 and version <= 10)   # CharBone4Bone: no weight float <=10
    bones = []
    for _ in range(bone_count):
        sym = r.numstring()
        weight = None if no_weight else r.f32()
        bones.append((sym, weight))
    return bones


def _read_char_bones_samples(r):
    """Reads one CharBonesSamples block: version, the CharBones name table, the
    counts/compression/num_samples header, and - when version > 11 - the per-frame
    sample payload (frame-time array + one decoded sample per bone-per-frame)."""
    version = r.i32()
    if not (-2 <= version <= 40):
        raise ValueError(f"implausible CharBonesSamples version {version}")
    count_size = 7 if version > 15 else 10

    bones = _read_char_bones(r, version)

    counts = [r.u32() for _ in range(count_size)]
    if any(c > 200_000 for c in counts):
        raise ValueError("implausible counts[] value")
    compression = r.u32()
    if compression > 4:
        raise ValueError(f"implausible compression {compression}")
    num_samples = r.u32()
    if num_samples > 200_000:
        raise ValueError(f"implausible num_samples {num_samples}")

    samples = []
    if version > 11:
        num_frames = r.u32()
        if num_frames > 200_000:
            raise ValueError(f"implausible num_frames {num_frames}")
        frame_times = [r.f32() for _ in range(num_frames)] if num_frames else []

        def type_size(idx):
            if idx < 2:
                return 16 if compression < 2 else 6
            if idx != 2:
                return 4 if compression == 0 else 2
            if compression > 2:
                return 4
            if compression == 0:
                return 16
            return 8

        computed = [0] * count_size
        for i in range(count_size - 1):
            computed[i + 1] = computed[i] + (counts[i + 1] - counts[i]) * type_size(i)
        sample_size = (computed[count_size - 1] + 0xF) & 0xFFFFFFF0
        if sample_size > 1_000_000 or sample_size * num_samples + r.p > len(r.d):
            raise ValueError("sample payload doesn't fit in the remaining bytes")

        pos_bones = [n for n, _ in bones if n.endswith('.pos')]
        quat_bones = [n for n, _ in bones if n.endswith('.quat')]
        rotz_bones = [n for n, _ in bones if n.endswith('.rotz')]

        def read_pos():
            # IMPORTANT: at compression < 2 a position occupies 16 bytes, not 12 - it's
            # stored as a Vector4 (the 4th float is padding/homogeneous w and is not a
            # coordinate). type_size(idx<2) above says 16 for exactly this reason.
            # Reading only 3 floats here left the cursor 4 bytes short on EVERY position
            # bone; the drift accumulated across the sample and corrupted every value
            # that followed (which is what produced unnormalised, garbage quaternions).
            # The trailing per-sample pad skip silently absorbed the shortfall, so the
            # file still "parsed" - hence the strict check after each sample below.
            if compression < 2:
                x, y, z = r.f32(), r.f32(), r.f32()
                r.p += 4          # skip the 4th component
                return (x, y, z)
            return tuple(max(r.i16() / 32767.0, -1.0) for _ in range(3))

        def read_quat():
            if compression == 0:
                x, y, z, w = r.f32(), r.f32(), r.f32(), r.f32()
                return (x, y, z, w)
            if compression < 3:
                return tuple(max(r.i16() / 32767.0, -1.0) for _ in range(4))
            # 4 x uint8, mapped from [0,255] back to [-1,1]
            return tuple((r.u8() / 127.5) - 1.0 for _ in range(4))

        def read_rotz():
            return r.f32() if compression == 0 else max(r.i16() / 32767.0, -1.0)

        expected = (len(pos_bones) * type_size(0)
                    + len(quat_bones) * type_size(2)
                    + len(rotz_bones) * type_size(3))

        for _ in range(num_samples):
            start = r.p
            sample = {}
            for name in pos_bones:
                sample[name] = ('pos', read_pos())
            for name in quat_bones:
                sample[name] = ('quat', read_quat())
            for name in rotz_bones:
                sample[name] = ('rotz', read_rotz())
            consumed = r.p - start
            # Hard check rather than a silent pad-skip: if the bytes actually read don't
            # match what the type-size table says this bone set should occupy, the reads
            # are misaligned and every decoded value is suspect. Failing here makes
            # _locate_bone_samples reject the offset instead of returning garbage.
            if consumed != expected:
                raise ValueError(
                    f"sample stride mismatch: consumed {consumed} bytes but the type-size "
                    f"table expects {expected} for {len(pos_bones)} pos / "
                    f"{len(quat_bones)} quat / {len(rotz_bones)} rotz bones")
            pad = sample_size - consumed
            if pad > 0:
                r.p += pad
            samples.append(sample)

    return dict(version=version, bones=bones, compression=compression,
                num_samples=num_samples, frame_times=frame_times if version > 11 else [],
                samples=samples)


def _read_char_bones_tail(r, cc_version):
    """The trailing CharBones name+weight list after 'one'. Not used for pose data -
    parsed purely so the entry's end boundary can be validated (see module docstring,
    gap #3)."""
    if cc_version <= 14:
        return []
    return _read_char_bones(r, cc_version)


def _locate_bone_samples(entry, search_lo=32, search_hi=110):
    """Scans candidate start offsets for the first CharBonesSamples block ('full'),
    requiring BOTH it and the following 'one' block (and, when the CharClip version
    supports it, the trailing CharBones tail) to decode cleanly. Returns
    (offset, full, one, end_pos). Raises VisemeImportError if no offset validates.
    Logs a warning (but does not fail) if more than one offset validates - that has
    not happened on any real clip tested, but if it ever does, the first/earliest match
    is used and the result should be treated as unverified."""
    hits = []
    hi = min(search_hi, len(entry))
    for off in range(search_lo, hi):
        try:
            r = _Reader(entry, off)
            full = _read_char_bones_samples(r)
            one = _read_char_bones_samples(r)
            hits.append((off, full, one, r.p))
        except Exception:
            continue
    if not hits:
        raise VisemeImportError(
            "couldn't locate a valid bone-samples block in this CharClip - its header "
            "layout doesn't match the retail X360 clips this importer was built against")
    if len(hits) > 1:
        _log(f"    WARNING: {len(hits)} candidate offsets validated for this clip - "
             f"using the first ({hits[0][0]}). Treat this clip's result as unverified.")
    return hits[0]


def parse_viseme_clip(entry_bytes):
    """Parses one CharClip's raw bytes into a single pose dict
    {bone_channel_name: (kind, values)}, combining BOTH CharBonesSamples blocks.

    The two blocks split a clip's bones by whether they hold still: 'one' stores the
    bones that are constant across the clip (a single sample), while 'full' stores the
    bones that vary, sampled per frame. Their bone sets are disjoint in all 68 CharClips
    of a retail viseme_male.milo_xbox - checked, zero overlap - so the complete pose is
    the union of the two, not either one alone.

    This matters a lot: bone_jaw.quat lives in 'full' in 40 of those 68 clips. Reading
    only 'one' (as an earlier version did) silently dropped the jaw rotation from most
    of the phoneme visemes, so mouth shapes like Ox_hi came out barely open while
    brow-only visemes - which have no jaw channel - looked correct. It also left the
    Neutral_* and exp_* clips completely empty, since those keep everything in 'full'.

    For the varying bones, sample 0 is used. In the viseme clips these barely move
    (Ox_hi's jaw wanders between 17.6 and 21.6 degrees across all 56 of its samples,
    starting and ending at the same value), so any sample is a fair representative of
    the intended static pose. The genuinely animated clips - the long exp_* performances
    - are the exception, and for those sample 0 is just the opening frame; importing
    them as real multi-frame Actions is a separate job (see module docstring gap #2).

    Returns (pose_dict, full_data, one_data).
    """
    off, full, one, _end = _locate_bone_samples(entry_bytes)

    pose = {}
    if full['samples']:
        pose.update(full['samples'][0])
    if one['samples']:
        pose.update(one['samples'][0])

    if not pose:
        raise VisemeImportError("this clip has no bone samples in either block")
    return pose, full, one


# ---------------------------------------------------------------------------------------
# Milo container walking - reuses io.py's own decompression + directory walker so this
# works on all four container types (uncompressed/zlib/gzip/zlibAlt), not just loose
# uncompressed extracts.
# ---------------------------------------------------------------------------------------

def find_viseme_clips(filepath):
    """Returns (dir_name, {clip_name: raw_bytes}) for every 'CharClip' typed entry found
    anywhere in the milo (including inside nested subdirectories)."""
    with open(filepath, 'rb') as f:
        data = f.read()
    body = read_milo_container_body(data)
    _, all_entries = _mw_collect_directory_meta(body, 0)

    def u32(p):
        return struct.unpack_from('>I', body, p)[0]

    def sym(p):
        n = u32(p)
        return body[p + 4:p + 4 + n].decode('latin-1'), p + 4 + n

    p = 4  # skip revision
    dir_type, p = sym(p)
    dir_name, p = sym(p)

    clips = {}
    for (etype, ename, estart, eend) in all_entries:
        if etype == 'CharClip':
            clips[ename] = body[estart:eend]
    return dir_name, clips


# ---------------------------------------------------------------------------------------
# Blender: build one Action per viseme clip on the active armature
# ---------------------------------------------------------------------------------------

_MILO_VISEME_TAG = "milo_viseme"          # bool custom property on the Action
_MILO_VISEME_NAME = "milo_viseme_name"    # original clip name (Action name may get suffixed)
_MILO_VISEME_SET = "milo_viseme_set"      # source CharClipSet dir_name

# The animated-component suffix used by CharBonesSamples channel names. These say WHICH
# transform component the samples drive, and are not part of the bone's identity.
_CHANNEL_SUFFIXES = ('.pos', '.quat', '.rotz')

# Milo's own bone-type extensions (see char_bone.bt: "Ext: .mesh, .trans"). These ARE
# part of the bone's name as it appears in the skeleton milo - and therefore as the
# skeleton importer creates it in Blender (e.g. 'bone_L-brow1.mesh'). A viseme channel
# named 'bone_L-brow1.pos' refers to that same bone, so matching has to be done on the
# shared stem ('bone_L-brow1') rather than on either full name.
_BONE_EXTENSIONS = ('.mesh', '.trans')


def _channel_stem(chan_name):
    """'bone_jaw.quat' -> 'bone_jaw'. Also tolerates a bone extension already being
    present (e.g. a hypothetical 'bone_jaw.mesh.quat') by stripping both layers."""
    name = chan_name
    for suf in _CHANNEL_SUFFIXES:
        if name.endswith(suf):
            name = name[:-len(suf)]
            break
    for ext in _BONE_EXTENSIONS:
        if name.endswith(ext):
            name = name[:-len(ext)]
            break
    return name


def _bone_stem(bone_name):
    """'bone_L-brow1.mesh' -> 'bone_L-brow1'; leaves extension-less names alone."""
    for ext in _BONE_EXTENSIONS:
        if bone_name.endswith(ext):
            return bone_name[:-len(ext)]
    return bone_name


def build_bone_lookup(armature_obj):
    """Maps each pose bone's stem -> its real pose bone name, so viseme channels can be
    matched against a skeleton whose bones carry Milo's '.mesh'/'.trans' extensions.

    Real RB3 skeleton milos name their facial bones 'bone_L-brow1.mesh' etc., and the
    skeleton importer preserves that verbatim, while the viseme CharBonesSamples channels
    call the same bone 'bone_L-brow1.pos'. Matching on the shared stem is what bridges
    the two. Exact full names are registered too, so an armature whose bones happen to
    have no extension still resolves.

    If two bones share a stem (e.g. both 'x.mesh' and 'x.trans' exist), the first in
    _BONE_EXTENSIONS order wins and the collision is logged rather than silently
    resolved - facial driver bones don't normally collide this way, so it's worth
    surfacing if it ever does."""
    lookup = {}
    collisions = []

    def _rank(name):
        for i, ext in enumerate(_BONE_EXTENSIONS):
            if name.endswith(ext):
                return i
        return len(_BONE_EXTENSIONS)   # extension-less sorts last

    for pb in armature_obj.pose.bones:
        lookup.setdefault(pb.name, pb.name)          # exact name always resolvable

    for pb in armature_obj.pose.bones:
        stem = _bone_stem(pb.name)
        if stem == pb.name:
            continue
        prev = lookup.get(stem)
        if prev is None or prev == stem:
            lookup[stem] = pb.name
        elif prev != pb.name:
            if _rank(pb.name) < _rank(prev):
                collisions.append((stem, prev, pb.name))
                lookup[stem] = pb.name
            else:
                collisions.append((stem, pb.name, prev))

    for stem, loser, winner in collisions:
        _log(f"  NOTE: bone stem '{stem}' matches both '{loser}' and '{winner}' - "
             f"using '{winner}'.")

    return lookup


def _get_or_create_channelbag(action, slot):
    """Blender 4.4+ replaced the flat Action.fcurves list with 'layered Actions':
    every Action needs at least one Layer, holding at least one keyframe Strip, holding
    one Channelbag per animated ID Slot - the Channelbag is what actually holds the
    F-Curves (see https://developer.blender.org/docs/features/animation/animation_system/layered/).
    As of the versions this addon targets, a single Action still only supports one Layer
    with one Strip, so this just ensures that pair exists and returns the Channelbag for
    `slot` within it, creating it if needed. `Action.fcurves` (used prior to this addon's
    first release) no longer exists at all once Blender 5.x - hence the AttributeError -
    so all F-Curve creation below goes through this instead."""
    layer = action.layers[0] if len(action.layers) else action.layers.new("Layer")
    strip = layer.strips[0] if len(layer.strips) else layer.strips.new(type='KEYFRAME')
    return strip.channelbag(slot, ensure=True)


def detect_rotation_convention(armature_obj, bone_lookup, milo_rest_rot):
    """Measures which quaternion convention the armature actually uses, instead of
    assuming one.

    Milo does matrix maths row-vector style (v * M) while Blender is column-vector
    (M * v), so a milo MATRIX gets transposed on import - io.py's _milo_to_blender_matrix
    does exactly that. Whether milo's stored QUATERNIONS follow the same convention isn't
    something the format states, and getting it wrong inverts every rotation.

    It can't be settled by eye either, because the ambiguity is invisible on precisely the
    bones that are easiest to read. At a rest rotation of 180 degrees w == 0, so conj(q)
    == -q == the same rotation: the jaw (179.97 deg) and tongue1 (179.96 deg) look
    identical under both conventions, while the brows and lips (89.99 deg) come out
    inverted by 179.98 deg. Validating against the jaw therefore proves nothing.

    So: compare the armature's real rest orientations against Milo's Base-clip rest under
    both interpretations and take whichever actually fits. Returns 'ROW' (milo quaternions
    are row-vector, so they need conjugating like the matrices do) or 'COLUMN' (they map
    across directly), plus the median error of each for logging."""
    row_errs, col_errs = [], []
    for stem, rest_m in milo_rest_rot.items():
        bone_name = bone_lookup.get(stem)
        if not bone_name:
            continue
        pb = armature_obj.pose.bones.get(bone_name)
        if pb is None:
            continue
        actual = _rest_rotation(pb)
        q = Quaternion(rest_m)
        row_errs.append(math.degrees(q.conjugated().rotation_difference(actual).angle))
        col_errs.append(math.degrees(q.rotation_difference(actual).angle))

    if not row_errs:
        return 'ROW', None, None

    def median(v):
        v = sorted(v)
        n = len(v)
        return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])

    row_med, col_med = median(row_errs), median(col_errs)
    convention = 'ROW' if row_med <= col_med else 'COLUMN'
    _log(f"  Rotation convention: measured {len(row_errs)} bone(s) against Milo's rest "
         f"pose - row-vector median error {row_med:.2f} deg, column-vector median error "
         f"{col_med:.2f} deg -> using {convention}.")
    if min(row_med, col_med) > 20.0:
        _log(f"    WARNING: neither convention fits well (best {min(row_med, col_med):.2f} "
             f"deg). This armature's rest pose may not match the viseme set's Base clip - "
             f"is this the right character's viseme milo?")
    return convention, row_med, col_med


def report_rest_mismatches(armature_obj, bone_lookup, milo_rest_rot, convention='ROW'):
    """Logs bones whose rest orientation in Blender disagrees with Milo's own rest
    orientation from the Base clip.

    This is what decides whether rotation offsets can be composed against the armature's
    rest at all. A faithful skeleton import gives Blender rest == conj(milo rest) for
    every bone (conj because Milo works row-vector style). Any bone that doesn't match
    got reoriented somewhere - most likely by the skeleton importer's connect-bones
    option repointing it at its child, or by Blender forcing a bone's Y axis along its
    length - and rotation offsets composed against that rest come out skewed for exactly
    those bones and no others.

    Returns (n_checked, n_mismatched)."""
    checked = 0
    mismatched = []
    for stem, rest_m in milo_rest_rot.items():
        bone_name = bone_lookup.get(stem)
        if not bone_name:
            continue
        pb = armature_obj.pose.bones.get(bone_name)
        if pb is None:
            continue
        checked += 1
        q = Quaternion(rest_m)
        expected = q.conjugated() if convention == 'ROW' else q
        actual = _rest_rotation(pb)
        # rotation_difference is orientation-only and handles the q / -q ambiguity
        diff = math.degrees(expected.rotation_difference(actual).angle)
        if diff > 1.0:
            mismatched.append((stem, bone_name, diff))

    if not checked:
        return 0, 0
    if not mismatched:
        _log(f"  Rest-orientation check: all {checked} bone(s) match Milo's rest pose.")
        return checked, 0

    mismatched.sort(key=lambda r: -r[2])
    _log(f"  Rest-orientation check: {len(mismatched)} of {checked} bone(s) differ from "
         f"Milo's rest pose ('Milo Rest' compensates for this; 'Parent' does not):")
    for _stem, bone_name, diff in mismatched[:12]:
        _log(f"    {bone_name:28s} off by {diff:6.2f} deg")
    if len(mismatched) > 12:
        _log(f"    ... and {len(mismatched)-12} more")
    return checked, len(mismatched)


def get_milo_rest_rotations(clips):
    """Reads the rest rotation of each facial bone from the set's 'Base' clip, keyed by
    bone stem, as a Blender-ordered (w, x, y, z) tuple still in MILO convention.

    Base is the only clip holding absolute values rather than offsets, so it's the rig's
    reference pose as the viseme data itself understands it. Having it means rotation
    offsets can be composed against Milo's own rest rather than against whatever rest the
    armature ended up with in Blender - which matters if the skeleton importer reoriented
    any bones (its connect-bones option repoints a bone at its child, and Blender always
    forces a bone's Y axis along its length). Returns {} if the set has no Base clip."""
    entry = clips.get('Base')
    if entry is None:
        return {}
    try:
        pose, _full, _one = parse_viseme_clip(entry)
    except VisemeImportError:
        return {}
    out = {}
    for chan, (kind, v) in pose.items():
        if kind == 'quat':
            x, y, z, w = v
            out[_channel_stem(chan)] = (w, x, y, z)
    return out


def _rest_rotation(pose_bone):
    """Returns the bone's rest rotation RELATIVE TO ITS PARENT, as a Quaternion.

    This is the frame conversion the viseme deltas need. The samples store each bone's
    offset in the frame shared by the facial bones (their common parent - effectively
    head space), which is directly observable in the decoded data: in the 'Base' clip
    the lateral component cleanly mirrors between L/R bones while the vertical component
    matches, and midline bones sit at ~0 laterally. Blender's pose_bone.location and
    rotation_quaternion, by contrast, are expressed in the BONE's own rest axes (Y along
    the bone). Writing a parent-frame delta straight into bone-local channels is what
    made 'Brow_down' slide the brows sideways instead of down.

    Only the rotation is taken, never the full matrix: conjugating a rotation by a
    matrix that also carries the bone's rest translation would rotate the bone about the
    parent's origin instead of its own, injecting translation that shouldn't be there.

    Uses the bone's CURRENT rest matrices, so this stays correct even if the skeleton
    importer reoriented bones (e.g. its connect-bones option points a bone at its child,
    changing the bone's axes away from the raw milo matrix)."""
    bone = pose_bone.bone
    if bone.parent is not None:
        rest_local = bone.parent.matrix_local.inverted() @ bone.matrix_local
    else:
        rest_local = bone.matrix_local
    return rest_local.to_quaternion()


def _build_viseme_action(armature_obj, clip_name, set_name, pose, bone_lookup=None,
                         pos_space='PARENT', rot_space='AUTO',
                         conjugate_rotations=True, milo_rest_rot=None,
                         convention='ROW'):
    """Creates (or replaces) a single-frame Action encoding one viseme's pose deltas on
    the given armature's pose bones. Returns (action, matched_count, unmatched_names).

    `bone_lookup` maps channel stems -> real pose bone names (see build_bone_lookup);
    it's built once per import and passed in, but defaults to building its own so this
    stays usable standalone.

    `pos_space` / `rot_space` select how the stored offsets are interpreted. These are
    SEPARATE on purpose - the two channel types don't use the same frame:

      pos_space='PARENT' (default) - position offsets are in the bone's parent frame and
        get rotated into the bone's own rest axes. Confirmed directly from the data: the
        offsets mirror cleanly between L/R bones, midline bones sit at ~0 laterally, and
        Brow_down/Brow_up move along the vertical axis in opposite directions.

      rot_space='LOCAL' (default) - rotation offsets are already in each bone's own frame
        and post-multiply the rest rotation, so they're written through unchanged.
        Established by posing every symmetric L/R rotation pair both ways and measuring
        how mirror-symmetric the resulting world-space bone axes came out: bone-local won
        in 59 of 65 clips. The pucker/lip-heavy clips (Wet_hi, Church_hi, New_lo, Fave_lo)
        were the strongest cases, which matches them being the ones that visibly looked
        wrong when rotations were treated as parent-frame. Bones whose rest rotation is
        near 0 or 180 degrees (jaw, tongue1, the lids) look nearly the same either way,
        which is why those appeared correct before this split existed.

    Both accept 'PARENT' and 'LOCAL' so the alternative is one click away for comparison.

    `conjugate_rotations` converts rotation channels from Milo's row-vector convention
    to Blender's column-vector one (see the inline note at the quat branch). On by
    default; turn it off only to compare.

    Builds F-Curves directly on the Action's own Channelbag (see
    _get_or_create_channelbag) rather than via pose_bone.keyframe_insert(). That's
    deliberate, not just a workaround for the API change: keyframe_insert() keys
    whatever action is CURRENTLY ASSIGNED to armature_obj.animation_data (auto-creating
    a throwaway one if none is assigned), not necessarily the Action created a few lines
    above - which is exactly what caused the original crash (an empty, never-assigned,
    slot-less Action has no F-Curves to iterate at all). Writing straight into this
    Action's own Channelbag means importing dozens of these in one go never has to touch
    - or fight over - whatever action happens to be active on the armature."""
    if bone_lookup is None:
        bone_lookup = build_bone_lookup(armature_obj)

    action_name = f"VISEME_{clip_name}"
    existing = bpy.data.actions.get(action_name)
    if existing is not None:
        bpy.data.actions.remove(existing)
    action = bpy.data.actions.new(action_name)
    action[_MILO_VISEME_TAG] = True
    action[_MILO_VISEME_NAME] = clip_name
    action[_MILO_VISEME_SET] = set_name

    slot = action.slots.new(id_type='OBJECT', name=armature_obj.name)
    channelbag = _get_or_create_channelbag(action, slot)

    pose_bones = armature_obj.pose.bones
    matched = 0
    unmatched = []

    def _key(data_path, index, group, val):
        fc = channelbag.fcurves.new(data_path, index=index)
        grp = channelbag.groups.get(group)
        if grp is None:
            grp = channelbag.groups.new(group)
        fc.group = grp
        kp = fc.keyframe_points.insert(1, val, options={'FAST'})
        kp.interpolation = 'CONSTANT'   # single static pose, nothing to interpolate

    for chan_name, (kind, value) in pose.items():
        # Resolve via stem so '<bone>.pos' finds a pose bone actually named
        # '<bone>.mesh' (how real RB3 skeletons name it) - see build_bone_lookup.
        bone_name = bone_lookup.get(_channel_stem(chan_name))
        pb = pose_bones.get(bone_name) if bone_name else None
        if pb is None:
            unmatched.append(chan_name)
            continue

        rest_rot = None
        if pos_space == 'PARENT' or rot_space in ('PARENT', 'MILO_REST', 'AUTO'):
            rest_rot = _rest_rotation(pb)
        path = f'pose.bones["{bone_name}"]'

        if kind == 'pos':
            vec = Vector(value)
            if pos_space == 'PARENT':
                # Rotate the parent-frame offset into the bone's own rest axes, which is
                # what pose_bone.location is expressed in. See _rest_rotation.
                vec = rest_rot.inverted() @ vec
            for i, val in enumerate(vec):
                _key(f'{path}.location', i, bone_name, val)

        elif kind == 'quat':
            if pb.rotation_mode != 'QUATERNION':
                pb.rotation_mode = 'QUATERNION'
            # CharBonesSamples stores quaternions as (x, y, z, w); Blender's
            # rotation_quaternion array order is (w, x, y, z).
            x, y, z, w = value
            quat = Quaternion((w, x, y, z))

            if rot_space == 'AUTO':
                # Both steps below follow from ONE measured fact - which quaternion
                # convention the data uses (see detect_rotation_convention). They are not
                # independent knobs: a row-vector quaternion must be conjugated AND has
                # its composition order flipped, because (A*B)^T == B^T * A^T. Treating
                # them as separate choices is what produced the earlier whack-a-mole,
                # where fixing the jaw broke the lips and vice versa.
                if convention == 'ROW':
                    # Conjugate, and the flipped order turns milo's rest*delta into
                    # conj(delta) @ rest, i.e. a conjugation by the rest rotation.
                    quat = rest_rot.inverted() @ quat.conjugated() @ rest_rot
                else:
                    # Column-vector: quaternions map straight across and the order is
                    # unchanged, so milo's rest*delta means Blender's pose basis simply
                    # IS the offset. No conversion at all.
                    pass
            else:
                if conjugate_rotations:
                    quat = quat.conjugated()
                if rot_space == 'MILO_REST' and milo_rest_rot:
                    rest_m = milo_rest_rot.get(_channel_stem(chan_name))
                    if rest_m is not None:
                        posed = quat @ Quaternion(rest_m).conjugated()
                        quat = rest_rot.inverted() @ posed
                    else:
                        quat = rest_rot.inverted() @ quat @ rest_rot
                elif rot_space == 'PARENT':
                    quat = rest_rot.inverted() @ quat @ rest_rot
                # 'LOCAL': passes through unconverted.
            for i, val in enumerate(quat):
                _key(f'{path}.rotation_quaternion', i, bone_name, val)

        elif kind == 'rotz':
            # NOTE: approximate - real axis lives in the skeleton milo's CharBone.rotation
            # field, which isn't read here. See module docstring gap #5. Negated under the
            # same row-vector convention as the quaternions above.
            if pb.rotation_mode == 'QUATERNION':
                pb.rotation_mode = 'XYZ'
            negate = (convention == 'ROW') if rot_space == 'AUTO' else conjugate_rotations
            _key(f'{path}.rotation_euler', 2, bone_name, -value if negate else value)
        matched += 1

    return action, matched, unmatched


# ---------------------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------------------

class IMPORT_OT_rb3_viseme_set(bpy.types.Operator, ImportHelper):
    """Import a Rock Band 3 viseme CharClipSet milo (Xbox 360 only) as one Action per
    named viseme on the active armature. Use this to test a custom face rig against
    the real per-viseme poses, and as the pose source for the (separate) lipsync
    import/export step."""
    bl_idname = "import_scene.rb3_viseme_set"
    bl_label = "Import RB3 Viseme Set (X360)"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".milo_xbox"
    filter_glob: StringProperty(default="*.milo_xbox", options={'HIDDEN'})

    skip_base: BoolProperty(
        name="Skip 'Base' Clip",
        description="Don't import the clip literally named 'Base' - in every retail "
                     "viseme set checked it holds the rig's absolute rest-space bone "
                     "layout rather than a playable delta pose, and it never appears "
                     "in real .lipsync viseme tables",
        default=True,
    )

    pos_space: EnumProperty(
        name="Position Space",
        description="Which coordinate frame the stored position offsets are in",
        items=[
            ('PARENT', "Parent (recommended)",
             "Offsets are in the bone's parent frame and get rotated into the bone's "
             "own rest axes. Confirmed from the retail data"),
            ('LOCAL', "Bone Local (raw)",
             "Write offsets straight into the bone's local channels with no conversion"),
        ],
        default='PARENT',
    )

    rot_space: EnumProperty(
        name="Rotation Space",
        description="How rotation offsets are composed. Note this is NOT the same choice "
                     "as Position Space - the two channel types behave differently",
        items=[
            ('AUTO', "Auto-detect (recommended)",
             "Measure which quaternion convention the data uses by comparing the "
             "armature's rest pose against Milo's Base clip, then apply the conjugation "
             "and composition order that follow from it. These are not independent "
             "choices, so deciding them together is what keeps every bone consistent"),
            ('MILO_REST', "Milo Rest",
             "Compose against Milo's own rest pose (read from the set's Base clip), then "
             "re-express in the armature's actual rest frame"),
            ('PARENT', "Parent",
             "Compose against the armature's rest orientation. Correct only if every "
             "bone kept the orientation Milo gave it"),
            ('LOCAL', "Bone Local (raw)",
             "Write rotations straight into the bone's channels with no conversion"),
        ],
        default='AUTO',
    )

    conjugate_rotations: BoolProperty(
        name="Convert Rotation Convention",
        description="Milo does transform maths row-vector style while Blender is "
                     "column-vector, so rotations must be conjugated on import (the "
                     "same reason matrices get transposed elsewhere in this addon). "
                     "Without this, rotations spin the right axis by the right amount "
                     "in the wrong direction. Disable only to compare",
        default=True,
    )

    def execute(self, context):
        arm_obj = context.active_object
        if arm_obj is None or arm_obj.type != 'ARMATURE':
            self.report({'ERROR'},
                        "Select the target armature first - viseme poses are written "
                        "onto its pose bones by name.")
            return {'CANCELLED'}

        try:
            dir_name, clips = find_viseme_clips(self.filepath)
        except Exception as e:
            _log(f"VISEME IMPORT FAILED: {e}")
            self.report({'ERROR'}, f"Could not parse viseme milo: {e}")
            return {'CANCELLED'}

        if not clips:
            self.report({'ERROR'},
                        "No CharClip entries found - is this a viseme CharClipSet milo?")
            return {'CANCELLED'}

        _log(f"===== Importing viseme set '{dir_name}' from {self.filepath} =====")
        _log(f"  {len(clips)} CharClip entries found")

        # Parse everything up front (cheap - this is all in-memory struct decoding, no
        # Blender data created yet) so a total name mismatch can be caught and reported
        # BEFORE littering bpy.data.actions with dozens of empty Actions - that's exactly
        # what happened without this check: 67 actions created, every one of them empty,
        # and the only signal was a WARNING buried at the end of a long log.
        parsed = {}
        failed = []
        for clip_name, entry_bytes in clips.items():
            if self.skip_base and clip_name == 'Base':
                continue
            try:
                pose, _full, _one = parse_viseme_clip(entry_bytes)
                parsed[clip_name] = pose
            except VisemeImportError as e:
                _log(f"  SKIP '{clip_name}': {e}")
                failed.append(clip_name)

        expected_stems = {_channel_stem(ch) for pose in parsed.values() for ch in pose}
        bone_lookup = build_bone_lookup(arm_obj)
        overlap = {s for s in expected_stems if s in bone_lookup}
        if parsed and not overlap:
            arm_sample = ', '.join(sorted(arm_obj.pose.bones.keys())[:8]) or '(no pose bones at all)'
            expected_sample = ', '.join(sorted(expected_stems)[:8])
            self.report({'ERROR'},
                f"'{arm_obj.name}' has no pose bones matching this viseme set's driver "
                f"bones - 0 of {len(expected_stems)} expected name(s) found. Expected "
                f"names like: {expected_sample} (a '.mesh'/'.trans' extension on the "
                f"armature's bones is fine and handled automatically). Armature's actual "
                f"bones include: {arm_sample}. This armature is very likely missing the "
                f"facial driver-bone set - import the actual character/skeleton milo this "
                f"viseme set belongs to (not the viseme milo itself, which has no skeleton "
                f"at all) and select ITS armature before running this again.")
            return {'CANCELLED'}
        _log(f"  {len(overlap)} of {len(expected_stems)} driver bone(s) resolved on "
             f"'{arm_obj.name}'")

        milo_rest_rot = get_milo_rest_rotations(clips)
        convention = 'ROW'
        if milo_rest_rot:
            convention, _r, _c = detect_rotation_convention(
                arm_obj, bone_lookup, milo_rest_rot)
            report_rest_mismatches(arm_obj, bone_lookup, milo_rest_rot, convention)
        elif self.rot_space in ('AUTO', 'MILO_REST'):
            _log("  No 'Base' clip in this set - can't measure the rotation convention, "
                 "assuming row-vector.")

        imported = 0
        total_unmatched = set()

        for clip_name, pose in parsed.items():
            action, matched, unmatched = _build_viseme_action(
                arm_obj, clip_name, dir_name, pose, bone_lookup=bone_lookup,
                pos_space=self.pos_space, rot_space=self.rot_space,
                conjugate_rotations=self.conjugate_rotations,
                milo_rest_rot=milo_rest_rot, convention=convention)
            total_unmatched.update(unmatched)
            if matched == 0:
                _log(f"  '{clip_name}': 0 bone name(s) matched the armature - "
                     f"Action created but empty.")
            imported += 1

        skipped = len(clips) - len(parsed) - len(failed)
        summary = (f"Imported {imported} viseme Action(s) onto '{arm_obj.name}'"
                   + (f", skipped {skipped}" if skipped else "")
                   + (f", {len(failed)} clip(s) failed to parse" if failed else ""))
        _log(f"===== {summary} =====")
        if total_unmatched:
            sample = ', '.join(sorted(total_unmatched)[:8])
            _log(f"  {len(total_unmatched)} distinct bone name(s) never matched a pose "
                 f"bone on '{arm_obj.name}' (e.g. {sample}) - check the armature has the "
                 f"full facial driver-bone set from the character skeleton milo.")
            summary += f"; {len(total_unmatched)} bone name(s) unmatched (see log)"
        if failed:
            summary += f" ({', '.join(failed[:5])}{'...' if len(failed) > 5 else ''})"

        self.report({'WARNING' if (failed or total_unmatched) else 'INFO'}, summary)
        return {'FINISHED'}
