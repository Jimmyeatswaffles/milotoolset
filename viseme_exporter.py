"""
Base-viseme injection (EXPERIMENTAL).

Rewrites the "Base" clip inside a vanilla viseme milo so it describes the selected
armature's face instead of the original character's, leaving every other clip in the file
untouched.

===========================================================================================
WHY THIS WORKS - confirmed against the Dance Central 3 decompilation
===========================================================================================
Facial animation is not applied to a character's own rest pose. CharFaceServo::Poll() runs
every frame and does:

    if (mBaseClip) {
        TryScaleDown();
        ScaleAddIdentity();
        mBaseClip->RotateBy(*this, mBaseClip->StartBeat());
        PoseMeshes();
    }

with the base clip resolved BY NAME in the viseme directory:

    mBaseClip = mClips->Find<CharClip>("Base", false);

and the header documenting the directory as "pointer to visemes, must contain Blink and
Base". RotateBy bottoms out in CharBones::RotateBy, which is an ADD (`*otherVecItr += v`)
matching bones BY NAME. Viseme clips accumulate first via CharFaceServo::ScaleAdd, so the
final pose each frame is:

    face = Base + sum(weight_i * viseme_delta_i)

The authoring side is the mirror image: CharClip::Relativize() subtracts the relative
clip's first frame (`*it -= v`), which is why 57 of 58 clips in a retail viseme milo carry
relative = "Base", and why CharFaceServo::ScaleAdd warns "playing non-relative clip %s, cut
it out!" for any clip that doesn't.

So the viseme deltas are character-agnostic and only Base is character-specific. Patching
Base alone retargets the entire set - all 58 clips - to a new face. That is the whole
premise of this exporter, and it is engine behaviour, not inference.

Two consequences worth knowing:
  - Base is applied EVERY frame, not once at load, and if mBaseClip is null the entire
    Poll() body is skipped and the face never poses at all. Base is mandatory.
  - Because bones are matched by name, a Base channel naming a bone the rig doesn't have
    makes the engine's match loop hit TestDstComplain and bail out early. That's why this
    exporter refuses to write a partial Base rather than silently leaving stale values.

===========================================================================================
ENCODING - all verified byte-for-byte against retail TBRB PS3 files
===========================================================================================
Positions are stored as 3 x int16 at compression >= 2, normalised by 32767 and scaled by
1300.0. That constant appears in the RE-notes template and is confirmed by regenerating
George's vanilla Base from his vanilla skeleton: 104 of 108 position int16s come out
byte-exact, with the remainder off by at most 2 steps. Crucially the residual error is
uniform regardless of a bone's magnitude (bone_forehead at 4.124 units errs by 0.010,
bone_tongue2 at 0.493 errs by 0.017) - a wrong scale would produce error proportional to
value, so 1300 is right rather than merely close.

Rounding is round-half (floor(x + 0.5)); truncation scores far worse on the same test.

Positions come from the bone's LOCAL rest matrix (relative to its parent), and rotations
are the quaternion of that same local matrix in Blender's column-vector convention with NO
conjugation. Milo stores matrices row-vector style, so its stored quaternion equals the
quaternion of the TRANSPOSED milo matrix - which is exactly Blender's matrix, since
io.py's _milo_to_blender_matrix performs that transpose on import. Verified: 10 of 17
quaternion channels reproduce to four decimals.

The seven that don't are authored deviations in vanilla Base rather than encoding errors -
the eyelids sit ~15 degrees off the skeleton's bind pose, and the tongue chain drifts
progressively. Vanilla Base is a neutral FACE pose, not a restatement of the bind pose.
Regenerating it from an armature necessarily discards those tweaks, which is what
`positions_only` exists for.

Quantisation is 1300/32767 = 0.0397 units per step, so injected values land on that grid
(+/- 0.02 units) no matter how precise the armature is. Fine for faces; just not exact.

===========================================================================================
SCOPE
===========================================================================================
The Beatles: Rock Band, PS3 and Xbox 360. Both are big-endian and share the milo revision
25 / CharClipSamples layout, so the console option only affects the expected file
extension and reporting - the object stream itself is identical. The game option exists so
other titles can be added without reworking the operator.

A self-test is built in: run this against a VANILLA viseme milo using an armature imported
from that same character's VANILLA skeleton, and every reported delta should be at or
below one quantisation step. Anything larger means the armature's rest pose doesn't match
what the file expects, and the log says so per bone.
"""

import math
import os
import struct

import bpy
from bpy.props import StringProperty, BoolProperty, EnumProperty
from bpy_extras.io_utils import ImportHelper

from .io import read_milo_container_body
from .utilities import _log, MiloWriter
from .viseme_importer import (
    _Reader, _read_char_bones_samples, _locate_bone_samples,
    _channel_stem, build_bone_lookup, VisemeImportError,
)


# Position quantisation scale - see the encoding notes above.
POS_SCALE = 1300.0
MILO_END_MARKER = b'\xAD\xDE\xAD\xDE'
MILO_MAGIC_UNCOMPRESSED = 0xCABEDEAF
BASE_CLIP_NAME = "Base"


class BaseInjectError(Exception):
    """Anything that stops a Base clip being located or rewritten."""
    pass


def _q16(value):
    """Quantise a normalised (-1..1) float to int16 using round-half, matching the
    original encoder. Verified against retail data: round-half reproduces 104/108
    position int16s exactly where truncation manages only 74."""
    n = math.floor(value * 32767.0 + 0.5)
    return max(-32767, min(32767, int(n)))


# ---------------------------------------------------------------------------------------
# Locating the Base clip
# ---------------------------------------------------------------------------------------

def read_dir_entries(body):
    """Reads the CharClipSet directory table: (revision, dir_type, dir_name, [names])."""
    p = [0]

    def u32():
        v = struct.unpack_from('>I', body, p[0])[0]; p[0] += 4; return v

    def sym():
        n = u32()
        if n > 4096 or p[0] + n > len(body):
            raise BaseInjectError(f"implausible symbol length {n} in the directory table")
        s = body[p[0]:p[0] + n].decode('latin-1'); p[0] += n
        return s

    revision = u32()
    dir_type = sym()
    dir_name = sym()
    u32(); u32()                       # two counts, not needed here
    entry_count = u32()
    if entry_count > 65535:
        raise BaseInjectError(f"implausible entry count {entry_count}")
    entries = [(sym(), sym()) for _ in range(entry_count)]
    return revision, dir_type, dir_name, entries


def split_entries(body, entry_count):
    """Returns the byte span of each directory entry.

    The generic milo walker in io.py can't traverse revision-25 CharClipSets, so entries
    are delimited by their 0xADDEADDE end markers instead: the first marker closes the
    directory body and each subsequent one closes an entry. That's only safe if the count
    lines up exactly, so it's asserted - if an entry's payload ever contained the marker
    bytes the split would be silently wrong, and a mis-split would corrupt the file."""
    marks = []
    i = 0
    while True:
        j = body.find(MILO_END_MARKER, i)
        if j < 0:
            break
        marks.append(j)
        i = j + 4
    if len(marks) != entry_count + 1:
        raise BaseInjectError(
            f"found {len(marks)} end markers but expected {entry_count + 1} "
            f"(one closing the directory body plus one per entry). This file's layout "
            f"isn't what this exporter understands - refusing to touch it.")
    return [(marks[k] + 4, marks[k + 1] + 4) for k in range(entry_count)]


def locate_base_payload(body):
    """Finds the Base clip's sample payload.

    Returns a dict with the payload's absolute offset in `body`, the ordered channel
    names, the compression mode, and the existing decoded values (used for the
    before/after report)."""
    revision, dir_type, dir_name, entries = read_dir_entries(body)
    names = [n for _t, n in entries]
    if BASE_CLIP_NAME not in names:
        raise BaseInjectError(
            f"this milo has no clip named '{BASE_CLIP_NAME}' (it has {len(names)} "
            f"entries). The engine resolves the base pose by that exact name, so a file "
            f"without it has no base pose to replace.")

    spans = split_entries(body, len(entries))
    idx = names.index(BASE_CLIP_NAME)
    e_start, e_end = spans[idx]
    chunk = body[e_start:e_end]

    off, _full, one, _endp = _locate_bone_samples(chunk)

    # Re-walk the 'one' block header to find exactly where its sample payload begins.
    r = _Reader(chunk, off)
    _read_char_bones_samples(r)          # skip 'full'
    one_start = r.p
    r2 = _Reader(chunk, one_start)
    version = r2.i32()
    count_size = 7 if version > 15 else 10
    bone_count = r2.i32()
    bones = []
    for _ in range(max(bone_count, 0)):
        nm = r2.numstring()
        w = None if (version != -1 and version <= 10) else r2.f32()
        bones.append((nm, w))
    for _ in range(count_size):
        r2.u32()
    compression = r2.u32()
    num_samples = r2.u32()
    if version > 11:
        num_frames = r2.u32()
        for _ in range(num_frames):
            r2.f32()
    payload_start = r2.p

    if num_samples != 1:
        raise BaseInjectError(
            f"Base holds {num_samples} samples; expected exactly 1. The engine reads its "
            f"first frame only, so a multi-sample Base isn't something this writes.")
    if compression < 2:
        raise BaseInjectError(
            f"Base uses compression {compression} (uncompressed floats). Only the "
            f"int16-quantised layout (compression 2+) used by TBRB is implemented.")

    pos_ch = [n for n, _ in bones if n.endswith('.pos')]
    quat_ch = [n for n, _ in bones if n.endswith('.quat')]
    rotz_ch = [n for n, _ in bones if n.endswith('.rotz')]
    if rotz_ch:
        raise BaseInjectError(
            f"Base contains {len(rotz_ch)} .rotz channel(s), which this writer doesn't "
            f"handle - their per-bone axis lives in the skeleton milo and isn't read here.")

    payload_len = len(pos_ch) * 6 + len(quat_ch) * 8

    # Decode what's currently stored, for the before/after comparison.
    existing = {}
    q = e_start + payload_start
    for nm in pos_ch:
        x, y, z = struct.unpack_from('>hhh', body, q); q += 6
        existing[nm] = tuple(c / 32767.0 * POS_SCALE for c in (x, y, z))
    for nm in quat_ch:
        x, y, z, w = struct.unpack_from('>hhhh', body, q); q += 8
        existing[nm] = tuple(c / 32767.0 for c in (x, y, z, w))

    return dict(revision=revision, dir_name=dir_name,
                payload_offset=e_start + payload_start, payload_len=payload_len,
                pos_channels=pos_ch, quat_channels=quat_ch,
                compression=compression, existing=existing,
                entry_count=len(entries))


# ---------------------------------------------------------------------------------------
# Reading the armature
# ---------------------------------------------------------------------------------------

def bone_local_rest(pose_bone):
    """The bone's rest transform relative to its parent, as (translation, quaternion).

    This is the same quantity Milo stores per bone: local position in the parent's frame,
    and the local rotation. Blender's matrix is already column-vector, which is what Milo's
    stored quaternion corresponds to once its row-vector matrix is transposed - so no
    conjugation is applied here. Uses the armature's CURRENT rest data, so it stays correct
    even if the skeleton importer reoriented bones."""
    bone = pose_bone.bone
    if bone.parent is not None:
        local = bone.parent.matrix_local.inverted() @ bone.matrix_local
    else:
        local = bone.matrix_local
    q = local.to_quaternion()
    t = local.to_translation()
    return (t.x, t.y, t.z), (q.x, q.y, q.z, q.w)


def build_base_payload(armature_obj, info, positions_only=False):
    """Encodes a replacement payload for Base from the armature.

    Returns (payload_bytes, report_rows, missing_channels). Refuses nothing itself - the
    caller decides what to do about missing bones - but never invents a value: a channel
    with no matching bone keeps whatever the file already held, and is reported."""
    lookup = build_bone_lookup(armature_obj)
    pose_bones = armature_obj.pose.bones

    out = bytearray()
    rows = []
    missing = []

    def resolve(chan):
        bone_name = lookup.get(_channel_stem(chan))
        return pose_bones.get(bone_name) if bone_name else None

    for chan in info['pos_channels']:
        pb = resolve(chan)
        old = info['existing'][chan]
        if pb is None:
            missing.append(chan)
            new = old                       # keep the file's value untouched
        else:
            new, _q = bone_local_rest(pb)
        out += struct.pack('>hhh', *[_q16(c / POS_SCALE) for c in new])
        rows.append((chan, old, new, max(abs(a - b) for a, b in zip(old, new))))

    for chan in info['quat_channels']:
        pb = resolve(chan)
        old = info['existing'][chan]
        if pb is None:
            missing.append(chan)
            new = old
        elif positions_only:
            new = old
        else:
            _t, new = bone_local_rest(pb)
            # q and -q are the same rotation; keep the sign closest to what's stored so
            # the before/after diff reflects real change rather than a sign flip.
            if sum(a * b for a, b in zip(old, new)) < 0:
                new = tuple(-c for c in new)
        out += struct.pack('>hhhh', *[_q16(c) for c in new])
        rows.append((chan, old, new, max(abs(a - b) for a, b in zip(old, new))))

    if len(out) != info['payload_len']:
        raise BaseInjectError(
            f"encoded {len(out)} bytes but the file's Base payload is "
            f"{info['payload_len']} - refusing to write a mismatched block.")
    return bytes(out), rows, missing


# ---------------------------------------------------------------------------------------
# Container write-back
# ---------------------------------------------------------------------------------------

def rebuild_uncompressed(body):
    """Wraps a patched body in an uncompressed milo container.

    Block sizes are recomputed from the body rather than carried over, because a
    compressed source decompresses into a single contiguous stream. The patch never
    changes the body's length, so for an already-uncompressed single-block file this
    reproduces the original layout exactly."""
    START_OFFSET = 0x810
    header = MiloWriter(big_endian=False)
    header.u32(MILO_MAGIC_UNCOMPRESSED)
    header.u32(START_OFFSET)
    header.u32(1)
    header.u32(len(body))
    header.u32(len(body))
    header.block(bytes(START_OFFSET - len(header.buf)))
    return bytes(header.buf) + bytes(body)


# ---------------------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------------------

class EXPORT_OT_milo_inject_base_viseme(bpy.types.Operator, ImportHelper):
    """Rewrite the Base clip inside a vanilla viseme milo so it matches the selected
    armature's face. Every other clip is left byte-for-byte untouched"""
    bl_idname = "export_scene.milo_inject_base_viseme"
    bl_label = "Inject New Base Viseme (Experimental)"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ""
    filter_glob: StringProperty(
        default="*.milo_ps3;*.milo_xbox", options={'HIDDEN'})

    game: EnumProperty(
        name="Game",
        description="Which title's viseme milo is being patched",
        items=[
            ('TBRB', "The Beatles: Rock Band",
             "Milo revision 25, CharClipSamples entries, int16-quantised samples"),
        ],
        default='TBRB',
    )

    console: EnumProperty(
        name="Console",
        description="Target console. Both are big-endian and share an identical object "
                     "stream, so this only affects the output file extension",
        items=[
            ('PS3', "PlayStation 3", "Writes .milo_ps3"),
            ('X360', "Xbox 360", "Writes .milo_xbox"),
        ],
        default='PS3',
    )

    positions_only: BoolProperty(
        name="Positions Only (keep vanilla rotations)",
        description="Replace only the position channels and leave the original rotations "
                     "alone. Vanilla Base is an authored neutral FACE pose, not a copy of "
                     "the bind pose - its eyelids sit about 15 degrees off and the tongue "
                     "chain is deliberately offset. Enable this to keep those authored "
                     "tweaks and fix only the bone placement",
        default=False,
    )

    overwrite: BoolProperty(
        name="Overwrite Source File",
        description="Write back over the selected file. When off, the patched milo is "
                     "saved alongside it with a _custombase suffix, leaving the vanilla "
                     "file intact",
        default=False,
    )

    def execute(self, context):
        arm_obj = context.active_object
        if arm_obj is None or arm_obj.type != 'ARMATURE':
            self.report({'ERROR'},
                        "Select the armature whose face should become the new Base.")
            return {'CANCELLED'}

        try:
            with open(self.filepath, 'rb') as f:
                raw = f.read()
            body = bytearray(read_milo_container_body(raw))
            info = locate_base_payload(bytes(body))
        except (BaseInjectError, VisemeImportError, ValueError) as e:
            _log(f"BASE INJECTION FAILED: {e}")
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        except Exception as e:
            _log(f"BASE INJECTION FAILED: {e}")
            self.report({'ERROR'}, f"Could not read this milo: {e}")
            return {'CANCELLED'}

        _log(f"===== Injecting Base viseme into {self.filepath} =====")
        _log(f"  {self.game} / {self.console}: milo revision {info['revision']}, "
             f"CharClipSet '{info['dir_name']}', {info['entry_count']} entries")
        _log(f"  Base payload: {info['payload_len']} bytes at body offset "
             f"{info['payload_offset']} "
             f"({len(info['pos_channels'])} pos, {len(info['quat_channels'])} quat, "
             f"compression {info['compression']})")

        try:
            payload, rows, missing = build_base_payload(
                arm_obj, info, positions_only=self.positions_only)
        except BaseInjectError as e:
            _log(f"BASE INJECTION FAILED: {e}")
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        if missing:
            # A Base naming a bone the rig lacks makes the engine's name-matching loop
            # bail out early (TestDstComplain), so a partial Base is worse than none.
            sample = ', '.join(sorted(set(missing))[:10])
            msg = (f"{len(set(missing))} of "
                   f"{len(info['pos_channels']) + len(info['quat_channels'])} Base "
                   f"channel(s) have no matching bone on '{arm_obj.name}': {sample}"
                   + (" ..." if len(set(missing)) > 10 else "")
                   + ". The engine matches face bones by name and stops at the first "
                     "mismatch, so nothing was written. Add these bones to the rig, or "
                     "use a rig exported for this game.")
            _log(f"BASE INJECTION ABORTED: {msg}")
            self.report({'ERROR'}, msg)
            return {'CANCELLED'}

        step = POS_SCALE / 32767.0
        _log(f"  quantisation step {step:.5f} units; changes below {step/2:.5f} are "
             f"rounding, not real movement")
        moved = [r for r in rows if r[0].endswith('.pos') and r[3] > step / 2]
        _log(f"  {len(moved)} of {len(info['pos_channels'])} position channel(s) moved "
             f"beyond one quantisation step:")
        for chan, old, new, d in sorted(moved, key=lambda r: -r[3])[:14]:
            _log(f"    {chan:26s} ({old[0]:7.3f},{old[1]:7.3f},{old[2]:7.3f}) -> "
                 f"({new[0]:7.3f},{new[1]:7.3f},{new[2]:7.3f})  delta {d:6.3f}")
        if len(moved) > 14:
            _log(f"    ... and {len(moved)-14} more")
        if not moved:
            _log("    (none - this armature's face already matches the file's Base. "
                 "That's the expected result when self-testing with a vanilla rig.)")

        if self.positions_only:
            _log("  Rotations left at their vanilla values (Positions Only enabled).")
        else:
            qmoved = [r for r in rows if r[0].endswith('.quat') and r[3] > 0.01]
            _log(f"  {len(qmoved)} of {len(info['quat_channels'])} rotation channel(s) "
                 f"changed materially.")

        # Splice the new payload in. Length is identical by construction, so no offset in
        # the file shifts and every other clip stays byte-for-byte as it was.
        o = info['payload_offset']
        body[o:o + info['payload_len']] = payload

        ext = '.milo_ps3' if self.console == 'PS3' else '.milo_xbox'
        if self.overwrite:
            out_path = self.filepath
        else:
            stem = self.filepath
            for known in ('.milo_ps3', '.milo_xbox'):
                if stem.lower().endswith(known):
                    stem = stem[:-len(known)]
                    break
            out_path = stem + '_custombase' + ext

        try:
            with open(out_path, 'wb') as f:
                f.write(rebuild_uncompressed(body))
        except OSError as e:
            _log(f"BASE INJECTION FAILED: could not write {out_path}: {e}")
            self.report({'ERROR'}, f"Could not write output: {e}")
            return {'CANCELLED'}

        summary = (f"Base injected from '{arm_obj.name}' -> {os.path.basename(out_path)} "
                   f"({len(moved)} bone(s) repositioned)")
        _log(f"===== {summary} =====")
        self.report({'INFO'}, summary)
        return {'FINISHED'}
