bl_info = {
    "name": "Milo Scene Exporter",
    "author": "jimmyeatwaffles",
    "version": (0, 2, 0),
    "blender": (5, 0, 0),
    "location": "File > Export > Milo Scene (.milo)",
    "description": "3D model expoter for Harmonix games using the milo archive format. Based on the gltfmilo CLI program by ihatecompvir.",
    "category": "Import-Export",
}

import bpy
import struct
from bpy.props import (
    StringProperty,
    BoolProperty,
    EnumProperty,
    IntProperty,
    FloatProperty,
    FloatVectorProperty,
    PointerProperty,
    CollectionProperty,
)
from bpy_extras.io_utils import ExportHelper, ImportHelper


# =====================================================================================
# Console logging
#
# Blender's operator report() (self.report in EXPORT_OT_milo_scene) shows messages in
# the Info editor / status bar, but requested logging is for the Blender system console
# (Window > Toggle System Console) - the same place print() output shows up. All export
# stages (mesh, texture, material, armature/bone) log a start line before doing real
# work and a result line with the relevant specs after, so a failed/hung export can be
# diagnosed from the console alone without needing to reproduce with breakpoints.
# =====================================================================================

def _log(msg):
    print(f"[Milo Export] {msg}")


_TEX_ENCODING_NAMES = {}  # populated after the TEX_ENCODING_* constants are defined below


# =====================================================================================
# SECTION 1 -- low-level binary writer
#
# Ported from MiloLib/Utils/Endian/EndianWriter.cs. Milo files have a little-endian
# header and a big-endian body (on every console/platform MiloLib supports, including
# Xbox 360 and PS3 - both are big-endian PowerPC).
# =====================================================================================

class MiloWriter:
    def __init__(self, big_endian=True):
        self.buf = bytearray()
        self.big_endian = big_endian

    def _fmt(self, base):
        return (">" if self.big_endian else "<") + base

    def u8(self, v):
        self.buf += struct.pack("B", v & 0xFF)

    def i16(self, v):
        self.buf += struct.pack(self._fmt("h"), v)

    def u16(self, v):
        self.buf += struct.pack(self._fmt("H"), v & 0xFFFF)

    def u32(self, v):
        self.buf += struct.pack(self._fmt("I"), v & 0xFFFFFFFF)

    def i32(self, v):
        self.buf += struct.pack(self._fmt("i"), v)

    def f32(self, v):
        self.buf += struct.pack(self._fmt("f"), v)

    def f16(self, v):
        self.buf += struct.pack(self._fmt("e"), v)

    def boolean(self, v):
        self.u8(1 if v else 0)

    def block(self, data: bytes):
        self.buf += data

    def symbol(self, s: str):
        # Symbol.Write: uint32 length prefix, then Latin-1 bytes, NO null terminator
        data = (s or "").encode("latin-1", errors="replace")
        self.u32(len(data))
        self.block(data)

    def __len__(self):
        return len(self.buf)


# =====================================================================================
# SECTION 2 -- shared struct helpers (Matrix, Sphere, compressed vertex formats)
# Ported from MiloLib/Classes/Matrix.cs, Sphere.cs and Assets/Rnd/Vertex.cs
# =====================================================================================

def write_matrix(w: MiloWriter, m12):
    """m12: iterable of 12 floats -> m11,m12,m13, m21,m22,m23, m31,m32,m33, m41,m42,m43"""
    for f in m12:
        w.f32(f)


IDENTITY_MATRIX = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)


def write_sphere(w: MiloWriter, x, y, z, radius):
    w.f32(x)
    w.f32(y)
    w.f32(z)
    w.f32(radius)


def write_color3(w: MiloWriter, r, g, b):
    w.f32(r)
    w.f32(g)
    w.f32(b)


def write_color4(w: MiloWriter, r, g, b, a):
    w.f32(r)
    w.f32(g)
    w.f32(b)
    w.f32(a)


def _to_snorm_bits(f, n):
    f = max(-1.0, min(1.0, f))
    maxv = (1 << (n - 1)) - 1
    s = int(f * maxv)  # truncation towards zero, matches C# (int) cast
    if s < 0:
        s += (1 << n)
    return s & ((1 << n) - 1)


def pack_signed_10_10_10_2(x, y, z, w):
    """Xbox 360 compressed vec4 (SignedCompressedVec4): used for normals and tangents."""
    xb = _to_snorm_bits(x, 10)
    yb = _to_snorm_bits(y, 10)
    zb = _to_snorm_bits(z, 10)
    wb = _to_snorm_bits(w, 2)
    return (xb) | (yb << 10) | (zb << 20) | (wb << 30)


def pack_unsigned_10_10_10_2(x, y, z, w):
    """Xbox 360 compressed vec4 (UnsignedCompressedVec4): used for bone weights."""
    xb = int(max(0.0, min(1.0, x)) * 1023.0) & 0x3FF
    yb = int(max(0.0, min(1.0, y)) * 1023.0) & 0x3FF
    zb = int(max(0.0, min(1.0, z)) * 1023.0) & 0x3FF
    wb = int(max(0.0, min(1.0, w)) * 3.0) & 0x003
    return xb | (yb << 10) | (zb << 20) | (wb << 30)


def pack_ps3_signed_11_11_10(x, y, z):
    """PS3 compressed vec3 (PS3SignedCompressedVec3): used for normals and tangents.
    Confirmed from MiloLib/Assets/Rnd/Vertex.cs: 11-bit x, 11-bit y, 10-bit z, signed
    (snorm), packed into a u32 as x | (y<<11) | (z<<22). No W component (unlike the
    Xbox 10-10-10-2 format), which is part of why the PS3 vertex is a different size."""
    xb = _to_snorm_bits(x, 11)
    yb = _to_snorm_bits(y, 11)
    zb = _to_snorm_bits(z, 10)
    return (xb & 0x7FF) | ((yb & 0x7FF) << 11) | ((zb & 0x3FF) << 22)


def pack_ps3_unsigned_11_11_10(x, y, z):
    """PS3 compressed vec3 (PS3UnsignedCompressedVec3): used for bone weights.
    Confirmed from MiloLib/Assets/Rnd/Vertex.cs: 11-bit x, 11-bit y, 10-bit z, unsigned.
    Note the scale factors differ per-channel exactly as MiloLib does it - x and y scale
    by 1023 (into 11-bit fields), z by 511 (into the 10-bit field)."""
    xb = int(max(0.0, min(1.0, x)) * 1023.0) & 0x7FF
    yb = int(max(0.0, min(1.0, y)) * 1023.0) & 0x7FF
    zb = int(max(0.0, min(1.0, z)) * 511.0) & 0x3FF
    return xb | (yb << 11) | (zb << 22)


# =====================================================================================
# SECTION 3 -- RB3 revision constants
# Ported from Source/Shared/GameRevisions.cs (RockBand3 entry only)
# =====================================================================================

RB3_MILO_REVISION = 28      # DirectoryMeta.revision (RB3 / DC1)
DC3_MILO_REVISION = 32      # DirectoryMeta.revision (Dance Central 3). Byte-identical to
                             # rev 28 except for one extra byte (observed 0x00) written after
                             # stringTableSize and before entryCount - confirmed by parsing real
                             # DC3 files at both the top level and the nested CharClipSet dir.
                             # Matching the container to the rev-32 CharClipSet is what stops
                             # DC3 misreading the clips (the "CharClip version 38 > 22" crash).
RB3_OBJDIR_REVISION = 27    # ObjectDir.revision
RB3_RNDDIR_REVISION = 10    # RndDir.revision
RB3_TRANS_REVISION = 9      # RndTrans.revision
RB3_DRAW_REVISION = 3       # RndDrawable.revision
DC3_DRAW_REVISION = 4       # RndDrawable.revision (Dance Central 3). Confirmed by byte-parsing
                             # every embedded RndDrawable in retail angel01.milo_xbox (9 meshes
                             # checked, every one): the revision field reads 4, not RB3's 3, and
                             # revision >= 4 adds ONE extra empty Symbol immediately after
                             # drawOrder (before whatever the caller writes next - e.g. RndMesh's
                             # `mat` field). Every one of the 9 sampled retail meshes has this
                             # field empty, so it's written as an empty symbol here too; nothing
                             # in the exporter's Blender-side data model maps to it yet. Writing
                             # DC3 Drawables at rev 3 (RB3's value) would desync every field that
                             # follows in the SAME asset for a version-aware reader - RndMesh's
                             # `mat`/geomOwner names land one symbol early, which is exactly what
                             # byte-parsing retail's meshes at rev-3 assumptions produced (mat
                             # read as empty, geomOwner read as the real material name, and a
                             # trailing real mesh-name symbol left over).
RB3_ANIM_REVISION = 4       # RndAnimatable.revision
RB3_MESH_REVISION = 38      # RndMesh.revision
OBJ_FIELDS_REVISION = 2     # Hmx::Object's own "note" revision (set explicitly by the original tool)

MAX_MILO_BLOCK_SIZE = 0x20000  # Confirmed directly from MiloLib's MiloFile.cs, which
                                # cites it as what "DirLoader::SaveObjects uses" (the
                                # real engine's own save routine). After each top-level
                                # entry finishes writing, if the number of bytes written
                                # since the last block boundary exceeds this, a new block
                                # boundary is recorded - so a single large asset (like a
                                # big DXT texture) just becomes its own block. This was
                                # missing entirely from this file: build_milo_bytes always
                                # wrote the whole body as ONE block regardless of size.
                                # Byte-for-byte comparison against a real CLI-exported
                                # reference file confirmed every individual asset (mesh,
                                # material, texture) was essentially the same size in both
                                # exports - the only structural difference was block count
                                # (CLI: 15 properly-sized blocks: this exporter: always 1,
                                # ballooning past 37MB once Ignore Texture Size Limits
                                # allowed 2048x2048 textures). MiloLib's own C# reader
                                # ignores blockSizes entirely for Type.Uncompressed (it
                                # just seeks to startOffset and reads through), so this
                                # metadata is inert as far as MiloEditor is concerned -
                                # but the CLI tool (known to load correctly on real
                                # hardware/Xenia) always produces it anyway, and the
                                # real Xbox 360 engine loader (not reflected in this Wii
                                # decompilation) most likely uses it for bounded/chunked
                                # reads or scratch-buffer allocation sizing during load,
                                # which a single 37MB block would blow past - matching
                                # the "exports fine, crashes on load" symptom exactly.

END_MARKER = b"\xAD\xDE\xAD\xDE"


# =====================================================================================
# SECTION 4 -- asset writers
# Ported from MiloLib/Assets/{Object.cs, ObjectDir.cs, Rnd/*.cs}
# Only the exact branch RB3 Xbox 360 takes at each fixed revision is implemented -
# the originals are heavily revision-gated for 20+ years of Milo format history,
# which we don't need to replicate for a single fixed target.
# =====================================================================================

def write_object_fields(w: MiloWriter, revision=OBJ_FIELDS_REVISION, alt_revision=0,
                         obj_type="", note=""):
    """Object.cs -> ObjectFields.Write()"""
    w.u32((alt_revision << 16) | revision)
    w.symbol(obj_type)
    w.boolean(False)  # DTBParent.hasTree
    if revision > 0:
        w.symbol(note)


def write_rnd_animatable(w: MiloWriter):
    """Assets/Rnd/RndAnimatable.cs, fixed at revision 4 (RB3)."""
    w.u32(RB3_ANIM_REVISION)
    w.f32(0.0)      # frame
    w.u32(0)        # rate (k30_fps)


def write_rnd_drawable(w: MiloWriter, showing=True, sphere=(0.0, 0.0, 0.0, 0.0), draw_order=0.0,
                        revision=RB3_DRAW_REVISION):
    """Assets/Rnd/RndDrawable.cs, fixed at revision 3 for RB3/DC1. Always called with
    skipMetadata=True by every caller, so objFields is never written here.

    `revision`: pass DC3_DRAW_REVISION (4) for Dance Central 3 - see that constant's
    comment. At revision >= 4, one extra empty Symbol is written after drawOrder (real
    DC3 files always leave it empty; nothing here authors it yet)."""
    w.u32(revision)
    w.boolean(showing)
    write_sphere(w, *sphere)
    w.f32(draw_order)
    if revision >= 4:
        w.symbol("")   # DC3-only extra field - see DC3_DRAW_REVISION


def write_rnd_trans(w: MiloWriter, local_xfm=IDENTITY_MATRIX, world_xfm=IDENTITY_MATRIX,
                     parent_obj="", standalone=False, skip_metadata=True):
    """Assets/Rnd/RndTrans.cs, fixed at revision 9 (RB3). Every existing embedded use
    in this file passes standalone=False (skipMetadata is irrelevant in that case,
    since the C# only writes objFields when standalone AND NOT skipMetadata are both
    true). standalone=True is for the new experimental "export armature as real Trans
    entries" feature - DirectoryMeta's own dispatch for top-level "Trans" entries
    calls Write(standalone=True, skipMetadata=False), so a real bone needs its own
    metadata written where an embedded mesh/drawable's trans does not."""
    w.u32(RB3_TRANS_REVISION)
    if standalone and not skip_metadata:
        write_object_fields(w)
    write_matrix(w, local_xfm)
    write_matrix(w, world_xfm)
    w.u32(0)            # constraint (kConstraintNone)
    w.symbol("")        # target
    w.boolean(False)    # preserveScale
    w.symbol(parent_obj)
    if standalone:
        w.block(END_MARKER)


# =====================================================================================
# CharMeshHide support (the ".hide" asset type)
#
# Confirmed directly against MiloLib's C# implementation (ihatecompvir/MiloEditor,
# MiloLib/Assets/Char/CharMeshHide.cs) AND independently verified byte-for-byte against
# a real reference file (facehair_bandana.hide - decoded by hand: revision=2, top-level
# flags=0, one Hide entry referencing "male_bandana.mesh" with flags=128/HideMask,
# show=False). Two DIFFERENT flags fields exist here and it's easy to conflate them:
#   - CharMeshHide's own top-level `flags` (CharMeshHideOptions below) - a bitmask of
#     hide-categories this asset itself requests/represents (e.g. HIDE_HEAD). This is
#     what MiloEditor's "Hide Head" checkbox actually edits.
#   - Each individual Hide entry's own `flags` - which of those same category bits
#     cause THAT entry's named mesh to be hidden, checked via HideDraws(x):
#       showing = hide.show != !(x & hide.flags)
#     i.e. an entirely separate per-mesh condition, unrelated to the top-level flags
#     field above unless something upstream deliberately feeds one into the other.
#
# IMPORTANT CAVEAT: grepping the actual RB3 Wii decompilation (DarkRTA/rb3) turns up
# no references to a "hide head" concept anywhere in the engine source at all - not in
# BandCharacter, CharMeshHide's own call sites, or anywhere else. This flag exists in
# the shared Milo format/tooling (used across multiple Harmonix games, some on
# different engines - RB4 is the one known to actually use head-hiding accessories),
# but nothing found so far in RB3 itself appears to read HIDE_HEAD. This export option
# exists to test that empirically on real hardware/Xenia rather than to assert it does
# anything in RB3 - see the operator tooltip.
# =====================================================================================

RB3_CHARMESHHIDE_REVISION = 2  # CharMeshHide.revision - matches the reference file exactly


class CharMeshHideOptions:
    """CharMeshHide.HideOptions, confirmed directly from MiloLib's C# enum - the exact
    bitmask values used both for CharMeshHide's own top-level `flags` field and (as
    far as we know so far) any per-Hide-entry `flags` values that are meant to key off
    those same categories (e.g. the reference file's own Hide entry uses HIDE_MASK)."""
    NONE = 0
    HIDE_LONG_COAT = 1
    HIDE_LONG_DRESS = 2
    HIDE_LONG_SLEEVE = 4
    HIDE_MID_SLEEVE = 8
    HIDE_FULL_SLEEVE = 16
    HIDE_HEAD = 32
    HIDE_LONG_GLOVE = 64
    HIDE_MASK = 128
    HIDE_LONG_BOOT = 256
    HIDE_SHORT_SLEEVE = 512
    HIDE_SHORT_BOOT = 1024
    HIDE_LONG_PANTS = 2048
    HIDE_GLASSES = 4096
    HIDE_VIGNETTE = 8192
    HIDE_SOCKS = 16384


def write_char_mesh_hide(w: MiloWriter, flags=0, hides=None, standalone=True):
    """Assets/Char/CharMeshHide.cs Write(), fixed at revision 2 (matches the reference
    file's own revision exactly). `hides` is a list of (draw_name, hide_flags, show)
    tuples for the asset's Hide list - `draw_name` is just a Symbol (ObjPtr<RndDrawable>
    reference), so nothing here restricts it to a mesh embedded in this same milo; once
    everything is merged into one runtime ObjectDir by the file-merger system, a name
    like "head.mesh" from a different, separately-loaded piece file should resolve the
    same way any other cross-entry symbol reference does elsewhere in this format (mat
    names, parent bone names, etc). Not exercised yet - hides is left empty by the
    current export option, which only sets the top-level `flags` bitmask."""
    if hides is None:
        hides = []
    w.u32(RB3_CHARMESHHIDE_REVISION)   # (altRevision=0 << 16) | 2

    write_object_fields(w)             # base.Write(standalone=False) -> objFields

    w.i32(flags)
    w.u32(len(hides))
    for (draw_name, hide_flags, show) in hides:
        w.symbol(draw_name)
        w.i32(hide_flags)
        if RB3_CHARMESHHIDE_REVISION > 1:   # always true at revision 2, mirrors the
            w.boolean(show)                 # C#'s own `if (revision > 1)` gate exactly

    if standalone:
        w.block(END_MARKER)


# =====================================================================================
# CharHair support (the ".hair" asset type - hair/cloth physics)
#
# Format confirmed against BOTH MiloLib's C# (MiloEditor/MiloLib/Assets/Char/CharHair.cs)
# and the RB3 Wii decompilation (DarkRTA/rb3 src/system/char/CharHair.cpp + .h). RB3
# uses CharHair revision 11 (ASSERT_REVS(11, 0) in the decomp's CharHair::Load), which
# determines exactly which per-Point fields are live - several fields present at other
# revisions (addToRadius, the pre-rev-3 unkInt/unkSym, the rev<9 bool before sideLength)
# are NOT written at revision 11, and getting that gating wrong would silently corrupt
# every following byte.
#
# CharHair extends RndHighlightable, but that's a pure-virtual passthrough that adds no
# serialized fields (confirmed: the decomp's CharHair::Load calls Hmx::Object::Load
# directly, and MiloLib's base.Read is just objFields) - so the body is simply objFields
# followed by the CharHair fields.
#
# Field order at revision 11 (from both sources):
#   objFields
#   stiffness, torsion, inertia, gravity, weight, friction   (6 floats)
#   minSlack, maxSlack                                        (2 floats, since rev>8)
#   strandCount (u32), then each Strand:
#     root (Symbol), angle (float), pointCount (u32), then each Point:
#       pos (Vector3 = 3 floats)
#       bone (Symbol)
#       length (float)
#       radius (float)
#       outerRadius (float, since rev>1)
#       sideLength (float, since rev>=8; NO leading bool since rev>=9)
#       unk5c (Vector3 = 3 floats, since rev>9)
#     baseMat (Matrix3 = 9 floats), rootMat (Matrix3 = 9 floats)
#     hookupFlags (int, since rev>2)
#   simulate (bool)
#   wind (Symbol, since rev>10)
#
# NOTE: SetRoot() would recompute pos/length/baseMat/rootMat from the bone chain
# (CharHair.cpp Strand::SetRoot: length = bone's local Y translation, pos = bone's world
# position, baseMat = root bone's local rotation) - BUT it is only called from the editor
# sync path, NOT at load. The decomp's Hookup() (which DOES run at load) only wires up
# collisions; it never touches the strand matrices. So everything we write here - the
# matrices especially - is used DIRECTLY by the runtime sim and must be correct. In
# particular baseMat/rootMat MUST equal the root bone's own Trans local rotation (verified
# byte-for-byte against a vanilla RB3 .hair); getting them wrong twists every strand.
# We still populate them from Blender's bone rest transforms so the file is valid and
# self-consistent even before the first Hookup.
# =====================================================================================

RB3_CHARHAIR_REVISION = 11  # ASSERT_REVS(11, 0) in the RB3 decomp's CharHair::Load
DEFAULT_HAIR_WIND = "world.wind"  # every real .hair file observed uses this wind object


def write_vector3(w: MiloWriter, x, y, z):
    w.f32(x)
    w.f32(y)
    w.f32(z)


def write_matrix3(w: MiloWriter, m9):
    """Hmx::Matrix3 - 9 floats, row-major (m11,m12,m13, m21,m22,m23, m31,m32,m33).
    Distinct from write_matrix (the 4x3/12-float RndTrans matrix)."""
    for f in m9:
        w.f32(f)


IDENTITY_MATRIX3 = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def write_char_hair(w: MiloWriter, settings, strands, standalone=True):
    """Assets/Char/CharHair.cs Write(), fixed at revision 11 (RB3). `settings` is a dict
    with keys: stiffness, torsion, inertia, gravity, weight, friction, min_slack,
    max_slack, simulate, wind. `strands` is a list of strand dicts, each with:
      root:       Symbol (str) - the root Trans/bone for this strand
      angle:      float - starting flip angle in degrees
      base_mat:   9-float Matrix3 (root bone's local rotation)
      root_mat:   9-float Matrix3 (base_mat rotated about X by `angle`; for angle 0 it
                  equals base_mat. Used DIRECTLY by the runtime sim - not recomputed at
                  load - so it must be the bone's real local rotation, not identity)
      hookup_flags: int
      points:     list of point dicts, each with:
        pos:         (x,y,z) world position of the bone
        bone:        Symbol (str) - the hair bone this point controls
        length:      float - bone's local-space length (local Y translation)
        radius:      float - collision radius
        outer_radius:float - flatten-against-collision distance
        side_length: float - base side length (>= 0 enables cloth-style sides;
                     -1.0 disables, matching the engine's own default)
    """
    w.u32(RB3_CHARHAIR_REVISION)   # (altRevision=0 << 16) | 11

    write_object_fields(w)         # base.Write(standalone=False) -> objFields

    w.f32(settings.get('stiffness', 0.1))
    w.f32(settings.get('torsion', 0.1))
    w.f32(settings.get('inertia', 0.65))
    w.f32(settings.get('gravity', 1.0))
    w.f32(settings.get('weight', 0.5))
    w.f32(settings.get('friction', 0.0))
    # revision(11) > 8: True
    w.f32(settings.get('min_slack', 0.0))
    w.f32(settings.get('max_slack', 0.0))

    w.u32(len(strands))
    for strand in strands:
        w.symbol(strand['root'])
        w.f32(strand.get('angle', 0.0))
        points = strand['points']
        w.u32(len(points))
        for pt in points:
            write_vector3(w, *pt['pos'])
            w.symbol(pt['bone'])
            w.f32(pt['length'])
            w.f32(pt.get('radius', 0.0))
            # revision(11) > 1: True
            w.f32(pt.get('outer_radius', -1.0))
            # revision(11) >= 8: True -> sideLength; revision >= 9 so NO leading bool
            w.f32(pt.get('side_length', -1.0))
            # revision(11) > 9: True -> unk5c (zeroed; engine recomputes)
            write_vector3(w, 0.0, 0.0, 0.0)
        write_matrix3(w, strand.get('base_mat', IDENTITY_MATRIX3))
        write_matrix3(w, strand.get('root_mat', IDENTITY_MATRIX3))
        # revision(11) > 2: True
        w.i32(strand.get('hookup_flags', 0))

    w.boolean(settings.get('simulate', True))

    # revision(11) > 10: True
    w.symbol(settings.get('wind', DEFAULT_HAIR_WIND))

    if standalone:
        w.block(END_MARKER)


# =====================================================================================
# OutfitConfig support (the ".cfg" asset type)
#
# Instruments (and character outfits) need an OutfitConfig or the game HARD-CRASHES on
# select: ClosetMgr::UpdateCurrentOutfitConfig() looks up "guitar.cfg" by name and gets
# null when it's missing (just a warning so far), but then ChooseColorPanel::Load() runs
# MILO_ASSERT(mCurrentOutfitConfig, 0x21) - a hard assert on that null - the instant you
# open the customize/color panel. That assert is the crash-on-select.
#
# Format confirmed two ways:
#  - RB3 decomp (src/system/bandobj/OutfitConfig.cpp): ASSERT_REVS(0x1B, 0) => revision 27,
#    LOAD_SUPERCLASS(Hmx::Object) => base is plain objFields (NOT RndDrawable).
#  - Byte-parsing the real vanilla guitar.cfg from greenday_blue.milo_xbox, which is
#    minimal: colors=[0,1,2], computeAO=true, every list (matSwaps/meshAOs/bandPatchMeshes/
#    piercings/overlays) empty, empty texBlender/bandLogo/wrinkleBlender, and a 20-byte
#    SHA1 digest.
#
# The SHA1 digest is NOT validated: grepping the whole decomp, mDigest is only ever read
# into the field (bs >> mDigest) and never compared or recomputed anywhere. So zeros are
# safe for a dummy config. This writes a minimal, valid, "does nothing but exist" config
# so the item stops crashing; real color-palette/mat-swap authoring can be layered on the
# empty lists later.
# =====================================================================================

RB3_OUTFITCONFIG_REVISION = 27  # ASSERT_REVS(0x1B, 0) in the RB3 decomp's OutfitConfig::Load

# Which OutfitConfig (.cfg) name the game expects per instrument. From the RB3 decomp's
# GetConfigNameFromAssetType (AssetTypes.cpp) and confirmed by byte-diffing real vanilla
# instruments (a real bass carries "bass.cfg", a real guitar "guitar.cfg"). ClosetMgr only
# does the config lookup for guitar/bass/drum, so those are the three that matter for the
# crash-on-select; the name MUST match the instrument or the customize panel asserts on null.
INSTRUMENT_CFG_NAMES = {
    'guitar': "guitar.cfg",
    'bass': "bass.cfg",
    'drum': "drum.cfg",
    # mic and keyboard are also valid instrument types (ClosetMgr::SetInstrumentType asserts
    # type is one of guitar/bass/drum/mic/keyboard), and GetConfigNameFromAssetType maps them
    # to mic.cfg / keyboard.cfg. The customize panel's null-config assert
    # (ChooseColorPanel::Load, MILO_ASSERT 0x21) is NOT special-cased per type, so mic and
    # keyboard need their config present too or they crash on select just like guitar/bass.
    'mic': "mic.cfg",
    'keyboard': "keyboard.cfg",
}


def write_outfit_config(w: MiloWriter, colors=(0, 1, 2), compute_ao=True, standalone=True):
    """Assets/Char/OutfitConfig.cs Write(), fixed at revision 27 (RB3). Writes a minimal
    valid config (empty matSwaps/meshAOs/patches/piercings/overlays) - just enough to
    satisfy the game's non-null OutfitConfig requirement so instruments/outfits don't
    crash on select. See the section comment above for the confirmed format + crash path."""
    w.u32(RB3_OUTFITCONFIG_REVISION)      # (altRevision=0 << 16) | 27

    write_object_fields(w)                 # base.Write(standalone=False) -> objFields

    # revision(27) > 4: colors[0], colors[1]; revision > 10: colors[2]
    w.i32(colors[0])
    w.i32(colors[1])
    w.i32(colors[2])

    # revision(27) > 3: matSwaps
    w.u32(0)          # matSwapCount (empty)

    w.u32(0)          # meshAOCount (empty)
    w.boolean(compute_ao)   # computeAO
    w.u32(0)          # bandPatchMeshCount (empty)
    w.u32(0)          # piercingCount (empty)

    w.symbol("")      # texBlender

    w.u32(0)          # overlaysCount (empty)

    w.symbol("")      # bandLogo

    w.block(bytes(20))  # sha1Digest - not validated by the game (see section comment)

    w.symbol("")      # wrinkleBlender

    if standalone:
        w.block(END_MARKER)


# =====================================================================================
# CharCollide support (the ".coll" asset type)
#
# A per-bone collision volume. Confirmed against BOTH the RB3 decomp
# (src/system/char/CharCollide.{h,cpp}, ASSERT_REVS(7,0)) and MiloLib
# (MiloLib/Assets/Char/CharCollide.cs) - the two agree field-for-field.
#
# Layout at revision 7:
#   (combined revision u32)
#   objFields                       <- LOAD_SUPERCLASS(Hmx::Object)
#   RndTransformable (rev 9)        <- LOAD_SUPERCLASS(RndTransformable): localXfm, worldXfm,
#                                      constraint, target, preserveScale, PARENT bone ptr.
#                                      This is exactly what write_rnd_trans already emits, so
#                                      the volume attaches to a bone the same way a Trans does.
#   shape        u32
#   origRadius0  f32
#   origLength0  f32   (rev > 4)
#   origLength1  f32   (rev > 2)
#   flags        i32   (rev > 1)
#   curRadius0   f32   (rev > 3)
#   -- rev > 5 block: --
#   origRadius1  f32
#   curRadius1   f32
#   curLength0   f32
#   curLength1   f32
#   unknownTransform (Transform = 12 f32)
#   mesh         symbol            (optional deform mesh; empty = none)
#   8x { unk0 i32, vec Vector3(3 f32) }
#   sha1Digest   20 bytes          (NOT validated by the game - only read into the field,
#                                   never compared/recomputed; zeros are safe, same as
#                                   OutfitConfig's digest)
#   meshYBias    bool
#
# "cur" values mirror "orig" values (the engine's CopyOriginalToCur), so we write the same
# number into both for a clean authored volume.
# =====================================================================================

RB3_CHARCOLLIDE_REVISION = 7  # ASSERT_REVS(7, 0) in CharCollide::Load


def write_char_collide(w: MiloWriter, shape=1, radius0=0.5, radius1=0.5,
                        length0=0.0, length1=1.0, flags=0, mesh_y_bias=False,
                        local_xfm=IDENTITY_MATRIX, world_xfm=IDENTITY_MATRIX,
                        parent_obj="", standalone=True):
    """Assets/Char/CharCollide.cs Write(), fixed at revision 7 (RB3). Emits a per-bone
    collision volume parented to `parent_obj` (a bone name). `shape` is the raw
    CharCollide::Shape int (0=plane,1=sphere,2=insideSphere,3=cigar,4=insideCigar).

    length0/length1 are only meaningful for the cigar shapes; radius1 likewise. For a
    sphere/plane the extra radius/length fields are still written (the format always
    includes them at rev 7) but the game ignores them for those shapes."""
    w.u32(RB3_CHARCOLLIDE_REVISION)             # (altRevision=0 << 16) | 7

    write_object_fields(w)                       # LOAD_SUPERCLASS(Hmx::Object)

    # LOAD_SUPERCLASS(RndTransformable): identical layout to a standalone-less RndTrans.
    # This is where the parent-bone attachment lives.
    write_rnd_trans(w, local_xfm=local_xfm, world_xfm=world_xfm, parent_obj=parent_obj,
                    standalone=False)

    w.u32(shape & 0xFFFFFFFF)                     # shape
    w.f32(radius0)                               # origRadius[0]
    w.f32(length0)                               # origLength[0]  (rev > 4)
    w.f32(length1)                               # origLength[1]  (rev > 2)
    w.i32(flags)                                 # flags          (rev > 1)
    w.f32(radius0)                               # curRadius[0]   (rev > 3) - mirrors orig

    # rev > 5 block
    w.f32(radius1)                               # origRadius[1]
    w.f32(radius1)                               # curRadius[1]   - mirrors orig
    w.f32(length0)                               # curLength[0]   - mirrors orig
    w.f32(length1)                               # curLength[1]   - mirrors orig
    write_matrix(w, IDENTITY_MATRIX)             # unknownTransform (12 f32)
    w.symbol("")                                 # mesh (no deform mesh)
    for _ in range(8):                           # 8x CharCollideStruct
        w.i32(0)                                 #   unk0
        w.f32(0.0); w.f32(0.0); w.f32(0.0)       #   vec (Vector3)
    w.block(bytes(20))                           # sha1Digest (not validated)
    w.boolean(mesh_y_bias)                       # meshYBias

    if standalone:
        w.block(END_MARKER)


# =====================================================================================
# SECTION 3b -- The Beatles: Rock Band revision constants
#
# Sourced from glTFMilo's Source/Shared/GameRevisions.cs (TheBeatlesRockBand entry) and
# then BYTE-VERIFIED by fully parsing real PS3 game files:
#   - george_skeleton.milo_ps3        (Character dir, 109 Trans bones + 4 CharCollide)
#   - george_headhands_long.milo_ps3  (Character dir, Tex/Mat/Mesh/Group/TexBlend...)
# The skeleton file's entire Character dir object round-trip-decodes to the exact object
# boundary using these revisions, and every Trans/CharCollide object decodes byte-clean.
#
# Confirmed layout facts that differ from RB3 and drive the writers below:
#   - DirectoryMeta.revision (milo)  = 25   (rev < 32, so NO extra DC3-style header byte)
#   - ObjectDir.revision             = 22   (rev < 27, so the unk1/unk2 u32 pair RB3's
#                                            rev-27 writer emits is ABSENT)
#   - Character.revision             = 15   (rev < 17 -> shadows are ONE bare Symbol, not
#                                            a count+list; rev NOT > 16 -> translucentGroup
#                                            is NOT written; rev > 14 -> minLod IS written,
#                                            real value -1)
#   - CharacterTesting.revision      = 10   (keeps older fields rev 15 dropped: unkSym
#                                            "none", clip2RealTime+bpm, unk2+unkFloat, and a
#                                            trailing "look at" bone Symbol)
#   - CharCollide.revision           = 5    (rev <= 5 -> the ENTIRE rev>5 block is omitted:
#                                            no unknownTransform / mesh / 8 structs / sha1 /
#                                            meshYBias - a compact 168-byte object)
#   - RndDir 10, Trans 9, Drawable 3, Animatable 4  (all identical to RB3, reused verbatim)
#   - objFields metadata is the same metaRev-2 / empty-type / hasTree-bool / note form the
#     RB3 path already writes (write_object_fields) - verified against the real bytes.
#
# The mesh-milo object revisions are recorded here for the next phase (mesh milo export)
# but are NOT yet wired into any writer - they need their own byte-parse of the real
# headhands/outfit milos before use (Mesh rev 33 in particular differs from RB3's rev 38).
# =====================================================================================
TBRB_MILO_REVISION = 25          # DirectoryMeta.revision
TBRB_OBJDIR_REVISION = 22        # ObjectDir.revision
TBRB_RNDDIR_REVISION = 10        # RndDir.revision (same as RB3)
TBRB_CHARACTER_REVISION = 15     # Character.revision
TBRB_CHARTEST_REVISION = 10      # Character.CharacterTesting.revision
TBRB_CHARCOLLIDE_REVISION = 5    # CharCollide.revision
# Mesh-milo revisions. CORRECTED against the retail files - glTFMilo's GameRevisions.cs
# lists Mesh as 33, but every one of the 41 Mesh objects across george_headhands_long
# (PS3) and walrus01 (Xbox) carries revision 36. Parsed with 36 they all decode to the
# exact object boundary; 33 does not. Trust the files, not the table.
TBRB_MESH_REVISION = 36          # RndMesh.revision  (verified; writer not yet implemented)
TBRB_MAT_REVISION = 55           # RndMat.revision   (verified, layout fully decoded)
TBRB_TEX_REVISION = 10           # RndTex.revision   (verified, layout fully decoded)
TBRB_GROUP_REVISION = 14         # RndGroup.revision (verified from entry headers)
TBRB_TEXBLENDCONTROLLER_REVISION = 1   # TexBlendController (face wrinkle blending)
TBRB_TEXBLENDER_REVISION = 2           # TexBlender

# Verified Tex/bitmap facts for the mesh-milo phase (from all 17 retail textures):
#   - Tex rev 10 is rev < 11, so optimizeForPS3 is NOT written (RB3's rev 11 does write it).
#   - useExternalPath is 1 in every retail TBRB texture, even though externalPath is empty
#     and the bitmap is embedded - the opposite of what the RB3 path writes.
#   - mipMapK is -8.0 (same as RB3).
#   - bitmap bpl = width * bpp / 8  (512x512 BC1 -> 256; 1024x512 BC3 -> 1024).
#   - MIP CHAINS ARE MANDATORY and stop when the smaller dimension reaches 16, NOT 4:
#     512x512 -> mipMaps=5 (6 levels), 256x256 -> 4, 512x256 -> 4, 64x64 -> 2.
#   - PS3 normal maps are BC1/DXT1 (enc 8); Xbox normal maps are ATI2/BC5 (enc 32),
#     matching the platform rule the RB3 texture path already implements.
TBRB_TEX_MIP_FLOOR = 16          # generate mips down to a 16px smaller dimension

# DC1 uses the exact same 16px mip floor as TBRB - confirmed by byte-parsing all 24 real
# textures embedded in retail emilia01.milo_xbox (nested inside its emilia01_textures.milo /
# emilia_shared_textures.milo sub-directories - see walk_all_entries). Every one of them, at
# every base size actually used (128, 256, 512) and including non-square textures, stops
# exactly when the smaller dimension would drop below 16:
#   256x256 ATI2 normal map  -> mipMaps=4  (256,128,64,32,16 - stops at 16x16)
#   512x512 ATI2 normal map  -> mipMaps=5  (512..16 - stops at 16x16)
#   128x128 BC1 diffuse      -> mipMaps=3  (128,64,32,16 - stops at 16x16)
#   512x128 BC3 hair diffuse -> mipMaps=3  (512x128 -> 64x16 - stops on the SHORTER axis)
# This is a completely separate constant from build_texture_mip_chain's own mip_floor=4
# default (kept for backwards compatibility with other callers) - DC1 callers must pass
# this explicitly, the same way TBRB already does with TBRB_TEX_MIP_FLOOR.
DC1_TEX_MIP_FLOOR = 16

# The revision the exporter WRITES for meshes. Confirmed as 34 by parsing a KNOWN-WORKING
# TBRB milo (custom character, glTFMilo meshes injected via MiloEditor) - every mesh in it
# is rev 34 with an 80-byte uncompressed vertex (rev 34 adds a w after position and an nw
# after normal vs rev 33's 72-byte layout). An earlier build wrote 33 and crashed on load.
# Retail's own meshes are 36 with a compressed vertex block; 34 is the uncompressed format
# with a live working precedent. Tangents are four plain floats here, so the RB3
# flat-normal-map bug cannot recur (see write_tbrb_rnd_mesh's docstring).
TBRB_MESH_WRITE_REVISION = 34

# TBRB skeleton milos point their single ObjectDir subdir at the shared wind milo, NOT
# char_shared.milo - confirmed by parsing george_skeleton.milo_ps3 (subDirs was exactly
# ["../../world/shared/wind.milo"]). This is inert for loading a leaf skeleton but is
# reproduced for fidelity.
TBRB_SKELETON_SUBDIR = "../../world/shared/wind.milo"

# CharacterTesting rev 10's trailing symbol is a "look at" bone reference; the real george
# skeleton uses bone_R-eye.mesh. Kept as a named default so a skeleton that happens not to
# contain that bone can fall back to empty rather than ship a dangling reference.
TBRB_CHARTEST_LOOK_BONE = "bone_R-eye.mesh"


# =====================================================================================
# TBRB reference skeleton data (parsed from the retail PS3 george_skeleton.milo_ps3)
#
# Used for VALIDATION ONLY - the exporter never injects these, it only warns when the
# armature being exported is missing bones the retail skeleton has. That warning exists
# because a skeleton milo with missing bones loads fine and then crashes once a song
# starts: the animation/look-at systems resolve driver bones BY NAME, and the retail
# character meshes' inverse-bind matrices are baked against these exact bones.
#
# The four collision volumes below are byte-exact copies of george's, including their
# local transforms relative to the parent bone. Note bone_head.mesh carries THREE of them
# (face/forehead/head), which the generic per-bone "<bone>.coll" collision UI cannot
# express - that's why TBRB gets this dedicated table instead.
# =====================================================================================
TBRB_REFERENCE_SKELETON_BONES = frozenset({
    'bone_L-ankle.mesh', 'bone_L-brow1.mesh', 'bone_L-brow2.mesh', 'bone_L-brow3.mesh',
    'bone_L-cheek.mesh', 'bone_L-cheek2.mesh', 'bone_L-clavicle.mesh', 'bone_L-crease.mesh',
    'bone_L-eye.mesh', 'bone_L-eye_back.mesh', 'bone_L-eyelid-low.mesh', 'bone_L-foreArm.mesh',
    'bone_L-foreTwist1.mesh', 'bone_L-foreTwist2.mesh', 'bone_L-hand.mesh',
    'bone_L-index01.mesh', 'bone_L-index02.mesh', 'bone_L-index03.mesh', 'bone_L-knee.mesh',
    'bone_L-lid.mesh', 'bone_L-lipcorner.mesh', 'bone_L-middlefinger01.mesh',
    'bone_L-middlefinger02.mesh', 'bone_L-middlefinger03.mesh', 'bone_L-nose.mesh',
    'bone_L-pinky01.mesh', 'bone_L-pinky02.mesh', 'bone_L-pinky03.mesh',
    'bone_L-ringfinger01.mesh', 'bone_L-ringfinger02.mesh', 'bone_L-ringfinger03.mesh',
    'bone_L-thigh.mesh', 'bone_L-thumb01.mesh', 'bone_L-thumb02.mesh', 'bone_L-thumb03.mesh',
    'bone_L-toe.mesh', 'bone_L-upperArm.mesh', 'bone_L-upperTwist1.mesh',
    'bone_L-upperTwist2.mesh', 'bone_R-ankle.mesh', 'bone_R-brow1.mesh', 'bone_R-brow2.mesh',
    'bone_R-brow3.mesh', 'bone_R-cheek.mesh', 'bone_R-cheek2.mesh', 'bone_R-clavicle.mesh',
    'bone_R-crease.mesh', 'bone_R-eye.mesh', 'bone_R-eye_back.mesh', 'bone_R-eyelid-low.mesh',
    'bone_R-foreArm.mesh', 'bone_R-foreTwist1.mesh', 'bone_R-foreTwist2.mesh',
    'bone_R-hand.mesh', 'bone_R-index01.mesh', 'bone_R-index02.mesh', 'bone_R-index03.mesh',
    'bone_R-knee.mesh', 'bone_R-lid.mesh', 'bone_R-lipcorner.mesh',
    'bone_R-middlefinger01.mesh', 'bone_R-middlefinger02.mesh', 'bone_R-middlefinger03.mesh',
    'bone_R-nose.mesh', 'bone_R-pinky01.mesh', 'bone_R-pinky02.mesh', 'bone_R-pinky03.mesh',
    'bone_R-ringfinger01.mesh', 'bone_R-ringfinger02.mesh', 'bone_R-ringfinger03.mesh',
    'bone_R-thigh.mesh', 'bone_R-thumb01.mesh', 'bone_R-thumb02.mesh', 'bone_R-thumb03.mesh',
    'bone_R-toe.mesh', 'bone_R-upperArm.mesh', 'bone_R-upperTwist1.mesh',
    'bone_R-upperTwist2.mesh', 'bone_brow-low.mesh', 'bone_brow-mid.mesh', 'bone_chin.mesh',
    'bone_eyes.mesh', 'bone_footik.mesh', 'bone_forehead.mesh', 'bone_guitar.mesh',
    'bone_head.mesh', 'bone_head_lookat.mesh', 'bone_head_nod.mesh', 'bone_jaw.mesh',
    'bone_liptop_left.mesh', 'bone_liptop_mid.mesh', 'bone_liptop_right.mesh',
    'bone_lowlip_left.mesh', 'bone_lowlip_mid.mesh', 'bone_lowlip_right.mesh',
    'bone_neck.mesh', 'bone_neckTwist.mesh', 'bone_nose.mesh', 'bone_pelvis.mesh',
    'bone_spine1.mesh', 'bone_spine2.mesh', 'bone_spine3.mesh', 'bone_tongue1.mesh',
    'bone_tongue2.mesh', 'bone_tongue3.mesh', 'bone_tongue4.mesh', 'spot_L-cheek.mesh',
    'spot_R-cheek.mesh', 'spot_look.mesh',
})

# (entry_name, parent_bone, shape, radius0, length0, length1, flags, local_xfm, world_xfm)
# Exact float32 values lifted from the retail george skeleton - do not round these; the
# rotation parts carry the head bone's axis convention and the tiny epsilons are what the
# retail file actually stores.
TBRB_STANDARD_COLLISIONS = (
    ('face.coll', 'bone_head.mesh', 1, 3.5, 0.0, 0.0, 64,
     (-8.485315629513934e-05, -0.0007023758953437209, -1.0, 5.039658503847022e-07, 1.0, -0.0007023758953437209, 1.0000001192092896, -6.322507601908001e-07, -8.485285070491955e-05, 1.3321876525878906, 0.8149709701538086, -0.0006557442247867584),
     (1.000000238418579, 1.1641532182693481e-10, 1.4551915228366852e-11, -1.1641532182693481e-10, 1.000000238418579, 2.6290081223123707e-13, 0.0, -2.1316282072803006e-14, 1.0, -2.6921043172478676e-09, -0.4348393678665161, 62.05038070678711)),
    ('forehead.coll', 'bone_head.mesh', 1, 3.5, 0.0, 0.0, 4,
     (-8.48532872623764e-05, -0.00070237583713606, -1.0, 5.039658503847022e-07, 1.0, -0.0007023758953437209, 1.000000238418579, -5.635645834445313e-07, -8.485288708470762e-05, 4.078254699707031, 0.5901411771774292, -0.0007308428175747395),
     (1.000000238418579, 1.7462298274040222e-10, -1.1641532182693481e-10, -1.1641532182693481e-10, 1.000000238418579, 2.6290081223123707e-13, -2.9103830456733704e-11, 6.868617674626876e-08, 1.0000001192092896, -1.877197064459324e-09, -0.6596674919128418, 64.79644775390625)),
    ('head.coll', 'bone_head.mesh', 1, 3.5, 0.0, 0.0, 8,
     (-8.48532872623764e-05, -0.00070237583713606, -1.0, 5.039658503847022e-07, 1.0, -0.0007023758953437209, 1.000000238418579, -5.635645834445313e-07, -8.485288708470762e-05, 4.060546875, -0.57911217212677, 9.191548451781273e-05),
     (1.000000238418579, 1.7462298274040222e-10, -1.1641532182693481e-10, -1.1641532182693481e-10, 1.000000238418579, 2.6290081223123707e-13, -2.9103830456733704e-11, 6.868617674626876e-08, 1.0000001192092896, -2.204615157097578e-09, -1.8289211988449097, 64.77873992919922)),
    ('neck.coll', 'bone_neck.mesh', 3, 2.0, 0.0, 2.0, 128,
     (1.0, 0.0, -0.0, -0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
     (-0.0003086627693846822, 0.3069963753223419, 0.9517107009887695, 5.298877658788115e-05, 0.9517108798027039, -0.3069964051246643, -1.0, -4.432826244737953e-05, -0.0003100250323768705, 0.00046827291953377426, -2.0362954139709473, 57.248836517333984)),
)


def build_tbrb_standard_collision_entries(bone_names):
    """Returns the retail TBRB head/neck collision volumes as CharCollide entries, in the
    same (entry_name, coll_dict) shape build_char_collide_entries produces.

    These are emitted verbatim from the retail values rather than derived from Blender,
    because (a) the generic collision UI is one-volume-per-bone and TBRB puts three on
    bone_head.mesh, and (b) hair/cloth in the outfit milo references these by name, so
    inventing different ones defeats the purpose. Volumes whose parent bone isn't present
    in the armature are skipped."""
    entries = []
    for (name, parent, shape, radius0, length0, length1, flags,
         local_xfm, world_xfm) in TBRB_STANDARD_COLLISIONS:
        if parent not in bone_names:
            _log(f"  Skipped standard collision '{name}': parent bone '{parent}' not in armature.")
            continue
        entries.append((name, {
            'shape': shape, 'radius0': radius0, 'radius1': radius0,
            'length0': length0, 'length1': length1, 'flags': flags,
            'mesh_y_bias': False,
            'local_xfm': local_xfm,
            # worldXfm is non-authoritative (the engine recomputes it from local + parent),
            # but the retail values are reproduced so these objects are byte-identical to
            # george's - that keeps them exactly diffable against a vanilla file.
            'world_xfm': world_xfm,
            'parent': parent,
        }))
    return entries


# =====================================================================================
# The Beatles: Rock Band skeleton-milo writers
#
# These are the TBRB (rev 25 milo) counterparts to write_object_dir_base / write_character
# / write_character_testing / write_char_collide. They are SEPARATE functions rather than
# revision parameters on the existing RB3 writers so the thoroughly-tested RB3/DC paths are
# untouched. Every field below was byte-verified against george_skeleton.milo_ps3 (see the
# SECTION 3b constant block).
# =====================================================================================

def write_tbrb_object_dir_base(w: MiloWriter, sub_dirs=None, viewports=None):
    """ObjectDir.Write at revision 22 (The Beatles: Rock Band).

    The only STRUCTURAL delta from RB3's rev-27 write_object_dir_base: rev 22 < 27, so the
    unk1/unk2 u32 pair is absent. Field values that differ from the RB3 defaults and were
    taken from the real george skeleton: currentViewportIdx = 0 (not 6) and inlineProxy =
    True (the real file carries 1 here with an empty proxyPath - inert but reproduced).

    The objFields tail is the single-boolean (root.hasTree=false) + note form, which parses
    cleanly against the real bytes (MiloLib models a 6-byte DTBParent there, but the actual
    on-disk form is the 1-byte one - the same convention write_object_dir_base already uses).
    Viewports are 7 identity matrices, which the engine ignores at load (RB3 exports use the
    same and load fine)."""
    if viewports is None:
        viewports = [IDENTITY_MATRIX] * 7
    if sub_dirs is None:
        sub_dirs = []

    w.u32(TBRB_OBJDIR_REVISION)    # ObjectDir's own revision/altRevision (22)
    w.u16(0)                       # objFields.altRevision
    w.u16(OBJ_FIELDS_REVISION)     # objFields.revision (2)
    w.symbol("")                   # objFields.type
    # rev 22 < 27 -> NO unk1/unk2 here (this is the whole structural delta vs RB3)
    w.u32(len(viewports))
    for vp in viewports:
        write_matrix(w, vp)
    w.u32(0)                       # currentViewportIdx (real george skeleton: 0)

    w.boolean(True)                # inlineProxy (real george skeleton: 1)
    w.symbol("")                   # proxyPath

    w.u32(len(sub_dirs))
    for path in sub_dirs:
        w.symbol(path)

    w.u8(0)                        # inlineSubDir (kInlineNever)
    w.u32(0)                       # inlineSubDirs.Count

    w.symbol("")                   # unknownString
    w.symbol("")                   # unknownCamReference

    w.boolean(False)               # objFields.root.hasTree
    w.symbol("")                   # objFields.note


def write_tbrb_character_testing(w: MiloWriter, look_bone=TBRB_CHARTEST_LOOK_BONE):
    """Character.CharacterTesting.Write at revision 10 (The Beatles: Rock Band).

    Byte-verified against george_skeleton.milo_ps3. Rev 10's tail is substantially longer
    than RB3's rev 15 - it retains several fields rev 15 dropped. Real george values are
    reproduced (distMap/unkSymbol "none", cycleTransition true, bpm 120, unkFloat 3.0, and
    the trailing look-at bone symbol)."""
    w.u32(TBRB_CHARTEST_REVISION)  # 10
    w.symbol("")            # driver
    w.symbol("")            # clip1
    w.symbol("")            # clip2
    w.symbol("")            # teleportTo
    w.symbol("")            # teleportFrom
    w.symbol("none")        # distMap                       (real: "none")
    # revision >= 6:
    w.u32(0)                # transition
    w.boolean(True)         # cycleTransition               (real: 1)
    w.u32(0)                # internalTransition
    # revision 10 is NOT < 10 -> no unk1
    w.boolean(False)        # metronome
    w.boolean(False)        # zeroTravel
    w.boolean(False)        # showScreenSize
    # revision < 0xC (12) -> unkSymbol
    w.symbol("none")        # unkSymbol                     (real: "none")
    w.boolean(False)        # footExtents
    # revision < 15 -> clip2RealTime + bpm
    w.boolean(False)        # clip2RealTime
    w.u32(120)              # bpm                           (real: 120)
    # revision != 6 and revision < 14 -> unk2 + unkFloat
    w.u32(0)                # unk2
    w.f32(3.0)              # unkFloat                      (real: 3.0)
    # revision < 14 && revision > 9 -> unkSymbol2 (the "look at" bone)
    w.symbol(look_bone)     # unkSymbol2                    (real: "bone_R-eye.mesh")


def write_tbrb_character(w: MiloWriter, root_name, bounding=(0.0, 0.0, 0.0, 0.0),
                          look_bone=TBRB_CHARTEST_LOOK_BONE, sub_dirs=None,
                          sphere_base=None):
    """Character.Write at revision 15 (The Beatles: Rock Band). Byte-verified against
    george_skeleton.milo_ps3 - the full Character dir object (Character -> RndDir ->
    ObjectDir -> anim/draw/trans -> LODs/shadow/bounding/charTest) round-trip-decodes to
    the exact object boundary.

    Deltas vs RB3's rev-17 write_character:
      - Character rev 15 (not 17)
      - rev < 17 -> shadows written as ONE bare Symbol (empty), not a u32 count + list
      - rev NOT > 16 -> translucentGroup is NOT written
      - rev > 14 -> minLod IS written (real value -1)
      - charTest is rev 10 (write_tbrb_character_testing), not rev 15
      - the single ObjectDir subdir defaults to wind.milo, not char_shared.milo
      - the draw sphere and the Character bounding sphere are the same value (as in the
        real file)."""
    if sub_dirs is None:
        sub_dirs = [TBRB_SKELETON_SUBDIR]

    w.u32(TBRB_CHARACTER_REVISION)          # 15

    # base.Write() -> RndDir.Write(standalone=False)
    w.u32(TBRB_RNDDIR_REVISION)             # 10
    write_tbrb_object_dir_base(w, sub_dirs=sub_dirs)
    write_rnd_animatable(w)                 # rev 4 (shared with RB3)
    write_rnd_drawable(w, sphere=bounding)  # rev 3 (shared); real draw sphere == bounding
    write_rnd_trans(w)                      # rev 9 embedded, identity, no parent
    w.symbol("")   # environ
    w.symbol("")   # testEvent

    # Character-specific fields (rev 15, non-proxy branch)
    w.u32(0)                    # lods.Count (a skeleton has no LOD groups)
    w.symbol("")               # rev < 17: exactly ONE shadow Symbol (empty)
    w.boolean(False)           # selfShadow
    w.symbol(sphere_base if sphere_base is not None else root_name)  # sphereBase
    write_sphere(w, *bounding) # bounding sphere
    w.boolean(False)           # frozen
    w.i32(-1)                  # minLod (real file: -1 == kLODAllFrames)
    # rev 15 is NOT > 16 -> translucentGroup is NOT written
    write_tbrb_character_testing(w, look_bone=look_bone)

    w.block(END_MARKER)        # Character.Write's own standalone=True end marker


def write_tbrb_char_collide(w: MiloWriter, shape=1, radius0=0.5, length0=0.0, length1=1.0,
                             flags=0, local_xfm=IDENTITY_MATRIX, world_xfm=IDENTITY_MATRIX,
                             parent_obj="", standalone=True):
    """CharCollide.Write at revision 5 (The Beatles: Rock Band). Byte-verified against the
    real face/forehead/head/neck .coll entries (each exactly 168 bytes).

    Rev 5 is the compact form: because rev is NOT > 5, the ENTIRE rev>5 block RB3's rev-7
    writer emits (origRadius[1], curRadius[1], curLength[0/1], unknownTransform, mesh symbol,
    8 CharCollideStructs, 20-byte sha1, meshYBias) is absent. What remains is objFields + an
    embedded (non-standalone) Trans + shape + origRadius0 + origLength0 + origLength1 + flags
    + curRadius0. The embedded Trans carries the parent-bone attachment."""
    w.u32(TBRB_CHARCOLLIDE_REVISION)         # 5
    write_object_fields(w)                    # Hmx::Object (metaRev 2, empty type, bool, note)
    # LOAD_SUPERCLASS(RndTransformable): embedded non-standalone Trans (no objFields, no
    # end marker) - this is where the parent-bone attachment lives.
    write_rnd_trans(w, local_xfm=local_xfm, world_xfm=world_xfm, parent_obj=parent_obj,
                    standalone=False)
    w.u32(shape & 0xFFFFFFFF)                 # shape
    w.f32(radius0)                            # origRadius[0]
    w.f32(length0)                            # origLength[0]  (rev > 4)
    w.f32(length1)                            # origLength[1]  (rev > 2)
    w.i32(flags)                              # flags          (rev > 1)
    w.f32(radius0)                            # curRadius[0]   (rev > 3) - mirrors orig
    # rev 5 is NOT > 5 -> the entire rev>5 block is omitted.
    if standalone:
        w.block(END_MARKER)


def build_tbrb_skeleton_milo_bytes(root_name, bone_trans_entries, char_collide_entries,
                                    look_bone=TBRB_CHARTEST_LOOK_BONE):
    """Build a complete The Beatles: Rock Band SKELETON milo: a Character directory
    (DirectoryMeta rev 25) containing ONLY Trans bones + CharCollide volumes - no meshes,
    materials, or textures. This is the separate per-Beatle skeleton milo
    (george_skeleton.milo_ps3 etc.); the mesh milo's meshes skin against these bones by
    name at runtime.

    Entry table order must match body write order exactly: all Trans bones first, then all
    CharCollide. Every object revision here is byte-verified against a real PS3
    george_skeleton.

    bone_trans_entries:   list of (bone_name, local_xfm12, world_xfm12, parent_obj)
                          (as produced by build_armature_trans_entries)
    char_collide_entries: list of (entry_name, coll_dict)
                          (as produced by build_char_collide_entries)
    """
    body = MiloWriter(big_endian=True)
    total_entries = len(bone_trans_entries) + len(char_collide_entries)

    # --- DirectoryMeta (rev 25) ---
    body.u32(TBRB_MILO_REVISION)          # revision (25)
    body.symbol("Character")              # type
    body.symbol(root_name)                # name
    body.i32((total_entries + 1) * 2)     # stringTableCount (matches MiloEditor & the real
                                           # files' (entryCount+1)*2 formula)
    body.u32(0)                           # stringTableSize (engine recalculates; real RB3/DC
                                           # exports ship 0 here and load fine)
    # rev 25 < 32 -> no extra DC3-style header byte
    body.i32(total_entries)               # entryCount
    for (bone_name, *_rest) in bone_trans_entries:
        body.symbol("Trans")
        body.symbol(bone_name)
    for (entry_name, *_rest) in char_collide_entries:
        body.symbol("CharCollide")
        body.symbol(entry_name)

    # Bounding sphere: centroid of bone world origins with radius 0. The real skeleton
    # carries a specific center with r=0 - a skeleton milo has no visible geometry, so its
    # bounds are inert for rendering; a rough centroid keeps tools happy without pretending
    # to a real radius.
    if bone_trans_entries:
        n = len(bone_trans_entries)
        cx = sum(b[2][9] for b in bone_trans_entries) / n
        cy = sum(b[2][10] for b in bone_trans_entries) / n
        cz = sum(b[2][11] for b in bone_trans_entries) / n
    else:
        cx = cy = cz = 0.0
    bounding = (cx, cy, cz, 0.0)

    # look-at bone: prefer the reference default if present, else fall back gracefully.
    bone_names = {b[0] for b in bone_trans_entries}
    if look_bone not in bone_names:
        look_bone = "bone_head.mesh" if "bone_head.mesh" in bone_names else \
                    (sorted(bone_names)[0] if bone_names else "")

    # --- Character dir object ---
    write_tbrb_character(body, root_name, bounding=bounding, look_bone=look_bone)

    # --- Trans bones (rev 9 standalone, WITH objFields), then CharCollide (rev 5) ---
    for (bone_name, local_xfm, world_xfm, parent_obj) in bone_trans_entries:
        write_rnd_trans(body, local_xfm=local_xfm, world_xfm=world_xfm,
                        parent_obj=parent_obj, standalone=True, skip_metadata=False)
    for (entry_name, coll) in char_collide_entries:
        write_tbrb_char_collide(
            body, shape=coll['shape'], radius0=coll['radius0'],
            length0=coll['length0'], length1=coll['length1'], flags=coll['flags'],
            local_xfm=coll['local_xfm'], world_xfm=coll['world_xfm'],
            parent_obj=coll['parent'], standalone=True)

    body_bytes = bytes(body.buf)

    # Block splitting: a skeleton is tiny (a real george skeleton is ~21 KB) so it is always
    # a single block, but keep the >MAX chunking correct for safety.
    if len(body_bytes) <= MAX_MILO_BLOCK_SIZE:
        block_sizes = [len(body_bytes)]
    else:
        block_sizes = []
        pos = 0
        while pos < len(body_bytes):
            block_sizes.append(min(MAX_MILO_BLOCK_SIZE, len(body_bytes) - pos))
            pos += MAX_MILO_BLOCK_SIZE

    header = MiloWriter(big_endian=False)
    START_OFFSET = 0x810
    header.u32(0xCABEDEAF)            # Type.Uncompressed
    header.u32(START_OFFSET)
    header.u32(len(block_sizes))
    header.u32(max(block_sizes))
    for sz in block_sizes:
        header.u32(sz)
    header.block(bytes(START_OFFSET - len(header.buf)))

    return bytes(header.buf) + body_bytes


def write_tbrb_rnd_mesh(w: MiloWriter, entry_name, local_xfm, world_xfm, parent_obj,
                         vertices, faces, mat_name="", bone_transforms=None,
                         force_white_vertex_color=False, write_tangents=True):
    """RndMesh.Write at revision 34 for The Beatles: Rock Band.

    WHY 34. Confirmed empirically against a KNOWN-WORKING TBRB milo (the same custom
    character, with meshes injected via glTFMilo + MiloEditor, that loads and plays in-game):
    every mesh in it is revision 34, not the 33 glTFMilo's revision table lists (MiloEditor
    upgrades them to 34 on save). An earlier plugin build wrote 33 and crashed on load; all
    six working meshes parse clean as 34. Retail's own meshes are 36 with a compressed
    vertex block, but 34 is the uncompressed format with a live working precedent, so it is
    the target.

    THE REV-34 QUIRK. Versus 33, revision 34 writes TWO extra padding floats: a `w` after
    the position and an `nw` after the normal (both gated on `meshVersion == 34` in the
    loader). Both are 0.0 in the working file. Missing them (i.e. writing the 33 layout under
    a 34 header, or writing a 33 header at all) desynchronises every subsequent vertex.

    WHY THE RB3 TANGENT BUG CANNOT RECUR HERE. On RB3 (rev 38) the vertex stores tangents in
    `packed3`/`packed4` (10-10-10-2). glTFMilo populates `tangent0..3` (plain floats) and
    never fills packed3/packed4, so its RB3 output carried zero tangents and normal maps went
    flat. At revision 34 the writer serialises `tangent0..3` directly as four floats - the
    same field - so the mismatch cannot occur. (Note: the working reference milo happens to
    carry ZERO tangents and still loads, which proves tangents are not load-critical; this
    writer fills real ones anyway, with tangent3 carrying Blender's raw bitangent_sign.)

    Vertex layout at revision 34 (uncompressed, 80 bytes/vertex, platform-independent):
        position   3 x f32  +  w   (1 x f32 = 0.0)   <- the w is rev-34-only
        normal     3 x f32  +  nw  (1 x f32 = 0.0)   <- the nw is rev-34-only
        weights    4 x f32
        uv         2 x f32
        bones      4 x u16      <- u16, NOT the u8 indices RB3's compressed vertex uses
        tangent    4 x f32
    There is no isNextGen/vertexSize/compressionType header (gated on rev >= 36), no
    keepMeshData (rev > 34) and no hasAOCalculation (rev > 37).
    """
    if bone_transforms is None:
        bone_transforms = []

    w.u32(TBRB_MESH_WRITE_REVISION)     # 34
    write_object_fields(w)
    write_rnd_trans(w, local_xfm, world_xfm, parent_obj)
    # Draw sphere: radius 10000, NOT zero. A zero-radius bounding sphere is degenerate and
    # gets the mesh culled (or crashes). Every mesh in a confirmed-working TBRB milo ships
    # sphere = (0,0,0,10000), i.e. "effectively never cull" - matched here exactly.
    write_rnd_drawable(w, sphere=(0.0, 0.0, 0.0, 10000.0))

    w.symbol(mat_name)                  # mat
    w.symbol(entry_name)                # geomOwner - real files self-reference
    w.u32(0)                            # mutable (kMutableNone)      rev >= 16
    w.u32(1)                            # volume (kVolumeTriangles)   rev > 17
    w.boolean(False)                    # bspNode.hasValue            rev > 18

    # --- vertices (no isNextGen header at rev 34; 80-byte uncompressed stride) ---
    w.u32(len(vertices))
    for v in vertices:
        w.f32(v["x"]); w.f32(v["y"]); w.f32(v["z"])
        w.f32(0.0)                      # position w - rev 34 only, retail/gltfMilo write 0
        w.f32(v["nx"]); w.f32(v["ny"]); w.f32(v["nz"])
        w.f32(0.0)                      # normal w (nw) - rev 34 only, written as 0
        weights = v.get("weights", (0.0, 0.0, 0.0, 0.0))
        for i in range(4):
            w.f32(weights[i] if i < len(weights) else 0.0)
        w.f32(v["u"])
        w.f32(v["v"])
        # Bone indices are u16 here. The RB3 Xbox-vs-PS3 index REVERSAL does not apply:
        # that quirk belongs to the compressed 10-10-10-2 weight vertex, where the bone
        # bytes pair against the packed weight order. At rev 33 weights and bones are both
        # plain arrays in matching slot order, so they are written straight through.
        bones = v.get("bones", (0, 0, 0, 0))
        for i in range(4):
            w.u16(bones[i] if i < len(bones) else 0)
        if write_tangents:
            w.f32(v.get("tx", 0.0)); w.f32(v.get("ty", 0.0)); w.f32(v.get("tz", 0.0))
            w.f32(v.get("tw", 1.0))
        else:
            w.f32(0.0); w.f32(0.0); w.f32(0.0); w.f32(0.0)

    # --- faces ---
    w.u32(len(faces))
    for (i1, i2, i3) in faces:
        w.u16(i1); w.u16(i2); w.u16(i3)

    group_sizes = compute_group_sizes(len(faces))   # rev > 0x17
    w.u32(len(group_sizes))
    for g in group_sizes:
        w.u8(g)

    # --- bone transforms ---
    # The reader peeks an int32: <= 0 means "no bone list" and consumes the word; > 0 means
    # it is the bone count and the list follows. THERE IS NO TRAILING END_MARKER AFTER THIS
    # FIELD (OR THE BONE LIST) AT ALL - the object's data simply ends here.
    #
    # This is confirmed against TWO independently-produced known-working reference files:
    #   - A facial-hair mesh from the original glTFMilo CLI with 14 bone transforms: the
    #     file ends immediately after the last bone matrix - no 0xADDEADDE, nothing.
    #   - A mouth-harmonica mesh from an unrelated/unknown custom-song compiler with ZERO
    #     bone transforms: the file ends immediately after a literal 0x00000000 "no bones"
    #     word - again no 0xADDEADDE, nothing.
    # Both agree: whatever the bone_transforms branch, nothing follows it.
    #
    # An EARLIER version of this comment/fix assumed 0xADDEADDE was written for the "no
    # bones" case instead of a literal 0, reasoning that a signed-negative sentinel could
    # double as both the "no bones" flag AND a terminal marker. That was a plausible-looking
    # but WRONG guess made without a second reference file to check it against - the
    # harmonica file above disproves it directly (it has a plain 0, not 0xADDEADDE, and
    # still nothing after it). The actual bug this whole section exists to fix was simpler
    # than either version: the ORIGINAL code wrote a literal 0 for "no bones" and THEN
    # unconditionally appended a separate END_MARKER afterward regardless of branch, which:
    #   1) left 4 orphaned, never-consumed trailing bytes after a no-bone-list mesh
    #      (0x00000000 gets consumed as the "no bones" word, then 0xADDEADDE dangles
    #      after it with nothing left in the object's Read() to consume it), and
    #   2) added an extra unwritten-by-spec 0xADDEADDE after a bone list that neither
    #      reference format ever has.
    # For a loose standalone Custom Song Asset .mesh file (no DirectoryMeta block-size
    # table to fall back on), trailing bytes the reader never consumes is exactly the kind
    # of malformed-asset condition that can bounce a custom song back to the main menu
    # right after the loading screen appears - so the fix is simply to never write
    # END_MARKER after this section, in either branch.
    if bone_transforms:
        w.u32(len(bone_transforms))
        for (bone_name, transform12) in bone_transforms:
            write_bone_transform(w, bone_name, transform12)
    else:
        w.u32(0)


def write_tbrb_rnd_tex(w: MiloWriter, width, height, encoding, bpp, block_data,
                        external_path="", platform='xbox360', num_mips=0):
    """RndTex.Write at revision 10 (The Beatles: Rock Band). Byte-layout verified against
    all 17 textures in the two retail milos.

    Two deltas from the RB3 rev-11 writer:
      1. revision 10 is < 11, so optimizeForPS3 is NOT written at all.
      2. useExternalPath is TRUE in every retail TBRB texture, even though externalPath is
         empty and the bitmap is embedded immediately after. The RB3 path writes False.
         Retail is reproduced here rather than the value that "looks" right.
    mipMapK stays -8.0, same as RB3."""
    w.u32(TBRB_TEX_REVISION)        # 10
    write_object_fields(w)          # revision > 8 -> objFields
    w.u32(width)
    w.u32(height)
    w.u32(bpp)
    w.symbol(external_path)
    w.f32(-8.0)                     # mipMapK (revision >= 8)
    w.u32(TEX_TYPE_REGULAR)         # type (revision > 6)
    # revision 10 < 11 -> NO optimizeForPS3 here (this is the whole structural delta)
    w.boolean(True)                 # useExternalPath - retail TBRB ships 1, see docstring
    write_rnd_bitmap(w, width, height, encoding, bpp, block_data, platform=platform,
                      num_mips=num_mips)
    w.block(END_MARKER)


def write_tbrb_rnd_mat(w: MiloWriter, settings, standalone=True, force_tbrb_defaults=False):
    """RndMat.Write at revision 55 (The Beatles: Rock Band). Field order verified by
    decoding all 12 retail materials - every one lands exactly on its object boundary.

    Takes the SAME settings dict gather_materials_and_textures() already produces for the
    RB3 path, so material authoring in the UI is shared between games.

    Deltas from the RB3 rev-68 writer, all revision-gated:
      - rev 55 is NOT > 60, so unkBool_0x3C / unkBool_0x42 are absent.
      - rev 55 IS < 63, so projLights IS present (RB3 drops it).
      - rev 55 falls in the 52..67 window, so a trailing block exists that RB3 has none of:
        unkInt3 (rev >= 53), unkColor2 (rev 53..59), unkSym2 (rev 54..61),
        ps3ForceTrilinear (rev 55..62).
      - rev 55 is NOT > 63, so the refract block is absent - refraction settings authored
        in the UI are silently unavailable on TBRB.
    shaderVariation in retail: 1 (kShaderVariationSkin) on face/hands, 2 on nails,
    0 on clothing.

    `force_tbrb_defaults`: the Blender material UI was authored for RB3, and several of its
    default values render a TBRB character solid black even though the file is structurally
    correct (confirmed by bisection - our RB3-defaulted export was black; a variant with
    these specific fields set to retail TBRB values rendered correctly). When True, the
    render-affecting fields below are overridden with byte-verified known-good values (from
    the confirmed-working Body_fix_ALL material) instead of the UI's RB3 defaults. This is a
    TESTING aid to prove the material writer is correct independent of UI value sync; the
    long-term fix is to expose TBRB-appropriate values in the UI (and gray out RB3-only
    controls like Refract Settings for pre-RB3 games). Texture names, base_color, blend,
    and z_mode still come from the authored material either way."""
    s = dict(settings)   # copy so overrides don't mutate the caller's dict
    if force_tbrb_defaults:
        s['cull'] = True
        s['specular_power'] = 2.0
        s['point_lights'] = False
        s['color_adjust'] = True
        s['rim_rgb'] = (0.40784, 0.36078, 0.29020)
        s['rim_power'] = 4.0
        s['rim_always_show'] = True
        s['specular2_power'] = 10.0
    w.u32(TBRB_MAT_REVISION)        # 55
    write_object_fields(w)
    w.i32(MAT_BLEND_VALUES[s['blend']])
    for c in s['base_color']:
        w.f32(c)
    # Order is prelit THEN use_environment - verified against PikminGuts92's re-notes
    # mat.bt 010 template ("Bool prelit;" immediately precedes "Bool use_environ;") and
    # matches the working RB3 rev-68 writer. These two were previously swapped here, which
    # inverted both fields on export: a material authored Pre-Lit ON / Use-Environment OFF
    # was written as prelit=0 / use_environ=1, disabling the vertex-color base lighting and
    # instead enabling environment modulation - rendering the character solid black in-game
    # despite a valid diffuse texture. Confirmed by byte-parsing a known-good TBRB milo
    # (prelit=1, use_environ=0) against our broken export (prelit=0, use_environ=1).
    w.boolean(s['pre_lit'])
    w.boolean(s['use_environment'])
    w.i32(MAT_ZMODE_VALUES[s['z_mode']])
    w.boolean(s['alpha_cut'])
    w.i32(s['alpha_threshold'])                 # rev > 37
    w.boolean(s['blend'] == 'kBlendSrcAlpha')   # alphaWrite - mirrors the RB3 path
    w.i32(MAT_TEXGEN_NONE)
    w.i32(MAT_TEXWRAP_REPEAT)
    write_matrix(w, IDENTITY_MATRIX)            # texXfm
    w.symbol(s.get('diffuse_tex_name', ""))
    w.symbol("")                                # nextPass
    w.boolean(s.get('intensify', False))
    w.boolean(s.get('cull', False))
    w.f32(float(s['emissive_multiplier']))
    for c in s['specular_rgb']:
        w.f32(c)
    w.f32(float(s['specular_power']))
    w.symbol(s.get('normal_tex_name', ""))
    w.symbol(s.get('emissive_tex_name', ""))
    w.symbol(s.get('specular_tex_name', ""))
    # rev 55 >= 51 -> no unkSymbol2 here
    # environment_map is a BoolProperty, not a symbol name. RB3 maps it to the
    # "instruments.cube" asset, but that is an RB3 asset and every retail TBRB material
    # ships an EMPTY environMap, so there is no cube map to point at here.
    w.symbol("")                                # environMap
    # rev 55 is NOT > 60 -> no unkBool_0x3C / unkBool_0x42
    w.boolean(True)                             # perPixelLit (rev > 25)
    w.i32(MAT_STENCIL_IGNORE)                   # rev > 27
    w.symbol("")                                # fur
    w.f32(float(s.get('de_normal', 0)))         # rev > 35 - EnumProperty, cast like RB3 does
    w.f32(float(s.get('anisotropy', 0.0)))
    w.f32(1.0)                                  # normalDetailTiling  (rev > 38)
    w.f32(float(s.get('normal_detail_strength', 0.0)))
    w.symbol("")                                # normalDetailMap
    # pointLights: all three retail TBRB materials in the reference milo carry 0, not the
    # True that Program.cs uses for RB3. Match retail - forcing point-light reception on
    # differs from every shipped TBRB material and is not what the game's shaders expect here.
    w.boolean(bool(s.get('point_lights', False)))  # pointLights (rev > 42, >= 45)
    w.boolean(True)                             # projLights  (rev < 63) - absent on RB3
    w.boolean(False)                            # fog        - always false (Program.cs 671)
    w.boolean(False)                            # fadeout    - CLI leaves at C# default
    w.boolean(bool(s.get('color_adjust', False)))  # colorAdjust (rev > 46)
    for c in s.get('rim_rgb', (0.0, 0.0, 0.0)):  # rev > 47
        w.f32(c)
    w.f32(float(s.get('rim_power', 4.0)))
    w.symbol("")                                # rimMap
    w.boolean(bool(s.get('rim_always_show', False)))  # rimAlwaysShow
    w.boolean(False)                            # screenAligned (rev > 48)
    w.i32(MAT_SHADER_VARIATION_VALUES[s.get('shader_variation', 'kShaderVariationNone')])
    for _ in range(3):                          # specular2RGB
        w.f32(0.0)
    w.f32(float(s.get('specular2_power', 0.0)))  # specular2Power (rev > 50)
    # --- rev 52..67 tail. Field identities and defaults are from PikminGuts92's re-notes
    # mat.bt: val_0x160 (rev >= 53, "Default: 1.0") then val_0x170..0x17c (rev > 54,
    # "Default: (1.0, 1.0, 1.0, 0.0)") then alpha_mask NumString (rev > 53) then
    # ps3_force_trilinear (rev > 54). The previous version of this writer mislabeled these
    # as unkInt3/unkColor2/unkSym2 and, critically, wrote val_0x160 as an int32(0) where a
    # float 1.0 belongs, AND emitted an extra stray symbol - desyncing the whole tail. The
    # byte-exact retail Miku_body.mat carries val_0x160=1.0 and val_0x170 block=(0,1,1,1),
    # so those exact values are reproduced here rather than the template's stated generic
    # defaults, since retail is the ground truth for what the game actually accepts.
    w.f32(1.0)                                  # val_0x160  (rev >= 53) - retail: 1.0
    w.f32(0.0)                                  # val_0x170  \
    w.f32(1.0)                                  # val_0x174   } (rev > 54) - retail: (0,1,1,1)
    w.f32(1.0)                                  # val_0x178   |
    w.f32(1.0)                                  # val_0x17c  /
    w.symbol("")                                # alpha_mask (ScreenMask) (rev > 53)
    w.boolean(False)                            # ps3_force_trilinear (rev > 54)
    # NO trailing END_MARKER - confirmed against a real vanilla PS3 keyboard material
    # dumped straight from the game (revision 55, same as this writer targets): every
    # field above lines up byte-for-byte at the identical offset, and the file ends the
    # instant ps3_force_trilinear is written - 0 bytes remaining, no 0xADDEADDE. This is
    # the exact same bug class already found and fixed in write_tbrb_rnd_mesh: `standalone`
    # used to unconditionally append END_MARKER here, which left 4 orphaned trailing bytes
    # on every loose Custom Song Asset .mat file (and, since the mesh-milo-container call
    # site at build_tbrb_mesh_milo_bytes also defaults standalone=True, likely desynced
    # every directory entry written after a material in a full milo too - Tex entries are
    # written immediately after Mat entries there, so any textured full-milo character
    # export carrying more than one material would hit the same 4-byte drift). `standalone`
    # is kept as a parameter for call-site clarity even though it's now a no-op here.


TBRB_CUSTOM_SONG_MAT_REVISION = 28  # See write_tbrb_custom_song_rnd_mat() below.


def write_tbrb_custom_song_rnd_mat(w: MiloWriter, settings, standalone=True):
    """RndMat.Write at revision 28 - the format actually proven to work for the TBRB
    Custom Song Asset loose-file workflow specifically (as opposed to write_tbrb_rnd_mat's
    revision 55, which matches a real vanilla PS3 character/instrument material dumped from
    inside an actual retail milo, but has NOT been confirmed to load through the loose
    'extras' folder + custom-song-compiler pipeline the way this revision has).

    Byte-verified field-for-field against a real reference material confirmed to load
    correctly through a custom song (a mouth-harmonica prop material, source compiler
    unknown, definitely not the glTFMilo CLI). Every field below sits at the exact same
    offset as that 166-byte reference. Two structural deltas from write_tbrb_rnd_mat's
    revision-55 layout, both because rev 28 predates the fields that were added later:
      - NO alpha_threshold field at all (that field is gated "revision > 37" in the
        higher-revision writer; 28 is below that, and the reference file's diffuse_tex
        symbol lands exactly 4 bytes earlier than it would if alpha_threshold were
        present - i.e. removing it, not zeroing it, is what makes the layout line up).
      - NO trailing END_MARKER, same as every other TBRB standalone asset type
        investigated here (RndMesh, and revision-55 RndMat) - the reference file ends
        immediately after its last byte with nothing appended.

    The reference file's tail (everything after specular_tex - presumably some
    combination of environMap/unkShort/perPixelLit/stencilMode/fur equivalents at this
    older revision) is 14 bytes, ALL ZERO. There's no way to recover individual field
    boundaries from an all-zero sample, so rather than guess names/types that could be
    wrong, this writes that tail as one literal 14-byte zero block - it reproduces the
    confirmed-working file exactly regardless of how those bytes actually subdivide.
    If a future reference ever needs one of those trailing fields to be non-default
    (e.g. a real per_pixel_lit=False or a real environMap), this block will need
    unpacking into real named fields at that point.

    `settings` only needs to supply the fields this revision actually uses: blend,
    base_color, pre_lit, use_environment, z_mode, alpha_cut, diffuse_tex_name,
    intensify, cull, emissive_multiplier, specular_rgb, specular_power,
    normal_tex_name, specular_tex_name. Every other key gather_materials_and_textures()
    populates (rim lighting, environment map, refraction, PS3 perf settings, etc.) is
    simply not part of this older format and is ignored here."""
    w.u32(TBRB_CUSTOM_SONG_MAT_REVISION)   # 28
    write_object_fields(w)
    w.i32(MAT_BLEND_VALUES[settings['blend']])
    for c in settings['base_color']:
        w.f32(c)
    w.boolean(settings['pre_lit'])
    w.boolean(settings['use_environment'])
    w.i32(MAT_ZMODE_VALUES[settings['z_mode']])
    w.boolean(settings['alpha_cut'])
    # NO alpha_threshold here - gated off below revision 37 in the newer writer, and
    # its absence is exactly what makes every field after this line line up with the
    # reference file.
    w.boolean(settings['blend'] == 'kBlendSrcAlpha')   # alphaWrite
    w.i32(MAT_TEXGEN_NONE)
    w.i32(MAT_TEXWRAP_REPEAT)
    write_matrix(w, IDENTITY_MATRIX)                   # texXfm
    w.symbol(settings.get('diffuse_tex_name', ""))
    w.symbol("")                                       # nextPass
    w.boolean(settings.get('intensify', False))
    w.boolean(settings.get('cull', False))
    w.f32(float(settings['emissive_multiplier']))
    for c in settings['specular_rgb']:
        w.f32(c)
    w.f32(float(settings['specular_power']))
    w.symbol(settings.get('normal_tex_name', ""))
    w.symbol(settings.get('emissive_tex_name', ""))
    w.symbol(settings.get('specular_tex_name', ""))
    # Unrecovered all-zero tail from the reference file - see docstring.
    w.block(b"\x00" * 14)
    # NO trailing END_MARKER - see docstring.


def write_tbrb_rnd_group(w: MiloWriter, object_names, sort_in_world=False):
    """RndGroup.Write at revision 14 (The Beatles: Rock Band). Verified against all 10
    retail groups across both milos.

    Layout: objFields (rev > 7) + anim + trans + draw + object symbol list (rev > 10) +
    environ (rev < 16) + drawOnly (rev > 12) + lod symbol + lodScreenSize (rev 12..15) +
    sortInWorld (rev > 13).

    Retail uses these purely as LOD buckets: hh_lod00/01/02.grp in the head milo, and
    lod0/1/2.grp in the outfit milo - where the outfit's lod0.grp lists hh_lod00.grp as a
    member, i.e. the outfit group pulls in the head/hands group by name."""
    w.u32(TBRB_GROUP_REVISION)      # 14
    write_object_fields(w)          # rev > 7
    write_rnd_animatable(w)
    write_rnd_trans(w)
    write_rnd_drawable(w, sphere=(0.0, 0.0, 0.0, 0.0))
    w.u32(len(object_names))        # rev > 10
    for n in object_names:
        w.symbol(n)
    w.symbol("")                    # environ   (rev < 16)
    w.symbol("")                    # drawOnly  (rev > 12)
    w.symbol("")                    # lod       (rev 12..15)
    w.f32(0.0)                      # lodScreenSize
    w.boolean(sort_in_world)        # rev > 13
    w.block(END_MARKER)


def build_tbrb_mesh_milo_bytes(root_name, mesh_entries, materials, textures,
                                skeleton_milo_name, platform='ps3', write_tangents=True,
                                sub_dirs=None, sphere_base="bone_pelvis.mesh",
                                force_tbrb_mat_defaults=False):
    """Build a complete The Beatles: Rock Band MESH milo: a Character directory
    (DirectoryMeta rev 25) holding Mesh + Mat + Tex entries only.

    Deliberately FLAT - no RndGroup LOD buckets. Retail groups its meshes into
    hh_lod00/01/02.grp (and the outfit's lod0.grp lists hh_lod00.grp as a member), but
    Saul's injection testing showed meshes display and deform correctly without them, and
    a flat layout matches how the DC and RB3 paths in this plugin already ship.
    write_tbrb_rnd_group() exists if grouping is needed later.

    The ObjectDir subdir points at the SKELETON milo (retail george_headhands_long carries
    exactly one subdir: "george_skeleton.milo"). That reference is how the mesh milo's bone
    names resolve, so the skeleton milo must be exported and installed alongside this one.

    Entry table order must match body write order exactly:
        Mesh -> Mat -> Tex
    """
    body = MiloWriter(big_endian=True)
    total_entries = len(mesh_entries) + len(materials) + len(textures)

    # --- DirectoryMeta (rev 25) ---
    body.u32(TBRB_MILO_REVISION)
    body.symbol("Character")
    body.symbol(root_name)
    body.i32((total_entries + 1) * 2)   # stringTableCount
    body.u32(0)                         # stringTableSize (engine recalculates)
    body.i32(total_entries)
    for (entry_name, *_rest) in mesh_entries:
        body.symbol("Mesh")
        body.symbol(entry_name)
    for (mat_entry_name, _settings) in materials:
        body.symbol("Mat")
        body.symbol(mat_entry_name)
    for (tex_entry_name, *_rest) in textures:
        body.symbol("Tex")
        body.symbol(tex_entry_name)

    # Bounding sphere over every vertex - retail ships a real one here (george's head milo
    # carries centre (0.90, -1.67, 42.02) radius 42.35) and the draw sphere matches it.
    min_c = [float('inf')] * 3
    max_c = [float('-inf')] * 3
    for entry in mesh_entries:
        for v in entry[4]:
            for i, k in enumerate(("x", "y", "z")):
                min_c[i] = min(min_c[i], v[k])
                max_c[i] = max(max_c[i], v[k])
    if min_c[0] == float('inf'):
        bounding = (0.0, 0.0, 0.0, 0.0)
    else:
        cx, cy, cz = ((min_c[i] + max_c[i]) * 0.5 for i in range(3))
        radius = max(
            ((v["x"] - cx) ** 2 + (v["y"] - cy) ** 2 + (v["z"] - cz) ** 2) ** 0.5
            for entry in mesh_entries for v in entry[4])
        bounding = (cx, cy, cz, radius)

    # --- Character dir object ---
    # look_bone is empty for a mesh milo (retail george_headhands_long and the outfit milos
    # both ship "" here). sphereBase is a BONE name in the retail files (bone_pelvis.mesh),
    # NOT the dir's own name - a name that doesn't resolve to a transform is a crash risk.
    # sub_dirs differ by milo kind: a head/hands milo references the skeleton milo, while an
    # outfit milo references the base head/hands milo + a shared outfit milo (it inherits the
    # skeleton transitively). Caller decides; default to the skeleton reference.
    if sub_dirs is None:
        sub_dirs = [skeleton_milo_name]
    write_tbrb_character(body, root_name, bounding=bounding, look_bone="",
                          sub_dirs=sub_dirs, sphere_base=sphere_base)

    # Block boundaries MUST fall on object boundaries (right after an 0xADDEADDE marker),
    # never mid-object. The game reads the milo block-by-block into a fixed ChunkStream
    # buffer; if a block boundary lands inside an object, a single read straddles two chunks
    # and the game asserts -> crash on the loading screen. This is exactly the algorithm the
    # RB3/DC build_milo_bytes path uses (from MiloLib's MiloFile.cs WriteHandler): after each
    # top-level object finishes, if the bytes written since the last boundary exceed
    # MAX_MILO_BLOCK_SIZE, close a block there. Large objects (a mesh can be >700 KB, far over
    # the 0x20000 threshold) therefore each become their own block rather than being sliced.
    block_sizes = []
    last_boundary = [0]

    def _mark():
        bytes_since = len(body.buf) - last_boundary[0]
        if bytes_since > MAX_MILO_BLOCK_SIZE:
            block_sizes.append(bytes_since)
            last_boundary[0] = len(body.buf)

    # --- Mesh bodies ---
    for (entry_name, local_xfm, world_xfm, parent_obj, vertices, faces,
         bone_transforms, mat_name, needs_ao_calc) in mesh_entries:
        write_tbrb_rnd_mesh(body, entry_name, local_xfm, world_xfm, parent_obj,
                             vertices, faces, mat_name=mat_name,
                             bone_transforms=bone_transforms,
                             force_white_vertex_color=needs_ao_calc,
                             write_tangents=write_tangents)
        _mark()

    # --- Mat bodies ---
    for (_mat_entry_name, settings) in materials:
        write_tbrb_rnd_mat(body, settings, force_tbrb_defaults=force_tbrb_mat_defaults)
        _mark()

    # --- Tex bodies ---
    for (_tex_entry_name, width, height, encoding, bpp, block_data, num_mips) in textures:
        write_tbrb_rnd_tex(body, width, height, encoding, bpp, block_data,
                            platform=platform, num_mips=num_mips)
        _mark()

    body_bytes = bytes(body.buf)

    # Trailing remainder after the last recorded boundary (or the whole body as one block if
    # nothing ever exceeded the threshold), matching MiloFile.cs.
    if block_sizes:
        remainder = len(body_bytes) - last_boundary[0]
        if remainder > 0:
            block_sizes.append(remainder)
    else:
        block_sizes = [len(body_bytes)]

    header = MiloWriter(big_endian=False)
    START_OFFSET = 0x810
    header.u32(0xCABEDEAF)
    header.u32(START_OFFSET)
    header.u32(len(block_sizes))
    header.u32(max(block_sizes))
    for sz in block_sizes:
        header.u32(sz)
    header.block(bytes(START_OFFSET - len(header.buf)))

    return bytes(header.buf) + body_bytes


def write_object_dir_base(w: MiloWriter, viewports=None, sub_dirs=None):
    """ObjectDir.cs Write(), fixed at revision 27 (RB3). `sub_dirs` is a list of
    plain path strings (ObjectDir.subDirs is a List<Symbol>, i.e. just external
    file references like "../../shared/char_shared.milo" for the Character
    type's shared-armature dependency - NOT embedded/nested directory content).
    standalone is always False here (RndDir/RndDir-subclass wraps the end
    marker itself)."""
    if viewports is None:
        viewports = [IDENTITY_MATRIX] * 7
    if sub_dirs is None:
        sub_dirs = []

    w.u32(RB3_OBJDIR_REVISION)     # ObjectDir's OWN revision/altRevision (separate from
                                    # objFields below - confirmed against a real CLI-exported
                                    # file; this field was missing entirely in the first version)
    w.u16(0)                       # objFields.altRevision
    w.u16(OBJ_FIELDS_REVISION)     # objFields.revision
    w.symbol("")                   # objFields.type

    w.u32(0)                       # unk1
    w.u32(0)                       # unk2
    w.u32(len(viewports))
    for vp in viewports:
        write_matrix(w, vp)
    w.u32(6)                       # currentViewportIdx

    w.boolean(False)               # inlineProxy
    w.symbol("")                   # proxyPath

    w.u32(len(sub_dirs))
    for path in sub_dirs:
        w.symbol(path)

    w.u8(0)                        # inlineSubDir (kInlineNever)
    w.u32(0)                       # inlineSubDirs.Count
    # inlineSubDirNames / referenceTypes / referenceTypesAlt / inlineSubDirs: all empty

    w.symbol("")                   # unknownString
    w.symbol("")                   # unknownCamReference

    w.boolean(False)               # objFields.root.hasTree
    w.symbol("")                   # objFields.note


def write_rnd_dir(w: MiloWriter, dc3=False):
    """Assets/Rnd/RndDir.cs Write(), fixed at revision 10 (RB3). This is the
    root directory object for the "other" export type. standalone=True here
    (DirectoryMeta.Write always calls the root directory's Write standalone).

    `dc3`: writes the embedded RndDrawable at DC3_DRAW_REVISION instead of RB3's - see
    that constant's comment. No effect for RB3/DC1."""
    draw_rev = DC3_DRAW_REVISION if dc3 else RB3_DRAW_REVISION
    w.u32(RB3_RNDDIR_REVISION)
    write_object_dir_base(w)

    write_rnd_animatable(w)
    write_rnd_drawable(w, sphere=(0.0, 0.0, 0.0, 10000.0),  # confirmed against real file
                        revision=draw_rev)
    write_rnd_trans(w)

    w.symbol("")   # environ
    w.symbol("")   # testEvent (written since revision(10) > 2 and != 9)

    w.block(END_MARKER)


RB3_CHARACTER_REVISION = 17          # Character.revision
RB3_CHARTEST_REVISION = 15           # Character.CharacterTesting.revision
CHAR_SHARED_MILO_PATH = "../../shared/char_shared.milo"  # matches DirBuilder.cs exactly

# Instruments (guitar/bass/etc.) must NOT use char_shared.milo (the human skeleton) as
# their subdir. Confirmed two ways:
#  - The RB3 decomp has a live assertion string in BandCharacter.cpp:
#    "instruments can only have one subdir, which is the resource or colorpalettes.milo"
#  - Byte-diffing a real vanilla guitar (greenday_blue.milo_xbox) against a broken custom
#    export: the real guitar's single subdir is exactly "../shared/colorpalettes.milo"
#    (28 bytes, one "../"), while the broken custom export wrongly carried
#    "../../shared/char_shared.milo" (the character path). Pointing a guitar at the
#    character skeleton is a likely hard-crash / misload, so instruments get this instead.
INSTRUMENT_SHARED_MILO_PATH = "../shared/colorpalettes.milo"


# =====================================================================================
# RB3 base skeleton bone names, ported verbatim from glTFMilo's Source/BoneNames.cs
# (commit "Import armature; get spec power and color from material"). These are all
# the bones that already exist in the shared character armature (char_shared.milo).
# The CLI tool generates a top-level "Trans" entry for every skin-joint bone in the
# glTF EXCEPT these - because re-emitting a bone that the shared armature already
# provides would duplicate it (and, per hard experience noted elsewhere in this file,
# embedding the base armature bones directly in a character milo crashes the game).
# So only CUSTOM bones (e.g. user-added hair physics bones) get written as Trans
# entries; they augment the shared skeleton rather than replacing it. A custom bone
# whose parent IS one of these base bones still references that base bone by name as
# its parentObj - that reference resolves at runtime against the merged shared
# armature, which is exactly how a hair strand attaches to e.g. bone_head.mesh.
# =====================================================================================

RB3_SKELETON_BONES = frozenset({
    "bone_drumbase.mesh", "bone_deform_drumkit.mesh", "bone_L-cymbal_stand.mesh",
    "bone_L-cymbal_shake.mesh", "bone_L-cymbal.mesh", "bone_target_L-cymbal.mesh",
    "bone_L-tip_L-cymbal.mesh", "bone_R-tip_L-cymbal.mesh", "bone_L-tom_shake.mesh",
    "bone_R-cymbal_stand.mesh", "bone_R-cymbal_shake.mesh", "bone_R-cymbal.mesh",
    "bone_target_R-cymbal.mesh", "bone_L-tip_R-cymbal.mesh", "bone_R-tip_R-cymbal.mesh",
    "bone_R-tom_shake.mesh", "bone_hihat_bottom_rot.mesh", "bone_hihat_pedal.mesh",
    "bone_hihat_pos.mesh", "bone_hihat_rot.mesh", "bone_target_hihat.mesh",
    "bone_L-tip_hihat.mesh", "bone_R-tip_hihat.mesh", "bone_kick.mesh", "bone_kickanim.mesh",
    "bone_cowbell_shake.mesh", "bone_target_R-cowbell.mesh", "bone_R-tip_cowbell.mesh",
    "bone_kick-face.mesh", "bone_kick_head_center.mesh", "bone_kick_head_middle.mesh",
    "bone_kick_head_outer.mesh", "bone_paddle1.mesh", "bone_paddle2.mesh",
    "bone_ride_stand.mesh", "bone_ride_shake.mesh", "bone_ride.mesh", "bone_target_ride.mesh",
    "bone_R-tip_ride.mesh", "bone_target_L-tom.mesh", "bone_L-tip_L-tom.mesh",
    "bone_R-tip_L-tom.mesh", "bone_target_R-tom.mesh", "bone_L-tip_R-tom.mesh",
    "bone_R-tip_R-tom.mesh", "bone_target_floor_tom.mesh", "bone_L-tip_floor_tom.mesh",
    "bone_R-tip_floor_tom.mesh", "bone_target_hihat_pedal.mesh", "bone_L-toe_hihat.mesh",
    "bone_target_kick_pedal.mesh", "bone_R-toe_kick.mesh", "bone_target_snare.mesh",
    "bone_L-tip_snare.mesh", "bone_R-tip_snare.mesh", "bone_deform_seatbase.mesh",
    "bone_deform_seat.mesh", "bone_floortombase.mesh", "bone_floortom_shake.mesh",
    "bone_snarebase_rot.mesh", "bone_snare_shake.mesh", "bone_keyboard_base.mesh",
    "bone_keyboard_shake.mesh", "bone_keyboard_height.mesh", "bone_key_a1.mesh",
    "bone_key_a2.mesh", "bone_key_a3.mesh", "bone_key_asharp1.mesh", "bone_key_asharp2.mesh",
    "bone_key_asharp3.mesh", "bone_key_b1.mesh", "bone_key_b2.mesh", "bone_key_b3.mesh",
    "bone_key_c1.mesh", "bone_key_c2.mesh", "bone_key_c3.mesh", "bone_key_c4.mesh",
    "bone_key_csharp1.mesh", "bone_key_csharp2.mesh", "bone_key_csharp3.mesh",
    "bone_key_d1.mesh", "bone_key_d2.mesh", "bone_key_d3.mesh", "bone_key_dsharp1.mesh",
    "bone_key_dsharp2.mesh", "bone_key_dsharp3.mesh", "bone_key_e1.mesh", "bone_key_e2.mesh",
    "bone_key_e3.mesh", "bone_key_f1.mesh", "bone_key_f2.mesh", "bone_key_f3.mesh",
    "bone_key_fsharp1.mesh", "bone_key_fsharp2.mesh", "bone_key_fsharp3.mesh",
    "bone_key_g1.mesh", "bone_key_g2.mesh", "bone_key_g3.mesh", "bone_key_gsharp1.mesh",
    "bone_key_gsharp2.mesh", "bone_key_gsharp3.mesh", "bone_target_keyboard_lh.mesh",
    "bone_keys_lh.mesh", "bone_target_keyboard_rh.mesh", "bone_keys_rh.mesh",
    "spot_keyboard_grab.mesh", "bone_L-hand_keys.mesh", "bone_R-hand_keys.mesh",
    "spot_keys_left.mesh", "spot_keys_left_lh.mesh", "spot_keys_right.mesh",
    "spot_keys_right_lh.mesh", "bone_mic_stand_bottom.mesh", "bone_mic_stand_top.mesh",
    "bone_L-hand_mic_stand.mesh", "bone_R-hand_mic_stand.mesh", "bone_mic.mesh",
    "bone_L-hand_mic.mesh", "bone_R-hand_mic.mesh", "bone_mic_mic_stand.mesh",
    "bone_pelvis.mesh", "bone_L-butt.mesh", "exo_L-butt.mesh", "bone_L-thigh.mesh",
    "bone_L-knee.mesh", "bone_L-ankle.mesh", "bone_L-toe.mesh", "exo_L-toe.mesh",
    "exo_L-ankle.mesh", "exo_L-knee.mesh", "exo_L-boot-size.mesh", "exo_L-calf.mesh",
    "exo_L-thigh.mesh", "exo_L-thigh_muscle.mesh", "bone_R-butt.mesh", "exo_R-butt.mesh",
    "bone_R-thigh.mesh", "bone_R-knee.mesh", "bone_R-ankle.mesh", "bone_R-toe.mesh",
    "exo_R-toe.mesh", "exo_R-ankle.mesh", "exo_R-knee.mesh", "exo_R-boot-size.mesh",
    "exo_R-calf.mesh", "exo_R-thigh.mesh", "exo_R-thigh_muscle.mesh", "bone_spine1.mesh",
    "bone_spine2.mesh", "bone_spine3.mesh", "bone_L-breast.mesh", "exo_L-breast.mesh",
    "bone_L-clavicle.mesh", "bone_L-deltoid-target.mesh", "bone_L-shoulderTwist1.mesh",
    "exo_L-shoulderTwist1.mesh", "bone_L-shoulderTwist2.mesh", "exo_L-shoulderTwist2.mesh",
    "bone_L-shoulderTwist3.mesh", "exo_L-shoulderTwist3.mesh", "bone_L-shoulderTwist4.mesh",
    "exo_L-shoulderTwist4.mesh", "bone_L-upperArm.mesh", "bone_L-foreArm.mesh",
    "bone_L-hand.mesh", "bone_L-index01.mesh", "bone_L-index02.mesh", "bone_L-index03.mesh",
    "exo_L-index03.mesh", "spot_L-index_tip.mesh", "exo_L-index02.mesh", "exo_L-index01.mesh",
    "bone_L-middlefinger01.mesh", "bone_L-middlefinger02.mesh", "bone_L-middlefinger03.mesh",
    "exo_L-middlefinger03.mesh", "spot_L-middlefinger_tip.mesh", "exo_L-middlefinger02.mesh",
    "exo_L-middlefinger01.mesh", "bone_L-pinky-base.mesh", "bone_L-pinky01.mesh",
    "bone_L-pinky02.mesh", "bone_L-pinky03.mesh", "exo_L-pinky03.mesh",
    "spot_L-pinky_tip.mesh", "exo_L-pinky02.mesh", "exo_L-pinky01.mesh",
    "exo_L-pinky-base.mesh", "bone_L-ringfinger-base.mesh", "bone_L-ringfinger01.mesh",
    "bone_L-ringfinger02.mesh", "bone_L-ringfinger03.mesh", "exo_L-ringfinger03.mesh",
    "spot_L-ringfinger_tip.mesh", "exo_L-ringfinger02.mesh", "exo_L-ringfinger01.mesh",
    "exo_L-ringfinger-base.mesh", "bone_L-thumb01.mesh", "bone_L-thumb02.mesh",
    "bone_L-thumb03.mesh", "exo_L-thumb03.mesh", "spot_L-thumb_tip.mesh", "exo_L-thumb02.mesh",
    "exo_L-thumb01.mesh", "bone_L-tip_keys.mesh", "bone_target_L-hand.mesh",
    "bone_L-stick.mesh", "bone_L-tip.mesh", "bone_tip.mesh", "exo_L-hand.mesh",
    "exo_L-forearm.mesh", "bone_L-foreTwist1.mesh", "bone_L-foreTwist2.mesh",
    "exo_L-foreTwist2.mesh", "exo_L-foreTwist1.mesh", "bone_L-shoulderTwist5.mesh",
    "exo_L-shoulderTwist5.mesh", "exo_L-upperArm.mesh", "bone_L-upperTwist1.mesh",
    "bone_L-upperTwist2.mesh", "exo_L-upperTwist2.mesh", "exo_L-upperTwist1.mesh",
    "exo_L-clavicle.mesh", "exo_L-shoulder.mesh", "bone_L-collar-base.mesh",
    "bone_L-collar.mesh", "exo_L-collar.mesh", "bone_L-deltoid-base.mesh",
    "bone_L-deltoid.mesh", "exo_L-deltoid.mesh", "bone_R-breast.mesh", "exo_R-breast.mesh",
    "bone_R-clavicle.mesh", "bone_R-deltoid-target.mesh", "bone_R-shoulderTwist1.mesh",
    "exo_R-shoulderTwist1.mesh", "bone_R-shoulderTwist2.mesh", "exo_R-shoulderTwist2.mesh",
    "bone_R-shoulderTwist3.mesh", "exo_R-shoulderTwist3.mesh", "bone_R-shoulderTwist4.mesh",
    "exo_R-shoulderTwist4.mesh", "bone_R-upperArm.mesh", "bone_R-foreArm.mesh",
    "bone_R-hand.mesh", "bone_R-index01.mesh", "bone_R-index02.mesh", "bone_R-index03.mesh",
    "exo_R-index03.mesh", "spot_R-index_tip.mesh", "exo_R-index02.mesh", "exo_R-index01.mesh",
    "bone_R-middlefinger01.mesh", "bone_R-middlefinger02.mesh", "bone_R-middlefinger03.mesh",
    "exo_R-middlefinger03.mesh", "spot_R-middlefinger_tip.mesh", "exo_R-middlefinger02.mesh",
    "exo_R-middlefinger01.mesh", "bone_R-pinky-base.mesh", "bone_R-pinky01.mesh",
    "bone_R-pinky02.mesh", "bone_R-pinky03.mesh", "exo_R-pinky03.mesh",
    "spot_R-pinky_tip.mesh", "exo_R-pinky02.mesh", "exo_R-pinky01.mesh",
    "exo_R-pinky-base.mesh", "bone_R-ringfinger-base.mesh", "bone_R-ringfinger01.mesh",
    "bone_R-ringfinger02.mesh", "bone_R-ringfinger03.mesh", "exo_R-ringfinger03.mesh",
    "spot_R-ringfinger_tip.mesh", "exo_R-ringfinger02.mesh", "exo_R-ringfinger01.mesh",
    "exo_R-ringfinger-base.mesh", "bone_R-thumb01.mesh", "bone_R-thumb02.mesh",
    "bone_R-thumb03.mesh", "bone_pick.mesh", "exo_R-thumb03.mesh", "spot_R-thumb_tip.mesh",
    "exo_R-thumb02.mesh", "exo_R-thumb01.mesh", "bone_R-tip_keys.mesh",
    "bone_target_R-hand.mesh", "bone_R-stick.mesh", "bone_R-tip.mesh", "exo_R-hand.mesh",
    "exo_R-forearm.mesh", "bone_R-foreTwist1.mesh", "bone_R-foreTwist2.mesh",
    "exo_R-foreTwist2.mesh", "exo_R-foreTwist1.mesh", "bone_R-shoulderTwist5.mesh",
    "exo_R-shoulderTwist5.mesh", "exo_R-upperArm.mesh", "bone_R-upperTwist1.mesh",
    "bone_R-upperTwist2.mesh", "exo_R-upperTwist2.mesh", "exo_R-upperTwist1.mesh",
    "exo_R-clavicle.mesh", "exo_R-shoulder.mesh", "bone_R-collar-base.mesh",
    "exo_R-collar.mesh", "bone_R-deltoid-base.mesh", "bone_R-deltoid.mesh",
    "exo_R-deltoid.mesh", "bone_neck.mesh", "bone_head_nod.mesh", "bone_head.mesh",
    "bone_L-brow1.mesh", "exo_L-brow1.mesh", "bone_L-brow2.mesh", "bone_L-brow2-eyeshape.mesh",
    "exo_L-brow2.mesh", "bone_L-brow3.mesh", "bone_L-brow3-eyeshape.mesh", "exo_L-brow3.mesh",
    "bone_L-cheek.mesh", "bone_L-cheek2.mesh", "bone_L-crease.mesh", "bone_L-eye-base.mesh",
    "bone_L-eye.mesh", "bone_L-lid.mesh", "bone_L-lid-blink.mesh", "bone_L-eyelid-low.mesh",
    "bone_L-eyelid-low-blink.mesh", "bone_L-lipcorner.mesh", "bone_L-nose.mesh",
    "bone_R-brow1.mesh", "exo_R-brow1.mesh", "bone_R-brow2.mesh", "exo_R-brow2.mesh",
    "bone_R-brow3.mesh", "bone_R-brow3-eyeshape.mesh", "exo_R-brow3.mesh", "bone_R-cheek.mesh",
    "bone_R-cheek2.mesh", "bone_R-crease.mesh", "bone_R-eye-base.mesh", "bone_R-eye.mesh",
    "bone_R-lid.mesh", "bone_R-lid-blink.mesh", "bone_R-eyelid-low.mesh",
    "bone_R-eyelid-low-blink.mesh", "bone_R-lipcorner.mesh", "bone_R-nose.mesh",
    "bone_brow-low.mesh", "bone_brow-mid.mesh", "bone_earrings_L.mesh", "bone_earrings_R.mesh",
    "bone_eyes.mesh", "bone_forehead.mesh", "bone_glasses.mesh", "bone_glasses_L.mesh",
    "bone_glasses_R.mesh", "bone_hair.mesh", "bone_jaw_base.mesh", "bone_jaw.mesh",
    "bone_chin.mesh", "bone_lowerteeth-base.mesh", "bone_lowlip_left.mesh",
    "bone_lowlip_mid.mesh", "bone_lowlip_right.mesh", "bone_upperteeth-base.mesh",
    "bone_tongue1.mesh", "bone_tongue2.mesh", "bone_tongue3.mesh", "bone_tongue4.mesh",
    "bone_liptop_left.mesh", "bone_liptop_mid.mesh", "bone_liptop_right.mesh",
    "bone_maskstrap.mesh", "bone_nose.mesh", "bone_target_mouth.mesh", "bone_mic_mouth.mesh",
    "exo_head.mesh", "bone_mic_stand_mouth.mesh", "bone_neckTwist.mesh", "exo_neckTwist.mesh",
    "exo_neck.mesh", "exo_neck-front.mesh", "exo_neckbase.mesh", "exo_spine3.mesh",
    "exo_armpit.mesh", "exo_chest-L1.mesh", "exo_chest-L2.mesh", "exo_chest-R1.mesh",
    "exo_chest-R2.mesh", "exo_chest-mid.mesh", "exo_spine2.mesh", "exo_lowerchest-L.mesh",
    "exo_lowerchest-R.mesh", "exo_lowerchest-mid.mesh", "exo_upperbelly-L.mesh",
    "exo_upperbelly-L1.mesh", "exo_upperbelly-L2.mesh", "exo_upperbelly-R.mesh",
    "exo_upperbelly-R1.mesh", "exo_upperbelly-R2.mesh", "exo_upperbelly-mid.mesh",
    "exo_stomach.mesh", "bone_target_belly.mesh", "bone_guitar.mesh",
    "bone_guitar_lh_mod.mesh", "bone_bridge.mesh", "bone_vibrate_hi.mesh",
    "bone_vibrate_low.mesh", "bone_nut.mesh", "bone_pos_string01.mesh",
    "bone_bend_string01.mesh", "bone_pos_string02.mesh", "bone_bend_string02.mesh",
    "bone_pos_string03.mesh", "bone_bend_string03.mesh", "bone_pos_string04.mesh",
    "bone_bend_string04.mesh", "bone_pos_string05.mesh", "bone_bend_string05.mesh",
    "bone_pos_string06.mesh", "bone_bend_string06.mesh", "bone_target_fret.mesh",
    "bone_tip_fret.mesh", "bone_target_pegs.mesh", "bone_L-hand_pegs.mesh",
    "bone_R-hand_pegs.mesh", "bone_target_strum.mesh", "bone_pick_strum.mesh",
    "spot_bridge01.mesh", "spot_bridge02.mesh", "spot_bridge03.mesh", "spot_bridge04.mesh",
    "spot_bridge05.mesh", "spot_bridge06.mesh", "spot_neck_fret01.mesh",
    "spot_neck_fret02.mesh", "spot_neck_fret03.mesh", "spot_neck_fret04.mesh",
    "spot_neck_fret05.mesh", "spot_neck_fret06.mesh", "bone_L-hand_fret.mesh",
    "bone_R-hand_fret.mesh", "spot_neck_fret07.mesh", "spot_neck_fret08.mesh",
    "spot_neck_fret09.mesh", "spot_neck_fret10.mesh", "spot_neck_fret11.mesh",
    "spot_neck_fret12.mesh", "spot_neck_fret13.mesh", "spot_neck_fret14.mesh",
    "spot_neck_fret15.mesh", "spot_neck_fret16.mesh", "spot_neck_fret17.mesh",
    "spot_neck_fret18.mesh", "spot_neck_fret19.mesh", "spot_neck_fret20.mesh",
    "spot_nut01.mesh", "spot_nut02.mesh", "spot_nut03.mesh", "spot_nut04.mesh",
    "spot_nut05.mesh", "spot_nut06.mesh", "exo_spine1.mesh", "exo_midbelly-L1.mesh",
    "exo_midbelly-L2.mesh", "exo_midbelly-L3.mesh", "exo_midbelly-R1.mesh",
    "exo_midbelly-R2.mesh", "exo_midbelly-R3.mesh", "exo_midbelly-mid.mesh",
    "bone_target_L-hip.mesh", "bone_L-hand_hip.mesh", "bone_mic_L-hip.mesh",
    "bone_target_R-hip.mesh", "bone_R-hand_hip.mesh", "bone_mic_R-hip.mesh", "exo_pelvis.mesh",
    "exo_butt.mesh", "exo_lowerbelly-L.mesh", "exo_lowerbelly-R.mesh",
    "exo_lowerbelly-mid.mesh", "bone_prop0.mesh", "bone_prop1.mesh", "bone_prop2.mesh",
    "bone_prop3.mesh", "spot_navel.mesh", "spot_neck.mesh", "neutral_bone",
})


def write_character_testing(w: MiloWriter, dist_map="none"):
    """Character.CharacterTesting.Write(), fixed at revision 15 (RB3). Ported
    directly from glTFMilo's own DirBuilder defaults - everything empty/false/0
    except distMap, which the CLI tool always sets to "none"."""
    w.u32(RB3_CHARTEST_REVISION)
    w.symbol("")            # driver
    w.symbol("")            # clip1
    w.symbol("")            # clip2
    w.symbol("")            # teleportTo
    w.symbol("")            # teleportFrom
    w.symbol(dist_map)      # distMap
    w.u32(0)                # transition
    w.boolean(False)        # cycleTransition
    w.u32(0)                # internalTransition
    w.boolean(False)        # metronome
    w.boolean(False)        # zeroTravel
    w.boolean(False)        # showScreenSize
    w.boolean(False)        # footExtents


def write_character(w: MiloWriter, root_name, is_instrument=False, dc3=False):
    """Assets/Char/Character.cs Write(), fixed at revision 17 (RB3). Character
    extends RndDir, so this wraps write_rnd_dir's structure (same anim/draw/trans/
    environ/testEvent) with LODs/shadows/selfShadow/sphereBase/bounding/frozen/
    minLod/translucentGroup/charTest layered on top. Ported from glTFMilo's own
    DirBuilder.BuildCharacterDirectory, which leaves nearly everything at
    defaults (empty LODs, empty shadows, sphereBase = the character's own name).

    NOTE: unlike the static-mesh RndDir path, this hasn't been verified against
    a real CLI-exported reference file yet - only source-read from glTFMilo/
    MiloLib. Flag anything that comes back wrong here in particular.

    NOTE 2: Character.revision itself is still hardcoded to RB3_CHARACTER_REVISION
    (17) even when dc3=True. Byte-parsing retail angel01.milo_xbox shows real DC3
    characters are revision 21 with a much larger Character-specific tail (populated
    LODs, non-empty shadows, and a CharClipSet embedded as entry 0) - that mismatch is
    real but is a SEPARATE, larger fix than the Drawable revision handled here, and is
    being tracked/fixed separately rather than folded into this change.

    `dc3`: writes the embedded RndDrawable at DC3_DRAW_REVISION instead of RB3's - see
    that constant's comment. No effect for RB3/DC1."""
    draw_rev = DC3_DRAW_REVISION if dc3 else RB3_DRAW_REVISION
    w.u32(RB3_CHARACTER_REVISION)

    # base.Write() -> RndDir.Write(standalone=False)
    w.u32(RB3_RNDDIR_REVISION)
    # Characters point at the shared human skeleton; instruments point at
    # colorpalettes.milo (NOT char_shared.milo - that's the character skeleton and
    # crashes/misloads on an instrument). See the INSTRUMENT_SHARED_MILO_PATH comment.
    if is_instrument:
        sub_dirs = [INSTRUMENT_SHARED_MILO_PATH]
    else:
        sub_dirs = [CHAR_SHARED_MILO_PATH]
    write_object_dir_base(w, sub_dirs=sub_dirs)
    write_rnd_animatable(w)
    write_rnd_drawable(w, sphere=(0.0, 0.0, 0.0, 10000.0), revision=draw_rev)
    write_rnd_trans(w)
    w.symbol("")   # environ
    w.symbol("")   # testEvent
    # (no end marker here - RndDir.Write's standalone is False when called as a base class)

    # Character-specific fields (revision(17) < 4 is False, and entry.isProxy is False
    # for the root directory, so the "!entry.isProxy" branch is always taken here)
    w.u32(0)                    # lods.Count (DirBuilder never populates this)
    w.u32(0)                    # shadows.Count (revision(17) >= 17 -> new list-style format)
    w.boolean(False)            # selfShadow
    w.symbol(root_name)         # sphereBase - DirBuilder sets this to the dir's own name
    write_sphere(w, 0.0, 0.0, 0.0, 0.0)   # bounding
    w.boolean(False)            # frozen
    w.i32(0)                    # minLod (kLODPerFrame)
    w.symbol("")                # translucentGroup
    write_character_testing(w, dist_map="none")

    w.block(END_MARKER)         # Character.Write's own standalone=True end marker


def compute_group_sizes(face_count):
    """groupSizes is a List<byte>, so each entry maxes out at 255. Confirmed against
    a real CLI-exported file: a 12-triangle cube produced groupSizes=[12] (a single
    group covering every face). The multi-group chunking below for >255 faces is
    an untested extrapolation - flag if a large mesh doesn't load right."""
    if face_count <= 0:
        return []
    sizes = []
    remaining = face_count
    while remaining > 0:
        chunk = min(remaining, 255)
        sizes.append(chunk)
        remaining -= chunk
    return sizes


MAX_BONES_PER_MESH = 40  # RndMesh::MaxBones() in the real engine (dc3-decomp Mesh.h) -
                          # exceeding this crashes the game. Confirmed independently by
                          # both the decompiled engine and the original glTFMilo CLI tool.


def write_bone_transform(w: MiloWriter, name, transform12):
    """RndMesh.BoneTransform.Write() - just a Symbol + Matrix, no revision gating."""
    w.symbol(name)
    write_matrix(w, transform12)


RB3_MAT_REVISION = 68  # 0x44 - RndMat.revision

# Dance Central 3 uses a completely restructured material class hierarchy (confirmed against
# rjkiv/dc3-decomp: RndMat -> BaseMaterial -> Hmx::Object) rather than RB3's single flat
# RndMat. This drives an entirely separate DC3 material writer (write_dc3_rnd_mat). The three
# revision numbers below are each written as their own u32, back to back, at the very start of
# the material body - matching packRevs(alt=0, rev) == rev for each class's SAVE_REVS:
#   RndMat::Save        -> SAVE_REVS(0x46, 0)  -> 0x00000046
#   BaseMaterial::Save  -> SAVE_REVS(8, 0)     -> 0x00000008
#   Hmx::Object::Save   -> bs << 2             -> 0x00000002
# Verified byte-for-byte against retail emilia01.milo_xbox (every Mat body starts
# 00 00 00 46 | 00 00 00 08 | 00 00 00 02 | ...).
DC3_MAT_REVISION = 0x46          # 70 - RndMat.revision (outer)
DC3_BASEMAT_REVISION = 8         # BaseMaterial.revision (superclass)
DC3_OBJ_REVISION = 2             # Hmx::Object.revision (base)

# The MetaMaterial (.mmat) template every DC3 material references as its trailing field.
# The runtime loads these into a GLOBAL template dir (RndMat::sMetaMaterials, populated from
# the game's own shared material resources - NOT shipped inside the character milo), and
# resolves this symbol by NAME against that dir, then pulls default/forced property values
# from it via UpdatePropertiesFromMetaMat(). A custom character therefore just needs to name a
# template the game already has loaded; it does not need to (and cannot usefully) embed the
# .mmat itself.
#
# CHOOSING THE RIGHT TEMPLATE MATTERS - it drives the actual shading path. Byte-parsing retail
# emilia01 shows exactly which template each surface type uses:
#     skin / body / face   -> char_basic_skin.mmat  (or char_basic_skin_rim.mmat for rim light)
#     clothing / outfit    -> char_basic_rim.mmat
#     hair                 -> char_basic_hair.mmat  (backfaces: char_basic_hair_backfaces.mmat)
#     eyes / teeth         -> char_basic_skin_nospec.mmat
#     LOD meshes           -> char_basic_lod.mmat
#     char_basic_color.mmat -> a FLAT, self-lit color template used only by retail's invisible
#                              utility materials. Pointing a real textured surface at it renders
#                              the surface as a uniform glowing color (no proper lighting) - this
#                              is the "glowing orange character" failure mode. Do NOT use it for
#                              a visible character surface.
# The default below is char_basic_skin.mmat: the correct choice for the most common case (a lit,
# textured character body/skin). Override per-material in Material Properties for clothing/hair/
# eyes so each surface uses its matching template.
DC3_DEFAULT_META_MATERIAL = "char_basic_skin.mmat"

# The game's Transform::Reset() leaves the two off-diagonal elements m13 and m21 as negative
# zero (0x80000000), so a freshly-Reset() texXfm - which is what every retail emilia material
# carries - serializes with -0.0 in those two slots rather than +0.0. Mathematically identical
# (-0.0 == 0.0, renders the same), but using it makes DC3 output byte-for-byte reproduce retail
# files. Confirmed against every material in emilia01.milo_xbox.
DC3_TEXXFM_IDENTITY = (1.0, 0.0, -0.0, -0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)

# =====================================================================================
# Texture support (Xbox 360 only)
#
# Ported from MiloLib/Assets/Rnd/RndBitmap.cs + RndTex.cs. The one detail that would
# have silently produced garbage textures if missed: RndBitmap.ConvertToImage() shows
# that on the Xbox 360 platform, raw DXT block bytes are stored byte-swapped in 2-byte
# pairs within every 4-byte group (a consequence of the console's big-endian CPU vs.
# the little-endian bit-packing DXT itself uses) - so every 4 bytes of standard DXT
# data needs (b1,b0,b3,b2) reordering before being written into the .milo.
#
# DXT1/BC1, DXT5/BC3, and BC5/ATI2 encoders below are a simple bounding-box block
# compressor (not the highest quality - no cluster-fit optimization - but produces
# correct, valid block data). Verified via round-trip against an independent decoder
# before integrating.
# =====================================================================================

RB3_TEX_REVISION = 11      # RndTex.revision
RB3_BITMAP_REVISION = 1    # RndBitmap.revision

TEX_ENCODING_DXT1 = 8      # RndBitmap.TextureEncoding.DXT1_BC1
TEX_ENCODING_DXT5 = 24     # RndBitmap.TextureEncoding.DXT5_BC3
TEX_ENCODING_ATI2 = 32     # RndBitmap.TextureEncoding.ATI2_BC5
TEX_TYPE_REGULAR = 1       # RndTex.Type.kRegular

_TEX_ENCODING_NAMES.update({
    TEX_ENCODING_DXT1: "DXT1/BC1",
    TEX_ENCODING_DXT5: "DXT5/BC3",
    TEX_ENCODING_ATI2: "ATI2/BC5",
})


def _color_to_565(r, g, b):
    return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)


def _565_to_rgb(c):
    r = (c >> 11) & 0x1F
    g = (c >> 5) & 0x3F
    b = c & 0x1F
    r = (r << 3) | (r >> 2)
    g = (g << 2) | (g >> 4)
    b = (b << 3) | (b >> 2)
    return r, g, b


def _pca_axis_3(pixels):
    """Returns (mean, axis) for a list of (r,g,b) tuples - the block's mean color and
    the dominant principal axis (unit vector) of its color distribution, found via a
    few iterations of power iteration on the 3x3 covariance matrix. This is the same
    class of approach real BC1 encoders use (PCA / cluster-fit) to pick a color line
    through the ACTUAL pixel distribution, instead of independently taking the max/min
    of each channel (which can synthesize an endpoint color that doesn't resemble any
    real pixel in the block - see _encode_bc1_block's docstring)."""
    n = len(pixels)
    mr = sum(p[0] for p in pixels) / n
    mg = sum(p[1] for p in pixels) / n
    mb = sum(p[2] for p in pixels) / n
    cov_rr = cov_rg = cov_rb = cov_gg = cov_gb = cov_bb = 0.0
    for (r, g, b) in pixels:
        dr, dg, db = r - mr, g - mg, b - mb
        cov_rr += dr * dr
        cov_rg += dr * dg
        cov_rb += dr * db
        cov_gg += dg * dg
        cov_gb += dg * db
        cov_bb += db * db

    # Seed power iteration along whichever channel has the widest range - converges
    # in very few iterations for the small, low-rank 4x4-pixel color distributions
    # found in a single block.
    r_range = max(p[0] for p in pixels) - min(p[0] for p in pixels)
    g_range = max(p[1] for p in pixels) - min(p[1] for p in pixels)
    b_range = max(p[2] for p in pixels) - min(p[2] for p in pixels)
    if r_range >= g_range and r_range >= b_range:
        ax, ay, az = 1.0, 0.0, 0.0
    elif g_range >= b_range:
        ax, ay, az = 0.0, 1.0, 0.0
    else:
        ax, ay, az = 0.0, 0.0, 1.0

    for _ in range(8):
        nx = cov_rr * ax + cov_rg * ay + cov_rb * az
        ny = cov_rg * ax + cov_gg * ay + cov_gb * az
        nz = cov_rb * ax + cov_gb * ay + cov_bb * az
        length = (nx * nx + ny * ny + nz * nz) ** 0.5
        if length < 1e-9:
            break
        ax, ay, az = nx / length, ny / length, nz / length

    return (mr, mg, mb), (ax, ay, az)


# Weight of c0 (vs c1) reconstructed by each of the 4 palette indices, in the
# opaque/4-unique-color palette ordering used below - index 0 is pure c0 (t=1),
# index 1 is pure c1 (t=0), indices 2/3 are the 2/3 and 1/3 blends.
_BC1_INDEX_C0_WEIGHT = (1.0, 0.0, 2.0 / 3.0, 1.0 / 3.0)


def _refine_bc1_endpoints(pixels, indices):
    """Given a fixed per-pixel index assignment (0-3), solves the 3x2 (per RGB
    channel) least-squares problem for the two endpoint colors that best reconstruct
    the block under that assignment - i.e. the true globally-optimal endpoints for
    THIS clustering, rather than the block's raw min/max. This is the same
    "endpoint refinement" step real cluster-fit BC1 encoders use after an initial
    PCA-based guess. Returns (c0_565, c1_565), or None if the system is degenerate
    (e.g. every pixel mapped to the same index, so there's nothing to solve for)."""
    ts = [_BC1_INDEX_C0_WEIGHT[(indices >> (2 * i)) & 0x3] for i in range(16)]
    sum_tt = sum(t * t for t in ts)
    sum_t1mt = sum(t * (1.0 - t) for t in ts)
    sum_1mt1mt = sum((1.0 - t) * (1.0 - t) for t in ts)
    det = sum_tt * sum_1mt1mt - sum_t1mt * sum_t1mt
    if abs(det) < 1e-6:
        return None

    c0 = [0, 0, 0]
    c1 = [0, 0, 0]
    for ch in range(3):
        sum_p_t = sum(ts[i] * pixels[i][ch] for i in range(16))
        sum_p_1mt = sum((1.0 - ts[i]) * pixels[i][ch] for i in range(16))
        c0v = (sum_p_t * sum_1mt1mt - sum_p_1mt * sum_t1mt) / det
        c1v = (sum_tt * sum_p_1mt - sum_t1mt * sum_p_t) / det
        c0[ch] = max(0, min(255, round(c0v)))
        c1[ch] = max(0, min(255, round(c1v)))

    return _color_to_565(*c0), _color_to_565(*c1)


def _encode_bc1_block(pixels, force_four_color=False):
    """pixels: 16 (r,g,b,a) tuples, 0-255. Returns 8 bytes.

    Endpoint selection previously took the max/min of each RGB channel
    INDEPENDENTLY across the block (max(rs), max(gs), max(bs)) - three values that
    can come from three different pixels, so the resulting "endpoint color" may not
    resemble any real pixel in the block at all. That's a known-inferior technique
    (this file's own earlier comments already flagged it as "not the highest
    quality - no cluster-fit optimization"), and byte-for-byte comparison against a
    real CLI-exported reference texture confirmed it measurably: a "blockiness"
    metric (edge energy at 4x4 block boundaries vs interior, in a smooth skin-tone
    gradient) came out visibly higher for this encoder's output than the CLI's.

    Replaced with PCA-based endpoint selection (fit a line through the actual 3D
    color distribution of the block via _pca_axis_3, take the extreme projections
    along it as the two endpoints), optionally followed by a least-squares
    refinement pass (_refine_bc1_endpoints) that re-solves for the best endpoint
    colors given the resulting index assignment - the same two-stage approach real
    cluster-fit BC1 encoders use, just without further iterating to full
    convergence.

    Both the PCA step and the refinement step are only ever KEPT if they measure a
    lower actual quantized reconstruction error than the alternative - confirmed
    necessary empirically: naive-endpoints-plus-refinement measured WORSE than
    plain naive endpoints on real texture content in some regions (least-squares
    is only optimal in continuous space; rounding to 5/6/5 bits can undo the
    improvement, especially on near-flat blocks where naive's own tie-break "+1"
    perturbation already dominates the result). Evaluating candidates against each
    other directly, rather than trusting any one approach unconditionally, is what
    makes this a guaranteed non-regression versus the original naive-only
    encoder - each candidate costs an extra index-assignment pass, so this is
    slower than a single fixed strategy, but still linear in block count."""
    has_transparent = (not force_four_color) and any(p[3] < 128 for p in pixels)

    def _order(c0_565, c1_565):
        if has_transparent:
            if c0_565 > c1_565:
                c0_565, c1_565 = c1_565, c0_565
        else:
            if c0_565 < c1_565:
                c0_565, c1_565 = c1_565, c0_565
            elif c0_565 == c1_565 and c0_565 < 0xFFFF:
                c0_565 += 1
        return c0_565, c1_565

    def _build_palette(c0_565, c1_565):
        r0, g0, b0 = _565_to_rgb(c0_565)
        r1, g1, b1 = _565_to_rgb(c1_565)
        if c0_565 > c1_565:
            return [(r0, g0, b0), (r1, g1, b1),
                    ((2*r0+r1)//3, (2*g0+g1)//3, (2*b0+b1)//3),
                    ((r0+2*r1)//3, (g0+2*g1)//3, (b0+2*b1)//3)]
        return [(r0, g0, b0), (r1, g1, b1),
                ((r0+r1)//2, (g0+g1)//2, (b0+b1)//2), (0, 0, 0)]

    def _assign_indices(palette):
        indices = 0
        for i, p in enumerate(pixels):
            if has_transparent and p[3] < 128:
                idx = 3
            else:
                best_idx, best_dist = 0, None
                for pi in range(4):
                    if has_transparent and pi == 3:
                        continue
                    pc = palette[pi]
                    d = (p[0]-pc[0])**2 + (p[1]-pc[1])**2 + (p[2]-pc[2])**2
                    if best_dist is None or d < best_dist:
                        best_dist, best_idx = d, pi
                idx = best_idx
            indices |= (idx << (2*i))
        return indices

    def _block_error(palette, indices):
        err = 0
        for i, p in enumerate(pixels):
            idx = (indices >> (2*i)) & 0x3
            pc = palette[idx]
            err += (p[0]-pc[0])**2 + (p[1]-pc[1])**2 + (p[2]-pc[2])**2
        return err

    def _candidate(c0_565, c1_565):
        c0_565, c1_565 = _order(c0_565, c1_565)
        palette = _build_palette(c0_565, c1_565)
        indices = _assign_indices(palette)
        return _block_error(palette, indices), c0_565, c1_565, indices

    rs = [p[0] for p in pixels]
    gs = [p[1] for p in pixels]
    bs = [p[2] for p in pixels]
    naive_c0 = _color_to_565(max(rs), max(gs), max(bs))
    naive_c1 = _color_to_565(min(rs), min(gs), min(bs))
    best = _candidate(naive_c0, naive_c1)

    fit_pixels = [(p[0], p[1], p[2]) for p in pixels if not (has_transparent and p[3] < 128)]
    if not fit_pixels:
        fit_pixels = [(p[0], p[1], p[2]) for p in pixels]

    if not all(p == fit_pixels[0] for p in fit_pixels):
        mean, axis = _pca_axis_3(fit_pixels)
        projs = [(p[0] - mean[0]) * axis[0] + (p[1] - mean[1]) * axis[1] + (p[2] - mean[2]) * axis[2]
                 for p in fit_pixels]
        t_max, t_min = max(projs), min(projs)

        def _endpoint(t):
            r = max(0, min(255, round(mean[0] + t * axis[0])))
            g = max(0, min(255, round(mean[1] + t * axis[1])))
            b = max(0, min(255, round(mean[2] + t * axis[2])))
            return r, g, b

        pca_c0 = _color_to_565(*_endpoint(t_max))
        pca_c1 = _color_to_565(*_endpoint(t_min))
        pca_candidate = _candidate(pca_c0, pca_c1)
        if pca_candidate[0] < best[0]:
            best = pca_candidate

    if not has_transparent:
        # Refinement only applies to the 4-unique-color palette ordering (index 3
        # is a real interpolated color there, not the punch-through-alpha "black"
        # slot used in transparent mode, so the linear system's weights above are
        # only valid for this branch). Refine on top of whichever candidate (naive
        # or PCA) is currently winning, using ITS index assignment.
        _, _, _, best_indices = best
        refined = _refine_bc1_endpoints([(p[0], p[1], p[2]) for p in pixels], best_indices)
        if refined is not None:
            refined_candidate = _candidate(*refined)
            if refined_candidate[0] < best[0]:
                best = refined_candidate

    _, c0_565, c1_565, indices = best
    return struct.pack('<HHI', c0_565, c1_565, indices)


def _encode_alpha_block(values):
    """values: 16 ints 0-255. BC3/BC5-style single-channel block -> 8 bytes."""
    a0, a1 = max(values), min(values)
    if a0 == a1:
        if a0 < 255:
            a0 += 1
        else:
            a1 -= 1
    palette = [a0, a1] + [round(((7-i)*a0 + i*a1) / 7) for i in range(1, 7)]
    indices = 0
    for i, v in enumerate(values):
        best_idx, best_dist = 0, None
        for pi, pv in enumerate(palette):
            d = abs(v - pv)
            if best_dist is None or d < best_dist:
                best_dist, best_idx = d, pi
        indices |= (best_idx << (3*i))
    return bytes([a0, a1]) + indices.to_bytes(6, 'little')


def _iter_blocks(width, height):
    for by in range(0, height, 4):
        for bx in range(0, width, 4):
            yield bx, by


def encode_bc1(rgba, width, height):
    """rgba: flat bytes, width*height*4 (RGBA8). Returns raw BC1/DXT1 block data."""
    out = bytearray()
    for bx, by in _iter_blocks(width, height):
        block = []
        for y in range(4):
            for x in range(4):
                px = min(bx + x, width - 1)
                py = min(by + y, height - 1)
                idx = (py * width + px) * 4
                block.append(tuple(rgba[idx:idx + 4]))
        out += _encode_bc1_block(block)
    return bytes(out)


def encode_bc3(rgba, width, height):
    """rgba: flat bytes, width*height*4 (RGBA8). Returns raw BC3/DXT5 block data."""
    out = bytearray()
    for bx, by in _iter_blocks(width, height):
        block, alphas = [], []
        for y in range(4):
            for x in range(4):
                px = min(bx + x, width - 1)
                py = min(by + y, height - 1)
                idx = (py * width + px) * 4
                p = tuple(rgba[idx:idx + 4])
                block.append(p)
                alphas.append(p[3])
        out += _encode_alpha_block(alphas)
        out += _encode_bc1_block(block, force_four_color=True)
    return bytes(out)


def encode_bc5(xy, width, height):
    """xy: flat bytes, width*height*2 (X,Y per pixel, 0-255). Returns raw BC5/ATI2 block data."""
    out = bytearray()
    for bx, by in _iter_blocks(width, height):
        xs, ys = [], []
        for y in range(4):
            for x in range(4):
                px = min(bx + x, width - 1)
                py = min(by + y, height - 1)
                idx = (py * width + px) * 2
                xs.append(xy[idx])
                ys.append(xy[idx + 1])
        out += _encode_alpha_block(xs)
        out += _encode_alpha_block(ys)
    return bytes(out)


def _encode_texture_level(rgba, width, height, encoder):
    """Encode a single RGBA8 level to BC block data. Returns (block_data, encoding, bpp).

    Factored out of register_texture so every mip level uses the identical encoding path.
    """
    if encoder == 'ati2':
        xy = bytes(b for i, b in enumerate(rgba) if i % 4 in (0, 1))  # R,G interleaved -> X,Y
        return encode_bc5(xy, width, height), TEX_ENCODING_ATI2, 8
    elif encoder == 'bc3':
        return encode_bc3(rgba, width, height), TEX_ENCODING_DXT5, 8
    else:
        return encode_bc1(rgba, width, height), TEX_ENCODING_DXT1, 4


def _halve_rgba8(rgba, width, height):
    """2x box-filter downsample of an RGBA8 buffer. width and height must be even.
    Returns (new_width, new_height, new_rgba)."""
    nw, nh = width // 2, height // 2
    out = bytearray(nw * nh * 4)
    for y in range(nh):
        r0 = (2 * y) * width
        r1 = (2 * y + 1) * width
        for x in range(nw):
            p00 = (r0 + 2 * x) * 4
            p10 = (r0 + 2 * x + 1) * 4
            p01 = (r1 + 2 * x) * 4
            p11 = (r1 + 2 * x + 1) * 4
            o = (y * nw + x) * 4
            for c in range(4):
                out[o + c] = (rgba[p00 + c] + rgba[p10 + c] + rgba[p01 + c] + rgba[p11 + c] + 2) >> 2
    return nw, nh, bytes(out)


def build_texture_mip_chain(rgba, width, height, encoder, generate_mips, mip_floor=4):
    """Encode a texture and, if generate_mips, a mip chain down to mip_floor (default 4x4,
    BC's smallest block).

    Returns (concatenated_block_data, encoding, bpp, num_extra_mips).

    num_extra_mips is exactly what the RndBitmap header's mipMaps byte stores: the number of
    levels BELOW the base. RndBitmap.NumMips() in the decompiled engine returns the length of
    the mMip linked list, i.e. it does NOT count the base level (a single-level texture stores
    0). The pixel data is laid out base -> smallest, concatenated - matching how the game and a
    vanilla DDS store it - so the caller's single Xbox 2-byte-pair swap covers the whole blob.

    Why this exists (Dance Central 1): a vanilla DC1 normal map ships a full ATI2/BC5 mip chain
    (angel's is 512->16, 6 levels). DC1's DXN sampling path renders a mip-less ATI2 surface
    BLACK, while RB3 renders the byte-identical single-level texture fine, and DXT1/DXT5
    (diffuse/spec) render mip-less on both. Supplying the chain is the fix for DC1's black
    normal-mapped characters; it is left off for RB3, which does not need it.

    mip_floor default here is 4 for backwards compatibility with existing non-DC1 callers,
    but DC1 itself should always pass DC1_TEX_MIP_FLOOR (16) - see that constant's comment.
    """
    base_block, encoding, bpp = _encode_texture_level(rgba, width, height, encoder)
    if not generate_mips:
        return base_block, encoding, bpp, 0
    chunks = [base_block]
    w, h, cur = width, height, rgba
    # Halve while the NEXT level stays >= mip_floor in both axes.
    #
    # mip_floor exists because the games disagree. DC1 ALSO stops at 16, exactly like TBRB
    # below - NOT at 4x4 as an earlier version of this comment incorrectly claimed. Byte-
    # parsing all 24 real textures in retail emilia01.milo_xbox (DC1) confirms every one
    # bottoms out at a 16px smaller dimension (256x256 ATI2 normal -> mipMaps=4, 512x512 ->
    # 5, 128x128 -> 3, 512x128 -> 3/64x16), never at 4x4. See DC1_TEX_MIP_FLOOR.
    # TBRB stops at 16: every retail TBRB texture bottoms out at a 16px smaller dimension
    # (512x512 -> mipMaps=5, 256x256 -> 4, 512x256 -> 4, 64x64 -> 2), never at 4. Running
    # a TBRB chain down to 4x4 would ship three levels the retail files never carry.
    while w >= mip_floor * 2 and h >= mip_floor * 2:
        w, h, cur = _halve_rgba8(cur, w, h)
        level_block, _, _ = _encode_texture_level(cur, w, h, encoder)
        chunks.append(level_block)
    return b''.join(chunks), encoding, bpp, len(chunks) - 1


def xbox_byte_swap(data):
    """RndBitmap.ConvertToImage()'s Xbox swap, applied in reverse for writing: every
    4-byte group gets its two 16-bit halves byte-swapped: (b0,b1,b2,b3) -> (b1,b0,b3,b2)."""
    out = bytearray(len(data))
    for i in range(0, len(data) - 3, 4):
        out[i] = data[i+1]
        out[i+1] = data[i]
        out[i+2] = data[i+3]
        out[i+3] = data[i+2]
    return bytes(out)


def write_rnd_bitmap(w: MiloWriter, width, height, encoding, bpp, block_data, platform='xbox360',
                      num_mips=0):
    """Assets/Rnd/RndBitmap.cs Write(), fixed at revision 1 (RB3).

    num_mips is the count of levels BELOW the base (RndBitmap.NumMips() = mMip chain length),
    and block_data must already contain base + those mip levels concatenated (base -> smallest).
    0 = a single base level, the historical behaviour.

    The 2-byte-pair byte-swap is XBOX-ONLY. Confirmed from two independent sources:
    MiloLib's RndBitmap only swaps `if (platform == Xbox)` in ConvertToImage, and the
    glTFMilo CLI only swaps the DXT block bytes when `meta.platform == Xbox` (Program.cs),
    storing them verbatim for PS3. So PS3 stores standard little-endian DXT blocks with
    no swap."""
    w.u8(RB3_BITMAP_REVISION)
    w.u8(bpp)
    w.u32(encoding)
    w.u8(num_mips)          # mipMaps = number of levels below the base (0 = base only, the
                             # CLI's GenerateMipmaps=false default). Dance Central 1 requires a
                             # real chain here for ATI2/BC5 normal maps or they render black.
    w.u16(width)
    w.u16(height)
    w.u16((width * bpp) // 8)  # bpl (bytes per line) - was hardcoded to 0, confirmed
                                # wrong against a real file (512-wide DXT1 has bpl=256,
                                # exactly width*bpp/8). Very likely the actual cause of
                                # textures rendering as garbage/another character's data
                                # in-game: if the engine uses this as the row stride/pitch
                                # when uploading pixel data to the GPU, a stride of 0 would
                                # make that copy degenerate, leaving GPU memory holding
                                # whatever was already there from a previously loaded
                                # texture - exactly matching the reported symptom.
    w.u16(0)                # wiiAlphaNum
    w.block(bytes(17))      # padding (revision != 2)
    if platform == 'ps3':
        w.block(block_data)             # PS3: store DXT blocks verbatim (no swap)
    else:
        w.block(xbox_byte_swap(block_data))   # Xbox 360: 2-byte-pair swap


def write_rnd_tex(w: MiloWriter, width, height, encoding, bpp, block_data, standalone=True,
                   external_path="", platform='xbox360', num_mips=0):
    """Assets/Rnd/RndTex.cs Write(), fixed at revision 11 (RB3). Embeds the bitmap
    directly (useExternalPath=False) rather than referencing an external file.

    mipMapK=-8.0 and optimizeForPS3=True are NOT platform-conditional despite the
    field name - confirmed by directly comparing a working CLI-exported .tex against
    ours: the CLI hardcodes both unconditionally for every texture (Program.cs sets
    tex.mipMapK = -8.0f and tex.optimizeForPS3 = true for every single texture type,
    Xbox included). Leaving mipMapK at 0.0 - the field's C# default, which seemed like
    a reasonable value for "no mips" - is suspected to be why textures rendered grey/
    corrupted in-game: the GPU likely uses this as an LOD/mip-bias hint, and with only
    one real mip level stored, an LOD bias expecting more mips could sample garbage."""
    w.u32(RB3_TEX_REVISION)         # (altRevision=0 << 16) | 11

    write_object_fields(w)          # base.Write(standalone=False) -> objFields (revision(11) > 8)

    w.u32(width)
    w.u32(height)
    w.u32(bpp)
    w.symbol(external_path)         # externalPath - purely informational since useExternalPath=False,
                                     # but the CLI sets it anyway (e.g. "MaterialName.png")
    w.f32(-8.0)                     # mipMapK (revision >= 8) - see docstring
    w.u32(TEX_TYPE_REGULAR)         # type (revision > 6)
    w.boolean(True)                 # optimizeForPS3 (revision >= 11) - see docstring
    w.boolean(False)                # useExternalPath (revision != 7) - embed the bitmap

    write_rnd_bitmap(w, width, height, encoding, bpp, block_data, platform=platform,
                      num_mips=num_mips)

    if standalone:
        w.block(END_MARKER)


# Byte-enum values, confirmed directly from MiloLib/Assets/Rnd/RndMat.cs - same
# identifiers used in the Blender EnumProperty items, so this is a direct lookup.
MAT_BLEND_VALUES = {
    'kBlendDest': 0, 'kBlendSrc': 1, 'kBlendAdd': 2, 'kBlendSrcAlpha': 3,
    'kBlendSrcAlphaAdd': 4, 'kBlendSubtract': 5, 'kBlendMultiply': 6, 'kPreMultAlpha': 7,
}
MAT_ZMODE_VALUES = {
    'kZModeDisable': 0, 'kZModeNormal': 1, 'kZModeTransparent': 2, 'kZModeForce': 3, 'kZModeDecal': 4,
}
MAT_STENCIL_IGNORE = 0        # kStencilIgnore - not exposed yet, CLI always uses this
MAT_TEXGEN_NONE = 0           # kTexGenNone - not exposed yet
MAT_TEXWRAP_REPEAT = 1        # kTexWrapRepeat - CLI's own fallback default
MAT_SHADER_VARIATION_NONE = 0 # kShaderVariationNone
# RndMat.ShaderVariation enum (byte), confirmed directly from MiloLib/Assets/Rnd/RndMat.cs
MAT_SHADER_VARIATION_VALUES = {
    'kShaderVariationNone': 0, 'kShaderVariationSkin': 1, 'kShaderVariationHair': 2,
}

IDENTITY_MATRIX_UNUSED = IDENTITY_MATRIX  # texXfm default


def write_rnd_mat(w: MiloWriter, settings, standalone=True):
    """Assets/Rnd/RndMat.cs Write(), fixed at revision 68 (RB3). `settings` is a dict
    with keys matching GltfMiloMaterialSettings: blend, base_color, pre_lit,
    use_environment, z_mode, alpha_cut, alpha_threshold, specular_rgb, specular_power,
    emissive_multiplier, environment_map, cull, de_normal, anisotropy,
    normal_detail_strength, rim_rgb, rim_power, shader_variation, refract_enabled,
    refract_strength (texture symbols - diffuse/normal/specular/emissive/
    refract_normal_map - default to "" if not set).

    Every field not yet exposed in the Blender UI uses glTFMilo's own CLI defaults,
    read directly from Source/glTFMilo/Program.cs's unconditional mat.* assignments -
    see the comments below for exactly which line each default came from."""
    w.u32(RB3_MAT_REVISION)  # (altRevision=0 << 16) | 68

    write_object_fields(w)   # base.Write(standalone=False) -> objFields (revision=2, same as everywhere else)

    w.i32(MAT_BLEND_VALUES[settings['blend']])
    write_color4(w, *settings['base_color'])
    w.boolean(settings['pre_lit'])
    w.boolean(settings['use_environment'])
    w.i32(MAT_ZMODE_VALUES[settings['z_mode']])
    w.boolean(settings['alpha_cut'])
    # revision(68) > 0x25(37): True
    w.i32(settings['alpha_threshold'])
    w.boolean(settings['blend'] == 'kBlendSrcAlpha')   # alphaWrite - not exposed yet; CLI only
                                                        # sets this true in its "BLEND" alpha-mode
                                                        # branch, which corresponds to this blend mode
    w.i32(MAT_TEXGEN_NONE)             # texGen (not exposed yet)
    w.i32(MAT_TEXWRAP_REPEAT)          # texWrap (not exposed yet - CLI's own fallback default)
    write_matrix(w, IDENTITY_MATRIX)   # texXfm (not exposed yet)
    w.symbol(settings.get('diffuse_tex_name', ''))
    w.symbol("")                       # nextPass
    w.boolean(settings.get('intensify', False))   # intensify - boosts (doubles) the base
                                                    # color for a stronger glow/bloom. Confirmed
                                                    # against neon_resource: every untextured
                                                    # bulb material (orange/red/blue/etc.) sets
                                                    # this True, while textured surfaces leave it
                                                    # False. This is the extra "pop" that makes a
                                                    # preLit bright-color surface read as a glowing
                                                    # neon bulb rather than just a brightly lit one.

    w.boolean(settings.get('cull', False))  # cull - now its own dedicated material
                                             # setting (GltfMiloMaterialSettings.cull)
                                             # for direct testing, rather than being
                                             # sourced from Blender's built-in
                                             # "Backface Culling" toggle
    w.f32(settings['emissive_multiplier'])
    write_color3(w, *settings['specular_rgb'])
    w.f32(settings['specular_power'])
    w.symbol(settings.get('normal_tex_name', ''))
    w.symbol(settings.get('emissive_tex_name', ''))
    w.symbol(settings.get('specular_tex_name', ''))
    # revision(68) < 51: False -> unkSymbol2 skipped
    w.symbol("instruments.cube" if settings['environment_map'] else "")  # environMap

    # revision > 25: True; revision == 68: True
    w.u16(0)                           # unkShort

    w.boolean(True)                    # perPixelLit - Program.cs line 664: always true

    # revision >= 27 and < 50: False (68 is outside) -> unkBool1 skipped

    w.i32(MAT_STENCIL_IGNORE)          # stencilMode - Program.cs line 662: always kStencilIgnore

    # revision < 33: False -> else:
    w.symbol("")                       # fur

    # revision >= 34 and < 49: False (68 outside) -> unkBool2/unkColor3/unkFloat3/unkSym3 skipped
    # revision <= 28: False -> no early return

    w.f32(float(settings.get('de_normal', 0)))   # deNormal - user-editable, restricted to -1/0/1
    w.f32(settings.get('anisotropy', 0.0))        # anisotropy - user-editable
    w.f32(1.0)                         # normalDetailTiling - Program.cs line 688
    w.f32(settings.get('normal_detail_strength', 0.0))  # normalDetailStrength - user-editable
    w.symbol("")                       # normalDetailMap

    w.boolean(True)                    # pointLights - Program.cs line 669: always true
    # revision < 0x3F(63): False (68 not < 63) -> projLights skipped
    w.boolean(False)                   # fog - Program.cs line 671: always false
    w.boolean(False)                   # fadeout - not set by CLI, C# default
    w.boolean(False)                   # colorAdjust - not set by CLI, C# default

    # revision > 47: True
    write_color3(w, *settings.get('rim_rgb', (0.0, 0.0, 0.0)))  # rimRGB - user-editable
    w.f32(settings.get('rim_power', 4.0))  # rimPower - user-editable
    w.symbol("")                       # rimMap
    w.boolean(False)                   # rimAlwaysShow

    # revision > 48: True
    w.boolean(False)                   # screenAligned

    # revision > 0x32(50): True
    w.i32(MAT_SHADER_VARIATION_VALUES[settings.get('shader_variation', 'kShaderVariationNone')])  # user-editable
    write_color3(w, 0.0, 0.0, 0.0)     # specular2RGB
    w.f32(0.0)                         # specular2Power - Program.cs line 694

    # revision >= 52 and <= 67: False (68 is outside) -> whole unkInt3/unkColor2/colors block skipped
    # revision >= 54 and <= 61: False -> unkSym2 skipped
    # revision >= 55 and <= 62: False -> perfSettings.ps3ForceTrilinear (standalone bool) skipped
    # revision == 0x38(56): False -> unkInt1/unkInt2 skipped

    # revision > 0x3E(62): True
    w.boolean(False)                   # perfSettings.recvProjLights
    w.boolean(False)                   # perfSettings.ps3ForceTrilinear
    # revision > 0x41(65): True
    w.boolean(False)                   # perfSettings.recvPointCubeTex

    # revision > 0x3F(63): True
    w.boolean(settings.get('refract_enabled', False))   # refractEnabled - user-editable
    w.f32(settings.get('refract_strength', 0.0))         # refractStrength - user-editable
    w.symbol(settings.get('refract_normal_map_tex_name', ''))  # refractNormalMap - user-editable

    if standalone:
        w.block(END_MARKER)


# DC3 BaseMaterial::Save constant defaults for fields the Blender UI doesn't expose yet. These
# are the C# constructor defaults from dc3-decomp's BaseMaterial::BaseMaterial(), used verbatim
# so a from-scratch DC3 material matches what the game would author for an untouched material.
DC3_MAT_ALLOW_DISTORTION_EFFECTS = True    # mAllowDistortionEffects(true)
DC3_MAT_SHOCKWAVE_MULT = 1.0               # mShockwaveMult(1)
DC3_MAT_WORLD_PROJECTION_TILING = 0.125    # mWorldProjectionTiling(0.125)
DC3_MAT_WORLD_PROJECTION_START_BLEND = 0.8 # mWorldProjectionStartBlend(0.8)
DC3_MAT_WORLD_PROJECTION_END_BLEND = 0.9   # mWorldProjectionEndBlend(0.9)


def _dc3_shader_variation_for_meta(meta_name, fallback='kShaderVariationNone'):
    """The material's shaderVariation must match the shading path of its MetaMaterial. Confirmed
    by byte-parsing retail emilia01: char_basic_skin*.mmat surfaces set kShaderVariationSkin(1),
    char_basic_hair*.mmat set kShaderVariationHair(2), and everything else (rim/lod/color/nospec)
    stays kShaderVariationNone(0). `fallback` is used when the name doesn't match a known family
    (e.g. a custom template) so an explicitly-authored shader variation isn't clobbered."""
    m = (meta_name or "").lower()
    if "char_basic_skin" in m:
        return 'kShaderVariationSkin'
    if "char_basic_hair" in m:
        return 'kShaderVariationHair'
    if m in ("char_basic_rim.mmat", "char_basic_lod.mmat", "char_basic_color.mmat"):
        return 'kShaderVariationNone'
    return fallback


def write_dc3_rnd_mat(w: MiloWriter, settings, standalone=True, force_dc3_defaults=False):
    """Assets/Rnd material writer for Dance Central 3, fixed at RndMat revision 0x46 (70).

    DC3 restructured the material into a class hierarchy (confirmed against rjkiv/dc3-decomp,
    src/system/rndobj/{Mat,BaseMaterial}.cpp) instead of RB3's single flat RndMat. The wire
    format is therefore:

        RndMat::Save         SAVE_REVS(0x46,0)              -> u32 0x00000046
                             SAVE_SUPERCLASS(BaseMaterial)
        BaseMaterial::Save     SAVE_REVS(8,0)              -> u32 0x00000008
                               SAVE_SUPERCLASS(Hmx::Object)
        Hmx::Object::Save        bs << 2 (SaveType)        -> u32 0x00000002
                                 bs << Type() (SaveType)   -> symbol ("")
                                 bs << (DataArray*)null    -> bool 0x00 (SaveRest)
                                 bs << 0  (empty note)     -> u32 0x00000000 (SaveRest)
                             ... BaseMaterial fields ...
                         bs << mMetaMaterial               -> symbol (the .mmat template)

    Two differences from the RB3 rev-68 writer matter and are exactly why an RB3-format
    material renders black on a normal-mapped DC3 mesh:

      1. Field ORDER differs. In DC3, specularRGB/normalMap/emissiveMap/specularMap are
         written as a contiguous run right after emissiveMultiplier, and there is NO separate
         specularPower float - the specular power is packed into specularRGB's alpha channel
         (Hmx::Color is 4 floats r,g,b,a; a == power). RB3 wrote 3-float specularRGB + a
         standalone specularPower float, which shifts normalMap 4 bytes late and derails every
         subsequent field on the DC3 loader.

      2. Every DC3 material carries a trailing MetaMaterial (.mmat) symbol. The runtime resolves
         it by name against a global template dir and calls UpdatePropertiesFromMetaMat(); a
         material without it (or read at the wrong offset) never gets its properties resolved.

    `settings` is the same dict write_rnd_mat consumes, plus optional 'meta_material' (the .mmat
    symbol; defaults to DC3_DEFAULT_META_MATERIAL). All fields not exposed in the Blender UI use
    dc3-decomp's own BaseMaterial constructor defaults (see the DC3_MAT_* constants above).

    `force_dc3_defaults`: like TBRB's force_tbrb_defaults, the Blender material UI was authored
    for RB3 and a couple of its render-affecting defaults are inverted relative to what DC3
    materials actually use. EVERY retail emilia01 material (byte-parsed from the real milo)
    writes prelit=0 and useEnviron=1; the RB3 UI defaults to the opposite (prelit=1,
    useEnviron=0), which disables environment lighting on a DC3 character. When True, prelit and
    useEnviron are snapped to the retail DC3 convention, AND shaderVariation is derived from the
    MetaMaterial name so the two always agree (see _dc3_shader_variation_for_meta): retail pairs
    char_basic_skin*.mmat with kShaderVariationSkin(1) and char_basic_hair*.mmat with
    kShaderVariationHair(2); a mismatch between the template's shading path and the material's
    shaderVariation is its own rendering bug. Texture names, base_color, blend, z_mode, specular,
    rim, and the MetaMaterial name still come from the authored material either way. Note the
    referenced MetaMaterial re-applies its own default/forced property values on load
    (UpdatePropertiesFromMetaMat), so the .mmat name matters most - but matching retail keeps the
    material self-consistent and byte diffs against retail clean."""
    s = dict(settings)   # copy so overrides don't mutate the caller's dict
    if force_dc3_defaults:
        s['pre_lit'] = False        # retail: prelit=0 on every emilia material
        s['use_environment'] = True # retail: useEnviron=1 on every emilia material
        # Keep shaderVariation consistent with the chosen MetaMaterial's shading path.
        s['shader_variation'] = _dc3_shader_variation_for_meta(
            s.get('meta_material', DC3_DEFAULT_META_MATERIAL),
            s.get('shader_variation', 'kShaderVariationNone'))
    settings = s

    # --- RndMat::Save / BaseMaterial::Save / Hmx::Object::Save headers ---
    w.u32(DC3_MAT_REVISION)          # RndMat  packRevs(0, 0x46)
    w.u32(DC3_BASEMAT_REVISION)      # BaseMaterial  packRevs(0, 8)
    w.u32(DC3_OBJ_REVISION)          # Hmx::Object  bs << 2
    w.symbol("")                     # Object type
    w.boolean(False)                 # null TypeProps (DataArray* == nullptr -> bool false)
    w.u32(0)                         # empty note (SaveRest: bs << 0, a 4-byte int)

    # --- BaseMaterial::Save body, in exact decomp order ---
    # bs << mBlend << mColor << mUseEnviron << mPrelit;
    w.i32(MAT_BLEND_VALUES[settings['blend']])
    write_color4(w, *settings['base_color'])
    w.boolean(settings['use_environment'])
    w.boolean(settings['pre_lit'])

    # bs << mZMode << mAlphaCut << mAlphaThreshold << mAlphaWrite;
    w.i32(MAT_ZMODE_VALUES[settings['z_mode']])
    w.boolean(settings['alpha_cut'])
    w.i32(settings['alpha_threshold'])
    w.boolean(settings['blend'] == 'kBlendSrcAlpha')   # alphaWrite - mirror the RB3 writer's
                                                        # rule (true only in the SrcAlpha blend
                                                        # branch); not user-exposed yet

    # bs << mTexGen << mTexWrap << mTexXfm << mDiffuseTex << mNextPass;
    w.i32(MAT_TEXGEN_NONE)
    w.i32(MAT_TEXWRAP_REPEAT)
    write_matrix(w, DC3_TEXXFM_IDENTITY)   # -0.0 off-diagonals match retail's Transform::Reset()
    w.symbol(settings.get('diffuse_tex_name', ''))
    w.symbol("")                     # nextPass (in-milo next-pass material chain; unused here)

    # bs << mIntensify << mCull << mEmissiveMultiplier;
    # (rev 8 >= 3, so cull is written as a full int, not the old bool)
    w.boolean(settings.get('intensify', False))
    w.i32(1 if settings.get('cull', False) else 0)   # kCullRegular(1) / kCullNone(0)
    w.f32(settings['emissive_multiplier'])

    # bs << mSpecularRGB << mNormalMap;
    # mSpecularRGB is a 4-float Hmx::Color: rgb + ALPHA==specular power. This is where DC3
    # diverges hardest from RB3 (no standalone specularPower float).
    sr = settings['specular_rgb']
    write_color4(w, sr[0], sr[1], sr[2], settings.get('specular_power', 0.0))
    w.symbol(settings.get('normal_tex_name', ''))

    # bs << mEmissiveMap << mSpecularMap;
    w.symbol(settings.get('emissive_tex_name', ''))
    w.symbol(settings.get('specular_tex_name', ''))

    # bs << mEnvironMap << mEnvironMapFalloff << mEnvironMapSpecMask;
    w.symbol("instruments.cube" if settings.get('environment_map') else "")
    w.boolean(False)                 # environMapFalloff (not exposed)
    w.boolean(False)                 # environMapSpecMask (not exposed)

    # bs << mPerPixelLit << mStencilMode;
    # perPixelLit must be True for any material that uses a normal or specular map (the
    # per-pixel lighting path is what samples them). Retail leaves it False on untextured
    # materials that defer entirely to their MetaMaterial (skin/color/LOD), but for a custom
    # textured surface it needs to be on. Default True, overridable via the 'per_pixel_lit'
    # setting, and force-True whenever a normal or specular map is present so a user can't
    # accidentally reintroduce the black-normal-map bug.
    needs_ppl = bool(settings.get('normal_tex_name') or settings.get('specular_tex_name'))
    w.boolean(settings.get('per_pixel_lit', True) or needs_ppl)
    w.i32(MAT_STENCIL_IGNORE)        # stencilMode (kStencilIgnore)

    # bs << mFur << mDeNormal << mAnisotropy;
    w.symbol("")                     # fur
    w.f32(float(settings.get('de_normal', 0)))
    w.f32(settings.get('anisotropy', 0.0))

    # bs << mNormDetailTiling << mNormDetailStrength << mNormDetailMap;
    w.f32(1.0)                       # normDetailTiling (ctor default 1)
    w.f32(settings.get('normal_detail_strength', 0.0))
    w.symbol("")                     # normDetailMap

    # bs << mPointLights << mFog << mFadeout << mColorAdjust;
    w.boolean(settings.get('point_lights', True))   # pointLights - retail always True on emilia
    w.boolean(False)                 # fog
    w.boolean(False)                 # fadeout
    w.boolean(False)                 # colorAdjust

    # bs << mRimRGB << mRimMap << mRimLightUnder;
    # mRimRGB is a 4-float Hmx::Color too (rgb + alpha==rim power; ctor default alpha 10).
    rr = settings.get('rim_rgb', (0.0, 0.0, 0.0))
    write_color4(w, rr[0], rr[1], rr[2], settings.get('rim_power', 4.0))
    w.symbol("")                     # rimMap
    w.boolean(False)                 # rimLightUnder

    # bs << mScreenAligned << mShaderVariation << mSpecular2RGB;
    w.boolean(False)                 # screenAligned
    w.i32(MAT_SHADER_VARIATION_VALUES[settings.get('shader_variation', 'kShaderVariationNone')])
    write_color4(w, 0.0, 0.0, 0.0, 10.0)   # specular2RGB (ctor default 0,0,0, alpha 10)

    # mPerfSettings.Save(bs): recvProjLights, ps3ForceTrilinear, recvPointCubeTex
    w.boolean(False)
    w.boolean(False)
    w.boolean(False)

    # bs << mRefractEnabled << mRefractStrength << mRefractNormalMap;
    w.boolean(settings.get('refract_enabled', False))
    w.f32(settings.get('refract_strength', 0.0))
    w.symbol(settings.get('refract_normal_map_tex_name', ''))

    # bs << mBloomMultiplier << mNeverFitToSpline;
    w.f32(1.0)                       # bloomMultiplier (ctor default 1)
    w.boolean(False)                 # neverFitToSpline

    # bs << mAllowDistortionEffects << mShockwaveMult;
    w.boolean(DC3_MAT_ALLOW_DISTORTION_EFFECTS)
    w.f32(DC3_MAT_SHOCKWAVE_MULT)

    # bs << mWorldProjectionTiling << mWorldProjectionStartBlend << mWorldProjectionEndBlend;
    w.f32(DC3_MAT_WORLD_PROJECTION_TILING)
    w.f32(DC3_MAT_WORLD_PROJECTION_START_BLEND)
    w.f32(DC3_MAT_WORLD_PROJECTION_END_BLEND)

    # bs << mDiffuseTex2;
    w.symbol("")                     # diffuseTex2 (second diffuse layer; unused here)

    # bs << mForceAlphaWrite;
    w.boolean(False)                 # forceAlphaWrite

    # --- back in RndMat::Save: bs << mMetaMaterial (the trailing .mmat template symbol) ---
    w.symbol(settings.get('meta_material', DC3_DEFAULT_META_MATERIAL))

    if standalone:
        w.block(END_MARKER)


def write_rnd_mesh(w: MiloWriter, entry_name, local_xfm, world_xfm, parent_obj,
                    vertices, faces, mat_name="", bone_transforms=None,
                    force_white_vertex_color=False, platform='xbox360',
                    write_tangents=True, dc3=False, dc1=False):
    """Assets/Rnd/RndMesh.cs Write(), fixed at revision 38 (RB3 AND DC3 - confirmed by
    byte-parsing retail angel01.milo_xbox: all 23 meshes are revision 38, same as RB3).
    Vertex format:
    compressed Xbox 360 (isNextGen=True, compressionType=1, 36 bytes/vertex),
    confirmed against a real CLI-exported character file with materials and
    skinning - switched over from the uncompressed format (which only matched a
    real reference for an unlit static prop, and is suspected of not being what
    the game's lit-rendering path actually expects).
    `vertices` is a list of dicts with keys: x,y,z, nx,ny,nz, u,v, r,g,b,a
    (vertex color - r/g/b/a as raw floats, matching the original tool's [likely
    unintentional but faithfully ported] direct float->int cast rather than a
    0-255 scale), and (for skinned meshes) weights=[w0..w3], bones=[b0..b3]
    (already resolved to local bone-list indices, already padded/ordered - see
    collect_mesh_entries).
    `faces` is a list of (i1, i2, i3) ushort tuples.
    `bone_transforms` is a list of (bone_name, transform12) - the mesh's local
    per-mesh bone list (RndMesh::MaxBones() == 40), or None/[] for a static mesh.

    `force_white_vertex_color`: confirmed against the glTFMilo reference tool
    (commit "Use full white for AO calculation if mesh has normal map (for
    now)") - vertex color here is a raw 0-255 value, not 0.0-1.0, and a mesh
    with no real vertex-color data defaults to (0,0,0,0) i.e. black. The
    engine's lit shader path multiplies its output by vertex color, so any
    mesh lacking real vertex-paint data renders black once it's routed onto
    that path. The reference tool papers over this by setting
    hasAOCalculation=True and forcing every vertex color to (255,255,255,255)
    - but only when the material has a normal map. It never extended this to
    emissive-only materials, which is why models using only an emissive
    texture (no normal map) still render black - same root cause, just an
    unfixed gap upstream. This port sets force_white_vertex_color for either
    case to close that gap.

    `dc3`: writes the mesh's embedded RndDrawable at DC3_DRAW_REVISION (4) instead of
    RB3/DC1's 3 - see that constant's comment. Confirmed required: byte-parsing every
    embedded Drawable in 9 sampled retail DC3 meshes reads revision 4, and writing 3
    instead leaves out one field the real reader expects at this exact spot, which
    shifts every subsequent symbol in this same mesh (mat, geomOwner) one field early.
    No effect for RB3/DC1."""
    if bone_transforms is None:
        bone_transforms = []
    draw_rev = DC3_DRAW_REVISION if dc3 else RB3_DRAW_REVISION

    w.u32(RB3_MESH_REVISION)

    write_object_fields(w)              # base.Write(standalone=False) -> Object.Write -> objFields.Write()

    write_rnd_trans(w, local_xfm, world_xfm, parent_obj)
    write_rnd_drawable(w, sphere=(0.0, 0.0, 0.0, 0.0), revision=draw_rev)

    w.symbol(mat_name)                  # mat
    # revision == 27 -> mat2: skipped (revision is 38)
    w.symbol(entry_name)                # geomOwner - real files self-reference their own name
    # revision < 13/15/14/3 blocks: all skipped at revision 38

    w.u32(0)                            # mutable (kMutableNone)
    w.u32(1)                            # volume (kVolumeTriangles)
    w.boolean(False)                    # bspNode.hasValue

    # --- vertices (compressed, isNextGen=True) ---
    # Xbox 360: compressionType=1, 36 bytes/vertex. PS3: compressionType=2, 40 bytes.
    # Both confirmed against MiloLib's RndMesh.WriteVertices for meshVersion 38. The two
    # layouts are genuinely different (not just endianness) - PS3 uses 11-11-10 packed
    # vec3s with no W, ARGB color order, and u16 bone indices, which is why its vertex is
    # 40 bytes vs Xbox's 36. Both consoles are big-endian PowerPC so the container/body
    # endianness itself is identical - only the per-vertex packing differs.
    is_ps3 = (platform == 'ps3')
    w.u32(len(vertices))
    w.boolean(True)                     # isNextGen
    if is_ps3:
        w.u32(40)                       # vertexSize (PS3)
        w.u32(2)                        # compressionType (PS3)
    else:
        w.u32(36)                       # vertexSize (Xbox 360)
        w.u32(1)                        # compressionType (Xbox 360)
    for v in vertices:
        w.f32(v["x"])
        w.f32(v["y"])
        w.f32(v["z"])
        if force_white_vertex_color:
            # Confirmed against glTFMilo's own fix: vertex color is raw 0-255,
            # not 0-1, and defaults to black (0,0,0,0) with no real vertex-paint
            # data. Force full white here to avoid the mesh rendering black.
            r = g = b = a = 255.0
        else:
            r = v.get("r", 1.0)
            g = v.get("g", 1.0)
            b = v.get("b", 1.0)
            a = v.get("a", 1.0)
        ri, gi, bi, ai = int(r) & 0xFF, int(g) & 0xFF, int(b) & 0xFF, int(a) & 0xFF
        weights = v.get("weights", (0.0, 0.0, 0.0, 0.0))
        bones = v.get("bones", (0, 0, 0, 0))

        if is_ps3:
            # PS3 layout (compressionType 2), from MiloLib WriteVertices:
            #   u,v (half) | norms(11-11-10) | tangents(11-11-10) | weights(11-11-10)
            #   | color(ARGB u32) | 4x u16 bones
            w.f16(v["u"])
            w.f16(v["v"])
            w.u32(pack_ps3_signed_11_11_10(v["nx"], v["ny"], v["nz"]))
            # tangent - PS3 DOES populate this (unlike the Xbox path just below, whose zero
            # tangent matches the glTFMilo CLI). Confirmed by byte-diffing a real glTFMilo
            # PS3 export: the reference stores a real unit-length 11-11-10 tangent. A zero
            # here gives a degenerate tangent basis, so normal-mapped lighting can't be
            # applied - surfaces read flat or near-black on lit/normal-mapped materials in
            # dim scenes. tx/ty/tz come from Blender's calc_tangents(), in the same axis
            # frame as the normals above so the basis stays consistent.
            # NOTE: 11 + 11 + 10 = 32 bits exactly, so unlike Xbox's 10-10-10-2 there is NO
            # W component here (MiloLib's PS3SignedCompressedVec3 hard-sets w = 0 on read).
            # Per-vertex handedness therefore cannot be stored on PS3; the shader derives the
            # bitangent as cross(N, T) with fixed handedness. On a mesh with MIRRORED UV
            # islands, normal-map detail on the mirrored half may appear inverted.
            if write_tangents:
                w.u32(pack_ps3_signed_11_11_10(
                    v.get("tx", 0.0), v.get("ty", 0.0), v.get("tz", 0.0)))
            else:
                w.u32(pack_ps3_signed_11_11_10(0.0, 0.0, 0.0))
            w.u32(pack_ps3_unsigned_11_11_10(weights[0], weights[1], weights[2]))
            # NOTE: PS3 color is ARGB order (a<<24 | r<<16 | g<<8 | b), NOT the Xbox
            # RGBA order - confirmed directly from MiloLib's compressionType==2 branch.
            w.u32((ai << 24) | (ri << 16) | (gi << 8) | bi)
            w.u16(bones[0])
            w.u16(bones[1])
            w.u16(bones[2])
            w.u16(bones[3])
        else:
            # Xbox 360 layout (compressionType 1):
            #   color(RGBA u32) | u,v (half) | norms(10-10-10-2) | tangents(10-10-10-2)
            #   | weights(10-10-10-2) | 4x u8 bones
            w.u32((ri << 24) | (gi << 16) | (bi << 8) | ai)
            w.f16(v["u"])
            w.f16(v["v"])
            w.u32(pack_signed_10_10_10_2(v["nx"], v["ny"], v["nz"], 0.0))
            # Tangent. Previously hardcoded to zero with a comment claiming the CLI "never
            # populates this" - that was WRONG: glTFMilo's Program.cs copies the glTF TANGENT
            # accessor into tangent0..3, exactly as the PS3 branch above already does.
            #
            # Confirmed against dc3-decomp (an Xbox 360 Harmonix decomp; the Wii RB3 decomp
            # stripped normal maps entirely so it can't answer this): CompressedVertex_Xbox is
            # byte-identical to this 36-byte layout with `tangent` at 0x18, and the D3D vertex
            # declaration binds it as D3DDECLUSAGE_TANGENT / D3DDECLTYPE_DEC4N - signed
            # 10-10-10-2, exactly pack_signed_10_10_10_2. It is a real GPU shader input.
            # A zero tangent makes the TBN basis degenerate, so a tangent-space normal map
            # collapses back to the geometric normal and the surface renders FLAT. Specular
            # kept working because specular only needs the vertex normal, never the tangent -
            # exactly the symptom split that was observed in-game.
            #
            # The engine will NOT generate these at this revision: dc3-decomp's RndMesh load
            # only calls MakeTangentsLate() when `d.rev < 0x1E` (< 30); RB3 meshes are rev 38.
            # Confirmed fixed in-game on Xbox 360 (fabric wrinkles reappear).
            #
            # Unlike PS3's 11-11-10, the Xbox 10-10-10-2 format has a 2-bit W, which carries
            # the bitangent sign / handedness (Blender's loop.bitangent_sign, +/-1).
            if write_tangents:
                w.u32(pack_signed_10_10_10_2(
                    v.get("tx", 0.0), v.get("ty", 0.0), v.get("tz", 0.0), v.get("tw", 1.0)))
            else:
                w.u32(pack_signed_10_10_10_2(0.0, 0.0, 0.0, 0.0))
            w.u32(pack_unsigned_10_10_10_2(weights[0], weights[1], weights[2], weights[3]))
            w.u8(bones[0])
            w.u8(bones[1])
            w.u8(bones[2])
            w.u8(bones[3])

    # --- faces ---
    w.u32(len(faces))
    for (i1, i2, i3) in faces:
        w.u16(i1)
        w.u16(i2)
        w.u16(i3)

    group_sizes = compute_group_sizes(len(faces))
    w.u32(len(group_sizes))
    for g in group_sizes:
        w.u8(g)

    w.u32(len(bone_transforms))
    for (bone_name, transform12) in bone_transforms:
        write_bone_transform(w, bone_name, transform12)

    w.boolean(False if dc1 else True)     # keepMeshData - confirmed FALSE in every real DC1
                                           # mesh sampled (7 vanilla meshes, with AND without
                                           # normal maps - wraps/jersey/cleats/trackshorts/head/
                                           # eyelashes/eye-L all ship keepMeshData=0). Left as
                                           # True for RB3/DC3, unchanged from prior verified
                                           # behaviour there.
    w.boolean(False if dc1 else force_white_vertex_color)
                                           # hasAOCalculation - real DC1 meshes NEVER set this,
                                           # even ones with real diffuse+normal+spec (wraps.mesh,
                                           # jersey.mesh) - they rely purely on the pre-baked
                                           # white vertex color already written above (which this
                                           # writer already does unconditionally when
                                           # force_white_vertex_color is True, independent of
                                           # this flag). Forcing the runtime AO-recompute path on
                                           # is the leading suspect for DC1's black-with-normal-
                                           # map symptom: it's the one place this writer diverges
                                           # from vanilla specifically on meshes that have a
                                           # normal map, which is exactly the reported failure
                                           # pattern. RB3 keeps its previously-verified behaviour
                                           # (force_white_vertex_color drives this flag) since
                                           # that combination was confirmed working there before
                                           # and hasn't been independently re-checked against a
                                           # real RB3 mesh byte-for-byte in this pass.
    # altRevision(0) > 1 / > 3: both skipped -> noQuant / unkBool3 not written
    # groupSizes[0](12) > 0 but parent.revision(28) is NOT < 25, so the trailing
    # groupSections block is still skipped even though groupSizes is non-empty here

    w.block(END_MARKER)


# =====================================================================================
# SECTION 5b -- Dance Central 1 support + donor-milo CharClipSet extraction
#
# DC1 uses byte-identical format revisions to RB3 (verified by parsing a real DC1
# character, mo01.milo_xbox, against RB3 files):
#     DirectoryMeta 28, Character 17, RndDir 10, ObjectDir 27,
#     Mesh 38, Mat 68, Tex 11, CharCollide 7,
#     vertex format = Xbox 360 compressionType 1 / 36 bytes
# so every asset writer in this file already emits DC1-valid data unchanged. DC1 was an
# Xbox 360 exclusive, so there is no PS3 path.
#
# The one thing we cannot author from Blender is the CharClipSet - the character's
# animation clip set. In a real DC1 character it is entry 0 of the milo and contains the
# dance animations (CharClip / CharClipGroup assets). This module lifts that whole asset
# out of a donor milo verbatim so a custom character can carry working animations.
#
# LAYOUT of a CharClipSet entry (confirmed by byte-parsing mo01 (DC1) and a DC3 milo):
#     [CharClipSet object fields ... ends with 0xADDEADDE]
#     [nested DirectoryMeta: rev 28 (DC1/RB3) or 32 (DC3), type "CharClipSet",
#      name e.g. "clips" / "clips1", entry table (CharClip / CharClipGroup ...), then the
#      nested dir's own object body, then each nested asset body - each ending 0xADDEADDE]
#
# DC3 note: DC3 milos are a newer format version. For CharClipSet extraction the only
# relevant differences are the nested DirectoryMeta revision (32 vs 28) and one extra byte
# inserted before entryCount at rev 32. We therefore locate the CharClipSet by TYPE, not by
# a fixed revision, and find entryCount by validation - so DC1, RB3 and DC3 donors all work.
#
# EXTRACTION strategy (validated on mo01/DC1 and a real DC3 milo):
#   1. Find the nested DirectoryMeta by its "CharClipSet" type symbol and confirm a plausible
#      revision precedes it (_find_charclipset_dir); reject the top-level entry-table
#      reference to "CharClipSet" because it does not parse as a valid entry table.
#   2. The 0xADDEADDE immediately before it terminates the CharClipSet object fields.
#   3. The 0xADDEADDE before THAT terminates the parent Character directory body - so the
#      CharClipSet entry body begins right after it. (This works because CharClipSet is
#      entry 0, i.e. the first asset written after the directory body.)
#   4. Parse the nested entry table for its entry count N (probing 0..4 padding bytes so the
#      rev-32 extra byte is absorbed), then walk (N + 1) end markers forward from the end of
#      that table: N asset bodies plus the nested directory's own object body. Where that
#      lands is the end of the whole CharClipSet entry.
#
# Step 4 self-validates: on DC1 the walk consumed 45 markers for 44 nested entries and landed
# on the next top-level entry; on DC3, 50 markers for 49 entries, likewise landing cleanly.
# extract_charclipset_from_donor() re-checks that marker arithmetic and refuses to return
# a range it cannot account for, rather than silently handing back garbage bytes.
# =====================================================================================

# The nested CharClipSet DirectoryMeta is located by its TYPE symbol (length-prefixed
# "CharClipSet"), not by a hardcoded directory revision, so donors from any game version
# are accepted. Verified against real files: Dance Central 1 / Rock Band 3 use DirectoryMeta
# revision 28, Dance Central 3 uses revision 32. The two headers are identical except that
# rev 32 inserts one extra byte between the string-table size and entryCount - handled
# empirically by _find_charclipset_dir (it locates the real entryCount by requiring a clean,
# non-empty clip table) rather than by a per-revision offset table, so future revisions with
# similar padding also work.
CHARCLIPSET_TYPE_SYMBOL = struct.pack(">I", 11) + b"CharClipSet"


class DonorExtractionError(Exception):
    """Raised when a donor milo doesn't contain a usable CharClipSet."""
    pass


def _donor_u32(data, p):
    return struct.unpack_from(">I", data, p)[0]


def _donor_symbol(data, p):
    n = _donor_u32(data, p)
    if n > 4096:
        raise DonorExtractionError(f"implausible symbol length {n} at {p:#x}")
    return data[p + 4:p + 4 + n].decode("latin-1"), p + 4 + n


def _find_charclipset_dir(data):
    """Locate the nested CharClipSet DirectoryMeta, version-agnostically.

    Returns (rev_pos, revision, dir_name, entry_count, entry_table_end, nested_kinds), or
    None if no valid CharClipSet is present. rev_pos is the offset of the directory's
    revision u32 (the first byte of the nested DirectoryMeta).

    It searches for every "CharClipSet" type symbol and, for each, tries to parse a nested
    DirectoryMeta at that point. A match must have a plausible revision AND a clean, non-empty
    table of clip entries. This does two useful things at once:
      * accepts any revision (DC1/RB3 = 28, DC3 = 32, ...) - newer revisions add up to a few
        padding/flag bytes before entryCount (DC3 = 1), so we probe offsets 0..4 and take the
        first that yields a valid table; and
      * automatically skips the top-level entry-table reference to "CharClipSet" (which is not
        a directory header and does not parse as a valid entry table).
    """
    search = 0
    while True:
        i = data.find(CHARCLIPSET_TYPE_SYMBOL, search)
        if i < 0:
            return None
        search = i + 1
        rev_pos = i - 4
        if rev_pos < 0:
            continue
        rev = _donor_u32(data, rev_pos)
        if not (16 <= rev <= 64):        # plausible DirectoryMeta revision
            continue
        try:
            p = i + len(CHARCLIPSET_TYPE_SYMBOL)   # just past the "CharClipSet" type symbol
            dname, p = _donor_symbol(data, p)      # directory name (e.g. "clips" / "clips1")
            p += 8                                  # stringTableCount(4) + stringTableSize(4)
            for extra in range(0, 5):               # rev-specific padding before entryCount
                q = p + extra
                entry_count = _donor_u32(data, q)
                if not (1 <= entry_count <= 8192):
                    continue
                qq = q + 4
                kinds = {}
                ok = True
                for _ in range(entry_count):
                    tl = _donor_u32(data, qq)
                    if not (3 <= tl <= 40):
                        ok = False
                        break
                    ttype = data[qq + 4:qq + 4 + tl]
                    if not all(0x20 <= b < 0x7f for b in ttype):
                        ok = False
                        break
                    qq += 4 + tl
                    nl = _donor_u32(data, qq)
                    if not (0 <= nl <= 256):
                        ok = False
                        break
                    qq += 4 + nl
                    k = ttype.decode("latin-1")
                    kinds[k] = kinds.get(k, 0) + 1
                if ok:
                    return (rev_pos, rev, dname, entry_count, qq, kinds)
        except Exception:
            continue


def _charclipset_top_level_name(data):
    """Return the name under which the CharClipSet appears in the donor's TOP-LEVEL entry
    table, or None. This is the name the character/game references it by ("clips" in both
    DC1 and DC3), which differs from the nested directory's own name (DC3 nests it as
    "clips1"). We must inject under the top-level name, or DC3 ends up with the embedded
    set under one name AND the separately-loaded char/.../clips.milo under "clips" - two
    conflicting clip dirs - which corrupts clip loading and misreads a valid clip's version
    (the "CharClip version 38 > 22" crash). DC1 only worked by luck: there the nested name
    already equals the top-level name ("clips")."""
    try:
        dir_off = struct.unpack_from("<I", data, 4)[0]
        p = dir_off
        _rev = _donor_u32(data, p); p += 4
        _t, p = _donor_symbol(data, p)          # dir type ("Character")
        _n, p = _donor_symbol(data, p)          # dir name
        p += 8                                   # stringTableCount + stringTableSize
        # Probe the rev-dependent padding before entryCount (DC3 rev 32 adds one byte).
        for extra in range(0, 5):
            q = p + extra
            ec = _donor_u32(data, q)
            if not (1 <= ec <= 65535):
                continue
            qq = q + 4
            ok = True
            entries = []
            for _ in range(ec):
                tl = _donor_u32(data, qq)
                if not (1 <= tl <= 64):
                    ok = False
                    break
                etype = data[qq + 4:qq + 4 + tl]
                if not all(0x20 <= b < 0x7f for b in etype):
                    ok = False
                    break
                qq += 4 + tl
                nl = _donor_u32(data, qq)
                if not (0 <= nl <= 256):
                    ok = False
                    break
                ename = data[qq + 4:qq + 4 + nl]
                qq += 4 + nl
                entries.append((etype.decode("latin-1"), ename.decode("latin-1")))
            if ok:
                for etype, ename in entries:
                    if etype == "CharClipSet":
                        return ename
                return None
    except Exception:
        return None
    return None


def extract_charclipset_from_donor(donor_path):
    """Lifts the CharClipSet asset out of a donor milo, verbatim.

    Returns (entry_name, raw_bytes) where raw_bytes is the complete serialized entry
    body (CharClipSet object fields + nested DirectoryMeta + all nested clip assets),
    ready to be written into another milo under an entry of type "CharClipSet".

    Accepts donors from any supported game version - Dance Central 1 / Rock Band 3
    (DirectoryMeta rev 28) and Dance Central 3 (rev 32) - by locating the CharClipSet by
    type rather than by a fixed revision.

    Raises DonorExtractionError with a specific reason if the donor can't be used."""
    with open(donor_path, "rb") as f:
        data = f.read()

    if len(data) < 0x820 or (_donor_u32(data, 0) != 0xCABEDEAF and
                             struct.unpack_from("<I", data, 0)[0] != 0xCABEDEAF):
        raise DonorExtractionError(
            "This doesn't look like an uncompressed Milo file (bad magic). Only "
            "uncompressed .milo_xbox donors are supported.")

    found = _find_charclipset_dir(data)
    if found is None:
        raise DonorExtractionError(
            "No CharClipSet found in the donor milo. Pick a full character milo "
            "(one that contains the character's animation clip set).")
    rev_pos, rev, dname, entry_count, entry_table_end, nested_kinds = found

    # The marker immediately before the nested dir ends the CharClipSet object fields;
    # the one before that ends the parent directory body, so the entry starts after it.
    m_end_obj = data.rfind(END_MARKER, 0, rev_pos)
    if m_end_obj < 0:
        raise DonorExtractionError("Malformed donor: no end marker before the CharClipSet.")
    m_prev = data.rfind(END_MARKER, 0, m_end_obj)
    if m_prev < 0:
        raise DonorExtractionError(
            "Malformed donor: couldn't find the directory body boundary before the "
            "CharClipSet.")
    start = m_prev + 4

    # Walk (entry_count + 1) end markers from the end of the nested entry table: one per
    # nested asset body, plus the nested directory's own object body. Record each marker's
    # trailing position: these are safe places to split the milo into read chunks, because
    # each falls right after an 0xADDEADDE marker (between assets) so no single read straddles
    # a chunk boundary - which the game's ChunkStream asserts against. Real DC3 files split the
    # clip set at exactly these points; without them a big clip set becomes one over-large
    # chunk that overruns the read buffer.
    walk = entry_table_end
    markers_needed = entry_count + 1
    split_points = [m_end_obj + 4]      # after the CharClipSet object fields
    for idx in range(markers_needed):
        m = data.find(END_MARKER, walk)
        if m < 0:
            raise DonorExtractionError(
                f"Ran out of end markers while walking the CharClipSet "
                f"({idx} of {markers_needed} found). The donor may be compressed or "
                f"use a layout this extractor doesn't understand.")
        walk = m + 4
        split_points.append(walk)
    end = walk

    if end <= start:
        raise DonorExtractionError("Computed an empty CharClipSet byte range.")

    raw = data[start:end]
    # Asset-boundary offsets relative to the start of `raw`, for chunk splitting. Drop the
    # final one (== len(raw), the end) and anything at 0.
    boundaries = sorted({p - start for p in split_points if start < p < end})
    # Inject under the name the character references (the donor's TOP-LEVEL entry name),
    # not the nested dir's own name - see _charclipset_top_level_name for why DC3 crashes
    # otherwise.
    top_name = _charclipset_top_level_name(data)
    entry_name = top_name if top_name else dname
    name_note = "" if entry_name == dname else f" (top-level entry name; nested dir is '{dname}')"
    _log(f"Donor CharClipSet '{entry_name}'{name_note} (DirectoryMeta rev {rev}): "
         f"{entry_count} nested entries "
         f"({', '.join(f'{k} x{v}' for k, v in sorted(nested_kinds.items()))}), "
         f"{len(raw)} bytes extracted from 0x{start:X}..0x{end:X}, "
         f"{len(boundaries)} internal split points.")
    return entry_name, raw, boundaries

# =====================================================================================
# SECTION 5 -- container (DirectoryMeta + MiloFile)
# Ported from MiloLib/Assets/DirectoryMeta.cs and MiloLib/MiloFile.cs
# Only the Type.Uncompressed path is implemented (matches the original CLI's
# fixed choice for its own output: MiloFile.Type.Uncompressed).
# =====================================================================================

def build_milo_bytes(root_name, mesh_entries, milo_type='other', materials=None, textures=None,
                      bone_trans_entries=None, char_mesh_hide_entries=None, char_hair_entries=None,
                      platform='xbox360', write_tangents=True, outfit_config_entries=None,
                      char_collide_entries=None, raw_entries=None, dir_revision=None,
                      force_dc3_mat_defaults=False, is_dc1=False):
    """mesh_entries: list of (entry_name, local_xfm, world_xfm, parent_obj, vertices,
    faces, bone_transforms, mat_name, needs_ao_calc). needs_ao_calc is True when the
    mesh's material has a normal and/or emissive texture - see write_rnd_mesh's
    force_white_vertex_color docstring for why this is required to avoid the mesh
    rendering black. milo_type: 'other' (plain RndDir, what we've
    validated against a real file) or 'character'/'instrument' (Character - source-read
    only, not yet verified against a real reference file). `materials` is a list of
    (mat_entry_name, settings_dict) and `textures` a list of (tex_entry_name, width,
    height, encoding, bpp, block_data) - see gather_materials_and_textures().
    `bone_trans_entries` (EXPERIMENTAL, see build_armature_trans_entries) is a list
    of (bone_name, local_xfm12, world_xfm12, parent_obj) - real top-level Trans
    entries for each armature bone, for testing whether a custom character needs
    its own embedded skeleton.
    `char_mesh_hide_entries` (EXPERIMENTAL, see write_char_mesh_hide) is a list of
    (entry_name, flags, hides) tuples - one per CharMeshHide asset to embed, for
    testing whether the top-level HIDE_HEAD flag does anything in RB3 at all (see
    write_char_mesh_hide's docstring caveat - nothing found in the RB3 decomp
    consumes this flag so far).
    `char_hair_entries` (EXPERIMENTAL, see write_char_hair) is a list of
    (entry_name, settings, strands) tuples - one per CharHair (.hair) asset to embed,
    for testing whether hair/cloth physics work from a generated .hair file alone
    (collision .coll files not yet implemented).
    `outfit_config_entries` (see write_outfit_config) is a list of
    (entry_name, colors, compute_ao) tuples - one per OutfitConfig (.cfg) asset. An
    instrument/outfit MUST have one or the game hard-asserts (crashes) on select when
    the customize/color panel loads. For now this is a minimal dummy config.

    IMPORTANT: the directory's entry table order must exactly match the order asset
    bodies are actually written in (confirmed against a real file - DirectoryMeta
    reads the table first, then reads each entry's body strictly in that same
    order), so the entry kinds are listed and written in the same fixed
    Mesh -> Trans -> CharMeshHide -> CharHair -> OutfitConfig -> Mat -> Tex sequence
    throughout this function."""
    if materials is None:
        materials = []
    if textures is None:
        textures = []
    if bone_trans_entries is None:
        bone_trans_entries = []
    if char_mesh_hide_entries is None:
        char_mesh_hide_entries = []
    if char_hair_entries is None:
        char_hair_entries = []
    if outfit_config_entries is None:
        outfit_config_entries = []
    if char_collide_entries is None:
        char_collide_entries = []
    if raw_entries is None:
        raw_entries = []
    if dir_revision is None:
        dir_revision = RB3_MILO_REVISION
    # Computed early (moved up from just before the material-writer branch below) so the
    # Character/RndDir/RndMesh writers can all pick DC3_DRAW_REVISION for their embedded
    # RndDrawable - see that constant's comment.
    is_dc3 = (dir_revision == DC3_MILO_REVISION)

    body = MiloWriter(big_endian=True)

    is_character = milo_type in ('character', 'instrument')
    dir_type = "Character" if is_character else "RndDir"

    total_entries = (len(mesh_entries) + len(bone_trans_entries) + len(char_mesh_hide_entries)
                      + len(char_hair_entries) + len(outfit_config_entries)
                      + len(char_collide_entries) + len(raw_entries)
                      + len(materials) + len(textures))

    body.u32(dir_revision)
    body.symbol(dir_type)             # DirectoryMeta.type
    body.symbol(root_name)            # DirectoryMeta.name
    body.i32((total_entries * 2) + 4)       # stringTableCount
    body.u32(0)                       # stringTableSize (game recalculates)
    if dir_revision >= DC3_MILO_REVISION:
        body.u8(0)                    # DC3 (rev >= 32) inserts one byte here before
                                       # entryCount - matches vanilla DC3 files (value 0).
    body.i32(total_entries)
    # raw pass-through entries (e.g. a CharClipSet lifted from a donor milo) go FIRST,
    # matching how a real DC1 character lays out its clip set as entry 0.
    for (raw_type, raw_name, _raw_bytes, *_rest) in raw_entries:
        body.symbol(raw_type)
        body.symbol(raw_name)
    for (entry_name, *_rest) in mesh_entries:
        body.symbol("Mesh")
        body.symbol(entry_name)
    for (bone_name, *_rest) in bone_trans_entries:
        body.symbol("Trans")
        body.symbol(bone_name)
    for (entry_name, *_rest) in char_collide_entries:
        body.symbol("CharCollide")
        body.symbol(entry_name)
    for (entry_name, *_rest) in char_mesh_hide_entries:
        body.symbol("CharMeshHide")
        body.symbol(entry_name)
    for (entry_name, *_rest) in char_hair_entries:
        body.symbol("CharHair")
        body.symbol(entry_name)
    for (entry_name, *_rest) in outfit_config_entries:
        body.symbol("OutfitConfig")
        body.symbol(entry_name)
    for (mat_entry_name, _settings) in materials:
        body.symbol("Mat")
        body.symbol(mat_entry_name)
    for (tex_entry_name, *_rest) in textures:
        body.symbol("Tex")
        body.symbol(tex_entry_name)

    if is_character:
        write_character(body, root_name, is_instrument=(milo_type == 'instrument'), dc3=is_dc3)
    else:
        write_rnd_dir(body, dc3=is_dc3)

    # Block-boundary tracking - confirmed algorithm from MiloLib's MiloFile.cs WriteHandler:
    # after each top-level entry finishes writing, if the bytes written since the last
    # boundary exceed MAX_MILO_BLOCK_SIZE, close a block there. Critically, the FIRST
    # check is measured from the absolute start of the body stream (position 0) - i.e.
    # it includes the header/table/root-dir-body bytes written above as part of the
    # first entry's accounting, exactly matching WriteHandler's own
    # `if (blockSizes.Count == 0) bytesWritten = currentPos` (absolute position, not
    # relative to some prior marker).
    block_sizes = []
    last_boundary = 0

    def _mark_block_boundary():
        bytes_since = len(body) - last_boundary
        if bytes_since > MAX_MILO_BLOCK_SIZE:
            block_sizes.append(bytes_since)
            return len(body)
        return last_boundary

    # Raw pass-through bodies first, matching their position in the entry table above.
    # These are already fully serialized (lifted verbatim from a donor milo). A large clip
    # set (a real DC3 one is ~2.4 MB) must be written across multiple read chunks or it
    # overruns the game's ChunkStream buffer; the chunks may only break at asset boundaries
    # (right after an 0xADDEADDE marker) or a single read straddles a chunk and the game
    # asserts. `raw_boundaries` (from extract_charclipset_from_donor) are exactly those safe
    # points, so we emit the raw in asset-sized segments and let the normal per-boundary
    # check close a block once the accumulated bytes exceed MAX_MILO_BLOCK_SIZE - reproducing
    # how a real DC3 file chunks its clip set.
    for raw_entry in raw_entries:
        raw_bytes = raw_entry[2]
        raw_boundaries = raw_entry[3] if len(raw_entry) > 3 else None
        if raw_boundaries:
            prev = 0
            for b in raw_boundaries:
                body.block(raw_bytes[prev:b])
                last_boundary = _mark_block_boundary()
                prev = b
            body.block(raw_bytes[prev:])
            last_boundary = _mark_block_boundary()
        else:
            body.block(raw_bytes)
            last_boundary = _mark_block_boundary()

    for (entry_name, local_xfm, world_xfm, parent_obj, vertices, faces, bone_transforms,
         mat_name, needs_ao_calc) in mesh_entries:
        write_rnd_mesh(body, entry_name, local_xfm, world_xfm, parent_obj, vertices, faces,
                        mat_name=mat_name, bone_transforms=bone_transforms,
                        force_white_vertex_color=needs_ao_calc, platform=platform,
                        write_tangents=write_tangents, dc3=is_dc3, dc1=is_dc1)
        last_boundary = _mark_block_boundary()

    for (bone_name, local_xfm, world_xfm, parent_obj) in bone_trans_entries:
        write_rnd_trans(body, local_xfm, world_xfm, parent_obj, standalone=True, skip_metadata=False)
        last_boundary = _mark_block_boundary()

    for (_entry_name, coll) in char_collide_entries:
        write_char_collide(body, shape=coll['shape'], radius0=coll['radius0'],
                            radius1=coll['radius1'], length0=coll['length0'],
                            length1=coll['length1'], flags=coll['flags'],
                            mesh_y_bias=coll['mesh_y_bias'],
                            local_xfm=coll['local_xfm'], world_xfm=coll['world_xfm'],
                            parent_obj=coll['parent'])
        last_boundary = _mark_block_boundary()

    for (_entry_name, flags, hides) in char_mesh_hide_entries:
        write_char_mesh_hide(body, flags=flags, hides=hides)
        last_boundary = _mark_block_boundary()

    for (_entry_name, hair_settings, strands) in char_hair_entries:
        write_char_hair(body, hair_settings, strands)
        last_boundary = _mark_block_boundary()

    for (_entry_name, colors, compute_ao) in outfit_config_entries:
        write_outfit_config(body, colors=colors, compute_ao=compute_ao)
        last_boundary = _mark_block_boundary()

    # DC3 (DirectoryMeta rev 32) uses its own rev-70 material class hierarchy; every other
    # supported game (RB3, DC1) uses the flat rev-68 RndMat. See write_dc3_rnd_mat for why an
    # RB3-format material renders normal-mapped DC3 surfaces black.
    for (_mat_entry_name, settings) in materials:
        if is_dc3:
            write_dc3_rnd_mat(body, settings, force_dc3_defaults=force_dc3_mat_defaults)
        else:
            write_rnd_mat(body, settings)
        last_boundary = _mark_block_boundary()

    for (tex_entry_name, width, height, encoding, bpp, block_data, num_mips) in textures:
        external_path = tex_entry_name.replace('.tex', '.png')
        write_rnd_tex(body, width, height, encoding, bpp, block_data, external_path=external_path,
                       platform=platform, num_mips=num_mips)
        last_boundary = _mark_block_boundary()

    # trailing remainder after the last recorded boundary (or the whole body, if no
    # entry ever individually exceeded MAX_MILO_BLOCK_SIZE) - matches MiloFile.cs exactly
    if block_sizes:
        remainder = len(body) - last_boundary
        if remainder > 0:
            block_sizes.append(remainder)
    else:
        block_sizes = [len(body)]

    body_bytes = bytes(body.buf)

    header = MiloWriter(big_endian=False)
    START_OFFSET = 0x810
    header.u32(0xCABEDEAF)            # Type.Uncompressed
    header.u32(START_OFFSET)
    header.u32(len(block_sizes))      # numBlocks
    header.u32(max(block_sizes))      # largestBlock
    for sz in block_sizes:
        header.u32(sz)                # blockSizes[i]
    header.block(bytes(START_OFFSET - len(header.buf)))

    return bytes(header.buf) + body_bytes


# =====================================================================================
# SECTION 5c -- Donor milo mesh injection
#
# Rebuilds a donor milo with its Mesh entries swapped for meshes exported from Blender,
# leaving EVERYTHING else byte-for-byte identical: the directory body (including inline
# subdirectories, LOD lists, sphereBase, minLod, charTest), the CharClipSet animations,
# materials, textures, Trans bones, CharCollide, CharHair - all passed through verbatim.
#
# This exists because a from-scratch DC1 export crashed, and the directory body is the
# most likely culprit: a real DC1 character carries a populated LOD list, sphereBase =
# "bone_pelvis.mesh" and minLod = -1, where our writer emits empty LODs, sphereBase =
# the root name and minLod = 0. Injection sidesteps every one of those differences by
# never regenerating the directory body, so a test isolates exactly one variable: the
# mesh data itself.
#
# The core requirement is being able to split a milo into per-asset byte ranges without
# understanding every asset type. That's what the walker below does:
#   - Known directory-style entries (CharClipSet et al) are [object fields][nested
#     DirectoryMeta], so they're descended into recursively.
#   - Every other asset is terminated by a single 0xADDEADDE end marker.
#   - The parent directory body is skipped by parsing as far as the inline-subdirectory
#     list, recursively skipping each inline subdir, then scanning to the next marker
#     (the remaining directory fields contain no nested markers).
#
# The walk is SELF-VALIDATING: walk_top_level_entries() checks that the final asset ends
# exactly at the end of the file. On a real DC1 character (mo01, 116 entries, 3 inline
# subdirs, 5.0 MB) and a real RB3 outfit (admiraljacket, 14 entries) it lands with zero
# bytes left over. inject_meshes_into_donor() refuses to proceed if that check fails,
# so a misparse can't silently produce a corrupt milo.
# =====================================================================================

MILO_END_MARKER = b'\xAD\xDE\xAD\xDE'

# Asset types serialized as [object fields][nested DirectoryMeta]. NOTE: CharClipGroup is
# deliberately NOT in this set - byte-walking a real CharClipSet proved each CharClipGroup
# consumes exactly one end marker like an ordinary asset.
MILO_DIR_ENTRY_TYPES = frozenset({
    "CharClipSet", "ObjectDir", "RndDir", "Character", "BandCharacter",
    "WorldDir", "PanelDir", "UIPanel", "WorldInstance",
})


class DonorInjectError(Exception):
    """Raised when a donor milo can't be parsed or safely rebuilt."""
    pass


def _mw_u32(d, p):
    return struct.unpack_from('>I', d, p)[0]


def _mw_sym(d, p):
    n = _mw_u32(d, p)
    if n > 4096:
        raise DonorInjectError(f"implausible symbol length {n} at offset {p:#x}")
    return d[p + 4:p + 4 + n].decode('latin-1'), p + 4 + n


def _mw_next_marker(d, p):
    m = d.find(MILO_END_MARKER, p)
    if m < 0:
        raise DonorInjectError(f"no end marker found after offset {p:#x}")
    return m + 4


def _mw_skip_dir_body(d, p, depth=0):
    """Skip an ObjectDir-derived directory body, returning the offset just past its end
    marker. Handles inline subdirectories recursively."""
    objdir_p = None
    for probe in range(0, 5):
        q = p + probe * 4
        rev = _mw_u32(d, q)
        if not (1 <= rev <= 40):
            continue
        alt = struct.unpack_from('>H', d, q + 4)[0]
        orev = struct.unpack_from('>H', d, q + 6)[0]
        if alt == 0 and 0 < orev <= 3:
            objdir_p = q
            break
    if objdir_p is None:
        raise DonorInjectError(f"couldn't locate the ObjectDir header at {p:#x}")

    q = objdir_p + 4      # ObjectDir revision
    q += 4                # objFields altRevision(u16) + revision(u16)
    _t, q = _mw_sym(d, q)     # objFields type
    q += 8                # unk1, unk2
    vpc = _mw_u32(d, q); q += 4
    q += vpc * 12 * 4     # viewport matrices
    q += 4                # currentViewportIdx
    q += 1                # inlineProxy
    _pp, q = _mw_sym(d, q)    # proxyPath
    subc = _mw_u32(d, q); q += 4
    for _ in range(subc):
        _s, q = _mw_sym(d, q)
    q += 1                # inlineSubDir type
    icount = _mw_u32(d, q); q += 4
    if icount:
        for _ in range(icount):
            _nm, q = _mw_sym(d, q)
        rtypes = [d[q + i] for i in range(icount)]; q += icount
        rtalt = [d[q + i] for i in range(icount)]; q += icount
        for i in range(icount):
            if rtypes[i] == 1 and rtalt[i] == 1:
                q += 1                       # inlineCached boolean
            if rtypes[i] == 3 and rtalt[i] == 1:
                continue                     # shared: no nested dir written
            q = _mw_skip_directory_meta(d, q, depth + 1)
    return _mw_next_marker(d, q)


def _mw_skip_asset(d, p, atype, depth=0):
    if atype in MILO_DIR_ENTRY_TYPES:
        q = _mw_next_marker(d, p)            # end of the object fields
        return _mw_skip_directory_meta(d, q, depth + 1)
    return _mw_next_marker(d, p)


def _mw_skip_directory_meta(d, p, depth=0):
    q = p
    q += 4                                   # revision
    _type, q = _mw_sym(d, q)
    _name, q = _mw_sym(d, q)
    q += 8                                   # stringTableCount + stringTableSize
    ec = _mw_u32(d, q); q += 4
    if ec > 65535:
        raise DonorInjectError(f"implausible nested entry count {ec} at {p:#x}")
    entries = []
    for _ in range(ec):
        t, q = _mw_sym(d, q)
        n, q = _mw_sym(d, q)
        entries.append((t, n))
    q = _mw_skip_dir_body(d, q, depth)
    for (t, _n) in entries:
        q = _mw_skip_asset(d, q, t, depth)
    return q


def _mw_collect_dir_body(d, p, depth=0):
    """Like _mw_skip_dir_body, but recurses into every inline nested subdirectory and
    returns (end_offset, collected_entries) where collected_entries is a flat list of
    (type, name, start, end) for every asset found inside those subdirectories (and
    anything nested further inside THEM), in file order.

    This exists because real DC1/DC3 character milos ship a big chunk of their skeleton
    and their per-material textures inside inline nested subdirectories (e.g. a shared
    '../female_skeleton.milo' body-skeleton container and per-character texture packs
    like 'emilia01_textures.milo') rather than as top-level entries - confirmed by byte-
    parsing retail emilia01.milo_xbox: 93 of its 173 real Trans (bone) entries and all 24
    of its real embedded textures live ONLY inside these nested containers, invisible to
    any walk that only looks at the top-level entry table. The subdirectory container
    itself is also recorded as a synthetic ('SubDir', name, start, end) entry so it's
    visible in listings even though it has no single (type,name) pair of its own."""
    objdir_p = None
    for probe in range(0, 5):
        q = p + probe * 4
        rev = _mw_u32(d, q)
        if not (1 <= rev <= 40):
            continue
        alt = struct.unpack_from('>H', d, q + 4)[0]
        orev = struct.unpack_from('>H', d, q + 6)[0]
        if alt == 0 and 0 < orev <= 3:
            objdir_p = q
            break
    if objdir_p is None:
        raise DonorInjectError(f"couldn't locate the ObjectDir header at {p:#x}")

    q = objdir_p + 4
    q += 4
    _t, q = _mw_sym(d, q)
    q += 8
    vpc = _mw_u32(d, q); q += 4
    q += vpc * 12 * 4
    q += 4
    q += 1
    _pp, q = _mw_sym(d, q)
    subc = _mw_u32(d, q); q += 4
    for _ in range(subc):
        _s, q = _mw_sym(d, q)
    q += 1
    icount = _mw_u32(d, q); q += 4
    collected = []
    if icount:
        names = []
        for _ in range(icount):
            nm, q = _mw_sym(d, q)
            names.append(nm)
        rtypes = [d[q + i] for i in range(icount)]; q += icount
        rtalt = [d[q + i] for i in range(icount)]; q += icount
        for i in range(icount):
            if rtypes[i] == 1 and rtalt[i] == 1:
                q += 1                       # inlineCached boolean
            if rtypes[i] == 3 and rtalt[i] == 1:
                continue                     # shared: no nested dir written
            s = q
            q, nested = _mw_collect_directory_meta(d, q, depth + 1)
            collected.append(('SubDir', names[i], s, q))
            collected.extend(nested)
    return _mw_next_marker(d, q), collected


def _mw_collect_asset(d, p, atype, aname, depth=0):
    """Like _mw_skip_asset, but returns (end_offset, nested_entries) - nested_entries is
    non-empty when `atype` is itself a directory-bearing type (CharClipSet, ObjectDir,
    RndDir, etc.) with its own nested DirectoryMeta full of child assets."""
    if atype in MILO_DIR_ENTRY_TYPES:
        q = _mw_next_marker(d, p)
        return _mw_collect_directory_meta(d, q, depth + 1)
    return _mw_next_marker(d, p), []


def _mw_collect_directory_meta(d, p, depth=0):
    """Like _mw_skip_directory_meta, but returns (end_offset, flat_entries) where
    flat_entries includes every asset declared directly in this DirectoryMeta's own
    entry table PLUS everything recursively found inside inline nested subdirectories
    (from the dir body) and inside dir-bearing typed entries."""
    q = p
    rev = _mw_u32(d, q); q += 4              # revision
    _type, q = _mw_sym(d, q)
    _name, q = _mw_sym(d, q)
    q += 8                                   # stringTableCount + stringTableSize
    if rev >= DC3_MILO_REVISION:              # DC3 nested dirs carry one extra header
        q += 1                               # byte vs RB3/DC1 - matches parse_milo_skeleton's
                                              # own top-level handling of the same field.
    ec = _mw_u32(d, q); q += 4
    if ec > 65535:
        raise DonorInjectError(f"implausible nested entry count {ec} at {p:#x}")
    entries_meta = []
    for _ in range(ec):
        t, q = _mw_sym(d, q)
        n, q = _mw_sym(d, q)
        entries_meta.append((t, n))
    q, body_nested = _mw_collect_dir_body(d, q, depth)
    collected = list(body_nested)
    for (t, n) in entries_meta:
        s = q
        q, nested = _mw_collect_asset(d, q, t, n, depth)
        collected.append((t, n, s, q))
        collected.extend(nested)
    return q, collected


def walk_all_entries(data):
    """Like walk_top_level_entries, but the returned 'entries' list is fully flattened:
    every asset found anywhere in the file, including those nested inside inline
    subdirectories. See _mw_collect_dir_body's docstring for why this matters."""
    if len(data) < 0x820:
        raise DonorInjectError("file is too small to be a milo")
    if struct.unpack_from('<I', data, 0)[0] != 0xCABEDEAF:
        raise DonorInjectError(
            "not an uncompressed Milo file (bad magic) - only uncompressed donors "
            "are supported")
    body_start = struct.unpack_from('<I', data, 4)[0]

    p = body_start
    revision = _mw_u32(data, p); p += 4
    dir_type, p = _mw_sym(data, p)
    dir_name, p = _mw_sym(data, p)
    p += 8
    ec = _mw_u32(data, p); p += 4
    entries_meta = []
    for _ in range(ec):
        t, p = _mw_sym(data, p)
        n, p = _mw_sym(data, p)
        entries_meta.append((t, n))
    table_end = p

    dir_body_end, body_nested = _mw_collect_dir_body(data, p)
    ranges = list(body_nested)
    q = dir_body_end
    for (t, n) in entries_meta:
        s = q
        q, nested = _mw_collect_asset(data, q, t, n)
        ranges.append((t, n, s, q))
        ranges.extend(nested)

    if q != len(data):
        raise DonorInjectError(
            f"asset walk didn't consume the file exactly (ended at {q:#x}, file is "
            f"{len(data):#x}, {len(data) - q} bytes unaccounted for).")

    return dict(revision=revision, dir_type=dir_type, dir_name=dir_name,
                body_start=body_start, table_end=table_end,
                dir_body_end=dir_body_end, entries=ranges)


def walk_top_level_entries(data):
    """Splits a milo into its top-level pieces.

    Returns dict with: revision, dir_type, dir_name, body_start, table_end,
    dir_body_end, and entries = [(type, name, start, end)] byte ranges.

    Raises DonorInjectError if the walk doesn't consume the file exactly - that check is
    what makes it safe to splice byte ranges."""
    if len(data) < 0x820:
        raise DonorInjectError("file is too small to be a milo")
    if struct.unpack_from('<I', data, 0)[0] != 0xCABEDEAF:
        raise DonorInjectError(
            "not an uncompressed Milo file (bad magic) - only uncompressed donors "
            "are supported")
    body_start = struct.unpack_from('<I', data, 4)[0]

    p = body_start
    revision = _mw_u32(data, p); p += 4
    dir_type, p = _mw_sym(data, p)
    dir_name, p = _mw_sym(data, p)
    p += 8                                   # stringTableCount + stringTableSize
    ec = _mw_u32(data, p); p += 4
    entries_meta = []
    for _ in range(ec):
        t, p = _mw_sym(data, p)
        n, p = _mw_sym(data, p)
        entries_meta.append((t, n))
    table_end = p

    dir_body_end = _mw_skip_dir_body(data, p)
    ranges = []
    q = dir_body_end
    for (t, n) in entries_meta:
        s = q
        q = _mw_skip_asset(data, q, t)
        ranges.append((t, n, s, q))

    if q != len(data):
        raise DonorInjectError(
            f"asset walk didn't consume the file exactly (ended at {q:#x}, file is "
            f"{len(data):#x}, {len(data) - q} bytes unaccounted for). Refusing to "
            f"rebuild from a parse that isn't provably correct.")

    return dict(revision=revision, dir_type=dir_type, dir_name=dir_name,
                body_start=body_start, table_end=table_end,
                dir_body_end=dir_body_end, entries=ranges)


def inject_meshes_into_donor(donor_path, mesh_entries, platform='xbox360',
                             write_tangents=True):
    """Rebuilds the donor milo with its Mesh entries replaced by `mesh_entries`
    (the same tuples build_milo_bytes takes). Everything else is preserved verbatim.

    Returns the complete new milo file bytes."""
    with open(donor_path, 'rb') as f:
        data = f.read()

    info = walk_top_level_entries(data)
    ranges = info['entries']

    donor_meshes = [(n, s, e) for (t, n, s, e) in ranges if t == 'Mesh']
    donor_mats = [n for (t, n, s, e) in ranges if t == 'Mat']
    _log(f"Donor '{donor_path}': {len(ranges)} top-level entries, "
         f"dir body 0x{info['table_end']:X}..0x{info['dir_body_end']:X} "
         f"({info['dir_body_end'] - info['table_end']} bytes, preserved verbatim).")
    _log(f"  Donor meshes being replaced ({len(donor_meshes)}): "
         f"{', '.join(n for n, s, e in donor_meshes) if donor_meshes else '(none)'}")
    _log(f"  Donor materials available ({len(donor_mats)}): "
         f"{', '.join(donor_mats) if donor_mats else '(none)'}")

    our_names = [e[0] for e in mesh_entries]
    _log(f"  Injecting {len(mesh_entries)} Blender mesh(es): {', '.join(our_names)}")

    # Name-matching guidance: other assets (LOD Groups especially) reference meshes BY
    # NAME. If our names don't match the donor's, those references dangle.
    donor_name_set = {n for n, s, e in donor_meshes}
    unmatched = [n for n in our_names if n not in donor_name_set]
    if unmatched and donor_name_set:
        _log(f"  WARNING: these exported meshes don't match any donor mesh name: "
             f"{', '.join(unmatched)}")
        _log(f"           Other donor assets (e.g. LOD Groups) reference meshes by name, "
             f"so unmatched names may leave dangling references. Consider naming the "
             f"Blender objects to match the donor mesh names above.")

    # Build the new entry list: donor order preserved, our meshes dropped in where the
    # donor's first Mesh was.
    new_entries = []      # (type, name, raw_bytes_or_None)
    inserted = False
    for (t, n, s, e) in ranges:
        if t == 'Mesh':
            if not inserted:
                for me in mesh_entries:
                    new_entries.append(('Mesh', me[0], None))
                inserted = True
            continue      # drop the donor mesh
        new_entries.append((t, n, data[s:e]))
    if not inserted:
        for me in mesh_entries:
            new_entries.append(('Mesh', me[0], None))

    # Serialize the new milo.
    body = MiloWriter(big_endian=True)
    total = len(new_entries)
    body.u32(info['revision'])
    body.symbol(info['dir_type'])
    body.symbol(info['dir_name'])
    body.i32((total * 2) + 4)          # stringTableCount
    body.u32(0)                        # stringTableSize (game recalculates)
    body.i32(total)
    for (t, n, _raw) in new_entries:
        body.symbol(t)
        body.symbol(n)

    # Directory body, verbatim from the donor.
    body.block(data[info['table_end']:info['dir_body_end']])

    block_sizes = []
    last_boundary = 0

    def _boundary():
        nonlocal last_boundary
        bytes_since = len(body) - last_boundary
        if bytes_since > MAX_MILO_BLOCK_SIZE:
            block_sizes.append(bytes_since)
            last_boundary = len(body)

    # Detected from the donor's own container revision rather than a separate parameter -
    # a DC3 donor (rev 32) always needs its replacement meshes' embedded RndDrawable at
    # DC3_DRAW_REVISION too, so this can't drift out of sync with what the donor's other
    # untouched assets (and the game) actually expect. See DC3_DRAW_REVISION's comment.
    donor_is_dc3 = (info['revision'] == DC3_MILO_REVISION)
    mesh_by_name = {me[0]: me for me in mesh_entries}
    for (t, n, raw) in new_entries:
        if raw is not None:
            body.block(raw)
        else:
            (entry_name, local_xfm, world_xfm, parent_obj, vertices, faces,
             bone_transforms, mat_name, needs_ao_calc) = mesh_by_name[n]
            write_rnd_mesh(body, entry_name, local_xfm, world_xfm, parent_obj,
                            vertices, faces, mat_name=mat_name,
                            bone_transforms=bone_transforms,
                            force_white_vertex_color=needs_ao_calc,
                            platform=platform, write_tangents=write_tangents,
                            dc3=donor_is_dc3,
                            # This donor-injection path is DC1-only in practice (DC3 donor
                            # mesh injection isn't implemented - see the caller's error
                            # message), so keepMeshData/hasAOCalculation should always match
                            # DC1's confirmed-real behaviour here, same as the from-scratch
                            # DC1 export path.
                            dc1=True)
        _boundary()

    if block_sizes:
        remainder = len(body) - last_boundary
        if remainder > 0:
            block_sizes.append(remainder)
    else:
        block_sizes = [len(body)]

    body_bytes = bytes(body.buf)

    header = MiloWriter(big_endian=False)
    START_OFFSET = 0x810
    header.u32(0xCABEDEAF)
    header.u32(START_OFFSET)
    header.u32(len(block_sizes))
    header.u32(max(block_sizes))
    for sz in block_sizes:
        header.u32(sz)
    header.block(bytes(START_OFFSET - len(header.buf)))

    _log(f"  Rebuilt donor: {total} entries, {len(body_bytes)} body bytes, "
         f"{len(block_sizes)} block(s).")
    return bytes(header.buf) + body_bytes


# =====================================================================================
# SECTION 6 -- Blender scene -> exporter data structures
#
# No Blender-Z-up -> Milo-Y-up conversion happens anywhere in this section (see the
# comment at the top of collect_mesh_entries for why - it was removed after being
# confirmed, byte-for-byte, to be causing mesh deformation on skinned characters).
# =====================================================================================


def _matrix_to_milo(m):
    """Blender mathutils.Matrix (column-vector convention) -> Milo's row-vector
    style 4x3 (basis vectors as rows, translation as the last row). This is the
    transpose of the 3x4 part of the Blender matrix.

    NOTE: this mapping is reconstructed from the Milo format's field layout,
    not verified against a known-good file yet. If meshes come in mirrored,
    rotated 90 degrees, or otherwise wrong in MiloEditor, this is the first
    place to check - try transposing differently or flipping the axis
    conversion sign.
    """
    return (
        m[0][0], m[1][0], m[2][0],
        m[0][1], m[1][1], m[2][1],
        m[0][2], m[1][2], m[2][2],
        m[0][3], m[1][3], m[2][3],
    )


class BoneLimitExceeded(Exception):
    """Raised (with a list of (mesh_name, bone_count) tuples) when one or more
    meshes reference more than MAX_BONES_PER_MESH distinct weighted bones -
    RndMesh::MaxBones() in the real engine is 40; exceeding it crashes the game."""
    def __init__(self, offenders):
        self.offenders = offenders
        msg = "; ".join(f"'{name}' uses {count} bones (max {MAX_BONES_PER_MESH})"
                         for name, count in offenders)
        super().__init__(msg)


def _find_armature_modifier(obj):
    for mod in obj.modifiers:
        if mod.type == 'ARMATURE' and mod.object is not None:
            return mod
    return None


_BONE_AXIS_CORRECTIONS = {
    'none': None,
    'x90': (1, 90),
    'xneg90': (1, -90),
    'y90': (2, 90),
    'yneg90': (2, -90),
    'z90': (3, 90),
    'zneg90': (3, -90),
}


def _bone_axis_correction_matrix(key):
    from mathutils import Matrix
    entry = _BONE_AXIS_CORRECTIONS.get(key)
    if entry is None:
        return Matrix.Identity(4)
    axis_index, degrees = entry
    axis = 'XYZ'[axis_index - 1]
    import math
    return Matrix.Rotation(math.radians(degrees), 4, axis)


def _resolve_stock_reference_armature(context, explicit_name, only_selected):
    """Find the pristine base-skeleton armature whose bone REST transforms should be
    used as the bind basis for boneTransforms offsets (the "stock rest" feature).

    RB3 always skins against the FIXED shared skeleton, never the armature embedded in
    a custom milo (embedding one crashes anyway). So the only offset basis that skins
    correctly is the stock skeleton's rest. If a modder edits the deform armature's
    rest pose to fit their character's proportions (the standard cross-game modding
    step), offsets computed from that edited rest no longer match what the game
    actually poses and the mesh explodes in-game. Reading the bind-rest from a pristine
    copy of the base skeleton instead keeps the REST pose correct no matter how the
    deform rig was edited; the only residual is mild animation drift (bones pivot about
    their stock positions), the same tolerable tradeoff seen in the GHWT: Definitive
    Edition mod (curled-fingertip artifacts on extreme poses, fine at rest).

    The mesh geometry and vertex weights STILL come from the working deform armature;
    only the offset matrix's bind-rest basis is taken from this reference.

    Resolution order: (1) an explicitly named object; (2) an armature whose name flags
    it as a reference ('stock' / 'reference' / '_ref' / '.ref'); (3) if exactly one
    armature in the export scope deforms no mesh at all, that one. Returns an Armature
    Object, or None if nothing unambiguous was found (caller then falls back to the
    deform rig's own rest, i.e. current behavior). The reference armature should be an
    UNEDITED base-skeleton import placed at the same world transform as the deform rig
    (typically identity/origin) so its bone rests match the game's runtime skeleton."""
    scene_arms = [o for o in context.scene.objects if o.type == 'ARMATURE']
    if explicit_name:
        obj = context.scene.objects.get(explicit_name)
        if obj is not None and obj.type == 'ARMATURE':
            return obj
    for o in scene_arms:
        low = o.name.lower()
        if 'stock' in low or 'reference' in low or '_ref' in low or '.ref' in low:
            return o
    # unused-armature heuristic: deform rigs are Armature-modifier targets of meshes;
    # a pristine reference deforms nothing.
    if only_selected and any(o.type == 'MESH' for o in context.selected_objects):
        meshes = [o for o in context.selected_objects if o.type == 'MESH']
    else:
        meshes = [o for o in context.scene.objects if o.type == 'MESH']
    deform_targets = set()
    for m in meshes:
        for mod in m.modifiers:
            if mod.type == 'ARMATURE' and mod.object is not None:
                deform_targets.add(mod.object.name)
    unused = [o for o in scene_arms if o.name not in deform_targets]
    if len(unused) == 1:
        return unused[0]
    return None


def collect_mesh_entries(context, root_name, only_selected=False, bone_axis_correction='none',
                          offset_multiply_order='mesh_first', stock_reference_armature=None,
                          platform='xbox360', include_tangents=True):
    from mathutils import Matrix

    # No Blender Z-up -> Milo Y-up conversion is applied anywhere below anymore.
    # Confirmed against a real reference file: raw Blender local vertex coordinates
    # (completely unconverted) match the CLI tool's own stored values exactly. The
    # CLI itself does no axis conversion at all - it copies glTF accessor data
    # through verbatim - and per direct confirmation, the game's renderer doesn't
    # care what "up axis" convention the source used; only MiloEditor's own preview
    # viewport displays it looking odd, which is a MiloEditor quirk, not a real
    # problem. Applying our own conversion to vertex data while bone rest transforms
    # used a DIFFERENT (or inconsistently combined) convention is what caused
    # per-bone mesh deformation on multi-bone meshes, even though it looked fine on
    # a single rigid attachment like the earlier hat test.
    axis_correction = _bone_axis_correction_matrix(bone_axis_correction)

    entries = []
    bone_limit_offenders = []
    materials_by_name = {}
    armatures_used = {}

    selected_mesh_objects = [o for o in context.selected_objects if o.type == 'MESH']
    if only_selected and selected_mesh_objects:
        candidate_objects = selected_mesh_objects
    else:
        candidate_objects = [o for o in context.scene.objects if o.type == 'MESH']

    for obj in candidate_objects:
        if obj.type != 'MESH':
            continue

        _log(f"Exporting mesh '{obj.name}'...")

        arm_mod = _find_armature_modifier(obj)
        armature_obj = arm_mod.object if arm_mod else None
        if armature_obj is not None:
            armatures_used[armature_obj.name] = armature_obj

        # Export REST-pose geometry, not whatever pose the armature is currently
        # posed in - the game applies skinning deformation itself at runtime using
        # the boneTransforms offsets below, so baking the current pose into vertex
        # positions here would be wrong (either double-deformed, or just wrong if
        # the rig isn't at rest when exporting). Temporarily hide the Armature
        # modifier so depsgraph evaluation gives us undeformed positions.
        restore_show_viewport = None
        if arm_mod is not None:
            restore_show_viewport = arm_mod.show_viewport
            arm_mod.show_viewport = False
            context.view_layer.update()

        depsgraph = context.evaluated_depsgraph_get()
        obj_eval = obj.evaluated_get(depsgraph)
        mesh = obj_eval.to_mesh()

        if mesh is None or len(mesh.polygons) == 0:
            _log(f"  Skipped '{obj.name}': no polygons.")
            if arm_mod is not None:
                arm_mod.show_viewport = restore_show_viewport
                context.view_layer.update()
            continue

        mesh.calc_loop_triangles()

        # --- pick which UV map to export ---
        # A RB3 mesh vertex carries exactly ONE UV set, sampled by the material's
        # texture. In Blender, the map a texture is authored against is the ACTIVE
        # RENDER map (the uv_layer with active_render=True, the little camera icon) -
        # that is also what a faithful glTF export feeds the material's TEXCOORD_0, and
        # therefore what the glTFMilo CLI reads. It is NOT necessarily the map that
        # happens to be HIGHLIGHTED for editing (mesh.uv_layers.active).
        #
        # Reading uv_layers.active was a latent bug that only bites models with MORE
        # THAN ONE UV map when the highlighted map isn't the texture map (e.g. a second
        # projection / lightmap / mirror-UV set left selected). Saul's own test models
        # have a single map so active == active_render and it looked fine locally, which
        # is exactly why this reproduced only for "some users, sometimes".
        #
        # Confirmed by byte-diffing this plugin's PS3 export of the jinx mod against the
        # glTFMilo CLI export of the SAME model:
        #   * CLI   : every UV inside [0,1],  ~11k verts, faces=19058  (clean, deduped)
        #   * plugin: UVs sprawling to U[-1,1] V[0,2] (~60% out of range), ~52k verts
        # Same face count both ways (identical triangles) but ~4.7x the vertices: the
        # wrong map's coords are un-normalized [-1,1], so v = 1-uv[1] lands in [0,2]
        # (hence the scrambled texturing in-game) AND those coords barely dedup (hence
        # the vertex blow-up, which on a heavier model would also trip the 65535 guard).
        export_uv = None
        if mesh.uv_layers:
            export_uv = next((l for l in mesh.uv_layers if l.active_render), None)
            if export_uv is None:                       # no render flag set (unusual)
                export_uv = mesh.uv_layers[0]           # match glTF TEXCOORD_0 ordering
            active_edit = mesh.uv_layers.active
            if len(mesh.uv_layers) > 1 and active_edit is not None \
                    and active_edit.name != export_uv.name:
                _log(f"  '{obj.name}': {len(mesh.uv_layers)} UV maps - exporting the "
                     f"render map '{export_uv.name}'; the highlighted map "
                     f"'{active_edit.name}' is NOT the texture map and was skipped.")
        has_uv = export_uv is not None
        uv_layer = export_uv.data if has_uv else None

        has_tangents = False
        if has_uv:
            try:
                # Tangents MUST be computed against the SAME UV map we export, or the
                # tangent basis won't line up with the exported UVs (scrambled normal
                # maps at seams). calc_tangents() otherwise defaults to the editing-
                # active map, so name the render map explicitly.
                mesh.calc_tangents(uvmap=export_uv.name)
                has_tangents = True
            except Exception:
                has_tangents = False

        color_layer = None
        if mesh.color_attributes and mesh.color_attributes.active_color:
            color_layer = mesh.color_attributes.active_color

        world_matrix = obj.matrix_world

        # --- resolve per-vertex bone weights (skinned meshes only) ---
        influencing_bones = []          # ordered, unique - local bone-list for this mesh
        vert_weight_data = {}           # vertex index -> up to 4 (bone_name, weight), sorted, normalized

        if armature_obj is not None:
            vgroup_names = {vg.index: vg.name for vg in obj.vertex_groups}
            bone_names_set = {b.name for b in armature_obj.data.bones}

            for i, bv in enumerate(mesh.vertices):
                infl = []
                for g in bv.groups:
                    name = vgroup_names.get(g.group)
                    if name is None or name not in bone_names_set or g.weight <= 0.0:
                        continue
                    infl.append((name, g.weight))
                infl.sort(key=lambda t: t[1], reverse=True)
                infl = infl[:4]
                total = sum(w for _, w in infl)
                if total > 0:
                    infl = [(n, w / total) for (n, w) in infl]
                for (name, _w) in infl:
                    if name not in influencing_bones:
                        influencing_bones.append(name)
                vert_weight_data[i] = infl

            if len(influencing_bones) > MAX_BONES_PER_MESH:
                _log(f"  FAILED '{obj.name}': uses {len(influencing_bones)} bones, "
                     f"exceeds the {MAX_BONES_PER_MESH}-bone-per-mesh limit.")
                bone_limit_offenders.append((obj.name, len(influencing_bones)))
                obj_eval.to_mesh_clear()
                if arm_mod is not None:
                    arm_mod.show_viewport = restore_show_viewport
                    context.view_layer.update()
                continue  # don't bother building this mesh - export will be cancelled anyway

        # dedup vertices by their full per-loop attribute signature
        vert_map = {}
        vertices = []
        faces = []

        def get_color(loop_index, vert_index):
            if color_layer is None:
                return (1.0, 1.0, 1.0, 1.0)
            if color_layer.domain == 'POINT':
                c = color_layer.data[vert_index].color
            else:
                c = color_layer.data[loop_index].color
            return (c[0], c[1], c[2], c[3] if len(c) > 3 else 1.0)

        for tri in mesh.loop_triangles:
            tri_indices = []
            for loop_index in tri.loops:
                loop = mesh.loops[loop_index]
                vert_index = loop.vertex_index
                co = mesh.vertices[vert_index].co
                normal = loop.normal

                if has_uv:
                    uv = uv_layer[loop_index].uv
                    u, v = uv[0], 1.0 - uv[1]
                else:
                    u, v = 0.0, 0.0

                if has_tangents:
                    tangent = loop.tangent
                    # Write Blender's RAW bitangent sign as the tangent W (handedness).
                    #
                    # This was briefly negated (a "flip handedness" experiment) to try to
                    # fix an inverted-normal report, but byte-diffing a KNOWN-GOOD export
                    # (emix2, working in-game) against a KNOWN-BROKEN one (derek, scrambled
                    # normal maps) proved the negation is what actually broke normal maps:
                    # per mesh block, emix2's stored handedness matched the RAW bitangent-sign
                    # convention and even included one mesh whose sign came out OPPOSITE to its
                    # neighbours (genuine per-UV-island variation), whereas derek was uniformly
                    # the NEGATED sign. So the game's shader wants the raw Blender convention.
                    #
                    # dc3-decomp confirms this W is a real DEC4N shader input
                    # (D3DDECLUSAGE_TANGENT, signed 10-10-10-2) that the engine does NOT
                    # recompute at RB3's mesh revision (MakeTangentsLate only runs for
                    # rev < 30; RB3 is 38), so whatever we write here is what the shader uses.
                    #
                    # NOTE: we export UVs V-flipped (v = 1.0 - uv[1], above), but the tangent
                    # vector and its bitangent_sign both come from the SAME calc_tangents()
                    # result, so they stay mutually consistent - the correct move is to write
                    # that sign through untouched, not to "correct" it for the V flip.
                    tw = loop.bitangent_sign
                else:
                    tangent = (1.0, 0.0, 0.0)
                    tw = 1.0

                r, g, b, a = get_color(loop_index, vert_index)

                weights4 = (0.0, 0.0, 0.0, 0.0)
                bones4 = (0, 0, 0, 0)
                if armature_obj is not None:
                    infl = vert_weight_data.get(vert_index, [])
                    w_list = [0.0, 0.0, 0.0, 0.0]
                    b_list = [0, 0, 0, 0]
                    # Weights stay in natural descending-weight order (w_list[0] = largest).
                    for i in range(4):
                        if i < len(infl):
                            _, wgt = infl[i]
                            w_list[i] = wgt
                    # Bone/weight pairing order is PLATFORM-SPECIFIC (confirmed from
                    # glTFMilo Program.cs's own vertex builder AND verified by byte-diffing
                    # real exports): Xbox 360 (compressionType 1) stores bones REVERSED
                    # relative to weights (bone0 <-> smallest-weight influence, bone3 <->
                    # primary), while PS3 (compressionType 2) stores them in NATURAL order
                    # (bone0 <-> primary, matching the weight order). Using the reversed order
                    # on PS3 pairs each weight with the wrong bone. It's invisible on the body
                    # at rest (the shared skeleton sits at bind pose, so every bone contributes
                    # ~identity), but on the HAIR - whose physics bones move - it drags scalp
                    # vertices onto swinging hair bones, which looks like broken/twisted hair.
                    last_valid = 0
                    for i in range(4):
                        src_idx = i if platform == 'ps3' else (3 - i)
                        if src_idx < len(infl):
                            name, _ = infl[src_idx]
                            b_list[i] = influencing_bones.index(name)
                            last_valid = b_list[i]
                        else:
                            b_list[i] = last_valid
                    weights4 = tuple(w_list)
                    bones4 = tuple(b_list)

                # Tangent must participate in the dedup key when exporting tangents. Two
                # loops can share position/normal/UV/color/weights but have DIFFERENT
                # tangents - most commonly opposite handedness across a mirrored-UV island
                # or a UV seam. Without this they'd collapse into one vertex that silently
                # keeps whichever tangent was seen first, giving wrong normal-map lighting
                # along seams. Handedness is only keyed where the format can STORE it
                # (Xbox's 2-bit w); PS3's 11-11-10 has no w, so keying on it there would
                # just emit duplicate vertices with byte-identical data.
                tangent_key = ()
                if include_tangents:
                    tangent_key = (
                        round(tangent[0], 4), round(tangent[1], 4), round(tangent[2], 4),
                    )
                    if platform != 'ps3':
                        tangent_key = tangent_key + (round(tw, 3),)

                key = (
                    round(co.x, 5), round(co.y, 5), round(co.z, 5),
                    round(normal.x, 4), round(normal.y, 4), round(normal.z, 4),
                    round(u, 5), round(v, 5),
                    round(r, 4), round(g, 4), round(b, 4), round(a, 4),
                    bones4, tuple(round(w, 4) for w in weights4),
                    tangent_key,
                )

                idx = vert_map.get(key)
                if idx is None:
                    idx = len(vertices)
                    vert_map[key] = idx
                    vertices.append({
                        "x": co.x, "y": co.y, "z": co.z,
                        "nx": normal.x, "ny": normal.y, "nz": normal.z,
                        "u": u, "v": v,
                        "tx": tangent[0], "ty": tangent[1], "tz": tangent[2], "tw": tw,
                        "r": r, "g": g, "b": b, "a": a,
                        "weights": weights4, "bones": bones4,
                    })
                tri_indices.append(idx)

            if len(vertices) > 65535:
                obj_eval.to_mesh_clear()
                if arm_mod is not None:
                    arm_mod.show_viewport = restore_show_viewport
                    context.view_layer.update()
                raise ValueError(
                    f"'{obj.name}' needs more than 65535 vertices after export "
                    f"processing - split it into multiple meshes."
                )

            faces.append(tuple(tri_indices))

        # --- boneTransforms: per-bone offset matrix (mesh bind pose -> bone-local
        # space), confirmed against dc3-decomp's RndMesh::SetBone() / SkinVertex():
        #   mOffset = meshWorldXfm(bind) "then" Invert(boneWorldXfm(bind))
        # (row-vector composition, mesh-world applied first). NOTE: this is the
        # OPPOSITE multiply order from what glTFMilo's own C# does for this same
        # field - I went with the decompiled engine's own SetBone() over the CLI
        # tool's implementation since the engine is the actual ground truth and
        # the CLI's skinning path doesn't have a confirmed-working reference file
        # the way the static mesh path did. Worth testing both if this is off.
        bone_transforms = []
        if armature_obj is not None:
            for name in influencing_bones:
                bone = armature_obj.data.bones.get(name)
                if bone is None:
                    continue
                # Bind-rest basis for this bone's offset matrix. Normally the mesh's own
                # deform armature. If a stock reference armature was supplied, read the
                # rest from the SAME-NAMED bone there instead - this decouples "what the
                # user did to the deform rig to fit their character" from "the rest pose
                # the game's fixed skeleton actually uses", so a rig with moved/rotated
                # bones still yields a correct REST pose in-game (see
                # _resolve_stock_reference_armature for the full rationale + tradeoff).
                # Falls back to the deform bone if the reference lacks this name.
                rest_arm = armature_obj
                rest_bone = bone
                if stock_reference_armature is not None:
                    ref_bone = stock_reference_armature.data.bones.get(name)
                    if ref_bone is not None:
                        rest_arm = stock_reference_armature
                        rest_bone = ref_bone
                bone_rest_world = rest_arm.matrix_world @ rest_bone.matrix_local @ axis_correction
                try:
                    if offset_multiply_order == 'bone_first':
                        # glTFMilo's own C# order: invert(bone) applied first, then mesh world
                        offset_col = world_matrix @ bone_rest_world.inverted()
                    else:
                        # decompiled engine's SetBone() order (default): mesh world first,
                        # then invert(bone)
                        offset_col = bone_rest_world.inverted() @ world_matrix
                except ValueError:
                    offset_col = Matrix.Identity(4)
                bone_transforms.append((name, _matrix_to_milo(offset_col)))

        mat_name = ""
        needs_ao_calc = False
        if obj.material_slots and obj.material_slots[0].material is not None:
            blender_mat = obj.material_slots[0].material
            mat_name = f"{sanitize_milo_name(blender_mat.name)}.mat"
            if blender_mat.name not in materials_by_name:
                materials_by_name[blender_mat.name] = blender_mat
            # Confirmed against the glTFMilo reference tool: meshes with no real
            # vertex-color data default to black vertex color, which the lit
            # shader path multiplies its output by - rendering the mesh black.
            # The reference tool works around this by forcing full-white vertex
            # color whenever the material has a normal map, but never extended
            # that to emissive-only materials (a gap in the reference tool
            # itself). We check both here to avoid the emissive-black bug.
            settings = blender_mat.gltfmilo_settings
            needs_ao_calc = (settings.normal_tex is not None) or (settings.emissive_tex is not None)

        entry_name = f"{sanitize_milo_name(obj.name)}.mesh"
        milo_matrix = _matrix_to_milo(world_matrix)
        # Default parent is the milo root dir. A per-object "Mesh Trans Parent" override
        # (Object Properties > Milo) lets a mesh point its RndTrans parentObj at an
        # existing in-game bone instead - used by the TBRB Custom Song Asset workflow so a
        # loose .mesh rides e.g. bone_piano_base.mesh the moment it loads. Written verbatim
        # as the Trans parent symbol; blank means "keep the root default".
        mesh_parent = root_name
        obj_settings = getattr(obj, "gltfmilo_object", None)
        if obj_settings is not None:
            override = (obj_settings.mesh_parent or "").strip()
            if override:
                mesh_parent = override
                _log(f"  '{entry_name}': mesh trans parent overridden to '{override}'.")
        # Since this mesh is parented directly to the root dir (no intermediate node,
        # same as a top-level glTF node in the original tool), local == world here.
        # Leaving local_xfm at identity while only setting world_xfm was the bug that
        # made meshes spawn at the origin in-game: the engine walks LOCAL transforms
        # up the parentObj chain at runtime, not the cached world matrix.
        entries.append((entry_name, milo_matrix, milo_matrix,
                         mesh_parent, vertices, faces, bone_transforms, mat_name, needs_ao_calc))

        bone_note = f", {len(bone_transforms)} bone(s)" if bone_transforms else ""
        mat_note = f", material '{mat_name}'" if mat_name else ", no material"
        _log(f"  Exported '{entry_name}': {len(vertices)} verts, {len(faces)} tris{bone_note}{mat_note}.")

        obj_eval.to_mesh_clear()
        if arm_mod is not None:
            arm_mod.show_viewport = restore_show_viewport
            context.view_layer.update()

    if bone_limit_offenders:
        raise BoneLimitExceeded(bone_limit_offenders)

    return entries, materials_by_name, armatures_used


import re

def sanitize_milo_name(name):
    """Milo/HMX symbol names end up as identifiers the game's resource lookups and
    DTA-based data files parse - unlike MiloEditor (which just treats entry names
    as opaque display strings and doesn't care what's in them), the actual game's
    loader may choke on spaces or other characters DTA-style tokenizing doesn't
    expect. Blender datablock names (mesh objects, materials, images) commonly
    contain spaces, dots-as-separators, or auto-numbered suffixes like "Diffuse
    Texture.001" - replace anything that isn't alphanumeric/underscore/period/
    hyphen with an underscore so every entry name in the file is a safe identifier."""
    return re.sub(r'[^A-Za-z0-9_.\-]', '_', name)


_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tga', '.bmp', '.tif', '.tiff', '.exr', '.dds')


def _strip_image_extension(name):
    lower = name.lower()
    for ext in _IMAGE_EXTENSIONS:
        if lower.endswith(ext):
            return name[:-len(ext)]
    return name


def _round_up_4(n):
    return ((n + 3) // 4) * 4


def _round_up_pow2(n, minimum=4):
    """Rounds n UP to the next power of 2, floored at `minimum`. Used for the TBRB
    Custom Song Asset 'Source .png' export path: p9songtool (the actual custom-song
    compiler's PNG -> .tex conversion, in Mackiloha's TextureExtensions.BitmapFromImage)
    hard-rejects any image whose width or height isn't an exact power of 2 - not just a
    multiple of 4 like this exporter's own DXT path requires - throwing and failing the
    whole song build. Rounding up (rather than down or to nearest) avoids cropping detail
    off the source image; the resulting canvas gets stretched to fill exactly like the
    existing DXT resize path already does non-uniformly per axis."""
    n = max(minimum, int(n))
    p = 1
    while p < n:
        p *= 2
    return p


MAX_MILO_TEX_SIZE = 2048  # Confirmed against the glTFMilo reference tool: its own code
                          # comment calls this "the limit to textures in Milo" - textures
                          # larger than 2048 on either axis crash the game at runtime,
                          # regardless of platform. This was unconditionally enforced in
                          # the reference tool originally; a later commit that changed the
                          # *soft* default downscale target from 2048 to 512 accidentally
                          # tied this hard ceiling to the same ignore-limits flag as the
                          # soft default, so enabling "ignore size limits" there (and here,
                          # before this fix) skips the real ceiling too, not just the soft
                          # one - producing textures that crash the game. This constant
                          # must always be enforced, independent of ignore_size_limits.


def _require_loaded_image_buffer(image):
    """Raises a clear, actionable error if `image`'s pixel buffer never actually decoded,
    BEFORE anything tries to .scale()/read .pixels on it and gets Blender's raw, unhelpful
    'RuntimeError: Error: Image 'X' failed to load image buffer'.

    Why this happens: Blender can read a DDS file's HEADER (width/height/etc - so
    image.size is non-zero and it shows up fine in the file browser and material preview
    thumbnail) without being able to actually DECODE its pixel data - Blender's built-in
    DDS support only covers a subset of real-world DDS variants (notably missing most
    BC7/DX10-header/typeless-format files, which are extremely common output from texture
    tools and game-asset extractors). The failure only surfaces the moment something
    (like this exporter) actually needs the decoded pixels, which is exactly the confusing
    experience a plugin user reported: a crash pointing at Blender's own scale()/pixels
    machinery with no indication the source file itself is the problem.

    image.has_data is Blender's own signal for "the pixel buffer is actually loaded",
    independent of whether the header parsed enough to report a size - checking it here
    lets us fail with a real explanation instead of Blender's raw RuntimeError."""
    if not image.has_data:
        import os
        ext = os.path.splitext(image.filepath or image.name)[1].lower()
        dds_note = (
            " This is a known Blender limitation with DDS files specifically: Blender's "
            "built-in DDS reader can parse the header (so it shows a correct size/name "
            "here) without being able to decode many real-world DDS variants - BC7, "
            "DX10-header, and typeless-format DDS files in particular. Re-export/convert "
            "this texture to PNG or TGA (e.g. in GIMP, Photoshop, or an online DDS "
            "converter) and re-link it in Blender, then try exporting again."
        ) if ext == '.dds' else (
            " Try re-saving/re-exporting this image in a different format (PNG or TGA "
            "are the safest bets) and re-linking it in Blender."
        )
        raise RuntimeError(
            f"Image '{image.name}' has a size but its pixel buffer never actually loaded "
            f"(Blender's own image.has_data is False for it) - this is NOT a bug in the "
            f"exporter, Blender itself failed to decode this file's pixel data.{dds_note}")


def get_image_rgba8(image, max_size=512, ignore_size_limits=False):
    """Returns (width, height, rgba_bytes) as flat RGBA8 bytes, downscaled to fit
    max_size unless ignore_size_limits, and rounded up to a multiple of 4 in each
    dimension (required for DXT block compression - a deliberate improvement over
    the CLI tool, which just errors out on non-multiple-of-4 textures instead).

    ignore_size_limits only skips the soft `max_size` default - it never skips
    MAX_MILO_TEX_SIZE (2048), which is a real engine/hardware ceiling, not a
    stylistic default. Without this second, unconditional clamp, "ignore size
    limits" could produce textures that either crash the game outright (over
    the real 2048 ceiling) or take an extremely long time to DXT-encode in
    this file's pure-Python block compressors, appearing to hang the export.

    Flips rows vertically: Blender's `image.pixels` is stored bottom-up, but the
    CLI preserves the image's original top-down file orientation with no flip -
    confirmed by testing (this was previously left unflipped on the assumption the
    two conventions already matched; they don't - textures came out upside-down)."""
    orig_w, orig_h = image.size[0], image.size[1]
    if orig_w == 0 or orig_h == 0:
        raise ValueError(f"Image '{image.name}' has no pixel data - is it missing or unpacked?")
    _require_loaded_image_buffer(image)

    target_w, target_h = orig_w, orig_h
    if not ignore_size_limits and (orig_w > max_size or orig_h > max_size):
        scale = min(max_size / orig_w, max_size / orig_h)
        target_w = max(1, round(orig_w * scale))
        target_h = max(1, round(orig_h * scale))

    # Real hard ceiling - always enforced, even when ignore_size_limits skipped the
    # soft default above. See MAX_MILO_TEX_SIZE docstring for why this must not be
    # gated behind ignore_size_limits the way the reference tool's own check is.
    if target_w > MAX_MILO_TEX_SIZE or target_h > MAX_MILO_TEX_SIZE:
        scale = min(MAX_MILO_TEX_SIZE / target_w, MAX_MILO_TEX_SIZE / target_h)
        target_w = max(1, round(target_w * scale))
        target_h = max(1, round(target_h * scale))

    target_w = max(4, _round_up_4(target_w))
    target_h = max(4, _round_up_4(target_h))

    work_img = image
    is_temp = False
    if (target_w, target_h) != (orig_w, orig_h):
        work_img = image.copy()
        work_img.scale(target_w, target_h)
        is_temp = True

    try:
        pixels = work_img.pixels[:]
        raw = bytes(max(0, min(255, round(v * 255))) for v in pixels)
        row_bytes = target_w * 4
        rows = [raw[i:i + row_bytes] for i in range(0, len(raw), row_bytes)]
        rgba = b"".join(reversed(rows))
    finally:
        if is_temp:
            bpy.data.images.remove(work_img)

    return target_w, target_h, rgba


def export_texture_as_png(image, out_path, max_size=512, ignore_size_limits=False):
    """Writes `image` to `out_path` as a plain, standard .png file - used only by the TBRB
    Custom Song Asset 'Texture Format = Source .png' option.

    The community's custom-song compiler (p9songtool, part of PikminGuts92's Mackiloha)
    expects loose .png source files in the extras folder and does its own PNG -> .tex
    conversion at song-package time, so this writes real uncompressed PNG bytes via
    Blender's own image writer rather than this exporter's compiled RndTex/DXT format.

    Applies the SAME soft/hard size-limit policy as get_image_rgba8 (soft `max_size`
    default unless ignore_size_limits, hard MAX_MILO_TEX_SIZE ceiling always enforced),
    but then rounds each axis UP TO THE NEXT POWER OF 2 rather than merely a multiple of
    4. This is not optional: byte-verified against Mackiloha's actual PNG-import code
    (TextureExtensions.BitmapFromImage) - it hard-rejects any image whose width or height
    isn't an exact power of 2 with "Invalid image resolution... must be a power of 2 and
    at least 4px", throwing and failing the ENTIRE song build, not just that one texture.
    A multiple-of-4 image (e.g. 500x500, which the DXT path would happily accept) passes
    this exporter's own checks but would still blow up the compiler at package time.

    Returns (width, height) of the written PNG."""
    orig_w, orig_h = image.size[0], image.size[1]
    if orig_w == 0 or orig_h == 0:
        raise ValueError(f"Image '{image.name}' has no pixel data - is it missing or unpacked?")
    _require_loaded_image_buffer(image)

    target_w, target_h = orig_w, orig_h
    if not ignore_size_limits and (orig_w > max_size or orig_h > max_size):
        scale = min(max_size / orig_w, max_size / orig_h)
        target_w = max(1, round(orig_w * scale))
        target_h = max(1, round(orig_h * scale))

    # Same real hard ceiling as get_image_rgba8 - always enforced regardless of
    # ignore_size_limits. See MAX_MILO_TEX_SIZE for why.
    if target_w > MAX_MILO_TEX_SIZE or target_h > MAX_MILO_TEX_SIZE:
        scale = min(MAX_MILO_TEX_SIZE / target_w, MAX_MILO_TEX_SIZE / target_h)
        target_w = max(1, round(target_w * scale))
        target_h = max(1, round(target_h * scale))

    # Power-of-2 rounding - see docstring. MAX_MILO_TEX_SIZE (2048) is itself a power of
    # 2, so rounding up can only push an already-near-ceiling value back over it; reclamp
    # by capping at MAX_MILO_TEX_SIZE rather than letting it round up past it.
    target_w = min(_round_up_pow2(target_w), MAX_MILO_TEX_SIZE)
    target_h = min(_round_up_pow2(target_h), MAX_MILO_TEX_SIZE)

    work_img = image
    is_temp = False
    if (target_w, target_h) != (orig_w, orig_h):
        work_img = image.copy()
        work_img.scale(target_w, target_h)
        is_temp = True

    try:
        prev_format = work_img.file_format
        try:
            work_img.file_format = 'PNG'
            work_img.filepath_raw = out_path
            work_img.save()
        finally:
            if not is_temp:
                # Only restore on the original (non-temp) image - a temp copy gets
                # deleted right after anyway.
                work_img.file_format = prev_format
    finally:
        if is_temp:
            bpy.data.images.remove(work_img)

    return target_w, target_h


def build_all_bone_trans_entries(armature_obj, root_name, label="skeleton"):
    """Builds Trans entries for EVERY bone in ONE armature, with no filtering of any kind.

    This is deliberately a separate function from build_armature_trans_entries rather than a
    flag on it. That function serves the RB3/DC paths and carries two silent-drop behaviours
    that must never apply to a skeleton milo:

      1. `skip_skeleton_bones` filters against RB3_SKELETON_BONES (490 names). 102 of the 109
         bones in the retail TBRB skeleton appear in that list, so if it were ever left on for
         TBRB it would gut the skeleton.
      2. It merges MULTIPLE armatures and de-duplicates by bone name, so when a scene holds
         two rigs (a live one and an `Armature_old`), whichever iterates first claims each
         name and the other's version is dropped without the counts looking wrong.

    A skeleton milo is exactly one skeleton, so this takes exactly one armature object and
    emits one Trans per bone - no skip list, no dedup, no vertex-group or deform check. A bone
    with no weights painted on it (facial crease/driver bones like bone_L-crease.mesh,
    bone_footik.mesh, bone_head_lookat.mesh) is still exported: the animation and look-at
    systems resolve those by NAME at runtime, and a missing one crashes the game on song start.

    The count is asserted at the end - if the emitted total ever fails to match the armature's
    bone count, that's a bug and the export should be treated as untrustworthy.
    """
    bones = list(armature_obj.data.bones)
    _log(f"Exporting {label} from armature '{armature_obj.name}': {len(bones)} bone(s) "
         f"(exporting ALL of them - no skip list, no weight/deform filtering).")

    entries = []
    skipped_root = 0
    for bone in bones:
        # A bone whose name equals the directory/root name (e.g. an armature-root bone
        # literally named 'george_skeleton') must NOT be emitted as a Trans: the retail
        # skeleton has no such bone, and a Trans sharing the directory's own name creates a
        # runtime name collision with the ObjectDir. Observed shipping in a custom export
        # (110 Trans vs the retail 109) - drop it here.
        if bone.name == root_name:
            skipped_root += 1
            _log(f"  Skipped bone '{bone.name}': shares the directory/root name - not a real "
                 f"skeleton bone, would collide with the ObjectDir at runtime.")
            continue
        world_xfm_blender = armature_obj.matrix_world @ bone.matrix_local
        if bone.parent is not None:
            parent_world_blender = armature_obj.matrix_world @ bone.parent.matrix_local
            local_xfm_blender = parent_world_blender.inverted() @ world_xfm_blender
            parent_obj = bone.parent.name
        else:
            local_xfm_blender = world_xfm_blender
            parent_obj = root_name
        entries.append((
            bone.name,
            _matrix_to_milo(local_xfm_blender),
            _matrix_to_milo(world_xfm_blender),
            parent_obj,
        ))

    if len(entries) != len(bones) - skipped_root:
        raise ValueError(
            f"Internal error: armature '{armature_obj.name}' has {len(bones)} bones "
            f"({skipped_root} root-named skipped) but {len(entries)} Trans entries were built.")

    # Blender enforces unique bone names within one armature, so a collision here would mean
    # something upstream mangled the list - check anyway rather than ship a short skeleton.
    names = [e[0] for e in entries]
    if len(set(names)) != len(names):
        from collections import Counter
        dupes = [n for n, c in Counter(names).items() if c > 1]
        raise ValueError(f"Duplicate bone name(s) in armature: {', '.join(dupes)}")

    _log(f"  Emitted {len(entries)} Trans entry(ies) - matches the armature bone count.")
    return entries


def _read_skeleton_milo_trans(path):
    """Parse a vanilla/reference TBRB skeleton milo (uncompressed, rev-25 Character dir)
    and return {bone_name: (local_xfm12, world_xfm12, parent_obj)} for every Trans entry.

    Used by the degenerate-bone repair below: the retail skeleton is the authoritative
    rest for the non-deforming driver bones (spot_*, *-crease, head_lookat) that some
    imported GLB rigs collapse onto a single wrong transform. We only read Trans objects
    (CharCollide bodies are skipped by object-boundary, not decoded). This mirrors the
    exact byte layout written by write_rnd_trans(standalone=True, skip_metadata=False)
    and write_object_fields - if either changes, this must change with it.

    Returns {} on any structural problem rather than raising, so a malformed or wrong-game
    reference file degrades to "no repair" instead of aborting the export."""
    try:
        with open(path, 'rb') as f:
            raw = f.read()
    except OSError:
        return {}
    if len(raw) < 12 or struct.unpack_from('<I', raw, 0)[0] != 0xCABEDEAF:
        return {}
    start = struct.unpack_from('<I', raw, 4)[0]
    body = raw[start:]
    p = 0
    be = '>'  # PS3 skeletons are big-endian; this repair path is PS3-only in practice

    def u32():
        nonlocal p
        v = struct.unpack_from(be + 'I', body, p)[0]; p += 4; return v

    def i32():
        nonlocal p
        v = struct.unpack_from(be + 'i', body, p)[0]; p += 4; return v

    def sym():
        nonlocal p
        n = u32()
        s = body[p:p + n].decode('latin-1'); p += n; return s

    try:
        _rev = u32()
        _type = sym()
        _name = sym()
        _strCount = i32()
        _strSize = u32()
        entry_count = i32()
        entries = []
        for _ in range(entry_count):
            et = sym(); en = sym()
            entries.append((et, en))
    except (struct.error, UnicodeDecodeError):
        return {}

    MARK = b'\xAD\xDE\xAD\xDE'
    # First object body is the Character dir - skip to its end marker.
    m = body.find(MARK, p)
    if m < 0:
        return {}
    obj_start = m + 4
    out = {}
    for (etype, ename) in entries:
        mend = body.find(MARK, obj_start)
        if mend < 0:
            break
        obj = body[obj_start:mend]
        if etype == 'Trans':
            try:
                op = 0

                def _u32():
                    nonlocal op
                    v = struct.unpack_from(be + 'I', obj, op)[0]; op += 4; return v

                def _sym():
                    nonlocal op
                    n = struct.unpack_from(be + 'I', obj, op)[0]; op += 4
                    s = obj[op:op + n].decode('latin-1'); op += n; return s

                def _mat12():
                    nonlocal op
                    vals = struct.unpack_from(be + '12f', obj, op); op += 48
                    return tuple(vals)

                def _u8():
                    nonlocal op
                    v = obj[op]; op += 1; return v

                _trev = _u32()
                _of_rev = _u32()      # write_object_fields: (alt<<16)|rev
                _otype = _sym()
                _hasTree = _u8()
                _note = _sym()        # revision > 0
                local = _mat12()
                world = _mat12()
                _constraint = _u32()
                _target = _sym()
                _preserve = _u8()
                parent = _sym()
                out[ename] = (local, world, parent)
            except (struct.error, UnicodeDecodeError, IndexError):
                pass
        obj_start = mend + 4
    return out


def _trans_world_translation(entry):
    """(local12, world12, parent) -> world (x,y,z) translation of that Trans."""
    w = entry[1] if len(entry) >= 2 else entry[0]
    return (w[9], w[10], w[11])


def repair_degenerate_skeleton_bones(bone_trans_entries, reference_trans, root_name,
                                     origin_eps=0.01):
    """Repair driver bones that collapsed to the armature origin on import.

    Some GLB rigs (e.g. a Miku mesh retargeted onto george's skeleton) import the
    non-deforming helper bones - spot_L/R-cheek.mesh, bone_L/R-crease.mesh,
    bone_head_lookat.mesh - stacked onto ONE identical transform that resolves to
    world (0,0,0). Weighted/positioned bones are unaffected; only these origin-collapsed
    drivers are. Left alone they drag the surrounding face skin (the mouth/cheek pull
    the user is seeing) and break the look-at.

    Fix: for each bone whose OWN exported world translation is ~origin while the retail
    reference skeleton has a genuinely non-origin world position for the same name,
    substitute the reference's local+world matrices (parent kept as-is if it matches,
    else taken from the reference too). This is the skeleton-milo analogue of the
    "stock rest offsets" fix already used on the mesh path.

    A bone that is SUPPOSED to sit at the origin (bone_eyes.mesh, bone_footik.mesh -
    origin in the retail file too) is left untouched, because the reference also has it
    at origin and the guard below requires the reference to be non-origin.

    Returns (repaired_entries, repaired_names). No reference -> returns input unchanged."""
    if not reference_trans:
        return bone_trans_entries, []
    repaired = []
    repaired_names = []
    for entry in bone_trans_entries:
        name, local, world, parent = entry
        wx, wy, wz = world[9], world[10], world[11]
        own_is_origin = (wx * wx + wy * wy + wz * wz) < (origin_eps * origin_eps)
        ref = reference_trans.get(name)
        if own_is_origin and ref is not None:
            rlocal, rworld, rparent = ref
            rx, ry, rz = rworld[9], rworld[10], rworld[11]
            ref_is_origin = (rx * rx + ry * ry + rz * rz) < (origin_eps * origin_eps)
            if not ref_is_origin:
                new_parent = parent if parent else rparent
                repaired.append((name, rlocal, rworld, new_parent))
                repaired_names.append(name)
                continue
        repaired.append(entry)
    return repaired, repaired_names


def build_armature_trans_entries(armatures_used, root_name, only_bones=None,
                                  skip_skeleton_bones=True, label="armature",
                                  zero_rotation=False):
    """Builds top-level "Trans" directory entries for armature bones, matching how the
    glTFMilo CLI imports an armature (Program.cs, commit "Import armature..."): for each
    skin-joint bone it creates an RndTrans whose localXfm/worldXfm come from the bone's
    rest transforms and whose parentObj is the parent BONE's name (or the directory name
    for a parentless bone).

    Two behaviours ported from the CLI that matter a lot:

    1. skip_skeleton_bones (default True): base RB3 skeleton bones (RB3_SKELETON_BONES,
       provided by the shared char_shared.milo) are NOT re-emitted - the CLI's
       `if (rb3SkeletonBones.Contains(node.Name)) continue;`. Re-emitting them would
       duplicate the shared armature and (per notes elsewhere in this file) embedding
       the base armature directly crashes the game. So only CUSTOM bones get Trans
       entries.

    2. parentObj is the raw parent bone name even when that parent is a base skeleton
       bone that we DIDN'T emit. That dangling-looking reference is correct: it resolves
       at runtime against the merged shared armature. This is how a custom hair bone
       attaches to e.g. bone_head.mesh.

    `only_bones` (optional): if given, a set of bone names to restrict output to (used by
    the hair path to emit exactly the physics bones referenced by hair strands, and
    nothing else). None means "all bones on the armature" (subject to the skeleton
    skip). Bone names are left raw/unsanitized to exactly match whatever the .hair
    strands and boneTransforms reference.

    `zero_rotation` (default False): write each bone's LOCAL rotation as identity,
    keeping only its local translation, so worldXfm ends up as "the parent's world
    rotation, offset by the bone's own translation" - i.e. no additional twist relative
    to the parent. REQUIRED for hair/cloth physics bones: a real vanilla .hair-driven
    Trans carries no rotation of its own - CharHair.cpp only ever reads each point's
    position and the Y-translation-derived length (see write_char_hair's module
    docstring), so copying Blender's actual bone rotation into these Trans entries
    (correct and necessary for regular skinned-mesh bones, where rotation matters for
    skinning) introduces a twist a real .hair bone never has. This was fixed once before
    by zeroing rotation on exactly these bones; that fix isn't in this unified function,
    which is presumably how it regressed when hair-bone export got folded into the same
    code path as regular armature export."""
    entries = []
    seen_names = set()
    skipped_skeleton = 0
    for armature_obj in armatures_used.values():
        candidate_bones = [b for b in armature_obj.data.bones
                           if only_bones is None or b.name in only_bones]
        _log(f"Exporting {label} '{armature_obj.name}': {len(candidate_bones)} candidate bone(s)...")
        for bone in candidate_bones:
            if skip_skeleton_bones and bone.name in RB3_SKELETON_BONES:
                skipped_skeleton += 1
                continue  # already provided by the shared armature - don't duplicate
            if bone.name in seen_names:
                _log(f"  Skipped bone '{bone.name}': already exported by another armature.")
                continue  # two armatures sharing a bone name - only write it once
            seen_names.add(bone.name)

            world_xfm_blender = armature_obj.matrix_world @ bone.matrix_local
            if bone.parent is not None:
                parent_world_blender = armature_obj.matrix_world @ bone.parent.matrix_local
                local_xfm_blender = parent_world_blender.inverted() @ world_xfm_blender
                # parentObj is the parent bone's name even if that parent is a base
                # skeleton bone we didn't emit - it resolves against the shared armature.
                parent_obj = bone.parent.name
            else:
                local_xfm_blender = world_xfm_blender
                parent_obj = root_name

            if zero_rotation:
                # Keep the bone's local translation (its position/length relative to its
                # parent), drop rotation/scale entirely - see this function's docstring.
                from mathutils import Matrix
                local_xfm_blender = Matrix.Translation(local_xfm_blender.to_translation())
                if bone.parent is not None:
                    world_xfm_blender = parent_world_blender @ local_xfm_blender
                else:
                    world_xfm_blender = local_xfm_blender

            entries.append((
                bone.name,
                _matrix_to_milo(local_xfm_blender),
                _matrix_to_milo(world_xfm_blender),
                parent_obj,
            ))
    if skipped_skeleton:
        _log(f"  Skipped {skipped_skeleton} base-skeleton bone(s) (already in shared armature).")
    _log(f"Exported {label}: {len(entries)} Trans entry(ies) total.")
    return entries


def _hair_bone_names(armatures_used):
    """Returns the set of all bone names referenced by any enabled hair strand across
    the given armatures - i.e. exactly the custom physics bones that need to exist as
    Trans entries for the .hair file to have something to drive."""
    names = set()
    for armature_obj in armatures_used.values():
        hair = getattr(armature_obj.data, "gltfmilo_hair", None)
        if hair is None or not hair.enabled:
            continue
        for strand in hair.strands:
            for item in strand.bones:
                names.add(item.name)
    return names


def gather_materials_and_textures(materials_by_name, max_tex_size=512, ignore_tex_size_limits=False,
                                   platform='xbox360', generate_mips=False, mip_floor=4):
    """Converts {material_name: bpy.types.Material} into:
    - materials: list of (mat_entry_name, settings_dict) for build_milo_bytes/write_rnd_mat
    - textures: list of (tex_entry_name, width, height, encoding, bpp, block_data, num_mips)
      for build_milo_bytes/write_rnd_tex

    Texture roles use fixed encodings, confirmed directly against the real game files
    (not guessed): diffuse = DXT1/BC1, specular = DXT5/BC3, normal = ATI2/BC5 (reading
    the image's R/G channels as X/Y - Z is reconstructed by the shader), emissive = DXT1/BC1.

    PS3 exception: normal maps use DXT1/BC1 instead of ATI2/BC5 - confirmed from the
    glTFMilo commit "Encode normal maps as BC1 on PS3" (the PS3 GPU/shader path doesn't
    use the two-channel BC5 normal format the Xbox path does).

    generate_mips: when True, every texture is written with a full mip chain (down to 4x4)
    instead of a single base level. Dance Central 1 needs this - a vanilla DC1 ATI2/BC5
    normal map ships a mip chain, and DC1 renders a mip-less ATI2 surface black. Off for
    RB3, which renders single-level ATI2 normal maps correctly."""
    materials = []
    texture_by_key = {}   # (image, suffix) -> tex_entry_name, dedups shared textures
    textures = []
    texture_images = {}   # entry_name -> source bpy.types.Image, for the PNG export path
    is_ps3 = (platform == 'ps3')

    def register_texture(image, suffix, encoder, role):
        key = (image, suffix)
        if key in texture_by_key:
            _log(f"  Texture '{image.name}' already exported as '{texture_by_key[key]}' - reusing.")
            return texture_by_key[key]
        base_name = sanitize_milo_name(_strip_image_extension(image.name))
        entry_name = f"{base_name}{suffix}.tex"
        # dedupe by name too, in case two different Image datablocks share a name
        existing_names = {t[0] for t in textures}
        if entry_name in existing_names:
            n = 2
            while f"{base_name}{suffix}_{n}.tex" in existing_names:
                n += 1
            entry_name = f"{base_name}{suffix}_{n}.tex"

        _log(f"Exporting texture '{image.name}' ({role}) -> '{entry_name}'...")

        width, height, rgba = get_image_rgba8(image, max_tex_size, ignore_tex_size_limits)

        block_data, encoding, bpp, num_mips = build_texture_mip_chain(
            rgba, width, height, encoder, generate_mips, mip_floor=mip_floor)

        textures.append((entry_name, width, height, encoding, bpp, block_data, num_mips))
        texture_by_key[key] = entry_name
        texture_images[entry_name] = image
        enc_name = _TEX_ENCODING_NAMES.get(encoding, str(encoding))
        mip_note = f", {num_mips + 1} mip level(s)" if num_mips else ""
        _log(f"  Exported '{entry_name}': {width}x{height}, {enc_name} ({bpp}bpp), "
             f"{len(block_data)} bytes compressed{mip_note}.")
        return entry_name

    # On PS3 normal maps are BC1, not ATI2/BC5 - the encoder differs, and BC1 reads the
    # full RGB (so we feed the normal map's RGB, same as diffuse) rather than the R/G-only
    # ATI2 path.
    normal_encoder = 'bc1' if is_ps3 else 'ati2'

    for name, blender_mat in materials_by_name.items():
        _log(f"Exporting material '{name}'...")
        s = blender_mat.gltfmilo_settings

        diffuse_tex_name = ""
        if s.diffuse_tex is not None:
            diffuse_tex_name = register_texture(s.diffuse_tex, "", 'bc1', 'diffuse')

        normal_tex_name = ""
        if s.normal_tex is not None:
            normal_tex_name = register_texture(s.normal_tex, "_norm", normal_encoder, 'normal')

        specular_tex_name = ""
        if s.specular_tex is not None:
            specular_tex_name = register_texture(s.specular_tex, "_spec", 'bc3', 'specular')

        emissive_tex_name = ""
        if s.emissive_tex is not None:
            emissive_tex_name = register_texture(s.emissive_tex, "_emissive", 'bc1', 'emissive')

        refract_normal_map_tex_name = ""
        if s.refract_normal_map is not None:
            refract_normal_map_tex_name = register_texture(s.refract_normal_map, "_refract_norm", normal_encoder, 'refract normal map')

        settings = {
            'blend': s.blend,
            'base_color': tuple(s.base_color),
            'pre_lit': s.pre_lit,
            'intensify': s.intensify,
            'use_environment': s.use_environment,
            'z_mode': s.z_mode,
            'alpha_cut': s.alpha_cut,
            'alpha_threshold': s.alpha_threshold,
            'specular_rgb': tuple(s.specular_rgb),
            'specular_power': s.specular_power,
            'emissive_multiplier': s.emissive_multiplier,
            'environment_map': s.environment_map,
            'cull': s.cull,
            'de_normal': s.de_normal,
            'anisotropy': s.anisotropy,
            'normal_detail_strength': s.normal_detail_strength,
            'rim_rgb': tuple(s.rim_rgb),
            'rim_power': s.rim_power,
            'shader_variation': s.shader_variation,
            'refract_enabled': s.refract_enabled,
            'refract_strength': s.refract_strength,
            'diffuse_tex_name': diffuse_tex_name,
            'normal_tex_name': normal_tex_name,
            'specular_tex_name': specular_tex_name,
            'emissive_tex_name': emissive_tex_name,
            'refract_normal_map_tex_name': refract_normal_map_tex_name,
            'meta_material': (s.meta_material or DC3_DEFAULT_META_MATERIAL),  # DC3 only; see write_dc3_rnd_mat
        }
        mat_entry_name = f"{sanitize_milo_name(name)}.mat"
        materials.append((mat_entry_name, settings))
        tex_notes = ", ".join(
            f"{label}='{val}'" for label, val in
            (("diffuse", diffuse_tex_name), ("normal", normal_tex_name),
             ("specular", specular_tex_name), ("emissive", emissive_tex_name))
            if val
        ) or "no textures"
        _log(f"  Exported '{mat_entry_name}': blend={settings['blend']}, {tex_notes}.")

    return materials, textures, texture_images


# =====================================================================================
# SECTION 7 -- glTFMilo Material Settings (per-Blender-Material properties)
#
# Ported field-for-field from MiloLib/Assets/Rnd/RndMat.cs. Values/enums here match
# the actual RndMat fields exactly (blend, color, preLit, useEnviron, zMode, alphaCut,
# alphaThreshold, diffuseTex, normalMap, specularMap, emissiveMap, specularRGB,
# specularPower, emissiveMultiplier, environMap) so a future exporter pass can read
# this property group straight across with no translation. Only the fields requested
# so far are user-editable; everything else uses glTFMilo's own CLI defaults (see
# Source/glTFMilo/Program.cs's unconditional mat.* assignments around line 646) and
# isn't exposed in the UI yet.
#
# This section only adds the UI / data storage. It does not yet write Mat/Tex assets
# into the exported .milo file - that's the next piece of work.
# =====================================================================================

# RndMat.Blend enum (byte) - values match MiloLib exactly
MAT_BLEND_ITEMS = (
    ('kBlendDest', "Dest", ""),
    ('kBlendSrc', "Src", "Default - opaque/cutout materials"),
    ('kBlendAdd', "Add", ""),
    ('kBlendSrcAlpha', "Src Alpha", "Standard alpha blending"),
    ('kBlendSrcAlphaAdd', "Src Alpha Add", ""),
    ('kBlendSubtract', "Subtract", ""),
    ('kBlendMultiply', "Multiply", ""),
    ('kPreMultAlpha', "Premultiplied Alpha", ""),
)

# RndMat.ZMode enum (byte)
MAT_ZMODE_ITEMS = (
    ('kZModeDisable', "Disable", ""),
    ('kZModeNormal', "Normal", "Default"),
    ('kZModeTransparent', "Transparent", ""),
    ('kZModeForce', "Force", ""),
    ('kZModeDecal', "Decal", ""),
)

# RndMat.ShaderVariation enum (byte)
MAT_SHADER_VARIATION_ITEMS = (
    ('kShaderVariationNone', "None", "Default - no special shader variation"),
    ('kShaderVariationSkin', "Skin", "Skin shading variation"),
    ('kShaderVariationHair', "Hair", "Hair shading variation"),
)


class GltfMiloMaterialSettings(bpy.types.PropertyGroup):
    blend: EnumProperty(
        name="Blend Mode",
        description="RndMat.blend - how to blend this material's poly into the screen",
        items=MAT_BLEND_ITEMS,
        default='kBlendSrc',
    )
    base_color: FloatVectorProperty(
        name="Base Color",
        description="RndMat.color - base material color",
        subtype='COLOR',
        size=4,
        min=0.0, max=1.0,
        default=(1.0, 1.0, 1.0, 1.0),
    )
    pre_lit: BoolProperty(
        name="Pre-Lit",
        description="RndMat.preLit - use vertex color and alpha for base or ambient",
        default=True,
    )
    intensify: BoolProperty(
        name="Intensify Base Color",
        description="RndMat.intensify - boosts (doubles) the base color, pushing bright "
                     "materials further into the venue's bloom range. This is the trick "
                     "RB3's neon-sign guitar uses on its bulbs: every untextured bulb "
                     "material has this enabled on top of a bright base color and Pre-Lit, "
                     "which is what makes them read as glowing neon rather than just brightly "
                     "lit. Pair with Pre-Lit + a bright base color + no diffuse texture",
        default=False,
    )
    use_environment: BoolProperty(
        name="Use Environment",
        description="RndMat.useEnviron - modulate with environment ambient and lights",
        default=False,
    )
    z_mode: EnumProperty(
        name="Z-Buffer Mode",
        description="RndMat.zMode - how to read and write the z-buffer",
        items=MAT_ZMODE_ITEMS,
        default='kZModeNormal',
    )
    alpha_cut: BoolProperty(
        name="Alpha Cut",
        description="RndMat.alphaCut - cut zero-alpha pixels from the z-buffer",
        default=False,
    )
    alpha_threshold: IntProperty(
        name="Alpha Threshold",
        description="RndMat.alphaThreshold - alpha level below which gets cut (0-255)",
        min=0, max=255,
        default=0,
    )
    diffuse_tex: PointerProperty(
        name="Diffuse Texture",
        description="RndMat.diffuseTex - base texture map, modulated with color and alpha",
        type=bpy.types.Image,
    )
    normal_tex: PointerProperty(
        name="Normal Map",
        description="RndMat.normalMap - texture map that defines lighting normals",
        type=bpy.types.Image,
    )
    specular_tex: PointerProperty(
        name="Specular Map",
        description="RndMat.specularMap - texture map for specular color and power",
        type=bpy.types.Image,
    )
    emissive_tex: PointerProperty(
        name="Emissive Map",
        description="RndMat.emissiveMap - map for self illumination",
        type=bpy.types.Image,
    )
    specular_rgb: FloatVectorProperty(
        name="Specular RGB",
        description="RndMat.specularRGB - color to use when not driven by a specular map",
        subtype='COLOR',
        size=3,
        min=0.0, max=1.0,
        default=(0.255, 0.255, 0.255),
    )
    specular_power: FloatProperty(
        name="Specular Power",
        description="RndMat.specularPower - power to use when not driven by a specular map",
        default=0.0,
    )
    emissive_multiplier: FloatProperty(
        name="Emissive Multiplier",
        description="RndMat.emissiveMultiplier - multiplier applied to emission",
        default=1.0,
    )
    environment_map: BoolProperty(
        name="Environment Map?",
        description='RndMat.environMap - when enabled, uses the game\'s base cube map '
                     '("instruments.cube") for reflections. No support yet for custom '
                     'cube maps',
        default=False,
    )
    cull: BoolProperty(
        name="Cull (Backface)",
        description="RndMat.cull - cull backfaces of this material's polygons "
                     "(equivalent to the CLI tool's !DoubleSided)",
        default=False,
    )
    de_normal: EnumProperty(
        name="De-Normal",
        description="RndMat.deNormal - stored as a float, but only -1, 0, or 1 are "
                     "meaningful values, so it's exposed as a 3-way choice here",
        items=(
            ('-1', "-1", ""),
            ('0', "0", ""),
            ('1', "1", ""),
        ),
        default='0',
    )
    anisotropy: FloatProperty(
        name="Anisotropy",
        description="RndMat.anisotropy",
        default=0.0,
    )
    normal_detail_strength: FloatProperty(
        name="Normal Detail Strength",
        description="RndMat.normalDetailStrength",
        default=0.0,
    )
    rim_rgb: FloatVectorProperty(
        name="Rim RGB",
        description="RndMat.rimRGB - rim lighting color",
        subtype='COLOR',
        size=3,
        min=0.0, max=1.0,
        default=(0.0, 0.0, 0.0),
    )
    rim_power: FloatProperty(
        name="Rim Power",
        description="RndMat.rimPower - rim lighting falloff power",
        default=4.0,
    )
    shader_variation: EnumProperty(
        name="Shader Variation",
        description="RndMat.shaderVariation - special shading path (skin/hair)",
        items=MAT_SHADER_VARIATION_ITEMS,
        default='kShaderVariationNone',
    )
    meta_material: StringProperty(
        name="MetaMaterial (DC3)",
        description="Dance Central 3 only. The .mmat material template this material derives "
                     "from - it drives the actual shading path, so picking the right one matters. "
                     "DC3 resolves it by name against the game's own globally-loaded template "
                     "library (it is NOT shipped inside your character milo) and pulls default/"
                     "forced property values from it. Templates retail Emilia uses, by surface: "
                     "skin/body/face -> 'char_basic_skin.mmat' (or 'char_basic_skin_rim.mmat' for "
                     "rim lighting); clothing -> 'char_basic_rim.mmat'; hair -> "
                     "'char_basic_hair.mmat'; eyes/teeth -> 'char_basic_skin_nospec.mmat'. AVOID "
                     "'char_basic_color.mmat' on a visible surface - it is a flat self-lit color "
                     "template and renders the character as a uniform glowing color instead of a "
                     "properly lit, textured surface. Ignored for RB3 and DC1, which use the flat "
                     "rev-68 material format",
        default=DC3_DEFAULT_META_MATERIAL,
    )

    # --- Refract settings (own collapsible sub-panel in the UI) ---
    refract_enabled: BoolProperty(
        name="Refract Enabled",
        description="RndMat.refractEnabled - enable screen-space refraction for this material",
        default=False,
    )
    refract_strength: FloatProperty(
        name="Refract Strength",
        description="RndMat.refractStrength - how strongly the material refracts",
        default=0.0,
    )
    refract_normal_map: PointerProperty(
        name="Refract Normal Map",
        description="RndMat.refractNormalMap - normal map that drives refraction "
                     "distortion (separate from the main Normal Map)",
        type=bpy.types.Image,
    )


class MATERIAL_OT_gltfmilo_autodetect_textures(bpy.types.Operator):
    """Scans this material's node tree for a Principled BSDF and fills the texture
    slots above from whatever Image Texture nodes feed its Base Color, Normal,
    Specular, and Emission inputs."""
    bl_idname = "material.gltfmilo_autodetect_textures"
    bl_label = "Auto-Detect Textures From Nodes"
    bl_options = {'REGISTER', 'UNDO'}

    @staticmethod
    def _trace_image(socket):
        """Walk backward from a shader input socket through Reroute/Normal Map
        nodes (and, as a fallback, any other single-input node) to find the
        Image Texture node feeding it."""
        if socket is None or not socket.is_linked:
            return None
        node = socket.links[0].from_node
        visited = set()
        while node is not None and id(node) not in visited:
            visited.add(id(node))
            if node.type == 'TEX_IMAGE':
                return node.image
            if node.type == 'REROUTE':
                in0 = node.inputs[0] if node.inputs else None
                node = in0.links[0].from_node if (in0 and in0.is_linked) else None
                continue
            if node.type == 'NORMAL_MAP':
                color_in = node.inputs.get('Color')
                node = color_in.links[0].from_node if (color_in and color_in.is_linked) else None
                continue
            # Fallback: follow the first linked input on any other node type
            next_node = None
            for inp in node.inputs:
                if inp.is_linked:
                    next_node = inp.links[0].from_node
                    break
            node = next_node
        return None

    def execute(self, context):
        mat = context.material
        if mat is None or mat.node_tree is None:
            self.report({'WARNING'}, "This material has no node tree to scan.")
            return {'CANCELLED'}

        principled = next((n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
        if principled is None:
            self.report({'WARNING'}, "No Principled BSDF node found in this material.")
            return {'CANCELLED'}

        settings = mat.gltfmilo_settings
        found = []

        img = self._trace_image(principled.inputs.get('Base Color'))
        if img:
            settings.diffuse_tex = img
            found.append("Diffuse")

        img = self._trace_image(principled.inputs.get('Normal'))
        if img:
            settings.normal_tex = img
            found.append("Normal")

        # "Specular Tint" is where the CLI tool's own convention plugs a specular
        # map in (confirmed against a real rig) - check that before "IOR Level"/
        # the older pre-4.0 "Specular" input name
        specular_input = (principled.inputs.get('Specular Tint')
                           or principled.inputs.get('Specular IOR Level')
                           or principled.inputs.get('Specular'))
        img = self._trace_image(specular_input)
        if img:
            settings.specular_tex = img
            found.append("Specular")

        emission_input = principled.inputs.get('Emission Color') or principled.inputs.get('Emission')
        img = self._trace_image(emission_input)
        if img:
            settings.emissive_tex = img
            found.append("Emissive")

        if found:
            self.report({'INFO'}, f"Auto-detected: {', '.join(found)}")
        else:
            self.report({'WARNING'}, "No connected image textures found on the Principled BSDF.")
        return {'FINISHED'}


class MATERIAL_PT_gltfmilo_settings(bpy.types.Panel):
    bl_label = "glTFMilo Material Settings"
    bl_idname = "MATERIAL_PT_gltfmilo_settings"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "material"

    @classmethod
    def poll(cls, context):
        return context.material is not None

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        settings = context.material.gltfmilo_settings

        layout.prop(settings, "blend")
        layout.prop(settings, "base_color")
        layout.prop(settings, "pre_lit")
        layout.prop(settings, "intensify")
        layout.prop(settings, "use_environment")
        layout.prop(settings, "z_mode")
        layout.prop(settings, "alpha_cut")
        layout.prop(settings, "alpha_threshold")

        box = layout.box()
        box.label(text="Textures", icon='TEXTURE')
        row = box.row(align=True)
        row.label(text="Diffuse")
        row.template_ID(settings, "diffuse_tex", open="image.open")
        row = box.row(align=True)
        row.label(text="Normal Map")
        row.template_ID(settings, "normal_tex", open="image.open")
        row = box.row(align=True)
        row.label(text="Specular")
        row.template_ID(settings, "specular_tex", open="image.open")
        row = box.row(align=True)
        row.label(text="Emissive")
        row.template_ID(settings, "emissive_tex", open="image.open")
        box.operator("material.gltfmilo_autodetect_textures", icon='VIEWZOOM')

        layout.prop(settings, "specular_rgb")
        layout.prop(settings, "specular_power")
        layout.prop(settings, "emissive_multiplier")
        layout.prop(settings, "environment_map")

        box = layout.box()
        box.label(text="Advanced", icon='MODIFIER')
        box.prop(settings, "cull")
        box.prop(settings, "de_normal")
        box.prop(settings, "anisotropy")
        box.prop(settings, "normal_detail_strength")
        box.prop(settings, "rim_rgb")
        box.prop(settings, "rim_power")
        box.prop(settings, "shader_variation")
        box.prop(settings, "meta_material")


class MATERIAL_PT_gltfmilo_refract(bpy.types.Panel):
    """Refract settings - a separate, collapsed-by-default sub-panel so the
    (rarely-used, easily-misunderstood) screen-space refraction options don't
    clutter the main material settings or confuse users."""
    bl_label = "Refract Settings"
    bl_idname = "MATERIAL_PT_gltfmilo_refract"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "material"
    bl_parent_id = "MATERIAL_PT_gltfmilo_settings"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return context.material is not None

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        settings = context.material.gltfmilo_settings

        layout.prop(settings, "refract_enabled")
        layout.prop(settings, "refract_strength")
        row = layout.row(align=True)
        row.label(text="Refract Normal Map")
        row.template_ID(settings, "refract_normal_map", open="image.open")

# =====================================================================================
# SECTION 7b -- glTFMilo Hair Setup (per-Armature CharHair authoring)
#
# Lets the user select the bones that make up a hair strand and record them (in order,
# root-first) as a strand. On export these become a CharHair (.hair) asset - see
# write_char_hair. Each strand stores just the ordered bone names; the actual point
# positions / lengths / matrices are derived from the armature's bone rest transforms
# at export time, and are used DIRECTLY by the runtime sim (the engine does NOT recompute
# them at load - Hookup() only sets up collisions), so they must be correct.
# =====================================================================================

class GltfMiloHairBoneItem(bpy.types.PropertyGroup):
    """One bone in a strand's ordered root-to-tip chain."""
    name: StringProperty(name="Bone")


class GltfMiloHairStrand(bpy.types.PropertyGroup):
    """One hair strand - an ordered chain of bones (root first) plus per-strand
    starting-flip angle. Physics params live on the armature-level settings, matching
    the .hair format (they're global to the CharHair asset, not per-strand)."""
    name: StringProperty(name="Strand Name", default="Strand")
    angle: FloatProperty(
        name="Angle",
        description="CharHair.Strand.angle - starting flip angle in degrees",
        default=0.0,
    )
    hookup_flags: IntProperty(
        name="Hookup Flags",
        description="CharHair.Strand.hookup_flags - a collision-group BITMASK. This strand's "
                     "points collide with a CharCollide volume only if this value shares at "
                     "least one bit with that volume's 'Collision Group Flags' "
                     "(hookup_flags & collide.flags != 0). Set this to the OR of the group "
                     "bits of every collision volume you want this strand to bump against. "
                     "e.g. 63 (= 1+2+4+8+16+32) collides with volumes in groups 1 through 6. "
                     "If 0, this strand collides with nothing",
        default=1, min=0,
    )
    bones: CollectionProperty(type=GltfMiloHairBoneItem)


class GltfMiloHairSettings(bpy.types.PropertyGroup):
    """Armature-level CharHair physics settings + the list of authored strands.
    Float defaults match the values seen in real RB3 .hair files (see the reference
    screenshot: stiffness/torsion 0.1, inertia 0.65, gravity 1.0, weight 0.5)."""
    enabled: BoolProperty(
        name="Export CharHair (.hair) for this Armature",
        description="When on (and 'Export Hair Physics' is enabled in the exporter), "
                     "embeds a CharHair asset built from the strands below",
        default=False,
    )
    simulate: BoolProperty(
        name="Simulate Physics?",
        description="CharHair.simulate - run the hair/cloth physics simulation",
        default=True,
    )
    stiffness: FloatProperty(name="Stiffness", description="CharHair.stiffness - stiffness of each strand", default=0.1)
    torsion: FloatProperty(name="Torsion", description="CharHair.torsion - rotational stiffness of each strand", default=0.1)
    inertia: FloatProperty(name="Inertia", description="CharHair.inertia - inertia of the hair, zero means none", default=0.65)
    gravity: FloatProperty(name="Gravity", description="CharHair.gravity - gravity of the hair, one is normal", default=1.0)
    weight: FloatProperty(name="Weight", description="CharHair.weight - weight of the hair, one is normal", default=0.5)
    friction: FloatProperty(name="Friction", description="CharHair.friction - hair friction against each other", default=0.0)
    min_slack: FloatProperty(name="Min Slack", description="CharHair.minSlack - if using sides, how far in it could go", default=0.0)
    max_slack: FloatProperty(name="Max Slack", description="CharHair.maxSlack - if using sides, how far out it could go", default=0.0)

    strands: CollectionProperty(type=GltfMiloHairStrand)
    active_strand_index: IntProperty(default=0)


def _selected_bone_names(context, armature_obj):
    """Returns the set of selected bone names, working across Blender modes. As of
    Blender 5.0 the `.select` attribute was removed from Bone (the object/pose-mode
    rest datablock), so iterating `armature_obj.data.bones` and reading `b.select`
    raises AttributeError. The mode-appropriate context selection lists are the
    supported way to read selection:
      - Pose mode:   context.selected_pose_bones (each has a .name)
      - Edit mode:   context.selected_bones / context.selected_editable_bones
      - Object mode: nothing is "selected" at the bone level; fall back to any pose
                     bones flagged, else empty.
    We collect just the NAMES here (never holding bone pointers across a mode switch,
    per Blender's own guidance) and resolve them against data.bones afterwards for the
    parent/child chain walk."""
    names = set()

    # Prefer whichever context list is populated for the current mode.
    for attr in ("selected_pose_bones", "selected_bones", "selected_editable_bones"):
        seq = getattr(context, attr, None)
        if seq:
            for b in seq:
                # Only keep bones that belong to THIS armature (context lists can span
                # multiple selected armatures).
                name = getattr(b, "name", None)
                if name and name in armature_obj.data.bones:
                    names.add(name)
            if names:
                return names

    return names


def _ordered_bones_from_selection(context, armature_obj):
    """Given an armature with some bones selected (in pose or edit mode), returns the
    selected bones (as data.bones entries) ordered root-first along their parent chain.
    Raises ValueError if the selection isn't a single connected chain (a strand must be
    one linear chain - the engine walks TransChildren().front() from the root, i.e. a
    single path)."""
    selected_set = _selected_bone_names(context, armature_obj)
    if not selected_set:
        raise ValueError(
            "No bones selected. Enter Pose or Edit mode and select the bones of a "
            "single hair strand first."
        )

    bones = armature_obj.data.bones
    selected = [bones[name] for name in selected_set]

    roots = [b for b in selected if (b.parent is None or b.parent.name not in selected_set)]
    if len(roots) != 1:
        raise ValueError(
            f"Selection has {len(roots)} root bones - a strand must be a single "
            f"connected chain with exactly one root (a bone whose parent is unselected)."
        )

    ordered = []
    current = roots[0]
    while current is not None:
        ordered.append(current)
        sel_children = [c for c in current.children if c.name in selected_set]
        if len(sel_children) > 1:
            raise ValueError(
                f"Bone '{current.name}' has {len(sel_children)} selected children - "
                f"a strand must be a single linear chain, not a branching tree."
            )
        current = sel_children[0] if sel_children else None

    if len(ordered) != len(selected):
        raise ValueError(
            "Selected bones don't form one connected chain - some selected bones "
            "aren't part of the root's parent chain."
        )
    return ordered


class ARMATURE_OT_gltfmilo_create_hair_strand(bpy.types.Operator):
    """Create a hair strand from the currently selected armature bones. The bones must
    form a single connected chain (one root, no branching); they're recorded root-first
    so the exporter knows the physics bone order and parenting chain."""
    bl_idname = "armature.gltfmilo_create_hair_strand"
    bl_label = "Create Hair Strand from Selected Bones"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and obj.type == 'ARMATURE'

    def execute(self, context):
        armature_obj = context.object
        try:
            ordered = _ordered_bones_from_selection(context, armature_obj)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        hair = armature_obj.data.gltfmilo_hair
        strand = hair.strands.add()
        strand.name = f"Strand {len(hair.strands)}"
        for bone in ordered:
            item = strand.bones.add()
            item.name = bone.name
        hair.active_strand_index = len(hair.strands) - 1

        self.report({'INFO'}, f"Created '{strand.name}' from {len(ordered)} bone(s): "
                              f"{' -> '.join(b.name for b in ordered)}")
        return {'FINISHED'}


class ARMATURE_OT_gltfmilo_remove_hair_strand(bpy.types.Operator):
    """Remove the selected hair strand."""
    bl_idname = "armature.gltfmilo_remove_hair_strand"
    bl_label = "Remove Hair Strand"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.object
        if obj is None or obj.type != 'ARMATURE':
            return False
        return len(obj.data.gltfmilo_hair.strands) > 0

    def execute(self, context):
        hair = context.object.data.gltfmilo_hair
        idx = hair.active_strand_index
        if 0 <= idx < len(hair.strands):
            hair.strands.remove(idx)
            hair.active_strand_index = max(0, idx - 1)
        return {'FINISHED'}


class DATA_UL_gltfmilo_hair_strands(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.label(text=item.name, icon='STRANDS')
        row.label(text=f"{len(item.bones)} bone(s)")


class DATA_PT_gltfmilo_hair(bpy.types.Panel):
    bl_label = "glTFMilo Hair Setup"
    bl_idname = "DATA_PT_gltfmilo_hair"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "data"

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == 'ARMATURE'

    def draw(self, context):
        layout = self.layout
        hair = context.object.data.gltfmilo_hair

        layout.prop(hair, "enabled")

        col = layout.column()
        col.enabled = hair.enabled

        col.operator("armature.gltfmilo_create_hair_strand", icon='ADD')

        row = col.row()
        row.template_list("DATA_UL_gltfmilo_hair_strands", "", hair, "strands",
                          hair, "active_strand_index", rows=3)
        side = row.column(align=True)
        side.operator("armature.gltfmilo_remove_hair_strand", icon='REMOVE', text="")

        if 0 <= hair.active_strand_index < len(hair.strands):
            strand = hair.strands[hair.active_strand_index]
            box = col.box()
            box.prop(strand, "name")
            box.prop(strand, "angle")
            box.prop(strand, "hookup_flags")
            box.label(text="Bones (root -> tip):")
            for item in strand.bones:
                box.label(text=f"    {item.name}", icon='BONE_DATA')

        col.separator()
        box = col.box()
        box.label(text="Physics Settings", icon='PHYSICS')
        box.prop(hair, "simulate")
        box.prop(hair, "stiffness")
        box.prop(hair, "torsion")
        box.prop(hair, "inertia")
        box.prop(hair, "gravity")
        box.prop(hair, "weight")
        box.prop(hair, "friction")
        box.prop(hair, "min_slack")
        box.prop(hair, "max_slack")
        box.label(text="Wind: world.wind (fixed)", icon='FORCE_WIND')


def build_char_hair_entries(armatures_used, root_name):
    """Builds CharHair (.hair) entries from any armatures whose gltfmilo_hair.enabled
    is set and that have at least one strand. Returns a list of
    (entry_name, settings_dict, strands_list) tuples for build_milo_bytes /
    write_char_hair. Point positions/lengths and the base/root matrices are derived
    from each bone's rest transform; the engine's Hookup() largely recomputes them at
    load, so these are a valid, self-consistent initial state rather than the final
    simulated one."""
    entries = []
    for armature_obj in armatures_used.values():
        hair = getattr(armature_obj.data, "gltfmilo_hair", None)
        if hair is None or not hair.enabled or len(hair.strands) == 0:
            continue

        settings = {
            'stiffness': hair.stiffness,
            'torsion': hair.torsion,
            'inertia': hair.inertia,
            'gravity': hair.gravity,
            'weight': hair.weight,
            'friction': hair.friction,
            'min_slack': hair.min_slack,
            'max_slack': hair.max_slack,
            'simulate': hair.simulate,
            'wind': DEFAULT_HAIR_WIND,
        }

        strands_out = []
        for strand in hair.strands:
            bone_names = [item.name for item in strand.bones]
            if not bone_names:
                continue

            points = []
            for bname in bone_names:
                bone = armature_obj.data.bones.get(bname)
                if bone is None:
                    continue
                world_head = armature_obj.matrix_world @ bone.head_local
                points.append({
                    'pos': (world_head.x, world_head.y, world_head.z),
                    'bone': bname,
                    'length': bone.length,
                    'radius': 0.0,
                    'outer_radius': -1.0,
                    'side_length': -1.0,
                })

            root_bone = armature_obj.data.bones.get(bone_names[0])
            if root_bone is not None:
                # The seed matrix MUST equal the root bone's OWN Trans local rotation,
                # exactly as the engine computes mBaseMat = mRoot->LocalXfm(). The decomp's
                # Hookup() only wires up collisions - it does NOT call SetRoot - so the
                # STORED matrix is used directly and drives every strand's twist reference.
                # Verified against a real vanilla RB3 .hair (female_hair_ladylayered):
                # mBaseMat == that bone's Trans local rotation byte-for-byte.
                #
                # The root bone's Trans entry is now written with zero_rotation=True (see
                # build_armature_trans_entries's docstring - hair-driven Trans bones carry
                # no rotation of their own in a real .hair file), so its local rotation is
                # always identity by construction. base_mat/root_mat must match that
                # exactly, so it's just IDENTITY_MATRIX3 directly rather than recomputed
                # from Blender's actual (rotated) bone matrix - reintroducing the real
                # rotation here would silently disagree with the zeroed Trans entry and
                # reproduce the exact twisting bug zero_rotation exists to fix.
                base_mat = IDENTITY_MATRIX3
            else:
                base_mat = IDENTITY_MATRIX3

            strands_out.append({
                'root': bone_names[0],
                'angle': strand.angle,
                'points': points,
                'base_mat': base_mat,
                'root_mat': base_mat,
                'hookup_flags': strand.hookup_flags,
            })

        if strands_out:
            entry_name = f"{sanitize_milo_name(root_name)}.hair"
            entries.append((entry_name, settings, strands_out))

    return entries


def build_char_collide_entries(armatures_used, root_name):
    """Builds CharCollide (.coll) entries from any bones whose gltfmilo_collision.enabled
    is set. Returns a list of (entry_name, coll_dict) tuples for build_milo_bytes /
    write_char_collide.

    Each volume parents to ITS OWN bone (the volume sits at the bone and is carried by
    it), so localXfm is identity and worldXfm is the bone's rest world transform. The
    entry is named "<bone>.coll". This mirrors how the volume attaches in the game: a
    CharCollide is an RndTransformable whose parent is the bone it belongs to."""
    entries = []
    seen_names = set()
    for armature_obj in armatures_used.values():
        arm = armature_obj.data
        for bone in arm.bones:
            coll = getattr(bone, "gltfmilo_collision", None)
            if coll is None or not coll.enabled:
                continue

            entry_name = f"{bone.name}.coll"
            if entry_name in seen_names:
                _log(f"  Skipped collision on '{bone.name}': already exported.")
                continue
            seen_names.add(entry_name)

            # The volume is parented to its own bone, so it lives at that bone's origin:
            # local transform is identity, world transform is the bone's rest world matrix.
            world_xfm_blender = armature_obj.matrix_world @ bone.matrix_local

            shape = coll.shape
            length0, length1 = coll.length0, coll.length1
            if length0 > length1:
                length0, length1 = length1, length0  # game requires length0 <= length1

            entries.append((entry_name, {
                'shape': COLL_SHAPE_TO_INT.get(shape, COLL_SHAPE_SPHERE),
                'radius0': coll.radius0,
                'radius1': coll.radius1,
                'length0': length0,
                'length1': length1,
                'flags': coll.flags,
                'mesh_y_bias': coll.mesh_y_bias,
                'local_xfm': IDENTITY_MATRIX,
                'world_xfm': _matrix_to_milo(world_xfm_blender),
                'parent': bone.name,
            }))
    return entries


def _on_milo_type_update(self, context):
    """When the user switches export Type, set sensible per-type defaults for the
    armature/hair toggles so they don't have to remember to tick them. Instruments
    need their bone rig embedded (that's what makes a guitar load and hold correctly),
    so armature export defaults ON for them; hair physics is enabled by default too so
    it can be experimented with on instruments. Switching away from instrument doesn't
    force anything back off - the user stays in control after the initial nudge."""
    if self.milo_type == 'instrument':
        self.export_armature = True
        self.export_hair = True


class EXPORT_OT_milo_scene(bpy.types.Operator, ExportHelper):
    """Export the current scene to a Milo (.milo) file"""
    bl_idname = "export_scene.milo"
    bl_label = "Export Milo Scene"
    bl_options = {'PRESET'}

    filename_ext = ".milo_xbox"
    filter_glob: StringProperty(
        default="*.milo_xbox;*.milo_ps3;*.milo",
        options={'HIDDEN'},
    )

    platform: EnumProperty(
        name="Platform",
        description="Target console for the exported Milo file",
        items=(
            ('xbox360', "Xbox 360", "Export for Xbox 360"),
            ('ps3', "PS3", "Export for PlayStation 3 (different vertex packing, no texture "
             "byte-swap, and BC1 normal maps - otherwise identical to Xbox)"),
        ),
        default='xbox360',
    )

    game: EnumProperty(
        name="Game",
        description="Target game / revision set",
        items=(
            ('rb3', "Rock Band 3", "Currently the most thoroughly tested path"),
            ('dc1', "Dance Central 1", "EXPERIMENTAL. DC1 uses byte-identical format "
             "revisions to RB3 (DirectoryMeta 28 / Mesh 38 / Mat 68 / Tex 11 / "
             "CharCollide 7), so the same asset writers are used. Xbox 360 only. "
             "Everything is written flat into the milo root - the game resolves objects "
             "by name (ObjectDir::FindObject checks its own entries before recursing into "
             "subdirectories), so the subdirectory layout real DC1 files use is a "
             "load-time sharing optimisation rather than a requirement"),
            ('dc3', "Dance Central 3", "EXPERIMENTAL. Same asset writers as DC1/RB3, but the "
             "container's DirectoryMeta is written at revision 32 (with the extra rev-32 "
             "header byte) so it is revision-consistent with a DC3 CharClipSet. Use a DC3 "
             "donor to copy that game's win/intro animation clips - injecting them into a "
             "rev-28 (DC1) container instead makes DC3 misread the clips and crash. Xbox 360 "
             "only"),
            ('rb2', "Rock Band 2", "Not yet implemented"),
            ('tbrb', "The Beatles: Rock Band", "Not yet implemented"),
        ),
        default='rb3',
    )

    donor_mode: EnumProperty(
        name="Donor Mode",
        description="What to do with the donor milo",
        items=(
            ('inject_meshes', "Inject Meshes Into Donor",
             "Rebuild the DONOR milo with its meshes replaced by the Blender meshes, "
             "leaving everything else byte-for-byte untouched (directory body, LOD lists, "
             "CharClipSet animations, materials, textures, bones, collision, hair). The "
             "output is the donor with new geometry - materials are NOT exported, so mesh "
             "material references should match the donor's existing material names"),
            ('charclipset', "Copy CharClipSet Only",
             "Build the milo from scratch as usual, but copy the CharClipSet (animation "
             "clip set) verbatim out of the donor as entry 0"),
        ),
        default='inject_meshes',
    )

    donor_milo: StringProperty(
        name="Donor Milo (CharClipSet)",
        description="EXPERIMENTAL, Dance Central only. Path to a real DC1 or DC3 character "
                     "milo to use as the donor. What gets taken from it depends on Donor "
                     "Mode above (DC3 supports CharClipSet copy only). The CharClipSet "
                     "extractor auto-detects the donor's revision (DC1 = 28, DC3 = 32). "
                     "Leave empty to export a fresh milo with no donor data",
        subtype='FILE_PATH',
        default="",
    )

    milo_type: EnumProperty(
        name="Type",
        description="Intended use of the exported Milo directory",
        items=(
            ('character', "Character", "Implemented, but not yet verified against a real "
             "reference file the way the static mesh path was - test carefully"),
            ('venue', "Venue", "Not yet implemented (needs RndGroup support)"),
            ('instrument', "Instrument", "For guitars/basses/etc. Uses the "
             "colorpalettes.milo shared subdir (not the character skeleton). The engine "
             "also expects the instrument's animation bone rig (bone_target_*, "
             "bone_pos_string*, spot_neck_fret*, etc.) to be present - export those from "
             "your guitar armature. Selecting this auto-enables armature + hair export"),
            ('custom_song_asset', "Custom Song Asset",
             "The Beatles: Rock Band ONLY. Instead of building a milo, export each mesh/"
             "material/texture as a LOOSE standalone .mesh, .mat and .tex file into the "
             "chosen folder. TBRB custom songs can load these from an 'extras' folder and "
             "wire them into the world at runtime via song.dta scripting (set_trans_parent, "
             "add_object, set diffuse_tex...). Use the per-object 'Milo Object > Trans "
             "Parent' setting to bake a bone parent into the .mesh so it rides an existing "
             "in-game object without a set_trans_parent call. No container, no LOD groups, "
             "no subdirectory references - just the raw assets"),
            ('other', "Other", "Plain RndDir - the most thoroughly validated path"),
        ),
        default='other',
        update=_on_milo_type_update,
    )

    instrument_type: EnumProperty(
        name="Instrument",
        description="Which instrument this milo is, for Type=Instrument. This selects the "
                     "OutfitConfig (.cfg) name the game looks up when you select the item - "
                     "it MUST match or the customize/color panel hard-crashes on select. "
                     "Confirmed from the RB3 decomp (GetConfigNameFromAssetType) and by "
                     "byte-diffing real vanilla instruments: guitar->guitar.cfg, "
                     "bass->bass.cfg, drum->drum.cfg, mic->mic.cfg, keyboard->keyboard.cfg",
        items=(
            ('guitar', "Guitar", "Emits guitar.cfg"),
            ('bass', "Bass", "Emits bass.cfg"),
            ('drum', "Drums", "Emits drum.cfg"),
            ('mic', "Microphone", "Emits mic.cfg"),
            ('keyboard', "Keyboard", "Emits keyboard.cfg"),
        ),
        default='guitar',
    )

    tbrb_target: EnumProperty(
        name="TBRB Milo",
        description="The Beatles: Rock Band splits a character across THREE milos - a "
                     "skeleton milo (bones + collision), a base head/hands milo, and an "
                     "outfit+hair milo. This picks which one to export",
        items=(
            ('skeleton', "Skeleton (bones + collision)",
             "Export <beatle>_skeleton.milo - the Trans bone hierarchy taken from the export "
             "armature plus any CharCollide (.coll) volumes. This is the piece the original "
             "glTFMilo CLI never exported. IMPLEMENTED and byte-verified against a real PS3 "
             "george_skeleton"),
            ('meshes', "Meshes (head+hands / outfit)",
             "Export the mesh milo: meshes + materials + textures, written flat with no LOD "
             "groups. Meshes are written at revision 33 (glTFMilo's target, confirmed working "
             "in-game) rather than retail's 36 - Harmonix loaders read older mesh revisions, "
             "and rev 33 stores tangents as plain floats so the RB3 flat-normal-map bug "
             "cannot occur. Export the skeleton milo separately and install both"),
        ),
        default='skeleton',
    )

    tbrb_beatle: EnumProperty(
        name="Beatle",
        description="Which Beatle this skeleton is for. This ONLY sets the output file name "
                     "(<beatle>_skeleton) - the skeleton content comes from your export "
                     "armature regardless. Pick 'Custom' to keep whatever name you typed",
        items=(
            ('george', "George", "Output george_skeleton"),
            ('john', "John", "Output john_skeleton"),
            ('paul', "Paul", "Output paul_skeleton"),
            ('ringo', "Ringo", "Output ringo_skeleton"),
            ('custom', "Custom (keep typed name)",
             "Use the filename you typed in the dialog instead of forcing <beatle>_skeleton"),
        ),
        default='george',
    )

    tbrb_mesh_kind: EnumProperty(
        name="Mesh Milo Kind",
        description="Which kind of mesh milo to write. This sets the subdirectory references, "
                     "which differ: an outfit milo points at the base head/hands milo plus a "
                     "shared outfit milo (and inherits the skeleton through them), while a "
                     "head/hands milo points at the skeleton milo directly",
        items=(
            ('outfit', "Outfit", "References base head/hands milo + shared outfit milo"),
            ('headhands', "Head / Hands", "References the skeleton milo directly"),
        ),
        default='outfit',
    )

    tbrb_base_milo: StringProperty(
        name="Base Head/Hands Milo",
        description="Outfit only: relative path to the base head/hands milo this outfit sits "
                     "on. Retail cavern02 uses '../george_headhands_short.milo'. Leave empty "
                     "to default to '../george_headhands_long.milo'",
        default="../george_headhands_long.milo",
    )

    tbrb_share_milo: StringProperty(
        name="Shared Outfit Milo",
        description="Outfit only: relative path to the shared outfit milo. Retail cavern02 "
                     "uses '../../shared/outfit/cavern02_share.milo'. Leave empty to omit it",
        default="",
    )

    tbrb_test_hardcode_subdirs: BoolProperty(
        name="TEST: Missing Sub Directory Theory",
        description="Isolation test. When on, ignores the fields above and writes the EXACT "
                     "two subdirectories from the retail cavern02 outfit milo "
                     "(../george_headhands_short.milo and "
                     "../../shared/outfit/cavern02_share.milo). Lets you confirm whether the "
                     "missing/wrong subdirs were the crash without changing anything else",
        default=False,
    )

    tbrb_skeleton_milo: StringProperty(
        name="Skeleton Milo Name",
        description="Name of the companion skeleton milo this mesh milo references as its "
                     "subdirectory. Retail george_headhands_long points at "
                     "'george_skeleton.milo', and that reference is how the mesh milo's bone "
                     "names resolve at runtime. Leave empty to derive it from the Beatle "
                     "dropdown (<beatle>_skeleton.milo)",
        default="",
    )

    tbrb_armature: StringProperty(
        name="Skeleton Armature",
        description="Name of the armature object to build the skeleton from. Leave empty to "
                     "use the active object (or the scene's only armature). Set this when the "
                     "scene holds more than one rig - the exporter refuses to merge armatures, "
                     "because merging de-duplicates by bone name and lets a stale rig silently "
                     "shadow the live one",
        default="",
    )

    tbrb_standard_collisions: BoolProperty(
        name="Retail Head/Neck Collisions",
        description="Emit the four collision volumes the retail TBRB skeleton ships "
                     "(face.coll, forehead.coll, head.coll, neck.coll) as byte-exact copies "
                     "of george's, including their local transforms. Hair and cloth in the "
                     "outfit milo collide against these BY NAME, so a skeleton without them "
                     "is missing something the rest of the character expects. This is "
                     "separate from the per-bone Milo Collision panel because TBRB puts "
                     "THREE volumes on bone_head.mesh, which the one-volume-per-bone UI "
                     "can't express. Turn off to use your own per-bone volumes instead",
        default=True,
    )

    tbrb_repair_degenerate_bones: BoolProperty(
        name="Repair Collapsed Driver Bones",
        description="When exporting a TBRB skeleton, detect non-deforming driver bones "
                     "(spot_*, *-crease, head_lookat) that imported collapsed onto the "
                     "armature origin and restore their rest transform from a reference "
                     "skeleton milo. Some retargeted GLB rigs stack these helper bones on one "
                     "identical wrong transform, which drags the face skin in-game (mouth/cheek "
                     "pull) and breaks the look-at. Only bones whose OWN world position is the "
                     "origin AND that the reference places off-origin are touched - bones you "
                     "moved on purpose are never overridden. Needs a Reference Skeleton Milo set",
        default=True,
    )

    tbrb_reference_skeleton_milo: StringProperty(
        name="Reference Skeleton Milo",
        description="Path to a pristine retail skeleton milo (e.g. the vanilla "
                     "george_skeleton.milo_ps3) used ONLY as the authoritative rest for "
                     "collapsed driver bones when 'Repair Collapsed Driver Bones' is on. The "
                     "mesh geometry, weights, and every bone you positioned still come from your "
                     "armature; this file just supplies correct positions for the origin-"
                     "collapsed helper bones. Leave empty to skip the repair",
        subtype='FILE_PATH',
        default="",
    )

    outfit_config: StringProperty(name="Outfit Config", subtype='FILE_PATH', default="")
    ignore_tex_size_limits: BoolProperty(
        name="Ignore Texture Size Limits",
        description="When off (default), textures are downscaled to fit within 512x512 "
                     "before export, matching the original CLI tool's behavior - this saves "
                     "space and console memory on real hardware. Turn this on to export "
                     "textures at up to 2048x2048 instead of 512x512 (2048 is a hard engine "
                     "limit, not a style choice - textures larger than that crash the game "
                     "regardless of platform, so this will still downscale anything bigger "
                     "than 2048x2048). Large textures near that ceiling can also take a "
                     "noticeably long time to compress during export.",
        default=False,
    )
    only_selected: BoolProperty(
        name="Selected Objects Only",
        description="Export only the currently selected mesh objects. If nothing is "
                     "selected, the whole scene is exported as usual",
        default=False,
    )
    bone_axis_correction: EnumProperty(
        name="Bone Axis Correction (experimental)",
        description="Rotates every bone's local frame by a fixed amount before computing "
                     "its skin offset. This is a debugging aid, not a real setting - if a "
                     "skinned mesh follows its bone correctly but is rotated by a clean "
                     "90/180 degrees, that points to a bone-local axis convention mismatch "
                     "rather than a position bug, and this lets you test corrections quickly "
                     "without re-rigging in Blender first. NOTE: reset to 'None' as the "
                     "default - the old -90 X default was tuned against the global Z-up/"
                     "Y-up conversion this version removes, so it no longer applies as a "
                     "starting point and needs to be re-tested from scratch if needed at all",
        items=(
            ('none', "None", ""),
            ('x90', "+90 X", ""),
            ('xneg90', "-90 X", ""),
            ('y90', "+90 Y", ""),
            ('yneg90', "-90 Y", ""),
            ('z90', "+90 Z", ""),
            ('zneg90', "-90 Z", ""),
        ),
        default='none',
    )
    offset_multiply_order: EnumProperty(
        name="Bone Offset Multiply Order (experimental)",
        description="The decompiled game engine's own SetBone() and glTFMilo's C# "
                     "implementation disagree on the multiply order for computing each "
                     "bone's skin offset matrix. 'Mesh First' (the default) matches the "
                     "decompiled engine; 'Bone First' matches glTFMilo's own CLI tool. "
                     "If skinned meshes deform incorrectly, try switching this alongside "
                     "the Bone Axis Correction above",
        items=(
            ('mesh_first', "Mesh First (engine order)", ""),
            ('bone_first', "Bone First (glTFMilo order)", ""),
        ),
        default='mesh_first',
    )
    export_armature: BoolProperty(
        name="Export Character Armature (Experimental)",
        description="Writes every armature bone as a real top-level Trans entry in the "
                     "milo, instead of relying purely on RB3's external shared-armature "
                     "system. Debugging aid for testing whether custom characters need "
                     "their own embedded skeleton. Only applies to Type=Character",
        default=False,
    )
    stock_rest_offsets: BoolProperty(
        name="Bind Offsets To Stock Rest (Experimental)",
        description="Compute each mesh's bone offset matrices against a pristine "
                     "reference copy of the base RB3 skeleton, instead of the deform "
                     "armature you actually rigged to. RB3 always skins against the "
                     "fixed shared skeleton, so if you moved or rotated bones to fit "
                     "your character's proportions, offsets taken from your edited rig "
                     "make the mesh deform wildly in-game. This keeps the REST pose "
                     "correct no matter how you edited the rig; the tradeoff is mild "
                     "animation drift (bones pivot about their stock positions), like "
                     "the GHWT Definitive Edition mod. Leave OFF if every bone is still "
                     "at its stock rest - it makes no difference then. Type=Character only",
        default=False,
    )
    reference_armature: StringProperty(
        name="Stock Reference Armature",
        description="Name of the pristine, UNEDITED base-skeleton armature to read stock "
                     "bone rest transforms from (for 'Bind Offsets To Stock Rest'). Keep "
                     "it at the same world transform as your deform rig. Leave blank to "
                     "auto-detect: an armature named like a reference "
                     "('stock'/'reference'/'_ref'), or the one armature in the scene that "
                     "deforms no mesh",
        default="",
    )
    hide_head: BoolProperty(
        name="Hide Head? (Experimental)",
        description="Embeds a CharMeshHide asset in the milo with its top-level "
                     "HIDE_HEAD flag set (the same field MiloEditor's own 'Hide Head' "
                     "checkbox edits) and an empty hides list. UNTESTED IN RB3: "
                     "searching the RB3 Wii decompilation (DarkRTA/rb3) turns up no "
                     "code anywhere that reads this flag - only RB4, a different "
                     "engine, is known to use head-hiding accessories. This option "
                     "exists to test on real hardware/Xenia whether RB3 secretly "
                     "honors it after all, or whether it's a vestigial field from "
                     "shared Milo tooling that RB3 never consumes. Only applies to "
                     "Type=Character",
        default=False,
    )
    export_hair: BoolProperty(
        name="Export Hair Physics (.hair) (Experimental)",
        description="Embeds a CharHair (.hair) asset for any armature that has "
                     "'Export CharHair' enabled in its glTFMilo Hair Setup (in the "
                     "armature Data properties). Format is confirmed against MiloLib "
                     "and the RB3 decomp, but whether hair physics actually work from "
                     "a generated .hair alone is untested - collision (.coll) files "
                     "aren't implemented yet, so collisions won't work either way. "
                     "Only applies to Type=Character",
        default=False,
    )
    export_tangents: BoolProperty(
        name="Export Vertex Tangents",
        description="Writes real per-vertex tangents instead of zeros. Normal maps need a "
                     "tangent basis (TBN); with zero tangents the basis is degenerate and "
                     "the normal map collapses back to the geometric normal, so surfaces "
                     "render FLAT - which is why normal maps appeared to do nothing while "
                     "specular maps (which don't need tangents) worked fine. The engine will "
                     "NOT generate tangents itself at RB3's mesh revision (38): the "
                     "decompiled engine only auto-generates them for revision < 30, so they "
                     "must be supplied in the file. Confirmed working in-game on Xbox 360 "
                     "(fabric wrinkles reappear). Leave ON; turn off only to A/B compare "
                     "against the old flat output. Also adds the tangent to the vertex dedup "
                     "key, which can raise the vertex count slightly at UV seams. PS3 NOTE: "
                     "the PS3 tangent format (11-11-10) has no W, so per-vertex handedness "
                     "isn't stored there - detail may invert on mirrored UV islands",
        default=True,
    )
    tbrb_force_mat_defaults: BoolProperty(
        name="Force TBRB Material Defaults (testing)",
        description="TBRB only. The Blender material UI defaults were authored for Rock "
                     "Band 3; several of those values render a TBRB character solid black "
                     "even though the .mat file is structurally correct. When ON, the "
                     "render-affecting material fields (cull, specular power, point lights, "
                     "color adjust, rim lighting, secondary specular) are overridden on "
                     "export with byte-verified known-good TBRB values, so you can confirm "
                     "the material writer itself works without hand-syncing every field in "
                     "the UI. Texture names, base color, blend mode and z-mode still come "
                     "from your material. This is a temporary testing switch - once the UI "
                     "exposes proper TBRB material values it can be turned off. Has no "
                     "effect on non-TBRB exports",
        default=True,
    )
    dc1_generate_mipmaps: BoolProperty(
        name="Generate Mipmaps (DC1)",
        description="Writes a full mip chain (down to a 16px smaller dimension, matching "
                     "every real DC1 texture - confirmed by byte-parsing retail emilia01) "
                     "for every texture instead of a single base level. Dance Central 1 "
                     "requires this for ATI2/BC5 normal "
                     "maps: a vanilla DC1 normal map ships a mip chain, and DC1 renders a "
                     "mip-less ATI2/DXN surface BLACK - the exact 'character turns black when "
                     "a normal map is present' bug. RB3 renders single-level ATI2 normal maps "
                     "fine, and DXT1/DXT5 (diffuse/spec) render mip-less on both, so this is "
                     "DC1-specific. Leave ON for DC1; turn OFF only to A/B confirm that the "
                     "mip chain is what fixes the black rendering. Has no effect for RB3",
        default=True,
    )
    dc3_force_mat_defaults: BoolProperty(
        name="Force DC3 Material Defaults",
        description="Dance Central 3 only. DC3 uses its own rev-70 material format (a "
                     "RndMat/BaseMaterial/MetaMaterial hierarchy) rather than the flat rev-68 "
                     "material RB3 and DC1 share - this is why a custom character with a normal "
                     "map renders solid black on DC3 even though the same export works on DC1. "
                     "The dedicated DC3 material writer fixes the format; this switch additionally "
                     "snaps the two render-affecting fields whose RB3 UI defaults are inverted "
                     "relative to DC3 (Pre-Lit and Use Environment) to the values every retail "
                     "Emilia material uses (prelit OFF, use-environment ON). Texture names, base "
                     "color, blend, z-mode, specular, rim and the MetaMaterial name still come "
                     "from your material. Leave ON unless you are deliberately hand-authoring "
                     "these bits. Has no effect on non-DC3 exports",
        default=True,
    )
    custom_song_texture_format: EnumProperty(
        name="Texture Format",
        description="TBRB Custom Song Asset only. Which loose file format textures are "
                     "written in",
        items=(
            ('tex', "Compiled .tex", "Write the compiled RndTex binary format directly, "
             "same as every other export path - the loose file is a ready-to-load .tex"),
            ('png', "Source .png", "Write a plain, uncompressed .png of the (resized) "
             "image instead of a compiled .tex. Matches the custom-song compiler's "
             "expected input: it takes loose .png source files and does its own PNG -> "
             ".tex conversion when it packages the song, so this skips this exporter's "
             "own DXT/BCn compression and just hands over the source image. Materials "
             "still reference the texture by its eventual '.tex' name, since that's what "
             "the finished, compiled asset the compiler produces will be called"),
        ),
        default='tex',
    )
    custom_song_mat_revision: EnumProperty(
        name="Material Revision",
        description="TBRB Custom Song Asset only. Which RndMat revision loose .mat files "
                     "are written at",
        items=(
            ('rev28', "Revision 28 (proven for custom songs)",
             "Older, simpler material format - diffuse/normal/specular map support only, "
             "no rim lighting/environment map/refraction. Byte-verified against a real "
             "reference material confirmed to load correctly through the custom-song "
             "'extras' folder pipeline. Recommended until revision 55 is confirmed working "
             "through that same pipeline"),
            ('rev55', "Revision 55 (matches vanilla game data)",
             "The same material format every other TBRB export path in this plugin uses, "
             "and byte-verified field-for-field against a real vanilla PS3 instrument "
             "material dumped from inside an actual retail milo - but NOT yet confirmed to "
             "load through the loose custom-song 'extras' folder pipeline itself. Supports "
             "rim lighting, environment map and refraction on top of diffuse/normal/"
             "specular"),
        ),
        default='rev28',
    )
    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        box = layout.box()
        box.label(text="Target", icon='CONSOLE')
        box.prop(self, "platform")
        box.prop(self, "game")
        box.prop(self, "milo_type")
        if self.milo_type == 'instrument':
            box.prop(self, "instrument_type")
        if self.milo_type == 'custom_song_asset':
            if self.game != 'tbrb':
                box.label(text="Custom Song Asset is TBRB-only - set Game to TBRB.",
                          icon='ERROR')
            else:
                sub = box.box()
                sub.label(text="Loose asset export (no milo)", icon='FILE_BLANK')
                sub.label(text="Writes .mesh/.mat/.tex into the chosen folder for the",
                          icon='INFO')
                sub.label(text="custom-song 'extras' + song.dta workflow.")
                sub.label(text="Set per-mesh parent in Object Properties > Milo Object.",
                          icon='INFO')
                sub.prop(self, "custom_song_mat_revision")
                if self.custom_song_mat_revision == 'rev28':
                    sub.label(text="Materials written at rev 28 - diffuse/normal/specular",
                              icon='INFO')
                    sub.label(text="map only, matches a confirmed-working reference file.")
                else:
                    sub.label(text="Materials written at rev 55 - matches vanilla game",
                              icon='INFO')
                    sub.label(text="data, but unconfirmed through this loose-file pipeline.")
                sub.prop(self, "custom_song_texture_format")
                if self.custom_song_texture_format == 'png':
                    sub.label(text="Textures written as loose .png source files - the",
                              icon='INFO')
                    sub.label(text="song compiler converts these to .tex at package time.")
                else:
                    sub.label(text="Textures written as compiled .tex (default, matches",
                              icon='INFO')
                    sub.label(text="every other export path).")
        if self.game in ('dc1', 'dc3'):
            sub = box.box()
            label = "Dance Central 1" if self.game == 'dc1' else "Dance Central 3"
            sub.label(text=f"{label} (experimental)", icon='ARMATURE_DATA')
            sub.prop(self, "dc1_generate_mipmaps")
            if not self.dc1_generate_mipmaps:
                sub.label(text="Mip-less ATI2 normal maps render BLACK in DC1.", icon='ERROR')
            sub.prop(self, "donor_milo")
            if self.game == 'dc3':
                sub.label(text="Container written at rev 32 to match DC3 clips.", icon='INFO')
                sub.prop(self, "dc3_force_mat_defaults")
                sub.label(text="DC3 uses its own rev-70 material format (fixes black",
                          icon='INFO')
                sub.label(text="normal-mapped surfaces). Set each material's MetaMaterial")
                sub.label(text="(.mmat) in Material Properties. Default char_basic_skin;")
                sub.label(text="clothing=char_basic_rim, hair=char_basic_hair,")
                sub.label(text="eyes/teeth=char_basic_skin_nospec.")
                sub.label(text="Avoid char_basic_color - it glows flat (no lighting).",
                          icon='ERROR')
                if not self.export_tangents:
                    sub.label(text="Tangents OFF - normal maps will be flat.", icon='ERROR')
                if self.donor_milo:
                    sub.label(text="Copies the DC3 donor's CharClipSet (win/intro anims).",
                              icon='INFO')
            elif self.donor_milo:
                sub.prop(self, "donor_mode")
                if self.donor_mode == 'inject_meshes':
                    sub.label(text="Output = donor with its meshes swapped for yours.",
                              icon='INFO')
                    sub.label(text="Name Blender objects to match donor mesh names.",
                              icon='INFO')
                else:
                    sub.label(text="Fresh milo + donor's CharClipSet as entry 0.",
                              icon='INFO')

        if self.game == 'tbrb' and self.milo_type != 'custom_song_asset':
            sub = box.box()
            sub.label(text="The Beatles: Rock Band (PS3 first)", icon='ARMATURE_DATA')
            sub.prop(self, "tbrb_target")
            if self.tbrb_target == 'skeleton':
                sub.prop(self, "tbrb_beatle")
                sub.prop(self, "tbrb_armature")
                sub.prop(self, "tbrb_standard_collisions")
                sub.prop(self, "tbrb_repair_degenerate_bones")
                if self.tbrb_repair_degenerate_bones:
                    sub.prop(self, "tbrb_reference_skeleton_milo")
                    if not self.tbrb_reference_skeleton_milo:
                        sub.label(text="Set a vanilla skeleton milo to enable the repair.",
                                  icon='ERROR')
                    else:
                        sub.label(text="Collapsed driver bones (spot_*/crease/lookat) restored "
                                       "from it.", icon='INFO')
                sub.label(text="Every bone in that armature is exported, weighted or not.",
                          icon='INFO')
                sub.label(text="Bones come from the export armature; missing retail bones "
                               "are reported.", icon='INFO')
                if self.platform != 'ps3':
                    sub.label(text="Start with PS3 - TBRB isn't testable on Xenia/360 yet.",
                              icon='INFO')
            else:
                sub.prop(self, "tbrb_mesh_kind")
                if self.tbrb_mesh_kind == 'outfit':
                    sub.prop(self, "tbrb_base_milo")
                    sub.prop(self, "tbrb_share_milo")
                else:
                    sub.prop(self, "tbrb_beatle")
                    sub.prop(self, "tbrb_skeleton_milo")
                sub.prop(self, "tbrb_test_hardcode_subdirs")
                sub.label(text="Flat milo: meshes + materials + textures, no LOD groups.",
                          icon='INFO')
                sub.label(text="sphereBase is set to bone_pelvis.mesh (matches retail).",
                          icon='INFO')
                if not self.export_tangents:
                    sub.label(text="Tangents OFF - normal maps will be flat.", icon='ERROR')

        box = layout.box()
        box.label(text="Scene", icon='SCENE_DATA')
        box.prop(self, "only_selected")

        box = layout.box()
        box.label(text="Materials & Textures", icon='MATERIAL')
        box.prop(self, "ignore_tex_size_limits")
        box.prop(self, "export_tangents")
        if self.export_tangents and self.platform == 'ps3':
            box.label(text="PS3: no per-vertex handedness (mirrored UVs may invert)",
                      icon='INFO')
        if self.game == 'tbrb':
            box.prop(self, "tbrb_force_mat_defaults")
            if self.tbrb_force_mat_defaults:
                box.label(text="TBRB render values forced; UI material fields partly ignored.",
                          icon='INFO')

        box = layout.box()
        box.label(text="Character", icon='ARMATURE_DATA')
        box.prop(self, "outfit_config")
        box.prop(self, "bone_axis_correction")
        row = box.row()
        row.enabled = (self.milo_type in ('character', 'instrument'))
        row.prop(self, "export_armature")
        if self.game in ('dc1', 'dc3'):
            row.enabled = False
            box.label(text="Dance Central always exports the full armature (no shared "
                           "skeleton).", icon='INFO')
        row = box.row()
        row.enabled = (self.milo_type == 'character')
        row.prop(self, "hide_head")
        row = box.row()
        row.enabled = (self.milo_type in ('character', 'instrument'))
        row.prop(self, "export_hair")
        row = box.row()
        row.enabled = (self.milo_type == 'character')
        row.prop(self, "stock_rest_offsets")
        row = box.row()
        row.enabled = (self.milo_type == 'character' and self.stock_rest_offsets)
        row.prop(self, "reference_armature")

    def _collect_armatures(self, context):
        """Return {name: armature_object} for every armature in the export scope. Unlike
        the mesh path (which infers armatures from mesh Armature modifiers), a skeleton-only
        export may have no meshes at all, so armature objects are gathered directly."""
        if self.only_selected and context.selected_objects:
            objs = context.selected_objects
        else:
            objs = context.scene.objects
        return {o.name: o for o in objs if o.type == 'ARMATURE'}

    def _export_tbrb_meshes(self, context):
        """Export a TBRB MESH milo: meshes + materials + textures, flat (no LOD groups).
        The companion skeleton milo must be exported separately and installed alongside -
        this milo's only subdir reference points at it, and that's how bone names resolve."""
        import os

        base, ext = os.path.splitext(self.filepath)
        if self.platform == 'ps3':
            ext = '.milo_ps3'
        elif ext.lower() != '.milo_xbox':
            ext = '.milo_xbox'
        self.filepath = base + ext
        root_name = os.path.splitext(os.path.basename(self.filepath))[0]

        skeleton_milo = (self.tbrb_skeleton_milo or "").strip()
        if not skeleton_milo:
            skeleton_milo = f"{self.tbrb_beatle}_skeleton.milo" \
                if self.tbrb_beatle != 'custom' else "george_skeleton.milo"

        # Resolve the subdirectory list by milo kind. An OUTFIT milo references the base
        # head/hands milo + a shared outfit milo (and inherits the skeleton transitively);
        # a HEAD/HANDS milo references the skeleton milo directly. sphereBase is a real bone
        # in every retail mesh milo (bone_pelvis.mesh), never the project name.
        sphere_base = "bone_pelvis.mesh"
        if self.tbrb_test_hardcode_subdirs:
            # Isolation test: emit the EXACT subdirs from the retail cavern02 outfit milo,
            # so "are the subdirs the crash?" can be answered without any other variable.
            sub_dirs = ["../george_headhands_short.milo",
                        "../../shared/outfit/cavern02_share.milo"]
            _log("  TEST MODE: hardcoding retail cavern02 subdirs "
                 "(../george_headhands_short.milo, ../../shared/outfit/cavern02_share.milo).")
        elif self.tbrb_mesh_kind == 'outfit':
            base_milo = (self.tbrb_base_milo or "").strip() or "../george_headhands_long.milo"
            share_milo = (self.tbrb_share_milo or "").strip()
            sub_dirs = [base_milo] + ([share_milo] if share_milo else [])
            _log(f"  Outfit subdirs: {sub_dirs}")
        else:
            sub_dirs = [skeleton_milo]
            _log(f"  Head/hands subdir: {sub_dirs}")

        _log(f"===== Starting TBRB MESH export: '{root_name}' -> {self.filepath} "
             f"(platform={self.platform}, kind={self.tbrb_mesh_kind}) =====")
        _log(f"  sphereBase='{sphere_base}'")
        if self.platform != 'ps3':
            _log("  NOTE: PS3 is the verified TBRB target so far.")

        _log("--- Meshes ---")
        try:
            entries, materials_by_name, armatures_used = collect_mesh_entries(
                context, root_name, only_selected=self.only_selected,
                bone_axis_correction=self.bone_axis_correction,
                offset_multiply_order=self.offset_multiply_order,
                stock_reference_armature=None,
                platform=self.platform,
                include_tangents=self.export_tangents)
        except BoneLimitExceeded as e:
            details = "; ".join(f"'{n}': {c} bones (max {MAX_BONES_PER_MESH})"
                                 for n, c in e.offenders)
            _log(f"EXPORT FAILED: bone limit exceeded - {details}")
            self.report({'ERROR'},
                        f"Export blocked - exceeds the {MAX_BONES_PER_MESH}-bone-per-mesh "
                        f"limit (this crashes the game): {details}.")
            return {'CANCELLED'}
        except ValueError as e:
            _log(f"EXPORT FAILED: {e}")
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        if not entries:
            _log("EXPORT FAILED: no mesh objects found.")
            self.report({'ERROR'}, "No mesh objects found to export.")
            return {'CANCELLED'}

        if not self.export_tangents:
            _log("  WARNING: tangents are DISABLED. At mesh revision 33 tangents are plain "
                 "floats and are written correctly, so there's no reason to turn them off "
                 "here - normal maps will render FLAT without them.")

        _log("--- Materials & Textures ---")
        # mip_floor=16: every retail TBRB texture bottoms out at a 16px smaller dimension,
        # never 4. generate_mips is forced on - retail ships full chains for every texture.
        materials, textures, _texture_images = gather_materials_and_textures(
            materials_by_name, max_tex_size=512,
            ignore_tex_size_limits=self.ignore_tex_size_limits,
            platform=self.platform, generate_mips=True,
            mip_floor=TBRB_TEX_MIP_FLOOR)
        for (tname, tw, th, enc, bpp, _blk, nmips) in textures:
            _log(f"  Tex '{tname}': {tw}x{th} bpp={bpp} enc={enc} mipMaps={nmips}")

        _log("--- Writing mesh container ---")
        data = build_tbrb_mesh_milo_bytes(
            root_name, entries, materials, textures,
            skeleton_milo_name=skeleton_milo,
            platform=self.platform, write_tangents=self.export_tangents,
            sub_dirs=sub_dirs, sphere_base=sphere_base,
            force_tbrb_mat_defaults=self.tbrb_force_mat_defaults)

        with open(self.filepath, 'wb') as f:
            f.write(data)

        total_verts = sum(len(e[4]) for e in entries)
        total_faces = sum(len(e[5]) for e in entries)
        summary = (f"Exported TBRB mesh milo '{root_name}': {len(entries)} mesh(es), "
                   f"{len(materials)} material(s), {len(textures)} texture(s), "
                   f"{total_verts} verts, {total_faces} tris -> {self.filepath} "
                   f"({len(data)} bytes)")
        _log(f"===== SUCCESS: {summary} =====")
        self.report({'INFO'}, summary)
        return {'FINISHED'}

    def _export_tbrb_custom_song_assets(self, context):
        """Export loose standalone .mesh / .mat / .tex files (no milo container).

        The TBRB custom-song community loads models by dropping loose milo assets into an
        'extras' folder that the packaging tool mounts into $world, then wiring them in with
        song.dta scripting (set_trans_parent onto a bone, add_object into a draw group,
        set diffuse_tex to rebind textures). This export produces exactly those loose files:
        one .mesh per Blender mesh, one .mat per material, one .tex per texture, each a
        complete standalone asset (revision + body + 0xADDEADDE), written into the folder the
        user picked in the export dialog. No directory container, no LOD groups, no subdir
        references - the packaging tool and the DTA do the assembly.

        Assets are written at the TBRB revisions (mesh 34, mat 55, tex 10) so they match what
        a TBRB milo carries and load the same way when mounted. The per-object 'Milo Object >
        Trans Parent' setting is honoured: if set, that mesh's RndTrans parentObj is baked to
        the named in-game bone (e.g. bone_piano_base.mesh), so the asset rides that bone the
        instant it loads without the DTA needing a set_trans_parent call."""
        import os

        out_dir = os.path.dirname(self.filepath)
        if not out_dir:
            out_dir = os.getcwd()

        _log(f"===== Starting TBRB CUSTOM SONG ASSET export (loose files) -> {out_dir} "
             f"(platform={self.platform}) =====")
        if self.platform != 'ps3':
            _log("  NOTE: PS3 is the verified TBRB target so far.")

        _log("--- Meshes ---")
        # root_name is only used as the default Trans parent for meshes with no per-object
        # override. For a loose asset there is no milo root dir to parent to, so default to
        # empty - a mesh with no explicit parent stays unparented until the DTA wires it in,
        # exactly like the community's existing set_trans_parent workflow. A per-object Trans
        # Parent override (e.g. bone_piano_base.mesh) replaces this.
        loose_root = ""
        try:
            entries, materials_by_name, armatures_used = collect_mesh_entries(
                context, loose_root, only_selected=self.only_selected,
                bone_axis_correction=self.bone_axis_correction,
                offset_multiply_order=self.offset_multiply_order,
                stock_reference_armature=None,
                platform=self.platform,
                include_tangents=self.export_tangents)
        except BoneLimitExceeded as e:
            details = "; ".join(f"'{n}': {c} bones (max {MAX_BONES_PER_MESH})"
                                 for n, c in e.offenders)
            _log(f"EXPORT FAILED: bone limit exceeded - {details}")
            self.report({'ERROR'},
                        f"Export blocked - exceeds the {MAX_BONES_PER_MESH}-bone-per-mesh "
                        f"limit (this crashes the game): {details}.")
            return {'CANCELLED'}
        except ValueError as e:
            _log(f"EXPORT FAILED: {e}")
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        if not entries:
            _log("EXPORT FAILED: no mesh objects found.")
            self.report({'ERROR'}, "No mesh objects found to export.")
            return {'CANCELLED'}

        if not self.export_tangents:
            _log("  WARNING: tangents are DISABLED - normal maps will render FLAT.")

        _log("--- Materials & Textures ---")
        materials, textures, texture_images = gather_materials_and_textures(
            materials_by_name, max_tex_size=512,
            ignore_tex_size_limits=self.ignore_tex_size_limits,
            platform=self.platform, generate_mips=True,
            mip_floor=TBRB_TEX_MIP_FLOOR)

        # --- write each asset as its own loose file ---
        written = []

        def _safe_filename(name):
            # Entry names already carry the extension (.mesh/.mat/.tex) and are sanitized to
            # milo-symbol-safe characters, which are also filesystem-safe. Guard anyway.
            return name.replace("/", "_").replace("\\", "_")

        _log("--- Writing loose assets ---")
        # Textures first (materials reference them by name; order on disk doesn't matter to
        # the game, but writing them first keeps the log readable).
        write_png = (self.custom_song_texture_format == 'png')
        for (tname, tw, th, enc, bpp, blk, nmips) in textures:
            if write_png:
                # Source .png: hand the compiled-name entry a .png extension instead, and
                # write real image bytes via Blender's own PNG writer rather than this
                # exporter's compiled RndTex/DXT format. The .mat still references
                # '<name>.tex' (unchanged) - that's the name the song compiler's own
                # PNG -> .tex conversion is expected to produce.
                png_name = tname[:-4] + ".png" if tname.lower().endswith(".tex") else tname + ".png"
                path = os.path.join(out_dir, _safe_filename(png_name))
                image = texture_images.get(tname)
                if image is None:
                    _log(f"  Tex  '{tname}': FAILED - no source image found for PNG export, "
                         f"skipping.")
                    self.report({'WARNING'}, f"Could not export '{tname}' as PNG - "
                                              f"source image not found.")
                    continue
                pw, ph = export_texture_as_png(image, path, max_size=512,
                                                ignore_size_limits=self.ignore_tex_size_limits)
                written.append(path)
                _log(f"  Tex  '{tname}': {pw}x{ph} -> '{os.path.basename(path)}' (png, "
                     f"{os.path.getsize(path)} bytes)")
            else:
                w = MiloWriter(big_endian=True)
                write_tbrb_rnd_tex(w, tw, th, enc, bpp, blk,
                                    external_path="", platform=self.platform, num_mips=nmips)
                path = os.path.join(out_dir, _safe_filename(tname))
                with open(path, 'wb') as f:
                    f.write(w.buf)
                written.append(path)
                _log(f"  Tex  '{tname}': {tw}x{th} bpp={bpp} enc={enc} mipMaps={nmips} "
                     f"-> {len(w.buf)} bytes")

        use_rev28_mat = (self.custom_song_mat_revision == 'rev28')
        for (mat_entry_name, settings) in materials:
            w = MiloWriter(big_endian=True)
            if use_rev28_mat:
                write_tbrb_custom_song_rnd_mat(w, settings, standalone=True)
            else:
                write_tbrb_rnd_mat(w, settings, standalone=True,
                                    force_tbrb_defaults=self.tbrb_force_mat_defaults)
            path = os.path.join(out_dir, _safe_filename(mat_entry_name))
            with open(path, 'wb') as f:
                f.write(w.buf)
            written.append(path)
            rev_note = "rev 28" if use_rev28_mat else "rev 55"
            _log(f"  Mat  '{mat_entry_name}' ({rev_note}): -> {len(w.buf)} bytes")

        for (entry_name, local_xfm, world_xfm, parent_obj, vertices, faces,
             bone_transforms, mat_name, needs_ao_calc) in entries:
            w = MiloWriter(big_endian=True)
            write_tbrb_rnd_mesh(w, entry_name, local_xfm, world_xfm, parent_obj,
                                 vertices, faces, mat_name=mat_name,
                                 bone_transforms=bone_transforms,
                                 force_white_vertex_color=needs_ao_calc,
                                 write_tangents=self.export_tangents)
            path = os.path.join(out_dir, _safe_filename(entry_name))
            with open(path, 'wb') as f:
                f.write(w.buf)
            written.append(path)
            parent_note = f", parent '{parent_obj}'" if parent_obj else ", no parent (wire in via DTA)"
            _log(f"  Mesh '{entry_name}': {len(vertices)} verts, {len(faces)} tris"
                 f"{parent_note} -> {len(w.buf)} bytes")

        summary = (f"Exported {len(entries)} loose .mesh, {len(materials)} .mat, "
                   f"{len(textures)} .tex file(s) -> {out_dir}")
        _log(f"===== SUCCESS: {summary} =====")
        self.report({'INFO'}, summary)
        return {'FINISHED'}

    def _resolve_tbrb_armature(self, context):
        """Resolve EXACTLY ONE armature to build the skeleton from, and say which.

        A skeleton milo is one skeleton. Silently merging every armature in the scene is how
        bones go missing: the merge de-duplicates by name, so a stale `Armature_old` sitting
        in the file can shadow the live rig's bones. Preference order is explicit name ->
        active object -> selected -> the scene's only armature. Anything ambiguous is an
        error the user resolves, not a guess.

        Returns (armature_object, error_message)."""
        scene_arms = [o for o in context.scene.objects if o.type == 'ARMATURE']
        if not scene_arms:
            return None, ("No armature found in the scene. A TBRB skeleton milo is built "
                          "from an armature's bones.")

        wanted = (self.tbrb_armature or "").strip()
        if wanted:
            for o in scene_arms:
                if o.name == wanted:
                    return o, None
            return None, (f"Armature '{wanted}' not found. Available: "
                          f"{', '.join(o.name for o in scene_arms)}")

        active = getattr(context.view_layer.objects, "active", None)
        if active is not None and active.type == 'ARMATURE':
            return active, None

        selected = [o for o in getattr(context, "selected_objects", []) or []
                    if o.type == 'ARMATURE']
        if len(selected) == 1:
            return selected[0], None
        if len(selected) > 1:
            return None, (f"{len(selected)} armatures selected. Select just one, make it "
                          f"active, or name it in 'Skeleton Armature'.")

        if len(scene_arms) == 1:
            return scene_arms[0], None
        return None, (f"The scene has {len(scene_arms)} armatures "
                      f"({', '.join(o.name for o in scene_arms)}). Make the one you want "
                      f"active, or name it in 'Skeleton Armature' - merging them would "
                      f"silently drop bones.")

    def _export_tbrb(self, context):
        """The Beatles: Rock Band export dispatch. Currently only the SKELETON milo
        (bones + collision) is implemented; the mesh milo is the next phase."""
        import os

        if self.platform not in ('xbox360', 'ps3'):
            self.report({'ERROR'}, "TBRB export supports Platform = PS3 (preferred) or "
                                    "Xbox 360.")
            return {'CANCELLED'}

        # Custom Song Asset export produces loose .mesh/.mat/.tex files regardless of the
        # TBRB Milo (skeleton/meshes) selector - it never builds a container.
        if self.milo_type == 'custom_song_asset':
            return self._export_tbrb_custom_song_assets(context)

        if self.tbrb_target == 'meshes':
            return self._export_tbrb_meshes(context)

        # Normalize output extension to the platform, then apply the Beatle filename.
        base, ext = os.path.splitext(self.filepath)
        if self.platform == 'ps3':
            ext = '.milo_ps3'
        elif ext.lower() not in ('.milo_xbox',):
            ext = '.milo_xbox'
        if self.tbrb_beatle != 'custom':
            base = os.path.join(os.path.dirname(base), f"{self.tbrb_beatle}_skeleton")
        self.filepath = base + ext
        root_name = os.path.splitext(os.path.basename(self.filepath))[0]

        _log(f"===== Starting TBRB SKELETON export: '{root_name}' -> {self.filepath} "
             f"(platform={self.platform}, beatle={self.tbrb_beatle}) =====")
        if self.platform != 'ps3':
            _log("  NOTE: exporting for Xbox 360, but TBRB isn't testable on Xenia/360 yet - "
                 "PS3 is the verified target.")

        armature_obj, err = self._resolve_tbrb_armature(context)
        if armature_obj is None:
            _log(f"EXPORT FAILED: {err}")
            self.report({'ERROR'}, err)
            return {'CANCELLED'}
        armatures_used = {armature_obj.name: armature_obj}
        _log(f"Skeleton armature: '{armature_obj.name}'")

        _log("--- Skeleton bones (Trans) ---")
        # EVERY bone, unconditionally. A skeleton milo IS the skeleton: unweighted driver and
        # facial bones (creases, footik, head_lookat, eyes) are resolved by name at runtime and
        # a missing one crashes on song start.
        try:
            bone_trans_entries = build_all_bone_trans_entries(
                armature_obj, root_name, label="TBRB skeleton")
        except ValueError as e:
            _log(f"EXPORT FAILED: {e}")
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        # Sort alphabetically to match how the retail skeleton is laid out (order is
        # name-resolved at runtime, so this is convention/tidiness, not correctness).
        bone_trans_entries.sort(key=lambda e: e[0])
        if not bone_trans_entries:
            _log("EXPORT FAILED: armature has no bones to export.")
            self.report({'ERROR'}, "The armature has no bones to export.")
            return {'CANCELLED'}

        # --- Bone completeness check against the retail skeleton -------------------
        # A skeleton milo missing bones LOADS FINE and then crashes once a song starts,
        # because the animation/look-at systems resolve driver bones by name and the retail
        # meshes' inverse-bind matrices are baked against the full bone set. Warn loudly.
        exported_bone_names = {b[0] for b in bone_trans_entries}
        missing_bones = sorted(TBRB_REFERENCE_SKELETON_BONES - exported_bone_names)
        if missing_bones:
            _log(f"  WARNING: {len(missing_bones)} bone(s) present in the retail TBRB "
                 f"skeleton are MISSING from this armature:")
            for n in missing_bones:
                _log(f"    - {n}")
            _log("  Missing driver bones (footik / head_lookat / eyes) are a known cause of "
                 "a crash on song start. Re-import the armature so it carries every bone.")
            self.report({'WARNING'},
                        f"{len(missing_bones)} retail bone(s) missing from the armature "
                        f"(e.g. {', '.join(missing_bones[:3])}) - this commonly crashes on "
                        f"song start. See the system console for the full list.")
        else:
            _log("  Bone check: armature contains every bone the retail TBRB skeleton has.")
        extra_bones = sorted(exported_bone_names - TBRB_REFERENCE_SKELETON_BONES)
        # A bone whose name is one ".mesh" away from a retail bone is almost always a typo,
        # not a custom bone. It's a silent double failure: the vertex group (named with the
        # suffix) never binds to the bone, AND the exported Trans has a name nothing in the
        # game looks up - so the bone is effectively still missing.
        suffix_typos = [n for n in extra_bones
                        if f"{n}.mesh" in TBRB_REFERENCE_SKELETON_BONES]
        if suffix_typos:
            _log(f"  WARNING: {len(suffix_typos)} bone(s) look like they are missing the "
                 f"'.mesh' suffix:")
            for n in suffix_typos:
                _log(f"    - '{n}'  should probably be  '{n}.mesh'")
            _log("  Rename these in Blender. As-is the vertex group (which DOES carry the "
                 "suffix) won't bind to the bone, and the game won't find the bone either.")
            self.report({'WARNING'},
                        f"{len(suffix_typos)} bone(s) appear to be missing the '.mesh' "
                        f"suffix (e.g. '{suffix_typos[0]}' -> '{suffix_typos[0]}.mesh'). "
                        f"See the system console.")
        other_extra = [n for n in extra_bones if n not in suffix_typos]
        if other_extra:
            _log(f"  NOTE: {len(other_extra)} bone(s) are not in the retail skeleton "
                 f"(custom bones are fine, but typos here silently do nothing): "
                 f"{', '.join(other_extra)}")

        # --- Repair origin-collapsed driver bones from a reference skeleton ---------
        if self.tbrb_repair_degenerate_bones and self.tbrb_reference_skeleton_milo:
            ref_path = bpy.path.abspath(self.tbrb_reference_skeleton_milo)
            _log(f"--- Collapsed-bone repair (reference: {ref_path}) ---")
            reference_trans = _read_skeleton_milo_trans(ref_path)
            if not reference_trans:
                _log("  WARNING: could not parse the reference skeleton milo (wrong file, "
                     "compressed, or non-PS3). No repair applied.")
                self.report({'WARNING'},
                            "Reference skeleton milo could not be parsed - collapsed-bone "
                            "repair was skipped. See the system console.")
            else:
                before = {b[0]: _trans_world_translation(b[1:]) for b in bone_trans_entries}
                bone_trans_entries, repaired = repair_degenerate_skeleton_bones(
                    bone_trans_entries, reference_trans, root_name)
                if repaired:
                    _log(f"  Repaired {len(repaired)} origin-collapsed driver bone(s) from the "
                         f"reference skeleton:")
                    after = {b[0]: _trans_world_translation(b[1:]) for b in bone_trans_entries}
                    for n in repaired:
                        bx = before.get(n, (0, 0, 0)); ax = after.get(n, (0, 0, 0))
                        _log(f"    {n:28s} world ({bx[0]:+.3f},{bx[1]:+.3f},{bx[2]:+.3f}) "
                             f"-> ({ax[0]:+.3f},{ax[1]:+.3f},{ax[2]:+.3f})")
                    self.report({'INFO'},
                                f"Repaired {len(repaired)} collapsed driver bone(s) from the "
                                f"reference skeleton.")
                else:
                    _log("  No origin-collapsed bones found - nothing to repair (rig is clean "
                         "or bones already positioned).")
        elif self.tbrb_repair_degenerate_bones:
            _log("--- Collapsed-bone repair: ENABLED but no Reference Skeleton Milo set - "
                 "skipped. Point it at the vanilla george_skeleton.milo_ps3 to enable. ---")

        # --- Collision volumes ------------------------------------------------------
        if self.tbrb_standard_collisions:
            char_collide_entries = build_tbrb_standard_collision_entries(exported_bone_names)
            _log(f"--- Collision volumes (CharCollide) ---\n"
                 f"Emitting the {len(char_collide_entries)} retail TBRB head/neck volume(s) "
                 f"(byte-exact copies of george's, including local transforms):")
        else:
            char_collide_entries = build_char_collide_entries(armatures_used, root_name)
            _log("--- Collision volumes (CharCollide) ---\n"
                 "Using per-bone authored volumes (retail standard volumes are OFF).")
        char_collide_entries.sort(key=lambda e: e[0])
        if char_collide_entries:
            for ename, coll in char_collide_entries:
                _log(f"  {ename}: shape={coll['shape']} r0={coll['radius0']} "
                     f"-> parent '{coll['parent']}'")
        else:
            _log("  NONE. The retail george skeleton ships 4 (face/forehead/head/neck); "
                 "hair and cloth in the outfit milo collide against these by name.")

        _log("--- Writing skeleton container ---")
        data = build_tbrb_skeleton_milo_bytes(root_name, bone_trans_entries,
                                               char_collide_entries)
        with open(self.filepath, 'wb') as f:
            f.write(data)

        summary = (f"Exported TBRB skeleton '{root_name}': {len(bone_trans_entries)} bone(s), "
                   f"{len(char_collide_entries)} collision volume(s) -> {self.filepath} "
                   f"({len(data)} bytes)")
        _log(f"===== SUCCESS: {summary} =====")
        self.report({'INFO'}, summary)
        return {'FINISHED'}

    def execute(self, context):
        if self.game == 'tbrb':
            return self._export_tbrb(context)
        if self.game not in ('rb3', 'dc1', 'dc3'):
            self.report(
                {'ERROR'},
                "Only Game=Rock Band 3, Dance Central 1 and Dance Central 3 are "
                "implemented so far."
            )
            return {'CANCELLED'}
        if self.platform not in ('xbox360', 'ps3'):
            self.report(
                {'ERROR'},
                "Only Platform=Xbox 360 and PS3 are implemented so far."
            )
            return {'CANCELLED'}
        if self.game in ('dc1', 'dc3') and self.platform != 'xbox360':
            self.report(
                {'ERROR'},
                "Dance Central was an Xbox 360 exclusive - set Platform to Xbox 360."
            )
            return {'CANCELLED'}
        if self.milo_type == 'custom_song_asset':
            self.report(
                {'ERROR'},
                "Type=Custom Song Asset is for The Beatles: Rock Band only - set Game to "
                "The Beatles: Rock Band."
            )
            return {'CANCELLED'}
        if self.milo_type not in ('other', 'character', 'instrument'):
            self.report(
                {'ERROR'},
                "Type=Venue isn't implemented yet - only Other, Character, and Instrument."
            )
            return {'CANCELLED'}

        import os
        # Normalize the output extension to match the target platform. ExportHelper's
        # fixed filename_ext is .milo_xbox; swap it to .milo_ps3 for PS3 so the file
        # name reflects what it actually is (and so MiloLib/tools pick the right platform).
        base, ext = os.path.splitext(self.filepath)
        if self.platform == 'ps3' and ext.lower() in ('.milo_xbox', ''):
            self.filepath = base + '.milo_ps3'
        elif self.platform == 'xbox360' and ext.lower() == '.milo_ps3':
            self.filepath = base + '.milo_xbox'

        root_name = os.path.splitext(os.path.basename(self.filepath))[0]

        _log(f"===== Starting Milo export: '{root_name}' -> {self.filepath} "
             f"(type={self.milo_type}, platform={self.platform}, game={self.game}) =====")

        stock_ref_arm = None
        if self.stock_rest_offsets:
            stock_ref_arm = _resolve_stock_reference_armature(
                context, self.reference_armature, self.only_selected)
            if stock_ref_arm is not None:
                _log(f"Stock-rest offsets ON - reading bind rest from reference armature "
                     f"'{stock_ref_arm.name}'")
            else:
                _log("Stock-rest offsets ON but no reference armature could be resolved - "
                     "falling back to the deform rig's own rest (no change). Name one in "
                     "'Stock Reference Armature', or add a pristine base-skeleton copy.")
                self.report({'WARNING'},
                            "Stock-rest offsets: no reference armature found; used the "
                            "deform rig's rest instead (see the export log).")

        if self.export_tangents:
            if self.platform == 'ps3':
                _log("Vertex tangents: ENABLED - real per-vertex tangents will be written. "
                     "PS3's 11-11-10 tangent has no W, so per-vertex handedness isn't stored; "
                     "normal-map detail may invert on mirrored UV islands.")
            else:
                _log("Vertex tangents: ENABLED - real per-vertex tangents (with raw Blender "
                     "handedness) will be written; tangent + handedness included in the vertex "
                     "dedup key.")
        else:
            _log("Vertex tangents: DISABLED - writing zero tangents. Normal maps will render "
                 "FLAT (degenerate TBN basis). Only use this to A/B compare.")

        _log("--- Meshes ---")
        try:
            entries, materials_by_name, armatures_used = collect_mesh_entries(
                context, root_name, only_selected=self.only_selected,
                bone_axis_correction=self.bone_axis_correction,
                offset_multiply_order=self.offset_multiply_order,
                stock_reference_armature=stock_ref_arm,
                platform=self.platform,
                include_tangents=self.export_tangents)
        except BoneLimitExceeded as e:
            details = "; ".join(f"'{name}': {count} bones (max {MAX_BONES_PER_MESH})"
                                 for name, count in e.offenders)
            _log(f"EXPORT FAILED: bone limit exceeded - {details}")
            self.report(
                {'ERROR'},
                f"Export blocked - exceeds the {MAX_BONES_PER_MESH}-bone-per-mesh "
                f"limit (this crashes the game): {details}. Split the offending "
                f"mesh(es) into separate objects so each stays under the limit."
            )
            return {'CANCELLED'}
        except ValueError as e:
            _log(f"EXPORT FAILED: {e}")
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        if not entries:
            _log("EXPORT FAILED: no mesh objects found in the scene to export.")
            self.report({'WARNING'}, "No mesh objects found in the scene to export.")
            return {'CANCELLED'}

        bone_trans_entries = []
        emitted_bone_names = set()
        # DC1 has NO shared skeleton milo - the character must carry its own complete bone
        # hierarchy - so armature export is forced on and nothing is filtered out. This
        # matters a lot: RB3_SKELETON_BONES contains 490 RB3 bone names, and DC1 uses the
        # same Harmonix naming convention, so the RB3 filter would silently drop ~100 of a
        # DC1 character's ~136 bones and ship a nearly-empty skeleton.
        force_full_armature = (self.game in ('dc1', 'dc3'))
        want_armature = (self.export_armature or force_full_armature)
        if want_armature and self.milo_type in ('character', 'instrument') and armatures_used:
            _log("--- Armature ---")
            if force_full_armature and not self.export_armature:
                _log("  DC1: forcing armature export on (DC1 characters must embed their "
                     "own skeleton).")
            # IMPORTANT: RB3_SKELETON_BONES is the COMBINED RB3 skeleton and it INCLUDES
            # the guitar rig bones (bone_target_strum.mesh, spot_neck_fret01.mesh, etc.).
            # For an RB3 character we skip those names because char_shared.milo already
            # provides them - but an instrument does NOT get char_shared, and neither does
            # a DC1 character, so for those we must NOT skip or the rig comes out empty.
            skip_shared = (self.milo_type == 'character' and not force_full_armature)
            bone_trans_entries = build_armature_trans_entries(
                armatures_used, root_name, skip_skeleton_bones=skip_shared,
                label=("DC1 armature (all bones)" if force_full_armature else
                       "instrument armature" if self.milo_type == 'instrument' else "armature"))
            emitted_bone_names = {b[0] for b in bone_trans_entries}
            if force_full_armature:
                total_bones = sum(len(a.data.bones) for a in armatures_used.values())
                _log(f"  DC1: emitted {len(bone_trans_entries)} of {total_bones} armature "
                     f"bone(s) as Trans entries (no skeleton filtering applied).")
                if len(bone_trans_entries) != total_bones:
                    _log(f"  WARNING: {total_bones - len(bone_trans_entries)} bone(s) were "
                         f"not emitted - duplicates across armatures are only written once.")

        char_mesh_hide_entries = []
        if self.hide_head and self.milo_type == 'character':
            hide_entry_name = f"{sanitize_milo_name(root_name)}.hide"
            _log(f"--- CharMeshHide ---\nExporting '{hide_entry_name}' "
                 f"(HIDE_HEAD flag, untested in RB3, hides list empty)...")
            char_mesh_hide_entries = [
                (hide_entry_name, CharMeshHideOptions.HIDE_HEAD, [])
            ]
            _log(f"  Exported '{hide_entry_name}'.")

        char_hair_entries = []
        if self.export_hair and self.milo_type in ('character', 'instrument') and armatures_used:
            _log("--- CharHair (.hair) ---")
            char_hair_entries = build_char_hair_entries(armatures_used, root_name)
            for (hair_entry_name, hair_settings, strands) in char_hair_entries:
                total_points = sum(len(s['points']) for s in strands)
                _log(f"Created CharHair '{hair_entry_name}': {len(strands)} strand(s), "
                     f"{total_points} point(s) total, simulate={hair_settings['simulate']}, "
                     f"wind={hair_settings['wind']}.")
                for s in strands:
                    _log(f"  Strand root '{s['root']}': "
                         f"{' -> '.join(p['bone'] for p in s['points'])}")
            if not char_hair_entries:
                _log("  No CharHair exported (no armature had 'Export CharHair' enabled "
                     "with at least one strand).")
            else:
                # A .hair file drives bones by name, so those bones MUST exist as real
                # Trans entries in the milo or the hair has nothing to move. Emit Trans
                # entries for exactly the hair physics bones (skipping any bone already
                # emitted above by Export Armature). This is what the glTFMilo CLI does
                # automatically when it imports the armature.
                hair_bones = _hair_bone_names(armatures_used)
                hair_bones_to_emit = hair_bones - emitted_bone_names
                if hair_bones_to_emit:
                    # For RB3 characters, base-skeleton hair bones come from
                    # char_shared.milo, so they're skipped (skip_skeleton_bones=True).
                    # Instruments get colorpalettes.milo and DC1 characters get no shared
                    # skeleton at all, so both must emit every hair bone verbatim - even
                    # ones whose names appear in the combined RB3_SKELETON_BONES list.
                    skip_shared = (self.milo_type == 'character' and not force_full_armature)
                    _log(f"--- Hair physics bones (Trans) ---\n"
                         f"Emitting up to {len(hair_bones_to_emit)} hair bone(s) as Trans "
                         f"entries so the .hair file has bones to drive.")
                    hair_bone_entries = build_armature_trans_entries(
                        armatures_used, root_name, only_bones=hair_bones_to_emit,
                        skip_skeleton_bones=skip_shared, label="hair bones",
                        zero_rotation=True)
                    for entry in hair_bone_entries:
                        _log(f"  Bone '{entry[0]}' -> parent '{entry[3]}'")
                    bone_trans_entries = bone_trans_entries + hair_bone_entries
                    emitted_bone_names |= {b[0] for b in hair_bone_entries}
                    if skip_shared:
                        # Only relevant for characters - flag base-skeleton hair bones that
                        # weren't re-emitted (they should already exist in the shared armature).
                        skipped = hair_bones & RB3_SKELETON_BONES
                        if skipped:
                            _log(f"  NOTE: {len(skipped)} hair bone(s) match base-skeleton "
                                 f"names and were not re-emitted (they should already exist "
                                 f"in the shared armature): {', '.join(sorted(skipped))}")

        outfit_config_entries = []
        if self.milo_type == 'instrument':
            # An instrument MUST carry an OutfitConfig or the game hard-asserts (crashes)
            # when you select it and the customize/color panel loads
            # (ChooseColorPanel::Load -> MILO_ASSERT(mCurrentOutfitConfig, 0x21)). The game
            # looks it up by an INSTRUMENT-SPECIFIC name, not always "guitar.cfg": ClosetMgr
            # calls GetConfigNameFromAssetType, which maps guitar->guitar.cfg, bass->bass.cfg,
            # drum->drum.cfg (confirmed in the RB3 decomp and by byte-diffing a real vanilla
            # bass, which carries bass.cfg). Emitting the wrong name (e.g. guitar.cfg on a
            # bass) reproduces the exact crash-on-select. We emit a minimal dummy config for
            # now; real color palettes / mat swaps can be authored into it later.
            cfg_name = INSTRUMENT_CFG_NAMES.get(self.instrument_type, "guitar.cfg")
            outfit_config_entries = [(cfg_name, (0, 1, 2), True)]
            _log(f"--- OutfitConfig (.cfg) ---\nEmitting dummy '{cfg_name}' for "
                 f"instrument_type='{self.instrument_type}' (colors=[0,1,2], computeAO=True, "
                 f"all lists empty) so the instrument doesn't crash on select.")

        # CharCollide (.coll) volumes - gathered from any bone with collision enabled in
        # the Bone properties "Milo Collision" panel. Independent of the export type; if a
        # bone has a volume authored, it gets emitted.
        char_collide_entries = []
        if armatures_used:
            char_collide_entries = build_char_collide_entries(armatures_used, root_name)
            if char_collide_entries:
                _log(f"--- CharCollide (.coll) ---\nEmitting {len(char_collide_entries)} "
                     f"collision volume(s):")
                for ename, coll in char_collide_entries:
                    _log(f"  {ename}: shape={coll['shape']} r0={coll['radius0']} "
                         f"r1={coll['radius1']} l0={coll['length0']} l1={coll['length1']} "
                         f"-> parent '{coll['parent']}'")

        # DC1 donor handling. Two modes:
        #   inject_meshes - rebuild the DONOR with its meshes swapped for ours, keeping
        #                   everything else byte-identical (this bypasses build_milo_bytes
        #                   entirely, so none of our directory-body assumptions apply).
        #   charclipset   - build fresh as usual, copying only the donor's CharClipSet.
        raw_entries = []
        donor_path = None
        if self.game in ('dc1', 'dc3') and self.donor_milo:
            import os as _os
            donor_path = bpy.path.abspath(self.donor_milo)
            if not _os.path.isfile(donor_path):
                _log(f"EXPORT FAILED: donor milo not found: {donor_path}")
                self.report({'ERROR'}, f"Donor milo not found: {donor_path}")
                return {'CANCELLED'}
        elif self.game in ('dc1', 'dc3'):
            _log("No donor milo set - exporting a fresh milo with no donor data.")

        if self.game == 'dc3' and donor_path and self.donor_mode == 'inject_meshes':
            self.report(
                {'ERROR'},
                "DC3 donor support currently covers 'Copy CharClipSet Only'. Mesh "
                "injection into a DC3 donor isn't implemented yet (its rev-32 container "
                "isn't handled by the donor walker). Set Donor Mode to 'Copy CharClipSet "
                "Only'.")
            return {'CANCELLED'}

        if donor_path and self.donor_mode == 'inject_meshes':
            _log("--- Donor mesh injection (DC1, experimental) ---")
            try:
                data = inject_meshes_into_donor(
                    donor_path, entries, platform=self.platform,
                    write_tangents=self.export_tangents)
            except DonorInjectError as e:
                _log(f"EXPORT FAILED: donor mesh injection failed - {e}")
                self.report({'ERROR'}, f"Donor mesh injection failed: {e}")
                return {'CANCELLED'}
            with open(self.filepath, 'wb') as f:
                f.write(data)
            summary = (f"Injected {len(entries)} mesh(es) into donor -> {self.filepath} "
                       f"({len(data)} bytes). Materials/textures NOT exported.")
            _log(f"===== SUCCESS: {summary} =====")
            self.report({'INFO'}, summary)
            return {'FINISHED'}

        if donor_path:
            _log("--- Donor CharClipSet (DC1/DC3, experimental) ---")
            try:
                clip_name, clip_bytes, clip_boundaries = extract_charclipset_from_donor(donor_path)
            except DonorExtractionError as e:
                _log(f"EXPORT FAILED: donor CharClipSet extraction failed - {e}")
                self.report({'ERROR'}, f"Donor CharClipSet extraction failed: {e}")
                return {'CANCELLED'}
            raw_entries.append(("CharClipSet", clip_name, clip_bytes, clip_boundaries))

        _log("--- Materials & Textures ---")
        # mip_floor: DC1's real mip chains stop the instant either dimension would drop
        # below 16 (confirmed by byte-parsing every one of 24 real textures embedded in
        # retail emilia01.milo_xbox's nested texture sub-milos - 128/256/512 bases and
        # both square and non-square textures all bottom out at a 16px smaller dimension,
        # NEVER at 4x4 - e.g. 256x256 ATI2 normal maps carry exactly mipMaps=4, landing at
        # 16x16, and 512x128 lands at 64x16). The function's own default (mip_floor=4)
        # was going two levels deeper than any real DC1 file ever does (down to 4x4/8x4),
        # which is a real, confirmed mismatch - left unfixed pending the same byte-level
        # check for DC3, whose real mip depth hasn't been independently verified yet.
        game_mip_floor = DC1_TEX_MIP_FLOOR if self.game == 'dc1' else 4
        materials, textures, _texture_images = gather_materials_and_textures(
            materials_by_name, max_tex_size=512, ignore_tex_size_limits=self.ignore_tex_size_limits,
            platform=self.platform,
            generate_mips=(self.game in ('dc1', 'dc3') and self.dc1_generate_mipmaps),
            mip_floor=game_mip_floor)

        _log("--- Writing container ---")
        data = build_milo_bytes(root_name, entries, milo_type=self.milo_type,
                                 materials=materials, textures=textures,
                                 bone_trans_entries=bone_trans_entries,
                                 char_mesh_hide_entries=char_mesh_hide_entries,
                                 char_hair_entries=char_hair_entries,
                                 platform=self.platform,
                                 write_tangents=self.export_tangents,
                                 outfit_config_entries=outfit_config_entries,
                                 char_collide_entries=char_collide_entries,
                                 raw_entries=raw_entries,
                                 dir_revision=(DC3_MILO_REVISION if self.game == 'dc3'
                                               else RB3_MILO_REVISION),
                                 force_dc3_mat_defaults=(self.game == 'dc3'
                                                         and self.dc3_force_mat_defaults),
                                 is_dc1=(self.game == 'dc1'))

        with open(self.filepath, 'wb') as f:
            f.write(data)

        total_verts = sum(len(e[4]) for e in entries)
        total_faces = sum(len(e[5]) for e in entries)
        bone_note = f", {len(bone_trans_entries)} bone(s) (experimental)" if bone_trans_entries else ""
        hide_note = ", 1 CharMeshHide (experimental, untested in RB3)" if char_mesh_hide_entries else ""
        hair_note = (f", {len(char_hair_entries)} CharHair (experimental, untested in RB3)"
                     if char_hair_entries else "")
        tangent_note = "" if self.export_tangents else ", tangents OFF (normal maps will be flat)"
        cfg_note = ", 1 OutfitConfig (.cfg)" if outfit_config_entries else ""
        clip_note = (f", CharClipSet '{raw_entries[0][1]}' from donor" if raw_entries else "")
        summary = (
            f"Exported {len(entries)} mesh(es), {len(materials)} material(s), "
            f"{len(textures)} texture(s){bone_note}{hide_note}{hair_note}{cfg_note}{clip_note}{tangent_note}, "
            f"{total_verts} verts, {total_faces} tris to {self.filepath}"
        )
        _log(f"===== SUCCESS: {summary} ({len(data)} bytes total) =====")
        self.report({'INFO'}, summary)
        return {'FINISHED'}


def menu_func_export(self, context):
    self.layout.operator(EXPORT_OT_milo_scene.bl_idname, text="Milo Scene (.milo)")


# =====================================================================================
# SECTION 8b -- CharCollide (.coll) authoring + live viewport wireframe preview
#
# CharCollide is RB3's per-bone collision volume (a ".coll" asset). Confirmed from the
# RB3 decomp (src/system/char/CharCollide.{h,cpp}): it extends RndTransformable (so it
# parents to a bone and has a transform), revision 7, with a Shape enum and radius/length
# tunables. This section only adds AUTHORING + a live wireframe PREVIEW in Blender - it
# does NOT export a .coll yet (that writer is the next piece of work). The volumes are
# stored per-bone (on bpy.types.Bone), matching how the game attaches them to specific
# bones. The wireframe shapes mirror CharCollide::Highlight() exactly:
#   kPlane        -> a plane at the bone, facing the bone-local X axis
#   kSphere       -> sphere of radius0
#   kInsideSphere -> same sphere (collide from inside)
#   kCigar        -> capsule: radius0 hemisphere at length0, radius1 at length1, along X
#   kInsideCigar  -> same capsule
# Field names/semantics match the decomp's SYNC_PROPs (shape, flags, radius0, radius1,
# length0, length1, mesh_y_bias) so the eventual exporter can read them straight across.
# =====================================================================================

import math as _math

# Shape enum values, verbatim from CharCollide::Shape in the decomp.
COLL_SHAPE_PLANE = 0
COLL_SHAPE_SPHERE = 1
COLL_SHAPE_INSIDE_SPHERE = 2
COLL_SHAPE_CIGAR = 3
COLL_SHAPE_INSIDE_CIGAR = 4

COLL_SHAPE_ITEMS = (
    ('plane', "Plane", "Flat collision plane (kPlane), faces the bone's local +X"),
    ('sphere', "Sphere", "Spherical volume of radius0 (kSphere)"),
    ('inside_sphere', "Inside Sphere", "Sphere that collides from the inside (kInsideSphere)"),
    ('cigar', "Cigar / Capsule", "Capsule between two hemispheres along local X (kCigar)"),
    ('inside_cigar', "Inside Cigar", "Capsule that collides from the inside (kInsideCigar)"),
)

COLL_SHAPE_TO_INT = {
    'plane': COLL_SHAPE_PLANE,
    'sphere': COLL_SHAPE_SPHERE,
    'inside_sphere': COLL_SHAPE_INSIDE_SPHERE,
    'cigar': COLL_SHAPE_CIGAR,
    'inside_cigar': COLL_SHAPE_INSIDE_CIGAR,
}


def _coll_redraw(self, context):
    """Tag all 3D viewports for redraw when a collision property changes, so the
    wireframe preview updates live as the user drags a slider."""
    if context is None:
        return
    wm = getattr(context, "window_manager", None)
    if wm is None:
        return
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


class GltfMiloObjectSettings(bpy.types.PropertyGroup):
    """Per-object Milo settings. Stored on bpy.types.Object.

    mesh_parent exposes the RndTrans `parentObj` field of the exported .mesh. In a
    normal milo the exporter forces every mesh to parent to the milo root dir, because
    that's how a self-contained scene hangs together. But the TBRB "Custom Song Asset"
    workflow ships a LOOSE .mesh that a song.dta wires into the world at runtime via
    set_trans_parent - and some assets (a keyboard replacement, a prop that rides a
    character bone) want to be parented to a bone that already exists in-game (e.g.
    bone_piano_base.mesh, bone_neck.mesh) so they inherit that bone's animation/placement
    without the DTA having to call set_trans_parent at all.

    Leave BLANK to keep the exporter's default parent (the milo root for a normal export).
    This value is written verbatim as the Trans parentObj symbol, so it must match the
    exact in-game object name including any .mesh suffix (bones in these skeletons are
    named like 'bone_piano_base.mesh')."""
    mesh_parent: StringProperty(
        name="Mesh Trans Parent",
        description="Optional .mesh trans bone parenting. The name of the object/bone this "
                     "mesh should be parented to (its RndTrans parentObj), e.g. "
                     "'bone_piano_base.mesh' or 'bone_neck.mesh'. Used by the TBRB Custom "
                     "Song Asset workflow so the loose .mesh rides an existing in-game bone "
                     "the moment it loads, instead of needing a set_trans_parent call in the "
                     "song.dta. Leave BLANK to keep whatever parent the exporter uses by "
                     "default (the milo root for a normal export)",
        default="",
    )


class GltfMiloCollisionSettings(bpy.types.PropertyGroup):
    """Per-bone CharCollide (.coll) settings. Stored on bpy.types.Bone."""
    enabled: BoolProperty(
        name="Export Collision Volume",
        description="Emit a CharCollide (.coll) volume attached to this bone. Also shows a "
                     "live wireframe preview of the volume in the viewport",
        default=False,
        update=_coll_redraw,
    )
    shape: EnumProperty(
        name="Shape",
        description="CharCollide.shape - the collision volume type",
        items=COLL_SHAPE_ITEMS,
        default='sphere',
        update=_coll_redraw,
    )
    radius0: FloatProperty(
        name="Radius 0",
        description="CharCollide.radius0 - radius of the sphere, or of the length0 hemisphere "
                     "if this is a cigar/capsule",
        default=0.5, min=0.0, soft_max=5.0,
        update=_coll_redraw,
    )
    radius1: FloatProperty(
        name="Radius 1",
        description="CharCollide.radius1 - cigar only: radius of the length1 hemisphere",
        default=0.5, min=0.0, soft_max=5.0,
        update=_coll_redraw,
    )
    length0: FloatProperty(
        name="Length 0",
        description="CharCollide.length0 - cigar only: placement of the radius0 hemisphere "
                     "along the bone's local X axis. Must be <= length1",
        default=0.0, soft_min=-5.0, soft_max=5.0,
        update=_coll_redraw,
    )
    length1: FloatProperty(
        name="Length 1",
        description="CharCollide.length1 - cigar only: placement of the radius1 hemisphere "
                     "along the bone's local X axis. Must be >= length0",
        default=1.0, soft_min=-5.0, soft_max=5.0,
        update=_coll_redraw,
    )
    flags: IntProperty(
        name="Collision Group Flags",
        description="CharCollide.flags - a collision-group BITMASK. This is the linchpin of "
                     "collision: a hair strand only collides with this volume when the "
                     "strand's own 'Hookup Flags' share at least one bit with this value "
                     "(strand.hookup_flags & collide.flags != 0). Vanilla characters give "
                     "each volume a distinct power-of-two bit (1, 2, 4, 8, 16, ...) so hair "
                     "can opt into specific volumes. If this is 0 the volume collides with "
                     "NOTHING (that's why an all-zero setup silently does nothing). Default "
                     "1 = collision group 1",
        default=1, min=0,
        update=_coll_redraw,
    )
    mesh_y_bias: BoolProperty(
        name="Mesh Y Bias",
        description="CharCollide.mesh_y_bias - for spheres/cigars, biases the fit toward the "
                     "bone's local +Y (green) axis. Used by the game for chest/back volumes",
        default=False,
        update=_coll_redraw,
    )


# ------------------------------------------------------------------------------------
# Live viewport wireframe preview (GPU draw handler)
# ------------------------------------------------------------------------------------
# Drawn in world space (POST_VIEW). For each visible armature, for each bone whose
# gltfmilo_collision.enabled is True, we draw the appropriate wireframe at the bone's
# world transform. The geometry mirrors CharCollide::Highlight().

_coll_draw_handle = None


def _coll_line_shader():
    import gpu
    # 3.x/4.x compatible uniform-color line shader
    try:
        return gpu.shader.from_builtin('UNIFORM_COLOR')
    except Exception:
        return gpu.shader.from_builtin('3D_UNIFORM_COLOR')


def _circle_points(center, axis_u, axis_v, radius, segments=24):
    """Return a list of points forming a circle in the plane spanned by axis_u/axis_v."""
    from mathutils import Vector
    pts = []
    for i in range(segments):
        a = (i / segments) * 2.0 * _math.pi
        pts.append(center + axis_u * (radius * _math.cos(a)) + axis_v * (radius * _math.sin(a)))
    return pts


def _loop_segments(points):
    """Convert a closed loop of points into GPU LINE segment pairs."""
    segs = []
    n = len(points)
    for i in range(n):
        segs.append(points[i])
        segs.append(points[(i + 1) % n])
    return segs


def _sphere_wire_segments(center, rx, ry, rz, radius, segments=24):
    """Three orthogonal great-circle rings (XY, XZ, YZ) approximating a sphere."""
    segs = []
    segs += _loop_segments(_circle_points(center, rx, ry, radius, segments))
    segs += _loop_segments(_circle_points(center, rx, rz, radius, segments))
    segs += _loop_segments(_circle_points(center, ry, rz, radius, segments))
    return segs


def _capsule_wire_segments(world_mtx, radius0, radius1, length0, length1, segments=16):
    """Cigar/capsule wireframe along local X: a hemisphere of radius0 at x=length0 and a
    hemisphere of radius1 at x=length1, joined by tube lines. Mirrors UtilDrawCigar's
    intent (two radii, two placements along X)."""
    from mathutils import Vector
    # Basis vectors (world) for the bone-local axes.
    origin = world_mtx.to_translation()
    m = world_mtx.to_3x3()
    x_axis = m @ Vector((1.0, 0.0, 0.0))
    y_axis = m @ Vector((0.0, 1.0, 0.0))
    z_axis = m @ Vector((0.0, 0.0, 1.0))
    for v in (x_axis, y_axis, z_axis):
        if v.length > 1e-8:
            v.normalize()

    c0 = origin + x_axis * length0
    c1 = origin + x_axis * length1
    segs = []
    # End rings (perpendicular to X)
    segs += _loop_segments(_circle_points(c0, y_axis, z_axis, radius0, segments))
    segs += _loop_segments(_circle_points(c1, y_axis, z_axis, radius1, segments))
    # Connecting tube lines (4 around)
    for ax in (y_axis, z_axis, -y_axis, -z_axis):
        segs.append(c0 + ax * radius0)
        segs.append(c1 + ax * radius1)
    # Hemisphere caps: half-rings in the XY and XZ planes at each end.
    # radius0 cap bulges toward -X, radius1 cap toward +X.
    def half_ring(center, axis_long, axis_side, radius, sign):
        pts = []
        steps = max(4, segments // 2)
        for i in range(steps + 1):
            a = (i / steps) * _math.pi
            pts.append(center + axis_long * (sign * radius * _math.sin(a))
                       + axis_side * (radius * _math.cos(a)))
        out = []
        for i in range(len(pts) - 1):
            out.append(pts[i]); out.append(pts[i + 1])
        return out
    segs += half_ring(c0, x_axis, y_axis, radius0, -1.0)
    segs += half_ring(c0, x_axis, z_axis, radius0, -1.0)
    segs += half_ring(c1, x_axis, y_axis, radius1, +1.0)
    segs += half_ring(c1, x_axis, z_axis, radius1, +1.0)
    return segs


def _plane_wire_segments(world_mtx, size=1.0):
    """A square plane facing bone-local +X, with a short normal stub, mirroring how
    CharCollide::Highlight draws a plane oriented by WorldXfm.m.x."""
    from mathutils import Vector
    origin = world_mtx.to_translation()
    m = world_mtx.to_3x3()
    x_axis = m @ Vector((1.0, 0.0, 0.0))
    y_axis = m @ Vector((0.0, 1.0, 0.0))
    z_axis = m @ Vector((0.0, 0.0, 1.0))
    for v in (x_axis, y_axis, z_axis):
        if v.length > 1e-8:
            v.normalize()
    corners = [
        origin + y_axis * size + z_axis * size,
        origin - y_axis * size + z_axis * size,
        origin - y_axis * size - z_axis * size,
        origin + y_axis * size - z_axis * size,
    ]
    segs = _loop_segments(corners)
    # normal stub (local +X)
    segs.append(origin)
    segs.append(origin + x_axis * size)
    return segs


def _bone_world_matrix(armature_obj, bone):
    """World matrix of a bone's rest position (armature world @ bone.matrix_local)."""
    return armature_obj.matrix_world @ bone.matrix_local


def _draw_collision_volumes():
    import gpu
    from gpu_extras.batch import batch_for_shader
    context = bpy.context
    if context is None:
        return
    # Gather segments per color; enabled volumes get a red-ish wire.
    all_segs = []
    for obj in context.view_layer.objects:
        if obj.type != 'ARMATURE' or not obj.visible_get():
            continue
        arm = obj.data
        for bone in arm.bones:
            coll = getattr(bone, "gltfmilo_collision", None)
            if coll is None or not coll.enabled:
                continue
            wm = _bone_world_matrix(obj, bone)
            origin = wm.to_translation()
            m = wm.to_3x3()
            from mathutils import Vector
            rx = m @ Vector((1.0, 0.0, 0.0)); ry = m @ Vector((0.0, 1.0, 0.0)); rz = m @ Vector((0.0, 0.0, 1.0))
            for v in (rx, ry, rz):
                if v.length > 1e-8:
                    v.normalize()
            shape = coll.shape
            if shape in ('sphere', 'inside_sphere'):
                all_segs += _sphere_wire_segments(origin, rx, ry, rz, coll.radius0)
            elif shape in ('cigar', 'inside_cigar'):
                lo, hi = coll.length0, coll.length1
                if lo > hi:
                    lo, hi = hi, lo
                all_segs += _capsule_wire_segments(wm, coll.radius0, coll.radius1, lo, hi)
            elif shape == 'plane':
                all_segs += _plane_wire_segments(wm, size=max(coll.radius0, 0.5))

    if not all_segs:
        return

    shader = _coll_line_shader()
    batch = batch_for_shader(shader, 'LINES', {"pos": all_segs})
    gpu.state.line_width_set(1.5)
    gpu.state.blend_set('ALPHA')
    try:
        gpu.state.depth_test_set('LESS_EQUAL')
    except Exception:
        pass
    shader.bind()
    shader.uniform_float("color", (1.0, 0.15, 0.15, 0.9))
    batch.draw(shader)
    gpu.state.blend_set('NONE')
    try:
        gpu.state.depth_test_set('NONE')
    except Exception:
        pass


def _register_coll_draw_handler():
    global _coll_draw_handle
    if _coll_draw_handle is None:
        _coll_draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw_collision_volumes, (), 'WINDOW', 'POST_VIEW')


def _unregister_coll_draw_handler():
    global _coll_draw_handle
    if _coll_draw_handle is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_coll_draw_handle, 'WINDOW')
        except Exception:
            pass
        _coll_draw_handle = None


class BONE_PT_gltfmilo_collision(bpy.types.Panel):
    bl_label = "Milo Collision (.coll)"
    bl_idname = "BONE_PT_gltfmilo_collision"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "bone"

    @classmethod
    def poll(cls, context):
        return context.bone is not None or context.edit_bone is not None

    def draw_header(self, context):
        bone = context.bone
        if bone is not None and hasattr(bone, "gltfmilo_collision"):
            self.layout.prop(bone.gltfmilo_collision, "enabled", text="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        bone = context.bone
        if bone is None:
            layout.label(text="Select a bone in Pose or Object mode to edit collision.",
                         icon='INFO')
            return

        coll = bone.gltfmilo_collision
        col = layout.column()
        col.active = coll.enabled
        col.prop(coll, "shape")

        shape = coll.shape
        if shape in ('sphere', 'inside_sphere'):
            col.prop(coll, "radius0", text="Radius")
        elif shape in ('cigar', 'inside_cigar'):
            col.prop(coll, "radius0")
            col.prop(coll, "radius1")
            col.prop(coll, "length0")
            col.prop(coll, "length1")
            if coll.length0 > coll.length1:
                col.label(text="length0 > length1 (will be swapped for preview)",
                          icon='ERROR')
            col.prop(coll, "mesh_y_bias")
        elif shape == 'plane':
            col.prop(coll, "radius0", text="Size")

        col.prop(coll, "flags")
        layout.label(text="Preview only - .coll export not wired up yet.", icon='INFO')


class OBJECT_PT_gltfmilo_object(bpy.types.Panel):
    """Object Properties panel exposing per-object Milo export settings - currently the
    optional .mesh trans parent used by the TBRB Custom Song Asset workflow."""
    bl_label = "Milo Object"
    bl_idname = "OBJECT_PT_gltfmilo_object"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "object"

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == 'MESH'

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        obj = context.object
        settings = getattr(obj, "gltfmilo_object", None)
        if settings is None:
            layout.label(text="No Milo object settings on this object.", icon='INFO')
            return

        col = layout.column()
        col.prop(settings, "mesh_parent", text="Trans Parent")
        if (settings.mesh_parent or "").strip():
            col.label(text="Mesh will parent to this existing in-game object/bone.",
                      icon='INFO')
        else:
            col.label(text="Blank = default parent (milo root on a normal export).",
                      icon='INFO')



# =====================================================================================
# SECTION 9 -- TBRB skeleton milo IMPORT
#
# Builds a Blender armature directly from a retail TBRB skeleton milo. This exists because
# a re-derived/approximated armature is worse than useless for skinning: the retail meshes'
# inverse-bind matrices are baked against the retail rest pose, so even sub-unit joint drift
# makes vertices fly apart at runtime. Importing the real skeleton removes rest-pose drift
# as a variable entirely.
#
# Two details that matter and are easy to get wrong:
#   1. worldXfm in the file is NON-AUTHORITATIVE. 57 of george's 109 bones ship an identity
#      worldXfm while carrying a real localXfm - the engine recomputes world from
#      local * parent. So the importer resolves world transforms by walking the parent chain
#      from localXfm, never by trusting the stored worldXfm.
#   2. Milo stores a 4x3 in ROW-vector convention; Blender is column-vector. The conversion
#      here is the exact inverse of the exporter's _matrix_to_milo, so import -> export is a
#      round trip.
# =====================================================================================

def _milo_to_blender_matrix(m12):
    """Inverse of _matrix_to_milo: Milo row-vector 4x3 -> Blender column-vector Matrix."""
    from mathutils import Matrix
    return Matrix((
        (m12[0], m12[3], m12[6], m12[9]),
        (m12[1], m12[4], m12[7], m12[10]),
        (m12[2], m12[5], m12[8], m12[11]),
        (0.0,    0.0,    0.0,    1.0),
    ))


# =====================================================================================
# General milo container reader (all four compression types)
#
# The TBRB skeleton parser below only ever needed to read UNCOMPRESSED (0xCABEDEAF)
# PS3 skeleton milos, because that's how retail TBRB ships them. RB3 and Dance Central
# character/skeleton milos are frequently COMPRESSED (zlib or gzip block streams), so a
# general importer has to decompress first. This mirrors MiloLib/MiloFile.cs exactly:
#
#   Header (ALWAYS little-endian):
#     u32 magic          - one of the four Type values below
#     u32 startOffset    - byte offset where the first block begins
#     u32 numBlocks
#     u32 largestBlock   - uncompressed size of the largest block (unused here)
#     u32 blockSizes[numBlocks]
#   Body (big-endian for every 360/PS3/Wii game this plugin targets):
#     numBlocks back-to-back chunks, concatenated after per-block decompression.
#
# The four container types (MiloFile.cs Type enum):
#   0xCABEDEAF Uncompressed   - blocks are raw body bytes (blockSize & 0xFFFFFF)
#   0xCBBEDEAF CompressedZlib - each block is a raw DEFLATE stream (no zlib header:
#                               MiloLib uses Inflater(true) == raw; Python: wbits=-15)
#   0xCCBEDEAF CompressedGzip - each block is a gzip stream
#   0xCDBEDEAF CompressedZlibAlt - per block: if (blockSize & 0xFF000000) the block is
#                               stored UNCOMPRESSED with size (blockSize & 0xFFFFFF);
#                               otherwise the block begins with a u32 uncompressed-size
#                               prefix followed by (blockSize-4) raw-DEFLATE bytes.
# =====================================================================================

MILO_MAGIC_UNCOMPRESSED = 0xCABEDEAF
MILO_MAGIC_ZLIB         = 0xCBBEDEAF
MILO_MAGIC_GZIP         = 0xCCBEDEAF
MILO_MAGIC_ZLIB_ALT     = 0xCDBEDEAF
_MILO_CONTAINER_MAGICS = (MILO_MAGIC_UNCOMPRESSED, MILO_MAGIC_ZLIB,
                          MILO_MAGIC_GZIP, MILO_MAGIC_ZLIB_ALT)


def read_milo_container_body(data):
    """Return the decompressed big-endian object-stream body of any milo container.

    Accepts all four MiloFile.cs container types (uncompressed / zlib / gzip / zlibAlt).
    Raises ValueError for anything that isn't a recognised milo header. The returned
    bytes are exactly what the object-stream parser should walk (the same bytes the
    uncompressed path already produced by concatenating raw blocks)."""
    import struct as _struct
    import zlib as _zlib
    import gzip as _gzip

    if len(data) < 16:
        raise ValueError("File is too small to be a milo.")
    magic = _struct.unpack_from('<I', data, 0)[0]
    if magic not in _MILO_CONTAINER_MAGICS:
        raise ValueError(
            f"Unsupported milo container (magic 0x{magic:08X}). Expected one of "
            f"uncompressed 0xCABEDEAF, zlib 0xCBBEDEAF, gzip 0xCCBEDEAF, "
            f"zlibAlt 0xCDBEDEAF.")

    start_offset = _struct.unpack_from('<I', data, 4)[0]
    num_blocks   = _struct.unpack_from('<I', data, 8)[0]
    # data[12] is largestBlock - not needed for decoding.
    block_sizes  = [_struct.unpack_from('<I', data, 16 + 4 * i)[0]
                    for i in range(num_blocks)]

    off = start_offset
    body = bytearray()

    if magic == MILO_MAGIC_UNCOMPRESSED:
        for s in block_sizes:
            n = s & 0xFFFFFF
            body += data[off:off + n]
            off += n
        return bytes(body)

    for s in block_sizes:
        if magic == MILO_MAGIC_ZLIB_ALT:
            uncompressed_flag = (s & 0xFF000000) != 0
            size = s & 0x00FFFFFF
            if uncompressed_flag:
                body += data[off:off + size]
                off += size
            else:
                # 4-byte uncompressed-size prefix (ignored - we let zlib size it),
                # then (size - 4) raw-DEFLATE bytes.
                raw = data[off + 4:off + 4 + (size - 4)]
                off += size
                body += _zlib.decompress(raw, -15)
        else:
            raw = data[off:off + s]
            off += s
            if magic == MILO_MAGIC_ZLIB:
                body += _zlib.decompress(raw, -15)   # raw DEFLATE, no header
            else:  # MILO_MAGIC_GZIP
                body += _gzip.decompress(raw)

    return bytes(body)


def parse_milo_skeleton(path):
    """Parse ANY Harmonix skeleton/character milo (RB3, DC1, DC3, TBRB) into
    (dir_name, bones, collisions) - the same tuple shape parse_tbrb_skeleton_milo
    returns, so the shared armature builder can consume either.

      bones:      list of (name, local_xfm12, parent_name) in file order
      collisions: list of (entry_name, parent_bone, shape, radius0, length0, length1, flags)

    Handles, vs the TBRB-only parser:
      * All four container compression types (via read_milo_container_body) - RB3/DC
        retail milos are usually zlib/gzip compressed.
      * BOTH CharCollide revisions: TBRB ships rev 5 (no rev>5 tail block); RB3/DC ship
        rev 7 (adds origRadius1/curRadius1/curLength0/curLength1). The revision is read
        from each CharCollide body and the rev>5 tail is skipped only when present, so
        the six values we surface (shape/radius0/length0/length1/flags) land correctly
        regardless of game.
      * DirectoryMeta rev >= 32 (DC3) extra header byte.

    The Trans (bone) layout is revision 9 in every one of these games, so bone reading
    is identical across all four - only the container and the CharCollide tail differ.

    IMPORTANT - why objects are NOT split by naive 0xADDEADDE scanning:
    A real character milo interleaves Trans/CharCollide entries with Mesh/Mat/Tex/CharHair
    objects whose vertex/pixel data contains the byte sequence 0xADDEADDE by coincidence,
    AND legitimately uses that marker to terminate nested structures. So the marker count
    far exceeds the entry count (observed 273 markers for 139 entries), and any "split on
    every marker, object[i] = entry[i]" scheme drifts almost immediately and then reads a
    Trans body off the end of the stream. Instead we walk the entry list in order and, for
    each Trans/CharCollide entry, FULLY DECODE the object at the current position (or, if the
    preceding unknown object left us mid-stream, resync forward to the next position where a
    valid object of the expected type decodes cleanly and lands on its terminating marker).
    Unknown object types (Mesh/Mat/Tex/Group/CharHair/CharClipSet/...) are never sized - the
    resync for the NEXT wanted entry skips over them, however many internal markers they hold.
    This is validated byte-clean against retail angel01 (DC1) and aubrey01 (DC3)."""
    import struct as _struct

    with open(path, 'rb') as f:
        data = f.read()

    body = read_milo_container_body(data)   # raises ValueError on a bad container
    n = len(body)

    def U32(p):
        return _struct.unpack_from('>I', body, p)[0]

    pos = 0

    def u8():
        nonlocal pos
        v = body[pos]; pos += 1; return v

    def u32():
        nonlocal pos
        v = _struct.unpack_from('>I', body, pos)[0]; pos += 4; return v

    def i32():
        nonlocal pos
        v = _struct.unpack_from('>i', body, pos)[0]; pos += 4; return v

    def sym():
        nonlocal pos
        ln = u32()
        v = body[pos:pos + ln].decode('latin1'); pos += ln; return v

    revision = u32()
    dir_type = sym()
    dir_name = sym()
    u32()   # stringTableCount
    u32()   # stringTableSize
    if revision >= DC3_MILO_REVISION:
        u8()
    entry_count = i32()
    if entry_count < 0 or entry_count > 100000:
        raise ValueError(
            f"Milo directory entry count {entry_count} is implausible - the body may be "
            f"the wrong endianness or this isn't a directory milo.")
    entries = [(sym(), sym()) for _ in range(entry_count)]
    entry_end = pos

    RB3_TRANS_REV = 9

    def try_read_trans(p):
        """Fully decode a rev-9 standalone Trans at byte offset p. Returns
        (name_ignored_here, local12, parent, end_after_marker) or None if it doesn't
        decode cleanly into a trailing 0xADDEADDE. Bounds/sanity checks make a spurious
        match inside mesh/pixel data astronomically unlikely."""
        try:
            if p + 4 > n or (U32(p) & 0xFFFF) != RB3_TRANS_REV:
                return None
            q = p + 4
            q += 4                                   # objFields combined revision
            tl = U32(q)
            if tl > 64:
                return None
            q += 4 + tl                              # objFields type symbol
            if body[q] not in (0, 1):
                return None
            q += 1                                   # hasTree bool
            nl = U32(q)
            if nl > 256:
                return None
            q += 4 + nl                              # note symbol
            local = _struct.unpack_from('>12f', body, q); q += 48
            q += 48                                  # worldXfm (ignored)
            q += 4                                   # constraint u32
            tl2 = U32(q)
            if tl2 > 256:
                return None
            q += 4 + tl2                             # target symbol
            if body[q] not in (0, 1):
                return None
            q += 1                                   # preserveScale bool
            pl = U32(q)
            if pl > 256:
                return None
            parent = body[q + 4:q + 4 + pl].decode('latin1'); q += 4 + pl
            if body[q:q + 4] != MILO_END_MARKER:
                return None
            return local, parent, q + 4
        except Exception:
            return None

    def try_read_collide(p):
        """Fully decode a rev-5 (TBRB) or rev-7 (RB3/DC) CharCollide at offset p.
        Returns (parent, shape, r0, l0, l1, flags, end_after_marker) or None."""
        try:
            if p + 4 > n:
                return None
            crev = U32(p) & 0xFFFF
            if crev not in (5, 7):
                return None
            q = p + 4
            q += 4                                   # objFields combined revision
            tl = U32(q)
            if tl > 64:
                return None
            q += 4 + tl
            if body[q] not in (0, 1):
                return None
            q += 1
            nl = U32(q)
            if nl > 256:
                return None
            q += 4 + nl
            if (U32(q) & 0xFFFF) != RB3_TRANS_REV:   # embedded RndTrans revision
                return None
            q += 4
            q += 96                                  # local + world matrices
            q += 4                                   # constraint
            tl2 = U32(q)
            if tl2 > 256:
                return None
            q += 4 + tl2                             # target
            if body[q] not in (0, 1):
                return None
            q += 1                                   # preserveScale
            pl = U32(q)
            if pl > 256:
                return None
            parent = body[q + 4:q + 4 + pl].decode('latin1'); q += 4 + pl
            shape = U32(q); q += 4
            r0 = _struct.unpack_from('>f', body, q)[0]; q += 4
            l0 = _struct.unpack_from('>f', body, q)[0]; q += 4
            l1 = _struct.unpack_from('>f', body, q)[0]; q += 4
            flags = _struct.unpack_from('>i', body, q)[0]; q += 4
            q += 4                                   # curRadius0 (rev > 3)
            if crev > 5:
                q += 16                              # origRadius1,curRadius1,curLength0,curLength1
                q += 48                              # unknownTransform (12 floats)
                ml = U32(q)
                if ml > 128:
                    return None
                q += 4 + ml                          # mesh symbol
                q += 8 * 16                          # 8x CharCollideStruct (i32 + 3 floats)
                q += 20                              # sha1
                q += 1                               # meshYBias
            if body[q:q + 4] != MILO_END_MARKER:
                return None
            return parent, shape, r0, l0, l1, flags, q + 4
        except Exception:
            return None

    bones = []
    collisions = []

    # Recursively collect every asset in the file - including ones nested inside inline
    # subdirectories - instead of walking only the top-level `entries` list. This matters
    # a lot in practice: byte-parsing retail emilia01.milo_xbox (DC1) shows 93 of its 173
    # real Trans (bone) entries live ONLY inside a nested '../female_skeleton.milo'
    # subdirectory (the shared base skeleton - arms, legs, fingers, spine), while the
    # 80 Trans entries visible at the top level are just the character-specific extras
    # (face, hair, jiggle, wrap bones) that parent ONTO that missing base skeleton. The
    # old code below only ever called resync() len(entries) times (once per TOP-LEVEL
    # declared entry), so it could find at most 80 bones here no matter how good the
    # marker-scanning was - and worse, because cur never advances across a non-Trans/
    # CharCollide top-level entry, the very first resync('Trans') call could scan straight
    # through the entire subdirectory region and return one of ITS bones mislabeled with
    # whatever name the outer loop happened to be on, silently corrupting bone data rather
    # than just omitting bones. Walking the real structure fixes both: every bone gets its
    # own correct name and exact byte offset, and none are skipped or misattributed.
    _, all_entries = _mw_collect_directory_meta(body, 0)

    def _decode_at(offset, probe):
        """Decode directly at a known-correct offset; only fall back to a short forward
        scan if that exact decode unexpectedly fails (defensive - the recursive walk
        above should always land exactly on the object, but this keeps behaviour at
        least as robust as the old resync() for anything unforeseen)."""
        r = probe(offset)
        if r is not None:
            return r
        s = offset
        while s < n - 4:
            m = body.find(MILO_END_MARKER, s)
            if m < 0:
                break
            r = probe(m + 4)
            if r is not None:
                return r
            s = m + 4
        return None

    for (etype, ename, estart, _eend) in all_entries:
        if etype == 'Trans':
            r = _decode_at(estart, try_read_trans)
            if r is None:
                continue
            local, parent, _endp = r
            bones.append((ename, local, parent))
        elif etype == 'CharCollide':
            r = _decode_at(estart, try_read_collide)
            if r is None:
                continue
            parent, shape, r0, l0, l1, flags, _endp = r
            collisions.append((ename, parent, shape, r0, l0, l1, flags))
        # Everything else (Mesh/Mat/Tex/Group/CharHair/CharClipSet/SubDir/...) is ignored -
        # the recursive walk above already accounts for their bytes structurally, so there's
        # no need to size or skip them manually here the way the old resync loop did.

    if not bones and entry_count == 0:
        raise ValueError(
            "This milo's directory has no top-level entries - its bones (if any) are stored "
            "inside the ObjectDir as animation/driver data, not as a plain rest-pose list "
            "(this is how RB3's shared world-base skeleton.milo is structured). Import the "
            "bone rest pose from a CHARACTER milo instead (e.g. a retail RB3 character), "
            "which stores its skeleton as top-level Trans entries.")

    return dir_name, bones, collisions


def parse_tbrb_skeleton_milo(path):
    """Parses a TBRB skeleton milo into (dir_name, bones, collisions).

    bones:      list of (name, local_xfm12, parent_name) in file order
    collisions: list of (entry_name, parent_bone, shape_int, radius0, length0, length1, flags)
    """
    import struct as _struct

    with open(path, 'rb') as f:
        data = f.read()
    if len(data) < 16:
        raise ValueError("File is too small to be a milo.")
    magic = _struct.unpack_from('<I', data, 0)[0]
    if magic != 0xCABEDEAF:
        raise ValueError(
            f"Unsupported milo container (magic 0x{magic:08X}). Only uncompressed milos "
            f"(0xCABEDEAF) are supported - retail TBRB skeleton milos are uncompressed.")
    start_offset = _struct.unpack_from('<I', data, 4)[0]
    block_count = _struct.unpack_from('<I', data, 8)[0]
    sizes = [_struct.unpack_from('<I', data, 16 + 4 * i)[0] for i in range(block_count)]
    body = bytearray()
    off = start_offset
    for s in sizes:
        n = s & 0xFFFFFF
        body += data[off:off + n]
        off += n
    body = bytes(body)

    pos = 0

    def u8():
        nonlocal pos
        v = body[pos]; pos += 1; return v

    def u32():
        nonlocal pos
        v = _struct.unpack_from('>I', body, pos)[0]; pos += 4; return v

    def i32():
        nonlocal pos
        v = _struct.unpack_from('>i', body, pos)[0]; pos += 4; return v

    def f32():
        nonlocal pos
        v = _struct.unpack_from('>f', body, pos)[0]; pos += 4; return v

    def sym():
        nonlocal pos
        n = u32()
        v = body[pos:pos + n].decode('latin1'); pos += n; return v

    def mat12():
        return tuple(f32() for _ in range(12))

    revision = u32()
    dir_type = sym()
    dir_name = sym()
    u32()   # stringTableCount
    u32()   # stringTableSize
    if revision >= DC3_MILO_REVISION:
        u8()
    entry_count = i32()
    entries = [(sym(), sym()) for _ in range(entry_count)]

    # Split the remaining body into per-object byte ranges on the 0xADDEADDE end marker.
    obj_ranges = []
    scan = pos
    while True:
        nxt = body.find(MILO_END_MARKER, scan)
        if nxt < 0:
            break
        obj_ranges.append((scan, nxt))
        scan = nxt + 4
    if len(obj_ranges) < entry_count + 1:
        raise ValueError(
            f"Milo looks malformed: {entry_count} entries but only {len(obj_ranges)} objects.")

    bones = []
    collisions = []
    for idx, (etype, ename) in enumerate(entries):
        pos = obj_ranges[idx + 1][0]     # +1 skips the directory object itself
        if etype == 'Trans':
            u32()                                    # combined revision
            u32(); sym(); u8(); sym()                # objFields
            local_xfm = mat12()
            mat12()                                  # worldXfm - deliberately ignored
            u32(); sym(); u8()                       # constraint, target, preserveScale
            parent = sym()
            bones.append((ename, local_xfm, parent))
        elif etype == 'CharCollide':
            u32()                                    # revision
            u32(); sym(); u8(); sym()                # objFields
            u32()                                    # embedded Trans revision
            mat12(); mat12()                         # local, world
            u32(); sym(); u8()                       # constraint, target, preserveScale
            parent = sym()
            shape = u32(); radius0 = f32(); length0 = f32(); length1 = f32(); flags = i32()
            collisions.append((ename, parent, shape, radius0, length0, length1, flags))

    return dir_name, bones, collisions


def _build_armature_from_skeleton(context, dir_name, bones, collisions,
                                  bone_display_length=0.5, connect_bones=False,
                                  import_collisions=True):
    """Shared skeleton -> Blender armature builder used by every game's import operator.

    Takes the (dir_name, bones, collisions) tuple that parse_milo_skeleton /
    parse_tbrb_skeleton_milo return and constructs the armature object, resolving each
    bone's world transform by walking the parent chain from LOCAL matrices only (the
    stored worldXfm is non-authoritative - mostly identity in retail files). Returns
    (arm_obj, applied_collision_count) or raises ValueError on a cyclic/invalid rig.

    This is deliberately game-agnostic: the bone (Trans rev 9) and the surfaced
    collision fields are identical across RB3/DC1/DC3/TBRB, so the only per-game
    differences live in the parser (container + CharCollide tail) and the operator
    wrappers (labels/defaults), not here."""
    local_by_name = {n: lx for (n, lx, _p) in bones}
    parent_by_name = {n: p for (n, _lx, p) in bones}
    world_cache = {}

    def world_of(name, _stack=None):
        if name in world_cache:
            return world_cache[name]
        _stack = _stack or set()
        if name in _stack:
            raise ValueError(f"Cyclic bone parenting at '{name}'")
        _stack = _stack | {name}
        local = _milo_to_blender_matrix(local_by_name[name])
        parent = parent_by_name.get(name)
        if parent in local_by_name and parent != name:
            result = world_of(parent, _stack) @ local
        else:
            result = local
        world_cache[name] = result
        return result

    for n in local_by_name:
        world_of(n)

    arm_data = bpy.data.armatures.new(dir_name)
    arm_obj = bpy.data.objects.new(dir_name, arm_data)
    context.scene.collection.objects.link(arm_obj)
    for o in context.selected_objects:
        o.select_set(False)
    arm_obj.select_set(True)
    context.view_layer.objects.active = arm_obj

    bpy.ops.object.mode_set(mode='EDIT')
    try:
        children = {}
        for n, _lx, p in bones:
            children.setdefault(p, []).append(n)

        edit_bones = {}
        for (name, _lx, _p) in bones:
            eb = arm_data.edit_bones.new(name)
            # A non-zero length must exist before assigning .matrix, or Blender discards
            # the bone. The matrix assignment then sets head/orientation/roll together.
            eb.head = (0.0, 0.0, 0.0)
            eb.tail = (0.0, bone_display_length, 0.0)
            eb.matrix = world_of(name)
            edit_bones[name] = eb
            if eb.name != name:
                _log(f"  WARNING: Blender renamed bone '{name}' -> '{eb.name}' "
                     f"(duplicate name). Export will not match the retail skeleton.")

        for (name, _lx, parent) in bones:
            if parent in edit_bones and parent != name:
                edit_bones[name].parent = edit_bones[parent]

        for (name, _lx, _p) in bones:
            eb = edit_bones[name]
            kids = [edit_bones[c] for c in children.get(name, []) if c in edit_bones]
            if len(kids) == 1:
                vec = kids[0].head - eb.head
                if vec.length > 1e-4:
                    eb.length = vec.length
                    if connect_bones:
                        eb.tail = kids[0].head
                        kids[0].use_connect = True
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')

    applied = 0
    if import_collisions:
        int_to_shape = {v: k for k, v in COLL_SHAPE_TO_INT.items()}
        seen_parents = set()
        for (ename, parent, shape, radius0, length0, length1, flags) in collisions:
            bone = arm_data.bones.get(parent)
            if bone is None:
                _log(f"  Collision '{ename}': parent bone '{parent}' not found, skipped.")
                continue
            if parent in seen_parents:
                _log(f"  Collision '{ename}': bone '{parent}' already holds a volume - "
                     f"only one per bone is storable, skipped.")
                continue
            seen_parents.add(parent)
            coll = bone.gltfmilo_collision
            coll.enabled = True
            coll.shape = int_to_shape.get(shape, 'sphere')
            coll.radius0 = radius0
            coll.radius1 = radius0
            coll.length0 = min(length0, length1)
            coll.length1 = max(length0, length1)
            coll.flags = flags
            applied += 1

    return arm_obj, applied


class IMPORT_OT_tbrb_skeleton(bpy.types.Operator, ImportHelper):
    """Import a TBRB skeleton milo as a Blender armature"""
    bl_idname = "import_scene.tbrb_skeleton"
    bl_label = "Import TBRB Skeleton Milo"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".milo_ps3"
    filter_glob: StringProperty(
        default="*.milo_ps3;*.milo_xbox;*.milo",
        options={'HIDDEN'},
    )

    bone_display_length: FloatProperty(
        name="Bone Display Length",
        description="Length given to bones that have no child to point at. Purely cosmetic - "
                     "Milo bones are transforms, so only the head position and orientation "
                     "are exported. Does NOT affect exported data",
        default=0.5, min=0.01, max=20.0,
    )

    connect_bones: BoolProperty(
        name="Auto-Connect Bones",
        description="Snap each bone's tail to its single child's head. Leave OFF unless you "
                     "want prettier viewport display: connecting bones makes Blender move a "
                     "child's head whenever the parent's tail moves, which is exactly how a "
                     "faithful rest pose gets silently corrupted during editing",
        default=False,
    )

    import_collisions: BoolProperty(
        name="Import Collision Volumes",
        description="Copy each CharCollide's shape/radius/length/flags onto its parent bone's "
                     "Milo Collision panel. NOTE: the panel holds ONE volume per bone, and "
                     "TBRB puts three on bone_head.mesh - only the first is stored. Export "
                     "with 'Retail Head/Neck Collisions' on to emit all four correctly",
        default=True,
    )

    def execute(self, context):
        try:
            dir_name, bones, collisions = parse_tbrb_skeleton_milo(self.filepath)
        except Exception as e:
            _log(f"IMPORT FAILED: {e}")
            self.report({'ERROR'}, f"Could not parse skeleton milo: {e}")
            return {'CANCELLED'}

        if not bones:
            self.report({'ERROR'}, "No Trans bones found - is this a skeleton milo?")
            return {'CANCELLED'}

        _log(f"===== Importing TBRB skeleton '{dir_name}' from {self.filepath} =====")
        _log(f"  {len(bones)} bone(s), {len(collisions)} collision volume(s)")

        try:
            _arm, applied = _build_armature_from_skeleton(
                context, dir_name, bones, collisions,
                bone_display_length=self.bone_display_length,
                connect_bones=self.connect_bones,
                import_collisions=self.import_collisions)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        summary = (f"Imported skeleton '{dir_name}': {len(bones)} bone(s), "
                   f"{applied} collision volume(s) applied")
        _log(f"===== SUCCESS: {summary} =====")
        self.report({'INFO'}, summary)
        return {'FINISHED'}


class _IMPORT_OT_milo_skeleton_base(bpy.types.Operator, ImportHelper):
    """Shared base for the RB3/DC1/DC3 skeleton importers. Each subclass only sets its
    bl_idname/bl_label, the display name shown in the File > Import menu, and its default
    file extension/filter. All of them use the general parse_milo_skeleton (which handles
    compressed containers and both CharCollide revisions) plus the shared armature builder,
    so the only real difference between games is the label and the filename filter."""
    bl_options = {'REGISTER', 'UNDO'}

    # Subclasses override these three:
    _game_label = "Milo"
    filename_ext = ".milo_xbox"
    filter_glob: StringProperty(
        default="*.milo_xbox;*.milo_ps3;*.milo_wii;*.milo",
        options={'HIDDEN'},
    )

    bone_display_length: FloatProperty(
        name="Bone Display Length",
        description="Length given to bones that have no child to point at. Purely cosmetic - "
                     "Milo bones are transforms, so only the head position and orientation "
                     "are exported. Does NOT affect exported data",
        default=0.5, min=0.01, max=20.0,
    )

    connect_bones: BoolProperty(
        name="Auto-Connect Bones",
        description="Snap each bone's tail to its single child's head. Leave OFF unless you "
                     "want prettier viewport display: connecting bones makes Blender move a "
                     "child's head whenever the parent's tail moves, which is exactly how a "
                     "faithful rest pose gets silently corrupted during editing",
        default=False,
    )

    import_collisions: BoolProperty(
        name="Import Collision Volumes",
        description="Copy each CharCollide's shape/radius/length/flags onto its parent bone's "
                     "Milo Collision panel. The panel holds ONE volume per bone; if a milo "
                     "puts more than one on the same bone only the first is stored",
        default=True,
    )

    def execute(self, context):
        try:
            dir_name, bones, collisions = parse_milo_skeleton(self.filepath)
        except Exception as e:
            _log(f"IMPORT FAILED: {e}")
            self.report({'ERROR'}, f"Could not parse {self._game_label} milo: {e}")
            return {'CANCELLED'}

        if not bones:
            self.report({'ERROR'},
                        "No Trans bones found - is this a skeleton/character milo with a "
                        "shared armature? (A mesh-only milo has no bones to import.)")
            return {'CANCELLED'}

        _log(f"===== Importing {self._game_label} skeleton '{dir_name}' "
             f"from {self.filepath} =====")
        _log(f"  {len(bones)} bone(s), {len(collisions)} collision volume(s)")

        try:
            _arm, applied = _build_armature_from_skeleton(
                context, dir_name, bones, collisions,
                bone_display_length=self.bone_display_length,
                connect_bones=self.connect_bones,
                import_collisions=self.import_collisions)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        summary = (f"Imported {self._game_label} skeleton '{dir_name}': "
                   f"{len(bones)} bone(s), {applied} collision volume(s) applied")
        _log(f"===== SUCCESS: {summary} =====")
        self.report({'INFO'}, summary)
        return {'FINISHED'}


class IMPORT_OT_rb3_skeleton(_IMPORT_OT_milo_skeleton_base):
    """Import a Rock Band 3 skeleton/character milo as a Blender armature.

    Use this to pull the authoritative RB3 shared skeleton (the fixed rig every RB3
    character is weighted to) straight out of a retail milo, so your character template's
    rest pose matches the game exactly. Handles compressed retail milos automatically."""
    bl_idname = "import_scene.rb3_skeleton"
    bl_label = "Import RB3 Skeleton Milo"
    _game_label = "RB3"
    filename_ext = ".milo_xbox"
    filter_glob: StringProperty(
        default="*.milo_xbox;*.milo_ps3;*.milo_wii;*.milo",
        options={'HIDDEN'},
    )


class IMPORT_OT_dc1_skeleton(_IMPORT_OT_milo_skeleton_base):
    """Import a Dance Central 1 skeleton/character milo as a Blender armature.

    DC1 shares RB3's DirectoryMeta revision (28) and CharCollide revision (7); the parser
    is identical to RB3's path. Use it to recover DC1's dancer skeleton for a correct
    character template rest pose."""
    bl_idname = "import_scene.dc1_skeleton"
    bl_label = "Import DC1 Skeleton Milo"
    _game_label = "DC1"
    filename_ext = ".milo_xbox"
    filter_glob: StringProperty(
        default="*.milo_xbox;*.milo;*.milo_ps3",
        options={'HIDDEN'},
    )


class IMPORT_OT_dc3_skeleton(_IMPORT_OT_milo_skeleton_base):
    """Import a Dance Central 3 skeleton/character milo as a Blender armature.

    DC3 uses DirectoryMeta revision 32 (an extra header byte vs RB3/DC1); the parser
    accounts for that automatically. Bone and CharCollide layouts are otherwise the same
    as RB3. Use it to recover DC3's dancer skeleton for a correct template rest pose."""
    bl_idname = "import_scene.dc3_skeleton"
    bl_label = "Import DC3 Skeleton Milo"
    _game_label = "DC3"
    filename_ext = ".milo_xbox"
    filter_glob: StringProperty(
        default="*.milo_xbox;*.milo",
        options={'HIDDEN'},
    )


class TOPBAR_MT_milo_skeleton_import(bpy.types.Menu):
    """File > Import > Milo Skeleton Importer submenu grouping every game's skeleton
    importer in one place instead of scattering four entries across the Import menu."""
    bl_idname = "TOPBAR_MT_milo_skeleton_import"
    bl_label = "Milo Skeleton Importer"

    def draw(self, context):
        layout = self.layout
        layout.operator(IMPORT_OT_rb3_skeleton.bl_idname,
                        text="Rock Band 3 (.milo_xbox)")
        layout.operator(IMPORT_OT_tbrb_skeleton.bl_idname,
                        text="The Beatles: Rock Band (.milo_ps3)")
        layout.operator(IMPORT_OT_dc1_skeleton.bl_idname,
                        text="Dance Central 1 (.milo_xbox)")
        layout.operator(IMPORT_OT_dc3_skeleton.bl_idname,
                        text="Dance Central 3 (.milo_xbox)")


def menu_func_import(self, context):
    self.layout.menu(TOPBAR_MT_milo_skeleton_import.bl_idname)


classes = (
    GltfMiloMaterialSettings,
    MATERIAL_OT_gltfmilo_autodetect_textures,
    MATERIAL_PT_gltfmilo_settings,
    MATERIAL_PT_gltfmilo_refract,
    GltfMiloHairBoneItem,
    GltfMiloHairStrand,
    GltfMiloHairSettings,
    ARMATURE_OT_gltfmilo_create_hair_strand,
    ARMATURE_OT_gltfmilo_remove_hair_strand,
    DATA_UL_gltfmilo_hair_strands,
    DATA_PT_gltfmilo_hair,
    GltfMiloObjectSettings,
    OBJECT_PT_gltfmilo_object,
    GltfMiloCollisionSettings,
    BONE_PT_gltfmilo_collision,
    EXPORT_OT_milo_scene,
    IMPORT_OT_tbrb_skeleton,
    IMPORT_OT_rb3_skeleton,
    IMPORT_OT_dc1_skeleton,
    IMPORT_OT_dc3_skeleton,
    TOPBAR_MT_milo_skeleton_import,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Material.gltfmilo_settings = PointerProperty(type=GltfMiloMaterialSettings)
    bpy.types.Armature.gltfmilo_hair = PointerProperty(type=GltfMiloHairSettings)
    bpy.types.Bone.gltfmilo_collision = PointerProperty(type=GltfMiloCollisionSettings)
    bpy.types.Object.gltfmilo_object = PointerProperty(type=GltfMiloObjectSettings)
    _register_coll_draw_handler()
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    _unregister_coll_draw_handler()
    del bpy.types.Object.gltfmilo_object
    del bpy.types.Bone.gltfmilo_collision
    del bpy.types.Armature.gltfmilo_hair
    del bpy.types.Material.gltfmilo_settings
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
