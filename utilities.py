import bpy
import struct
import re


def _log(msg):
    print(f"[Milo Export] {msg}")


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


RB3_MILO_REVISION = 28


DC3_MILO_REVISION = 32


RB3_OBJDIR_REVISION = 27


RB3_RNDDIR_REVISION = 10


RB3_TRANS_REVISION = 9


RB3_DRAW_REVISION = 3


DC3_DRAW_REVISION = 4


RB3_ANIM_REVISION = 4


OBJ_FIELDS_REVISION = 2


MAX_MILO_BLOCK_SIZE = 0x20000


END_MARKER = b"\xAD\xDE\xAD\xDE"


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
    w.u32(0)


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
        w.symbol("")


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


TBRB_MILO_REVISION = 25


TBRB_OBJDIR_REVISION = 22


TBRB_RNDDIR_REVISION = 10


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
    w.symbol("")


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


MAX_BONES_PER_MESH = 40


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


