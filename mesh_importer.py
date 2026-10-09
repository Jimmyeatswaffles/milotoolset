"""
Green Day: Rock Band mesh importer.

Reads the Mesh entries out of a GDRB character milo and builds Blender meshes with UVs,
custom normals, per-bone vertex groups and placeholder materials. Textures are NOT
imported - materials are created as named stand-ins so the material assignment and slot
layout survive, and real texture work can happen later.

===========================================================================================
FORMAT - verified against a retail billiejoe_head_21.milo_xbox
===========================================================================================
Milo revision 25 Character dir, big-endian, 45 entries (21 Mesh, 8 Tex, 6 Mat, 6
TexBlendController, 3 Group, 1 TexBlender). RndMesh version 37 is GDRB; 36 is TBRB and 38
is RB3/DC2, and all three share the same "next-gen" packed vertex block, so this reads 36
as well - though only 37 has been checked against real data.

Each mesh carries a 36-byte packed vertex, confirmed by decoding the real file and testing
both candidate layouts documented in the format notes:

    offset  0  float x, y, z
    offset 12  u32   colour, RGBA
    offset 16  half  u, v
    offset 20  u32   normal,  signed 10/10/10/2
    offset 24  u32   tangent, signed 10/10/10/2
    offset 28  u32   weights, unsigned 10/10/10/2
    offset 32  u8    bone indices x4

This is byte-identical to what model_exporter.write_rnd_mesh already emits for RB3
revision 38 on Xbox 360, which is a useful independent cross-check.

The alternative layout the notes describe (UVs immediately after the position, then two
compressed vectors, then u16 bone indices) decodes the same bytes into garbage: normals
come out with lengths from 0.27 to 1.41, weights summing to 2.56, bone indices reaching
65279 and NaN UVs. The layout above gives unit-length normals (0.997-1.000), weights
summing to 0.899-1.000, bone indices spanning 1-29, and UVs inside 0-1. The format notes
distinguish the two by peeking at the word after the position and testing for 0xFFFFFFFF;
that is really just a white vertex colour, so this reader keys off the vertex size and
compression type in the header instead, which is what actually determines the layout.

Vertex positions are in character space, not mesh-local: the head mesh spans Z 61.1-70.9,
matching bone_head's world height in billiejoe_skeleton. No per-mesh transform is applied.

UVs are stored V-flipped relative to Blender, so import inverts V. That matches
model_exporter's export path, which writes `v = 1.0 - uv[1]`.

Bone indices are local to a per-mesh bone table that follows the faces, capped at
RndMesh::MaxBones() == 40 - the head mesh uses 30, the eyes 2. Names there carry the
'.mesh' extension exactly as in the skeleton milo, so vertex groups line up with an
imported armature without renaming.

===========================================================================================
KNOWN GAPS
===========================================================================================
1. Textures aren't read. Materials are created as placeholders named after the milo's Mat
   entry so slots and assignments are preserved, but they have no maps.
2. Tangents are decoded but discarded - Blender recomputes tangents from UVs when it needs
   them, and a stored tangent can't be assigned directly anyway.
3. Vertex colours are read but not applied. Every vertex in the sampled file is white
   (0xFFFFFFFF), so there's nothing to preserve yet; the value is exposed in the parsed
   data if that changes.
4. Group entries (hh_lod00/01/02) are not parsed. LOD detection is by name instead, which
   also survives the misspelled 'billiejoel_tongueLOD02.mesh' present in the retail file.
"""

import re
import struct

import bpy
from bpy.props import StringProperty, BoolProperty
from bpy_extras.io_utils import ImportHelper

from .io import read_milo_container_body
from .utilities import _log


MILO_END_MARKER = b'\xAD\xDE\xAD\xDE'
# RndMesh versions sharing the packed next-gen vertex block.
GDRB_MESH_VERSION = 37
TBRB_MESH_VERSION = 36
NG_VERTEX_SIZE = 36
NG_COMPRESSION_XBOX = 1
# RndMesh::MaxBones() is documented as 40, but retail data exceeds it: 21st.milo_xbox's
# 21st_skin.mesh uses 41 bones and 21st_skinLOD01.mesh uses 43, and both decode perfectly
# (unit-length normals, weight sums of exactly 1.0, zero out-of-range bone slots). Those
# two meshes are the character's arms and hands, so enforcing 40 as a hard limit silently
# dropped the arms from every outfit import. Treat 40 as advisory and only reject counts
# implausible enough to mean a genuine mis-parse.
DOCUMENTED_MAX_BONES = 40
MAX_MESH_BONES = 255

# One quantisation step of a 10-bit weight is 1/1023 ~= 0.000978. Deriving the fourth
# weight as the remainder leaves that much rounding residue on vertices that really only
# use three bones, and the unused fourth slot's bone byte is padding that can point past
# the mesh's bone table. In the retail head milo this accounts for exactly 22 slots, every
# one of them carrying precisely 1/1023. Dropping anything at or below a step avoids
# creating vertex-group entries out of rounding noise.
WEIGHT_EPSILON = 1.5 / 1023.0

_LOD_RE = re.compile(r'LOD\s*\d+', re.IGNORECASE)
_SHADOW_RE = re.compile(r'shadow', re.IGNORECASE)


class MeshImportError(Exception):
    """Raised when a milo can't be read as a GDRB/TBRB mesh container."""
    pass


_LOD_NUM_RE = re.compile(r'LOD\s*(\d+)', re.IGNORECASE)


def is_lod_name(name):
    """True for the reduced-detail copies. Matched on the name because GDRB's retail file
    ships 'billiejoel_tongueLOD02.mesh' - a typo'd stem that no base-name pairing would
    catch, but which the LOD suffix still identifies correctly.

    Only a LOD NUMBER above 0 counts. GDRB leaves its full-detail meshes unnumbered
    ('billiejoe_head.1.mesh', then 'billiejoe_headLOD01'), but TBRB numbers every mesh, with
    '_lod00' as the full-detail one ('g_head_lod00', '_lod01', '_lod02'). Treating any LOD
    suffix as reduced, as an earlier version did, would have dropped every TBRB mesh."""
    m = _LOD_NUM_RE.search(name)
    return m is not None and int(m.group(1)) > 0


def is_shadow_name(name):
    """True for the flat blob mesh used to cast the character's shadow. An outfit milo
    carries one 'shadow.mesh' alongside a matching 'shadow.grp'. It is not body geometry -
    it spans the character's whole height at very low density (480 verts over Z 0-73.9) -
    so it just clutters the viewport when what you want is the model."""
    return bool(_SHADOW_RE.search(name))


# ---------------------------------------------------------------------------------------
# Binary helpers
# ---------------------------------------------------------------------------------------

def _u32(b, o):
    return struct.unpack_from('>I', b, o)[0]


def _numstring(b, o, max_len=512):
    n = _u32(b, o)
    if n > max_len or o + 4 + n > len(b):
        return None, o
    return b[o + 4:o + 4 + n].decode('latin-1'), o + 4 + n


def _decode_snorm_10_10_10_2(v):
    def sx(bits, width):
        mask = (1 << width) - 1
        bits &= mask
        return bits - (1 << width) if bits & (1 << (width - 1)) else bits
    x = max(sx(v, 10) / 511.0, -1.0)
    y = max(sx(v >> 10, 10) / 511.0, -1.0)
    z = max(sx(v >> 20, 10) / 511.0, -1.0)
    w = max(sx(v >> 30, 2) / 1.0, -1.0)
    return x, y, z, w


def _decode_unorm_10_10_10_2(v):
    return ((v & 1023) / 1023.0, ((v >> 10) & 1023) / 1023.0,
            ((v >> 20) & 1023) / 1023.0, ((v >> 30) & 3) / 3.0)


# ---------------------------------------------------------------------------------------
# Container / entry walking
# ---------------------------------------------------------------------------------------

def _read_dir_entries(body):
    p = [0]

    def u32():
        v = _u32(body, p[0]); p[0] += 4; return v

    def sym():
        n = u32()
        if n > 4096 or p[0] + n > len(body):
            raise MeshImportError("implausible symbol length in the directory table")
        s = body[p[0]:p[0] + n].decode('latin-1'); p[0] += n
        return s

    revision = u32()
    dir_type = sym()
    dir_name = sym()
    u32(); u32()
    count = u32()
    if count > 65535:
        raise MeshImportError(f"implausible entry count {count}")
    return revision, dir_type, dir_name, [(sym(), sym()) for _ in range(count)]


def _entry_spans(body, entry_count):
    """Entry byte ranges, delimited by end markers. The generic walker can't traverse
    revision-25 dirs, and a count mismatch means the split would be silently wrong, so
    it's asserted rather than assumed."""
    marks = []
    i = 0
    while True:
        j = body.find(MILO_END_MARKER, i)
        if j < 0:
            break
        marks.append(j)
        i = j + 4
    if len(marks) != entry_count + 1:
        raise MeshImportError(
            f"found {len(marks)} end markers but expected {entry_count + 1} - this "
            f"file's layout isn't what this importer understands")
    return [(marks[k] + 4, marks[k + 1]) for k in range(entry_count)]


# ---------------------------------------------------------------------------------------
# Mesh decoding
# ---------------------------------------------------------------------------------------

def _find_vertex_block(chunk):
    """Locates the packed vertex block by its header signature: isNextGen==1, then the
    vertex size and compression type. Scanning for this is deliberate - everything before
    it is an embedded Trans and Draw whose length varies with revision, and resyncing on a
    signature that's then structurally validated is more robust than tracking every
    version-conditional field in those two objects. Returns (vert_count, payload_offset)."""
    sig = bytes([1]) + struct.pack('>I', NG_VERTEX_SIZE) + struct.pack('>I',
                                                                      NG_COMPRESSION_XBOX)
    start = 0
    while True:
        h = chunk.find(sig, start)
        if h < 0:
            raise MeshImportError(
                "no next-gen vertex block found - this mesh may use an uncompressed or "
                "PS3 vertex layout, which isn't supported yet")
        if h >= 4:
            count = _u32(chunk, h - 4)
            payload = h + 9
            # Structural check: the vertices must fit, and a plausible face count has to
            # follow them. A stray signature-shaped byte run won't satisfy both.
            if 0 < count < 1_000_000 and payload + count * NG_VERTEX_SIZE + 4 <= len(chunk):
                fo = payload + count * NG_VERTEX_SIZE
                fcount = _u32(chunk, fo)
                if 0 < fcount < 1_000_000 and fo + 4 + fcount * 6 <= len(chunk):
                    return count, payload
        start = h + 1


def _find_trans(chunk, limit):
    """(local, world, parent) from a mesh's RndTrans data, or None. The two 4x3 matrices are
    followed by a constraint (int), a target symbol, a preserve-scale flag and the parent's
    name; located by that structure - both matrices' rotation parts orthonormal (or uniformly
    scaled), sane small fields after them, a printable parent name - rather than at a fixed
    offset, since the header in front of them varies."""
    for o in range(4, max(5, min(limit, 400) - 110)):
        try:
            local = struct.unpack_from('>12f', chunk, o)
            world = struct.unpack_from('>12f', chunk, o + 48)
        except struct.error:
            return None
        if not all(abs(x) < 1e5 for x in local + world):
            continue
        ok = True
        for m in (local, world):
            rows = [m[0:3], m[3:6], m[6:9]]
            lens = [sum(x * x for x in r) ** 0.5 for r in rows]
            if not all(0.01 < ln < 100 for ln in lens) or max(lens) - min(lens) > 1e-3 * max(lens):
                ok = False
                break
            for i in range(3):
                for j in range(i + 1, 3):
                    if abs(sum(a * b for a, b in zip(rows[i], rows[j]))) > 1e-3 * lens[i] * lens[j]:
                        ok = False
        if not ok:
            continue
        q = o + 96
        constraint = _u32(chunk, q); q += 4
        if constraint > 16:
            continue
        target, q = _numstring(chunk, q, max_len=128)
        if target is None:
            continue
        if chunk[q] not in (0, 1):
            continue
        q += 1
        parent, q2 = _numstring(chunk, q, max_len=128)
        if parent is None or not all(32 <= ord(ch) < 127 for ch in parent):
            continue
        return local, world, parent
    return None


def parse_mesh_entry(chunk, name):
    """Decodes one Mesh entry into a dict, or raises MeshImportError."""
    version = _u32(chunk, 0)
    if version not in (GDRB_MESH_VERSION, TBRB_MESH_VERSION):
        raise MeshImportError(
            f"RndMesh version {version} (expected {TBRB_MESH_VERSION} for TBRB or "
            f"{GDRB_MESH_VERSION} for GDRB)")

    vcount, vstart = _find_vertex_block(chunk)

    # mat and geomOwner sit immediately before mutable/volume/bsp, which in turn sit
    # 9 bytes before the vertex count. Resolve them by finding the pair of strings that
    # lands exactly on that boundary.
    boundary = vstart - 9 - 4 - 9
    mat_name = geom_owner = ""
    for s in range(8, max(9, boundary)):
        m, o1 = _numstring(chunk, s)
        if m is None:
            continue
        g, o2 = _numstring(chunk, o1)
        if g is None:
            continue
        if o2 == boundary and all(32 <= ord(ch) < 127 for ch in (m + g)):
            mat_name, geom_owner = m, g
            break

    verts = []
    for i in range(vcount):
        o = vstart + i * NG_VERTEX_SIZE
        x, y, z = struct.unpack_from('>fff', chunk, o)
        colour = _u32(chunk, o + 12)
        u, v = struct.unpack_from('>ee', chunk, o + 16)
        nx, ny, nz, _nw = _decode_snorm_10_10_10_2(_u32(chunk, o + 20))
        # tangent at o+24 is decoded by the engine but not used here (see gap #2)
        wx, wy, wz, _w2bit = _decode_unorm_10_10_10_2(_u32(chunk, o + 28))
        b0, b1, b2, b3 = struct.unpack_from('>4B', chunk, o + 32)

        # The fourth weight is DERIVED, not stored: the packed vec4's w field is only two
        # bits, far too coarse to be a real weight, and the format notes say as much. The
        # data agrees - across all 6494 vertices in the retail file, wx+wy+wz never once
        # exceeds 1.0, which is exactly the invariant that has to hold for the remainder
        # to be the missing weight. Taking the stored 2-bit field at face value instead
        # gave weight sums as low as 0.684, i.e. vertices quietly under-skinned.
        ww = max(0.0, 1.0 - (wx + wy + wz))

        # Bone indices are stored in REVERSE order relative to the weight components: the
        # first byte pairs with the w component and the last with x. Confirmed both ways -
        # the format notes annotate them as (w, z, y, x) with a default of [3,2,1,0], and
        # pairing them forward leaves 22 non-zero-weight slots pointing past the end of
        # the mesh's own bone table, while pairing them reversed leaves exactly none.
        bones = (b3, b2, b1, b0)

        verts.append(dict(co=(x, y, z), normal=(nx, ny, nz),
                          # stored V-flipped relative to Blender; see module docstring
                          uv=(u, 1.0 - v), colour=colour,
                          weights=(wx, wy, wz, ww), bones=bones))

    fo = vstart + vcount * NG_VERTEX_SIZE
    fcount = _u32(chunk, fo)
    fo += 4
    faces = [struct.unpack_from('>3H', chunk, fo + i * 6) for i in range(fcount)]
    o = fo + fcount * 6

    group_count = _u32(chunk, o); o += 4 + group_count

    bone_count = _u32(chunk, o); o += 4
    if bone_count > MAX_MESH_BONES:
        raise MeshImportError(
            f"mesh claims {bone_count} bones, which is implausible - the bone table was "
            f"probably mis-located")
    bone_names = []
    bone_mats = []
    for _ in range(bone_count):
        nm, o = _numstring(chunk, o)
        if nm is None:
            raise MeshImportError("malformed bone table")
        # 4x3 row-vector matrix the engine skins through: vertex x this x bone's pose. For
        # nearly every mesh it's the inverse of the bone's rest pose, so it cancels at rest;
        # see apply_bind_corrections for the meshes where it doesn't.
        bone_mats.append(struct.unpack_from('>12f', chunk, o))
        o += 48
        bone_names.append(nm)

    trans = _find_trans(chunk, vstart)
    return dict(name=name, version=version, mat=mat_name, geom_owner=geom_owner,
                verts=verts, faces=faces, bone_names=bone_names, bone_mats=bone_mats,
                local_xfm=trans[0] if trans else None,
                world_xfm=trans[1] if trans else None,
                trans_parent=trans[2] if trans else '',
                is_lod=is_lod_name(name), is_shadow=is_shadow_name(name),
                # TBRB's Blend_* meshes: no bones, no material - region masks a TexBlender
                # uses to place wrinkles, not part of the character's visible geometry.
                is_helper=(not bone_names and not mat_name),
                over_documented_bone_cap=bone_count > DOCUMENTED_MAX_BONES)


# ---------------------------------------------------------------------------------------
# Bind matrices
# ---------------------------------------------------------------------------------------
#
# The engine skins a vertex as  vertex x (the mesh's matrix for that bone) x (the bone's
# current pose). Nearly every mesh stores the inverse of the bone's rest pose there, so at
# rest the two cancel and the stored positions are already in place - which is all the
# importer used to rely on. Not every mesh does: Ringo's teeth (ringo_headhands_long) are
# modelled oversized and shrunk into the mouth by SCALED matrices (row lengths 0.899, 0.945,
# 0.814 on both bone_head and bone_jaw, where every other mesh's are 1.0). Stored as-is they
# float about 13.5 units above the head; skinned through their matrices they land in the
# mouth around the tongue (upper Z 59.5-60.0, lower 59.1-59.6; tongue 59.0-59.7).
#
# So each mesh's vertices are put through  matrix x rest pose  for their bones, which is
# where the game draws them at rest - and is left untouched whenever that's the identity.
# Each bone's rest pose comes from the file itself - the inverse of the unscaled matrix its
# meshes store for that bone - keeping the character consistent with itself. Where the
# meshes disagree, the selected armature's skeleton decides (without one, meshes using that
# bone are left as stored); and for bones no mesh stores unscaled, the armature's rest pose is
# used.
#
# Other cases, checked on retail files. Corrected: Billie Joe's lowest-detail eye is scaled
# like Ringo's teeth (2.60 units off the full-detail eye as stored, 0.03 corrected), and
# George's LOD0 and LOD1 hands are bound up to 3 units off his skeleton (his LOD2 hands match
# it exactly) - corrected when his armature is selected. Left alone: matrices that would MIRROR a mesh (negative determinant) -
# 21st's LOD01 shoes and shadow - since no placement fix mirrors; see apply_bind_corrections.

def _m4(m):
    return [[m[0], m[1], m[2], 0.0], [m[3], m[4], m[5], 0.0],
            [m[6], m[7], m[8], 0.0], [m[9], m[10], m[11], 1.0]]


def _mul4(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def _inv4(m):
    n = 4
    a = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(m)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(a[r][c]))
        if abs(a[p][c]) < 1e-12:
            raise ValueError("singular matrix")
        a[c], a[p] = a[p], a[c]
        pv = a[c][c]
        a[c] = [x / pv for x in a[c]]
        for r in range(n):
            if r != c:
                f = a[r][c]
                a[r] = [x - f * y for x, y in zip(a[r], a[c])]
    return [row[n:] for row in a]


def _is_rigid(m, tol=1e-3):
    rows = [m[0:3], m[3:6], m[6:9]]
    for i in range(3):
        if abs(sum(x * x for x in rows[i]) - 1.0) > tol:
            return False
        for j in range(i + 1, 3):
            if abs(sum(x * y for x, y in zip(rows[i], rows[j]))) > tol:
                return False
    return True


def _det3(m):
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def _is_identity(m, tol=1e-3):
    return all(abs(m[i][j] - (1.0 if i == j else 0.0)) <= tol for i in range(4) for j in range(4))


def _armature_rest(armature_obj, bone_name):
    """A bone's rest pose from the armature, as an engine row-vector matrix."""
    if armature_obj is None:
        return None
    b = armature_obj.data.bones.get(bone_name)
    if b is None:
        return None
    ml = b.matrix_local
    return [[ml[0][0], ml[1][0], ml[2][0], 0.0], [ml[0][1], ml[1][1], ml[2][1], 0.0],
            [ml[0][2], ml[1][2], ml[2][2], 0.0], [ml[0][3], ml[1][3], ml[2][3], 1.0]]


def apply_bind_corrections(meshes, armature_obj=None, rest_lookup=None, extra_rest=None):
    """Moves vertices (and normals) of any mesh whose bone-table matrices don't cancel at
    rest to where the game draws them. Returns [(mesh name, note)] for the log.

    `rest_lookup(bone name)` -> the bone's rest pose as an engine row-vector matrix, or None;
    by default it reads the armature.

    Rest poses: with this character's skeleton (the armature), the skeleton is the reference
    for every bone - the game skins against it. George's full-detail hands show why: they're
    bound to a slightly different skeleton (every hand and finger bone turned 3 degrees, up to
    3 units off), and only his lowest-detail hands match the real one, so trusting the file's
    own majority moved the wrong mesh. Without a skeleton, each bone's rest pose comes from the
    unscaled matrix its meshes store; where they disagree, nothing is guessed and those meshes
    stay as stored. An armature that doesn't match the file (another character's) is ignored.

    Corrections are skipped when they would only nudge a mesh imperceptibly (under 0.05
    units), mirror it (negative determinant: 21st's LOD01 shoes and shadow), or only spin it
    in place (its centre moving under 0.1 units: George's eyeballs, which the skeleton would
    turn 8 and 90 degrees about their centres - the game aims the eyes every frame anyway)."""
    if rest_lookup is None:
        def rest_lookup(name):
            return _armature_rest(armature_obj, name)

    variants = {}
    for m in meshes:
        for name, mat in zip(m.get('bone_names', []), m.get('bone_mats', [])):
            if not _is_rigid(mat):
                continue
            groups = variants.setdefault(name, [])
            for g in groups:
                if max(abs(a - b) for a, b in zip(g[0], mat)) < 1e-3:
                    g[1] += 1
                    break
            else:
                groups.append([mat, 1])

    def _mismatch(mat, sk):
        c = _mul4(_m4(mat), sk)
        return max(abs(c[i][j] - (1.0 if i == j else 0.0)) for i in range(4) for j in range(4))

    notes = []
    # Is the armature this character's skeleton? Compare it with the file's own rest poses.
    skeleton = {}
    for name in variants:
        sk = rest_lookup(name)
        if sk is not None:
            skeleton[name] = sk
    use_skeleton = False
    armature_ok = True                  # nothing to compare against: trust it as a fallback
    if skeleton:
        fits = sorted(min(_mismatch(g[0], skeleton[n]) for g in variants[n]) for n in skeleton)
        use_skeleton = fits[len(fits) // 2] <= 0.5
        armature_ok = use_skeleton
        if not use_skeleton:
            notes.append(("(armature)", f"its rest pose doesn't match this file (median "
                                        f"mismatch {fits[len(fits) // 2]:.2f}) - using the "
                                        f"file's own data instead"))

    rest = {}
    for name, groups in variants.items():
        if use_skeleton and name in skeleton:
            rest[name] = skeleton[name]
            continue
        if len(groups) > 1:
            continue                       # the file disagrees with itself: don't guess
        try:
            rest[name] = _inv4(_m4(groups[0][0]))
        except ValueError:
            pass

    for m in meshes:
        names, mats = m.get('bone_names', []), m.get('bone_mats', [])
        if not names:
            continue
        corr, missing = [], []
        for name, mat in zip(names, mats):
            r = rest.get(name)
            # The armature fills in bones the file has no usable matrix for - but never once
            # it's been rejected as another character's skeleton.
            if r is None and armature_ok and (use_skeleton or name not in variants):
                r = rest_lookup(name)
            # Bones the file defines itself (an outfit's Trans objects) carry their own rest
            # pose: straw's hair bones, which no skeleton has.
            if r is None and extra_rest and name in extra_rest:
                r = extra_rest[name]
            if r is None:
                missing.append(name)
                corr.append(None)
                continue
            corr.append(_mul4(_m4(mat), r))
        if all(c is None or _is_identity(c) for c in corr):
            continue
        mirrored = [n for n, c in zip(names, corr) if c is not None and _det3(c) < 0.0]
        if mirrored:
            notes.append((m['name'], f"bone-table matrices would mirror it ("
                                     f"{', '.join(mirrored[:3])}"
                                     f"{'...' if len(mirrored) > 3 else ''}) - left as stored"))
            continue
        if missing:
            notes.append((m['name'], f"bone-table matrices don't cancel at rest, but no rest "
                                     f"pose is known for {', '.join(missing[:3])}"
                                     f"{'...' if len(missing) > 3 else ''} - left as stored"))
            continue
        # Normals take the inverse transpose of each matrix's 3x3 part (the teeth are
        # scaled unevenly).
        ninv = []
        for c in corr:
            a3 = [row[:3] + [0.0] for row in c[:3]] + [[0.0, 0.0, 0.0, 1.0]]
            ninv.append(_inv4(a3))
        new_cos, new_nrm = [], []
        for v in m['verts']:
            px, py, pz = v['co']
            nx, ny, nz = v['normal']
            acc = [0.0, 0.0, 0.0]
            nacc = [0.0, 0.0, 0.0]
            total = 0.0
            for slot in range(4):
                w = v['weights'][slot]
                bi = v['bones'][slot]
                if w <= WEIGHT_EPSILON or bi >= len(corr):
                    continue
                c, ni = corr[bi], ninv[bi]
                for j in range(3):
                    acc[j] += w * (px * c[0][j] + py * c[1][j] + pz * c[2][j] + c[3][j])
                    # n x (C^-1)^T  ==  sum_k n_k * C^-1[j][k]
                    nacc[j] += w * (nx * ni[j][0] + ny * ni[j][1] + nz * ni[j][2])
                total += w
            if total <= 0.0:
                new_cos.append(v['co'])
                new_nrm.append(v['normal'])
                continue
            ln = sum(x * x for x in nacc) ** 0.5 or 1.0
            new_cos.append(tuple(x / total for x in acc))
            new_nrm.append(tuple(x / ln for x in nacc))
        moves = [sum((a - b) ** 2 for a, b in zip(n, v['co'])) ** 0.5
                 for n, v in zip(new_cos, m['verts'])]
        shift = max(moves) if moves else 0.0
        if shift < 0.05:
            continue                                   # imperceptible
        cnt = len(new_cos)
        c_old = [sum(v['co'][i] for v in m['verts']) / cnt for i in range(3)]
        c_new = [sum(p[i] for p in new_cos) / cnt for i in range(3)]
        if sum((a - b) ** 2 for a, b in zip(c_old, c_new)) ** 0.5 < 0.1:
            notes.append((m['name'], f"bone-table matrices would only spin it in place (up to "
                                     f"{shift:.2f} units at the edge, centre fixed) - left as "
                                     f"stored"))
            continue
        for v, co, n in zip(m['verts'], new_cos, new_nrm):
            v['co'] = co
            v['normal'] = n
        notes.append((m['name'], f"placed through its bone-table matrices (moved up to "
                                 f"{shift:.2f} units)"))
    return notes


def read_trans_bones(filepath):
    """{name: (world 4x4, parent)} for the Trans objects in a milo - bones an outfit defines
    for itself, like straw's four hair bones (bone_hair_l1-01 ...), each parented to
    bone_head with a stored world placement. Those placements are the bones' rest poses."""
    with open(filepath, 'rb') as f:
        body = read_milo_container_body(f.read())
    _rev, _dt, _dn, entries = _read_dir_entries(body)
    spans = _entry_spans(body, len(entries))
    out = {}
    for (etype, ename), (s, e) in zip(entries, spans):
        if etype != 'Trans':
            continue
        tr = _find_trans(body[s:e], e - s)
        if tr is not None:
            out[ename] = (_m4(tr[1]), tr[2])
    return out


def remap_unknown_bones(meshes, trans_bones, armature_obj):
    """Points weights for bones the armature doesn't have at the nearest ancestor it does,
    following the file's own Trans parents. Outfit hair bones aren't on the character's
    skeleton, so without this their vertices would deform with nothing and the hair would
    stay behind when the head moves; this way it moves rigidly with bone_head. Returns
    [(mesh, note)]."""
    if armature_obj is None:
        return []
    bones = armature_obj.data.bones
    notes = []
    for m in meshes:
        remap = {}
        for bn in m.get('bone_names', []):
            if bn in bones:
                continue
            cur, seen = bn, set()
            while cur in trans_bones and cur not in seen:
                seen.add(cur)
                cur = trans_bones[cur][1]
                if cur in bones:
                    remap[bn] = cur
                    break
        if remap:
            m['bone_remap'] = remap
            notes.append((m['name'], "weights for bones the armature lacks moved to their "
                                     "parent: " + ", ".join(f"{a} -> {b}" for a, b in
                                                             sorted(remap.items()))))
    return notes


def place_rigid_meshes(meshes):
    """Meshes with no bones aren't skinned: the game draws them through their own transform,
    hanging from a parent. john_headhands_long's eyeballs are like this - modelled around the
    origin, no bone table, parented to bone_L-eye / bone_R-eye with a stored world placement
    that puts them in the sockets (left eye at X -1.25, Z 65.65). Ignoring the transform left
    them on the floor at the scene's centre. Their vertices are put through the stored world
    transform here, and build_mesh_object binds them to the parent bone. Skinned meshes are
    left alone - the game places those through their bones, not their own transform."""
    notes = []
    for m in meshes:
        w = m.get('world_xfm')
        if m.get('bone_names') or w is None or m.get('is_helper'):
            continue
        M = _m4(w)
        if _is_identity(M, tol=1e-5):
            continue
        rows = [w[0:3], w[3:6], w[6:9]]
        ninv = _inv4([list(r) + [0.0] for r in rows] + [[0.0, 0.0, 0.0, 1.0]])
        for v in m['verts']:
            px, py, pz = v['co']
            nx, ny, nz = v['normal']
            v['co'] = tuple(px * M[0][j] + py * M[1][j] + pz * M[2][j] + M[3][j]
                            for j in range(3))
            n = [nx * ninv[j][0] + ny * ninv[j][1] + nz * ninv[j][2] for j in range(3)]
            ln = sum(x * x for x in n) ** 0.5 or 1.0
            v['normal'] = tuple(x / ln for x in n)
        m['rigid_parent'] = m.get('trans_parent', '')
        notes.append((m['name'], f"no bones - placed by its own transform at "
                                 f"({w[9]:.2f}, {w[10]:.2f}, {w[11]:.2f})"
                                 + (f", parented to {m['rigid_parent']}"
                                    if m['rigid_parent'] else "")))
    return notes


def parse_gdrb_meshes(filepath):
    """Returns (dir_name, [mesh dicts], [(name, reason)] for entries that failed)."""
    with open(filepath, 'rb') as f:
        body = read_milo_container_body(f.read())
    revision, dir_type, dir_name, entries = _read_dir_entries(body)
    spans = _entry_spans(body, len(entries))

    meshes = []
    failed = []
    for (etype, ename), (s, e) in zip(entries, spans):
        if etype != 'Mesh':
            continue
        try:
            meshes.append(parse_mesh_entry(body[s:e], ename))
        except (MeshImportError, struct.error) as ex:
            failed.append((ename, str(ex)))
    return dir_name, meshes, failed


# ---------------------------------------------------------------------------------------
# Blender construction
# ---------------------------------------------------------------------------------------

def _placeholder_material(mat_name):
    """A named stand-in so material slots and assignments survive import. Textures aren't
    read (see gap #1), so this carries no maps - it exists to keep the mapping from mesh
    to milo Mat visible and re-linkable later."""
    if not mat_name:
        mat_name = "milo_unassigned"
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(mat_name)
        mat.use_nodes = True
        mat["milo_material"] = mat_name
    return mat


def build_mesh_object(context, mesh_data, armature_obj=None):
    """Creates one Blender object from a parsed mesh dict."""
    name = mesh_data['name']
    verts = mesh_data['verts']
    faces = mesh_data['faces']

    me = bpy.data.meshes.new(name)
    me.from_pydata([v['co'] for v in verts], [], [list(f) for f in faces])
    me.update()

    uv_layer = me.uv_layers.new(name="UVMap")
    for loop in me.loops:
        uv_layer.data[loop.index].uv = verts[loop.vertex_index]['uv']

    me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    try:
        me.normals_split_custom_set_from_vertices([v['normal'] for v in verts])
    except (RuntimeError, AttributeError) as e:
        # Degenerate or zero-length normals make Blender reject the whole set; the mesh is
        # still perfectly usable with computed normals, so this is a note, not a failure.
        _log(f"    '{name}': custom normals not applied ({e}); using computed normals")

    me.materials.append(_placeholder_material(mesh_data['mat']))

    obj = bpy.data.objects.new(name, me)
    obj["milo_mesh_version"] = mesh_data['version']
    obj["milo_material"] = mesh_data['mat']
    obj["milo_is_lod"] = mesh_data['is_lod']
    obj["milo_is_shadow"] = mesh_data['is_shadow']
    context.collection.objects.link(obj)

    # Vertex groups. Bone indices are local to this mesh's own bone table, so they're
    # resolved through it rather than used directly.
    bone_names = mesh_data['bone_names']
    # Bones the armature lacks but the file parents to one it has (outfit hair bones hang
    # from bone_head) hand their weights to that parent; see remap_unknown_bones. Several
    # table entries can then share a group, so groups are keyed - and weights totalled - by
    # destination name.
    remap = mesh_data.get('bone_remap', {})
    dest = [remap.get(bn, bn) for bn in bone_names]
    groups = {}
    for d in dest:
        if d not in groups:
            groups[d] = obj.vertex_groups.get(d) or obj.vertex_groups.new(name=d)
    dropped = 0
    for vi, v in enumerate(verts):
        # Total the weight per bone BEFORE writing it. The same bone often appears in more
        # than one of a vertex's four slots, and the engine's skinning adds every slot's
        # contribution, so a bone listed twice is meant to get the sum. Writing each slot
        # with 'REPLACE' kept only the last one instead: on billiejoe's brow mesh that hit
        # 29 of 160 vertices and threw away 0.39 of their weight on average, so those
        # vertices moved about 40% less than their neighbours - the sawtooth seen on the
        # brows whenever a viseme moved them. The head mesh had 63 such vertices, losing
        # up to 0.95.
        per_bone = {}
        for slot in range(4):
            w = v['weights'][slot]
            if w <= WEIGHT_EPSILON:
                continue
            bi = v['bones'][slot]
            if bi >= len(bone_names):
                # Real weight aimed at a bone this mesh doesn't list - that would be a
                # decode error rather than rounding, so it's worth surfacing.
                dropped += 1
                continue
            per_bone[dest[bi]] = per_bone.get(dest[bi], 0.0) + w
        for d, w in per_bone.items():
            groups[d].add([vi], w, 'REPLACE')
    if dropped:
        _log(f"    '{name}': {dropped} weight(s) referenced a bone outside this mesh's "
             f"bone table and were skipped")

    rigid_parent = mesh_data.get('rigid_parent')
    if rigid_parent and armature_obj is not None and rigid_parent in armature_obj.data.bones:
        # A bone-less mesh hangs from its parent bone: give every vertex full weight to it
        # so it follows that bone like the skinned meshes follow theirs.
        vg = obj.vertex_groups.get(rigid_parent) or obj.vertex_groups.new(name=rigid_parent)
        vg.add(list(range(len(verts))), 1.0, 'REPLACE')
    elif rigid_parent and armature_obj is not None:
        _log(f"    '{name}': its parent '{rigid_parent}' isn't on the armature, so it won't "
             f"follow any bone")

    if armature_obj is not None:
        obj.parent = armature_obj
        mod = obj.modifiers.new(name="Armature", type='ARMATURE')
        mod.object = armature_obj

    return obj


# ---------------------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------------------

class _IMPORT_OT_milo_meshes_base(bpy.types.Operator, ImportHelper):
    """Shared mesh-import logic for the revision-25 games. Not registered itself - GDRB and
    TBRB each subclass it, setting only their id, label and _game_label."""
    bl_options = {'REGISTER', 'UNDO'}
    _game_label = "Milo"

    filename_ext = ".milo_xbox"
    filter_glob: StringProperty(
        default="*.milo_xbox;*.milo_ps3;*.milo", options={'HIDDEN'})

    exclude_lod: BoolProperty(
        name="Exclude LOD Meshes",
        description="Skip the reduced-detail copies (names containing LOD01, LOD02 and "
                     "so on). A retail head milo is mostly LODs - 14 of its 21 meshes - "
                     "so leaving this on keeps only the full-detail originals",
        default=True,
    )

    exclude_shadow: BoolProperty(
        name="Exclude Shadow Mesh",
        description="Skip the flat shadow-caster blob an outfit milo ships alongside the "
                     "real geometry. It spans the character's full height at very low "
                     "density and isn't body geometry",
        default=True,
    )

    import_textures: BoolProperty(
        name="Import Textures",
        description="Decode the milo's textures and build a simple node tree for each "
                     "material: diffuse to Base Color, normal map through a Normal Map node, "
                     "specular to Specular Tint",
        default=True,
    )

    flip_normal_green: BoolProperty(
        name="Flip Normal Map Green",
        description="Convert the normal maps' Y from DirectX (down) to Blender's OpenGL "
                     "(up). The retail maps test as DirectX; turn this off only if bumps "
                     "light up as dents",
        default=True,
    )

    exclude_helpers: BoolProperty(
        name="Exclude Helper Meshes",
        description="Skip meshes with no bones and no material. In TBRB these are the "
                     "Blend_* patches a TexBlender uses to place facial wrinkles; they "
                     "aren't visible geometry and would float over the face",
        default=True,
    )

    parent_to_armature: BoolProperty(
        name="Parent To Active Armature",
        description="Parent the imported meshes to the selected armature and add an "
                     "Armature modifier. Vertex groups are created either way; this only "
                     "hooks them up to a rig",
        default=True,
    )

    def execute(self, context):
        armature_obj = None
        active = context.active_object
        if self.parent_to_armature and active is not None and active.type == 'ARMATURE':
            armature_obj = active

        try:
            dir_name, meshes, failed = parse_gdrb_meshes(self.filepath)
        except Exception as e:
            _log(f"{self._game_label} MESH IMPORT FAILED: {e}")
            self.report({'ERROR'}, f"Could not parse mesh milo: {e}")
            return {'CANCELLED'}

        if not meshes and not failed:
            self.report({'ERROR'},
                        "No Mesh entries found - is this a character/mesh milo? "
                        "(A skeleton milo holds only Trans and CharCollide entries.)")
            return {'CANCELLED'}

        _log(f"===== Importing {self._game_label} meshes from '{dir_name}' "
             f"({self.filepath}) =====")
        try:
            trans_bones = read_trans_bones(self.filepath)
        except Exception as e:
            _log(f"  couldn't read the file's own bones ({e}) - carrying on without them")
            trans_bones = {}
        for name, note in place_rigid_meshes(meshes):
            _log(f"  '{name}': {note}")
        for name, note in apply_bind_corrections(
                meshes, armature_obj, extra_rest={n: t[0] for n, t in trans_bones.items()}):
            _log(f"  '{name}': {note}")
        for name, note in remap_unknown_bones(meshes, trans_bones, armature_obj):
            _log(f"  '{name}': {note}")
        lods = [m for m in meshes if m['is_lod']]
        shadows = [m for m in meshes if m['is_shadow']]
        _log(f"  {len(meshes)} mesh(es) parsed, {len(lods)} LOD, {len(shadows)} shadow"
             + (f", {len(failed)} failed" if failed else ""))

        heavy = [m for m in meshes if m.get('over_documented_bone_cap')]
        if heavy:
            _log(f"  {len(heavy)} mesh(es) exceed the documented {DOCUMENTED_MAX_BONES}-bone "
                 f"cap (max {max(len(m['bone_names']) for m in heavy)}): "
                 f"{', '.join(m['name'] for m in heavy[:4])} - imported normally.")

        wanted = [m for m in meshes
                  if not (self.exclude_lod and m['is_lod'])
                  and not (self.exclude_shadow and m['is_shadow'])
                  and not (self.exclude_helpers and m.get('is_helper'))]
        helpers = [m for m in meshes if m.get('is_helper')]
        if helpers and self.exclude_helpers:
            _log(f"  {len(helpers)} helper mesh(es) skipped (no bones, no material): "
                 f"{', '.join(m['name'] for m in helpers)}")

        # A mesh that exists ONLY as a LOD has no full-detail counterpart, so excluding
        # LODs removes that piece of the character entirely rather than just downgrading
        # it. Worth saying out loud - it looks like missing geometry otherwise.
        if self.exclude_lod:
            kept_stems = {_LOD_RE.sub('', m['name']).replace('_.', '.').lower()
                          for m in wanted}
            orphan_lods = []
            for m in lods:
                stem = _LOD_RE.sub('', m['name']).replace('_.', '.').lower()
                if stem not in kept_stems:
                    orphan_lods.append(m['name'])
            if orphan_lods:
                _log(f"  NOTE: {len(orphan_lods)} mesh(es) exist only as a LOD, so "
                     f"excluding LODs drops them completely: {', '.join(orphan_lods)}")
        created = 0
        total_v = total_f = 0
        unmatched_bones = set()
        arm_bones = set(armature_obj.data.bones.keys()) if armature_obj else set()

        for m in wanted:
            build_mesh_object(context, m, armature_obj)
            created += 1
            total_v += len(m['verts'])
            total_f += len(m['faces'])
            _log(f"    {m['name']:34s} {len(m['verts']):5d}v {len(m['faces']):5d}f  "
                 f"{len(m['bone_names']):2d} bone(s)  mat='{m['mat']}'")
            if arm_bones:
                unmatched_bones.update(b for b in m['bone_names'] if b not in arm_bones)

        for nm, reason in failed:
            _log(f"    SKIPPED {nm}: {reason}")

        tex_summary = ""
        if self.import_textures:
            # Imported here rather than at module level: texture_importer itself uses this
            # module's directory helpers.
            from .texture_importer import apply_textures
            wanted_mats = sorted({m['mat'] for m in wanted if m['mat']})
            try:
                done, n_images, tex_notes = apply_textures(
                    self.filepath, wanted_mats, self.flip_normal_green)
                _log(f"  textures: {done} of {len(wanted_mats)} material(s) textured, "
                     f"{n_images} image(s) decoded")
                for n in tex_notes:
                    _log(f"    {n}")
                tex_summary = f"; {done} material(s) textured"
            except Exception as e:
                _log(f"  textures: could not be imported ({e})")
                tex_summary = "; textures failed (see log)"

        if unmatched_bones:
            sample = ', '.join(sorted(unmatched_bones)[:8])
            _log(f"  {len(unmatched_bones)} vertex group(s) name bones the armature "
                 f"doesn't have (e.g. {sample}) - those groups will have no effect until "
                 f"the matching skeleton is imported.")

        summary = (f"Imported {created} mesh(es) from '{dir_name}': "
                   f"{total_v} verts, {total_f} faces"
                   + (f"; {len(lods)} LOD skipped" if self.exclude_lod and lods else "")
                   + (f"; {len(shadows)} shadow skipped"
                      if self.exclude_shadow and shadows else "")
                   + (f"; {len(failed)} failed (see log)" if failed else "")
                   + tex_summary)
        if armature_obj is not None:
            summary += f"; parented to '{armature_obj.name}'"
        _log(f"===== {summary} =====")
        self.report({'WARNING' if failed else 'INFO'}, summary)
        return {'FINISHED'}


class IMPORT_OT_gdrb_meshes(_IMPORT_OT_milo_meshes_base):
    """Import the meshes from a Green Day: Rock Band character milo, with UVs, per-bone
    vertex groups and textured materials"""
    bl_idname = "import_scene.gdrb_meshes"
    bl_label = "Import GDRB Meshes Milo"
    _game_label = "GDRB"


class IMPORT_OT_tbrb_meshes(_IMPORT_OT_milo_meshes_base):
    """Import the meshes from a The Beatles: Rock Band character milo, with UVs, per-bone
    vertex groups and textured materials.

    TBRB is one revision behind GDRB on every object it uses - RndMesh 36, RndTex 10,
    RndMat 55 against 37, 11 and 56 - with the same vertex layout, so it shares GDRB's
    reader. Verified on two retail Xbox 360 milos (george_headhands_long, straw): all 44
    meshes parse with unit normals, weights summing to exactly 1.0 and no stray bone slots.
    What differs is handled where it's read: '_lod00' naming (see is_lod_name), helper
    meshes, and a render-target head normal map (see texture_importer)."""
    bl_idname = "import_scene.tbrb_meshes"
    bl_label = "Import TBRB Meshes Milo"
    _game_label = "TBRB"
