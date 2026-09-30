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


def is_lod_name(name):
    """True for the reduced-detail copies. Matched on the name because the retail file
    ships 'billiejoel_tongueLOD02.mesh' - a typo'd stem that no base-name pairing would
    catch, but which the LOD suffix still identifies correctly."""
    return bool(_LOD_RE.search(name))


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
    for _ in range(bone_count):
        nm, o = _numstring(chunk, o)
        if nm is None:
            raise MeshImportError("malformed bone table")
        o += 48                       # 4x3 transform matrix, unused for import
        bone_names.append(nm)

    return dict(name=name, version=version, mat=mat_name, geom_owner=geom_owner,
                verts=verts, faces=faces, bone_names=bone_names,
                is_lod=is_lod_name(name), is_shadow=is_shadow_name(name),
                over_documented_bone_cap=bone_count > DOCUMENTED_MAX_BONES)


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
    groups = {}
    for bn in bone_names:
        groups[bn] = obj.vertex_groups.new(name=bn)
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
            per_bone[bi] = per_bone.get(bi, 0.0) + w
        for bi, w in per_bone.items():
            groups[bone_names[bi]].add([vi], w, 'REPLACE')
    if dropped:
        _log(f"    '{name}': {dropped} weight(s) referenced a bone outside this mesh's "
             f"bone table and were skipped")

    if armature_obj is not None:
        obj.parent = armature_obj
        mod = obj.modifiers.new(name="Armature", type='ARMATURE')
        mod.object = armature_obj

    return obj


# ---------------------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------------------

class IMPORT_OT_gdrb_meshes(bpy.types.Operator, ImportHelper):
    """Import the meshes from a Green Day: Rock Band character milo, with UVs, per-bone
    vertex groups and placeholder materials. Textures are not imported"""
    bl_idname = "import_scene.gdrb_meshes"
    bl_label = "Import GDRB Meshes Milo"
    bl_options = {'REGISTER', 'UNDO'}

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
            _log(f"GDRB MESH IMPORT FAILED: {e}")
            self.report({'ERROR'}, f"Could not parse mesh milo: {e}")
            return {'CANCELLED'}

        if not meshes and not failed:
            self.report({'ERROR'},
                        "No Mesh entries found - is this a character/mesh milo? "
                        "(A skeleton milo holds only Trans and CharCollide entries.)")
            return {'CANCELLED'}

        _log(f"===== Importing GDRB meshes from '{dir_name}' ({self.filepath}) =====")
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
                  and not (self.exclude_shadow and m['is_shadow'])]

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
                   + (f"; {len(failed)} failed (see log)" if failed else ""))
        if armature_obj is not None:
            summary += f"; parented to '{armature_obj.name}'"
        _log(f"===== {summary} =====")
        self.report({'WARNING' if failed else 'INFO'}, summary)
        return {'FINISHED'}
