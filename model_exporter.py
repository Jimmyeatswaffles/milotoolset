import bpy
import struct

from .utilities import (
    MiloWriter, write_object_fields, write_rnd_animatable, write_rnd_drawable,
    write_rnd_trans, write_matrix, write_sphere, write_color3, write_color4,
    pack_signed_10_10_10_2, pack_unsigned_10_10_10_2, pack_ps3_signed_11_11_10,
    pack_ps3_unsigned_11_11_10, IDENTITY_MATRIX, END_MARKER, sanitize_milo_name,
    _matrix_to_milo, BoneLimitExceeded, _find_armature_modifier,
    _bone_axis_correction_matrix, _resolve_stock_reference_armature, _log,
    RB3_TRANS_REVISION, RB3_ANIM_REVISION, RB3_DRAW_REVISION, DC3_DRAW_REVISION,
    OBJ_FIELDS_REVISION, MAX_MILO_BLOCK_SIZE, MAX_BONES_PER_MESH,
    TBRB_MILO_REVISION, TBRB_OBJDIR_REVISION, TBRB_RNDDIR_REVISION,
)
from .texture_exporter import (
    write_tbrb_rnd_mat, write_tbrb_rnd_tex,
)
from .skeleton_exporter import (
    write_tbrb_character, write_bone_transform,
)


RB3_MESH_REVISION = 38


TBRB_MESH_REVISION = 36


TBRB_GROUP_REVISION = 14


TBRB_MESH_WRITE_REVISION = 34


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


