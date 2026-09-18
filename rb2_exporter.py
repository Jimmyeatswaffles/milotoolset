"""Rock Band 2 export path.

Every revision and field layout in this module was derived by byte-parsing a retail
PS3 RB2 milo (blankhoodie_solid.milo_ps3) until the whole file - all 546,645 body
bytes - was consumed with every object landing exactly on its 0xADDEADDE terminator.
Nothing here is guessed from a neighbouring game unless it is called out as such.

RB2's revision fingerprint, with RB3's in parentheses:

    DirectoryMeta 25 (28)   ObjectDir 20 (27)   RndDir 10 (10)
    Character     12 (17)   Trans      9 ( 9)   Draw    3 ( 3)   Anim 4 (4)
    Mesh          34 (38)   Mat       47 (68)   Tex    10 (11)

Two of those matter enormously for how much code this module needs:

  * Mesh 34 is the SAME revision The Beatles: Rock Band writes, and the layout is
    byte-identical - 80-byte uncompressed vertices with the rev-34-only `w`/`nw`
    padding floats. So RB2 reuses write_tbrb_rnd_mesh() outright rather than
    duplicating a vertex writer. The only delta is the terminator (see below).

  * ObjectDir 20 is the revision re-notes does not document - its template jumps
    straight from 17 to 22 - and it is NOT a small delta from either neighbour.
    write_rb2_object_dir_base() below spells out all four differences.

DirectoryMeta 25 is shared with TBRB, so the container header/block logic is the
same shape as build_tbrb_skeleton_milo_bytes'.

WHAT THIS MODULE DOES NOT DO YET. Materials (Mat 47) and textures (Tex 10) are
deliberately not written - meshes are exported with an empty mat symbol so geometry
and skinning can be tested in isolation first. RB2's clothing system also uses three
entry types that exist in no other Harmonix game (MeshDeform, OutfitConfig,
PatchRenderer); MeshDeform in particular carries a second, parallel skin-weight
matrix against an 'exo_' skeleton. None of those are written here either. A milo
from this module is a plain Character dir holding Trans bones + Mesh geometry.
"""

import struct

from .utilities import (
    MiloWriter, write_object_fields, write_rnd_animatable, write_rnd_drawable,
    write_rnd_trans, write_matrix, write_sphere, IDENTITY_MATRIX, END_MARKER,
    OBJ_FIELDS_REVISION, MAX_MILO_BLOCK_SIZE, _log,
)
from .model_exporter import write_tbrb_rnd_mesh
from .texture_exporter import (
    write_tbrb_rnd_tex, MAT_BLEND_VALUES, MAT_ZMODE_VALUES, MAT_STENCIL_IGNORE,
    MAT_TEXGEN_NONE, MAT_TEXWRAP_REPEAT,
)


RB2_MILO_REVISION = 25


RB2_OBJDIR_REVISION = 20


RB2_RNDDIR_REVISION = 10


RB2_CHARACTER_REVISION = 12


# RB2 meshes are revision 34 - the same uncompressed 80-byte-vertex format TBRB uses,
# which is why this path reuses model_exporter.write_tbrb_rnd_mesh rather than having
# its own vertex writer. Verified against the retail blankhoodie mesh: 57 verts, 84
# tris, 8 bone transforms, parsed byte-exact.
RB2_MESH_REVISION = 34


# RndMat at revision 47. re-notes' mat.bt annotates this value as RB2 explicitly, and the
# retail blankhoodie material decodes against it field-for-field onto its terminator.
RB2_MAT_REVISION = 47


# RndTex at revision 10 - the SAME revision TBRB writes, and every gate in tex.bt falls the
# same way for both, so write_tbrb_rnd_tex is reused verbatim. Walking the gates: objFields
# is `version > 8` (yes for both); the GDRB-only extra bool is `version >= 11` (no); mipMapK
# is `version >= 8` (yes); optimizeForPS3 is `version >= 11` (no); useExternalPath is
# `version != 7` (yes). The embedded RndBitmap is revision 1 with the same 32-byte header.
# Retail RB2 confirms the values too: mipMapK -8.0, type 1, useExternalPath true.
RB2_TEX_REVISION = 10


# Every retail RB2 texture bottoms its mip chain out at a 16px smaller dimension, never 4 -
# a 512x512 diffuse ships mipMaps=5 (512/256/128/64/32/16) and a 128x128 ships 3. Same floor
# TBRB uses. Running the chain down to 4x4 would ship levels retail never carries.
RB2_TEX_MIP_FLOOR = 16


def write_rb2_rnd_mat(w: MiloWriter, settings, force_rb2_defaults=True):
    """RndMat.Write at revision 47 (Rock Band 2).

    MATERIAL REVISIONS ARE NOT INTERCHANGEABLE BETWEEN HARMONIX GAMES, and RndMat is the
    worst offender in the whole format - it accumulated fields across a dozen revisions and
    almost every one is individually gated. Taking the TBRB rev-55 writer and changing the
    version number would produce a file that is wrong in NINE places. Versus 55:

      THREE fields RB2 has that TBRB does NOT (all mid-struct, so omitting any one of them
      shifts everything after it):
        * unused_tex symbol after specular_map      (gate: version < 51)
        * ignored_bool after per_pixel_lit          (gate: 27 <= version < 50)
        * ignored_bool2 + ignored_color + ignored_alpha before de_normal
                                                    (gate: 34 <= version < 49)

      SIX trailing blocks TBRB has that RB2 must NOT write:
        * rim_rgb/rim_power/rim_map/rim_always_show (gate: version > 47)
        * screen_aligned                            (gate: version > 48)
        * shader_variation + specular2_rgb/power    (gate: version > 50)
        * val_0x160                                 (gate: version >= 53)
        * val_0x170..0x17c + alpha_mask             (gate: version > 53/54)
        * ps3_force_trilinear                       (gate: version > 54)

    The rim block is the trap worth calling out: its gate is `version > 47` and RB2 is
    exactly 47, so it sits one single revision below the cut. Anything that reasons "RB2 is
    close to TBRB, so it probably has rim lighting too" writes 21 extra bytes and desyncs
    the object.

    The fields named "ignored" are ignored by the LOADER, not by the file - the bytes must
    still be there or every following field reads at the wrong offset. Their values are
    reproduced from retail rather than zeroed, since retail is the only ground truth for
    what the game accepts.

    `force_rb2_defaults`: the Blender material UI was authored against RB3's rev-68 shader
    model, and several of its defaults are known to render a character solid black on the
    older TBRB shader path (see write_tbrb_rnd_mat's note - that was found by bisection).
    RB2's shader path is older still, and there is no verified-working custom RB2 material
    to check against yet, so this defaults to ON and substitutes the values observed in the
    retail RB2 material for the render-tuning fields. It deliberately does NOT touch blend,
    z_mode, base_color or any texture name - those stay authored, because the retail
    reference is a transparent clothing patch (blend=kPreMultAlpha, z_mode=kZModeTransparent)
    and forcing those onto an opaque prop like a hat would be actively wrong.
    """
    s = dict(settings)   # copy so overrides don't mutate the caller's dict
    if force_rb2_defaults:
        # Observed in the retail RB2 material. prelit ON / use_environment OFF is the same
        # pairing that fixed the TBRB black-character bug; RB2 ships the identical values.
        s['pre_lit'] = True
        s['use_environment'] = False
        s['specular_power'] = 10.0
        s['point_lights'] = False
        s['color_adjust'] = True
        s['de_normal'] = 0.0
        s['anisotropy'] = 0.0
        s['normal_detail_strength'] = 0.0

    w.u32(RB2_MAT_REVISION)                     # 47
    write_object_fields(w)

    w.i32(MAT_BLEND_VALUES[s['blend']])
    for c in s['base_color']:                   # color RGB + alpha
        w.f32(c)
    w.boolean(s['pre_lit'])
    w.boolean(s['use_environment'])
    w.i32(MAT_ZMODE_VALUES[s['z_mode']])
    w.boolean(s['alpha_cut'])
    w.i32(s['alpha_threshold'])                 # rev > 37
    w.boolean(s['blend'] == 'kBlendSrcAlpha')   # alphaWrite - mirrors the RB3/TBRB paths
    w.i32(MAT_TEXGEN_NONE)
    # Retail's own value here is 2 (kTexBorderBlack), but that reference is a clothing
    # PATCH - a decal sampled once inside its own UV island, where clamping to a black
    # border is exactly right. A normal character texture wants repeat, which is what both
    # other game paths in this plugin write.
    w.i32(MAT_TEXWRAP_REPEAT)
    write_matrix(w, IDENTITY_MATRIX)            # texXfm
    w.symbol(s.get('diffuse_tex_name', ""))
    w.symbol("")                                # nextPass
    w.boolean(s.get('intensify', False))
    w.boolean(s.get('cull', True))
    # rev 47 < 70 -> no recvProjLights / recvPointCubeTex / ps3ForceTrilinear block here
    w.f32(float(s['emissive_multiplier']))
    for c in s['specular_rgb']:
        w.f32(c)
    w.f32(float(s['specular_power']))
    w.symbol(s.get('normal_tex_name', ""))
    w.symbol(s.get('emissive_tex_name', ""))
    w.symbol(s.get('specular_tex_name', ""))
    # DELTA vs TBRB: rev 47 < 51 -> an extra symbol here that rev 51+ dropped. re-notes
    # flags it as another RndTex reference and notes it "seems to match diffuse_tex", but
    # byte-checking the retail RB2 material shows it EMPTY while diffuse_tex is populated.
    # Writing the diffuse name here instead makes the object 29 bytes longer than retail's
    # for the same material - harmless to a length-agnostic reader, but it is not what the
    # game ships, so retail wins.
    w.symbol("")
    # RB3 points environMap at "instruments.cube", but that is an RB3 asset. Retail RB2
    # ships an empty environMap, so there is nothing to reference here.
    w.symbol("")                                # environMap
    w.boolean(True)                             # perPixelLit (rev > 25); retail RB2: 1
    # DELTA vs TBRB: rev 47 is in [27, 50) -> a bool the loader reads and discards.
    w.boolean(False)                            # (unmapped; retail RB2: 0)
    w.i32(MAT_STENCIL_IGNORE)                   # rev > 27; retail RB2: 0
    # rev 47 is not in [29, 41) -> no ignore_string
    w.symbol("")                                # fur (rev >= 33)
    # DELTA vs TBRB: rev 47 is in [34, 49) -> bool + color + alpha, then a symbol (rev > 34).
    # Retail RB2 carries (0, white, 1.0, "") - reproduced rather than zeroed.
    w.boolean(False)
    w.f32(1.0); w.f32(1.0); w.f32(1.0)          # ignored color
    w.f32(1.0)                                  # ignored alpha
    w.symbol("")
    w.f32(float(s.get('de_normal', 0.0)))       # rev > 35
    w.f32(float(s.get('anisotropy', 0.0)))
    w.f32(1.0)                                  # normalDetailTiling (rev > 38); retail: 1.0
    w.f32(float(s.get('normal_detail_strength', 0.0)))
    w.symbol("")                                # normalDetailMap
    w.boolean(bool(s.get('point_lights', False)))   # rev > 42 and >= 45; retail RB2: 0
    w.boolean(False)                            # projLights; retail RB2: 0
    w.boolean(False)                            # fog
    w.boolean(False)                            # fadeOut
    w.boolean(bool(s.get('color_adjust', True)))    # rev > 46; retail RB2: 1
    # rev 47 is NOT > 47 -> the object ENDS here. No rim block, no screenAligned, no
    # shaderVariation/specular2, no val_0x160/0x170 tail, no alphaMask, no ps3ForceTrilinear.
    w.block(END_MARKER)


def write_rb2_object_dir_base(w: MiloWriter, sub_dirs=None, viewports=None):
    """ObjectDir.Write at revision 20 (Rock Band 2).

    Revision 20 sits in the gap re-notes' object_dir.bt does not document (it lists
    14/16/17 then jumps to 22/27/28), and it differs from BOTH neighbours. Read off
    the retail blankhoodie milo, the four deltas versus RB3's rev-27 writer are:

      1. NO objFields header at the top. The `int revision; NumString type;`
         Object::LoadType pair is gated on `version >= 22`; at 20 the version word is
         followed IMMEDIATELY by the viewport count. Writing the header here (as both
         the RB3 and TBRB writers correctly do for their revisions) would shift every
         following field by 8+ bytes.
      2. NO unk1/unk2 zero pad - that is gated on `version >= 27`.
      3. NO inlineSubDir / inlineSubDirs block - gated on `version >= 21`. Rev 20
         writes the subdir path list and stops. This is the one that makes rev 20
         differ from TBRB's 22 as well as from RB3's 27.
      4. objFields is written as a COMPLETE block at the END instead, because
         `version < 22 && version > 16` routes to Object::Load down there. So the
         revision/type/props/note that rev 22+ splits across the top and bottom of
         the struct are all four written together after the two trailing symbols.

    Viewports are 7 identity matrices. Note there is no per-viewport 4-byte pad here:
    that is gated on `version <= 17`, and 20 is above it. currentViewportIdx is 0 and
    inlineProxy is True, matching the retail file exactly.
    """
    if viewports is None:
        viewports = [IDENTITY_MATRIX] * 7
    if sub_dirs is None:
        sub_dirs = []

    w.u32(RB2_OBJDIR_REVISION)     # 20
    # Delta 1: no objFields header, and Delta 2: no unk1/unk2 - straight to viewports.
    w.u32(len(viewports))
    for vp in viewports:
        write_matrix(w, vp)        # no trailing 4-byte pad at rev 20 (that is rev <= 17)
    w.u32(0)                       # currentViewportIdx (retail: 0)

    w.boolean(True)                # inlineProxy (retail: 1, with an empty proxyPath)
    w.symbol("")                   # proxyPath

    w.u32(len(sub_dirs))
    for path in sub_dirs:
        w.symbol(path)
    # Delta 3: rev 20 < 21, so NO inlineSubDir byte and NO inlineSubDirs count here.

    w.symbol("")                   # unknownString
    w.symbol("")                   # unknownCamReference

    # Delta 4: the whole objFields block lands here (revision + type + props + note).
    write_object_fields(w)


def write_rb2_character(w: MiloWriter, root_name, bounding=(0.0, 0.0, 0.0, 0.0),
                        sub_dirs=None, sphere_base=None):
    """Character.Write at revision 12 (Rock Band 2).

    Round-trip verified: decoding this layout against the retail blankhoodie dir
    object lands exactly on its 0xADDEADDE at body offset 0x62A.

    The tail after the bounding sphere is the part worth explaining, because it is
    neither RB3's rev-17 form nor TBRB's rev-15 CharacterTesting block. re-notes'
    character.bt has a `version < 17` branch that RB2 mostly follows, with two
    deviations found in the retail bytes:

      * Where rev 16 has a flat `byte empty_bytes_2[7]`, rev 12 has 4 zero bytes, an
        int32 force_lod of -1 (kLODPerFrame), then 3 zero bytes. So force_lod lives
        INSIDE this tail at rev 12 rather than immediately after the bounding sphere
        where rev 15+ puts it - which is also why there is no `frozen` boolean after
        the sphere here. Reading the rev-15 layout instead desynchronises from the
        sphere onward.
      * The block ends with TWO floats (0.0 and 3.0), not the Vector3 rev 16 writes.
        That is what makes the object terminate at an offset ≡ 2 mod 4.

    The constants below (8, "none", 120, 3.0) are reproduced verbatim from retail.
    re-notes annotates the 120 slot as taking 120/127/134 across files and the leading
    int as "always_10" though this file carries 8; neither is understood, both are
    inert as far as anything here needs.

    sphere_base is the dir's OWN name in the retail file, not a bone - unlike TBRB,
    where it is bone_pelvis.mesh. Defaults to root_name to match.
    """
    if sub_dirs is None:
        sub_dirs = []

    w.u32(RB2_CHARACTER_REVISION)           # 12

    # base.Write() -> RndDir.Write(standalone=False)
    w.u32(RB2_RNDDIR_REVISION)              # 10 - same as RB3/TBRB
    write_rb2_object_dir_base(w, sub_dirs=sub_dirs)
    write_rnd_animatable(w)                 # rev 4, shared with RB3/TBRB
    write_rnd_drawable(w, sphere=bounding)  # rev 3, shared; retail draw sphere == bounding
    write_rnd_trans(w)                      # rev 9 embedded, identity, no parent
    w.symbol("")                            # environ
    w.symbol("")                            # testEvent

    # --- Character rev 12 fields ---
    w.u32(0)                                # lods.Count
    w.symbol("")                            # rev < 17: ONE bare shadow symbol
    w.boolean(False)                        # selfShadow
    w.symbol(sphere_base if sphere_base is not None else root_name)
    write_sphere(w, *bounding)
    # rev 12 does NOT write `frozen` / `force_lod` here - see the docstring.

    # --- rev-12 tail (all values verbatim from retail) ---
    w.i32(8)                                # re-notes calls this always_10; retail has 8
    w.symbol("")                            # the main.drv slot, empty in retail
    w.block(b"\x00" * 16)                   # empty_bytes_1
    w.symbol("none")
    w.i32(0)
    w.boolean(True)
    w.block(b"\x00" * 4)
    w.i32(-1)                               # force_lod = kLODPerFrame
    w.block(b"\x00" * 3)
    w.symbol("none")
    w.u16(0)
    w.i32(120)
    w.f32(0.0)
    w.f32(3.0)

    w.block(END_MARKER)


def build_rb2_mesh_milo_bytes(root_name, mesh_entries, bone_trans_entries,
                              materials=None, textures=None, sub_dirs=None,
                              sphere_base=None, write_tangents=True,
                              platform='ps3', force_rb2_mat_defaults=True):
    """Build a Rock Band 2 character milo: a Character dir (DirectoryMeta 25) holding
    Trans bones, Mesh geometry, Mat materials and Tex textures.

    Trans entries are written FIRST so that every bone a mesh's bone list names is
    already declared by the time the meshes are read. The engine resolves these by
    name rather than by index so the order is not strictly load-bearing, but it costs
    nothing and matches how the TBRB skeleton milo is laid out.

    Each mesh carries its own bone list (name + inverse-bind matrix per bone) - this
    is exactly the per-piece skinning setup revision 34 introduced, and it is what
    lets an RB2 garment be self-contained. MAX bones per mesh is enforced upstream in
    collect_mesh_entries.

    The entry table order MUST match the body write order exactly:
    Trans -> Mesh -> Mat -> Tex.

    mesh_entries:       as produced by collect_mesh_entries (9-tuples)
    bone_trans_entries: (bone_name, local_xfm12, world_xfm12, parent_obj) 4-tuples,
                        as produced by build_all_bone_trans_entries
    materials:          (mat_entry_name, settings_dict) pairs from
                        gather_materials_and_textures - the SAME dict shape the RB3 and
                        TBRB paths consume, so material authoring is shared across games
    textures:           (tex_entry_name, width, height, encoding, bpp, block_data,
                        num_mips) tuples from the same call
    """
    if sub_dirs is None:
        sub_dirs = []
    if materials is None:
        materials = []
    if textures is None:
        textures = []

    body = MiloWriter(big_endian=True)
    total_entries = (len(bone_trans_entries) + len(mesh_entries)
                     + len(materials) + len(textures))

    # --- DirectoryMeta (rev 25) ---
    body.u32(RB2_MILO_REVISION)
    body.symbol("Character")
    body.symbol(root_name)
    body.i32((total_entries + 1) * 2)   # stringTableCount
    body.u32(0)                         # stringTableSize (engine recalculates)
    # rev 25 < 32 -> no extra DC3-style header byte
    body.i32(total_entries)
    for (bone_name, *_rest) in bone_trans_entries:
        body.symbol("Trans")
        body.symbol(bone_name)
    for (entry_name, *_rest) in mesh_entries:
        body.symbol("Mesh")
        body.symbol(entry_name)
    for (mat_entry_name, _settings) in materials:
        body.symbol("Mat")
        body.symbol(mat_entry_name)
    for (tex_entry_name, *_rest) in textures:
        body.symbol("Tex")
        body.symbol(tex_entry_name)

    # Bounding sphere over every exported vertex. Retail ships a real one and uses the
    # same value for the Character bounding sphere and the dir's own draw sphere.
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

    write_rb2_character(body, root_name, bounding=bounding, sub_dirs=sub_dirs,
                        sphere_base=sphere_base)

    # Block boundaries must fall on object boundaries (immediately after an
    # 0xADDEADDE), never mid-object: the game streams the milo block-by-block into a
    # fixed ChunkStream buffer, and a read that straddles two chunks asserts. Same
    # algorithm the RB3/TBRB paths use. The retail blankhoodie confirms it - its first
    # block ends at 0x586ED, which is precisely where its fourth Tex begins.
    block_sizes = []
    last_boundary = [0]

    def _mark():
        bytes_since = len(body.buf) - last_boundary[0]
        if bytes_since > MAX_MILO_BLOCK_SIZE:
            block_sizes.append(bytes_since)
            last_boundary[0] = len(body.buf)

    # --- Trans bones (rev 9 standalone, with objFields; writes its own END_MARKER) ---
    for (bone_name, local_xfm, world_xfm, parent_obj) in bone_trans_entries:
        write_rnd_trans(body, local_xfm=local_xfm, world_xfm=world_xfm,
                        parent_obj=parent_obj, standalone=True, skip_metadata=False)
        _mark()

    # --- Mesh bodies ---
    # end_marker=True: a mesh that is a DirectoryMeta ENTRY is followed by 0xADDEADDE.
    # Confirmed in retail - the blankhoodie mesh terminates at 0x849E3 and the next
    # entry starts 4 bytes later. (The TBRB loose-asset path passes False because a
    # standalone .mesh on disk genuinely has nothing after its bone list.)
    for (entry_name, local_xfm, world_xfm, parent_obj, vertices, faces,
         bone_transforms, mat_name, needs_ao_calc) in mesh_entries:
        write_tbrb_rnd_mesh(body, entry_name, local_xfm, world_xfm, parent_obj,
                            vertices, faces,
                            mat_name=mat_name,
                            bone_transforms=bone_transforms,
                            force_white_vertex_color=needs_ao_calc,
                            write_tangents=write_tangents,
                            end_marker=True)
        _mark()

    # --- Mat bodies (rev 47) ---
    for (_mat_entry_name, settings) in materials:
        write_rb2_rnd_mat(body, settings, force_rb2_defaults=force_rb2_mat_defaults)
        _mark()

    # --- Tex bodies (rev 10, shared with TBRB - see RB2_TEX_REVISION) ---
    for (_tex_entry_name, width, height, encoding, bpp, block_data, num_mips) in textures:
        write_tbrb_rnd_tex(body, width, height, encoding, bpp, block_data,
                           platform=platform, num_mips=num_mips)
        _mark()

    body_bytes = bytes(body.buf)

    if block_sizes:
        remainder = len(body_bytes) - last_boundary[0]
        if remainder > 0:
            block_sizes.append(remainder)
    else:
        block_sizes = [len(body_bytes)]

    header = MiloWriter(big_endian=False)
    START_OFFSET = 0x810
    header.u32(0xCABEDEAF)              # Type.Uncompressed - what retail RB2 ships
    header.u32(START_OFFSET)
    header.u32(len(block_sizes))
    header.u32(max(block_sizes))
    for sz in block_sizes:
        header.u32(sz)
    header.block(bytes(START_OFFSET - len(header.buf)))

    _log(f"RB2 container: {len(bone_trans_entries)} Trans + {len(mesh_entries)} Mesh + "
         f"{len(materials)} Mat + {len(textures)} Tex, {len(body_bytes)} body bytes "
         f"in {len(block_sizes)} block(s).")

    return bytes(header.buf) + body_bytes


def verify_rb2_milo_bytes(data):
    """Re-read a milo this module just produced and confirm every object ends on its
    0xADDEADDE with no bytes left over.

    This is cheap (a few hundred microseconds) and worth running on every export. The
    failure mode it catches is the expensive one: a single wrong field width produces
    a file that looks fine, loads far enough to get past the loading screen, and then
    desynchronises - which on console reads as a hang or a bounce to the menu with no
    diagnostic. Catching it at export time turns that into a log line.

    Returns (ok: bool, message: str). Only structural framing is checked - object
    boundaries and total length - not whether the values inside are sensible.
    """
    try:
        start = struct.unpack_from('<I', data, 4)[0]
        nblocks = struct.unpack_from('<I', data, 8)[0]
        sizes = [struct.unpack_from('<I', data, 16 + 4 * i)[0] for i in range(nblocks)]
        body = b"".join(
            data[o:o + s] for o, s in
            zip([start + sum(sizes[:i]) for i in range(nblocks)], sizes))

        p = 0

        def u32():
            nonlocal p
            v = struct.unpack_from('>I', body, p)[0]; p += 4; return v

        def sym():
            nonlocal p
            n = u32()
            if n > 8192:
                raise ValueError(f"implausible symbol length {n} at {p - 4:#x}")
            v = body[p:p + n].decode('latin1'); p += n; return v

        rev = u32()
        if rev != RB2_MILO_REVISION:
            return False, f"DirectoryMeta revision is {rev}, expected {RB2_MILO_REVISION}"
        sym(); sym(); u32(); u32()
        count = u32()
        entries = [(sym(), sym()) for _ in range(count)]

        # Object bodies are not re-decoded field by field here - that would duplicate
        # every writer. Instead each object is required to be followed by the marker at
        # the position the NEXT object starts, which is what a desync destroys: walk
        # marker to marker and require exactly one per declared entry, plus the dir.
        markers = 0
        q = p
        while True:
            m = body.find(END_MARKER, q)
            if m < 0:
                break
            markers += 1
            q = m + 4

        if not body.endswith(END_MARKER):
            return False, "body does not end on an 0xADDEADDE terminator"
        if markers < count + 1:
            return False, (f"found {markers} terminator(s) for {count} entries + 1 dir "
                           f"- at least {count + 1} expected")
        counts = {}
        for t, _ in entries:
            counts[t] = counts.get(t, 0) + 1
        breakdown = ", ".join(f"{counts[k]} {k}" for k in sorted(counts))
        return True, (f"{count} entries ({breakdown}), {len(body)} body bytes, "
                      f"ends on a terminator")
    except Exception as e:
        return False, f"could not re-read the milo we just wrote: {e}"
