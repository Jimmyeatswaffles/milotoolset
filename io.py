import bpy
import struct

from .utilities import (
    MiloWriter, write_object_fields, write_rnd_trans, write_matrix, sanitize_milo_name,
    _log, RB3_MILO_REVISION, DC3_MILO_REVISION, MAX_MILO_BLOCK_SIZE, END_MARKER,
    write_rnd_dir,
)
from .model_exporter import write_rnd_mesh
from .texture_exporter import write_rnd_mat, write_dc3_rnd_mat, write_rnd_tex
from .physics_exporter import (
    write_char_hair, write_char_collide, write_outfit_config, write_char_mesh_hide,
    COLL_SHAPE_TO_INT,
)
from .skeleton_exporter import write_bone_transform, write_character


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


MILO_END_MARKER = b'\xAD\xDE\xAD\xDE'


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


def _milo_to_blender_matrix(m12):
    """Inverse of _matrix_to_milo: Milo row-vector 4x3 -> Blender column-vector Matrix."""
    from mathutils import Matrix
    return Matrix((
        (m12[0], m12[3], m12[6], m12[9]),
        (m12[1], m12[4], m12[7], m12[10]),
        (m12[2], m12[5], m12[8], m12[11]),
        (0.0,    0.0,    0.0,    1.0),
    ))


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


