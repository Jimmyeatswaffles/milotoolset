"""Guitar Hero 2 (Xbox 360) export path.

Everything here is the mirror image of the GH2 IMPORT path in io.py
(parse_gh2_skeleton / parse_gh2_meshes / _try_parse_gh2_mesh_body), and every
revision and field layout was derived by byte-parsing a retail GH2 X360 milo
(goth2.milo_xbox) rather than assumed from a neighbouring game. Where a value is
reused from another game's writer it is called out as such.

The one thing that makes this path unlike every other exporter in this addon:

    THE OBJECT BODY IS LITTLE-ENDIAN.

RB3/DC1/DC3/TBRB/RB2 all write big-endian object bodies on Xbox/PS3. GH2 X360 does
not - confirmed on import, where the top-level DirectoryMeta only parses to sane
values (revision 25, type "BandCharacter", 321 entries) when read LE. So every
MiloWriter in this module is constructed with big_endian=False. The outer
compression/block header was ALREADY little-endian in every game, so that part is
unchanged.

GH2's revision fingerprint, with RB3's in parentheses:

    DirectoryMeta  25 (28)    ObjectDir 17 (27)    RndDir  9 (10)
    BandCharacter   1 (n/a)   Character 10 (17)    Trans   9 ( 9)
    Draw            3 ( 3)    Mesh      28 (38)    Mat    28 (68)   Tex 10 (11)

Three of those need explaining because they change how much code this module needs:

  * ObjectDir 17 is the TOP of re-notes' documented 14/16/17 range, and unlike RB2's
    rev 20 it DOES carry the per-viewport 4-byte pad (that field is gated on
    `version <= 17`). Confirmed empirically rather than assumed: decoding the retail
    dir body with a 48-byte viewport stride yields only 2 of 7 viewports with
    unit-length rotation rows, while a 52-byte stride yields 7 of 7.

  * Trans 9 is the same revision number RB3 writes AND, unusually, the same byte
    layout - just little-endian. (An early assumption during the import work that
    GH2's rev-9 Trans differed structurally was wrong; a byte-exact test proved it
    identical.) So bone writing is genuinely "the RB3 layout, endian-flipped".

  * Mesh 28 is NOT close to any other supported game. Vertices are 48 bytes of
    PLAIN float32 - no packed 10:10:10:2 normals, no float16 UVs, no separate bone
    INDEX bytes - and the per-mesh bone palette is a fixed FOUR slots. See
    write_gh2_rnd_mesh and GH2_MAX_BONES_PER_MESH.

WHAT THIS MODULE DOES NOT DO YET. Materials (Mat 28) and textures (Tex 10) are
deliberately not written - meshes are exported referencing a material NAME only, so
geometry and skinning can be validated in isolation first, exactly as the RB2 path
did at the same stage. A milo from this module is a BandCharacter dir holding Trans
bones + Mesh geometry.

THE DIRECTORY BODY IS TAKEN FROM A DONOR. The ~390 bytes of BandCharacter config
that follow the viewport list are character BEHAVIOUR data, not geometry: the retail
bytes contain gameplay strings like "guitarist" and "in_solo". Those are not
derivable from a Blender scene, and fabricating them risks a character that loads but
misbehaves in ways that would be very hard to trace back here. So this module lifts
the directory body verbatim out of a donor GH2 milo (the same approach io.py already
takes for CharClipSet, and for the same reason) and writes fresh Trans/Mesh entries
around it. That is a real limitation, stated plainly: you need one retail GH2
character milo on hand to export a custom one.
"""

import struct

from .utilities import (
    MiloWriter, write_object_fields, write_matrix,
    MAX_MILO_BLOCK_SIZE, END_MARKER, _log,
)


GH2_MILO_REVISION = 25            # DirectoryMeta, type "BandCharacter"
GH2_DIR_TYPE = "BandCharacter"

GH2_TRANS_REVISION = 9            # same number AND layout as RB3, just LE
GH2_MESH_REVISION = 28
GH2_DRAW_REVISION = 3             # shared with RB3

# objFields revision. RB3/TBRB/RB2 all write 2 (utilities.OBJ_FIELDS_REVISION); every
# GH2 object in the retail file carries 1. The value is load-bearing: the note symbol
# is gated on `revision > 0`, so a 0 here would drop a field, and the import path
# reads this same word back to decide whether to expect one.
GH2_OBJ_FIELDS_REVISION = 1

# A GH2 mesh's bone palette is exactly FOUR slots, always written, with unused slots
# carrying an EMPTY symbol (and still a full 48-byte matrix). Confirmed arithmetically
# across the retail file: the trailing block after the bone-name list measures
# 192 + 4*(4 - names_present) bytes in every single mesh, i.e. 4 * 48 bytes of matrix
# plus 4 bytes for each unused slot's zero-length symbol. This is also exactly WHY a
# GH2 character ships as ~140 small mesh chunks instead of a few big ones, and why
# this exporter has to split by bone count (see split_mesh_by_bone_limit).
GH2_MAX_BONES_PER_MESH = 4

# Per-vertex size: 12 float32s = position(3) + normal(3) + weights(4) + uv(2).
GH2_VERTEX_SIZE = 48


class GH2DonorError(Exception):
    """Raised when a donor GH2 milo can't supply a usable BandCharacter directory body."""


def _d_u32(data, p):
    return struct.unpack_from('<I', data, p)[0]


def _d_symbol(data, p):
    n = _d_u32(data, p)
    return data[p + 4:p + 4 + n].decode('latin1'), p + 4 + n


def read_milo_body_le(data):
    """Concatenate a GH2 milo's block payloads into one object-body byte string.

    Identical in shape to io.py's read_milo_container_body - the container header is
    little-endian in every supported game - but reproduced here so the export path
    doesn't import from the import path just for this.
    """
    magic = _d_u32(data, 0)
    if magic != 0xCABEDEAF:
        raise GH2DonorError(
            f"Donor's container magic is {magic:#x}, not the uncompressed 0xCABEDEAF "
            f"that retail GH2 X360 characters ship as.")
    start_offset = _d_u32(data, 4)
    num_blocks = _d_u32(data, 8)
    off = start_offset
    body = bytearray()
    for i in range(num_blocks):
        size = _d_u32(data, 16 + 4 * i) & 0xFFFFFF
        body += data[off:off + size]
        off += size
    return bytes(body)


def extract_gh2_dir_body(donor_path):
    """Lift the BandCharacter directory body out of a donor GH2 milo, verbatim.

    Returns (dir_type, dir_name, dir_body_bytes). dir_body_bytes spans from just after
    the donor's entry table up to and INCLUDING its first 0xADDEADDE terminator - i.e.
    the complete BandCharacter object, viewports and character config and all.

    See this module's docstring for why this is lifted rather than synthesized. The
    bytes are game config, not geometry, so they transplant cleanly onto a different
    character: nothing inside them is a per-vertex or per-bone reference. The one
    thing that IS name-bound - the directory's own name - lives in the DirectoryMeta
    header this module writes fresh, not in here.
    """
    with open(donor_path, 'rb') as f:
        data = f.read()
    body = read_milo_body_le(data)

    try:
        p = 0
        revision = _d_u32(body, p); p += 4
        dir_type, p = _d_symbol(body, p)
        dir_name, p = _d_symbol(body, p)
        p += 8                                   # stringTableCount + stringTableSize
        entry_count = _d_u32(body, p); p += 4
        if entry_count > 100000:
            raise GH2DonorError(
                f"Donor entry count {entry_count} is implausible - the body may not be "
                f"little-endian, so this probably isn't a GH2 X360 milo.")
        for _ in range(entry_count):
            _t, p = _d_symbol(body, p)
            _n, p = _d_symbol(body, p)
    except GH2DonorError:
        raise
    except Exception as e:
        raise GH2DonorError(f"Couldn't parse the donor's directory header: {e}")

    if revision != GH2_MILO_REVISION or dir_type != GH2_DIR_TYPE:
        raise GH2DonorError(
            f"Donor is DirectoryMeta revision {revision} type '{dir_type}', but a GH2 "
            f"character milo should be revision {GH2_MILO_REVISION} type "
            f"'{GH2_DIR_TYPE}'. Pick a GH2 Xbox 360 character milo as the donor.")

    end = body.find(END_MARKER, p)
    if end < 0:
        raise GH2DonorError(
            "Couldn't find the end of the donor's BandCharacter directory body "
            "(no 0xADDEADDE terminator after its entry table).")
    return dir_type, dir_name, body[p:end + 4]


def write_gh2_object_fields(w):
    """objFields at GH2's revision 1 - see GH2_OBJ_FIELDS_REVISION."""
    write_object_fields(w, revision=GH2_OBJ_FIELDS_REVISION)


def write_gh2_rnd_trans(w, local_xfm, world_xfm, parent_name="",
                        standalone=True):
    """RndTrans at revision 9.

    Byte-for-byte the RB3 rev-9 layout, just written little-endian - see this module's
    docstring. `standalone=True` writes the objFields header and the trailing
    0xADDEADDE that a Trans which is its own directory ENTRY carries; embedded copies
    (inside a Mesh) pass False, which matches how the import path reads them back.
    """
    w.u32(GH2_TRANS_REVISION)
    if standalone:
        write_gh2_object_fields(w)
    write_matrix(w, local_xfm)
    write_matrix(w, world_xfm)
    w.u32(0)                 # constraint = kConstraintNone
    w.symbol("")             # target
    w.boolean(False)         # preserveScale
    w.symbol(parent_name)
    if standalone:
        w.block(END_MARKER)


def write_gh2_rnd_mesh(w, entry_name, local_xfm, world_xfm, parent_name,
                       vertices, faces, bone_palette, mat_name=""):
    """RndMesh at revision 28.

    Layout mirrors _try_parse_gh2_mesh_body in io.py exactly (that function is the
    reader for what this writes), so the two stay in lockstep:

        u32 revision(28) | objFields | embedded RndTrans(9) | RndDrawable(3)
        | symbol mat | symbol geomOwner | u32 mutable | u32 volume | u8 bspNode
        | u32 vertexCount | vertices... | u32 faceCount | faces(u16 x3)...
        | u32 groupCount | group sizes (u8 each)
        | FOUR bone symbols | FOUR bone matrices | END_MARKER

    vertices: dicts with x,y,z, nx,ny,nz, u,v and a `weights` list of
              (bone_name, weight) pairs - the same shape parse_gh2_meshes RETURNS, so
              an imported mesh round-trips through here without reshaping.
    bone_palette: list of up to GH2_MAX_BONES_PER_MESH (bone_name, inv_bind_xfm12)
              pairs. Slots beyond what's supplied are padded with an empty symbol and
              an identity matrix - which is exactly what retail does (see
              GH2_MAX_BONES_PER_MESH).

    Each vertex's four weight floats are written POSITIONALLY against that palette:
    slot i's float is this vertex's weight for bone_palette[i]. That positional
    mapping (rather than a separate bone-index array, which this format simply does
    not have) is why the palette is capped at four and why meshes must be split.
    """
    w.u32(GH2_MESH_REVISION)
    write_gh2_object_fields(w)

    # --- embedded RndTrans (not standalone: no objFields, no terminator) ---
    write_gh2_rnd_trans(w, local_xfm, world_xfm, parent_name, standalone=False)

    # --- RndDrawable rev 3 (shared shape with RB3, LE here) ---
    w.u32(GH2_DRAW_REVISION)
    w.boolean(True)          # showing
    w.f32(0.0); w.f32(0.0); w.f32(0.0); w.f32(0.0)   # bounding sphere
    w.f32(0.0)               # drawOrder
    # rev 3 < 4, so no trailing drawable symbol (that gate is the DC3-only one).

    w.symbol(mat_name)
    w.symbol(entry_name)     # geomOwner - retail always names the mesh ITSELF here
                             # (verified: 143/143 entries in goth2.milo_xbox match)
    w.u32(0)                 # mutable
    w.u32(1)                 # volume (retail value for character meshes)
    w.u8(0)                  # bspNode.hasValue

    # --- vertices: 12 plain float32s each, no packing anywhere ---
    slot_of = {name: i for i, (name, _m) in enumerate(bone_palette)}
    w.u32(len(vertices))
    for v in vertices:
        w.f32(v["x"]); w.f32(v["y"]); w.f32(v["z"])
        w.f32(v["nx"]); w.f32(v["ny"]); w.f32(v["nz"])
        slots = [0.0, 0.0, 0.0, 0.0]
        for bone_name, weight in v.get("weights", ()):
            i = slot_of.get(bone_name)
            if i is not None:
                slots[i] += weight
        w.f32(slots[0]); w.f32(slots[1]); w.f32(slots[2]); w.f32(slots[3])
        w.f32(v["u"]); w.f32(v["v"])

    # --- faces ---
    w.u32(len(faces))
    for (i0, i1, i2) in faces:
        w.u16(i0); w.u16(i1); w.u16(i2)

    # --- face groups. A List<byte>, so each entry caps at 255; retail splits a
    # 254-face mesh as [91, 92, 71] and a 121-face one as [104, 17], i.e. the exact
    # chunking isn't meaningful beyond "sums to the face count". ---
    remaining = len(faces)
    groups = []
    while remaining > 0:
        take = min(remaining, 255)
        groups.append(take)
        remaining -= take
    w.u32(len(groups))
    for g in groups:
        w.u8(g)

    # --- bone palette: FOUR names, then FOUR matrices (see GH2_MAX_BONES_PER_MESH) ---
    if len(bone_palette) > GH2_MAX_BONES_PER_MESH:
        raise ValueError(
            f"Mesh '{entry_name}' has {len(bone_palette)} bones but GH2 meshes hold at "
            f"most {GH2_MAX_BONES_PER_MESH}. It should have been split upstream by "
            f"split_mesh_by_bone_limit.")
    for i in range(GH2_MAX_BONES_PER_MESH):
        w.symbol(bone_palette[i][0] if i < len(bone_palette) else "")
    for i in range(GH2_MAX_BONES_PER_MESH):
        if i < len(bone_palette):
            write_matrix(w, bone_palette[i][1])
        else:
            write_matrix(w, (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
                             0.0, 0.0, 0.0))

    w.block(END_MARKER)


def split_mesh_by_bone_limit(vertices, faces, max_bones=GH2_MAX_BONES_PER_MESH):
    """Split one mesh's geometry into chunks that each reference at most `max_bones`
    distinct bones, returning a list of (chunk_vertices, chunk_faces, bone_names).

    This is the step that turns a normal Blender character into GH2-shaped data, and
    it is not optional: the format has no per-vertex bone-index array, so a vertex's
    four weight floats are positional against a four-slot palette (see
    write_gh2_rnd_mesh). A whole-body mesh weighted to 60 bones cannot be expressed as
    one GH2 mesh at any quality setting - retail ships goth2 as ~140 chunks for exactly
    this reason, which is what the IMPORT path's family-merging undoes for the user.
    Exporting simply has to redo it.

    The algorithm is fill-greedy: open a chunk, seed it with the first unassigned
    face, then repeatedly sweep the remaining unassigned faces and absorb every one
    whose bones still fit inside the chunk's four-bone budget, preferring faces that
    add the fewest NEW bones so the budget is spent slowly. Close the chunk when
    nothing else fits and open the next.

    The obvious simpler version - sort faces by bone set, then walk once and flush
    whenever the running union would overflow - was tried first and is much worse: on
    a mesh whose faces don't already share bone sets it degenerates to roughly one
    face per chunk, because a single face pulling in a fourth bone forces a flush that
    throws away the three bones already paid for. The sweep avoids that by filling
    each chunk from the whole remaining pool instead of only from what comes next in
    order.

    Greedy rather than optimal because the optimal partition (fewest chunks) is a
    set-cover-flavoured problem; chunk COUNT has no correctness consequence here, only
    the per-chunk bone cap does, so a cheap good-enough packing is the right trade.

    Vertices are re-indexed per chunk, and a vertex used by faces in two different
    chunks is duplicated into both - unavoidable, and exactly what retail's chunk
    boundaries show.

    A single face needing more than max_bones distinct bones cannot be placed at all;
    those are returned in their own chunk with the bone set TRUNCATED to the
    highest-weighted max_bones, and the caller is expected to warn. Truncating loses a
    little skinning fidelity on that face but keeps the mesh exportable, which is the
    better failure mode than dropping geometry silently.
    """
    def face_bones(f):
        s = set()
        for vi in f:
            for bone_name, weight in vertices[vi].get("weights", ()):
                if weight > 1e-4:
                    s.add(bone_name)
        return s

    annotated = []
    for f in faces:
        fb = face_bones(f)
        if len(fb) > max_bones:
            # Keep the heaviest contributors for this face - see docstring.
            totals = {}
            for vi in f:
                for bone_name, weight in vertices[vi].get("weights", ()):
                    totals[bone_name] = totals.get(bone_name, 0.0) + weight
            fb = set(sorted(totals, key=lambda b: -totals[b])[:max_bones])
        annotated.append((f, fb))

    chunks = []
    unassigned = list(range(len(annotated)))
    taken = [False] * len(annotated)

    while True:
        seed = next((i for i in unassigned if not taken[i]), None)
        if seed is None:
            break
        taken[seed] = True
        cur_faces = [annotated[seed][0]]
        cur_bones = set(annotated[seed][1])

        # Sweep the remaining pool, cheapest-first, until nothing else fits. Repeat
        # the sweep after each successful absorb only in the sense that `cur_bones`
        # grows - one pass ordered by added-cost is enough in practice and keeps this
        # near-linear per chunk rather than quadratic per face.
        while len(cur_bones) < max_bones:
            best = None
            best_cost = None
            for i in unassigned:
                if taken[i]:
                    continue
                fb = annotated[i][1]
                cost = len(fb - cur_bones)
                if len(cur_bones | fb) > max_bones:
                    continue
                if best_cost is None or cost < best_cost:
                    best, best_cost = i, cost
                    if cost == 0:
                        break
            if best is None:
                break
            taken[best] = True
            cur_faces.append(annotated[best][0])
            cur_bones |= annotated[best][1]

        # Anything that needs no new bones at all can still be swept up for free.
        for i in unassigned:
            if not taken[i] and annotated[i][1] <= cur_bones:
                taken[i] = True
                cur_faces.append(annotated[i][0])

        unassigned = [i for i in unassigned if not taken[i]]
        chunks.append((cur_faces, cur_bones))

    out = []
    for cfaces, cbones in chunks:
        bone_names = sorted(cbones)
        remap = {}
        cverts = []
        newfaces = []
        for f in cfaces:
            nf = []
            for vi in f:
                if vi not in remap:
                    remap[vi] = len(cverts)
                    src = vertices[vi]
                    # Drop weights pointing outside this chunk's palette, then
                    # renormalize so the remaining weights still sum to 1 - otherwise
                    # a vertex whose bone got truncated away would deform toward the
                    # origin instead of staying put.
                    kept = [(b, wt) for (b, wt) in src.get("weights", ())
                            if b in cbones and wt > 1e-4]
                    total = sum(wt for _b, wt in kept)
                    if total > 1e-6:
                        kept = [(b, wt / total) for (b, wt) in kept]
                    elif bone_names:
                        kept = [(bone_names[0], 1.0)]
                    cverts.append({**src, "weights": kept})
                nf.append(remap[vi])
            newfaces.append(tuple(nf))
        out.append((cverts, newfaces, bone_names))
    return out


def build_gh2_character_milo_bytes(root_name, mesh_entries, bone_trans_entries,
                                   dir_body):
    """Build a GH2 Xbox 360 character milo: a BandCharacter dir (DirectoryMeta 25,
    LITTLE-ENDIAN body) holding Trans bones and Mesh geometry.

    Trans entries are written first, then Mesh - and the entry TABLE order must match
    the body write order exactly, same invariant as every other exporter here. The
    import path's sequential walker also depends on it (it maps entry N to the Nth
    0xADDEADDE), so a mismatch would break round-tripping as well as the game.

    root_name:          the character/directory name (e.g. "mycharacter")
    mesh_entries:       (entry_name, local_xfm, world_xfm, parent_name, vertices,
                        faces, bone_palette, mat_name) 8-tuples
    bone_trans_entries: (bone_name, local_xfm, world_xfm, parent_name) 4-tuples
    dir_body:           verbatim BandCharacter body from extract_gh2_dir_body
    """
    body = MiloWriter(big_endian=False)     # <-- the GH2 difference; see module docstring
    total_entries = len(bone_trans_entries) + len(mesh_entries)

    # --- DirectoryMeta (rev 25) ---
    body.u32(GH2_MILO_REVISION)
    body.symbol(GH2_DIR_TYPE)
    body.symbol(root_name)
    stc, sts = gh2_string_table_hint(
        GH2_DIR_TYPE, root_name,
        [("Trans", e[0]) for e in bone_trans_entries]
        + [("Mesh", e[0]) for e in mesh_entries])
    body.i32(stc)                       # stringTableCount - see gh2_string_table_hint
    body.u32(sts)                       # stringTableSize (never 0: breaks Milo Editor)
    body.i32(total_entries)
    for (bone_name, *_rest) in bone_trans_entries:
        body.symbol("Trans")
        body.symbol(bone_name)
    for (entry_name, *_rest) in mesh_entries:
        body.symbol("Mesh")
        body.symbol(entry_name)

    # --- BandCharacter directory body, lifted from the donor ---
    body.block(dir_body)

    # Block boundaries must land on object boundaries (right after an 0xADDEADDE),
    # never mid-object - the game streams blocks into a fixed buffer and a straddling
    # read asserts. Same algorithm the RB2/RB3/TBRB paths use.
    block_sizes = []
    last_boundary = [0]

    def _mark():
        bytes_since = len(body.buf) - last_boundary[0]
        if bytes_since > MAX_MILO_BLOCK_SIZE:
            block_sizes.append(bytes_since)
            last_boundary[0] = len(body.buf)

    _mark()

    for (bone_name, local_xfm, world_xfm, parent_name) in bone_trans_entries:
        write_gh2_rnd_trans(body, local_xfm, world_xfm, parent_name, standalone=True)
        _mark()

    for (entry_name, local_xfm, world_xfm, parent_name, vertices, faces,
         bone_palette, mat_name) in mesh_entries:
        write_gh2_rnd_mesh(body, entry_name, local_xfm, world_xfm, parent_name,
                           vertices, faces, bone_palette, mat_name=mat_name)
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
    header.u32(0xCABEDEAF)                  # Type.Uncompressed - what retail GH2 ships
    header.u32(START_OFFSET)
    header.u32(len(block_sizes))
    header.u32(max(block_sizes))
    for sz in block_sizes:
        header.u32(sz)
    header.block(bytes(START_OFFSET - len(header.buf)))

    _log(f"GH2 container: {len(bone_trans_entries)} Trans + {len(mesh_entries)} Mesh, "
         f"{len(body_bytes)} body bytes in {len(block_sizes)} block(s).")

    return bytes(header.buf) + body_bytes


def verify_gh2_milo_bytes(data):
    """Re-parse a just-built GH2 milo and confirm every declared entry lands exactly on
    its own 0xADDEADDE terminator, with nothing left over.

    This is a self-check against the class of bug that is hardest to notice otherwise:
    a field written at the wrong width or endianness still produces a file, and the
    game just fails to load it with no useful diagnostic. Returns (ok, message).
    """
    try:
        body = read_milo_body_le(data)
        p = 0
        revision = _d_u32(body, p); p += 4
        dir_type, p = _d_symbol(body, p)
        dir_name, p = _d_symbol(body, p)
        p += 8
        entry_count = _d_u32(body, p); p += 4
        entries = []
        for _ in range(entry_count):
            t, p = _d_symbol(body, p)
            n, p = _d_symbol(body, p)
            entries.append((t, n))

        markers = []
        sf = p
        while True:
            m = body.find(END_MARKER, sf)
            if m < 0:
                break
            markers.append(m)
            sf = m + 4

        expected = entry_count + 1          # one per entry, plus the directory body
        if len(markers) != expected:
            return False, (f"Expected {expected} object terminators "
                           f"({entry_count} entries + 1 directory), found "
                           f"{len(markers)}.")
        if markers[-1] + 4 != len(body):
            return False, (f"Last terminator ends at {markers[-1] + 4} but the body is "
                           f"{len(body)} bytes - trailing data after the final object.")
        return True, (f"OK: revision {revision}, type '{dir_type}', name '{dir_name}', "
                      f"{entry_count} entries, all landing on their terminators, "
                      f"{len(body)} body bytes.")
    except Exception as e:
        return False, f"Verification failed while re-parsing: {e}"


def gh2_string_table_hint(dir_type, dir_name, entries):
    """Return (stringTableCount, stringTableSize) for a DirectoryMeta header.

    These two fields were being written as `(entries+1)*2` and `0`. The GAME accepts
    that - every milo exported that way loads and plays - but Milo Editor refuses to
    open the file, which cost a debugging session to track down. Retail never ships a
    zero here: goth2 carries 674 / 5800.

    Neither field is an exact, derivable quantity. Checked against retail goth2: the
    count is ~2x the number of unique symbols in its entry table (343 unique -> 674)
    and the size sits just above their combined byte length (5787 -> 5800). Scanning
    every symbol in the whole file instead gives 449 unique / 6683 bytes, which
    brackets the real values from the other side. So these read as PREALLOCATION HINTS
    for the engine's symbol pool, carrying deliberate headroom, rather than a checksum.

    That shapes the safe failure mode: over-estimating just reserves memory that goes
    unused, while under-estimating (and certainly zero) is what breaks a reader that
    trusts the number. So this computes the entry table's unique symbols and applies
    the same ~2x / +slack headroom retail uses, rather than trying to reproduce an
    exact value that isn't exact in the source material either.
    """
    uniq = {dir_type, dir_name}
    for (t, n) in entries:
        uniq.add(t)
        uniq.add(n)
    count = len(uniq) * 2
    size = sum(len(s) + 1 for s in uniq) + 64
    return count, size


def _parse_donor_entries(body):
    """Split a GH2 milo body into (dir_body, [(type, name, start, end), ...]).

    Entry N's body runs from the previous object's terminator to its own, which is the
    same marker-order assumption the import path uses - and which holds because the
    entry table order and the body write order are the same thing in this format.
    """
    p = 0
    _rev = _d_u32(body, p); p += 4
    _t, p = _d_symbol(body, p)
    _n, p = _d_symbol(body, p)
    p += 8
    entry_count = _d_u32(body, p); p += 4
    decl = []
    for _ in range(entry_count):
        t, p = _d_symbol(body, p)
        n, p = _d_symbol(body, p)
        decl.append((t, n))
    table_end = p

    markers = []
    sf = table_end
    while True:
        m = body.find(END_MARKER, sf)
        if m < 0:
            break
        markers.append(m)
        sf = m + 4
    if len(markers) < entry_count + 1:
        raise GH2DonorError(
            f"Donor has {entry_count} entries but only {len(markers)} object "
            f"terminators - can't map entries to bodies.")

    dir_body = body[table_end:markers[0] + 4]
    spans = []
    cur = markers[0] + 4
    for i, (t, n) in enumerate(decl):
        end = markers[i + 1] + 4
        spans.append((t, n, cur, end))
        cur = end
    return dir_body, spans


def _parse_donor_mesh_header(body, s):
    """Read just the reusable header fields off a donor mesh: returns
    (local_xfm12, world_xfm12, parent_name, mat_name).

    Injection keeps every one of these from the donor rather than inventing them, so
    the rebuilt mesh stays wired into the donor's own material list, bone parenting
    and LOD groups exactly as it was. Only the geometry underneath is swapped.
    """
    q = s
    q += 4                                   # mesh revision
    combined = _d_u32(body, q); q += 4
    tl = _d_u32(body, q); q += 4 + tl
    q += 1
    if combined & 0xFFFF > 0:
        nl = _d_u32(body, q); q += 4 + nl
    q += 4                                   # embedded trans revision
    local = struct.unpack_from('<12f', body, q); q += 48
    world = struct.unpack_from('<12f', body, q); q += 48
    q += 4                                   # constraint
    tl2 = _d_u32(body, q); q += 4 + tl2      # target
    q += 1                                   # preserveScale
    pl = _d_u32(body, q); q += 4
    parent = body[q:q + pl].decode('latin1'); q += pl
    draw_rev = _d_u32(body, q); q += 4
    q += 1 + 16 + 4                          # showing + sphere + drawOrder
    if draw_rev >= 4:
        dl = _d_u32(body, q); q += 4 + dl
    ml = _d_u32(body, q); q += 4
    mat = body[q:q + ml].decode('latin1')
    return local, world, parent, mat


def build_gh2_injected_milo_bytes(donor_path, chunks, target_family=None,
                                  lod_group="lod0.grp"):
    """Rebuild a donor GH2 milo with its character geometry replaced by `chunks`,
    leaving every other entry byte-for-byte untouched.

    THIS IS THE PATH THAT LOADS IN-GAME, and the reason is worth recording because the
    from-scratch path looked correct and still crashed on the loading screen. A GH2
    BandCharacter directory body is NOT self-contained: it names `lod0.grp`,
    `lod1.grp`, `main.drv` and `shadow` directly. A fresh milo holding only Trans +
    Mesh entries leaves all four dangling, and the donor's whole driver layer
    (CharDriver, CharIKHand, CharEyes, CharHair, FaceFxLipSyncServo, OutfitLoader, ...)
    plus its Mat/Tex entries are missing too, so the character fails to resolve its own
    references while loading. The directory body is a set of pointers INTO the entry
    list, so the entry list has to stay intact.

    Injection keeps the donor's directory body, skeleton, materials, textures, drivers
    and groups, and swaps only the geometry of one mesh family.

    GROWING THE TABLE. A custom mesh usually splits into more four-bone chunks than the
    donor happens to have slots for (a real case: 118 chunks against funk1's 60). Since
    the entry table is ours to rewrite, extra chunks are appended as NEW Mesh entries -
    but that alone would leave them invisible, because GH2 draws a character through
    its LOD groups and a mesh absent from lod0.grp never renders. So the group is
    rewritten too (see write_gh2_group), with the new names added and any emptied
    donor slots removed from it. Unused donor slots are still kept as entries, rewritten
    as EMPTY meshes rather than deleted, so anything else pointing at them by name
    still resolves.

    chunks: (vertices, faces, bone_palette) triples from split_mesh_by_bone_limit.
    target_family: mesh-name family to replace; defaults to the donor's directory name.
    lod_group: the group whose membership tracks the replaced family.
    """
    with open(donor_path, 'rb') as f:
        raw = f.read()
    body = read_milo_body_le(raw)

    p = 0
    _rev = _d_u32(body, p); p += 4
    _dtype, p = _d_symbol(body, p)
    dir_name, p = _d_symbol(body, p)

    dir_body, spans = _parse_donor_entries(body)
    family = target_family or dir_name

    def fam_of(name):
        base = name[:-5] if name.endswith('.mesh') else name
        return base.split('.', 1)[0]

    slots = [i for i, (t, n, _s, _e) in enumerate(spans)
             if t == 'Mesh' and fam_of(n) == family]
    if not slots:
        raise GH2DonorError(
            f"Donor has no mesh family named '{family}' to replace. Families present: "
            f"{sorted({fam_of(n) for t, n, _s, _e in spans if t == 'Mesh'})}")

    used_names = {n for _t, n, _s, _e in spans}
    slot_to_chunk = {si: chunks[i] for i, si in enumerate(slots) if i < len(chunks)}

    # Extra chunks beyond the donor's slots become brand-new entries.
    extra = []
    k = 0
    for chunk in chunks[len(slots):]:
        while True:
            k += 1
            nm = f"{family}.x{k}.mesh"
            if nm not in used_names:
                break
        used_names.add(nm)
        extra.append((nm, chunk))

    # Membership for the LOD group: every slot that actually carries geometry, plus
    # every new entry; emptied slots drop out so the game isn't asked to draw nothing.
    filled_slot_names = [spans[si][1] for si in slots if si in slot_to_chunk]
    emptied_slot_names = {spans[si][1] for si in slots if si not in slot_to_chunk}
    new_names = [nm for nm, _c in extra]

    # Header + freshly written entry table (the table changes because entries are added).
    out = MiloWriter(big_endian=False)
    total_entries = len(spans) + len(extra)
    out.u32(GH2_MILO_REVISION)
    out.symbol(GH2_DIR_TYPE)
    out.symbol(dir_name)
    stc, sts = gh2_string_table_hint(
        GH2_DIR_TYPE, dir_name,
        [(t, n) for (t, n, _s, _e) in spans] + [("Mesh", nm) for nm, _c in extra])
    out.i32(stc)
    out.u32(sts)
    out.i32(total_entries)
    for (t, n, _s, _e) in spans:
        out.symbol(t)
        out.symbol(n)
    for (nm, _c) in extra:
        out.symbol("Mesh")
        out.symbol(nm)

    out.block(dir_body)

    replaced = 0
    emptied = 0
    regrouped = 0

    for i, (t, n, s, e) in enumerate(spans):
        if t == 'Mesh' and fam_of(n) == family:
            local, world, parent, mat = _parse_donor_mesh_header(body, s)
            if i in slot_to_chunk:
                cverts, cfaces, cpalette = slot_to_chunk[i]
                replaced += 1
            else:
                cverts, cfaces, cpalette = [], [], []
                emptied += 1
            write_gh2_rnd_mesh(out, n, local, world, parent,
                               cverts, cfaces, cpalette, mat_name=mat)
        elif t == 'Group' and n == lod_group:
            names, frame, flag, gparent = _parse_gh2_group(body, s, e)
            kept = [x for x in names
                    if x not in emptied_slot_names and fam_of(x) != family]
            merged = kept + filled_slot_names + new_names
            write_gh2_group(out, n, merged, anim_frame=frame, anim_flag=flag,
                            parent_name=gparent)
            regrouped = len(merged)
        else:
            out.block(body[s:e])

    # New entries' bodies, in the same order they were declared above. They borrow the
    # first replaced slot's header so they inherit the donor's material and bone
    # parenting rather than referencing names that may not exist.
    if extra:
        ref_local, ref_world, ref_parent, ref_mat = _parse_donor_mesh_header(
            body, spans[slots[0]][2])
        for nm, (cverts, cfaces, cpalette) in extra:
            write_gh2_rnd_mesh(out, nm, ref_local, ref_world, ref_parent,
                               cverts, cfaces, cpalette, mat_name=ref_mat)

    body_bytes = bytes(out.buf)

    block_sizes = []
    sf = 0
    last = 0
    while True:
        m = body_bytes.find(END_MARKER, sf)
        if m < 0:
            break
        sf = m + 4
        if sf - last > MAX_MILO_BLOCK_SIZE:
            block_sizes.append(sf - last)
            last = sf
    if last < len(body_bytes):
        block_sizes.append(len(body_bytes) - last)
    if not block_sizes:
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

    _log(f"GH2 injection into '{dir_name}': {replaced} slot(s) replaced, "
         f"{len(extra)} new mesh entr(ies) added, {emptied} slot(s) emptied, "
         f"'{lod_group}' now lists {regrouped} object(s). "
         f"{len(body_bytes)} body bytes in {len(block_sizes)} block(s).")

    return bytes(header.buf) + body_bytes


def check_gh2_references(data, expected_missing=()):
    """Confirm every name the directory body and the meshes point at actually exists as
    an entry. Returns (ok, message).

    This is the check that would have caught the loading-screen crash before it ever
    reached the console: a milo with a dangling `lod0.grp` / `main.drv` reference is
    structurally valid - every object still lands on its terminator - so the existing
    terminator check passes it happily. Name resolution is a separate failure mode and
    needs its own test.
    """
    try:
        body = read_milo_body_le(data)
        dir_body, spans = _parse_donor_entries(body)
        names = {n for _t, n, _s, _e in spans}

        # Length-prefixed printable strings inside the directory body are its
        # references (group names, driver name, shadow group, and the dir's own name).
        refs = set()
        p = 0
        while p + 4 <= len(dir_body):
            n = struct.unpack_from('<I', dir_body, p)[0]
            if 1 <= n <= 64 and p + 4 + n <= len(dir_body):
                s = dir_body[p + 4:p + 4 + n]
                if all(32 <= c < 127 for c in s):
                    refs.add(s.decode())
                    p += 4 + n
                    continue
            p += 1

        # A reference that looks like an asset (has a dot-suffix) must resolve; bare
        # words in here are state names like "in_solo"/"guitarist", not entries.
        dangling = sorted(r for r in refs
                          if ('.' in r) and r not in names
                          and r not in expected_missing)
        if dangling:
            return False, (f"Directory body references {len(dangling)} name(s) that "
                           f"aren't entries in this milo: {dangling}. The game resolves "
                           f"these while loading, so this would crash.")
        return True, f"OK: all directory references resolve ({len(names)} entries)."
    except Exception as e:
        return False, f"Reference check failed: {e}"


# Suffixes that denote an in-milo entry rather than an external file path. Used by the
# reference checks to decide whether a name is supposed to resolve to an entry.
GH2_ASSET_SUFFIXES = frozenset((
    'mesh', 'grp', 'drv', 'mat', 'tex', 'hair', 'coll', 'trans', 'ik', 'anim', 'gem',
))


GH2_GROUP_REVISION = 12
GH2_ANIM_REVISION = 4


def write_gh2_group(w, name, object_names, anim_frame=-1.08, anim_flag=1,
                    parent_name=""):
    """Group at revision 12 - a named list of objects that render together.

    Decoded field-for-field off funk1's `shadow` group and confirmed against
    lod0.grp / lod1.grp:

        u32 revision(12) | objFields | RndAnimatable(4) {f32 frame, u32}
        | embedded RndTrans(9) | RndDrawable(3)
        | u32 objectCount | objectName symbols...
        | 12 zero bytes | END_MARKER

    These matter more than they look: GH2 renders a character through its LOD groups,
    so a mesh that exists as an entry but is not listed in lod0.grp simply never draws.
    That is why injection can't just append meshes to the entry table and stop - the
    group membership has to be rewritten to match, which is what this writer is for.
    """
    w.u32(GH2_GROUP_REVISION)
    write_gh2_object_fields(w)

    w.u32(GH2_ANIM_REVISION)
    w.f32(anim_frame)
    w.u32(anim_flag)

    write_gh2_rnd_trans(w, (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                        parent_name, standalone=False)

    w.u32(GH2_DRAW_REVISION)
    w.boolean(True)
    w.f32(0.0); w.f32(0.0); w.f32(0.0); w.f32(0.0)
    w.f32(0.0)

    w.u32(len(object_names))
    for n in object_names:
        w.symbol(n)

    w.block(b"\x00" * 12)
    w.block(END_MARKER)


def _parse_gh2_group(body, s, e):
    """Read a Group's object-name list back. Returns (names, anim_frame, anim_flag,
    parent_name) so a rewritten group keeps the donor's own animation/parent values
    instead of having them invented."""
    q = s
    q += 4                                   # revision
    combined = _d_u32(body, q); q += 4
    tl = _d_u32(body, q); q += 4 + tl
    q += 1
    if combined & 0xFFFF > 0:
        nl = _d_u32(body, q); q += 4 + nl
    q += 4                                   # anim revision
    anim_frame = struct.unpack_from('<f', body, q)[0]; q += 4
    anim_flag = _d_u32(body, q); q += 4
    q += 4                                   # trans revision
    q += 96                                  # local + world
    q += 4                                   # constraint
    tl2 = _d_u32(body, q); q += 4 + tl2
    q += 1
    pl = _d_u32(body, q); q += 4
    parent = body[q:q + pl].decode('latin1'); q += pl
    draw_rev = _d_u32(body, q); q += 4
    q += 1 + 16 + 4
    if draw_rev >= 4:
        dl = _d_u32(body, q); q += 4 + dl
    count = _d_u32(body, q); q += 4
    names = []
    for _ in range(count):
        n, q = _d_symbol(body, q)
        names.append(n)
    return names, anim_frame, anim_flag, parent


# Entry types this addon can author from a Blender scene today. Anything NOT in here
# is a "special" that gets borrowed from the donor in reverse-donor mode.
#
# Mat/Tex are listed because the reverse-donor builder accepts them once the material
# and texture writers land; until then the operator simply passes none and the closure
# below borrows the donor's, which is exactly the behaviour we want in the meantime.
GH2_AUTHORABLE_TYPES = ('Trans', 'Mesh', 'Group', 'Mat', 'Tex', 'CharHair',
                        'CharCollide')

# CharHair is the one special confirmed NOT to be character-locked - it is an optional
# physics extra rather than something the character needs to load - so reverse-donor
# drops it by default instead of borrowing it. Everything else (the IK chains, driver,
# eyes, twist/servo bones, lipsync servo, outfit loader) is treated as required until
# a strip test proves otherwise.
GH2_OPTIONAL_DONOR_TYPES = ('CharHair',)


# The shadow mesh family is deliberately NOT carried over. Harmonix engines of this era
# fall back to casting a shadow from the main mesh when no dedicated shadow mesh is
# present (confirmed behaviour in The Beatles: Rock Band; assumed to hold for GH2,
# which is a generation earlier but the same lineage - flagging that it is an
# assumption here rather than something verified in GH2 itself).
#
# Note what is dropped and what is NOT: the shadow MESHES are skipped, but the
# directory body still names a `shadow` group, so reverse-donor authors that group
# EMPTY rather than omitting it. Omitting it entirely would leave the directory
# pointing at a group that doesn't exist - the exact dangling-reference crash this
# whole path exists to avoid. An empty group resolves fine and carries no geometry,
# which is what "no shadow mesh" is supposed to mean.
GH2_SHADOW_FAMILY = 'shadow'


# Donor mesh families that must not appear on a custom character, but whose ENTRIES
# still have to exist. The eyes and teeth belong to the donor's head, so they render
# floating on top of a custom model - but simply deleting them would break the things
# that reference them by name: lod0.grp/lod1.grp, both CharLookAt entries (l-eye.lookat
# / r-eye.lookat) and the FaceFxLipSyncServo. Dropping those drivers to suit the eyes
# would cost lipsync and head tracking.
#
# So these are BLANKED instead of removed: the entry is rewritten with zero vertices
# and zero faces, keeping its name, material and bone parenting intact. Every reference
# still resolves, every driver still finds its target, and nothing draws - which is
# what "ignore these" actually needs to mean here. Same trick the injector already uses
# for donor slots it doesn't fill.
GH2_BLANK_MESH_FAMILIES = ('eye-L', 'eye-R', 'teeth-upper', 'teeth-lower')


def _fam_of(name):
    """Mesh-name family: "funk1.12.mesh" -> "funk1", "shadow.3.mesh" -> "shadow"."""
    base = name[:-5] if name.endswith('.mesh') else name
    return base.split('.', 1)[0]


def _entry_name_refs(body, s, e, known_names):
    """Return the set of entry names a given object body references.

    Works by scanning for length-prefixed printable symbols and keeping the ones that
    match a known entry name. That is deliberately conservative: it can only ever
    return real entry names, so it never invents a dependency, though it could in
    principle miss a reference stored some other way. Since the whole point is to
    guarantee nothing dangles, missing one would show up immediately in
    check_gh2_references rather than silently shipping.
    """
    out = set()
    p = s
    while p + 4 <= e:
        n = struct.unpack_from('<I', body, p)[0]
        if 1 <= n <= 128 and p + 4 + n <= e:
            raw = body[p + 4:p + 4 + n]
            if all(32 <= c < 127 for c in raw):
                nm = raw.decode()
                if nm in known_names:
                    out.add(nm)
                p += 4 + n
                continue
        p += 1
    return out


def build_gh2_reverse_donor_milo_bytes(donor_path, mesh_entries, bone_trans_entries,
                                       groups=None, extra_entries=None,
                                       drop_types=GH2_OPTIONAL_DONOR_TYPES):
    """Build a GH2 character milo from AUTHORED Blender data, borrowing only the
    special entries we can't create yet from a donor. The reverse of
    build_gh2_injected_milo_bytes.

    Injection rebuilds the donor and swaps geometry into it, so the output is always
    "that character, with new meshes". This instead makes the Blender scene the source
    of truth - our Trans bones, our Mesh chunks, our Groups - and pulls across only
    what a GH2 character needs and Blender can't express: the driver, the IK chains,
    the eye/twist/servo bones, the lipsync servo, the outfit loader.

    THE CLOSURE IS THE IMPORTANT PART. The first GH2 export crashed because a copied
    directory body referenced entries that didn't exist. Copying a hand-picked list of
    "specials" has exactly the same failure mode one level down: a CharIKHand can name
    bones, a CharEyes can name eye meshes, a Group can name meshes. So rather than
    trusting a fixed list, this walks references transitively - anything reachable from
    the directory body or from an already-included entry, that we did not author
    ourselves, gets pulled in from the donor verbatim, repeatedly, until nothing new is
    reachable. That makes a dangling reference structurally impossible instead of
    something to remember to check.

    Bone names matter: donor specials reference bones by name, so a rig authored
    against the imported GH2 skeleton works, while a renamed rig would leave the
    closure pulling in donor bones to satisfy the drivers. That is reported rather than
    silently patched, since it usually means the rig isn't what the user thinks it is.

    mesh_entries:       8-tuples, as build_gh2_character_milo_bytes takes
    bone_trans_entries: 4-tuples, likewise
    groups:             (group_name, [object_name, ...]) pairs - typically
                        lod0.grp / lod1.grp / shadow. Authored so they list OUR meshes.
    extra_entries:      optional (type, name, body_bytes) triples already serialized
    drop_types:         donor types to skip entirely even if unreferenced-safe
    """
    if groups is None:
        groups = []
    if extra_entries is None:
        extra_entries = []

    with open(donor_path, 'rb') as f:
        raw = f.read()
    donor_body = read_milo_body_le(raw)

    p = 0
    _rev = _d_u32(donor_body, p); p += 4
    _dtype, p = _d_symbol(donor_body, p)
    donor_dir_name, p = _d_symbol(donor_body, p)

    dir_body, donor_spans = _parse_donor_entries(donor_body)
    donor_by_name = {n: (t, s, e) for (t, n, s, e) in donor_spans}
    donor_names = set(donor_by_name)

    # --- serialize everything we author, so all entries are uniform from here on ---
    authored = []          # (type, name, bytes)

    # ROOT BONES MUST PARENT TO THE CHARACTER DIRECTORY, NOT TO NOTHING.
    #
    # A Blender root bone has no parent, so _collect_gh2_bones emits parent="". Retail
    # does not: funk1's bone_pelvis.mesh and bone_door.mesh both name the directory
    # itself ("funk1") as their parent, and so does the inject-mode output, which is
    # why inject moved correctly and the first reverse-donor build did not.
    #
    # The symptom this causes is deceptive: the character still ANIMATES, because the
    # bone-to-bone hierarchy underneath is intact, but it cannot be MOVED, because the
    # game repositions the character by transforming its directory node and nothing
    # under an unparented root inherits that. The result is a rig that performs
    # perfectly while welded to stage centre.
    #
    # The directory name is the authority here (not the Blender armature's name), since
    # that is what the entry is actually called in this file.
    reparented = 0
    for (bone_name, local_xfm, world_xfm, parent_name) in bone_trans_entries:
        if not parent_name:
            parent_name = donor_dir_name
            reparented += 1
        w = MiloWriter(big_endian=False)
        write_gh2_rnd_trans(w, local_xfm, world_xfm, parent_name, standalone=True)
        authored.append(('Trans', bone_name, bytes(w.buf)))

    for (entry_name, local_xfm, world_xfm, parent_name, vertices, faces,
         bone_palette, mat_name) in mesh_entries:
        w = MiloWriter(big_endian=False)
        write_gh2_rnd_mesh(w, entry_name, local_xfm, world_xfm, parent_name,
                           vertices, faces, bone_palette, mat_name=mat_name)
        authored.append(('Mesh', entry_name, bytes(w.buf)))

    for (group_name, object_names) in groups:
        w = MiloWriter(big_endian=False)
        write_gh2_group(w, group_name, object_names)
        authored.append(('Group', group_name, bytes(w.buf)))

    for (t, n, blob) in extra_entries:
        authored.append((t, n, blob))

    authored_names = {n for (_t, n, _b) in authored}

    # --- transitive closure over donor specials ---
    # Seed from the directory body's own references, since that is what the game
    # resolves first and what crashed the very first export.
    borrowed = {}          # name -> (type, bytes)
    blanked = []           # donor meshes kept as entries but emptied
    pending = set(_entry_name_refs(dir_body, 0, len(dir_body), donor_names))

    # Seed from what WE authored as well, not just the directory body. An authored mesh
    # names a material ("goth2_body.mat"), and until this addon writes materials that
    # name can only be satisfied by the donor - so the Mat, and transitively the Tex
    # entries it points at, have to be pulled across.
    #
    # Missing this was a real hole rather than a theoretical one: it stayed invisible
    # only because meshes were being exported with an EMPTY material name, so there was
    # nothing to dangle. The moment a material name is set - i.e. the moment materials
    # start working - every mesh would reference a Mat that isn't in the file. The
    # useful side effect is that a mesh tagged with a donor material name now drags that
    # material and its textures in automatically, which is what stops the character
    # rendering as a white silhouette before a real .mat writer exists.
    for (_t, _n, blob) in authored:
        pending |= _entry_name_refs(blob, 0, len(blob), donor_names)

    # Every donor entry whose type we cannot author is required up front - these are
    # the drivers/IK/etc. the character needs regardless of who references them.
    for (t, n, s, e) in donor_spans:
        if t in drop_types:
            continue
        if t not in GH2_AUTHORABLE_TYPES:
            pending.add(n)

    while pending:
        name = pending.pop()
        if name in borrowed or name in authored_names:
            continue
        info = donor_by_name.get(name)
        if info is None:
            continue
        t, s, e = info
        if t in drop_types:
            continue
        # Never borrow the donor's shadow geometry - see GH2_SHADOW_FAMILY. This is
        # belt-and-braces: with the shadow group authored empty nothing should
        # reference these anyway, but a donor special that names one directly would
        # otherwise drag the whole family back in.
        if t == 'Mesh' and _fam_of(name) == GH2_SHADOW_FAMILY:
            continue
        if t == 'Mesh' and _fam_of(name) in GH2_BLANK_MESH_FAMILIES:
            # Keep the entry, drop the geometry - see GH2_BLANK_MESH_FAMILIES.
            local, world, parent, _mat = _parse_donor_mesh_header(donor_body, s)
            bw = MiloWriter(big_endian=False)
            # Material deliberately cleared: a blanked mesh draws nothing, so keeping
            # its material reference would drag the donor's Mat - and transitively its
            # Tex entries - into the file to serve geometry that doesn't exist.
            write_gh2_rnd_mesh(bw, name, local, world, parent, [], [], [],
                               mat_name="")
            borrowed[name] = (t, bytes(bw.buf))
            blanked.append(name)
            for ref in _entry_name_refs(donor_body, s, e, donor_names):
                if ref not in borrowed and ref not in authored_names:
                    pending.add(ref)
            continue
        borrowed[name] = (t, donor_body[s:e])
        for ref in _entry_name_refs(donor_body, s, e, donor_names):
            if ref not in borrowed and ref not in authored_names:
                pending.add(ref)

    # --- assemble ---
    entries = list(authored) + [(t, n, b) for n, (t, b) in borrowed.items()]

    out = MiloWriter(big_endian=False)
    out.u32(GH2_MILO_REVISION)
    out.symbol(GH2_DIR_TYPE)
    # Keep the DONOR's directory name: the directory body we copy embeds that name, and
    # the game looks the character up by the filename/dir it was registered under. A
    # custom name here would disagree with the body and with whatever the game expects
    # to find, which is a needless way to reintroduce the original crash.
    out.symbol(donor_dir_name)
    stc, sts = gh2_string_table_hint(
        GH2_DIR_TYPE, donor_dir_name, [(t, n) for (t, n, _b) in entries])
    out.i32(stc)
    out.u32(sts)
    out.i32(len(entries))
    for (t, n, _b) in entries:
        out.symbol(t)
        out.symbol(n)

    out.block(dir_body)
    for (_t, _n, b) in entries:
        out.block(b)

    body_bytes = bytes(out.buf)

    block_sizes = []
    sf = 0
    last = 0
    while True:
        m = body_bytes.find(END_MARKER, sf)
        if m < 0:
            break
        sf = m + 4
        if sf - last > MAX_MILO_BLOCK_SIZE:
            block_sizes.append(sf - last)
            last = sf
    if last < len(body_bytes):
        block_sizes.append(len(body_bytes) - last)
    if not block_sizes:
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

    from collections import Counter
    bt = Counter(t for (t, _b) in borrowed.values())
    if blanked:
        _log(f"  blanked {len(blanked)} donor mesh(es) so they resolve but never draw: "
             f"{', '.join(sorted(blanked))}")
    if reparented:
        _log(f"  reparented {reparented} root bone(s) onto the character directory "
             f"'{donor_dir_name}' - without this the rig animates but can't be moved.")
    _log(f"GH2 reverse-donor '{donor_dir_name}': authored {len(authored)} entr(ies) "
         f"({Counter(t for t, _n, _b in authored)}), borrowed {len(borrowed)} from "
         f"donor ({dict(bt)}). {len(body_bytes)} body bytes in {len(block_sizes)} "
         f"block(s).")

    return bytes(header.buf) + body_bytes, dict(bt)


def check_gh2_all_references(data, expected_missing=()):
    """Whole-file reference check: every entry name referenced by the directory body OR
    by any object body must exist as an entry. Returns (ok, message).

    `expected_missing` names references the caller has deliberately left unresolved -
    used by the "omit LOD groups" experiment, where the point IS to ship a milo whose
    directory names groups that aren't there, to find out whether the game needs them.
    Without this the check would correctly refuse to write the very file being tested.

    check_gh2_references only looks at the directory body, which is where the first
    crash came from. This goes one level deeper, because borrowed specials reference
    things too - a CharIKHand names bones, a Group names meshes - and those dangle just
    as fatally. Only names that look like assets (they carry a dot-suffix such as
    .mesh/.grp/.drv/.mat/.tex) are required to resolve; bare words in these bodies are
    state names like "guitarist" or "in_solo", not entries.
    """
    try:
        body = read_milo_body_le(data)
        dir_body, spans = _parse_donor_entries(body)
        names = {n for _t, n, _s, _e in spans}

        def refs_in(blob, s, e):
            out = set()
            p = s
            while p + 4 <= e:
                n = struct.unpack_from('<I', blob, p)[0]
                if 1 <= n <= 128 and p + 4 + n <= e:
                    raw = blob[p + 4:p + 4 + n]
                    if all(32 <= c < 127 for c in raw):
                        out.add(raw.decode())
                        p += 4 + n
                        continue
                p += 1
            return out

        # What counts as "looks like an entry name". Calibrating this PURELY off the
        # file's own entry names was a bug: a broken export that contains nothing but
        # .mesh entries would have a suffix set of {"mesh"}, so a dangling "lod0.grp"
        # or "main.drv" reference wouldn't even be considered - the check would pass
        # exactly the file that crashed. So the known GH2 asset suffixes are always
        # included, and the file's own are added on top to stay open-ended.
        suffixes = {n.rsplit('.', 1)[1] for n in names if '.' in n} | GH2_ASSET_SUFFIXES

        def is_entry_ref(r):
            # External asset paths are not entry references: retail bodies legitimately
            # carry things like "../textures/funk1_head.bmp" and
            # "../../../shared/ng/cheat_headflames.milo", which resolve on disk, not in
            # this milo. Flagging those was a false positive in the first version.
            if '/' in r or '\\' in r or r.startswith('.'):
                return False
            if '.' not in r:
                return False
            return r.rsplit('.', 1)[1] in suffixes

        suspicious = set()
        for r in refs_in(dir_body, 0, len(dir_body)):
            if is_entry_ref(r) and r not in names and r not in expected_missing:
                suspicious.add(('<directory>', r))
        for (t, n, s, e) in spans:
            for r in refs_in(body, s, e):
                if (is_entry_ref(r) and r not in names and r != n
                        and r not in expected_missing):
                    suspicious.add((n, r))

        if suspicious:
            sample = sorted(suspicious)[:8]
            return False, (f"{len(suspicious)} unresolved reference(s), e.g. "
                           + "; ".join(f"{a} -> {b}" for a, b in sample))
        return True, f"OK: all references resolve across {len(names)} entries."
    except Exception as e:
        return False, f"Whole-file reference check failed: {e}"


def check_gh2_root_bones(data):
    """Confirm every root Trans parents onto the character directory rather than onto
    nothing. Returns (ok, message).

    This exists because of a bug the other two checks sailed straight past: a milo with
    unparented root bones is structurally perfect (every object lands on its
    terminator) and has zero dangling references (an empty parent points at nothing, so
    there is nothing to dangle). It loads, and the character even animates correctly -
    it just can't be moved, because the game positions a character by transforming its
    directory node and an unparented root doesn't inherit that. The failure is visible
    only in-game, as a performer welded to stage centre.

    Retail is unambiguous here: funk1's two root bones (bone_pelvis.mesh,
    bone_door.mesh) both name the directory. So "root parent is empty" is treated as an
    error rather than a warning.
    """
    try:
        body = read_milo_body_le(data)
        p = 0
        _rev = _d_u32(body, p); p += 4
        _dtype, p = _d_symbol(body, p)
        dir_name, p = _d_symbol(body, p)

        _dir_body, spans = _parse_donor_entries(body)
        bone_names = {n for t, n, _s, _e in spans if t == 'Trans'}

        orphans = []
        for (t, n, s, e) in spans:
            if t != 'Trans':
                continue
            q = s
            q += 4
            combined = _d_u32(body, q); q += 4
            tl = _d_u32(body, q); q += 4 + tl
            q += 1
            if combined & 0xFFFF > 0:
                nl = _d_u32(body, q); q += 4 + nl
            q += 96                              # local + world
            q += 4                               # constraint
            tl2 = _d_u32(body, q); q += 4 + tl2
            q += 1
            pl = _d_u32(body, q); q += 4
            parent = body[q:q + pl].decode('latin1')
            if parent in bone_names:
                continue                         # a normal child bone
            if parent != dir_name:
                orphans.append((n, parent))

        if orphans:
            sample = ", ".join(f"{n} (parent {p!r})" for n, p in orphans[:5])
            return False, (f"{len(orphans)} root bone(s) not parented to the character "
                           f"directory '{dir_name}': {sample}. The character would "
                           f"animate but stay stuck at stage centre.")
        return True, f"OK: all root bones parent onto '{dir_name}'."
    except Exception as e:
        return False, f"Root-bone check failed: {e}"


# RndTex at revision 10 and its embedded RndBitmap at revision 1. Both byte-verified
# against all 8 textures in retail goth2.milo_xbox: for every one, the decoded
# width/height/bpp/encoding/mip-count predicts the pixel-data length EXACTLY.
#
#   Tex rev 10: objFields | u32 width | u32 height | u32 bpp | symbol externalPath
#               | f32 mipMapK(-8.0) | u32 type(1) | bool useExternalPath
#               | RndBitmap | END_MARKER
#
# This is the same layout The Beatles: Rock Band writes - also revision 10 - so the
# structural work is already done in texture_exporter.write_tbrb_rnd_tex, and the
# embedded RndBitmap (rev 1, with its u16 bpl + u16 wiiAlphaNum + 17 pad = the 19
# trailing bytes seen in GH2) is byte-identical to the RB3/TBRB one. Only two things
# differ for GH2 and neither is structural: the body is little-endian (the MiloWriter
# handles that) and objFields carries revision 1 rather than 2.
GH2_TEX_REVISION = 10

# Encodings seen in retail GH2, matching texture_exporter's constants:
#   8  = DXT1 (4 bpp) - used by the small utility/tint texture
#   24 = DXT5 (8 bpp) - every diffuse and specular map
#   32 = DXT5-class normal map (8 bpp) - every normal map
# Retail bottoms its mip chains out at 16 px, never 4: a 512 texture ships 5 mips
# below the base (512/256/128/64/32/16) and a 1024 ships 6. Same floor TBRB and RB2
# use. Running the chain to 4x4 would ship levels the game never carries.
GH2_TEX_MIP_FLOOR = 16


def write_gh2_rnd_tex(w, width, height, encoding, bpp, block_data,
                      external_path="", platform='xbox360', num_mips=0):
    """RndTex.Write at revision 10, GH2 flavour.

    Deliberately mirrors texture_exporter.write_tbrb_rnd_tex field for field rather
    than reimplementing it - see GH2_TEX_REVISION for why they're the same shape. The
    single delta is objFields revision 1 (GH2) vs 2 (everything else), which is why
    this can't just call the TBRB writer directly.

    `platform` is accepted for symmetry with the other writers but GH2 is Xbox 360
    only, so the Xbox 2-byte-pair swap in write_rnd_bitmap always applies.
    """
    from .texture_exporter import write_rnd_bitmap, TEX_TYPE_REGULAR

    w.u32(GH2_TEX_REVISION)
    write_gh2_object_fields(w)      # revision > 8 -> objFields, but at GH2's rev 1
    w.u32(width)
    w.u32(height)
    w.u32(bpp)
    w.symbol(external_path)
    w.f32(-8.0)                     # mipMapK - retail GH2 ships -8.0 in every texture
    w.u32(TEX_TYPE_REGULAR)         # type = 1
    # revision 10 < 11 -> no optimizeForPS3 field at all
    w.boolean(True)                 # useExternalPath - retail GH2 ships 1
    write_rnd_bitmap(w, width, height, encoding, bpp, block_data,
                     platform=platform, num_mips=num_mips)
    w.block(END_MARKER)


def build_gh2_texture_blocks(rgba, width, height, is_normal_map=False,
                             generate_mips=True):
    """Encode a Blender RGBA buffer into GH2's texture format.

    Returns (block_data, encoding, bpp, num_mips) ready for write_gh2_rnd_tex.

    TWO BUGS LIVED HERE, both of which produced a character that rendered but looked
    wrong, so they're worth recording:

    1. `build_texture_mip_chain`'s `encoder` argument is a STRING ('ati2' / 'bc3' /
       anything else = BC1), not a callable. Passing the `encode_bc3` FUNCTION fell
       through to the BC1 branch, so the pixels came out BC1 at 4bpp - and this
       function then overrode the returned encoding with DXT5 (24) anyway. The result
       was BC1 data labelled DXT5: the GPU read 8bpp DXT5 blocks out of a 4bpp BC1
       buffer, which scrambles colour and produces garbage in the alpha channel. The
       fix is to pass the string AND to trust the encoding/bpp the encoder reports back
       rather than asserting one.

    2. Normal maps are genuinely ATI2/BC5 (encoding 32, 8bpp), not DXT5 wearing a
       different tag. An earlier comment here claimed the block format was identical
       and only the tag differed - that was wrong, and it is why normal maps have to go
       through the 'ati2' encoder.

    Diffuse and specular use BC1 unless the image actually carries transparency, in
    which case BC3 is needed to preserve it. Retail does both: funk1_body.tex is BC1
    (encoding 8, 4bpp) while goth2_head.tex is DXT5 (encoding 24, 8bpp), so picking per
    image rather than fixing one format matches the source material and avoids paying
    8bpp for textures with no alpha.
    """
    from .texture_exporter import build_texture_mip_chain

    if is_normal_map:
        encoder = 'ati2'
    else:
        # Any pixel below fully-opaque means the alpha channel carries information BC1
        # cannot store (it has 1 bit of alpha at most).
        has_alpha = any(rgba[i] < 255 for i in range(3, len(rgba), 4))
        encoder = 'bc3' if has_alpha else 'bc1'

    block_data, encoding, bpp, num_mips = build_texture_mip_chain(
        rgba, width, height, encoder, generate_mips,
        mip_floor=GH2_TEX_MIP_FLOOR)
    return block_data, encoding, bpp, num_mips


# RndMat at revision 28. Decoded by taking texture_exporter.write_rnd_mat's rev-68
# field list and walking its revision gates down to 28 - then verified byte-exactly:
# all five retail materials in goth2.milo_xbox consume to their terminator with zero
# bytes left over, yielding sensible values throughout (real texture names, blend
# modes 1/2/3, specular powers 40/80/10/15, perPixelLit=1, stencil=0).
#
# Three gates do the heavy lifting, and they cut a lot of the modern material away:
#
#   * `revision > 37` is FALSE, so alphaThreshold is absent.
#   * `revision < 51` is TRUE, so an extra symbol slot IS present between specularMap
#     and environMap. Retail uses it: the two "skin" materials put goth2_skin.tex
#     there, which is the fourth texture slot noticed early in this project.
#   * `revision <= 28` triggers the EARLY RETURN. Everything RB3 writes after the
#     stencil field - deNormal, anisotropy, normal-detail, point lights, rim lighting,
#     shader variation, the perf-settings bools, refraction - simply does not exist at
#     this revision. That is why a GH2 material is ~220 bytes against RB3's much larger
#     one, and it is what makes this writer short.
#
# One correction worth recording: the rev-68 writer emits a u16 `unkShort` before
# perPixelLit. At revision 28 that field is NOT present - including it over-runs every
# retail material by exactly 2 bytes. The tail here is bool + bool + i32.
GH2_MAT_REVISION = 28


def write_gh2_rnd_mat(w, diffuse_tex="", normal_tex="", specular_tex="",
                      skin_tex="", base_color=(1.0, 1.0, 1.0, 1.0),
                      blend=1, z_mode=1, cull=True, pre_lit=True,
                      use_environment=True, emissive_multiplier=1.0,
                      specular_rgb=(0.0, 0.0, 0.0), specular_power=40.0,
                      standalone=True):
    """RndMat.Write at revision 28 (Guitar Hero 2).

    Defaults reproduce a retail OPAQUE character body material.

    blend defaults to 1 (kBlendSrc). It was briefly 3 (kBlendSrcAlpha), copied from
    goth2_body.mat without checking what the value meant - kBlendSrcAlpha turns on
    alpha blending, which made every exported character semi-transparent in-game. The
    character being replaced here, funk1, uses kBlendSrc on its body and head, which is
    the right default for opaque skin and cloth; a material that genuinely needs
    blending should pass blend explicitly.

    `skin_tex` is the rev<51 symbol slot described above - retail puts the shared skin
    tint texture there on "skin" materials and leaves it empty elsewhere.
    """
    from .texture_exporter import (
        MAT_STENCIL_IGNORE, MAT_TEXGEN_NONE, MAT_TEXWRAP_REPEAT,
    )
    from .utilities import IDENTITY_MATRIX

    w.u32(GH2_MAT_REVISION)
    write_gh2_object_fields(w)

    w.i32(blend)
    w.f32(base_color[0]); w.f32(base_color[1])
    w.f32(base_color[2]); w.f32(base_color[3])
    w.boolean(pre_lit)
    w.boolean(use_environment)
    w.i32(z_mode)
    w.boolean(False)                     # alphaCut
    # revision 28 is NOT > 37 -> no alphaThreshold field
    w.boolean(False)                     # alphaWrite
    w.i32(MAT_TEXGEN_NONE)
    w.i32(MAT_TEXWRAP_REPEAT)
    write_matrix(w, IDENTITY_MATRIX)     # texXfm
    w.symbol(diffuse_tex)
    w.symbol("")                         # nextPass
    w.boolean(False)                     # intensify
    w.boolean(cull)
    w.f32(emissive_multiplier)
    w.f32(specular_rgb[0]); w.f32(specular_rgb[1]); w.f32(specular_rgb[2])
    w.f32(specular_power)
    w.symbol(normal_tex)
    w.symbol("")                         # emissiveMap
    w.symbol(specular_tex)
    w.symbol(skin_tex)                   # revision < 51 -> this slot exists
    w.symbol("")                         # environMap
    # NO u16 unkShort at this revision - see GH2_MAT_REVISION
    w.boolean(True)                      # perPixelLit (1 in every retail material)
    w.boolean(False)                     # unkBool1 (27 <= revision < 50)
    w.i32(MAT_STENCIL_IGNORE)            # stencilMode (0 in every retail material)
    # revision <= 28 -> EARLY RETURN: nothing else is written.

    if standalone:
        w.block(END_MARKER)
