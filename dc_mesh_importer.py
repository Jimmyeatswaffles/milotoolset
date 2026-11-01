"""
Dance Central mesh importer.

Dance Central character milos carry the same RndMesh (version 38, 36-byte packed vertices)
as Rock Band 3, so the vertex/face/bone-table decoding is mesh_importer.parse_mesh_entry,
unchanged. What differs is everything around the meshes, which is why DC has its own
operator rather than reusing the RB3 one.

===========================================================================================
VERIFIED ON - retail Dance Central 1 angel01.milo_xbox (revision 28 Character, 24 meshes)
===========================================================================================

1. THE SKINNED MESHES ARE STORED TURNED AROUND.
   Every mesh's bone table puts its bones at the skeleton's rest pose turned 180 degrees about
   Z and shifted -0.392 along Y. It's one rigid transform for the whole character: across all
   64 bones the meshes share with the skeleton it deviates by at most 0.00009, and all 112
   bones the meshes bind to (skeleton + the file's face, hair and scarf bones) land exactly on
   their chained rest poses once it's undone. The game cancels it through the bind matrices
   at draw time; Blender, drawing the stored vertices at rest, showed the character facing
   away from its armature. The importer measures that transform rather than assuming it -
   a rigid fit of the meshes' bind positions to the reference skeleton's bone heads, with
   outliers dropped - and applies it to the skinned meshes.

   The reference is the armature in the scene, if there is one; otherwise the skeleton the
   milo inlines for itself (here 'Character uniq0', 74 bones) plus the file's own bones,
   chained from their local transforms exactly as the skeleton importer builds an armature.
   So a character imported first lines up with a skeleton imported afterwards too.

2. BONE-LESS MESHES (eyes, teeth) need their parent bone.
   Their stored world transforms are unreliable - the teeth's carry no translation at all -
   but parent rest x local transform puts them exactly in place: the teeth's locals cancel
   their raw offsets (upper teeth modelled around x = 62 under a local of x = -61.86 from
   bone_head). The parent bones are mostly face bones (bone_L-eye, bone_jaw) that DC keeps
   in the CHARACTER milo, not the skeleton milo, so the armature usually lacks them. With an
   armature in the scene, the file's bones it doesn't have are added to it first (see
   mesh_importer.add_file_bones), then each bone-less mesh is placed under its parent and
   bound to it with full weight.

3. LODS ARE DECLARED BY GROUP, NOT ONLY BY NAME.
   'lod0.grp' and 'lod1.grp' list the meshes of each level. Meshes in neither group (arms,
   legs, hair, hat, eyes, teeth, tongue - none of them duplicates of grouped geometry) are
   shared by every level and always imported.

4. TEXBLEND MASK MESHES.
   A TexBlendController names a mesh it uses as a region mask for wrinkle normal maps
   ('angel01_wrinkle.mesh', 74% coincident with the jacket). It has a material, so the
   generic helper rule (no bones and no material) misses it, and imported it z-fights the
   jacket. Such meshes are skipped by default.

5. LIMB MESHES.
   'left_arm', 'right_arm', 'left_leg' and 'right_leg' sit outside both LOD groups and
   don't duplicate any grouped geometry, but they aren't part of the outfit as worn: their
   materials don't match the outfit's (right_arm uses angel01_body_lod.mat, left_leg
   angel01_jacket_lod.mat). They're treated as helpers with their own import option.

6. NESTED DIRECTORIES. Handled by mesh_importer.walk_milo_entries.

8. DANCE CENTRAL 3 (verified on tan01.milo_xbox).
   Same RndMesh 38 and the same 180-degree storage (offset +0.248 in Y here). Differences:
   directory tables gain a byte at revision 32 (mesh_importer); textures sit in the root,
   with RndBitmap revision 2 (texture_importer); materials are revision 70, built from
   MetaMaterial templates whose values the files already carry (texture_importer); there
   are no lod groups, so '_lod' names mark the LODs; eyelashes.mesh is stored turned 90
   degrees where the rest are 180 (align_meshes fits it alone); and hair and cloth back
   faces come from a next-pass material rather than their own mesh (backface_passes).

7. TEXTURES AND MATERIALS.
   The textures live in the nested RndDirs (inline copies of the character's textures milo);
   texture_importer reads them there, along with DC's RB3-layout RndTex (revision 11) and
   RndMat (revision 68). Materials use the shared Milo Character Shader. The specular maps
   use an alpha-gloss trick like GDRB's, but the alpha holds the specular EXPONENT itself
   rather than a 0-1 scale on the material's power - see texture_importer.SPEC_ALPHA_EXPONENT.
   The jacket's normal map is a TexBlender render target; its base map is used. Textures
   from outside the milo (the hair's hair_detail_normals.tex, teeth_diff.tex) are looked up
   through the milos it references, and listed in the log when they can't be found.
"""

import os
import re
import struct

import bpy
from bpy.props import StringProperty, BoolProperty
from bpy_extras.io_utils import ImportHelper

from .io import read_milo_container_body
from .utilities import _log
from . import mesh_importer as _mi


# Bind positions within this distance of the fitted transform count as agreeing with it.
ALIGN_TOLERANCE = 0.05
# A fitted transform this close to the identity is treated as "already aligned".
IDENTITY_TOLERANCE = 1e-3

_LOD_GROUP_RE = re.compile(r'lod\s*(\d+)', re.IGNORECASE)
# Dance Central 3 has no lod groups; its reduced meshes are named '<base>_lod', optionally
# followed by '.N' (tan01: tan01_lod.mesh, tan01_lod.1.mesh, tan_head_lod.mesh).
_LOD_SUFFIX_RE = re.compile(r'_lod\d*(\.\d+)?\.mesh$', re.IGNORECASE)
_LIMB_RE = re.compile(r'^(left|right)_(arm|leg)(\.\d+)?\.mesh$', re.IGNORECASE)


def is_limb_name(name):
    """left_arm.mesh, right_leg.mesh and so on - see the module notes."""
    return bool(_LIMB_RE.match(name))


# ---------------------------------------------------------------------------------------
# Matrices (numpy, column-vector convention - the same layout as a Blender Matrix)
# ---------------------------------------------------------------------------------------

def _np():
    import numpy
    return numpy


def _col(m12):
    """Milo row-vector 4x3 -> column-vector 4x4, as io._milo_to_blender_matrix."""
    np = _np()
    m = np.eye(4)
    m[:3, :3] = np.array(m12[:9], dtype=float).reshape(3, 3).T
    m[:3, 3] = m12[9:12]
    return m


def _transform_mesh(m, xfm):
    """Moves a parsed mesh's vertices (and normals) through a column-vector 4x4."""
    np = _np()
    if not m['verts']:
        return
    co = np.array([v['co'] for v in m['verts']], dtype=float)
    nr = np.array([v['normal'] for v in m['verts']], dtype=float)
    lin = xfm[:3, :3]
    co = co @ lin.T + xfm[:3, 3]
    try:
        nlin = np.linalg.inv(lin).T          # handles scaled parents correctly
    except np.linalg.LinAlgError:
        nlin = lin
    nr = nr @ nlin.T
    ln = np.linalg.norm(nr, axis=1)
    ln[ln < 1e-12] = 1.0
    nr /= ln[:, None]
    for v, c, n in zip(m['verts'], co.tolist(), nr.tolist()):
        v['co'] = tuple(c)
        v['normal'] = tuple(n)


# ---------------------------------------------------------------------------------------
# Reading the file
# ---------------------------------------------------------------------------------------

def _sym_bytes(name):
    b = name.encode('latin-1')
    return struct.pack('>I', len(b)) + b


def _group_members(chunk, known):
    """The object list of a Group entry: a count followed by that many names, every one of
    them an object in the directory. Located by that structure rather than a fixed offset."""
    for o in range(0, max(0, len(chunk) - 4)):
        n = struct.unpack_from('>I', chunk, o)[0]
        if not 1 <= n <= 4096:
            continue
        p, names = o + 4, []
        for _ in range(n):
            if p + 4 > len(chunk):
                break
            ln = struct.unpack_from('>I', chunk, p)[0]
            if not 1 <= ln <= 256 or p + 4 + ln > len(chunk):
                break
            nm = chunk[p + 4:p + 4 + ln].decode('latin-1', 'replace')
            if nm not in known:
                break
            names.append(nm)
            p += 4 + ln
        if len(names) == n:
            return names
    return []


def read_dc_structure(body):
    """(walked entries, {mesh: lod level}, {mesh names used as TexBlend masks},
    {bone: (local 4x3, parent)} for every Trans in the file, nested skeleton included)."""
    walked = _mi.walk_milo_entries(body)
    root_names = {n for d, _p, _t, n, _s, _e in walked if d == 0}
    mesh_names = [n for d, _p, t, n, _s, _e in walked if d == 0 and t == 'Mesh']

    lod_of = {}
    for d, _p, t, n, s, e in walked:
        if d != 0 or t != 'Group':
            continue
        mm = _LOD_GROUP_RE.search(n)
        if not mm:
            continue
        level = int(mm.group(1))
        for member in _group_members(body[s:e], root_names):
            # A mesh listed at several levels counts as its most detailed one.
            lod_of[member] = min(level, lod_of.get(member, level))

    masks = set()
    for d, _p, t, n, s, e in walked:
        if t in ('TexBlendController', 'TexBlender'):
            chunk = body[s:e]
            masks.update(mn for mn in mesh_names if _sym_bytes(mn) in chunk)

    bones = {}
    # Root-level Trans last, so they win if a name ever appears in both places.
    trans = sorted((x for x in walked if x[2] == 'Trans'), key=lambda x: -x[0])
    for d, _p, _t, n, s, e in trans:
        tr = _mi._find_trans(body[s:e], e - s)
        if tr is not None:
            bones[n] = (tr[0], tr[2])
    return walked, lod_of, masks, bones


def chain_rest_poses(file_bones):
    """{bone: column 4x4 rest pose} chained from local transforms only - the same way the
    skeleton importer builds an armature, so the result matches one."""
    np = _np()
    out = {}

    def world(name, stack=()):
        if name in out:
            return out[name]
        local, parent = file_bones[name]
        m = _col(local)
        if parent and parent != name and parent in file_bones and parent not in stack:
            m = world(parent, stack + (name,)) @ m
        out[name] = m
        return m

    for n in file_bones:
        world(n)
    return out


def bones_needed(meshes, file_bones):
    """The file's bones the meshes actually use - their bone tables and the bone-less meshes'
    parents - plus every ancestor, so the chain to the skeleton is complete. Leaves out the
    attachment points ('spot_*.trans') that aren't part of the rig."""
    used = set()
    for m in meshes:
        used.update(m.get('bone_names') or ())
        if not m.get('bone_names') and m.get('trans_parent'):
            used.add(m['trans_parent'])
    out = set()
    for n in used:
        seen = set()
        while n in file_bones and n not in seen:
            seen.add(n)
            out.add(n)
            n = file_bones[n][1]
    return out


def parse_dc_meshes(filepath):
    """(dir_name, meshes, failed, file bones) for a Dance Central character milo. Meshes carry
    the RB3 parse plus 'lod_level' and DC's LOD and helper flags."""
    with open(filepath, 'rb') as f:
        body = read_milo_container_body(f.read())
    _rev, _dt, dir_name, _entries = _mi._read_dir_entries(body)
    walked, lod_of, masks, bones = read_dc_structure(body)

    meshes, failed = [], []
    for d, _p, t, n, s, e in walked:
        if d != 0 or t != 'Mesh':
            continue
        try:
            m = _mi.parse_mesh_entry(body[s:e], n)
        except (_mi.MeshImportError, struct.error) as ex:
            failed.append((n, str(ex)))
            continue
        level = lod_of.get(n)
        m['lod_level'] = level
        m['is_lod'] = (level is not None and level > 0) or (
            level is None and (m['is_lod'] or bool(_LOD_SUFFIX_RE.search(n))))
        m['is_texblend_mask'] = n in masks
        m['is_limb'] = is_limb_name(n)
        m['is_helper'] = m['is_helper'] or m['is_texblend_mask']
        meshes.append(m)
    if not meshes and not failed:
        raise _mi.MeshImportError("no Mesh entries in this milo's root directory")
    return dir_name, meshes, failed, bones


# ---------------------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------------------

def _bind_positions(meshes):
    """{bone: bind position in the meshes' own space}, first mesh wins."""
    np = _np()
    out = {}
    for m in meshes:
        for bn, bm in zip(m.get('bone_names') or (), m.get('bone_mats') or ()):
            if bn in out:
                continue
            try:
                out[bn] = np.linalg.inv(_col(bm))[:3, 3]
            except np.linalg.LinAlgError:
                pass
    return out


def _rigid_fit(src, dst):
    """Rotation R and translation t minimising |R src + t - dst| (Kabsch, no reflection)."""
    np = _np()
    cs, cd = src.mean(0), dst.mean(0)
    h = (src - cs).T @ (dst - cd)
    u, _s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, cd - r @ cs


def solve_alignment(meshes, rest):
    """The transform taking the skinned meshes into the reference skeleton's space, fitted on
    the bones they share. Returns (4x4 or None, inliers, shared) - None when there are too
    few shared bones, or they don't agree on one transform."""
    np = _np()
    binds = _bind_positions(meshes)
    shared = [n for n in binds if n in rest]
    if len(shared) < 3:
        return None, 0, len(shared)
    src = np.array([binds[n] for n in shared])
    dst = np.array([rest[n][:3, 3] for n in shared])
    keep = np.ones(len(shared), dtype=bool)
    for _ in range(5):
        if keep.sum() < 3:
            return None, int(keep.sum()), len(shared)
        r, t = _rigid_fit(src[keep], dst[keep])
        res = np.linalg.norm(src @ r.T + t - dst, axis=1)
        new_keep = res < max(ALIGN_TOLERANCE, 3.0 * float(np.median(res)))
        if (new_keep == keep).all():
            break
        keep = new_keep
    if keep.sum() < max(3, len(shared) // 2):
        return None, int(keep.sum()), len(shared)
    # Rank-deficient sets (all bones on a line) can't fix a rotation.
    if np.linalg.matrix_rank(src[keep] - src[keep].mean(0), tol=1e-3) < 2:
        return None, int(keep.sum()), len(shared)
    x = np.eye(4)
    x[:3, :3], x[:3, 3] = r, t
    return x, int(keep.sum()), len(shared)


def _rotation_note(x):
    np = _np()
    ang = np.degrees(np.arccos(np.clip((np.trace(x[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))
    t = x[:3, 3]
    return f"{ang:.1f} deg, offset ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})"


def align_meshes(meshes, rest, global_align):
    """Moves every skinned mesh into the reference skeleton's space - normally all by the one
    transform fitted on the whole character. A mesh whose own bones disagree with it gets a
    fit of its own: Dance Central 3's tan01 stores eyelashes.mesh turned 90 degrees where
    every other mesh is turned 180, so the global fit put them ~100 units off; fitted alone
    (3 of 3 bones agreeing) they land on the eyelids. Records each mesh's transform as
    '_align'. Returns [(mesh, note)] for the ones handled separately."""
    np = _np()
    notes = []
    for m in meshes:
        if not m.get('bone_names'):
            continue
        xfm = global_align
        binds = _bind_positions([m])
        shared = [n for n in binds if n in rest]
        if shared:
            res = max(float(np.linalg.norm(global_align[:3, :3] @ binds[n]
                                           + global_align[:3, 3] - rest[n][:3, 3]))
                      for n in shared)
            if res > ALIGN_TOLERANCE * 10:
                own, inl, sh = solve_alignment([m], rest)
                if own is not None and inl == sh:
                    xfm = own
                    notes.append((m['name'], f"stored in its own space - aligned separately "
                                             f"({_rotation_note(own)}; {inl} of {sh} bones)"))
                else:
                    notes.append((m['name'], f"disagrees with the character's alignment by "
                                             f"up to {res:.1f} and has too few bones to fit "
                                             f"alone - left with the character's"))
        if np.abs(xfm - np.eye(4)).max() < IDENTITY_TOLERANCE:
            m['_align'] = None
            continue
        _transform_mesh(m, xfm)
        m['_align'] = xfm
    return notes


def backface_passes(meshes, mats):
    """Dance Central 3 draws back faces with a second pass instead of a separate mesh: a
    material names a next-pass material with reversed culling (cull 2), which redraws the
    same triangles from behind - tan01's tan_hair.mat -> tan_hair_backfaces.mat and
    tan01_cloth.mat -> tan01_cloth_backface.mat, from the char_basic_*_backface(s)
    templates. A Blender mesh has one material per face, so each such mesh gets a copy with
    its winding reversed and normals flipped, carrying the next-pass material - the shape
    DC1 and RB3 characters already ship their backface meshes in. Returns the copies."""
    out = []
    for m in meshes:
        slots = mats.get(m.get('mat'))
        nxt = slots and slots.get('next_pass')
        nslots = mats.get(nxt) if nxt else None
        if not nslots or nslots.get('cull') != 2:
            continue
        copy = dict(m)
        base = m['name'][:-5] if m['name'].endswith('.mesh') else m['name']
        copy['name'] = f"{base}_backface.mesh"
        copy['mat'] = nxt
        copy['faces'] = [(f[0], f[2], f[1]) for f in m['faces']]
        copy['verts'] = [dict(v, normal=tuple(-c for c in v['normal'])) for v in m['verts']]
        copy['is_next_pass'] = True
        out.append(copy)
    return out


def place_dc_rigid_meshes(meshes, rest):
    """Bone-less meshes go under their parent bone: parent rest x their own local transform.
    Records 'rigid_parent' so build_mesh_object binds them to it. Returns [(mesh, note)]."""
    notes = []
    for m in meshes:
        if m.get('bone_names') or m.get('is_helper'):
            continue
        parent = m.get('trans_parent', '')
        local = m.get('local_xfm')
        if parent in rest and local is not None:
            _transform_mesh(m, rest[parent] @ _col(local))
            m['rigid_parent'] = parent
            p = rest[parent][:3, 3]
            notes.append((m['name'], f"placed under '{parent}' at "
                                     f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})"))
        else:
            # Nothing to hang it from: fall back to the stored world transform.
            m['rigid_parent'] = parent
            w = m.get('world_xfm')
            if w is not None:
                _transform_mesh(m, _col(w))
            notes.append((m['name'], f"parent '{parent or '-'}' unknown - placed by its "
                                     f"stored transform, which may be off"))
    return notes


# ---------------------------------------------------------------------------------------
# Blender side
# ---------------------------------------------------------------------------------------

def _armature_rest(armature_obj):
    np = _np()
    return {b.name: np.array([list(r) for r in b.matrix_local], dtype=float)
            for b in armature_obj.data.bones}


def find_scene_armature(context, meshes):
    """The armature these meshes belong to: the active object if it's an armature sharing
    bones with them, otherwise the scene armature sharing the most. None if none shares any -
    an unrelated rig in the scene is left alone."""
    wanted = set()
    for m in meshes:
        wanted.update(m.get('bone_names') or ())
        if m.get('trans_parent'):
            wanted.add(m['trans_parent'])

    def score(o):
        return sum(1 for n in wanted if n in o.data.bones)

    active = context.active_object
    if active is not None and active.type == 'ARMATURE' and score(active) > 0:
        return active
    best, best_score = None, 0
    for o in context.scene.objects:
        if o.type == 'ARMATURE':
            s = score(o)
            if s > best_score:
                best, best_score = o, s
    return best


class _IMPORT_OT_dc_meshes_base(bpy.types.Operator, ImportHelper):
    """Shared by the Dance Central importers. Everything that differs between DC1 and DC3 is
    detected from the file itself (directory, bitmap and material revisions, lod groups or
    lod names, next-pass back faces), so the two only differ in name."""
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".milo_xbox"
    filter_glob: StringProperty(default="*.milo_xbox", options={'HIDDEN'})

    import_lods: BoolProperty(
        name="Import LODs",
        description="Also import the reduced-detail meshes (members of lod1.grp and up, or "
                     "named like LOD1). Meshes no LOD group lists - arms, legs, hair, eyes, "
                     "teeth - are shared by every level and always imported",
        default=False,
    )
    import_limbs: BoolProperty(
        name="Import Limb Meshes",
        description="Also import left_arm, right_arm, left_leg and right_leg. They sit "
                     "outside the outfit's LOD groups with materials that don't match it, "
                     "and aren't part of the character as worn",
        default=False,
    )
    import_helpers: BoolProperty(
        name="Import Helper Meshes",
        description="Also import meshes a TexBlendController uses as a wrinkle-map region "
                     "mask, and meshes with no bones and no material. They aren't visible "
                     "geometry and z-fight the clothing they cover",
        default=False,
    )
    exclude_shadow: BoolProperty(
        name="Exclude Shadow Mesh",
        description="Skip the flat shadow-caster mesh, if the milo has one",
        default=True,
    )
    use_scene_armature: BoolProperty(
        name="Use Scene Armature",
        description="Find the character's armature in the scene (the active one, or whichever "
                     "shares the most bones with these meshes), turn the meshes to face the "
                     "same way, and parent them to it",
        default=True,
    )
    add_file_bones: BoolProperty(
        name="Add Face Bones From This File",
        description="Add the bones this milo defines that the armature lacks - Dance Central "
                     "keeps the face rig (jaw, eyes, lids, brows, lips, tongue) and outfit "
                     "physics bones in the character milo, not the skeleton. The eyes and "
                     "teeth need them to be parented, and the head is weighted to them",
        default=True,
    )

    import_textures: BoolProperty(
        name="Import Textures",
        description="Decode the milo's textures and rebuild each material with the Milo "
                     "Character Shader: added Blinn-Phong specular with the specular map's "
                     "alpha as the exponent, detail normals, two-lobe hair highlights, "
                     "culling. Also fills each material's glTFMilo settings panel",
        default=True,
    )
    flip_normal_green: BoolProperty(
        name="Flip Normal Map Green",
        description="Convert the normal maps' Y from DirectX (down) to Blender's OpenGL "
                     "(up). Turn this off only if bumps light up as dents",
        default=True,
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "import_lods")
        layout.prop(self, "import_limbs")
        layout.prop(self, "import_helpers")
        layout.prop(self, "exclude_shadow")
        layout.separator()
        layout.prop(self, "import_textures")
        row = layout.row()
        row.enabled = self.import_textures
        row.prop(self, "flip_normal_green")
        layout.separator()
        layout.prop(self, "use_scene_armature")
        row = layout.row()
        row.enabled = self.use_scene_armature
        row.prop(self, "add_file_bones")

    def execute(self, context):
        np = _np()
        try:
            dir_name, meshes, failed, file_bones = parse_dc_meshes(self.filepath)
        except Exception as e:
            _log(f"DC MESH IMPORT FAILED: {e}")
            self.report({'ERROR'}, f"Could not parse Dance Central milo: {e}")
            return {'CANCELLED'}

        _log(f"===== Importing Dance Central meshes from '{dir_name}' ({self.filepath}) =====")
        armature_obj = find_scene_armature(context, meshes) if self.use_scene_armature else None
        if armature_obj is not None and armature_obj.mode == 'EDIT':
            bpy.ops.object.mode_set(mode='OBJECT')
        if armature_obj is not None:
            _log(f"  using armature '{armature_obj.name}'")
            rest, ref_label = _armature_rest(armature_obj), f"armature '{armature_obj.name}'"
        elif file_bones:
            rest, ref_label = chain_rest_poses(file_bones), "the milo's own skeleton"
        else:
            rest, ref_label = {}, None

        # 1. Turn the skinned meshes to face the way the skeleton does.
        align, inliers, shared = solve_alignment(meshes, rest) if rest else (None, 0, 0)
        if align is None:
            _log(f"  orientation: left as stored ({shared} bone(s) shared with "
                 f"{ref_label or 'no reference skeleton'}, {inliers} agreeing - not enough)")
        else:
            _log(f"  orientation: aligned to {ref_label} ({_rotation_note(align)}; "
                 f"{inliers} of {shared} shared bones agree)")
            for name, note in align_meshes(meshes, rest, align):
                _log(f"  '{name}': {note}")

        # 2. Give the armature the bones the file defines - DC's face rig lives here.
        if armature_obj is not None and self.add_file_bones and file_bones:
            from mathutils import Matrix
            bind_world = {}
            for m in meshes:
                for bn, bm in zip(m.get('bone_names') or (), m.get('bone_mats') or ()):
                    if bn in bind_world:
                        continue
                    try:
                        w = np.linalg.inv(_col(bm))
                    except np.linalg.LinAlgError:
                        continue
                    if m.get('_align') is not None:
                        w = m['_align'] @ w
                    bind_world[bn] = Matrix(w.tolist())
            needed = bones_needed(meshes, file_bones)
            try:
                added = _mi.add_file_bones(context, armature_obj,
                                           {n: file_bones[n] for n in needed},
                                           bind_world=bind_world)
            except Exception as e:
                _log(f"  couldn't add this file's bones ({e}) - carrying on without them")
                added = []
            if added:
                _log(f"  added {len(added)} bone(s) to '{armature_obj.name}': "
                     + ", ".join(n.rsplit('.', 1)[0] for n, _p in added[:12])
                     + (" ..." if len(added) > 12 else ""))
            rest = _armature_rest(armature_obj)
        elif armature_obj is not None:
            # Bones the armature lacks still get placed and resolved through the file's own.
            for n, w in chain_rest_poses(file_bones).items():
                rest.setdefault(n, w)

        # 3. Eyes, teeth: under their parent bone.
        for name, note in place_dc_rigid_meshes(meshes, rest):
            _log(f"  '{name}': {note}")

        # Weights for bones still missing go to the nearest ancestor the armature has.
        trans_bones = {n: (None, p) for n, (_l, p) in file_bones.items()}
        for name, note in _mi.remap_unknown_bones(meshes, trans_bones, armature_obj):
            _log(f"  '{name}': {note}")

        # Back faces drawn by a next pass (DC3) become meshes of their own.
        try:
            from . import texture_importer as _ti
            _t, mats_info, _f = _ti.read_milo_materials(self.filepath)
        except Exception as e:
            mats_info = {}
            _log(f"  materials couldn't be read for next passes ({e})")
        extra = backface_passes(meshes, mats_info)
        for c in extra:
            _log(f"  '{c['name']}': back faces of '{c['name'][:-14]}.mesh' "
                 f"drawn with next pass '{c['mat']}'")
        meshes = meshes + extra

        # 4. Pick what to build.
        lods = [m for m in meshes if m['is_lod']]
        helpers = [m for m in meshes if m.get('is_helper')]
        limbs = [m for m in meshes if m.get('is_limb')]
        wanted = [m for m in meshes
                  if (self.import_lods or not m['is_lod'])
                  and (self.import_limbs or not m.get('is_limb'))
                  and (self.import_helpers or not m.get('is_helper'))
                  and not (self.exclude_shadow and m['is_shadow'])]
        if lods and not self.import_lods:
            _log(f"  {len(lods)} LOD mesh(es) skipped: {', '.join(m['name'] for m in lods)}")
        if limbs and not self.import_limbs:
            _log(f"  {len(limbs)} limb mesh(es) skipped: {', '.join(m['name'] for m in limbs)}")
        if helpers and not self.import_helpers:
            _log(f"  {len(helpers)} helper mesh(es) skipped: "
                 f"{', '.join(m['name'] for m in helpers)}")

        source = os.path.basename(self.filepath)
        total_v = total_f = 0
        unmatched = set()
        arm_bones = set(armature_obj.data.bones.keys()) if armature_obj else set()
        for m in wanted:
            _mi.build_mesh_object(context, m, armature_obj, source)
            total_v += len(m['verts'])
            total_f += len(m['faces'])
            _log(f"    {m['name']:34s} {len(m['verts']):5d}v {len(m['faces']):5d}f  "
                 f"{len(m['bone_names']):2d} bone(s)  mat='{m['mat']}'"
                 + (f"  lod{m['lod_level']}" if m.get('lod_level') is not None else ""))
            if arm_bones:
                remap = m.get('bone_remap', {})
                unmatched.update(remap.get(b, b) for b in m['bone_names']
                                 if remap.get(b, b) not in arm_bones)
        for nm, reason in failed:
            _log(f"    SKIPPED {nm}: {reason}")

        # 5. Textures and materials.
        tex_summary, tex_warning = "", False
        if self.import_textures:
            from . import texture_importer as _ti
            wanted_mats = sorted({m['mat'] for m in wanted if m['mat']})
            try:
                done, n_images, tex_notes, tex_missing = _ti.apply_textures(
                    self.filepath, wanted_mats, self.flip_normal_green, source=source,
                    spec_alpha=_ti.SPEC_ALPHA_EXPONENT)
                _log(f"  textures: {done} of {len(wanted_mats)} material(s) textured, "
                     f"{n_images} image(s) decoded")
                for n in tex_notes:
                    _log(f"    {n}")
                tex_summary = f"; {done} material(s) textured"
                if tex_missing:
                    tex_warning = True
                    tex_summary += (f"; {len(tex_missing)} texture(s) not found "
                                    f"(see log)")
            except Exception as e:
                _log(f"  textures: could not be imported ({e})")
                tex_summary, tex_warning = "; textures failed (see log)", True
        if unmatched:
            _log(f"  {len(unmatched)} vertex group(s) name bones the armature doesn't have "
                 f"(e.g. {', '.join(sorted(unmatched)[:8])})")

        summary = (f"Imported {len(wanted)} mesh(es) from '{dir_name}': {total_v} verts, "
                   f"{total_f} faces"
                   + (f"; {len(lods)} LOD skipped" if lods and not self.import_lods else "")
                   + (f"; {len(limbs)} limb skipped"
                      if limbs and not self.import_limbs else "")
                   + (f"; {len(helpers)} helper skipped"
                      if helpers and not self.import_helpers else "")
                   + (f"; {len(failed)} failed (see log)" if failed else "")
                   + (f"; parented to '{armature_obj.name}'" if armature_obj else
                      "; no matching armature in the scene")
                   + tex_summary)
        _log(f"===== {summary} =====")
        self.report({'WARNING' if (failed or tex_warning) else 'INFO'}, summary)
        return {'FINISHED'}


class IMPORT_OT_dc_meshes(_IMPORT_OT_dc_meshes_base):
    """Import the meshes from a Dance Central character milo, turned to face the same way as
    the skeleton, with per-bone vertex groups and textured materials. Uses the armature
    already in the scene, adding the character's face bones to it and binding the eyes and
    teeth to their bones"""
    bl_idname = "import_scene.dc_meshes"
    bl_label = "Import Dance Central Meshes"


class IMPORT_OT_dc3_meshes(_IMPORT_OT_dc_meshes_base):
    """Import the meshes from a Dance Central 3 character milo, turned to face the same way
    as the skeleton, with per-bone vertex groups and textured materials. DC3 adds a
    revision-32 directory table, revision-70 materials built from MetaMaterial templates,
    back faces drawn by a next pass, and a mesh stored in a space of its own (eyelashes);
    see the module notes"""
    bl_idname = "import_scene.dc3_meshes"
    bl_label = "Import Dance Central 3 Meshes"
