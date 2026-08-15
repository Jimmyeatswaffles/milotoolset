import bpy
import struct

from .utilities import (
    MiloWriter, write_object_fields, write_rnd_trans, write_matrix, write_sphere,
    IDENTITY_MATRIX, END_MARKER, sanitize_milo_name, _matrix_to_milo, _log,
)


RB3_CHARMESHHIDE_REVISION = 2


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


RB3_CHARHAIR_REVISION = 11


DEFAULT_HAIR_WIND = "world.wind"


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


RB3_OUTFITCONFIG_REVISION = 27


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


RB3_CHARCOLLIDE_REVISION = 7


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


TBRB_CHARCOLLIDE_REVISION = 5


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


# =====================================================================================
# Per-game CharHair (.hair) profiles
#
# The .hair format is byte-identical across these games, but what the ENGINE expects in
# a few fields is not, so each game gets its own profile rather than one shared set of
# assumptions. Add a game here rather than sprinkling `if game == ...` through the
# builder.
#
#   zero_rotation - whether hair physics bones are written with their local rotation
#                   flattened to identity (and the strand's base/root matrices with them).
#                   The hard invariant, verified byte-for-byte on retail files, is that a
#                   strand's base_mat EQUALS its root bone's Trans local rotation. Zeroing
#                   both satisfies it; keeping both real also satisfies it. Mixing them is
#                   what twists strands. Retail RB3 (hair_ladylayered) and retail DC1 both
#                   ship REAL rotations - base_mat deviates from identity by 1.0-1.99 in
#                   every strand - so False matches shipped data on both.
# =====================================================================================
HAIR_GAME_PROFILES = {
    'rb3':  {'zero_rotation': False},
    'dc1':  {'zero_rotation': False},
    'dc3':  {'zero_rotation': False},
    # Not yet checked against retail files for these two - they inherit the RB3-style
    # profile until someone diffs a real one.
    'rb2':  {'zero_rotation': False},
    'tbrb': {'zero_rotation': False},
}

DEFAULT_HAIR_PROFILE = {'zero_rotation': False}


def hair_profile_for_game(game):
    """Returns the CharHair export profile for a game id, falling back to the default
    for anything not listed in HAIR_GAME_PROFILES."""
    return HAIR_GAME_PROFILES.get(game, DEFAULT_HAIR_PROFILE)


def build_char_hair_entries(armatures_used, root_name, zero_rotation=False):
    """Builds CharHair (.hair) entries from any armatures whose gltfmilo_hair.enabled
    is set and that have at least one strand. Returns a list of
    (entry_name, settings_dict, strands_list) tuples for build_milo_bytes /
    write_char_hair. Point positions/lengths and the base/root matrices are derived
    from each bone's rest transform; the engine's Hookup() largely recomputes them at
    load, so these are a valid, self-consistent initial state rather than the final
    simulated one.

    `zero_rotation` MUST match what was passed to build_armature_trans_entries for the
    same hair bones. The hard requirement is that a strand's base/root matrices EQUAL the
    root bone's Trans local rotation - verified byte-for-byte against retail files, where
    the two are identical to 0.0000. There are two self-consistent ways to satisfy that:

      zero_rotation=True  - zero the bones' local rotation AND write identity matrices
                            here. This is the RB3 path and is known-good there.
      zero_rotation=False - keep the bones' real rotations AND write the real rotation
                            here. This is what retail Dance Central does (checked against
                            a vanilla DC1 character: every strand's base_mat matches its
                            root bone's Trans local rotation exactly, and neither is
                            identity). It also preserves the orientations the rig was
                            authored with, instead of collapsing every physics bone onto
                            its parent's direction.

    What actually breaks is MIXING them - real rotations on the Trans entries with
    identity matrices here (or vice versa) is the disagreement that twists strands."""
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

            # Gather the chain's world head positions first: each point's `length` refers
            # to the gap with the PREVIOUS point, so the whole chain has to be known.
            chain = []
            for bname in bone_names:
                bone = armature_obj.data.bones.get(bname)
                if bone is None:
                    continue
                world_head = armature_obj.matrix_world @ bone.head_local
                chain.append((bname, bone, world_head))

            points = []
            for idx, (bname, bone, world_head) in enumerate(chain):
                # `length` is the rest span the solver holds each segment to, and it means
                # "the span of MY bone" - the distance from this point to the next one in
                # the strand. The tip has no next point, so retail repeats the previous
                # span rather than inventing a value; every vanilla strand checked (RB3's
                # hair_ladylayered and a vanilla DC1 character, 33 segments) ends with its
                # last two lengths identical.
                #
                # Three earlier mistakes, all of which shipped and all of which are worth
                # naming so they don't come back:
                #   * Blender's bone.length is the bone's own head->tail DISPLAY length. It
                #     only equals the chain gap when bones are connected head-to-tail, so
                #     disconnected physics bones all report the same arbitrary value.
                #   * Falling back to bone.length for the TIP point gave it a meaningless
                #     span (1.0 on a default-length bone) and collapsed the end of a strand.
                #   * Measuring the FIRST point back to its anchor bone stretches the strand
                #     badly: the anchor is usually bone_head.mesh, whose origin sits deep in
                #     the skull, so that distance came out 2.2-2.8x the real segment span
                #     (5.36 where the actual gap was 2.27) and the root segment visibly
                #     over-extended. A single-bone strand got ONLY that inflated number.
                if idx + 1 < len(chain):
                    seg_length = (chain[idx + 1][2] - world_head).length
                elif len(chain) >= 2:
                    seg_length = points[-1]['length']      # tip: repeat the previous span
                else:
                    seg_length = bone.length               # lone bone: nothing to measure
                points.append({
                    'pos': (world_head.x, world_head.y, world_head.z),
                    'bone': bname,
                    'length': seg_length,
                    'radius': 0.0,
                    'outer_radius': -1.0,
                    'side_length': -1.0,
                })

            root_bone = armature_obj.data.bones.get(bone_names[0])
            if root_bone is None or zero_rotation:
                # Bones were written with their local rotation zeroed, so the strand's
                # reference frame has to be identity to match them.
                base_mat = IDENTITY_MATRIX3
            else:
                # Bones keep their real rotation, so the strand's reference frame must be
                # the root bone's real Trans LOCAL rotation - the same value
                # build_armature_trans_entries writes into that bone's localXfm. Retail
                # DC1 files store exactly this (base_mat == root bone's Trans local
                # rotation, matching to 0.0000). Taking the first 9 floats of the milo
                # 4x3 is the rotation basis; the last 3 are translation, which Matrix3
                # doesn't carry.
                world_xfm = armature_obj.matrix_world @ root_bone.matrix_local
                if root_bone.parent is not None:
                    parent_world = armature_obj.matrix_world @ root_bone.parent.matrix_local
                    local_xfm = parent_world.inverted() @ world_xfm
                else:
                    local_xfm = world_xfm
                base_mat = tuple(_matrix_to_milo(local_xfm)[:9])

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


