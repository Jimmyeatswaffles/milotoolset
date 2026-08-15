import bpy
import struct

from .utilities import (
    MiloWriter, write_object_fields, write_rnd_animatable, write_rnd_drawable,
    write_rnd_trans, write_matrix, write_sphere, IDENTITY_MATRIX, END_MARKER,
    sanitize_milo_name, _matrix_to_milo, BoneLimitExceeded, _find_armature_modifier,
    _bone_axis_correction_matrix, _resolve_stock_reference_armature, _log,
    RB3_TRANS_REVISION, RB3_ANIM_REVISION, RB3_DRAW_REVISION, DC3_DRAW_REVISION,
    RB3_OBJDIR_REVISION, RB3_RNDDIR_REVISION, MAX_MILO_BLOCK_SIZE, OBJ_FIELDS_REVISION,
    TBRB_MILO_REVISION, TBRB_OBJDIR_REVISION, TBRB_RNDDIR_REVISION,
    write_object_dir_base, write_rnd_dir,
)
from .physics_exporter import (
    write_tbrb_char_collide, build_tbrb_standard_collision_entries,
    RB3_CHARCOLLIDE_REVISION, TBRB_CHARCOLLIDE_REVISION,
)


TBRB_CHARACTER_REVISION = 15


TBRB_CHARTEST_REVISION = 10


TBRB_SKELETON_SUBDIR = "../../world/shared/wind.milo"


TBRB_CHARTEST_LOOK_BONE = "bone_R-eye.mesh"


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
    w.symbol("")


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
    w.symbol(look_bone)


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


RB3_CHARACTER_REVISION = 17


RB3_CHARTEST_REVISION = 15


CHAR_SHARED_MILO_PATH = "../../shared/char_shared.milo"


INSTRUMENT_SHARED_MILO_PATH = "../shared/colorpalettes.milo"


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
    w.boolean(False)


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

    w.block(END_MARKER)


def write_bone_transform(w: MiloWriter, name, transform12):
    """RndMesh.BoneTransform.Write() - just a Symbol + Matrix, no revision gating."""
    w.symbol(name)
    write_matrix(w, transform12)


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
                                  zero_rotation=False, exclude_bones=None):
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

    `exclude_bones` (optional): a set of bone names to leave out entirely. This exists
    so a caller that emits the WHOLE armature in one pass can still hand specific bones
    off to a second pass with different options - specifically the hair physics bones,
    which must be written with zero_rotation=True. Without it, a full-armature export
    (DC1/DC3, where nothing is filtered) claims the hair bones first with their rotations
    intact, and the dedicated zero_rotation pass below then finds nothing left to emit -
    which is exactly how the hair rotation fix silently stopped applying on the Dance
    Central path while continuing to work on RB3 (where the full-armature pass doesn't run).

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
                           if (only_bones is None or b.name in only_bones)
                           and (not exclude_bones or b.name not in exclude_bones)]
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


