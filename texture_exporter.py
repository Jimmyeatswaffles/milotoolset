import bpy
import struct

from .utilities import (
    MiloWriter, write_object_fields, write_rnd_animatable, write_rnd_drawable,
    write_rnd_trans, write_matrix, write_sphere, write_color3, write_color4,
    IDENTITY_MATRIX, END_MARKER, sanitize_milo_name, _log,
    RB3_TRANS_REVISION, RB3_DRAW_REVISION, DC3_DRAW_REVISION, OBJ_FIELDS_REVISION,
)


_TEX_ENCODING_NAMES = {}


TBRB_MAT_REVISION = 55


TBRB_TEX_REVISION = 10


TBRB_TEXBLENDCONTROLLER_REVISION = 1


TBRB_TEXBLENDER_REVISION = 2


TBRB_TEX_MIP_FLOOR = 16


DC1_TEX_MIP_FLOOR = 16


def write_tbrb_rnd_tex(w: MiloWriter, width, height, encoding, bpp, block_data,
                        external_path="", platform='xbox360', num_mips=0):
    """RndTex.Write at revision 10 (The Beatles: Rock Band). Byte-layout verified against
    all 17 textures in the two retail milos.

    Two deltas from the RB3 rev-11 writer:
      1. revision 10 is < 11, so optimizeForPS3 is NOT written at all.
      2. useExternalPath is TRUE in every retail TBRB texture, even though externalPath is
         empty and the bitmap is embedded immediately after. The RB3 path writes False.
         Retail is reproduced here rather than the value that "looks" right.
    mipMapK stays -8.0, same as RB3."""
    w.u32(TBRB_TEX_REVISION)        # 10
    write_object_fields(w)          # revision > 8 -> objFields
    w.u32(width)
    w.u32(height)
    w.u32(bpp)
    w.symbol(external_path)
    w.f32(-8.0)                     # mipMapK (revision >= 8)
    w.u32(TEX_TYPE_REGULAR)         # type (revision > 6)
    # revision 10 < 11 -> NO optimizeForPS3 here (this is the whole structural delta)
    w.boolean(True)                 # useExternalPath - retail TBRB ships 1, see docstring
    write_rnd_bitmap(w, width, height, encoding, bpp, block_data, platform=platform,
                      num_mips=num_mips)
    w.block(END_MARKER)


def write_tbrb_rnd_mat(w: MiloWriter, settings, standalone=True, force_tbrb_defaults=False):
    """RndMat.Write at revision 55 (The Beatles: Rock Band). Field order verified by
    decoding all 12 retail materials - every one lands exactly on its object boundary.

    Takes the SAME settings dict gather_materials_and_textures() already produces for the
    RB3 path, so material authoring in the UI is shared between games.

    Deltas from the RB3 rev-68 writer, all revision-gated:
      - rev 55 is NOT > 60, so unkBool_0x3C / unkBool_0x42 are absent.
      - rev 55 IS < 63, so projLights IS present (RB3 drops it).
      - rev 55 falls in the 52..67 window, so a trailing block exists that RB3 has none of:
        unkInt3 (rev >= 53), unkColor2 (rev 53..59), unkSym2 (rev 54..61),
        ps3ForceTrilinear (rev 55..62).
      - rev 55 is NOT > 63, so the refract block is absent - refraction settings authored
        in the UI are silently unavailable on TBRB.
    shaderVariation in retail: 1 (kShaderVariationSkin) on face/hands, 2 on nails,
    0 on clothing.

    `force_tbrb_defaults`: the Blender material UI was authored for RB3, and several of its
    default values render a TBRB character solid black even though the file is structurally
    correct (confirmed by bisection - our RB3-defaulted export was black; a variant with
    these specific fields set to retail TBRB values rendered correctly). When True, the
    render-affecting fields below are overridden with byte-verified known-good values (from
    the confirmed-working Body_fix_ALL material) instead of the UI's RB3 defaults. This is a
    TESTING aid to prove the material writer is correct independent of UI value sync; the
    long-term fix is to expose TBRB-appropriate values in the UI (and gray out RB3-only
    controls like Refract Settings for pre-RB3 games). Texture names, base_color, blend,
    and z_mode still come from the authored material either way."""
    s = dict(settings)   # copy so overrides don't mutate the caller's dict
    if force_tbrb_defaults:
        s['cull'] = True
        s['specular_power'] = 2.0
        s['point_lights'] = False
        s['color_adjust'] = True
        s['rim_rgb'] = (0.40784, 0.36078, 0.29020)
        s['rim_power'] = 4.0
        s['rim_always_show'] = True
        s['specular2_power'] = 10.0
    w.u32(TBRB_MAT_REVISION)        # 55
    write_object_fields(w)
    w.i32(MAT_BLEND_VALUES[s['blend']])
    for c in s['base_color']:
        w.f32(c)
    # Order is prelit THEN use_environment - verified against PikminGuts92's re-notes
    # mat.bt 010 template ("Bool prelit;" immediately precedes "Bool use_environ;") and
    # matches the working RB3 rev-68 writer. These two were previously swapped here, which
    # inverted both fields on export: a material authored Pre-Lit ON / Use-Environment OFF
    # was written as prelit=0 / use_environ=1, disabling the vertex-color base lighting and
    # instead enabling environment modulation - rendering the character solid black in-game
    # despite a valid diffuse texture. Confirmed by byte-parsing a known-good TBRB milo
    # (prelit=1, use_environ=0) against our broken export (prelit=0, use_environ=1).
    w.boolean(s['pre_lit'])
    w.boolean(s['use_environment'])
    w.i32(MAT_ZMODE_VALUES[s['z_mode']])
    w.boolean(s['alpha_cut'])
    w.i32(s['alpha_threshold'])                 # rev > 37
    w.boolean(s['blend'] == 'kBlendSrcAlpha')   # alphaWrite - mirrors the RB3 path
    w.i32(MAT_TEXGEN_NONE)
    w.i32(MAT_TEXWRAP_REPEAT)
    write_matrix(w, IDENTITY_MATRIX)            # texXfm
    w.symbol(s.get('diffuse_tex_name', ""))
    w.symbol("")                                # nextPass
    w.boolean(s.get('intensify', False))
    w.boolean(s.get('cull', False))
    w.f32(float(s['emissive_multiplier']))
    for c in s['specular_rgb']:
        w.f32(c)
    w.f32(float(s['specular_power']))
    w.symbol(s.get('normal_tex_name', ""))
    w.symbol(s.get('emissive_tex_name', ""))
    w.symbol(s.get('specular_tex_name', ""))
    # rev 55 >= 51 -> no unkSymbol2 here
    # environment_map is a BoolProperty, not a symbol name. RB3 maps it to the
    # "instruments.cube" asset, but that is an RB3 asset and every retail TBRB material
    # ships an EMPTY environMap, so there is no cube map to point at here.
    w.symbol("")                                # environMap
    # rev 55 is NOT > 60 -> no unkBool_0x3C / unkBool_0x42
    w.boolean(True)                             # perPixelLit (rev > 25)
    w.i32(MAT_STENCIL_IGNORE)                   # rev > 27
    w.symbol("")                                # fur
    w.f32(float(s.get('de_normal', 0)))         # rev > 35 - EnumProperty, cast like RB3 does
    w.f32(float(s.get('anisotropy', 0.0)))
    w.f32(1.0)                                  # normalDetailTiling  (rev > 38)
    w.f32(float(s.get('normal_detail_strength', 0.0)))
    w.symbol("")                                # normalDetailMap
    # pointLights: all three retail TBRB materials in the reference milo carry 0, not the
    # True that Program.cs uses for RB3. Match retail - forcing point-light reception on
    # differs from every shipped TBRB material and is not what the game's shaders expect here.
    w.boolean(bool(s.get('point_lights', False)))  # pointLights (rev > 42, >= 45)
    w.boolean(True)                             # projLights  (rev < 63) - absent on RB3
    w.boolean(False)                            # fog        - always false (Program.cs 671)
    w.boolean(False)                            # fadeout    - CLI leaves at C# default
    w.boolean(bool(s.get('color_adjust', False)))  # colorAdjust (rev > 46)
    for c in s.get('rim_rgb', (0.0, 0.0, 0.0)):  # rev > 47
        w.f32(c)
    w.f32(float(s.get('rim_power', 4.0)))
    w.symbol("")                                # rimMap
    w.boolean(bool(s.get('rim_always_show', False)))  # rimAlwaysShow
    w.boolean(False)                            # screenAligned (rev > 48)
    w.i32(MAT_SHADER_VARIATION_VALUES[s.get('shader_variation', 'kShaderVariationNone')])
    for _ in range(3):                          # specular2RGB
        w.f32(0.0)
    w.f32(float(s.get('specular2_power', 0.0)))  # specular2Power (rev > 50)
    # --- rev 52..67 tail. Field identities and defaults are from PikminGuts92's re-notes
    # mat.bt: val_0x160 (rev >= 53, "Default: 1.0") then val_0x170..0x17c (rev > 54,
    # "Default: (1.0, 1.0, 1.0, 0.0)") then alpha_mask NumString (rev > 53) then
    # ps3_force_trilinear (rev > 54). The previous version of this writer mislabeled these
    # as unkInt3/unkColor2/unkSym2 and, critically, wrote val_0x160 as an int32(0) where a
    # float 1.0 belongs, AND emitted an extra stray symbol - desyncing the whole tail. The
    # byte-exact retail Miku_body.mat carries val_0x160=1.0 and val_0x170 block=(0,1,1,1),
    # so those exact values are reproduced here rather than the template's stated generic
    # defaults, since retail is the ground truth for what the game actually accepts.
    w.f32(1.0)                                  # val_0x160  (rev >= 53) - retail: 1.0
    w.f32(0.0)                                  # val_0x170  \
    w.f32(1.0)                                  # val_0x174   } (rev > 54) - retail: (0,1,1,1)
    w.f32(1.0)                                  # val_0x178   |
    w.f32(1.0)                                  # val_0x17c  /
    w.symbol("")                                # alpha_mask (ScreenMask) (rev > 53)
    w.boolean(False)


TBRB_CUSTOM_SONG_MAT_REVISION = 28


def write_tbrb_custom_song_rnd_mat(w: MiloWriter, settings, standalone=True):
    """RndMat.Write at revision 28 - the format actually proven to work for the TBRB
    Custom Song Asset loose-file workflow specifically (as opposed to write_tbrb_rnd_mat's
    revision 55, which matches a real vanilla PS3 character/instrument material dumped from
    inside an actual retail milo, but has NOT been confirmed to load through the loose
    'extras' folder + custom-song-compiler pipeline the way this revision has).

    Byte-verified field-for-field against a real reference material confirmed to load
    correctly through a custom song (a mouth-harmonica prop material, source compiler
    unknown, definitely not the glTFMilo CLI). Every field below sits at the exact same
    offset as that 166-byte reference. Two structural deltas from write_tbrb_rnd_mat's
    revision-55 layout, both because rev 28 predates the fields that were added later:
      - NO alpha_threshold field at all (that field is gated "revision > 37" in the
        higher-revision writer; 28 is below that, and the reference file's diffuse_tex
        symbol lands exactly 4 bytes earlier than it would if alpha_threshold were
        present - i.e. removing it, not zeroing it, is what makes the layout line up).
      - NO trailing END_MARKER, same as every other TBRB standalone asset type
        investigated here (RndMesh, and revision-55 RndMat) - the reference file ends
        immediately after its last byte with nothing appended.

    The reference file's tail (everything after specular_tex - presumably some
    combination of environMap/unkShort/perPixelLit/stencilMode/fur equivalents at this
    older revision) is 14 bytes, ALL ZERO. There's no way to recover individual field
    boundaries from an all-zero sample, so rather than guess names/types that could be
    wrong, this writes that tail as one literal 14-byte zero block - it reproduces the
    confirmed-working file exactly regardless of how those bytes actually subdivide.
    If a future reference ever needs one of those trailing fields to be non-default
    (e.g. a real per_pixel_lit=False or a real environMap), this block will need
    unpacking into real named fields at that point.

    `settings` only needs to supply the fields this revision actually uses: blend,
    base_color, pre_lit, use_environment, z_mode, alpha_cut, diffuse_tex_name,
    intensify, cull, emissive_multiplier, specular_rgb, specular_power,
    normal_tex_name, specular_tex_name. Every other key gather_materials_and_textures()
    populates (rim lighting, environment map, refraction, PS3 perf settings, etc.) is
    simply not part of this older format and is ignored here."""
    w.u32(TBRB_CUSTOM_SONG_MAT_REVISION)   # 28
    write_object_fields(w)
    w.i32(MAT_BLEND_VALUES[settings['blend']])
    for c in settings['base_color']:
        w.f32(c)
    w.boolean(settings['pre_lit'])
    w.boolean(settings['use_environment'])
    w.i32(MAT_ZMODE_VALUES[settings['z_mode']])
    w.boolean(settings['alpha_cut'])
    # NO alpha_threshold here - gated off below revision 37 in the newer writer, and
    # its absence is exactly what makes every field after this line line up with the
    # reference file.
    w.boolean(settings['blend'] == 'kBlendSrcAlpha')   # alphaWrite
    w.i32(MAT_TEXGEN_NONE)
    w.i32(MAT_TEXWRAP_REPEAT)
    write_matrix(w, IDENTITY_MATRIX)                   # texXfm
    w.symbol(settings.get('diffuse_tex_name', ""))
    w.symbol("")                                       # nextPass
    w.boolean(settings.get('intensify', False))
    w.boolean(settings.get('cull', False))
    w.f32(float(settings['emissive_multiplier']))
    for c in settings['specular_rgb']:
        w.f32(c)
    w.f32(float(settings['specular_power']))
    w.symbol(settings.get('normal_tex_name', ""))
    w.symbol(settings.get('emissive_tex_name', ""))
    w.symbol(settings.get('specular_tex_name', ""))
    # Unrecovered all-zero tail from the reference file - see docstring.
    w.block(b"\x00" * 14)


RB3_MAT_REVISION = 68


DC3_MAT_REVISION = 0x46


DC3_BASEMAT_REVISION = 8


DC3_OBJ_REVISION = 2


DC3_DEFAULT_META_MATERIAL = "char_basic_skin.mmat"


DC3_TEXXFM_IDENTITY = (1.0, 0.0, -0.0, -0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)


RB3_TEX_REVISION = 11


RB3_BITMAP_REVISION = 1


TEX_ENCODING_DXT1 = 8


TEX_ENCODING_DXT5 = 24


TEX_ENCODING_ATI2 = 32


TEX_TYPE_REGULAR = 1


_TEX_ENCODING_NAMES.update({
    TEX_ENCODING_DXT1: "DXT1/BC1",
    TEX_ENCODING_DXT5: "DXT5/BC3",
    TEX_ENCODING_ATI2: "ATI2/BC5",
})


def _color_to_565(r, g, b):
    return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)


def _565_to_rgb(c):
    r = (c >> 11) & 0x1F
    g = (c >> 5) & 0x3F
    b = c & 0x1F
    r = (r << 3) | (r >> 2)
    g = (g << 2) | (g >> 4)
    b = (b << 3) | (b >> 2)
    return r, g, b


def _pca_axis_3(pixels):
    """Returns (mean, axis) for a list of (r,g,b) tuples - the block's mean color and
    the dominant principal axis (unit vector) of its color distribution, found via a
    few iterations of power iteration on the 3x3 covariance matrix. This is the same
    class of approach real BC1 encoders use (PCA / cluster-fit) to pick a color line
    through the ACTUAL pixel distribution, instead of independently taking the max/min
    of each channel (which can synthesize an endpoint color that doesn't resemble any
    real pixel in the block - see _encode_bc1_block's docstring)."""
    n = len(pixels)
    mr = sum(p[0] for p in pixels) / n
    mg = sum(p[1] for p in pixels) / n
    mb = sum(p[2] for p in pixels) / n
    cov_rr = cov_rg = cov_rb = cov_gg = cov_gb = cov_bb = 0.0
    for (r, g, b) in pixels:
        dr, dg, db = r - mr, g - mg, b - mb
        cov_rr += dr * dr
        cov_rg += dr * dg
        cov_rb += dr * db
        cov_gg += dg * dg
        cov_gb += dg * db
        cov_bb += db * db

    # Seed power iteration along whichever channel has the widest range - converges
    # in very few iterations for the small, low-rank 4x4-pixel color distributions
    # found in a single block.
    r_range = max(p[0] for p in pixels) - min(p[0] for p in pixels)
    g_range = max(p[1] for p in pixels) - min(p[1] for p in pixels)
    b_range = max(p[2] for p in pixels) - min(p[2] for p in pixels)
    if r_range >= g_range and r_range >= b_range:
        ax, ay, az = 1.0, 0.0, 0.0
    elif g_range >= b_range:
        ax, ay, az = 0.0, 1.0, 0.0
    else:
        ax, ay, az = 0.0, 0.0, 1.0

    for _ in range(8):
        nx = cov_rr * ax + cov_rg * ay + cov_rb * az
        ny = cov_rg * ax + cov_gg * ay + cov_gb * az
        nz = cov_rb * ax + cov_gb * ay + cov_bb * az
        length = (nx * nx + ny * ny + nz * nz) ** 0.5
        if length < 1e-9:
            break
        ax, ay, az = nx / length, ny / length, nz / length

    return (mr, mg, mb), (ax, ay, az)


_BC1_INDEX_C0_WEIGHT = (1.0, 0.0, 2.0 / 3.0, 1.0 / 3.0)


def _refine_bc1_endpoints(pixels, indices):
    """Given a fixed per-pixel index assignment (0-3), solves the 3x2 (per RGB
    channel) least-squares problem for the two endpoint colors that best reconstruct
    the block under that assignment - i.e. the true globally-optimal endpoints for
    THIS clustering, rather than the block's raw min/max. This is the same
    "endpoint refinement" step real cluster-fit BC1 encoders use after an initial
    PCA-based guess. Returns (c0_565, c1_565), or None if the system is degenerate
    (e.g. every pixel mapped to the same index, so there's nothing to solve for)."""
    ts = [_BC1_INDEX_C0_WEIGHT[(indices >> (2 * i)) & 0x3] for i in range(16)]
    sum_tt = sum(t * t for t in ts)
    sum_t1mt = sum(t * (1.0 - t) for t in ts)
    sum_1mt1mt = sum((1.0 - t) * (1.0 - t) for t in ts)
    det = sum_tt * sum_1mt1mt - sum_t1mt * sum_t1mt
    if abs(det) < 1e-6:
        return None

    c0 = [0, 0, 0]
    c1 = [0, 0, 0]
    for ch in range(3):
        sum_p_t = sum(ts[i] * pixels[i][ch] for i in range(16))
        sum_p_1mt = sum((1.0 - ts[i]) * pixels[i][ch] for i in range(16))
        c0v = (sum_p_t * sum_1mt1mt - sum_p_1mt * sum_t1mt) / det
        c1v = (sum_tt * sum_p_1mt - sum_t1mt * sum_p_t) / det
        c0[ch] = max(0, min(255, round(c0v)))
        c1[ch] = max(0, min(255, round(c1v)))

    return _color_to_565(*c0), _color_to_565(*c1)


def _encode_bc1_block(pixels, force_four_color=False):
    """pixels: 16 (r,g,b,a) tuples, 0-255. Returns 8 bytes.

    Endpoint selection previously took the max/min of each RGB channel
    INDEPENDENTLY across the block (max(rs), max(gs), max(bs)) - three values that
    can come from three different pixels, so the resulting "endpoint color" may not
    resemble any real pixel in the block at all. That's a known-inferior technique
    (this file's own earlier comments already flagged it as "not the highest
    quality - no cluster-fit optimization"), and byte-for-byte comparison against a
    real CLI-exported reference texture confirmed it measurably: a "blockiness"
    metric (edge energy at 4x4 block boundaries vs interior, in a smooth skin-tone
    gradient) came out visibly higher for this encoder's output than the CLI's.

    Replaced with PCA-based endpoint selection (fit a line through the actual 3D
    color distribution of the block via _pca_axis_3, take the extreme projections
    along it as the two endpoints), optionally followed by a least-squares
    refinement pass (_refine_bc1_endpoints) that re-solves for the best endpoint
    colors given the resulting index assignment - the same two-stage approach real
    cluster-fit BC1 encoders use, just without further iterating to full
    convergence.

    Both the PCA step and the refinement step are only ever KEPT if they measure a
    lower actual quantized reconstruction error than the alternative - confirmed
    necessary empirically: naive-endpoints-plus-refinement measured WORSE than
    plain naive endpoints on real texture content in some regions (least-squares
    is only optimal in continuous space; rounding to 5/6/5 bits can undo the
    improvement, especially on near-flat blocks where naive's own tie-break "+1"
    perturbation already dominates the result). Evaluating candidates against each
    other directly, rather than trusting any one approach unconditionally, is what
    makes this a guaranteed non-regression versus the original naive-only
    encoder - each candidate costs an extra index-assignment pass, so this is
    slower than a single fixed strategy, but still linear in block count."""
    has_transparent = (not force_four_color) and any(p[3] < 128 for p in pixels)

    def _order(c0_565, c1_565):
        if has_transparent:
            if c0_565 > c1_565:
                c0_565, c1_565 = c1_565, c0_565
        else:
            if c0_565 < c1_565:
                c0_565, c1_565 = c1_565, c0_565
            elif c0_565 == c1_565 and c0_565 < 0xFFFF:
                c0_565 += 1
        return c0_565, c1_565

    def _build_palette(c0_565, c1_565):
        r0, g0, b0 = _565_to_rgb(c0_565)
        r1, g1, b1 = _565_to_rgb(c1_565)
        if c0_565 > c1_565:
            return [(r0, g0, b0), (r1, g1, b1),
                    ((2*r0+r1)//3, (2*g0+g1)//3, (2*b0+b1)//3),
                    ((r0+2*r1)//3, (g0+2*g1)//3, (b0+2*b1)//3)]
        return [(r0, g0, b0), (r1, g1, b1),
                ((r0+r1)//2, (g0+g1)//2, (b0+b1)//2), (0, 0, 0)]

    def _assign_indices(palette):
        indices = 0
        for i, p in enumerate(pixels):
            if has_transparent and p[3] < 128:
                idx = 3
            else:
                best_idx, best_dist = 0, None
                for pi in range(4):
                    if has_transparent and pi == 3:
                        continue
                    pc = palette[pi]
                    d = (p[0]-pc[0])**2 + (p[1]-pc[1])**2 + (p[2]-pc[2])**2
                    if best_dist is None or d < best_dist:
                        best_dist, best_idx = d, pi
                idx = best_idx
            indices |= (idx << (2*i))
        return indices

    def _block_error(palette, indices):
        err = 0
        for i, p in enumerate(pixels):
            idx = (indices >> (2*i)) & 0x3
            pc = palette[idx]
            err += (p[0]-pc[0])**2 + (p[1]-pc[1])**2 + (p[2]-pc[2])**2
        return err

    def _candidate(c0_565, c1_565):
        c0_565, c1_565 = _order(c0_565, c1_565)
        palette = _build_palette(c0_565, c1_565)
        indices = _assign_indices(palette)
        return _block_error(palette, indices), c0_565, c1_565, indices

    rs = [p[0] for p in pixels]
    gs = [p[1] for p in pixels]
    bs = [p[2] for p in pixels]
    naive_c0 = _color_to_565(max(rs), max(gs), max(bs))
    naive_c1 = _color_to_565(min(rs), min(gs), min(bs))
    best = _candidate(naive_c0, naive_c1)

    fit_pixels = [(p[0], p[1], p[2]) for p in pixels if not (has_transparent and p[3] < 128)]
    if not fit_pixels:
        fit_pixels = [(p[0], p[1], p[2]) for p in pixels]

    if not all(p == fit_pixels[0] for p in fit_pixels):
        mean, axis = _pca_axis_3(fit_pixels)
        projs = [(p[0] - mean[0]) * axis[0] + (p[1] - mean[1]) * axis[1] + (p[2] - mean[2]) * axis[2]
                 for p in fit_pixels]
        t_max, t_min = max(projs), min(projs)

        def _endpoint(t):
            r = max(0, min(255, round(mean[0] + t * axis[0])))
            g = max(0, min(255, round(mean[1] + t * axis[1])))
            b = max(0, min(255, round(mean[2] + t * axis[2])))
            return r, g, b

        pca_c0 = _color_to_565(*_endpoint(t_max))
        pca_c1 = _color_to_565(*_endpoint(t_min))
        pca_candidate = _candidate(pca_c0, pca_c1)
        if pca_candidate[0] < best[0]:
            best = pca_candidate

    if not has_transparent:
        # Refinement only applies to the 4-unique-color palette ordering (index 3
        # is a real interpolated color there, not the punch-through-alpha "black"
        # slot used in transparent mode, so the linear system's weights above are
        # only valid for this branch). Refine on top of whichever candidate (naive
        # or PCA) is currently winning, using ITS index assignment.
        _, _, _, best_indices = best
        refined = _refine_bc1_endpoints([(p[0], p[1], p[2]) for p in pixels], best_indices)
        if refined is not None:
            refined_candidate = _candidate(*refined)
            if refined_candidate[0] < best[0]:
                best = refined_candidate

    _, c0_565, c1_565, indices = best
    return struct.pack('<HHI', c0_565, c1_565, indices)


def _encode_alpha_block(values):
    """values: 16 ints 0-255. BC3/BC5-style single-channel block -> 8 bytes."""
    a0, a1 = max(values), min(values)
    if a0 == a1:
        if a0 < 255:
            a0 += 1
        else:
            a1 -= 1
    palette = [a0, a1] + [round(((7-i)*a0 + i*a1) / 7) for i in range(1, 7)]
    indices = 0
    for i, v in enumerate(values):
        best_idx, best_dist = 0, None
        for pi, pv in enumerate(palette):
            d = abs(v - pv)
            if best_dist is None or d < best_dist:
                best_dist, best_idx = d, pi
        indices |= (best_idx << (3*i))
    return bytes([a0, a1]) + indices.to_bytes(6, 'little')


def _iter_blocks(width, height):
    for by in range(0, height, 4):
        for bx in range(0, width, 4):
            yield bx, by


def encode_bc1(rgba, width, height):
    """rgba: flat bytes, width*height*4 (RGBA8). Returns raw BC1/DXT1 block data."""
    out = bytearray()
    for bx, by in _iter_blocks(width, height):
        block = []
        for y in range(4):
            for x in range(4):
                px = min(bx + x, width - 1)
                py = min(by + y, height - 1)
                idx = (py * width + px) * 4
                block.append(tuple(rgba[idx:idx + 4]))
        out += _encode_bc1_block(block)
    return bytes(out)


def encode_bc3(rgba, width, height):
    """rgba: flat bytes, width*height*4 (RGBA8). Returns raw BC3/DXT5 block data."""
    out = bytearray()
    for bx, by in _iter_blocks(width, height):
        block, alphas = [], []
        for y in range(4):
            for x in range(4):
                px = min(bx + x, width - 1)
                py = min(by + y, height - 1)
                idx = (py * width + px) * 4
                p = tuple(rgba[idx:idx + 4])
                block.append(p)
                alphas.append(p[3])
        out += _encode_alpha_block(alphas)
        out += _encode_bc1_block(block, force_four_color=True)
    return bytes(out)


def encode_bc5(xy, width, height):
    """xy: flat bytes, width*height*2 (X,Y per pixel, 0-255). Returns raw BC5/ATI2 block data."""
    out = bytearray()
    for bx, by in _iter_blocks(width, height):
        xs, ys = [], []
        for y in range(4):
            for x in range(4):
                px = min(bx + x, width - 1)
                py = min(by + y, height - 1)
                idx = (py * width + px) * 2
                xs.append(xy[idx])
                ys.append(xy[idx + 1])
        out += _encode_alpha_block(xs)
        out += _encode_alpha_block(ys)
    return bytes(out)


def _encode_texture_level(rgba, width, height, encoder):
    """Encode a single RGBA8 level to BC block data. Returns (block_data, encoding, bpp).

    Factored out of register_texture so every mip level uses the identical encoding path.
    """
    if encoder == 'ati2':
        xy = bytes(b for i, b in enumerate(rgba) if i % 4 in (0, 1))  # R,G interleaved -> X,Y
        return encode_bc5(xy, width, height), TEX_ENCODING_ATI2, 8
    elif encoder == 'bc3':
        return encode_bc3(rgba, width, height), TEX_ENCODING_DXT5, 8
    else:
        return encode_bc1(rgba, width, height), TEX_ENCODING_DXT1, 4


def _halve_rgba8(rgba, width, height):
    """2x box-filter downsample of an RGBA8 buffer. width and height must be even.
    Returns (new_width, new_height, new_rgba)."""
    nw, nh = width // 2, height // 2
    out = bytearray(nw * nh * 4)
    for y in range(nh):
        r0 = (2 * y) * width
        r1 = (2 * y + 1) * width
        for x in range(nw):
            p00 = (r0 + 2 * x) * 4
            p10 = (r0 + 2 * x + 1) * 4
            p01 = (r1 + 2 * x) * 4
            p11 = (r1 + 2 * x + 1) * 4
            o = (y * nw + x) * 4
            for c in range(4):
                out[o + c] = (rgba[p00 + c] + rgba[p10 + c] + rgba[p01 + c] + rgba[p11 + c] + 2) >> 2
    return nw, nh, bytes(out)


def build_texture_mip_chain(rgba, width, height, encoder, generate_mips, mip_floor=4):
    """Encode a texture and, if generate_mips, a mip chain down to mip_floor (default 4x4,
    BC's smallest block).

    Returns (concatenated_block_data, encoding, bpp, num_extra_mips).

    num_extra_mips is exactly what the RndBitmap header's mipMaps byte stores: the number of
    levels BELOW the base. RndBitmap.NumMips() in the decompiled engine returns the length of
    the mMip linked list, i.e. it does NOT count the base level (a single-level texture stores
    0). The pixel data is laid out base -> smallest, concatenated - matching how the game and a
    vanilla DDS store it - so the caller's single Xbox 2-byte-pair swap covers the whole blob.

    Why this exists (Dance Central 1): a vanilla DC1 normal map ships a full ATI2/BC5 mip chain
    (angel's is 512->16, 6 levels). DC1's DXN sampling path renders a mip-less ATI2 surface
    BLACK, while RB3 renders the byte-identical single-level texture fine, and DXT1/DXT5
    (diffuse/spec) render mip-less on both. Supplying the chain is the fix for DC1's black
    normal-mapped characters; it is left off for RB3, which does not need it.

    mip_floor default here is 4 for backwards compatibility with existing non-DC1 callers,
    but DC1 itself should always pass DC1_TEX_MIP_FLOOR (16) - see that constant's comment.
    """
    base_block, encoding, bpp = _encode_texture_level(rgba, width, height, encoder)
    if not generate_mips:
        return base_block, encoding, bpp, 0
    chunks = [base_block]
    w, h, cur = width, height, rgba
    # Halve while the NEXT level stays >= mip_floor in both axes.
    #
    # mip_floor exists because the games disagree. DC1 ALSO stops at 16, exactly like TBRB
    # below - NOT at 4x4 as an earlier version of this comment incorrectly claimed. Byte-
    # parsing all 24 real textures in retail emilia01.milo_xbox (DC1) confirms every one
    # bottoms out at a 16px smaller dimension (256x256 ATI2 normal -> mipMaps=4, 512x512 ->
    # 5, 128x128 -> 3, 512x128 -> 3/64x16), never at 4x4. See DC1_TEX_MIP_FLOOR.
    # TBRB stops at 16: every retail TBRB texture bottoms out at a 16px smaller dimension
    # (512x512 -> mipMaps=5, 256x256 -> 4, 512x256 -> 4, 64x64 -> 2), never at 4. Running
    # a TBRB chain down to 4x4 would ship three levels the retail files never carry.
    while w >= mip_floor * 2 and h >= mip_floor * 2:
        w, h, cur = _halve_rgba8(cur, w, h)
        level_block, _, _ = _encode_texture_level(cur, w, h, encoder)
        chunks.append(level_block)
    return b''.join(chunks), encoding, bpp, len(chunks) - 1


def xbox_byte_swap(data):
    """RndBitmap.ConvertToImage()'s Xbox swap, applied in reverse for writing: every
    4-byte group gets its two 16-bit halves byte-swapped: (b0,b1,b2,b3) -> (b1,b0,b3,b2)."""
    out = bytearray(len(data))
    for i in range(0, len(data) - 3, 4):
        out[i] = data[i+1]
        out[i+1] = data[i]
        out[i+2] = data[i+3]
        out[i+3] = data[i+2]
    return bytes(out)


def write_rnd_bitmap(w: MiloWriter, width, height, encoding, bpp, block_data, platform='xbox360',
                      num_mips=0):
    """Assets/Rnd/RndBitmap.cs Write(), fixed at revision 1 (RB3).

    num_mips is the count of levels BELOW the base (RndBitmap.NumMips() = mMip chain length),
    and block_data must already contain base + those mip levels concatenated (base -> smallest).
    0 = a single base level, the historical behaviour.

    The 2-byte-pair byte-swap is XBOX-ONLY. Confirmed from two independent sources:
    MiloLib's RndBitmap only swaps `if (platform == Xbox)` in ConvertToImage, and the
    glTFMilo CLI only swaps the DXT block bytes when `meta.platform == Xbox` (Program.cs),
    storing them verbatim for PS3. So PS3 stores standard little-endian DXT blocks with
    no swap."""
    w.u8(RB3_BITMAP_REVISION)
    w.u8(bpp)
    w.u32(encoding)
    w.u8(num_mips)          # mipMaps = number of levels below the base (0 = base only, the
                             # CLI's GenerateMipmaps=false default). Dance Central 1 requires a
                             # real chain here for ATI2/BC5 normal maps or they render black.
    w.u16(width)
    w.u16(height)
    w.u16((width * bpp) // 8)  # bpl (bytes per line) - was hardcoded to 0, confirmed
                                # wrong against a real file (512-wide DXT1 has bpl=256,
                                # exactly width*bpp/8). Very likely the actual cause of
                                # textures rendering as garbage/another character's data
                                # in-game: if the engine uses this as the row stride/pitch
                                # when uploading pixel data to the GPU, a stride of 0 would
                                # make that copy degenerate, leaving GPU memory holding
                                # whatever was already there from a previously loaded
                                # texture - exactly matching the reported symptom.
    w.u16(0)                # wiiAlphaNum
    w.block(bytes(17))      # padding (revision != 2)
    if platform == 'ps3':
        w.block(block_data)             # PS3: store DXT blocks verbatim (no swap)
    else:
        w.block(xbox_byte_swap(block_data))


def write_rnd_tex(w: MiloWriter, width, height, encoding, bpp, block_data, standalone=True,
                   external_path="", platform='xbox360', num_mips=0):
    """Assets/Rnd/RndTex.cs Write(), fixed at revision 11 (RB3). Embeds the bitmap
    directly (useExternalPath=False) rather than referencing an external file.

    mipMapK=-8.0 and optimizeForPS3=True are NOT platform-conditional despite the
    field name - confirmed by directly comparing a working CLI-exported .tex against
    ours: the CLI hardcodes both unconditionally for every texture (Program.cs sets
    tex.mipMapK = -8.0f and tex.optimizeForPS3 = true for every single texture type,
    Xbox included). Leaving mipMapK at 0.0 - the field's C# default, which seemed like
    a reasonable value for "no mips" - is suspected to be why textures rendered grey/
    corrupted in-game: the GPU likely uses this as an LOD/mip-bias hint, and with only
    one real mip level stored, an LOD bias expecting more mips could sample garbage."""
    w.u32(RB3_TEX_REVISION)         # (altRevision=0 << 16) | 11

    write_object_fields(w)          # base.Write(standalone=False) -> objFields (revision(11) > 8)

    w.u32(width)
    w.u32(height)
    w.u32(bpp)
    w.symbol(external_path)         # externalPath - purely informational since useExternalPath=False,
                                     # but the CLI sets it anyway (e.g. "MaterialName.png")
    w.f32(-8.0)                     # mipMapK (revision >= 8) - see docstring
    w.u32(TEX_TYPE_REGULAR)         # type (revision > 6)
    w.boolean(True)                 # optimizeForPS3 (revision >= 11) - see docstring
    w.boolean(False)                # useExternalPath (revision != 7) - embed the bitmap

    write_rnd_bitmap(w, width, height, encoding, bpp, block_data, platform=platform,
                      num_mips=num_mips)

    if standalone:
        w.block(END_MARKER)


MAT_BLEND_VALUES = {
    'kBlendDest': 0, 'kBlendSrc': 1, 'kBlendAdd': 2, 'kBlendSrcAlpha': 3,
    'kBlendSrcAlphaAdd': 4, 'kBlendSubtract': 5, 'kBlendMultiply': 6, 'kPreMultAlpha': 7,
}


MAT_ZMODE_VALUES = {
    'kZModeDisable': 0, 'kZModeNormal': 1, 'kZModeTransparent': 2, 'kZModeForce': 3, 'kZModeDecal': 4,
}


MAT_STENCIL_IGNORE = 0


MAT_TEXGEN_NONE = 0


MAT_TEXWRAP_REPEAT = 1


MAT_SHADER_VARIATION_NONE = 0


MAT_SHADER_VARIATION_VALUES = {
    'kShaderVariationNone': 0, 'kShaderVariationSkin': 1, 'kShaderVariationHair': 2,
}


IDENTITY_MATRIX_UNUSED = IDENTITY_MATRIX


def write_rnd_mat(w: MiloWriter, settings, standalone=True):
    """Assets/Rnd/RndMat.cs Write(), fixed at revision 68 (RB3). `settings` is a dict
    with keys matching GltfMiloMaterialSettings: blend, base_color, pre_lit,
    use_environment, z_mode, alpha_cut, alpha_threshold, specular_rgb, specular_power,
    emissive_multiplier, environment_map, cull, de_normal, anisotropy,
    normal_detail_strength, rim_rgb, rim_power, shader_variation, refract_enabled,
    refract_strength (texture symbols - diffuse/normal/specular/emissive/
    refract_normal_map - default to "" if not set).

    Every field not yet exposed in the Blender UI uses glTFMilo's own CLI defaults,
    read directly from Source/glTFMilo/Program.cs's unconditional mat.* assignments -
    see the comments below for exactly which line each default came from."""
    w.u32(RB3_MAT_REVISION)  # (altRevision=0 << 16) | 68

    write_object_fields(w)   # base.Write(standalone=False) -> objFields (revision=2, same as everywhere else)

    w.i32(MAT_BLEND_VALUES[settings['blend']])
    write_color4(w, *settings['base_color'])
    w.boolean(settings['pre_lit'])
    w.boolean(settings['use_environment'])
    w.i32(MAT_ZMODE_VALUES[settings['z_mode']])
    w.boolean(settings['alpha_cut'])
    # revision(68) > 0x25(37): True
    w.i32(settings['alpha_threshold'])
    w.boolean(settings['blend'] == 'kBlendSrcAlpha')   # alphaWrite - not exposed yet; CLI only
                                                        # sets this true in its "BLEND" alpha-mode
                                                        # branch, which corresponds to this blend mode
    w.i32(MAT_TEXGEN_NONE)             # texGen (not exposed yet)
    w.i32(MAT_TEXWRAP_REPEAT)          # texWrap (not exposed yet - CLI's own fallback default)
    write_matrix(w, IDENTITY_MATRIX)   # texXfm (not exposed yet)
    w.symbol(settings.get('diffuse_tex_name', ''))
    w.symbol("")                       # nextPass
    w.boolean(settings.get('intensify', False))   # intensify - boosts (doubles) the base
                                                    # color for a stronger glow/bloom. Confirmed
                                                    # against neon_resource: every untextured
                                                    # bulb material (orange/red/blue/etc.) sets
                                                    # this True, while textured surfaces leave it
                                                    # False. This is the extra "pop" that makes a
                                                    # preLit bright-color surface read as a glowing
                                                    # neon bulb rather than just a brightly lit one.

    w.boolean(settings.get('cull', False))  # cull - now its own dedicated material
                                             # setting (GltfMiloMaterialSettings.cull)
                                             # for direct testing, rather than being
                                             # sourced from Blender's built-in
                                             # "Backface Culling" toggle
    w.f32(settings['emissive_multiplier'])
    write_color3(w, *settings['specular_rgb'])
    w.f32(settings['specular_power'])
    w.symbol(settings.get('normal_tex_name', ''))
    w.symbol(settings.get('emissive_tex_name', ''))
    w.symbol(settings.get('specular_tex_name', ''))
    # revision(68) < 51: False -> unkSymbol2 skipped
    w.symbol("instruments.cube" if settings['environment_map'] else "")  # environMap

    # revision > 25: True; revision == 68: True
    w.u16(0)                           # unkShort

    w.boolean(True)                    # perPixelLit - Program.cs line 664: always true

    # revision >= 27 and < 50: False (68 is outside) -> unkBool1 skipped

    w.i32(MAT_STENCIL_IGNORE)          # stencilMode - Program.cs line 662: always kStencilIgnore

    # revision < 33: False -> else:
    w.symbol("")                       # fur

    # revision >= 34 and < 49: False (68 outside) -> unkBool2/unkColor3/unkFloat3/unkSym3 skipped
    # revision <= 28: False -> no early return

    w.f32(float(settings.get('de_normal', 0)))   # deNormal - user-editable, restricted to -1/0/1
    w.f32(settings.get('anisotropy', 0.0))        # anisotropy - user-editable
    w.f32(1.0)                         # normalDetailTiling - Program.cs line 688
    w.f32(settings.get('normal_detail_strength', 0.0))  # normalDetailStrength - user-editable
    w.symbol("")                       # normalDetailMap

    w.boolean(True)                    # pointLights - Program.cs line 669: always true
    # revision < 0x3F(63): False (68 not < 63) -> projLights skipped
    w.boolean(False)                   # fog - Program.cs line 671: always false
    w.boolean(False)                   # fadeout - not set by CLI, C# default
    w.boolean(False)                   # colorAdjust - not set by CLI, C# default

    # revision > 47: True
    write_color3(w, *settings.get('rim_rgb', (0.0, 0.0, 0.0)))  # rimRGB - user-editable
    w.f32(settings.get('rim_power', 4.0))  # rimPower - user-editable
    w.symbol("")                       # rimMap
    w.boolean(False)                   # rimAlwaysShow

    # revision > 48: True
    w.boolean(False)                   # screenAligned

    # revision > 0x32(50): True
    w.i32(MAT_SHADER_VARIATION_VALUES[settings.get('shader_variation', 'kShaderVariationNone')])  # user-editable
    write_color3(w, 0.0, 0.0, 0.0)     # specular2RGB
    w.f32(0.0)                         # specular2Power - Program.cs line 694

    # revision >= 52 and <= 67: False (68 is outside) -> whole unkInt3/unkColor2/colors block skipped
    # revision >= 54 and <= 61: False -> unkSym2 skipped
    # revision >= 55 and <= 62: False -> perfSettings.ps3ForceTrilinear (standalone bool) skipped
    # revision == 0x38(56): False -> unkInt1/unkInt2 skipped

    # revision > 0x3E(62): True
    w.boolean(False)                   # perfSettings.recvProjLights
    w.boolean(False)                   # perfSettings.ps3ForceTrilinear
    # revision > 0x41(65): True
    w.boolean(False)                   # perfSettings.recvPointCubeTex

    # revision > 0x3F(63): True
    w.boolean(settings.get('refract_enabled', False))   # refractEnabled - user-editable
    w.f32(settings.get('refract_strength', 0.0))         # refractStrength - user-editable
    w.symbol(settings.get('refract_normal_map_tex_name', ''))  # refractNormalMap - user-editable

    if standalone:
        w.block(END_MARKER)


DC3_MAT_ALLOW_DISTORTION_EFFECTS = True


DC3_MAT_SHOCKWAVE_MULT = 1.0


DC3_MAT_WORLD_PROJECTION_TILING = 0.125


DC3_MAT_WORLD_PROJECTION_START_BLEND = 0.8


DC3_MAT_WORLD_PROJECTION_END_BLEND = 0.9


def _dc3_shader_variation_for_meta(meta_name, fallback='kShaderVariationNone'):
    """The material's shaderVariation must match the shading path of its MetaMaterial. Confirmed
    by byte-parsing retail emilia01: char_basic_skin*.mmat surfaces set kShaderVariationSkin(1),
    char_basic_hair*.mmat set kShaderVariationHair(2), and everything else (rim/lod/color/nospec)
    stays kShaderVariationNone(0). `fallback` is used when the name doesn't match a known family
    (e.g. a custom template) so an explicitly-authored shader variation isn't clobbered."""
    m = (meta_name or "").lower()
    if "char_basic_skin" in m:
        return 'kShaderVariationSkin'
    if "char_basic_hair" in m:
        return 'kShaderVariationHair'
    if m in ("char_basic_rim.mmat", "char_basic_lod.mmat", "char_basic_color.mmat"):
        return 'kShaderVariationNone'
    return fallback


def write_dc3_rnd_mat(w: MiloWriter, settings, standalone=True, force_dc3_defaults=False):
    """Assets/Rnd material writer for Dance Central 3, fixed at RndMat revision 0x46 (70).

    DC3 restructured the material into a class hierarchy (confirmed against rjkiv/dc3-decomp,
    src/system/rndobj/{Mat,BaseMaterial}.cpp) instead of RB3's single flat RndMat. The wire
    format is therefore:

        RndMat::Save         SAVE_REVS(0x46,0)              -> u32 0x00000046
                             SAVE_SUPERCLASS(BaseMaterial)
        BaseMaterial::Save     SAVE_REVS(8,0)              -> u32 0x00000008
                               SAVE_SUPERCLASS(Hmx::Object)
        Hmx::Object::Save        bs << 2 (SaveType)        -> u32 0x00000002
                                 bs << Type() (SaveType)   -> symbol ("")
                                 bs << (DataArray*)null    -> bool 0x00 (SaveRest)
                                 bs << 0  (empty note)     -> u32 0x00000000 (SaveRest)
                             ... BaseMaterial fields ...
                         bs << mMetaMaterial               -> symbol (the .mmat template)

    Two differences from the RB3 rev-68 writer matter and are exactly why an RB3-format
    material renders black on a normal-mapped DC3 mesh:

      1. Field ORDER differs. In DC3, specularRGB/normalMap/emissiveMap/specularMap are
         written as a contiguous run right after emissiveMultiplier, and there is NO separate
         specularPower float - the specular power is packed into specularRGB's alpha channel
         (Hmx::Color is 4 floats r,g,b,a; a == power). RB3 wrote 3-float specularRGB + a
         standalone specularPower float, which shifts normalMap 4 bytes late and derails every
         subsequent field on the DC3 loader.

      2. Every DC3 material carries a trailing MetaMaterial (.mmat) symbol. The runtime resolves
         it by name against a global template dir and calls UpdatePropertiesFromMetaMat(); a
         material without it (or read at the wrong offset) never gets its properties resolved.

    `settings` is the same dict write_rnd_mat consumes, plus optional 'meta_material' (the .mmat
    symbol; defaults to DC3_DEFAULT_META_MATERIAL). All fields not exposed in the Blender UI use
    dc3-decomp's own BaseMaterial constructor defaults (see the DC3_MAT_* constants above).

    `force_dc3_defaults`: like TBRB's force_tbrb_defaults, the Blender material UI was authored
    for RB3 and a couple of its render-affecting defaults are inverted relative to what DC3
    materials actually use. EVERY retail emilia01 material (byte-parsed from the real milo)
    writes prelit=0 and useEnviron=1; the RB3 UI defaults to the opposite (prelit=1,
    useEnviron=0), which disables environment lighting on a DC3 character. When True, prelit and
    useEnviron are snapped to the retail DC3 convention, AND shaderVariation is derived from the
    MetaMaterial name so the two always agree (see _dc3_shader_variation_for_meta): retail pairs
    char_basic_skin*.mmat with kShaderVariationSkin(1) and char_basic_hair*.mmat with
    kShaderVariationHair(2); a mismatch between the template's shading path and the material's
    shaderVariation is its own rendering bug. Texture names, base_color, blend, z_mode, specular,
    rim, and the MetaMaterial name still come from the authored material either way. Note the
    referenced MetaMaterial re-applies its own default/forced property values on load
    (UpdatePropertiesFromMetaMat), so the .mmat name matters most - but matching retail keeps the
    material self-consistent and byte diffs against retail clean."""
    s = dict(settings)   # copy so overrides don't mutate the caller's dict
    if force_dc3_defaults:
        s['pre_lit'] = False        # retail: prelit=0 on every emilia material
        s['use_environment'] = True # retail: useEnviron=1 on every emilia material
        # Keep shaderVariation consistent with the chosen MetaMaterial's shading path.
        s['shader_variation'] = _dc3_shader_variation_for_meta(
            s.get('meta_material', DC3_DEFAULT_META_MATERIAL),
            s.get('shader_variation', 'kShaderVariationNone'))
    settings = s

    # --- RndMat::Save / BaseMaterial::Save / Hmx::Object::Save headers ---
    w.u32(DC3_MAT_REVISION)          # RndMat  packRevs(0, 0x46)
    w.u32(DC3_BASEMAT_REVISION)      # BaseMaterial  packRevs(0, 8)
    w.u32(DC3_OBJ_REVISION)          # Hmx::Object  bs << 2
    w.symbol("")                     # Object type
    w.boolean(False)                 # null TypeProps (DataArray* == nullptr -> bool false)
    w.u32(0)                         # empty note (SaveRest: bs << 0, a 4-byte int)

    # --- BaseMaterial::Save body, in exact decomp order ---
    # bs << mBlend << mColor << mUseEnviron << mPrelit;
    w.i32(MAT_BLEND_VALUES[settings['blend']])
    write_color4(w, *settings['base_color'])
    w.boolean(settings['use_environment'])
    w.boolean(settings['pre_lit'])

    # bs << mZMode << mAlphaCut << mAlphaThreshold << mAlphaWrite;
    w.i32(MAT_ZMODE_VALUES[settings['z_mode']])
    w.boolean(settings['alpha_cut'])
    w.i32(settings['alpha_threshold'])
    w.boolean(settings['blend'] == 'kBlendSrcAlpha')   # alphaWrite - mirror the RB3 writer's
                                                        # rule (true only in the SrcAlpha blend
                                                        # branch); not user-exposed yet

    # bs << mTexGen << mTexWrap << mTexXfm << mDiffuseTex << mNextPass;
    w.i32(MAT_TEXGEN_NONE)
    w.i32(MAT_TEXWRAP_REPEAT)
    write_matrix(w, DC3_TEXXFM_IDENTITY)   # -0.0 off-diagonals match retail's Transform::Reset()
    w.symbol(settings.get('diffuse_tex_name', ''))
    w.symbol("")                     # nextPass (in-milo next-pass material chain; unused here)

    # bs << mIntensify << mCull << mEmissiveMultiplier;
    # (rev 8 >= 3, so cull is written as a full int, not the old bool)
    w.boolean(settings.get('intensify', False))
    w.i32(1 if settings.get('cull', False) else 0)   # kCullRegular(1) / kCullNone(0)
    w.f32(settings['emissive_multiplier'])

    # bs << mSpecularRGB << mNormalMap;
    # mSpecularRGB is a 4-float Hmx::Color: rgb + ALPHA==specular power. This is where DC3
    # diverges hardest from RB3 (no standalone specularPower float).
    sr = settings['specular_rgb']
    write_color4(w, sr[0], sr[1], sr[2], settings.get('specular_power', 0.0))
    w.symbol(settings.get('normal_tex_name', ''))

    # bs << mEmissiveMap << mSpecularMap;
    w.symbol(settings.get('emissive_tex_name', ''))
    w.symbol(settings.get('specular_tex_name', ''))

    # bs << mEnvironMap << mEnvironMapFalloff << mEnvironMapSpecMask;
    w.symbol("instruments.cube" if settings.get('environment_map') else "")
    w.boolean(False)                 # environMapFalloff (not exposed)
    w.boolean(False)                 # environMapSpecMask (not exposed)

    # bs << mPerPixelLit << mStencilMode;
    # perPixelLit must be True for any material that uses a normal or specular map (the
    # per-pixel lighting path is what samples them). Retail leaves it False on untextured
    # materials that defer entirely to their MetaMaterial (skin/color/LOD), but for a custom
    # textured surface it needs to be on. Default True, overridable via the 'per_pixel_lit'
    # setting, and force-True whenever a normal or specular map is present so a user can't
    # accidentally reintroduce the black-normal-map bug.
    needs_ppl = bool(settings.get('normal_tex_name') or settings.get('specular_tex_name'))
    w.boolean(settings.get('per_pixel_lit', True) or needs_ppl)
    w.i32(MAT_STENCIL_IGNORE)        # stencilMode (kStencilIgnore)

    # bs << mFur << mDeNormal << mAnisotropy;
    w.symbol("")                     # fur
    w.f32(float(settings.get('de_normal', 0)))
    w.f32(settings.get('anisotropy', 0.0))

    # bs << mNormDetailTiling << mNormDetailStrength << mNormDetailMap;
    w.f32(1.0)                       # normDetailTiling (ctor default 1)
    w.f32(settings.get('normal_detail_strength', 0.0))
    w.symbol("")                     # normDetailMap

    # bs << mPointLights << mFog << mFadeout << mColorAdjust;
    w.boolean(settings.get('point_lights', True))   # pointLights - retail always True on emilia
    w.boolean(False)                 # fog
    w.boolean(False)                 # fadeout
    w.boolean(False)                 # colorAdjust

    # bs << mRimRGB << mRimMap << mRimLightUnder;
    # mRimRGB is a 4-float Hmx::Color too (rgb + alpha==rim power; ctor default alpha 10).
    rr = settings.get('rim_rgb', (0.0, 0.0, 0.0))
    write_color4(w, rr[0], rr[1], rr[2], settings.get('rim_power', 4.0))
    w.symbol("")                     # rimMap
    w.boolean(False)                 # rimLightUnder

    # bs << mScreenAligned << mShaderVariation << mSpecular2RGB;
    w.boolean(False)                 # screenAligned
    w.i32(MAT_SHADER_VARIATION_VALUES[settings.get('shader_variation', 'kShaderVariationNone')])
    write_color4(w, 0.0, 0.0, 0.0, 10.0)   # specular2RGB (ctor default 0,0,0, alpha 10)

    # mPerfSettings.Save(bs): recvProjLights, ps3ForceTrilinear, recvPointCubeTex
    w.boolean(False)
    w.boolean(False)
    w.boolean(False)

    # bs << mRefractEnabled << mRefractStrength << mRefractNormalMap;
    w.boolean(settings.get('refract_enabled', False))
    w.f32(settings.get('refract_strength', 0.0))
    w.symbol(settings.get('refract_normal_map_tex_name', ''))

    # bs << mBloomMultiplier << mNeverFitToSpline;
    w.f32(1.0)                       # bloomMultiplier (ctor default 1)
    w.boolean(False)                 # neverFitToSpline

    # bs << mAllowDistortionEffects << mShockwaveMult;
    w.boolean(DC3_MAT_ALLOW_DISTORTION_EFFECTS)
    w.f32(DC3_MAT_SHOCKWAVE_MULT)

    # bs << mWorldProjectionTiling << mWorldProjectionStartBlend << mWorldProjectionEndBlend;
    w.f32(DC3_MAT_WORLD_PROJECTION_TILING)
    w.f32(DC3_MAT_WORLD_PROJECTION_START_BLEND)
    w.f32(DC3_MAT_WORLD_PROJECTION_END_BLEND)

    # bs << mDiffuseTex2;
    w.symbol("")                     # diffuseTex2 (second diffuse layer; unused here)

    # bs << mForceAlphaWrite;
    w.boolean(False)                 # forceAlphaWrite

    # --- back in RndMat::Save: bs << mMetaMaterial (the trailing .mmat template symbol) ---
    w.symbol(settings.get('meta_material', DC3_DEFAULT_META_MATERIAL))

    if standalone:
        w.block(END_MARKER)


_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tga', '.bmp', '.tif', '.tiff', '.exr', '.dds')


def _strip_image_extension(name):
    lower = name.lower()
    for ext in _IMAGE_EXTENSIONS:
        if lower.endswith(ext):
            return name[:-len(ext)]
    return name


def _round_up_4(n):
    return ((n + 3) // 4) * 4


def _round_up_pow2(n, minimum=4):
    """Rounds n UP to the next power of 2, floored at `minimum`. Used for the TBRB
    Custom Song Asset 'Source .png' export path: p9songtool (the actual custom-song
    compiler's PNG -> .tex conversion, in Mackiloha's TextureExtensions.BitmapFromImage)
    hard-rejects any image whose width or height isn't an exact power of 2 - not just a
    multiple of 4 like this exporter's own DXT path requires - throwing and failing the
    whole song build. Rounding up (rather than down or to nearest) avoids cropping detail
    off the source image; the resulting canvas gets stretched to fill exactly like the
    existing DXT resize path already does non-uniformly per axis."""
    n = max(minimum, int(n))
    p = 1
    while p < n:
        p *= 2
    return p


MAX_MILO_TEX_SIZE = 2048


def _require_loaded_image_buffer(image):
    """Raises a clear, actionable error if `image`'s pixel buffer never actually decoded,
    BEFORE anything tries to .scale()/read .pixels on it and gets Blender's raw, unhelpful
    'RuntimeError: Error: Image 'X' failed to load image buffer'.

    Why this happens: Blender can read a DDS file's HEADER (width/height/etc - so
    image.size is non-zero and it shows up fine in the file browser and material preview
    thumbnail) without being able to actually DECODE its pixel data - Blender's built-in
    DDS support only covers a subset of real-world DDS variants (notably missing most
    BC7/DX10-header/typeless-format files, which are extremely common output from texture
    tools and game-asset extractors). The failure only surfaces the moment something
    (like this exporter) actually needs the decoded pixels, which is exactly the confusing
    experience a plugin user reported: a crash pointing at Blender's own scale()/pixels
    machinery with no indication the source file itself is the problem.

    image.has_data is Blender's own signal for "the pixel buffer is actually loaded",
    independent of whether the header parsed enough to report a size - checking it here
    lets us fail with a real explanation instead of Blender's raw RuntimeError."""
    if not image.has_data:
        import os
        ext = os.path.splitext(image.filepath or image.name)[1].lower()
        dds_note = (
            " This is a known Blender limitation with DDS files specifically: Blender's "
            "built-in DDS reader can parse the header (so it shows a correct size/name "
            "here) without being able to decode many real-world DDS variants - BC7, "
            "DX10-header, and typeless-format DDS files in particular. Re-export/convert "
            "this texture to PNG or TGA (e.g. in GIMP, Photoshop, or an online DDS "
            "converter) and re-link it in Blender, then try exporting again."
        ) if ext == '.dds' else (
            " Try re-saving/re-exporting this image in a different format (PNG or TGA "
            "are the safest bets) and re-linking it in Blender."
        )
        raise RuntimeError(
            f"Image '{image.name}' has a size but its pixel buffer never actually loaded "
            f"(Blender's own image.has_data is False for it) - this is NOT a bug in the "
            f"exporter, Blender itself failed to decode this file's pixel data.{dds_note}")


def get_image_rgba8(image, max_size=512, ignore_size_limits=False):
    """Returns (width, height, rgba_bytes) as flat RGBA8 bytes, downscaled to fit
    max_size unless ignore_size_limits, and rounded up to a multiple of 4 in each
    dimension (required for DXT block compression - a deliberate improvement over
    the CLI tool, which just errors out on non-multiple-of-4 textures instead).

    ignore_size_limits only skips the soft `max_size` default - it never skips
    MAX_MILO_TEX_SIZE (2048), which is a real engine/hardware ceiling, not a
    stylistic default. Without this second, unconditional clamp, "ignore size
    limits" could produce textures that either crash the game outright (over
    the real 2048 ceiling) or take an extremely long time to DXT-encode in
    this file's pure-Python block compressors, appearing to hang the export.

    Flips rows vertically: Blender's `image.pixels` is stored bottom-up, but the
    CLI preserves the image's original top-down file orientation with no flip -
    confirmed by testing (this was previously left unflipped on the assumption the
    two conventions already matched; they don't - textures came out upside-down)."""
    orig_w, orig_h = image.size[0], image.size[1]
    if orig_w == 0 or orig_h == 0:
        raise ValueError(f"Image '{image.name}' has no pixel data - is it missing or unpacked?")
    _require_loaded_image_buffer(image)

    target_w, target_h = orig_w, orig_h
    if not ignore_size_limits and (orig_w > max_size or orig_h > max_size):
        scale = min(max_size / orig_w, max_size / orig_h)
        target_w = max(1, round(orig_w * scale))
        target_h = max(1, round(orig_h * scale))

    # Real hard ceiling - always enforced, even when ignore_size_limits skipped the
    # soft default above. See MAX_MILO_TEX_SIZE docstring for why this must not be
    # gated behind ignore_size_limits the way the reference tool's own check is.
    if target_w > MAX_MILO_TEX_SIZE or target_h > MAX_MILO_TEX_SIZE:
        scale = min(MAX_MILO_TEX_SIZE / target_w, MAX_MILO_TEX_SIZE / target_h)
        target_w = max(1, round(target_w * scale))
        target_h = max(1, round(target_h * scale))

    target_w = max(4, _round_up_4(target_w))
    target_h = max(4, _round_up_4(target_h))

    work_img = image
    is_temp = False
    if (target_w, target_h) != (orig_w, orig_h):
        work_img = image.copy()
        work_img.scale(target_w, target_h)
        is_temp = True

    try:
        pixels = work_img.pixels[:]
        raw = bytes(max(0, min(255, round(v * 255))) for v in pixels)
        row_bytes = target_w * 4
        rows = [raw[i:i + row_bytes] for i in range(0, len(raw), row_bytes)]
        rgba = b"".join(reversed(rows))
    finally:
        if is_temp:
            bpy.data.images.remove(work_img)

    return target_w, target_h, rgba


def export_texture_as_png(image, out_path, max_size=512, ignore_size_limits=False):
    """Writes `image` to `out_path` as a plain, standard .png file - used only by the TBRB
    Custom Song Asset 'Texture Format = Source .png' option.

    The community's custom-song compiler (p9songtool, part of PikminGuts92's Mackiloha)
    expects loose .png source files in the extras folder and does its own PNG -> .tex
    conversion at song-package time, so this writes real uncompressed PNG bytes via
    Blender's own image writer rather than this exporter's compiled RndTex/DXT format.

    Applies the SAME soft/hard size-limit policy as get_image_rgba8 (soft `max_size`
    default unless ignore_size_limits, hard MAX_MILO_TEX_SIZE ceiling always enforced),
    but then rounds each axis UP TO THE NEXT POWER OF 2 rather than merely a multiple of
    4. This is not optional: byte-verified against Mackiloha's actual PNG-import code
    (TextureExtensions.BitmapFromImage) - it hard-rejects any image whose width or height
    isn't an exact power of 2 with "Invalid image resolution... must be a power of 2 and
    at least 4px", throwing and failing the ENTIRE song build, not just that one texture.
    A multiple-of-4 image (e.g. 500x500, which the DXT path would happily accept) passes
    this exporter's own checks but would still blow up the compiler at package time.

    Returns (width, height) of the written PNG."""
    orig_w, orig_h = image.size[0], image.size[1]
    if orig_w == 0 or orig_h == 0:
        raise ValueError(f"Image '{image.name}' has no pixel data - is it missing or unpacked?")
    _require_loaded_image_buffer(image)

    target_w, target_h = orig_w, orig_h
    if not ignore_size_limits and (orig_w > max_size or orig_h > max_size):
        scale = min(max_size / orig_w, max_size / orig_h)
        target_w = max(1, round(orig_w * scale))
        target_h = max(1, round(orig_h * scale))

    # Same real hard ceiling as get_image_rgba8 - always enforced regardless of
    # ignore_size_limits. See MAX_MILO_TEX_SIZE for why.
    if target_w > MAX_MILO_TEX_SIZE or target_h > MAX_MILO_TEX_SIZE:
        scale = min(MAX_MILO_TEX_SIZE / target_w, MAX_MILO_TEX_SIZE / target_h)
        target_w = max(1, round(target_w * scale))
        target_h = max(1, round(target_h * scale))

    # Power-of-2 rounding - see docstring. MAX_MILO_TEX_SIZE (2048) is itself a power of
    # 2, so rounding up can only push an already-near-ceiling value back over it; reclamp
    # by capping at MAX_MILO_TEX_SIZE rather than letting it round up past it.
    target_w = min(_round_up_pow2(target_w), MAX_MILO_TEX_SIZE)
    target_h = min(_round_up_pow2(target_h), MAX_MILO_TEX_SIZE)

    work_img = image
    is_temp = False
    if (target_w, target_h) != (orig_w, orig_h):
        work_img = image.copy()
        work_img.scale(target_w, target_h)
        is_temp = True

    try:
        prev_format = work_img.file_format
        try:
            work_img.file_format = 'PNG'
            work_img.filepath_raw = out_path
            work_img.save()
        finally:
            if not is_temp:
                # Only restore on the original (non-temp) image - a temp copy gets
                # deleted right after anyway.
                work_img.file_format = prev_format
    finally:
        if is_temp:
            bpy.data.images.remove(work_img)

    return target_w, target_h


def gather_materials_and_textures(materials_by_name, max_tex_size=512, ignore_tex_size_limits=False,
                                   platform='xbox360', generate_mips=False, mip_floor=4):
    """Converts {material_name: bpy.types.Material} into:
    - materials: list of (mat_entry_name, settings_dict) for build_milo_bytes/write_rnd_mat
    - textures: list of (tex_entry_name, width, height, encoding, bpp, block_data, num_mips)
      for build_milo_bytes/write_rnd_tex

    Texture roles use fixed encodings, confirmed directly against the real game files
    (not guessed): diffuse = DXT1/BC1, specular = DXT5/BC3, normal = ATI2/BC5 (reading
    the image's R/G channels as X/Y - Z is reconstructed by the shader), emissive = DXT1/BC1.

    PS3 exception: normal maps use DXT1/BC1 instead of ATI2/BC5 - confirmed from the
    glTFMilo commit "Encode normal maps as BC1 on PS3" (the PS3 GPU/shader path doesn't
    use the two-channel BC5 normal format the Xbox path does).

    generate_mips: when True, every texture is written with a full mip chain (down to 4x4)
    instead of a single base level. Dance Central 1 needs this - a vanilla DC1 ATI2/BC5
    normal map ships a mip chain, and DC1 renders a mip-less ATI2 surface black. Off for
    RB3, which renders single-level ATI2 normal maps correctly."""
    materials = []
    texture_by_key = {}   # (image, suffix) -> tex_entry_name, dedups shared textures
    textures = []
    texture_images = {}   # entry_name -> source bpy.types.Image, for the PNG export path
    is_ps3 = (platform == 'ps3')

    def register_texture(image, suffix, encoder, role):
        key = (image, suffix)
        if key in texture_by_key:
            _log(f"  Texture '{image.name}' already exported as '{texture_by_key[key]}' - reusing.")
            return texture_by_key[key]
        base_name = sanitize_milo_name(_strip_image_extension(image.name))
        entry_name = f"{base_name}{suffix}.tex"
        # dedupe by name too, in case two different Image datablocks share a name
        existing_names = {t[0] for t in textures}
        if entry_name in existing_names:
            n = 2
            while f"{base_name}{suffix}_{n}.tex" in existing_names:
                n += 1
            entry_name = f"{base_name}{suffix}_{n}.tex"

        _log(f"Exporting texture '{image.name}' ({role}) -> '{entry_name}'...")

        width, height, rgba = get_image_rgba8(image, max_tex_size, ignore_tex_size_limits)

        block_data, encoding, bpp, num_mips = build_texture_mip_chain(
            rgba, width, height, encoder, generate_mips, mip_floor=mip_floor)

        textures.append((entry_name, width, height, encoding, bpp, block_data, num_mips))
        texture_by_key[key] = entry_name
        texture_images[entry_name] = image
        enc_name = _TEX_ENCODING_NAMES.get(encoding, str(encoding))
        mip_note = f", {num_mips + 1} mip level(s)" if num_mips else ""
        _log(f"  Exported '{entry_name}': {width}x{height}, {enc_name} ({bpp}bpp), "
             f"{len(block_data)} bytes compressed{mip_note}.")
        return entry_name

    # On PS3 normal maps are BC1, not ATI2/BC5 - the encoder differs, and BC1 reads the
    # full RGB (so we feed the normal map's RGB, same as diffuse) rather than the R/G-only
    # ATI2 path.
    normal_encoder = 'bc1' if is_ps3 else 'ati2'

    for name, blender_mat in materials_by_name.items():
        _log(f"Exporting material '{name}'...")
        s = blender_mat.gltfmilo_settings

        diffuse_tex_name = ""
        if s.diffuse_tex is not None:
            diffuse_tex_name = register_texture(s.diffuse_tex, "", 'bc1', 'diffuse')

        normal_tex_name = ""
        if s.normal_tex is not None:
            normal_tex_name = register_texture(s.normal_tex, "_norm", normal_encoder, 'normal')

        specular_tex_name = ""
        if s.specular_tex is not None:
            specular_tex_name = register_texture(s.specular_tex, "_spec", 'bc3', 'specular')

        emissive_tex_name = ""
        if s.emissive_tex is not None:
            emissive_tex_name = register_texture(s.emissive_tex, "_emissive", 'bc1', 'emissive')

        refract_normal_map_tex_name = ""
        if s.refract_normal_map is not None:
            refract_normal_map_tex_name = register_texture(s.refract_normal_map, "_refract_norm", normal_encoder, 'refract normal map')

        settings = {
            'blend': s.blend,
            'base_color': tuple(s.base_color),
            'pre_lit': s.pre_lit,
            'intensify': s.intensify,
            'use_environment': s.use_environment,
            'z_mode': s.z_mode,
            'alpha_cut': s.alpha_cut,
            'alpha_threshold': s.alpha_threshold,
            'specular_rgb': tuple(s.specular_rgb),
            'specular_power': s.specular_power,
            'emissive_multiplier': s.emissive_multiplier,
            'environment_map': s.environment_map,
            'cull': s.cull,
            'de_normal': s.de_normal,
            'anisotropy': s.anisotropy,
            'normal_detail_strength': s.normal_detail_strength,
            'rim_rgb': tuple(s.rim_rgb),
            'rim_power': s.rim_power,
            'shader_variation': s.shader_variation,
            'refract_enabled': s.refract_enabled,
            'refract_strength': s.refract_strength,
            'diffuse_tex_name': diffuse_tex_name,
            'normal_tex_name': normal_tex_name,
            'specular_tex_name': specular_tex_name,
            'emissive_tex_name': emissive_tex_name,
            'refract_normal_map_tex_name': refract_normal_map_tex_name,
            'meta_material': (s.meta_material or DC3_DEFAULT_META_MATERIAL),  # DC3 only; see write_dc3_rnd_mat
        }
        mat_entry_name = f"{sanitize_milo_name(name)}.mat"
        materials.append((mat_entry_name, settings))
        tex_notes = ", ".join(
            f"{label}='{val}'" for label, val in
            (("diffuse", diffuse_tex_name), ("normal", normal_tex_name),
             ("specular", specular_tex_name), ("emissive", emissive_tex_name))
            if val
        ) or "no textures"
        _log(f"  Exported '{mat_entry_name}': blend={settings['blend']}, {tex_notes}.")

    return materials, textures, texture_images


