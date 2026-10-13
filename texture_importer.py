"""
Green Day: Rock Band texture and material importer.

Decodes the textures stored in a GDRB milo and builds a simple Principled BSDF node tree
for each material, so imported characters can be previewed with their real textures.

===========================================================================================
TEXTURE FORMAT - verified on all 28 textures in three retail GDRB Xbox 360 milos
===========================================================================================
RndTex revision 11 (GDRB) or 10 (TBRB) wrapping an RndBitmap revision 1. Width, height and
bpp follow a 14-byte object header (13 bytes at revision 10), then the source path, mip bias, type and one flag, then the bitmap: revision,
bpp, encoding, mip count, width, height, bytes per line, a Wii field and 17 bytes of padding,
then the pixel data for the base level followed by each smaller mip. Parsed this way, every
texture's payload comes out exactly the size its dimensions, format and mip count predict.

Encodings: 8 = DXT1 (colour maps), 24 = DXT5 (most specular maps), 32 = ATI2 (normal maps).
Xbox 360 stores the blocks with every pair of bytes swapped, but NOT tiled - matching how
texture_exporter writes them, which is proven in-game. Only the base level is decoded.

===========================================================================================
NORMAL MAPS - why the Xbox ones look green, and what's done about it
===========================================================================================
ATI2 holds just two channels. A normal is unit length, so Z can be rebuilt from X and Y;
decoded naively there is no blue at all, which is why these maps look green and yellow in
viewers. Three corrections turn them into a standard map for Blender's Normal Map node:

  1. The channels are SWAPPED: the first ATI2 block is Y, the second is X. That matches a
     known quirk of D3D9's ATI2 format relative to modern BC5, and it was established from
     the data rather than assumed - see below.
  2. Y points DOWN the image (DirectX convention). Blender's node expects OpenGL, Y up, so
     Y is inverted.
  3. Z is rebuilt as sqrt(1 - X^2 - Y^2).

Both the order and the direction were settled with a consistency test. A normal map
describes the slope of a surface, so its X and Y are the two halves of a gradient, and a
true gradient field has no curl. Measuring the curl for each candidate interpretation, on
five different retail normal maps, "swapped, Y down" came out about twice as consistent as
any other in every case (1.1x on the pants). The test fixes Y's direction relative to X, on
the near-universal assumption that red means "to the right"; if a map ever lights inside
out, that assumption is what failed, and the green flip can be turned off.

===========================================================================================
MATERIALS - what's read
===========================================================================================
RndMat revision 56 (GDRB) or 55 (TBRB). Everything is read in file order, the same order
texture_exporter's revision-55 writer uses:

  blend, colour (RGBA), prelit, useEnviron, zMode, alphaCut, alphaThreshold, alphaWrite,
  texGen, texWrap, texXfm, DIFFUSE, nextPass, intensify, cull, emissiveMultiplier,
  specularRGB + specularPower, NORMAL, EMISSIVE, SPECULAR, environMap,
  [56: an int (3 on every textured material, 0 on an unused one), stencil]
  [55: perPixelLit (bool), stencil], fur, deNormal, anisotropy,
  normalDetailTiling, normalDetailStrength, NORMAL DETAIL MAP,
  pointLights, projLights, fog, fadeout, colorAdjust,
  rimRGB + rimPower, rimMap, rimAlwaysShow, screenAligned, shaderVariation,
  specular2RGB + specular2Power, then a 25-byte tail (a float, a colour, the alpha-mask
  symbol, ps3ForceTrilinear).

The whole object must be consumed exactly, or only the fields up to the specular slot (the
part every revision shares) are trusted and the rest fall back to neutral values. All six
materials in 21st.milo_xbox land exactly on their object boundary. What they say:

  * Clothing: specular power 8-15 with a coloured specular map and a grey rim light
    (0.35, power 4; the shirt's is warm tan). The pants and shirt show their rim
    regardless of lighting (rimAlwaysShow); the shoes don't.
  * Skin: shaderVariation 1, no specular at all (black specularRGB, no map), the same rim.
  * Hair: shaderVariation 2, a tight primary highlight (power 200), a SECOND broad, warm
    highlight (specular2 = 0.30/0.22/0.15, power 10), a warm rim at power 25, and a detail
    normal map (hair_detail_normals.tex, fine vertical strands) tiled 5x at strength 0.5.
  * All opaque (blend kSrc, no alpha cut) with backfaces culled. The hair is opaque too:
    its DXT1 diffuse has no transparent texels.

===========================================================================================
MATERIALS - how they're rebuilt in Blender
===========================================================================================
The game lights these with a Blinn-Phong model: a diffuse term, plus specular added on top
(specularRGB x specular map x (N.H)^power), plus a rim term at grazing angles. Blender's
Principled BSDF can't express that - its specular is a Fresnel dielectric with a fixed
strength - so every material uses one shared node group, "Milo Character Shader", that
rebuilds the game's terms from separate parts:

  * Diffuse: diffuse texture x material colour (x2 with intensify), through a Principled
    BSDF with its own specular switched off.
  * Specular: a Glossy BSDF ADDED on top, as the game adds it. Power n becomes roughness
    (2/(n+2))^0.25 (the standard Blinn-Phong to Beckmann match), and its colour is
    specularRGB x specular map x 8/(n+7): the total light the game's lobe,
    spec x (N.H)^n x N.L, reflects over the hemisphere. For tight lobes that's also the
    same peak brightness; for broad ones (skin, power ~5) it keeps the lobe from reading
    as a pale layer under environment lighting, which the game's specular never gets.
  * Gloss: the specular map's ALPHA scales the power, as the material editor describes
    the map ("specular colour and power"). The pants' cloth is ~0.9, the vest's leather
    is lower, giving it a broader sheen. Turned off for hair (see below).
  * Hair: a second Glossy lobe from specular2, and both lobes stretched across the strands
    (anisotropic, along the UV's U axis - the hair textures' strands all run along V).
    The hair specular map's alpha only reaches 31/255, which as gloss would flatten the
    power-200 highlight to ~8; it looks like the strand-shift/sparkle mask of a two-lobe
    hair shader rather than gloss, so it isn't used.
  * Rim: rimRGB x (1 - N.V)^rimPower. With rimAlwaysShow it's added as emission (shown
    regardless of light, as the flag says); without it, it becomes the Principled sheen,
    so it only appears where the surface is lit.
  * Normals: the normal map, with the detail map (tiled by normalDetailTiling) blended in.
    normalDetailStrength s weights the detail by s/(1+s), so it saturates instead of
    multiplying the detail's slope: TBRB skin's 5.0 on a subtle pore map would otherwise
    turn the face to sandpaper.
  * Alpha, blend and cull follow the material: alpha-blended for kSrcAlpha/premultiplied,
    cut at alphaThreshold for alphaCut, additive for the add modes, backface culling from
    cull.

Each value is an input on the group node, so it can be tuned per material afterwards, and
the material's glTFMilo settings panel is filled from the same data.

===========================================================================================
TBRB - same textures, same shader, textures in a shared milo
===========================================================================================
TBRB's materials are revision 55 and carry the same fields; all five in budokan01 (John's
Budokan outfit) read to their last byte. Its textures use the same encodings (DXT1 colour
and specular, ATI2 normals) and the same shared hair_detail_normals.tex. Differences seen:
specularRGB is 1.0 and the specular maps alone set the colour (they're DXT1, so there's no
gloss alpha), the hair's second highlight is a strong gold (1.0/0.69/0.25), and its rim is
shown regardless of light.

The outfit milo only holds the hair's textures. The clothing's (pant/shoe/torso_bud02) live
in the outfit's shared milo, which the file names as '../../shared/outfit/
budokan01_share.milo' - shared because every Beatle in that outfit uses them. The importer
follows those references on disk automatically, using the game's folder layout (see
find_shared_milo), and follows a shared milo's own references in turn. Textures it still
can't find are listed in a warning; those slots are left empty.

===========================================================================================
KNOWN GAPS
===========================================================================================
1. Lighting is approximated, not reproduced: the game's light rig, ambient, and the skin
   variation's own shading (shaderVariation 1) aren't modelled - skin uses the standard
   path. The gloss and hair-alpha readings above are inferred from the data, not confirmed
   against the game's shaders.
2. The eyes' reflection pass isn't used.
3. TBRB's animated wrinkles aren't reproduced. Its head normal map is a render target that a
   TexBlender fills each frame, blending wrinkle maps into the base map in regions defined by
   the Blend_* helper meshes, as bone pairs move apart. The static base map is used instead.
4. kBlendMultiply/kBlendSubtract/kBlendDest materials are drawn opaque.
"""

import hashlib
import os
import re
import struct

import bpy
import numpy as np

from .io import read_milo_container_body
from .utilities import _log
from .mesh_importer import _read_dir_entries, _entry_spans


# RndTex::Type bit for render targets (kRendered = 2; kRenderedNoZ = 0x22 also has it).
TEX_RENDERED = 2

ENC_DXT1 = 8
ENC_DXT5 = 24
ENC_ATI2 = 32
_ENC_NAMES = {ENC_DXT1: "DXT1", ENC_DXT5: "DXT5", ENC_ATI2: "ATI2"}
_BLOCK_BYTES = {ENC_DXT1: 8, ENC_DXT5: 16, ENC_ATI2: 16}

# Offset of the width, by texture revision. GDRB's revision 11 has one more header byte
# than TBRB's revision 10; everything after the width is laid out the same.
TEX_HEADER_END = {10: 17, 11: 18}
MAT_OBJECT_HEADER_END = 17     # revision + 13-byte object header, before blend
# Material revisions sharing the slot layout read here: 55 is TBRB, 56 is GDRB.
MAT_REVISIONS = (55, 56)


class TextureImportError(Exception):
    pass


# ---------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------

class _R:
    def __init__(self, data, p=0):
        self.d = data
        self.p = p

    def u8(self):
        v = self.d[self.p]; self.p += 1; return v

    def u16(self):
        v = struct.unpack_from('>H', self.d, self.p)[0]; self.p += 2; return v

    def u32(self):
        v = struct.unpack_from('>I', self.d, self.p)[0]; self.p += 4; return v

    def f32(self):
        v = struct.unpack_from('>f', self.d, self.p)[0]; self.p += 4; return v

    def sym(self):
        n = self.u32()
        if n > 1024 or self.p + n > len(self.d):
            raise TextureImportError(f"implausible string length {n}")
        s = self.d[self.p:self.p + n].decode('latin-1'); self.p += n
        return s


def parse_tex(data, name):
    """RndTex revision 11 + RndBitmap revision 1. Returns a dict with the bitmap header
    and the base level's pixel data."""
    r = _R(data)
    rev = r.u32()
    if rev not in TEX_HEADER_END:
        raise TextureImportError(f"'{name}' is texture revision {rev}; only 10 (TBRB) and "
                                 f"11 (GDRB) are supported")
    r.p = TEX_HEADER_END[rev]
    r.u32(); r.u32(); r.u32()            # width, height, bpp (repeated in the bitmap)
    r.sym()                              # source path
    r.f32()                              # mip bias
    tex_type = r.u32()
    r.u8()                               # flag
    r.u8()                               # bitmap revision
    bpp = r.u8()
    enc = r.u32()
    mips = r.u8()
    w = r.u16()
    h = r.u16()
    r.u16(); r.u16()                     # bytes per line, Wii field
    r.p += 17
    if (tex_type & TEX_RENDERED) and w == 0 and h == 0:
        # A render target: the game draws into it at runtime, so there are no pixels to
        # decode. TBRB's head uses one as its normal map - the TexBlender's output.
        return dict(width=0, height=0, bpp=bpp, encoding=enc, mips=0, data=b'',
                    render_target=True)
    if enc not in _BLOCK_BYTES:
        raise TextureImportError(f"'{name}' uses encoding {enc}, which isn't supported")
    payload = data[r.p:]
    total = 0
    for i in range(mips + 1):
        total += max(1, (max(w >> i, 1) + 3) // 4) * max(1, (max(h >> i, 1) + 3) // 4) \
            * _BLOCK_BYTES[enc]
    if total != len(payload):
        raise TextureImportError(
            f"'{name}' pixel data is {len(payload)} bytes but {w}x{h} "
            f"{_ENC_NAMES[enc]} with {mips} mip(s) should be {total}")
    base = max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * _BLOCK_BYTES[enc]
    return dict(width=w, height=h, bpp=bpp, encoding=enc, mips=mips, data=payload[:base])


BLEND_DEST, BLEND_SRC, BLEND_ADD, BLEND_SRC_ALPHA, BLEND_SRC_ALPHA_ADD = 0, 1, 2, 3, 4
BLEND_SUBTRACT, BLEND_MULTIPLY, BLEND_PREMULT = 5, 6, 7
SHADER_VARIATION_SKIN, SHADER_VARIATION_HAIR = 1, 2
MAT_TAIL_BYTES = 25          # float, colour, alpha-mask symbol, ps3ForceTrilinear


def _mat_tail(r, rev, end):
    """Everything after the specular slot. Returns a dict, or None when the bytes don't
    land exactly on the object's end with plausible values."""
    t = {}
    t['environ_map'] = r.sym()
    if rev >= 56:
        t['lighting_mode'] = r.u32()     # 3 on every textured GDRB material; meaning unknown
    else:
        t['per_pixel_lit'] = bool(r.u8())
    t['stencil'] = r.u32()
    t['fur'] = r.sym()
    t['de_normal'] = r.f32()
    t['anisotropy'] = r.f32()
    t['detail_tiling'] = r.f32()
    t['detail_strength'] = r.f32()
    t['detail_normal'] = r.sym()
    t['point_lights'] = bool(r.u8())
    t['proj_lights'] = bool(r.u8())
    t['fog'] = bool(r.u8())
    t['fadeout'] = bool(r.u8())
    t['color_adjust'] = bool(r.u8())
    t['rim_rgb'] = (r.f32(), r.f32(), r.f32())
    t['rim_power'] = r.f32()
    t['rim_map'] = r.sym()
    t['rim_always_show'] = bool(r.u8())
    t['screen_aligned'] = bool(r.u8())
    t['shader_variation'] = r.u32()
    t['specular2_rgb'] = (r.f32(), r.f32(), r.f32())
    t['specular2_power'] = r.f32()
    if end - r.p != MAT_TAIL_BYTES:
        return None
    for key in ('environ_map', 'fur', 'detail_normal', 'rim_map'):
        if t[key] and not t[key].lower().endswith(('.tex', '.cube', '.fur')):
            return None
    if t['shader_variation'] > 2:
        return None
    floats = [t['detail_tiling'], t['detail_strength'], t['rim_power'],
              t['specular2_power'], *t['rim_rgb'], *t['specular2_rgb']]
    if not all(-1e4 < f < 1e4 for f in floats):
        return None
    return t


_MAT_TAIL_DEFAULTS = dict(
    environ_map='', stencil=0, fur='', de_normal=0.0, anisotropy=0.0, detail_tiling=1.0,
    detail_strength=0.0, detail_normal='', point_lights=False, proj_lights=False,
    fog=False, fadeout=False, color_adjust=False, rim_rgb=(0.0, 0.0, 0.0), rim_power=4.0,
    rim_map='', rim_always_show=False, screen_aligned=False, shader_variation=0,
    specular2_rgb=(0.0, 0.0, 0.0), specular2_power=0.0)


def parse_mat(data, name):
    """RndMat revision 55/56. Returns every field the material shader uses (see the module
    docstring for the layout). 'tail_ok' is False when the part after the specular slot
    couldn't be read cleanly; those fields then hold neutral defaults."""
    r = _R(data)
    rev = r.u32()
    if rev not in MAT_REVISIONS:
        raise TextureImportError(f"'{name}' is material revision {rev}; only 55 (TBRB) and "
                                 f"56 (GDRB) are supported")
    r.p = MAT_OBJECT_HEADER_END
    m = dict(revision=rev)
    m['blend'] = r.u32()
    m['color'] = (r.f32(), r.f32(), r.f32(), r.f32())
    m['prelit'] = bool(r.u8())
    m['use_environ'] = bool(r.u8())
    m['z_mode'] = r.u32()
    m['alpha_cut'] = bool(r.u8())
    m['alpha_threshold'] = r.u32()
    m['alpha_write'] = bool(r.u8())
    m['tex_gen'] = r.u32()
    m['tex_wrap'] = r.u32()
    r.p += 48                                           # texture transform
    m['diffuse'] = r.sym()
    m['next_pass'] = r.sym()
    m['intensify'] = bool(r.u8())
    m['cull'] = bool(r.u8())
    m['emissive_multiplier'] = r.f32()
    m['specular_rgb'] = (r.f32(), r.f32(), r.f32())
    m['specular_power'] = r.f32()
    m['normal'] = r.sym()
    m['emissive'] = r.sym()
    m['specular'] = r.sym()
    tail = None
    try:
        tail = _mat_tail(_R(data, r.p), rev, len(data))
    except (TextureImportError, struct.error, IndexError):
        tail = None
    m.update(tail if tail is not None else _MAT_TAIL_DEFAULTS)
    m['tail_ok'] = tail is not None
    return m


def _symbols(data):
    """Every plausible length-prefixed string in an object, in order."""
    out, p = [], 0
    while p + 4 <= len(data):
        n = struct.unpack_from('>I', data, p)[0]
        if 3 <= n <= 128 and p + 4 + n <= len(data):
            raw = data[p + 4:p + 4 + n]
            if all(32 <= b < 127 for b in raw):
                out.append(raw.decode('latin-1'))
                p += 4 + n
                continue
        p += 1
    return out


def parse_tex_blender(data):
    """A TexBlender's output texture and its base map, as (output, base).

    TBRB's head shows the layout: the output render target first, then the base normal map,
    then the wrinkle map(s), then its TexBlendControllers and the target mesh. With no
    wrinkles active the blended result is just the base map, so that's what a material
    pointing at the render target is given for a static preview."""
    texs = [s for s in _symbols(data) if s.lower().endswith('.tex')]
    if len(texs) < 2:
        return None
    return texs[0], texs[1]


def read_milo_materials(filepath):
    """Returns ({tex_name: tex dict}, {mat_name: slots}, [(name, reason)] failures).
    Texture dicts for a TexBlender's output carry 'blend_base', the name of its base map."""
    with open(filepath, 'rb') as f:
        body = read_milo_container_body(f.read())
    _rev, _dt, _dn, entries = _read_dir_entries(body)
    spans = _entry_spans(body, len(entries))
    texs, mats, failed, blenders = {}, {}, [], []
    for (etype, ename), (s, e) in zip(entries, spans):
        try:
            if etype == 'Tex':
                texs[ename] = parse_tex(body[s:e], ename)
            elif etype == 'Mat':
                mats[ename] = parse_mat(body[s:e], ename)
            elif etype == 'TexBlender':
                pair = parse_tex_blender(body[s:e])
                if pair:
                    blenders.append(pair)
        except (TextureImportError, struct.error, IndexError) as ex:
            failed.append((ename, str(ex)))
    for output, base in blenders:
        if output in texs and texs[output].get('render_target'):
            texs[output]['blend_base'] = base
    return texs, mats, failed


_MILO_REF_RE = re.compile(rb'[A-Za-z0-9_./\\-]+\.milo(?![A-Za-z0-9_])')


def referenced_milos(filepath):
    """The other milos a milo names (as source-tree paths like
    '../../shared/outfit/budokan01_share.milo'), in order of appearance."""
    with open(filepath, 'rb') as f:
        body = read_milo_container_body(f.read())
    out = []
    for m in _MILO_REF_RE.findall(body):
        ref = m.decode('latin-1').replace('\\', '/')
        if ref not in out:
            out.append(ref)
    return out


def _platform_suffix(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    return ext if ext.startswith('.milo') else '.milo_xbox'


def _source_dir(milo_path):
    """The folder a milo's references are relative to. Built files sit in a 'gen' folder
    beside their source (char/john/outfit/gen/budokan01.milo_xbox), but the references were
    written from the source folder (char/john/outfit), so 'gen' is stepped out of."""
    here = os.path.dirname(os.path.abspath(milo_path))
    return os.path.dirname(here) if os.path.basename(here).lower() == 'gen' else here


def _char_root(milo_path):
    """The nearest folder above a milo named 'char' (the game's character root), else the
    folder three levels up. Used only for the fallback search."""
    d = os.path.dirname(os.path.abspath(milo_path))
    for _ in range(6):
        if os.path.basename(d).lower() == 'char':
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    d = os.path.dirname(os.path.abspath(milo_path))
    for _ in range(3):
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return d


def find_shared_milo(filepath, ref, max_dirs=4000):
    """Locates a referenced milo on disk, or None. The game's layout is tried first:

        char/john/outfit/gen/budokan01.milo_xbox
          refs '../../shared/outfit/budokan01_share.milo'   (from char/john/outfit)
          ->    char/shared/outfit/gen/budokan01_share.milo_xbox

    then the same without the 'gen' step (for flattened extractions), the file name next to
    this one, and finally a bounded search under the char folder."""
    suffix = _platform_suffix(filepath)
    stem = os.path.splitext(ref.replace('\\', '/'))[0]
    name = os.path.basename(stem)
    self_path = os.path.normcase(os.path.abspath(filepath))
    cands = []
    for base in (_source_dir(filepath), os.path.dirname(os.path.abspath(filepath))):
        target = os.path.normpath(os.path.join(base, stem))
        tdir = os.path.dirname(target)
        cands += [os.path.join(tdir, 'gen', name + suffix),
                  os.path.join(tdir, name + suffix),
                  os.path.join(tdir, 'gen', name + '.milo'),
                  os.path.join(tdir, name + '.milo')]
    cands.append(os.path.join(os.path.dirname(os.path.abspath(filepath)), name + suffix))
    for cand in cands:
        if os.path.isfile(cand) and os.path.normcase(os.path.abspath(cand)) != self_path:
            return cand
    wanted = {name.lower() + suffix, name.lower() + '.milo'}
    seen = 0
    for dirpath, dirnames, filenames in os.walk(_char_root(filepath)):
        seen += 1
        if seen > max_dirs:
            break
        dirnames.sort()
        for fn in filenames:
            if fn.lower() in wanted:
                return os.path.join(dirpath, fn)
    return None


def _ref_priority(ref):
    """Shared outfit milos first: they're where an outfit's textures live, while a head
    milo reference ('../john_headhands_short.milo') rarely holds what's missing."""
    r = ref.lower()
    return (0 if ('shared' in r or '_share' in r) else 1)


def _shared_textures(filepath, missing, notes, max_files=8, max_depth=2):
    """Textures the materials name but this milo doesn't hold, gathered from the milos it
    references. TBRB outfits keep their clothing textures in a shared outfit milo every
    Beatle in that outfit points at (budokan01 -> ../../shared/outfit/budokan01_share.milo);
    an outfit can name more than one, and a shared milo can point on to another, so
    references are followed breadth-first (up to max_depth hops, max_files files) until
    nothing is missing."""
    found = {}
    visited = {os.path.normcase(os.path.abspath(filepath))}
    queue = [(filepath, 0)]
    unfound = []
    loaded = 0
    while queue and (missing - set(found)) and loaded < max_files:
        src, depth = queue.pop(0)
        if depth >= max_depth:
            continue
        try:
            refs = sorted(referenced_milos(src), key=_ref_priority)
        except Exception:
            continue
        for ref in refs:
            if not (missing - set(found)) or loaded >= max_files:
                break
            p = find_shared_milo(src, ref)
            if p is None:
                if ref not in unfound:
                    unfound.append(ref)
                continue
            key = os.path.normcase(os.path.abspath(p))
            if key in visited:
                continue
            visited.add(key)
            loaded += 1
            try:
                texs, _mats, _failed = read_milo_materials(p)
            except Exception as e:
                notes.append(f"couldn't read '{os.path.basename(p)}': {e}")
                continue
            got = [n for n in missing if n in texs and n not in found]
            for n in got:
                found[n] = texs[n]
            if got:
                notes.append(f"{len(got)} texture(s) from '{os.path.relpath(p, _char_root(filepath))}'")
            queue.append((p, depth + 1))
    if missing - set(found) and unfound:
        notes.append("referenced milo(s) not found on disk: "
                     + ", ".join(f"'{r}'" for r in unfound))
    return found


def _byteswap16(data):
    a = np.frombuffer(data, dtype=np.uint8)
    if a.size % 2:
        a = a[:-1]
    return a.reshape(-1, 2)[:, ::-1].reshape(-1).tobytes()


def _blocks(buf, w, h, size):
    bw, bh = max(1, (w + 3) // 4), max(1, (h + 3) // 4)
    return np.frombuffer(buf[:bw * bh * size], dtype=np.uint8).reshape(bh, bw, size), bw, bh


def _rgb565(c):
    r = ((c >> 11) & 31).astype(np.float32) * (255.0 / 31.0)
    g = ((c >> 5) & 63).astype(np.float32) * (255.0 / 63.0)
    b = (c & 31).astype(np.float32) * (255.0 / 31.0)
    return np.stack([r, g, b], -1)


def _color_block(b, force_four):
    c0 = b[..., 0].astype(np.uint16) | (b[..., 1].astype(np.uint16) << 8)
    c1 = b[..., 2].astype(np.uint16) | (b[..., 3].astype(np.uint16) << 8)
    p0, p1 = _rgb565(c0), _rgb565(c1)
    four = (c0 > c1) | force_four
    p2 = np.where(four[..., None], (2 * p0 + p1) / 3, (p0 + p1) / 2)
    p3 = np.where(four[..., None], (p0 + 2 * p1) / 3, 0.0)
    pal = np.stack([p0, p1, p2, p3], -2)
    idx = (b[..., 4:8].astype(np.uint32)[..., None] >> (np.arange(4) * 2)) & 3
    alpha = np.where((~four)[..., None, None] & (idx == 3), 0.0, 255.0)
    rgb = np.take_along_axis(pal[:, :, None, None, :, :],
                             idx[..., None, None].repeat(3, -1), axis=4)[..., 0, :]
    return rgb, alpha


def _alpha_block(b):
    a0 = b[..., 0].astype(np.float32)
    a1 = b[..., 1].astype(np.float32)
    eight = a0 > a1
    i8 = [a0 * (7 - i) / 7 + a1 * i / 7 for i in range(1, 7)]
    i6 = [a0 * (5 - i) / 5 + a1 * i / 5 for i in range(1, 5)] \
        + [np.zeros_like(a0), np.full_like(a0, 255.0)]
    extra = np.stack([np.where(eight, i8[i], i6[i]) for i in range(6)], -1)
    pal = np.concatenate([a0[..., None], a1[..., None], extra], -1)
    bits = np.zeros(b.shape[:2], dtype=np.uint64)
    for i in range(6):
        bits |= b[..., 2 + i].astype(np.uint64) << np.uint64(8 * i)
    idx = ((bits[..., None] >> (np.arange(16, dtype=np.uint64) * np.uint64(3)))
           & np.uint64(7)).astype(np.int64)
    return np.take_along_axis(pal, idx, axis=-1).reshape(b.shape[0], b.shape[1], 4, 4)


def decode_rgba(tex):
    """Decodes a texture's base level to a float32 (height, width, 4) array in 0-255,
    top row first. ATI2 comes back as its two raw channels (see decode_normal)."""
    w, h, enc = tex['width'], tex['height'], tex['encoding']
    data = _byteswap16(tex['data'])
    if enc == ENC_DXT1:
        b, bw, bh = _blocks(data, w, h, 8)
        rgb, a = _color_block(b, False)
        img = np.concatenate([rgb, a[..., None]], -1)
    elif enc == ENC_DXT5:
        b, bw, bh = _blocks(data, w, h, 16)
        rgb, _ = _color_block(b[..., 8:], True)
        a = _alpha_block(b[..., :8])
        img = np.concatenate([rgb, a[..., None]], -1)
    elif enc == ENC_ATI2:
        b, bw, bh = _blocks(data, w, h, 16)
        c0 = _alpha_block(b[..., :8])
        c1 = _alpha_block(b[..., 8:])
        img = np.stack([c0, c1, np.zeros_like(c0), np.full_like(c0, 255.0)], -1)
    else:
        raise TextureImportError(f"unsupported encoding {enc}")
    img = img.astype(np.float32).transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 4)
    return img[:h, :w]


def decode_normal(tex, flip_green=True):
    """An ATI2 normal map as a standard OpenGL-convention RGB normal map (see the module
    docstring for why each step is needed)."""
    raw = decode_rgba(tex)
    y = raw[..., 0] / 127.5 - 1.0         # first block is Y
    x = raw[..., 1] / 127.5 - 1.0         # second block is X
    if flip_green:
        y = -y                            # DirectX (down) -> OpenGL (up)
    z = np.sqrt(np.clip(1.0 - x * x - y * y, 0.0, 1.0))
    out = np.empty(raw.shape, dtype=np.float32)
    out[..., 0] = (x + 1.0) * 127.5
    out[..., 1] = (y + 1.0) * 127.5
    out[..., 2] = (z + 1.0) * 127.5
    out[..., 3] = 255.0
    return out


# ---------------------------------------------------------------------------------------
# Blender images and materials
# ---------------------------------------------------------------------------------------

def _image_name(tex_name):
    return tex_name[:-4] if tex_name.lower().endswith('.tex') else tex_name


def build_image(tex_name, tex, role, flip_green=True):
    """Creates or refreshes the Blender image for one texture. role: 'diffuse', 'normal'
    or 'specular'.

    Images are matched by texture name AND pixel content, not by name alone. Texture names
    repeat across milos with different pixels - every Beatle's head milo has its own
    'hand_diffuse.tex' and 'hand_spec.tex' - and matching by name let the second import
    overwrite the first character's images. Now an image is refreshed in place only when it
    holds the same texture data (a re-import, where e.g. the normal-map green flip may have
    changed), and identical textures (skin_detail_normals, eye_cubetex) are still shared;
    different pixels get a new image, which Blender names 'hand_diffuse.001'.

    One texture can also be needed in two colour spaces - 21st_shoes.mat uses
    'billiejoe shoes spec.tex' as both diffuse (sRGB) and specular (Non-Color) - so each
    colour space gets its own image."""
    colorspace = 'sRGB' if role == 'diffuse' else 'Non-Color'
    if role == 'normal' and tex['encoding'] == ENC_ATI2:
        rgba = decode_normal(tex, flip_green)
    else:
        rgba = decode_rgba(tex)
    h, w = rgba.shape[:2]

    digest = hashlib.sha1(tex['data']).hexdigest()[:16]
    img = None
    for cand in bpy.data.images:
        if (cand.get("milo_texture") == tex_name and cand.get("milo_hash") == digest
                and cand.get("milo_colorspace") == colorspace):
            img = cand
            break
    if img is not None and tuple(img.size) != (w, h):
        bpy.data.images.remove(img)
        img = None
    if img is None:
        base = _image_name(tex_name)
        name = base
        other = bpy.data.images.get(base)
        if other is not None and other.get("milo_texture") == tex_name \
                and other.get("milo_hash") == digest:
            # same texture already here in the other colour space
            name = f"{base}_{'color' if colorspace == 'sRGB' else 'data'}"
        img = bpy.data.images.new(name, width=w, height=h, alpha=True)
    img.colorspace_settings.name = colorspace
    # Blender stores pixels bottom row first; the decoded image is top row first. Flipping
    # here is what makes the mesh importer's V-flipped UVs land on the right texels.
    pixels = np.ascontiguousarray(np.flipud(rgba) / 255.0, dtype=np.float32)
    img.pixels.foreach_set(pixels.ravel())
    img.update()
    try:
        img.pack()                        # keep it in the .blend - it has no file on disk
    except RuntimeError:
        pass
    img["milo_texture"] = tex_name
    img["milo_colorspace"] = colorspace
    img["milo_hash"] = digest
    return img


def _input(node, *names):
    for n in names:
        sock = node.inputs.get(n)
        if sock is not None:
            return sock
    return None


def _srgb_to_linear(c):
    c = max(0.0, float(c))
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _lin3(rgb):
    return tuple(_srgb_to_linear(c) for c in rgb[:3])


SHADER_GROUP_NAME = "Milo Character Shader"
SHADER_GROUP_VERSION = 4
SHADER_NODE_NAME = "MiloCharShader"

# (name, socket type, default, min, max). Colours are linear (Blender's working space);
# the material's own sRGB values are converted when they're written to the node.
_GROUP_INPUTS = (
    ("Diffuse", 'NodeSocketColor', (1.0, 1.0, 1.0, 1.0), None, None),
    ("Diffuse Alpha", 'NodeSocketFloat', 1.0, 0.0, 1.0),
    ("Material Color", 'NodeSocketColor', (1.0, 1.0, 1.0, 1.0), None, None),
    ("Material Alpha", 'NodeSocketFloat', 1.0, 0.0, 1.0),
    ("Intensify", 'NodeSocketFloat', 0.0, 0.0, 1.0),
    ("Normal Map", 'NodeSocketColor', (0.5, 0.5, 1.0, 1.0), None, None),
    ("Detail Normal", 'NodeSocketColor', (0.5, 0.5, 1.0, 1.0), None, None),
    ("Detail Strength", 'NodeSocketFloat', 0.0, 0.0, 4.0),
    ("Specular Map", 'NodeSocketColor', (1.0, 1.0, 1.0, 1.0), None, None),
    ("Gloss Map", 'NodeSocketFloat', 1.0, 0.0, 1.0),
    ("Gloss Map Influence", 'NodeSocketFloat', 1.0, 0.0, 1.0),
    ("Specular Color", 'NodeSocketColor', (0.0, 0.0, 0.0, 1.0), None, None),
    ("Specular Power", 'NodeSocketFloat', 10.0, 0.0, 10000.0),
    ("Specular 2 Color", 'NodeSocketColor', (0.0, 0.0, 0.0, 1.0), None, None),
    ("Specular 2 Power", 'NodeSocketFloat', 10.0, 0.0, 10000.0),
    ("Hair Anisotropy", 'NodeSocketFloat', 0.0, -1.0, 1.0),
    ("Rim Color", 'NodeSocketColor', (0.0, 0.0, 0.0, 1.0), None, None),
    ("Rim Power", 'NodeSocketFloat', 4.0, 0.0, 1000.0),
    ("Rim Always Show", 'NodeSocketFloat', 0.0, 0.0, 1.0),
    ("Rim Strength", 'NodeSocketFloat', 1.0, 0.0, 10.0),
    ("Emissive", 'NodeSocketColor', (0.0, 0.0, 0.0, 1.0), None, None),
    ("Emissive Multiplier", 'NodeSocketFloat', 1.0, 0.0, 100.0),
    ("Use Alpha", 'NodeSocketFloat', 0.0, 0.0, 1.0),
    ("Alpha Cut", 'NodeSocketFloat', 0.0, 0.0, 1.0),
    ("Alpha Threshold", 'NodeSocketFloat', 0.0, 0.0, 1.0),
    ("Additive", 'NodeSocketFloat', 0.0, 0.0, 1.0),
)


def ensure_shader_group():
    """The shared "Milo Character Shader" group (see the module docstring). Rebuilt when
    its stored version is older than this code's, so an improved group reaches .blend
    files imported with an older version on the next import."""
    ng = bpy.data.node_groups.get(SHADER_GROUP_NAME)
    if ng is not None and ng.get("milo_shader_version") == SHADER_GROUP_VERSION:
        return ng
    if ng is None:
        ng = bpy.data.node_groups.new(SHADER_GROUP_NAME, 'ShaderNodeTree')
    ng.interface.clear()
    ng.nodes.clear()
    for name, kind, default, lo, hi in _GROUP_INPUTS:
        it = ng.interface.new_socket(name=name, in_out='INPUT', socket_type=kind)
        it.default_value = default
        if lo is not None:
            it.min_value, it.max_value = lo, hi
    ng.interface.new_socket(name="Shader", in_out='OUTPUT', socket_type='NodeSocketShader')

    nodes, links = ng.nodes, ng.links
    gin = nodes.new('NodeGroupInput'); gin.location = (-1800, 0)
    gout = nodes.new('NodeGroupOutput'); gout.location = (1400, 0)
    I = gin.outputs

    def node(kind, x, y, **attrs):
        n = nodes.new(kind)
        n.location = (x, y)
        for k, v in attrs.items():
            setattr(n, k, v)
        return n

    def math(op, a, b=None, x=0, y=0, clamp=False):
        n = node('ShaderNodeMath', x, y, operation=op, use_clamp=clamp)
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = float(v)
            else:
                links.new(v, n.inputs[i])
        return n.outputs[0]

    def vmath(op, a, b=None, c=None, x=0, y=0):
        n = node('ShaderNodeVectorMath', x, y, operation=op)
        for i, v in enumerate((a, b, c)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = (float(v),) * 3
            elif isinstance(v, tuple):
                n.inputs[i].default_value = v
            else:
                links.new(v, n.inputs[i])
        return n

    def mul_rgb(a, b, x, y):
        n = node('ShaderNodeMix', x, y, data_type='RGBA', blend_type='MULTIPLY')
        n.inputs['Factor'].default_value = 1.0
        links.new(a, n.inputs[6])
        links.new(b, n.inputs[7])
        return n.outputs[2]

    # ---- normals: whiteout blend of the normal map and the tiled detail map ----
    # The material's strength is NOT a plain multiplier on the detail's slope: TBRB skin
    # uses 5.0 on a subtle map (mean tilt ~5 degrees), which multiplied out becomes ~25
    # degrees of noise - the "sandpaper" face. Strength s is read as the detail's weight
    # against the base in an additive blend, s/(1+s): 0.5 -> 1/3 (GDRB hair), 1 -> 1/2,
    # 5 -> 5/6 (TBRB skin). The result never exceeds the detail map's own slope.
    det_w = math('DIVIDE', I["Detail Strength"],
                 math('ADD', I["Detail Strength"], 1.0, x=-1300, y=-1000), x=-1300, y=-900)
    n1 = vmath('MULTIPLY_ADD', I["Normal Map"], 2.0, -1.0, x=-1500, y=-600)
    n2 = vmath('MULTIPLY_ADD', I["Detail Normal"], 2.0, -1.0, x=-1500, y=-800)
    s1 = node('ShaderNodeSeparateXYZ', -1300, -600)
    s2 = node('ShaderNodeSeparateXYZ', -1300, -800)
    links.new(n1.outputs[0], s1.inputs[0])
    links.new(n2.outputs[0], s2.inputs[0])
    dx = math('MULTIPLY', s2.outputs[0], det_w, x=-1100, y=-760)
    dy = math('MULTIPLY', s2.outputs[1], det_w, x=-1100, y=-900)
    cx = math('ADD', s1.outputs[0], dx, x=-900, y=-560)
    cy = math('ADD', s1.outputs[1], dy, x=-900, y=-700)
    # z: lerp(1, z2, strength) so strength 0 leaves the base map untouched
    z2 = math('SUBTRACT', s2.outputs[2], 1.0, x=-1100, y=-1040)
    z2 = math('MULTIPLY', z2, det_w, x=-900, y=-1040)
    z2 = math('ADD', z2, 1.0, x=-900, y=-1180)
    cz = math('MULTIPLY', s1.outputs[2], z2, x=-700, y=-840)
    comb = node('ShaderNodeCombineXYZ', -500, -650)
    links.new(cx, comb.inputs[0]); links.new(cy, comb.inputs[1]); links.new(cz, comb.inputs[2])
    nrm = vmath('NORMALIZE', comb.outputs[0], x=-330, y=-650)
    enc = vmath('MULTIPLY_ADD', nrm.outputs[0], 0.5, 0.5, x=-160, y=-650)
    nmap = node('ShaderNodeNormalMap', 20, -650, space='TANGENT')
    links.new(enc.outputs[0], nmap.inputs['Color'])
    normal = nmap.outputs['Normal']

    # ---- diffuse: texture x material colour (x2 with intensify) ----
    base = mul_rgb(I["Diffuse"], I["Material Color"], -1300, 500)
    boost = math('ADD', I["Intensify"], 1.0, x=-1300, y=300)
    base_v = vmath('SCALE', base, x=-1100, y=480)
    links.new(boost, base_v.inputs['Scale'])
    base = base_v.outputs[0]

    # ---- rim: rimRGB x (1 - N.V)^power ----
    lw = node('ShaderNodeLayerWeight', -700, 900)
    lw.inputs['Blend'].default_value = 0.5
    links.new(normal, lw.inputs['Normal'])
    rim_f = math('POWER', lw.outputs['Facing'], I["Rim Power"], x=-500, y=900, clamp=True)
    rim_f = math('MULTIPLY', rim_f, I["Rim Strength"], x=-330, y=900)
    rim_f = math('MULTIPLY', rim_f, I["Rim Always Show"], x=-160, y=900)
    rim_c = vmath('SCALE', I["Rim Color"], x=20, y=900)
    links.new(rim_f, rim_c.inputs['Scale'])
    emis = vmath('SCALE', I["Emissive"], x=20, y=1060)
    links.new(I["Emissive Multiplier"], emis.inputs['Scale'])
    emis_total = vmath('ADD', rim_c.outputs[0], emis.outputs[0], x=200, y=980)

    # lit rim -> sheen; roughness from the rim power
    not_always = math('SUBTRACT', 1.0, I["Rim Always Show"], x=-500, y=700, clamp=True)
    sheen_w = math('MULTIPLY', not_always, I["Rim Strength"], x=-330, y=700)
    sq = math('SQRT', I["Rim Power"], x=-500, y=560)
    sq = math('MAXIMUM', sq, 0.001, x=-330, y=560)
    # 0.6/sqrt(power): power 4 -> 0.3, 25 -> 0.12. Tight enough to stay at the silhouette
    # like the game's (1-N.V)^power; broader sheen hazes the whole surface.
    sheen_r = math('DIVIDE', 0.6, sq, x=-160, y=560, clamp=True)

    bsdf = node('ShaderNodeBsdfPrincipled', 450, 600)
    links.new(base, bsdf.inputs['Base Color'])
    for nm, v in (('Metallic', 0.0), ('Roughness', 1.0), ('IOR', 1.5),
                  ('Specular IOR Level', 0.0), ('Emission Strength', 1.0)):
        sock = _input(bsdf, nm)
        if sock is not None:
            sock.default_value = v
    links.new(normal, bsdf.inputs['Normal'])
    for nm, src in (('Sheen Weight', sheen_w), ('Sheen Roughness', sheen_r)):
        sock = _input(bsdf, nm)
        if sock is not None:
            links.new(src, sock)
    sock = _input(bsdf, 'Sheen Tint')
    if sock is not None:
        links.new(I["Rim Color"], sock)
    sock = _input(bsdf, 'Emission Color', 'Emission')
    if sock is not None:
        links.new(emis_total.outputs[0], sock)

    # ---- specular lobes (added on top, as the game does) ----
    # The specular map is stored Non-Color (its alpha is data), but its RGB is a gamma-space
    # colour like the material's specular colour, so it's linearised here too. Left raw, a
    # mid-grey map (TBRB faces average ~0.3) counted ~4x too strong and washed skin white.
    spec_map = node('ShaderNodeGamma', -1300, -100)
    spec_map.inputs['Gamma'].default_value = 2.2
    links.new(I["Specular Map"], spec_map.inputs['Color'])
    tangent = node('ShaderNodeTangent', 200, -350, direction_type='UV_MAP')
    gloss = node('ShaderNodeMix', -1300, 100, data_type='FLOAT')
    gloss.inputs[2].default_value = 1.0
    links.new(I["Gloss Map Influence"], gloss.inputs[0])
    links.new(I["Gloss Map"], gloss.inputs[3])

    def lobe(color_in, power_in, scale_power_by_gloss, y):
        n = power_in
        if scale_power_by_gloss:
            n = math('MULTIPLY', n, gloss.outputs[0], x=-1100, y=y)
        n = math('MAXIMUM', n, 1.0, x=-900, y=y)
        n2 = math('ADD', n, 2.0, x=-700, y=y)
        rough = math('POWER', math('DIVIDE', 2.0, n2, x=-500, y=y), 0.25, x=-330, y=y)
        # Reflected energy matched to the game's lobe, spec x (N.H)^n x N.L, over the
        # hemisphere: 8/(n+7) (fit within 10% from n=1 to 200). The earlier 8/(n+2)
        # matched only the peak and ignored N.L, overstating broad lobes ~1.5x - and a
        # broad lobe under Blender's environment light reads as a pale layer over the
        # whole surface (George's "make-up"), since the game's ambient never reaches its
        # specular. Tight lobes are unchanged: n=50 -> 0.14 either way.
        n7 = math('ADD', n, 7.0, x=-700, y=y - 140)
        energy = math('MINIMUM', math('DIVIDE', 8.0, n7, x=-500, y=y - 140), 1.0,
                      x=-330, y=y - 140)
        col = mul_rgb(color_in, spec_map.outputs['Color'], -500, y - 300)
        col_v = vmath('SCALE', col, x=-160, y=y - 200)
        links.new(energy, col_v.inputs['Scale'])
        g = node('ShaderNodeBsdfGlossy', 450, y)
        try:
            # Beckmann is what the power->roughness match (alpha^2 = 2/(n+2)) is exact
            # for; GGX's long tail adds a broad haze the game's Blinn-Phong doesn't have.
            g.distribution = 'BECKMANN'
        except (TypeError, AttributeError):
            pass
        links.new(col_v.outputs[0], g.inputs['Color'])
        links.new(rough, g.inputs['Roughness'])
        links.new(normal, g.inputs['Normal'])
        for nm, src in (('Anisotropy', I["Hair Anisotropy"]),
                        ('Tangent', tangent.outputs['Tangent'])):
            sock = _input(g, nm)
            if sock is not None:
                links.new(src, sock)
        return g.outputs[0]

    spec1 = lobe(I["Specular Color"], I["Specular Power"], True, 100)
    spec2 = lobe(I["Specular 2 Color"], I["Specular 2 Power"], False, -250)

    add1 = node('ShaderNodeAddShader', 750, 200)
    links.new(spec1, add1.inputs[0]); links.new(spec2, add1.inputs[1])
    surface = node('ShaderNodeAddShader', 900, 400)
    links.new(bsdf.outputs[0], surface.inputs[0]); links.new(add1.outputs[0], surface.inputs[1])

    # ---- alpha: diffuse x material alpha, optionally cut at the threshold ----
    a = math('MULTIPLY', I["Diffuse Alpha"], I["Material Alpha"], x=-1300, y=-1250)
    cut = math('GREATER_THAN', a, I["Alpha Threshold"], x=-1100, y=-1400)
    am = node('ShaderNodeMix', -900, -1250, data_type='FLOAT')
    links.new(I["Alpha Cut"], am.inputs[0]); links.new(a, am.inputs[2])
    links.new(cut, am.inputs[3])
    use = math('MAXIMUM', I["Use Alpha"], I["Alpha Cut"], x=-900, y=-1450)
    alpha = node('ShaderNodeMix', -700, -1300, data_type='FLOAT')
    alpha.inputs[2].default_value = 1.0
    links.new(use, alpha.inputs[0]); links.new(am.outputs[0], alpha.inputs[3])

    transp = node('ShaderNodeBsdfTransparent', 900, 0)
    blended = node('ShaderNodeMixShader', 1050, 300)
    links.new(alpha.outputs[0], blended.inputs[0])
    links.new(transp.outputs[0], blended.inputs[1])
    links.new(surface.outputs[0], blended.inputs[2])

    # additive (kBlendAdd / kBlendSrcAlphaAdd): background + colour x alpha
    add_col = vmath('SCALE', base, x=700, y=-200)
    links.new(alpha.outputs[0], add_col.inputs['Scale'])
    add_em = node('ShaderNodeEmission', 900, -200)
    links.new(add_col.outputs[0], add_em.inputs['Color'])
    add_sh = node('ShaderNodeAddShader', 1050, -100)
    links.new(transp.outputs[0], add_sh.inputs[0]); links.new(add_em.outputs[0], add_sh.inputs[1])
    final = node('ShaderNodeMixShader', 1220, 100)
    links.new(I["Additive"], final.inputs[0])
    links.new(blended.outputs[0], final.inputs[1])
    links.new(add_sh.outputs[0], final.inputs[2])
    links.new(final.outputs[0], gout.inputs["Shader"])

    ng["milo_shader_version"] = SHADER_GROUP_VERSION
    return ng


def _set_render_method(mat, method):
    """'OPAQUE', 'CLIP' or 'BLEND', across the Blender versions that renamed these."""
    if hasattr(mat, 'surface_render_method'):
        try:
            mat.surface_render_method = 'BLENDED' if method == 'BLEND' else 'DITHERED'
        except (TypeError, AttributeError):
            pass
    if hasattr(mat, 'blend_method'):
        try:
            mat.blend_method = method
        except (TypeError, AttributeError):
            pass


# Per-game scale on normalDetailStrength, calibrated by eye in Blender rather than taken
# from the game's shader. TBRB authors 5.0 on its skin; 0.5 matches the game's look (tested
# on George), while GDRB's hair value of 0.5 already looks right as is.
DETAIL_STRENGTH_SCALE = {55: 0.1}


def _is_hair(slots):
    return (slots.get('shader_variation') == SHADER_VARIATION_HAIR
            and max(slots.get('specular2_rgb') or (0.0,)) > 0.0)


def build_material_tree(mat, slots, images):
    """Replaces a material's nodes with image nodes feeding the Milo Character Shader group,
    its inputs set from the parsed material. images: {'diffuse', 'normal', 'specular',
    'emissive', 'detail'} -> Blender image or None."""
    try:
        mat.use_nodes = True
    except AttributeError:
        pass
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    for n in list(nodes):
        nodes.remove(n)

    out = nodes.new('ShaderNodeOutputMaterial'); out.location = (500, 300)
    grp = nodes.new('ShaderNodeGroup')
    grp.node_tree = ensure_shader_group()
    grp.name = SHADER_NODE_NAME
    grp.label = SHADER_GROUP_NAME
    grp.location = (150, 300)
    grp.width = 260
    links.new(grp.outputs["Shader"], out.inputs['Surface'])
    gi = grp.inputs

    def tex_node(img, y, label, x=-450):
        n = nodes.new('ShaderNodeTexImage')
        n.image = img
        n.label = label
        n.name = f"Milo_{label.replace(' ', '')}"
        n.location = (x, y)
        return n

    col = slots.get('color') or (1.0, 1.0, 1.0, 1.0)
    gi["Material Color"].default_value = (*_lin3(col), 1.0)
    gi["Material Alpha"].default_value = float(col[3])
    gi["Intensify"].default_value = 1.0 if slots.get('intensify') else 0.0

    if images.get('diffuse') is not None:
        d = tex_node(images['diffuse'], 700, "Diffuse")
        links.new(d.outputs['Color'], gi["Diffuse"])
        links.new(d.outputs['Alpha'], gi["Diffuse Alpha"])
    if images.get('normal') is not None:
        n = tex_node(images['normal'], 400, "Normal")
        links.new(n.outputs['Color'], gi["Normal Map"])
    if images.get('detail') is not None:
        uv = nodes.new('ShaderNodeTexCoord'); uv.location = (-1000, 150)
        mp = nodes.new('ShaderNodeMapping'); mp.location = (-800, 150)
        t = float(slots.get('detail_tiling') or 1.0)
        mp.inputs['Scale'].default_value = (t, t, 1.0)
        links.new(uv.outputs['UV'], mp.inputs['Vector'])
        dn = tex_node(images['detail'], 150, "Detail Normal")
        links.new(mp.outputs['Vector'], dn.inputs['Vector'])
        links.new(dn.outputs['Color'], gi["Detail Normal"])
        gi["Detail Strength"].default_value = (float(slots.get('detail_strength') or 0.0)
                                               * DETAIL_STRENGTH_SCALE.get(
                                                   slots.get('revision'), 1.0))

    hair = _is_hair(slots)
    if images.get('specular') is not None:
        sp = tex_node(images['specular'], -150, "Specular")
        links.new(sp.outputs['Color'], gi["Specular Map"])
        links.new(sp.outputs['Alpha'], gi["Gloss Map"])
    gi["Gloss Map Influence"].default_value = 0.0 if hair else 1.0
    gi["Specular Color"].default_value = (*_lin3(slots.get('specular_rgb') or (0, 0, 0)), 1.0)
    gi["Specular Power"].default_value = float(slots.get('specular_power') or 0.0)
    gi["Specular 2 Color"].default_value = (*_lin3(slots.get('specular2_rgb') or (0, 0, 0)),
                                            1.0)
    gi["Specular 2 Power"].default_value = float(slots.get('specular2_power') or 10.0)
    gi["Hair Anisotropy"].default_value = 0.8 if hair else 0.0

    gi["Rim Color"].default_value = (*_lin3(slots.get('rim_rgb') or (0, 0, 0)), 1.0)
    gi["Rim Power"].default_value = float(slots.get('rim_power') or 4.0)
    gi["Rim Always Show"].default_value = 1.0 if slots.get('rim_always_show') else 0.0

    if images.get('emissive') is not None:
        e = tex_node(images['emissive'], -450, "Emissive")
        links.new(e.outputs['Color'], gi["Emissive"])
    gi["Emissive Multiplier"].default_value = float(slots.get('emissive_multiplier', 1.0))

    blend = slots.get('blend', BLEND_SRC)
    alpha_cut = bool(slots.get('alpha_cut'))
    use_alpha = blend in (BLEND_SRC_ALPHA, BLEND_PREMULT, BLEND_SRC_ALPHA_ADD)
    gi["Use Alpha"].default_value = 1.0 if use_alpha else 0.0
    gi["Alpha Cut"].default_value = 1.0 if alpha_cut else 0.0
    gi["Alpha Threshold"].default_value = min(1.0, slots.get('alpha_threshold', 0) / 255.0)
    gi["Additive"].default_value = 1.0 if blend in (BLEND_ADD, BLEND_SRC_ALPHA_ADD) else 0.0
    if use_alpha or blend in (BLEND_ADD, BLEND_SRC_ALPHA_ADD):
        _set_render_method(mat, 'BLEND')
    elif alpha_cut:
        _set_render_method(mat, 'CLIP')
    else:
        _set_render_method(mat, 'OPAQUE')
    mat.use_backface_culling = bool(slots.get('cull'))
    mat["milo_textured"] = True


_BLEND_ENUM = ('kBlendDest', 'kBlendSrc', 'kBlendAdd', 'kBlendSrcAlpha', 'kBlendSrcAlphaAdd',
               'kBlendSubtract', 'kBlendMultiply', 'kPreMultAlpha')
_ZMODE_ENUM = ('kZModeDisable', 'kZModeNormal', 'kZModeTransparent', 'kZModeForce',
               'kZModeDecal')
_VARIATION_ENUM = ('kShaderVariationNone', 'kShaderVariationSkin', 'kShaderVariationHair')


def fill_material_settings(mat, slots, images):
    """Copies the parsed values into the material's glTFMilo settings panel, so an imported
    material shows (and would export with) the game's own values. The panel's colours are
    the game's raw values, not linearised. Fields the panel has no control for are kept as
    custom properties on the material."""
    st = getattr(mat, 'gltfmilo_settings', None)
    if st is None:
        return

    def put(attr, value):
        try:
            setattr(st, attr, value)
        except (AttributeError, TypeError, ValueError):
            pass

    def clamp01(v):
        return max(0.0, min(1.0, float(v)))

    blend = slots.get('blend', BLEND_SRC)
    if 0 <= blend < len(_BLEND_ENUM):
        put('blend', _BLEND_ENUM[blend])
    put('base_color', tuple(clamp01(c) for c in slots.get('color', (1, 1, 1, 1))))
    put('pre_lit', bool(slots.get('prelit')))
    put('intensify', bool(slots.get('intensify')))
    put('use_environment', bool(slots.get('use_environ')))
    zm = slots.get('z_mode', 1)
    if 0 <= zm < len(_ZMODE_ENUM):
        put('z_mode', _ZMODE_ENUM[zm])
    put('alpha_cut', bool(slots.get('alpha_cut')))
    put('alpha_threshold', int(min(255, slots.get('alpha_threshold', 0))))
    for attr, role in (('diffuse_tex', 'diffuse'), ('normal_tex', 'normal'),
                       ('specular_tex', 'specular'), ('emissive_tex', 'emissive')):
        put(attr, images.get(role))
    put('specular_rgb', tuple(clamp01(c) for c in slots.get('specular_rgb', (0, 0, 0))))
    put('specular_power', float(slots.get('specular_power', 0.0)))
    put('emissive_multiplier', float(slots.get('emissive_multiplier', 1.0)))
    put('environment_map', bool(slots.get('environ_map')))
    put('cull', bool(slots.get('cull')))
    dn = slots.get('de_normal', 0.0)
    put('de_normal', '-1' if dn < -0.5 else ('1' if dn > 0.5 else '0'))
    put('anisotropy', float(slots.get('anisotropy', 0.0)))
    put('normal_detail_strength', float(slots.get('detail_strength', 0.0)))
    put('rim_rgb', tuple(clamp01(c) for c in slots.get('rim_rgb', (0, 0, 0))))
    put('rim_power', float(slots.get('rim_power', 4.0)))
    sv = slots.get('shader_variation', 0)
    if 0 <= sv < len(_VARIATION_ENUM):
        put('shader_variation', _VARIATION_ENUM[sv])

    mat["milo_rim_always_show"] = bool(slots.get('rim_always_show'))
    mat["milo_specular2_rgb"] = list(slots.get('specular2_rgb', (0.0, 0.0, 0.0)))
    mat["milo_specular2_power"] = float(slots.get('specular2_power', 0.0))
    mat["milo_normal_detail_map"] = slots.get('detail_normal', '')
    mat["milo_normal_detail_tiling"] = float(slots.get('detail_tiling', 1.0))
    mat["milo_color_adjust"] = bool(slots.get('color_adjust'))
    mat["milo_point_lights"] = bool(slots.get('point_lights'))


def _viewport_color(mat, slots, diffuse_img):
    """Solid-mode colour: the diffuse texture's average x the material colour."""
    rgb = list(_lin3(slots.get('color') or (1, 1, 1)))
    if diffuse_img is not None:
        try:
            px = np.empty(len(diffuse_img.pixels), dtype=np.float32)
            diffuse_img.pixels.foreach_get(px)
            mean = px.reshape(-1, 4)[:, :3].mean(0)
            rgb = [rgb[i] * float(mean[i]) for i in range(3)]
        except Exception:
            pass
    try:
        mat.diffuse_color = (*rgb, 1.0)
        mat.roughness = 1.0
        mat.metallic = 0.0
    except (AttributeError, TypeError):
        pass


def _tex_identity(name, texs):
    """What a texture slot really holds: a hash of its pixels, following a TexBlender's
    render target to its base map, or the name when the texture isn't available."""
    if not name:
        return None
    t = texs.get(name)
    if t is not None and t.get('render_target') and t.get('blend_base') in texs:
        name = t['blend_base']
        t = texs[name]
    if t is None or not t.get('data'):
        return ('name', name)
    return ('pixels', hashlib.sha1(t['data']).hexdigest())


def _borrowed_specular(mat_name, slots, mats, texs):
    """(material, texture) to lend a material that has no specular map, or None.

    Some TBRB head milos leave the map off: Ringo's hands.mat has none while his nails.mat,
    on the same hand textures, uses hand_spec.tex; John's SHORT-hands milo has none on
    j_head/j_head01/j_hands, while his long-hands milo - identical textures - has
    john_spec.tex on all three. With no map the material's specular colour (0.925 at power
    5) covers the whole surface and, under Blender's environment lighting, turns it white.

    So a material missing the map borrows one from a material in the same milo with the
    same diffuse texture. "Same" means the same pixels, not the same name: John's milo
    carries the face texture twice, as j_diffuse.tex (head) and john_diffuse.tex
    (eyebrows), byte-for-byte identical, and only the eyebrow material keeps john_spec.tex.
    A lender with the same normal map too is preferred. Materials that never use specular
    (black specular colour) borrow nothing."""
    if slots.get('specular') or not slots.get('diffuse'):
        return None
    if max(slots.get('specular_rgb') or (0.0,)) <= 0.0:
        return None
    my_diff = _tex_identity(slots['diffuse'], texs)
    my_norm = _tex_identity(slots.get('normal'), texs)
    best = None
    for other, o in sorted(mats.items()):
        if other == mat_name or not o.get('specular'):
            continue
        if _tex_identity(o.get('diffuse'), texs) != my_diff:
            continue
        rank = 0 if (my_norm is not None
                     and _tex_identity(o.get('normal'), texs) == my_norm) else 1
        if best is None or rank < best[0]:
            best = (rank, other, o['specular'])
    return (best[1], best[2]) if best else None


def apply_textures(filepath, material_names, flip_green=True, source=None):
    """Builds the Milo Character Shader node tree for the named materials from a milo's own
    Tex and Mat entries, and fills each material's glTFMilo settings. Textures the materials
    name but this milo doesn't hold are looked up automatically in the milos it references
    (see find_shared_milo). Returns (textured count, image count, [notes], [names of
    textures that couldn't be found anywhere]). source: the milo's file name, scoping its
    materials as mesh_importer.milo_material does (defaults to this file's name)."""
    from .mesh_importer import milo_material
    if source is None:
        source = os.path.basename(filepath)
    texs, mats, failed = read_milo_materials(filepath)
    notes = [f"skipped {n}: {r}" for n, r in failed]
    images = {}

    wanted = set()
    for mat_name in material_names:
        slots = mats.get(mat_name) or {}
        for key in ('diffuse', 'normal', 'specular', 'emissive', 'detail_normal'):
            if slots.get(key):
                wanted.add(slots[key])
    for mat_name in material_names:
        slots = mats.get(mat_name)
        if slots:
            b = _borrowed_specular(mat_name, slots, mats, texs)
            if b:
                wanted.add(b[1])
    missing = {n for n in wanted if n not in texs}
    if missing:
        texs.update(_shared_textures(filepath, missing, notes))
    not_found = sorted(n for n in missing if n not in texs)

    def image_for(tex_name, role):
        if not tex_name:
            return None
        if tex_name not in texs:
            return None             # reported once, in not_found
        if texs[tex_name].get('render_target'):
            base = texs[tex_name].get('blend_base')
            if base and base in texs and not texs[base].get('render_target'):
                notes.append(f"'{tex_name}' is filled by a TexBlender at runtime; using its "
                             f"base map '{base}' (wrinkles not shown)")
                tex_name = base
            else:
                notes.append(f"'{tex_name}' is a runtime render target with no pixels; "
                             f"slot left empty")
                return None
        img_role = 'normal' if role == 'detail' else role
        key = (tex_name, img_role)
        if key not in images:
            try:
                images[key] = build_image(tex_name, texs[tex_name], img_role, flip_green)
            except (TextureImportError, ValueError) as e:
                notes.append(f"could not decode '{tex_name}': {e}")
                images[key] = None
        return images[key]

    done = 0
    for mat_name in material_names:
        slots = mats.get(mat_name)
        if slots is None:
            notes.append(f"material '{mat_name}' isn't defined in this milo")
            continue
        mat = milo_material(mat_name, source)
        if not slots.get('tail_ok'):
            notes.append(f"'{mat_name}': rim, detail-normal and hair settings couldn't be "
                         f"read; neutral values used")
        imgs = {role: image_for(slots.get(key), role)
                for role, key in (('diffuse', 'diffuse'), ('normal', 'normal'),
                                  ('specular', 'specular'), ('emissive', 'emissive'),
                                  ('detail', 'detail_normal'))}
        own_imgs = dict(imgs)
        borrowed = _borrowed_specular(mat_name, slots, mats, texs)
        if imgs['specular'] is None and borrowed:
            imgs['specular'] = image_for(borrowed[1], 'specular')
            if imgs['specular'] is not None:
                notes.append(f"'{mat_name}' has no specular map; using '{borrowed[1]}' from "
                             f"'{borrowed[0]}', which shares its diffuse texture")
        build_material_tree(mat, slots, imgs)
        if imgs['specular'] is not None and own_imgs['specular'] is None:
            # A borrowed map's alpha is tuned to the lender's specular power, not this
            # material's, so only its colour (where the surface is shiny) is used.
            grp = mat.node_tree.nodes.get(SHADER_NODE_NAME)
            if grp is not None:
                grp.inputs["Gloss Map Influence"].default_value = 0.0
        # The panel keeps the material's own slots, so a re-export writes what the game
        # file had rather than the borrowed preview map.
        fill_material_settings(mat, slots, own_imgs)
        mat["milo_borrowed_specular"] = borrowed[1] if (borrowed and imgs['specular']
                                                        is not None
                                                        and own_imgs['specular'] is None) \
            else ""
        _viewport_color(mat, slots, imgs.get('diffuse'))
        if slots.get('blend') in (BLEND_SUBTRACT, BLEND_MULTIPLY, BLEND_DEST):
            notes.append(f"'{mat_name}': blend mode {slots['blend']} has no Blender "
                         f"equivalent here; drawn opaque")
        done += 1
    if not_found:
        notes.append(f"{len(not_found)} texture(s) not found in this milo or any milo it "
                     f"references: " + ", ".join(not_found))
    return done, len([i for i in images.values() if i is not None]), notes, not_found
