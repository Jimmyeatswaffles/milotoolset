"""
Rock Band 2 character-creator textures.

Adds a "Rock Band 2 Character Creator Texture" panel to the Material tab. Its "Load
Customizable Textures" button reads an RB2 texture-variant milo (armyjacket_dirty and
the like), replaces the importer's placeholder material with a real node tree, and -
when the item's OutfitConfig (.cfg) says it's colourable - enables one or two colour
pickers filled from the game's ColorPalette milo (colorpalettes.milo_xbox).

===========================================================================================
FORMAT - verified on armyjacket_dirty.milo_xbox and colorpalettes.milo_xbox (RB2, Xbox 360)
===========================================================================================
A texture-variant milo is a revision-25 Character dir holding Tex entries and an
OutfitConfig. The textures don't say which material they belong to; the DIRECTORY's own
property DataArray does, as symbol/value pairs:

    target          torso_armyjacket.mat            (Mat this slot dresses)
    diffuse_tex     armyjacket_dirty_diffuse.tex    DXT5: greyscale RGB + alpha
    two_color_mask  armyjacket_dirty_mask.tex       DXT1 greyscale
    normal_map      armyjacket_dirty_norm.tex       ATI2
    specular_map    (empty)
    specular_rgb    0x353535                        packed int
    specular_power  15.0
    target2 / diffuse_tex2 / two_color2 / normal_map2 / specular_rgb2 / ...  next slot

Items that carry their own geometry (rings, glasses, facial hair) have no directory
properties at all; their texture names come from their own Mat entries (RndMat revision
47, read here). The Mat's colour is the cfg's default palette colour baked in - goggles
carry cloth_dulled[6], chops hair[12], the chief ring metal[0] - so the game tints those
items through the material colour: diffuse x colour, or just the colour when there's no
diffuse (the ring). The same node group covers both, with Diffuse defaulting to white.

Note 'two_color_mask' (a texture) and 'two_color' (an int flag) are DIFFERENT keys - the
strap jacket carries both. The flag is logged but not used; the cfg decides the channels.

How the two masks combine was established from the pixels and confirmed by recolouring
with the cfg's own defaults (olive body, dark red plaid lining - an army jacket):

    diffuse alpha   0 = colour 1, 1 = colour 2   (the lining is the only alpha-white area)
    two_color_mask  1 = keep the diffuse's own colour (patches, tag, buckles, stitching -
                    the only saturated areas of an otherwise greyscale diffuse)

    result = mix(diffuse * mix(colour1, colour2, alpha), diffuse, mask)   (linear space)

An item with only ONE colour uses its diffuse alpha the other way - as the keep mask:
rolledjeans_cowspot's alpha holds the cow spots and the rolled cuffs, oringhoodie_cotton
_belt's the metal O-ring and grommets, rolledjeans_cowspot_belt's the buckle; with no
second colour to select, they stay their own colour while the rest takes the palette
(a two_color_mask texture, where there is one, still takes precedence).

OutfitConfig revision 3: a 17-byte object header, then a count of colour CHOICES (one
swatch pick each in the game's UI), each

    u32 default colour index, u32 n, n x symbol material,
    symbol palette-for-colour-1, symbol palette-for-colour-2, u32 (0 in retail)

A choice drives colour 1 of its materials if the first palette is filled, colour 2 if
the second is - or BOTH from one index when both are, as facehair.cfg does with
(12, hair.pal, hair_secondary.pal). torso.cfg is two single-channel choices: (30,
cloth_dulled.pal, '') and (14, '', cloth_dulled.pal). Verified end to end on torso.cfg x2,
glasses.cfg, rings.cfg and facehair.cfg; a cfg that doesn't parse falls back to collecting
its .mat / .pal names.

Palettes stored inside an item's own milo (rings_chief.pal, facehair_color.pal) are not
referenced by its cfg and rings_chief holds debug colours, so the global palettes milo is
what's used.

Milo shader (optional, "Use Milo Shader" on import): the same game-style shading the GDRB
and TBRB importers use (texture_importer's "Milo Character Shader" group - Blinn-Phong
specular added on top, the specular map's alpha as gloss, alpha/additive blending). RB2's
material revision 47 has no rim light, second highlight or shader variation, so only the
diffuse, normal and specular parts apply. RB2 specular maps follow the same pattern as the
later games': a greyscale RGB mask with gloss in the alpha (basictightpants_jeans_spec:
denim ~0.37, rivets and zip ~0.75+).

Everything the creator system does - two-colour tinting, makeup, fur, baking - still runs
through the Principled BSDF's Base Color and Alpha inputs, so that node stays as the colour
hub; with the Milo shader on it's simply no longer connected to the output. Instead
sync_milo_shader() feeds whatever reaches the Principled BSDF into the Milo group, and is
called again whenever those links change (makeup, fur, bake, restore).

ColorPalette revision 1: the same 17-byte header, a u32 count, then count x 4 floats. Only
RGB is meaningful (the fourth float is uninitialised junk). The values are exact 8-bit
sRGB (0.549 = 140/255), so they're converted to linear before going into a node.
"""

import json
import os
import re
import struct

import bpy
import bpy.utils.previews
from bpy.props import (BoolProperty, EnumProperty, IntProperty, PointerProperty,
                       StringProperty)
from bpy_extras.io_utils import ImportHelper

from .io import read_milo_container_body
from .texture_importer import (parse_tex, build_image, TextureImportError,
                               TEX_RENDERED)
from .utilities import _log


MILO_END_MARKER = b'\xAD\xDE\xAD\xDE'
OBJECT_HEADER_END = 17          # revision + 13-byte object header (rev-25 dirs)
PALETTE_REVISION = 1
GROUP_NAME = "RB2 Two-Color"
GROUP_VERSION = 1
GROUP_NODE_NAME = "RB2CC_TwoColor"
PALETTE_MILO_NAMES = ("colorpalettes.milo_xbox", "colorpalettes.milo_ps3",
                      "colorpalettes.milo_wii", "colorpalettes.milo")

# DataArray node types used by the directory properties.
_DA_INT, _DA_FLOAT, _DA_OBJECT, _DA_SYMBOL, _DA_STRING = 0, 1, 4, 5, 0x12
_PROP_KEYS = ('target', 'diffuse_tex', 'two_color_mask', 'two_color', 'normal_map',
              'specular_map', 'specular_rgb', 'specular_power')
_TEX_KEYS = ('diffuse_tex', 'two_color_mask', 'normal_map', 'specular_map')
MAT_REVISION_RB2 = 47
BLEND_ADD, BLEND_SRC_ALPHA, BLEND_PREMULT = 2, 3, 7
_PROP_KEY_RE = re.compile(r'^([a-z_]+?)(\d*)$')
_BLENDER_DUP_RE = re.compile(r'\.\d{3}$')


class CreatorTextureError(Exception):
    pass


# ---------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------

def _u32(b, o):
    return struct.unpack_from('>I', b, o)[0]


def _sym_at(b, o, max_len=512):
    n = _u32(b, o)
    if n > max_len or o + 4 + n > len(b):
        raise CreatorTextureError(f"implausible string length {n} at {o:#x}")
    return b[o + 4:o + 4 + n].decode('latin-1'), o + 4 + n


def parse_dir_texture_slots(meta):
    """The directory's material slots, from the symbol/value pairs in its property
    DataArray. Returns [slot dict] in slot order (target, target2, ...)."""
    found = {}
    p = 0
    while p + 12 <= len(meta):
        if _u32(meta, p) != _DA_SYMBOL:
            p += 1
            continue
        n = _u32(meta, p + 4)
        if not 3 <= n <= 40 or p + 8 + n + 8 > len(meta):
            p += 1
            continue
        raw = meta[p + 8:p + 8 + n]
        if not all(97 <= c <= 122 or c == 95 or 48 <= c <= 57 for c in raw):
            p += 1
            continue
        key = raw.decode('latin-1')
        m = _PROP_KEY_RE.match(key)
        if not m or m.group(1) not in _PROP_KEYS:
            p += 1
            continue
        q = p + 8 + n
        vtype = _u32(meta, q)
        try:
            if vtype in (_DA_OBJECT, _DA_SYMBOL, _DA_STRING):
                value, end = _sym_at(meta, q + 4)
            elif vtype == _DA_INT:
                value, end = _u32(meta, q + 4), q + 8
            elif vtype == _DA_FLOAT:
                value, end = struct.unpack_from('>f', meta, q + 4)[0], q + 8
            else:
                p += 1
                continue
        except (CreatorTextureError, struct.error):
            p += 1
            continue
        base = m.group(1)
        if base in _TEX_KEYS and not isinstance(value, str):
            value = ''
        slot = int(m.group(2) or 1)
        found.setdefault(slot, {})[base] = value
        p = end
    slots = []
    for k in sorted(found):
        s = found[k]
        if s.get('target'):
            s['slot'] = k
            slots.append(s)
    return slots


def parse_outfit_config(data):
    """OutfitConfig revision 3 -> [{'index', 'mats', 'palette1', 'palette2'}]."""
    try:
        p = OBJECT_HEADER_END
        count = _u32(data, p); p += 4
        if count > 16:
            raise CreatorTextureError(f"implausible colour count {count}")
        out = []
        for k in range(count):
            idx = _u32(data, p); p += 4
            n = _u32(data, p); p += 4
            if n > 64:
                raise CreatorTextureError(f"implausible material count {n}")
            mats = []
            for _ in range(n):
                s, p = _sym_at(data, p)
                mats.append(s)
            pal_a, p = _sym_at(data, p)
            pal_b, p = _sym_at(data, p)
            p += 4
            out.append(dict(index=idx, mats=mats, palette1=pal_a, palette2=pal_b))
        if p != len(data):
            raise CreatorTextureError(f"parsed {p} of {len(data)} bytes")
        return out, None
    except (CreatorTextureError, struct.error) as e:
        # Unverified layout: keep the names so the panel can still say what it found.
        return _outfit_config_fallback(data), f"layout not recognised ({e}); guessed"


def _outfit_config_fallback(data):
    syms, p = [], 0
    while p + 4 <= len(data):
        n = _u32(data, p)
        if 3 <= n <= 128 and p + 4 + n <= len(data) and \
                all(32 <= c < 127 for c in data[p + 4:p + 4 + n]):
            syms.append(data[p + 4:p + 4 + n].decode('latin-1'))
            p += 4 + n
            continue
        p += 1
    mats = [s for s in syms if s.lower().endswith('.mat')]
    pals = [s for s in syms if s.lower().endswith('.pal')]
    return [dict(index=0, mats=mats, palette1=pal if i == 0 else '',
                 palette2=pal if i == 1 else '') for i, pal in enumerate(pals[:2])]


_DA_WORD = {0, 1, 6, 8, 9, 36}
_DA_TEXT = {2, 3, 4, 5, 7, 18, 32, 33, 34, 35, 37}
_DA_TREE = {16, 17, 19}


def _skip_data_array(c, p, header, depth=0):
    """Steps over a binary DataArray: u16 node count, u32 line, a u16 field (header 8) or
    not (header 6), then the nodes."""
    if depth > 32:
        raise CreatorTextureError("property tree nested implausibly deep")
    count = struct.unpack_from('>h', c, p)[0]
    if not 0 <= count <= 4096:
        raise CreatorTextureError(f"implausible property node count {count}")
    p += header
    for _ in range(count):
        t = _u32(c, p); p += 4
        if t in _DA_WORD:
            p += 4
        elif t in _DA_TEXT:
            p += 4 + _u32(c, p)
        elif t in _DA_TREE:
            p = _skip_data_array(c, p, header, depth + 1)
        elif t == 20:
            p += 4 + abs(struct.unpack_from('>i', c, p)[0])
        else:
            raise CreatorTextureError(f"unknown property node type {t}")
        if p > len(c):
            raise CreatorTextureError("property tree runs past the object")
    return p


def _object_header_end(c):
    """Offset past the Hmx::Object header: revision, type, has-tree flag, the property
    tree if present (malerings_chief.mat carries one), then the note."""
    _t, p = _sym_at(c, 8)
    has_tree = c[p]; p += 1
    if not has_tree:
        return _sym_at(c, p)[1]
    for header in (8, 6):
        try:
            q = _skip_data_array(c, p, header)
            note, q = _sym_at(c, q)
            if all(32 <= ord(ch) < 127 for ch in note):
                return q
        except (CreatorTextureError, struct.error, IndexError):
            pass
    raise CreatorTextureError("couldn't step over the object's property tree")


def parse_rb2_mat(data, name):
    """RndMat revision 47 (see rb2_exporter.write_rb2_rnd_mat for the field order) ->
    the slot dict the directory properties would give, plus the material colour."""
    rev = _u32(data, 0)
    if rev != MAT_REVISION_RB2:
        raise CreatorTextureError(f"'{name}' is material revision {rev}; RB2 is 47")
    p = _object_header_end(data)
    blend = struct.unpack_from('>i', data, p)[0]
    color = struct.unpack_from('>4f', data, p + 4)
    p += 4 + 16 + 1 + 1 + 4 + 1 + 4 + 1 + 4 + 4 + 48     # blend .. texture transform
    diffuse, p = _sym_at(data, p)
    _next_pass, p = _sym_at(data, p)
    p += 1 + 1 + 4                                        # intensify, cull, emissive mult
    spec_rgb = struct.unpack_from('>3f', data, p); p += 12
    spec_power = struct.unpack_from('>f', data, p)[0]; p += 4
    normal, p = _sym_at(data, p)
    _emissive, p = _sym_at(data, p)
    specular, p = _sym_at(data, p)
    return dict(target=name, diffuse_tex=diffuse, normal_map=normal, specular_map=specular,
                specular_rgb=tuple(spec_rgb), specular_power=spec_power,
                mat_color=tuple(color), blend=blend, from_mat=True)


def parse_palette(data, name):
    rev = _u32(data, 0)
    if rev != PALETTE_REVISION:
        raise CreatorTextureError(f"'{name}' is ColorPalette revision {rev}")
    n = _u32(data, OBJECT_HEADER_END)
    if OBJECT_HEADER_END + 4 + 16 * n != len(data):
        raise CreatorTextureError(f"'{name}' size doesn't match its {n} colours")
    return [list(struct.unpack_from('>3f', data, OBJECT_HEADER_END + 4 + 16 * i))
            for i in range(n)]


def walk_milo(body):
    """Every object in a revision-25 milo, nested directories included, as
    (dir path, type, name, start, end) with end at the object's end marker.

    A directory is its entry table, a metadata block ending in a marker, then one
    marker-terminated chunk per entry. An entry that is itself a directory (head_2's
    'visemes' CharClipSet and 'head_skin_proxy' RndDir) is followed by its contents as an
    inline directory - the same layout again, recognised by a revision of 25 and a type and
    name that repeat the entry's. Verified to land exactly on the end of every milo used
    here, including head_2 (90 top-level entries, 249 objects in all)."""
    def table(p):
        _rev = _u32(body, p)
        dtype, p = _sym_at(body, p + 4)
        dname, p = _sym_at(body, p)
        p += 8
        count = _u32(body, p); p += 4
        if count > 65535:
            raise CreatorTextureError(f"implausible entry count {count}")
        ents = []
        for _ in range(count):
            t, p = _sym_at(body, p)
            n, p = _sym_at(body, p)
            ents.append((t, n))
        return dtype, dname, ents, p

    def inline_dir(p, etype, ename):
        try:
            if p + 8 > len(body) or _u32(body, p) != 25:
                return False
            t, q = _sym_at(body, p + 4)
            n, _q = _sym_at(body, q)
            return t == etype and n == ename
        except (CreatorTextureError, struct.error):
            return False

    out, metas = [], []

    def walk(p, path, depth):
        if depth > 16:
            raise CreatorTextureError("directories nested implausibly deep")
        _dtype, dname, ents, q = table(p)
        j = body.find(MILO_END_MARKER, q)
        if j < 0:
            raise CreatorTextureError(f"directory '{dname}' has no metadata end marker")
        metas.append((path + dname + '/', body[q:j]))
        p = j + 4
        for etype, ename in ents:
            if inline_dir(p, etype, ename):
                p = walk(p, path + dname + '/', depth + 1)
                continue
            j = body.find(MILO_END_MARKER, p)
            if j < 0:
                raise CreatorTextureError(f"'{ename}' has no end marker")
            out.append((path + dname + '/', etype, ename, p, j))
            p = j + 4
            if inline_dir(p, etype, ename):
                p = walk(p, path + dname + '/', depth + 1)
        return p

    walk(0, '', 0)
    return out, metas


def _read_milo(filepath):
    """(top dir name, objects, body, top-level metadata). Objects are dicts with 'path',
    'type', 'name', 'data' and 'top' (True for the top-level directory's own entries)."""
    with open(filepath, 'rb') as f:
        body = read_milo_container_body(f.read())
    objs, metas = walk_milo(body)
    top_path, meta = metas[0]
    dir_name = top_path.strip('/')
    return dir_name, [dict(path=d, type=t, name=n, data=body[a:z], top=(d == top_path))
                      for d, t, n, a, z in objs], body, meta


def read_palette_milo(filepath):
    """{palette name: [[r, g, b] sRGB, ...]} from a ColorPalette milo."""
    _dn, objs, _body, _meta = _read_milo(filepath)
    pals = {}
    for o in objs:
        if o['type'] == 'ColorPalette' and o['name'] not in pals:
            try:
                pals[o['name']] = parse_palette(o['data'], o['name'])
            except (CreatorTextureError, struct.error) as ex:
                _log(f"  palette '{o['name']}' skipped: {ex}")
    if not pals:
        raise CreatorTextureError("no ColorPalette entries in this milo")
    return pals


# ---------------------------------------------------------------------------------------
# Makeup
# ---------------------------------------------------------------------------------------
#
# Eye and lip makeup (head_1_lips_1, head_5_eyes_5 ...) is a milo holding one texture -
# makeup_lips.tex: 128x128 DXT5, the colour in RGB and the shape in alpha. The head's skin
# proxy places it: its ScreenMask makeup_lips.mask draws makeup_lips.mat into the head
# texture over a rectangle - x, y (from the top), width, height in 0..1 - that each head
# sets for itself (head_2: 0.389 0.641 0.240 0.215, head_7: 0.381 0.629 0.209 0.172);
# composited there, head_1_lips_1 lands on the mouth of both. The proxy's draw order puts
# makeup after the skin and its untinted overlay pass.
#
# Each region is drawn twice: makeup_*.mat alpha-blended at its colour's 0.7, then
# makeup_*_overlay.mat at 0.5. The proxy's properties name the overlay materials
# tattoo_overlay_mat / tattoo_overlay_mat2, and skin_tattoo_overlay.pal runs parallel to
# skin.pal, so the second pass is taken to be the makeup tinted by the skin tone's
# skin_tattoo_overlay colour - a reading of the names, not something the files spell out.

MAKEUP_MASKS = {'makeup_lips.mask': 'lips', 'makeup_eyes.mask': 'eyes'}
MAKEUP_KINDS = ('lips', 'eyes')
MAKEUP_ALPHA = 0.7
MAKEUP_OVERLAY_ALPHA = 0.5
OVERLAY_PALETTE = 'skin_tattoo_overlay.pal'


def parse_screenmask_rect(data):
    """(x, y, w, h) of a ScreenMask: its material's name, then a colour, then the rect."""
    i = data.find(b'.mat')
    if i < 0:
        return None
    start = i + 4
    for k in range(max(0, i - 128), i):
        if _u32(data, k) == start - (k + 4) and start - (k + 4) > 0:
            break
    else:
        return None
    if start + 32 > len(data):
        return None
    rect = struct.unpack_from('>4f', data, start + 16)
    if not all(-0.01 <= v <= 1.01 for v in rect) or rect[2] <= 0 or rect[3] <= 0:
        return None
    return tuple(rect)


class MiloTextures:
    """What one texture/item milo offers: its material slots, textures, cfg choices."""

    def __init__(self, dir_name, slots, texs, choices, cfgs, notes, rendered,
                 skin_source=None, makeup_rects=None):
        self.dir_name = dir_name
        self.slots = slots          # [slot dict]
        self.texs = texs            # {tex name: parsed tex}, real pixels preferred
        self.choices = choices      # [cfg colour choice]
        self.cfgs = cfgs            # [cfg names]
        self.notes = notes          # [str]
        self.rendered = rendered    # {tex names drawn at runtime somewhere in this milo}
        # The diffuse a nested skin proxy's skin.mat paints from - the real head texture
        # behind a head's runtime render target, whatever either is called.
        self.skin_source = skin_source
        # {'eyes'|'lips': (x, y from top, w, h)} - where the head's skin proxy draws makeup.
        self.makeup_rects = makeup_rects or {}


def _has_pixels(tex):
    return tex is not None and not tex.get('render_target') and tex.get('width', 0) > 0


def read_texture_milo(filepath):
    """Reads a texture or item milo into a MiloTextures.

    Slots come from the top directory's properties (texture-variant milos) and its own Mat
    entries (self-contained items); where both name the same material, the directory's
    non-empty values win and the Mat supplies its colour and blend. Textures are gathered
    from EVERY level: head_2's top-level head_head2_diffuse.tex is an empty render target
    the game paints at runtime from its nested head_skin_proxy, which holds the real 512x512
    texture under the same name - so real pixels win over a render target. A placeholder
    (dummy_*) texture name is blanked but remembered in the slot's 'placeholders'."""
    dir_name, objs, _body, meta = _read_milo(filepath)
    texs, choices, cfgs, notes, mats, rendered = {}, [], [], [], {}, set()
    skin_source = None
    makeup_rects = {}
    furs, mat_raw = {}, {}
    for o in objs:
        etype, ename, data = o['type'], o['name'], o['data']
        try:
            if etype == 'ScreenMask' and ename in MAKEUP_MASKS:
                rect = parse_screenmask_rect(data)
                if rect:
                    makeup_rects.setdefault(MAKEUP_MASKS[ename], rect)
                continue
            if etype == 'Fur':
                from .rb2_fur import parse_fur
                furs[ename] = parse_fur(data)
                continue
            if etype == 'Tex':
                tex = parse_tex(data, ename)
                if tex.get('render_target'):
                    rendered.add(ename)
                if ename not in texs or (_has_pixels(tex) and not _has_pixels(texs[ename])):
                    texs[ename] = tex
            elif not o['top']:
                # head_N's head_skin_proxy: skin.mat holds the real head diffuse that the
                # TexRenderer paints into the head's render target. head_2 names both
                # head_head2_diffuse.tex; head_7's target is doe_diffuse.tex while the
                # proxy's texture is head_head7_diffuse.tex - so follow skin.mat.
                if etype == 'Mat' and ename == 'skin.mat' and skin_source is None:
                    skin_source = parse_rb2_mat(data, ename).get('diffuse_tex') or None
                continue
            elif etype == 'Mat':
                mats[ename] = parse_rb2_mat(data, ename)
                mat_raw[ename] = data
            elif etype == 'OutfitConfig':
                entries_cfg, warn = parse_outfit_config(data)
                cfgs.append(ename)
                for c in entries_cfg:
                    c['cfg'] = ename
                choices.extend(entries_cfg)
                if warn:
                    notes.append(f"{ename}: {warn}")
        except (TextureImportError, CreatorTextureError, struct.error, IndexError) as ex:
            notes.append(f"skipped {ename}: {ex}")
    # A material that names a Fur object (torso_furbikini_fur.mat -> Fur.fur) draws fur.
    for mname, raw in mat_raw.items():
        for fname, fur in furs.items():
            if fname.encode('latin-1') in raw:
                mats[mname]['fur'] = dict(fur, name=fname)
    slots = []
    for sl in parse_dir_texture_slots(meta):
        m = mats.pop(sl['target'], None)
        if m is not None:
            merged = dict(m)
            merged.update({k: v for k, v in sl.items() if v not in ('', None)})
            sl = merged
        slots.append(sl)
    slots.extend(mats.values())
    for sl in slots:
        sl['placeholders'] = set()
        for k in _TEX_KEYS:
            v = sl.get(k)
            if isinstance(v, str) and v.lower().startswith('dummy_'):
                sl['placeholders'].add(k)
                sl['placeholder_names'] = (sl.get('placeholder_names', '') + ' ' + v).strip()
                sl[k] = ''
    return MiloTextures(dir_name, slots, texs, choices, cfgs, notes, rendered, skin_source,
                        makeup_rects)


# ---------------------------------------------------------------------------------------
# Palettes: cached on the scene so a saved .blend keeps working without the milo
# ---------------------------------------------------------------------------------------

_PALETTES = {'key': None, 'data': {}, 'names': []}
_ENUM_CACHE = {}
_PREVIEWS = None


def _scene(context=None):
    ctx = context or bpy.context
    return getattr(ctx, 'scene', None) or bpy.context.scene


def _palettes(context=None):
    scn = _scene(context)
    raw = scn.get("rb2cc_palettes", "") if scn is not None else ""
    if raw != _PALETTES['key']:
        try:
            data = json.loads(raw) if raw else {}
        except ValueError:
            data = {}
        _PALETTES.update(key=raw, data=data, names=sorted(data))
        _clear_enum_cache()
    return _PALETTES['data']


def _clear_enum_cache():
    _ENUM_CACHE.clear()
    if _PREVIEWS is not None:
        _PREVIEWS.clear()


def store_palettes(scene, pals, source):
    scene["rb2cc_palettes"] = json.dumps(pals)
    scene["rb2cc_palettes_source"] = source
    _palettes()


def find_palette_milo(start_path, max_up=4, max_dirs=3000):
    """Looks for colorpalettes.milo_* beside the texture milo, then in each parent folder
    and a few levels below it - RB2 extractions keep palettes in a shared folder."""
    d = os.path.dirname(os.path.abspath(start_path))
    seen = 0
    for _ in range(max_up + 1):
        for nm in PALETTE_MILO_NAMES:
            if os.path.isfile(os.path.join(d, nm)):
                return os.path.join(d, nm)
        for root, dirs, files in os.walk(d):
            depth = root[len(d):].count(os.sep)
            if depth >= 3:
                dirs[:] = []
            seen += 1
            if seen > max_dirs:
                return None
            for nm in PALETTE_MILO_NAMES:
                if nm in files:
                    return os.path.join(root, nm)
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def ensure_palettes(context, texture_milo_path=None):
    """Loads palettes if the scene has none. Returns (ok, message)."""
    scn = context.scene
    if _palettes(context):
        return True, ""
    path = bpy.path.abspath(scn.rb2cc.palette_milo) if scn.rb2cc.palette_milo else ""
    if not (path and os.path.isfile(path)) and texture_milo_path:
        path = find_palette_milo(texture_milo_path) or ""
        if path:
            scn.rb2cc["palette_milo"] = path   # set without re-triggering the update
    if not (path and os.path.isfile(path)):
        return False, ("Color palettes milo not found - set 'Color Palettes Milo' in the "
                       "panel (colorpalettes.milo_xbox)")
    try:
        store_palettes(scn, read_palette_milo(path), path)
    except Exception as e:
        return False, f"Couldn't read palettes from {os.path.basename(path)}: {e}"
    _log(f"RB2 creator: loaded {len(_palettes(context))} palette(s) from {path}")
    return True, ""


def _srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _palette_color(context, palette, index):
    cols = _palettes(context).get(palette)
    if not cols:
        return None
    return cols[max(0, min(index, len(cols) - 1))]


# ---------------------------------------------------------------------------------------
# Skin and eyes: character-wide colours, and finding skin textures on disk
# ---------------------------------------------------------------------------------------
#
# Skin is composited at runtime. head_2's head_skin_proxy shows how: skin.mask draws
# skin.mat (the head diffuse) full frame in a colour set at runtime, then overlay.mask draws
# the same diffuse again, untinted and alpha-blended - so the result is
#
#     mix(diffuse x skin tone, diffuse, diffuse alpha)
#
# which is the two-colour group with the diffuse alpha as the keep mask (it's white over
# the teeth / gums / tongue atlas). The eyes are the same with eyes.pal: a grey diffuse
# whose alpha keeps the sclera white and lets the iris take the colour. Neither colour is
# in any .cfg - heads.cfg only drives the eyebrows - so both are one choice for the whole
# character, stored on the scene.
#
# Body skin materials ship a placeholder diffuse (naked.milo: torso_naked.mat ->
# dummy_hands.tex; malebarefeet_fleshy: feet_skin.mat -> dummy_feet.tex) because the real
# one lives elsewhere. Candidate names follow the retail naming (head_head2_norm ->
# head_head2_diffuse, legs_naked_norm -> legs_naked_diffuse) and are looked up in this
# milo first, then in milos near it on disk.

KIND_ITEMS = (
    ('ITEM', "Item Colours", "Colours come from the item's .cfg"),
    ('SKIN', "Skin", "Tinted by the character's skin tone (skin.pal)"),
    ('EYES', "Eyes", "Tinted by the character's eye colour (eyes.pal)"),
)
CHARACTER_PALETTES = {'SKIN': 'skin.pal', 'EYES': 'eyes.pal'}
_SKIN_HINT_RE = re.compile(r'naked|skin|head|feet|foot|hand|legs|body|torso_nude',
                           re.IGNORECASE)
_EYE_RE = re.compile(r'(^|_)eyes?(\.|_)', re.IGNORECASE)
_ROLE_SUFFIX = {'diffuse_tex': '_diffuse', 'normal_map': '_norm', 'specular_map': '_spec'}
# Skin parts painted into another part's texture. Bare feet are UV-mapped into the lower-
# body skin: the soles, tops and heels sit in legs_naked_norm's right-hand column exactly
# where malebarefeet_fleshy_norm has them, and legs_naked_diffuse is plain skin there.
# The feet bring their own normal and spec maps (the 'fleshy' detail) and borrow only the
# diffuse. (The hands need no entry: their material is torso_naked.mat itself, so the
# torso's names are found from the material name.)
#
# Pants and jackets carry the bare skin they leave showing as their own meshes
# (wreckedjeans_skin.5.mesh -> legs_skin.mat, bulkyjacket_skin.3.mesh -> torso_naked.mat),
# painted from the same body atlases: legs_naked_* and torso_naked_* in each gender's
# char/<gender>/outfits/gen/static_*_textures milo (the torso diffuse in
# torso/naked/gen/torso_skin_subdir).
SHARED_SKIN_ATLAS = (
    (re.compile(r'feet|foot|toe', re.IGNORECASE), 'legs_naked'),
    (re.compile(r'legs?(_|\.|$)', re.IGNORECASE), 'legs_naked'),
    (re.compile(r'torso', re.IGNORECASE), 'torso_naked'),
)
_TEX_INDEX = {}           # milo path -> (mtime, size, {tex name with pixels: (start, end)})
_SEARCH_CACHE = {}
MAX_SEARCH_FILES = 2500


_SKIN_MAT_RE = re.compile(r'(^|_)(skin|naked)(_|\.|$)', re.IGNORECASE)


def skin_slot_for(material_name):
    """A slot for a skin material an outfit doesn't define itself (legs_skin.mat,
    torso_naked.mat on a jacket's skin mesh) - its textures are found by name - or None
    when the name doesn't look like skin."""
    if not _SKIN_MAT_RE.search(material_name):
        return None
    return dict(target=material_name, diffuse_tex='', normal_map='', specular_map='',
                placeholders={'diffuse_tex'}, placeholder_names='')


def classify_slot(slot, choices, rendered):
    """'EYES', 'SKIN' or 'ITEM' for one material slot."""
    target = slot['target']
    if any(target in c['mats'] for c in choices):
        return 'ITEM'
    diffuse = slot.get('diffuse_tex') or ''
    if _EYE_RE.search(target) and diffuse and 'cube' not in diffuse.lower():
        return 'EYES'
    if slot.get('blend') in (BLEND_ADD, BLEND_SRC_ALPHA, BLEND_PREMULT) \
            or 'patch' in target.lower():
        return 'ITEM'          # overlays (face-paint patch, makeup), never the skin itself
    if _SKIN_MAT_RE.search(target):
        # A skin material even when it names a real texture: hotpants' legs_override_skin
        # .mat paints the legs from hotpants_camo_legs_diff.tex - the legs skin with fishnet
        # stockings baked in, laid out like legs_naked_diffuse - and is tinted by the skin
        # tone like any other skin.
        return 'SKIN'
    runtime = 'diffuse_tex' in slot.get('placeholders', ()) or diffuse in rendered
    names = target + ' ' + diffuse + ' ' + (slot.get('normal_map') or '')
    if runtime and _SKIN_HINT_RE.search(names):
        return 'SKIN'
    return 'ITEM'


def candidate_names(slot, role):
    """Names a skin texture for `role` is likely stored under, best first."""
    out = []
    cur = slot.get(role)
    if isinstance(cur, str) and cur:
        out.append(cur)
    want = _ROLE_SUFFIX[role]
    for other, suf in _ROLE_SUFFIX.items():
        v = slot.get(other)
        if other != role and isinstance(v, str) and v.lower().endswith(suf + '.tex'):
            out.append(v[:-len(suf + '.tex')] + want + '.tex')
    stem = slot['target'][:-4] if slot['target'].lower().endswith('.mat') else slot['target']
    out.append(stem + want + '.tex')
    names = ' '.join(str(slot.get(k) or '') for k in ('target',) + _TEX_KEYS) + ' ' + \
        slot.get('placeholder_names', '')                # e.g. dummy_feet.tex
    for pattern, base in SHARED_SKIN_ATLAS:
        if pattern.search(names):
            out.append(base + want + '.tex')
    seen, uniq = set(), []
    for n in out:
        if n.lower() not in seen:
            seen.add(n.lower())
            uniq.append(n)
    return uniq


def _milo_files_near(start_path, extra_dir, max_up=3, depth=3):
    """Milo files to search: the chosen folder (deep), then the loaded milo's folder and
    its parents (shallow), each file once."""
    seen = set()
    roots = []
    if extra_dir and os.path.isdir(extra_dir):
        roots.append((extra_dir, 6))
    d = os.path.dirname(os.path.abspath(start_path))
    for _ in range(max_up + 1):
        roots.append((d, depth))
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    for root, max_depth in roots:
        for r, dirs, files in os.walk(root):
            if r[len(root):].count(os.sep) >= max_depth:
                dirs[:] = []
            for f in sorted(files):
                low = f.lower()
                if '.milo' not in low:
                    continue
                full = os.path.join(r, f)
                if full in seen:
                    continue
                seen.add(full)
                yield full
                if len(seen) >= MAX_SEARCH_FILES:
                    return


def _textures_in_file(path, names):
    """{name: parsed tex} for those of `names` this milo holds with real pixels. Only the
    Tex names each file holds are cached (by mtime and size), so a repeat search skips
    files that can't help without reading them."""
    try:
        st = os.stat(path)
    except OSError:
        return {}
    cached = _TEX_INDEX.get(path)
    if cached is not None and cached[0] == st.st_mtime and cached[1] == st.st_size \
            and not any(n in cached[2] for n in names):
        return {}
    try:
        with open(path, 'rb') as f:
            data = f.read()
    except OSError:
        return {}
    wanted = [n.encode('latin-1') for n in names]
    if len(data) >= 4 and struct.unpack_from('<I', data, 0)[0] == 0xCABEDEAF \
            and not any(w in data for w in wanted):
        return {}                         # uncompressed and doesn't mention them
    try:
        body = read_milo_container_body(data)
        if not any(w in body for w in wanted):
            return {}
        objs, _metas = walk_milo(body)
    except Exception:
        return {}
    _TEX_INDEX[path] = (st.st_mtime, st.st_size, {n for _d, t, n, _a, _z in objs if t == 'Tex'})
    found = {}
    for _d, t, n, a, z in objs:
        if t != 'Tex' or n not in names or n in found:
            continue
        try:
            tex = parse_tex(body[a:z], n)
        except (TextureImportError, struct.error, IndexError):
            continue
        if _has_pixels(tex):
            found[n] = tex
    return found


def find_textures_on_disk(names, start_path, extra_dir=""):
    """(name, tex, milo path) for the first of `names` found in a milo near start_path."""
    key = (tuple(names), os.path.dirname(os.path.abspath(start_path)), extra_dir)
    if key in _SEARCH_CACHE:
        return _SEARCH_CACHE[key]
    result = None
    start = os.path.abspath(start_path)
    for path in _milo_files_near(start_path, extra_dir):
        if os.path.abspath(path) == start:
            continue
        found = _textures_in_file(path, names)
        for n in names:
            if n in found:
                result = (n, found[n], path)
                break
        if result:
            break
    if result is not None:
        _SEARCH_CACHE[key] = result      # misses aren't cached: the user may add files
    return result


# ---------------------------------------------------------------------------------------
# Node setup
# ---------------------------------------------------------------------------------------

def _sock(sockets, identifier, fallback_index):
    for s in sockets:
        if s.identifier == identifier:
            return s
    return sockets[fallback_index]


def _mix(nodes, blend, x, y):
    n = nodes.new('ShaderNodeMix')
    n.data_type = 'RGBA'
    n.blend_type = blend
    n.location = (x, y)
    return n


def ensure_two_color_group():
    """mix(diffuse * mix(colour1, colour2, alpha * use2), diffuse, keep mask)."""
    ng = bpy.data.node_groups.get(GROUP_NAME)
    if ng is not None and ng.get("rb2cc_version") == GROUP_VERSION:
        return ng
    if ng is None:
        ng = bpy.data.node_groups.new(GROUP_NAME, 'ShaderNodeTree')
    ng.interface.clear()
    ng.nodes.clear()

    def sock(name, kind, default, in_out='INPUT'):
        it = ng.interface.new_socket(name=name, in_out=in_out, socket_type=kind)
        if default is not None:
            it.default_value = default
        if kind == 'NodeSocketFloat':
            it.min_value, it.max_value = 0.0, 1.0
        return it

    sock("Diffuse", 'NodeSocketColor', (0.8, 0.8, 0.8, 1.0))
    sock("Diffuse Alpha", 'NodeSocketFloat', 0.0)
    sock("Keep Mask", 'NodeSocketFloat', 0.0)
    sock("Color 1", 'NodeSocketColor', (0.5, 0.5, 0.5, 1.0))
    sock("Color 2", 'NodeSocketColor', (0.5, 0.5, 0.5, 1.0))
    sock("Use Color 2", 'NodeSocketFloat', 0.0)
    sock("Color", 'NodeSocketColor', None, 'OUTPUT')

    nodes, links = ng.nodes, ng.links
    gin = nodes.new('NodeGroupInput'); gin.location = (-700, 0)
    gout = nodes.new('NodeGroupOutput'); gout.location = (500, 0)

    sel = nodes.new('ShaderNodeMath'); sel.operation = 'MULTIPLY'; sel.location = (-450, 150)
    sel.use_clamp = True
    links.new(gin.outputs["Diffuse Alpha"], sel.inputs[0])
    links.new(gin.outputs["Use Color 2"], sel.inputs[1])

    pick = _mix(nodes, 'MIX', -200, 150)
    links.new(sel.outputs[0], _sock(pick.inputs, 'Factor_Float', 0))
    links.new(gin.outputs["Color 1"], _sock(pick.inputs, 'A_Color', 6))
    links.new(gin.outputs["Color 2"], _sock(pick.inputs, 'B_Color', 7))

    tint = _mix(nodes, 'MULTIPLY', 50, 50)
    _sock(tint.inputs, 'Factor_Float', 0).default_value = 1.0
    links.new(gin.outputs["Diffuse"], _sock(tint.inputs, 'A_Color', 6))
    links.new(_sock(pick.outputs, 'Result_Color', 2), _sock(tint.inputs, 'B_Color', 7))

    keep = _mix(nodes, 'MIX', 280, 0)
    links.new(gin.outputs["Keep Mask"], _sock(keep.inputs, 'Factor_Float', 0))
    links.new(_sock(tint.outputs, 'Result_Color', 2), _sock(keep.inputs, 'A_Color', 6))
    links.new(gin.outputs["Diffuse"], _sock(keep.inputs, 'B_Color', 7))
    links.new(_sock(keep.outputs, 'Result_Color', 2), gout.inputs["Color"])
    ng["rb2cc_version"] = GROUP_VERSION
    return ng


def _input(node, *names):
    for n in names:
        s = node.inputs.get(n)
        if s is not None:
            return s
    return None


def alpha_keeps(pg):
    """Whether the diffuse alpha is the keep mask (skin, eyes, one-colour items) rather
    than the colour-2 selector (two-colour items)."""
    if pg.id_data.get("rb2_fur"):
        return False             # fur: the diffuse alpha is the strand mask (coverage)
    return pg.kind in CHARACTER_PALETTES or not pg.uses2


def wire_diffuse_alpha(mat, keep):
    """Routes the diffuse alpha to the keep mask (skin, eyes) or to the colour-2 selector
    (clothing). An item with its own keep-mask texture keeps that texture."""
    nt = mat.node_tree
    grp = nt.nodes.get(GROUP_NODE_NAME) if nt else None
    diff = nt.nodes.get("RB2CC_Diffuse") if nt else None
    if grp is None or diff is None:
        return
    for sock_name in ("Keep Mask", "Diffuse Alpha"):
        for link in list(grp.inputs[sock_name].links):
            if link.from_node is diff:
                nt.links.remove(link)
    if keep:
        if not grp.inputs["Keep Mask"].is_linked:
            nt.links.new(diff.outputs['Alpha'], grp.inputs["Keep Mask"])
    else:
        nt.links.new(diff.outputs['Alpha'], grp.inputs["Diffuse Alpha"])


def build_creator_material(mat, slot, images, keep_alpha=False, milo_shader=False):
    """Replaces the material's nodes. images: {'diffuse', 'mask', 'normal', 'specular'}.

    Base colour always goes through the two-colour group so one path covers every item:
    a texture-variant clothing item (diffuse + alpha + keep mask), a self-contained item
    tinted through its material colour (diffuse x colour), and an item with no diffuse at
    all (the colour itself - the group's Diffuse input defaults to white)."""
    try:
        mat.use_nodes = True
    except AttributeError:
        pass
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    nodes.clear()

    out = nodes.new('ShaderNodeOutputMaterial'); out.location = (700, 300)
    bsdf = nodes.new('ShaderNodeBsdfPrincipled'); bsdf.location = (350, 300)
    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])

    power = float(slot.get('specular_power') or 10.0)
    for name, value in (('Metallic', 0.0), ('IOR', 1.5),
                        ('Roughness', max(0.05, min(1.0, (2.0 / (power + 2.0)) ** 0.5)))):
        s = _input(bsdf, name)
        if s is not None:
            s.default_value = value

    def tex_node(img, y, label):
        n = nodes.new('ShaderNodeTexImage')
        n.image = img
        n.label = label
        n.location = (-650, y)
        return n

    grp = nodes.new('ShaderNodeGroup')
    grp.node_tree = ensure_two_color_group()
    grp.name = GROUP_NODE_NAME
    grp.label = "RB2 Two-Color"
    grp.location = (0, 450)
    grp.inputs["Diffuse"].default_value = (1.0, 1.0, 1.0, 1.0)
    mat_color = slot.get('mat_color')
    if mat_color:
        base_rgb = [_srgb_to_linear(max(0.0, min(1.0, c))) for c in mat_color[:3]]
    else:
        base_rgb = [1.0, 1.0, 1.0]
    grp.inputs["Color 1"].default_value = (*base_rgb, 1.0)
    mat["rb2cc_base_color"] = base_rgb       # restored when no palette drives colour 1
    base = _input(bsdf, 'Base Color')
    if base is not None:
        links.new(grp.outputs["Color"], base)

    diff = images.get('diffuse')
    d = None
    if images.get('mask') is not None:
        m = tex_node(images['mask'], 250, "Two-Color Keep Mask")
        links.new(m.outputs['Color'], grp.inputs["Keep Mask"])
    if diff is not None:
        d = tex_node(diff, 550, "Diffuse")
        d.name = "RB2CC_Diffuse"
        links.new(d.outputs['Color'], grp.inputs["Diffuse"])
        wire_diffuse_alpha(mat, keep_alpha)

    # Transparency: only the alpha-blended materials (the goggles' lenses). Their alpha is
    # the material colour's, times the diffuse alpha when there's a diffuse and it isn't
    # already serving as the colour-2 mask.
    blend = slot.get('blend')
    alpha_sock = _input(bsdf, 'Alpha')
    if blend in (BLEND_SRC_ALPHA, BLEND_PREMULT) and alpha_sock is not None:
        alpha = float(mat_color[3]) if mat_color else 1.0
        if d is not None:
            mul = nodes.new('ShaderNodeMath'); mul.operation = 'MULTIPLY'
            mul.location = (0, 100)
            mul.inputs[1].default_value = alpha
            links.new(d.outputs['Alpha'], mul.inputs[0])
            links.new(mul.outputs[0], alpha_sock)
            mat["rb2cc_alpha_from_diffuse"] = True
        else:
            alpha_sock.default_value = alpha
        for attr, value in (('surface_render_method', 'BLENDED'), ('blend_method', 'BLEND')):
            try:
                setattr(mat, attr, value)
                break
            except (AttributeError, TypeError):
                continue

    if images.get('normal') is not None:
        nm = tex_node(images['normal'], -50, "Normal")
        nmap = nodes.new('ShaderNodeNormalMap'); nmap.location = (0, -50)
        nmap.space = 'TANGENT'
        links.new(nm.outputs['Color'], nmap.inputs['Color'])
        s = _input(bsdf, 'Normal')
        if s is not None:
            links.new(nmap.outputs['Normal'], s)

    tint_sock = _input(bsdf, 'Specular Tint')
    spec = slot.get('specular_rgb')
    if images.get('specular') is not None:
        sp = tex_node(images['specular'], -350, "Specular")
        if tint_sock is not None:
            links.new(sp.outputs['Color'], tint_sock)
    elif tint_sock is not None and spec is not None:
        if isinstance(spec, int):
            spec = [((spec >> sh) & 0xFF) / 255.0 for sh in (16, 8, 0)]
        tint_sock.default_value = (*[_srgb_to_linear(max(0.0, min(1.0, c)))
                                     for c in spec[:3]], 1.0)
    if blend == BLEND_ADD:
        # Additive pass (eyespec.mat: the eyes' reflection): light added on top, never
        # an opaque surface hiding what's behind it.
        trans = nodes.new('ShaderNodeBsdfTransparent'); trans.location = (350, 650)
        emit = nodes.new('ShaderNodeEmission'); emit.location = (350, 550)
        add = nodes.new('ShaderNodeAddShader'); add.location = (550, 600)
        links.new(grp.outputs["Color"], emit.inputs['Color'])
        links.new(trans.outputs[0], add.inputs[0])
        links.new(emit.outputs[0], add.inputs[1])
        links.new(add.outputs[0], out.inputs['Surface'])
        for attr, value in (('surface_render_method', 'BLENDED'), ('blend_method', 'BLEND')):
            try:
                setattr(mat, attr, value)
                break
            except (AttributeError, TypeError):
                continue
    mat["milo_textured"] = True

    # What the Milo shader needs that the Principled tree doesn't keep.
    spec_rgb = slot.get('specular_rgb')
    if isinstance(spec_rgb, int):
        spec_rgb = [((spec_rgb >> sh) & 0xFF) / 255.0 for sh in (16, 8, 0)]
    mat[MILO_SPEC_KEY] = [float(c) for c in (spec_rgb or (0.0, 0.0, 0.0))[:3]]
    mat[MILO_POWER_KEY] = float(slot.get('specular_power') or 10.0)
    mat[MILO_BLEND_KEY] = int(blend) if blend is not None else 1
    set_milo_shader(mat, milo_shader)


MILO_SHADER_KEY = "rb2_milo_shader"
MILO_SPEC_KEY = "rb2_milo_specular_rgb"
MILO_POWER_KEY = "rb2_milo_specular_power"
MILO_BLEND_KEY = "rb2_milo_blend"


def _principled(nt):
    return next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)


def _material_output(nt):
    outs = [n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL']
    return next((n for n in outs if getattr(n, 'is_active_output', False)), None) \
        or (outs[0] if outs else None)


def set_milo_shader(mat, enabled):
    """Switches a dressed material between the plain Principled BSDF and the Milo
    Character Shader. Either way the creator nodes are untouched; only what reaches the
    Material Output changes."""
    nt = mat.node_tree
    if nt is None:
        return
    from .texture_importer import SHADER_NODE_NAME
    mat[MILO_SHADER_KEY] = bool(enabled)
    if enabled:
        sync_milo_shader(mat)
        return
    grp = nt.nodes.get(SHADER_NODE_NAME)
    if grp is not None:
        nt.nodes.remove(grp)
    out, bsdf = _material_output(nt), _principled(nt)
    if out is None or bsdf is None:
        return
    add = next((n for n in nt.nodes if n.type == 'ADD_SHADER'), None)
    if mat.get(MILO_BLEND_KEY) == BLEND_ADD and add is not None:
        nt.links.new(add.outputs[0], out.inputs['Surface'])
    else:
        nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    bsdf.label = ""


def sync_milo_shader(mat):
    """Feeds the Milo Character Shader from whatever reaches the Principled BSDF - the
    two-colour result (with makeup / fur tint / a baked image), its alpha, the normal map
    and the specular map - and connects it to the Material Output. No-op unless the
    material uses the Milo shader."""
    if not mat.get(MILO_SHADER_KEY) or mat.node_tree is None:
        return
    from .texture_importer import (ensure_shader_group, SHADER_NODE_NAME, SHADER_GROUP_NAME,
                                   _lin3)
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    bsdf, out = _principled(nt), _material_output(nt)
    if bsdf is None or out is None:
        return
    grp = nodes.get(SHADER_NODE_NAME)
    if grp is None or grp.node_tree is None:
        if grp is not None:
            nodes.remove(grp)
        grp = nodes.new('ShaderNodeGroup')
        grp.name = SHADER_NODE_NAME
    grp.node_tree = ensure_shader_group()
    grp.label = SHADER_GROUP_NAME
    grp.width = 260
    grp.location = (bsdf.location.x, bsdf.location.y - 750)
    gi = grp.inputs
    for sock in gi:
        for l in list(sock.links):
            links.remove(l)

    def feed(src_sock, dst):
        node, outp = _source(src_sock)
        if node is not None:
            links.new(outp, dst)
            return True
        return False

    # colour
    base = bsdf.inputs.get('Base Color')
    if not feed(base, gi["Diffuse"]) and base is not None:
        gi["Diffuse"].default_value = tuple(base.default_value)
    gi["Material Color"].default_value = (1.0, 1.0, 1.0, 1.0)  # already in the colour

    # alpha: only where the Principled tree uses it as coverage (blended items, fur)
    alpha = bsdf.inputs.get('Alpha')
    blend = mat.get(MILO_BLEND_KEY, 1)
    if alpha is not None and feed(alpha, gi["Diffuse Alpha"]):
        gi["Use Alpha"].default_value = 1.0
        gi["Material Alpha"].default_value = 1.0
    elif alpha is not None and alpha.default_value < 1.0:
        gi["Use Alpha"].default_value = 1.0
        gi["Material Alpha"].default_value = float(alpha.default_value)
    else:
        gi["Use Alpha"].default_value = 0.0
    gi["Additive"].default_value = 1.0 if blend == BLEND_ADD else 0.0

    # normal map: the colour going into the Principled's Normal Map node
    nnode, _o = _source(bsdf.inputs.get('Normal'))
    if nnode is not None and nnode.type == 'NORMAL_MAP':
        feed(nnode.inputs.get('Color'), gi["Normal Map"])
    gi["Detail Strength"].default_value = 0.0

    # specular: the map's colour as the mask, its alpha as gloss
    snode, _o = _source(bsdf.inputs.get('Specular Tint'))
    if snode is not None and getattr(snode, 'image', None) is not None:
        links.new(snode.outputs['Color'], gi["Specular Map"])
        links.new(snode.outputs['Alpha'], gi["Gloss Map"])
        gi["Gloss Map Influence"].default_value = 1.0
    else:
        gi["Specular Map"].default_value = (1.0, 1.0, 1.0, 1.0)
        gi["Gloss Map Influence"].default_value = 0.0
    spec = mat.get(MILO_SPEC_KEY) or [0.0, 0.0, 0.0]
    gi["Specular Color"].default_value = (*_lin3(list(spec)), 1.0)
    gi["Specular Power"].default_value = float(mat.get(MILO_POWER_KEY, 10.0))
    gi["Specular 2 Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    gi["Hair Anisotropy"].default_value = 0.0
    gi["Rim Color"].default_value = (0.0, 0.0, 0.0, 1.0)   # RB2 materials have no rim
    gi["Emissive"].default_value = (0.0, 0.0, 0.0, 1.0)

    links.new(grp.outputs["Shader"], out.inputs['Surface'])
    bsdf.label = "RB2 colour hub (feeds the Milo shader)"


def apply_material_colors(mat, context=None):
    """Pushes the panel's colour choices into the material's two-colour node."""
    pg = mat.rb2cc
    nt = mat.node_tree
    node = nt.nodes.get(GROUP_NODE_NAME) if nt else None
    if node is None:
        return
    if pg.kind in CHARACTER_PALETTES:
        scn = _scene(context)
        idx = scn.rb2cc.skin_index if pg.kind == 'SKIN' else scn.rb2cc.eye_index
        rgb = _palette_color(context, CHARACTER_PALETTES[pg.kind], idx)
        if rgb is not None:
            node.inputs["Color 1"].default_value = (*[_srgb_to_linear(c) for c in rgb], 1.0)
        node.inputs["Use Color 2"].default_value = 0.0
        tint = nt.nodes.get("RB2CC_MakeupTint")
        if tint is not None and pg.kind == 'SKIN':
            trgb = _palette_color(context, OVERLAY_PALETTE, scn.rb2cc.skin_index)
            if trgb is not None:
                tint.outputs[0].default_value = (*[_srgb_to_linear(c) for c in trgb], 1.0)
        return
    if pg.link2:
        pg["sel_index2"] = pg.sel_index1      # one pick drives both channels
    if not pg.uses1:
        base = mat.get("rb2cc_base_color")
        if base is not None and len(base) == 3:
            node.inputs["Color 1"].default_value = (*base, 1.0)
    for ch in (1, 2):
        if not getattr(pg, f"uses{ch}"):
            continue
        rgb = _palette_color(context, getattr(pg, f"sel_palette{ch}"),
                             getattr(pg, f"sel_index{ch}"))
        if rgb is not None:
            node.inputs[f"Color {ch}"].default_value = (
                *[_srgb_to_linear(c) for c in rgb], 1.0)
    node.inputs["Use Color 2"].default_value = 1.0 if pg.uses2 else 0.0


def read_makeup_milo(filepath):
    """(kind 'eyes'|'lips' or None, texture name, parsed tex) from a makeup milo."""
    _dn, objs, _body, _meta = _read_milo(filepath)
    for o in objs:
        if o['type'] != 'Tex':
            continue
        tex = parse_tex(o['data'], o['name'])
        if not _has_pixels(tex):
            continue
        low = (o['name'] + ' ' + os.path.basename(filepath)).lower()
        kind = 'lips' if 'lips' in low else ('eyes' if 'eye' in low else None)
        return kind, o['name'], tex
    raise CreatorTextureError("no makeup texture in this milo")


def set_makeup(mat, kind, filepath, context=None):
    """Loads a makeup milo onto a head skin material's eye or lip slot (kind None: from
    the milo's names). Returns the kind used."""
    got, tex_name, tex = read_makeup_milo(filepath)
    kind = kind or got
    if kind not in MAKEUP_KINDS:
        raise CreatorTextureError("can't tell whether this is eye or lip makeup")
    stem = os.path.splitext(os.path.basename(filepath))[0]
    img = build_image(f"{stem}_{tex_name}", tex, 'diffuse', False)
    state = json.loads(mat.get("rb2_makeup", "{}"))
    state[kind] = dict(image=img.name, source=filepath)
    mat["rb2_makeup"] = json.dumps(state)
    rebuild_makeup(mat, context)
    return kind


def clear_makeup(mat, kind, context=None):
    state = json.loads(mat.get("rb2_makeup", "{}"))
    state.pop(kind, None)
    mat["rb2_makeup"] = json.dumps(state)
    rebuild_makeup(mat, context)


def rebuild_makeup(mat, context=None):
    """(Re)builds the makeup layers between the two-colour group and the BSDF's base
    colour: per region, the makeup over the skin at 0.7, then tinted by the skin tone's
    overlay colour at 0.5 - placed through the head's UVs into the proxy's rectangle."""
    nt = mat.node_tree
    if nt is None:
        return
    nodes, links = nt.nodes, nt.links
    for n in [n for n in nodes if n.name.startswith("RB2CC_Makeup")]:
        nodes.remove(n)
    grp = nodes.get(GROUP_NODE_NAME)
    bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if grp is None or bsdf is None:
        return
    base = _input(bsdf, 'Base Color')
    rects = json.loads(mat.get("rb2_makeup_rects", "{}"))
    state = json.loads(mat.get("rb2_makeup", "{}"))
    prev = grp.outputs["Color"]
    layers = [(k, state[k], rects[k]) for k in MAKEUP_KINDS
              if k in state and k in rects and bpy.data.images.get(state[k]['image'])]
    if layers:
        uv = nodes.new('ShaderNodeUVMap'); uv.name = "RB2CC_MakeupUV"
        uv.location = (-200, 900)
        tint = nodes.new('ShaderNodeRGB'); tint.name = "RB2CC_MakeupTint"
        tint.label = "Makeup overlay tint (skin_tattoo_overlay)"
        tint.location = (-200, 1100)
        tint.outputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
        x0 = 250
        for i, (kind, st, (x, y, w, h)) in enumerate(layers):
            def node(t, name, dx, dy):
                n = nodes.new(t)
                n.name = f"RB2CC_Makeup_{kind}_{name}"
                n.location = (x0 + dx + i * 700, 900 + dy)
                return n
            mp = node('ShaderNodeMapping', "Map", 0, 0)
            mp.vector_type = 'POINT'
            yb = 1.0 - y - h                         # the rect's y runs from the top
            mp.inputs['Location'].default_value = (-x / w, -yb / h, 0.0)
            mp.inputs['Scale'].default_value = (1.0 / w, 1.0 / h, 1.0)
            links.new(uv.outputs['UV'], mp.inputs['Vector'])
            tx = node('ShaderNodeTexImage', "Tex", 0, -300)
            tx.image = bpy.data.images[st['image']]
            tx.extension = 'CLIP'
            tx.label = f"Makeup ({kind})"
            links.new(mp.outputs['Vector'], tx.inputs['Vector'])

            a1 = node('ShaderNodeMath', "A1", 250, -200)
            a1.operation = 'MULTIPLY'; a1.inputs[1].default_value = MAKEUP_ALPHA
            links.new(tx.outputs['Alpha'], a1.inputs[0])
            m1 = _mix(nodes, 'MIX', 0, 0)
            m1.name = f"RB2CC_Makeup_{kind}_Mix1"
            m1.location = (x0 + 450 + i * 700, 900)
            links.new(a1.outputs[0], _sock(m1.inputs, 'Factor_Float', 0))
            links.new(prev, _sock(m1.inputs, 'A_Color', 6))
            links.new(tx.outputs['Color'], _sock(m1.inputs, 'B_Color', 7))

            tm = _mix(nodes, 'MULTIPLY', 0, 0)
            tm.name = f"RB2CC_Makeup_{kind}_Tint"
            tm.location = (x0 + 250 + i * 700, 650)
            _sock(tm.inputs, 'Factor_Float', 0).default_value = 1.0
            links.new(tx.outputs['Color'], _sock(tm.inputs, 'A_Color', 6))
            links.new(tint.outputs[0], _sock(tm.inputs, 'B_Color', 7))
            a2 = node('ShaderNodeMath', "A2", 250, -400)
            a2.operation = 'MULTIPLY'; a2.inputs[1].default_value = MAKEUP_OVERLAY_ALPHA
            links.new(tx.outputs['Alpha'], a2.inputs[0])
            m2 = _mix(nodes, 'MIX', 0, 0)
            m2.name = f"RB2CC_Makeup_{kind}_Mix2"
            m2.location = (x0 + 600 + i * 700, 900)
            links.new(a2.outputs[0], _sock(m2.inputs, 'Factor_Float', 0))
            links.new(_sock(m1.outputs, 'Result_Color', 2), _sock(m2.inputs, 'A_Color', 6))
            links.new(_sock(tm.outputs, 'Result_Color', 2), _sock(m2.inputs, 'B_Color', 7))
            prev = _sock(m2.outputs, 'Result_Color', 2)
    if base is not None:
        links.new(prev, base)
    apply_material_colors(mat, context)
    sync_milo_shader(mat)


# ---------------------------------------------------------------------------------------
# Baking the character's diffuse textures
# ---------------------------------------------------------------------------------------
#
# For exporting, each dressed material's live colouring is written into one plain texture.
# Blender's own bake only runs in Cycles and renders through the mesh's UVs (selection,
# margins, resampling); everything these materials do happens in the texture's own pixel
# space, so the result is computed straight from the source pixels instead - exact, at the
# game texture's resolution, with no render engine involved. It follows the node group:
#
#     out = mix(diffuse x mix(colour1, colour2, alpha x use2), diffuse, keep)    (linear)
#
# then the makeup layers in their rectangles (heads), then the flat fur tint (fur). The
# baked image keeps the diffuse alpha only where the material uses alpha as transparency
# (fur, alpha-cut cards); elsewhere the alpha is the keep / colour-2 mask, baked in already.

BAKED_NODE = "RB2CC_Baked"


def _pixels(img):
    """(H, W, 4) float array, rows bottom-up as Blender stores them."""
    import numpy as np
    w, h = img.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    return buf.reshape(h, w, 4)


def _to_linear(x):
    import numpy as np
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _to_srgb(x):
    import numpy as np
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def _resample(arr, h, w):
    """Nearest-neighbour resize of an (H, W, C) array to (h, w, C)."""
    import numpy as np
    ys = (np.arange(h) * arr.shape[0] // h)
    xs = (np.arange(w) * arr.shape[1] // w)
    return arr[ys][:, xs]


def _source(sock):
    """(node, output socket) linked into an input, or (None, None)."""
    if sock is None or not sock.is_linked:
        return None, None
    link = sock.links[0]
    return link.from_node, link.from_socket


def bake_material(mat):
    """Computes the material's baked diffuse. Returns the image, or None with a reason."""
    import numpy as np
    nt = mat.node_tree
    if nt is None:
        return None, "no nodes"
    grp = nt.nodes.get(GROUP_NODE_NAME)
    bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if grp is None or bsdf is None:
        return None, "not a Rock Band 2 creator material"
    dnode, _ds = _source(grp.inputs["Diffuse"])
    if dnode is None or getattr(dnode, 'image', None) is None:
        return None, "no diffuse texture (its colour is a plain value - nothing to bake)"
    dimg = dnode.image
    px = _pixels(dimg)
    h, w = px.shape[:2]
    d = _to_linear(px[..., :3].astype(np.float64))
    a_tex = px[..., 3].astype(np.float64)

    def value_or_linked(sock):
        """(H, W) array for a float input: the texture feeding it or its value."""
        node, out = _source(sock)
        if node is None:
            return np.full((h, w), float(sock.default_value))
        if node is dnode and out.name == 'Alpha':
            return a_tex
        if getattr(node, 'image', None) is not None:
            arr = _pixels(node.image)
            if arr.shape[:2] != (h, w):
                arr = _resample(arr, h, w)
            return arr[..., 0].astype(np.float64) if out.name == 'Color' \
                else arr[..., 3].astype(np.float64)
        return np.zeros((h, w))

    c1 = np.array(grp.inputs["Color 1"].default_value[:3])
    c2 = np.array(grp.inputs["Color 2"].default_value[:3])
    use2 = float(grp.inputs["Use Color 2"].default_value)
    sel = np.clip(value_or_linked(grp.inputs["Diffuse Alpha"]) * use2, 0.0, 1.0)[..., None]
    keep = np.clip(value_or_linked(grp.inputs["Keep Mask"]), 0.0, 1.0)[..., None]
    out = d * (c1 * (1 - sel) + c2 * sel)
    out = out * (1 - keep) + d * keep

    # makeup, in each layer's rectangle of the texture
    rects = json.loads(mat.get("rb2_makeup_rects", "{}"))
    state = json.loads(mat.get("rb2_makeup", "{}"))
    tint_node = nt.nodes.get("RB2CC_MakeupTint")
    tint = np.array(tint_node.outputs[0].default_value[:3]) if tint_node else np.ones(3)
    for kind in MAKEUP_KINDS:
        st, rect = state.get(kind), rects.get(kind)
        mimg = bpy.data.images.get(st['image']) if st else None
        if mimg is None or not rect:
            continue
        x, y, rw, rh = rect
        yb = 1.0 - y - rh
        mk = _pixels(mimg)
        mh, mw = mk.shape[:2]
        vs = (np.arange(h) + 0.5) / h
        us = (np.arange(w) + 0.5) / w
        mu = (us - x) / rw
        mv = (vs - yb) / rh
        cols = np.clip((mu * mw).astype(int), 0, mw - 1)
        rows = np.clip((mv * mh).astype(int), 0, mh - 1)
        inside = ((mv >= 0) & (mv < 1))[:, None] & ((mu >= 0) & (mu < 1))[None, :]
        sample = mk[rows][:, cols]
        m_rgb = _to_linear(sample[..., :3].astype(np.float64))
        m_a = np.where(inside, sample[..., 3], 0.0)[..., None]
        out = out * (1 - m_a * MAKEUP_ALPHA) + m_rgb * (m_a * MAKEUP_ALPHA)
        out = out * (1 - m_a * MAKEUP_OVERLAY_ALPHA) + (m_rgb * tint) * (m_a * MAKEUP_OVERLAY_ALPHA)

    # fur: the flat (half-way) tint - shells can't be baked into one texture
    if mat.get("rb2_fur"):
        import ast
        try:
            fur = ast.literal_eval(mat.get("rb2_fur_params", "{}"))
            mid = [(_srgb_to_linear(b) + _srgb_to_linear(e)) / 2
                   for b, e in zip(fur['base_tint'][:3], fur['end_tint'][:3])]
            out = out * np.array(mid)
        except (ValueError, SyntaxError, KeyError):
            pass

    alpha_in = bsdf.inputs.get('Alpha')
    alpha = a_tex if (alpha_in is not None and alpha_in.is_linked) else np.ones((h, w))
    rgba = np.concatenate([_to_srgb(out), alpha[..., None]], axis=2).astype(np.float32)

    name = f"{mat.name}_baked"
    old = bpy.data.images.get(name)
    if old is not None:
        bpy.data.images.remove(old)          # a fresh, unpacked image saves cleanly
    img = bpy.data.images.new(name, width=w, height=h, alpha=True)
    img.colorspace_settings.name = 'sRGB'
    img.pixels.foreach_set(rgba.ravel())
    img.update()
    img["rb2_baked_from"] = mat.name
    return img, ""


def use_baked(mat, img):
    """Points the material's base colour (and alpha, when it's used) at the baked image,
    remembering what fed them so the live colours can be restored."""
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    base = bsdf.inputs.get('Base Color')
    alpha = bsdf.inputs.get('Alpha')
    if not mat.get("rb2_live_sources"):
        saved = {}
        for key, sock in (('base', base), ('alpha', alpha)):
            node, out = _source(sock)
            if node is not None:
                saved[key] = [node.name, out.identifier]
        mat["rb2_live_sources"] = json.dumps(saved)
    node = nt.nodes.get(BAKED_NODE) or nt.nodes.new('ShaderNodeTexImage')
    node.name = BAKED_NODE
    node.label = "Baked Diffuse"
    node.image = img
    node.location = (bsdf.location.x - 350, bsdf.location.y + 350)
    nt.links.new(node.outputs['Color'], base)
    if alpha is not None and alpha.is_linked:
        nt.links.new(node.outputs['Alpha'], alpha)
    nt.nodes.active = node                       # also what most exporters look for
    mat["rb2_baked"] = True
    sync_milo_shader(mat)


def restore_live(mat):
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    saved = json.loads(mat.get("rb2_live_sources", "{}"))
    for key, sock_name in (('base', 'Base Color'), ('alpha', 'Alpha')):
        sock = bsdf.inputs.get(sock_name) if bsdf else None
        src = saved.get(key)
        if sock is None or not src:
            continue
        node = nt.nodes.get(src[0])
        out = next((o for o in node.outputs if o.identifier == src[1]), None) if node else None
        if out is not None:
            nt.links.new(out, sock)
    baked = nt.nodes.get(BAKED_NODE)
    if baked is not None:
        nt.nodes.remove(baked)
    for k in ("rb2_baked", "rb2_live_sources"):
        if k in mat:
            del mat[k]
    sync_milo_shader(mat)


# ---------------------------------------------------------------------------------------
# Material properties
# ---------------------------------------------------------------------------------------

def _icon_for(palette, index, rgb):
    if _PREVIEWS is None:
        return 0
    key = f"{palette}:{index}"
    pv = _PREVIEWS.get(key)
    if pv is None:
        pv = _PREVIEWS.new(key)
        px = [rgb[0], rgb[1], rgb[2], 1.0]
        pv.icon_size = (16, 16)
        pv.icon_pixels_float = px * 256
        pv.image_size = (64, 64)
        pv.image_pixels_float = px * 4096
    return pv.icon_id


def _PALETTE_NAMES_CACHE():
    _palettes()
    return _PALETTES['names']


def _palette_items(self, context):
    names = _PALETTE_NAMES_CACHE()
    key = ('palettes', tuple(names))
    if key not in _ENUM_CACHE:
        if names:
            _ENUM_CACHE[key] = [(n, n.rsplit('.', 1)[0],
                                 f"{len(_PALETTES['data'][n])} colours", i)
                                for i, n in enumerate(names)]
        else:
            _ENUM_CACHE[key] = [('NONE', "(no palettes loaded)", "", 0)]
    return _ENUM_CACHE[key]


def _color_items(palette):
    key = ('colors', palette)
    if key not in _ENUM_CACHE:
        cols = _palettes().get(palette)
        if cols:
            items = []
            for i, c in enumerate(cols):
                hexv = '#%02X%02X%02X' % tuple(int(round(v * 255)) for v in c)
                items.append((str(i), str(i), f"{palette} [{i}] {hexv}",
                              _icon_for(palette, i, c), i))
            _ENUM_CACHE[key] = items
        else:
            _ENUM_CACHE[key] = [('0', "-", "", 0, 0)]
    return _ENUM_CACHE[key]


def _make_palette_accessors(ch):
    def get(self):
        names = _PALETTE_NAMES_CACHE()
        name = getattr(self, f"sel_palette{ch}")
        return names.index(name) if name in names else 0

    def set_(self, value):
        names = _PALETTE_NAMES_CACHE()
        if 0 <= value < len(names):
            self[f"sel_palette{ch}"] = names[value]
            n = len(_palettes()[names[value]])
            if getattr(self, f"sel_index{ch}") >= n:
                self[f"sel_index{ch}"] = 0
            apply_material_colors(self.id_data)
    return get, set_


def _make_color_accessors(ch):
    def items(self, context):
        return _color_items(getattr(self, f"sel_palette{ch}"))

    def get(self):
        cols = _palettes().get(getattr(self, f"sel_palette{ch}"))
        i = getattr(self, f"sel_index{ch}")
        return i if cols and 0 <= i < len(cols) else 0

    def set_(self, value):
        self[f"sel_index{ch}"] = value
        apply_material_colors(self.id_data)
    return items, get, set_


_pg1, _ps1 = _make_palette_accessors(1)
_pg2, _ps2 = _make_palette_accessors(2)
_ci1, _cg1, _cs1 = _make_color_accessors(1)
_ci2, _cg2, _cs2 = _make_color_accessors(2)


def _kind_update(self, context):
    wire_diffuse_alpha(self.id_data, alpha_keeps(self))
    apply_material_colors(self.id_data, context)


class RB2CCMaterialSettings(bpy.types.PropertyGroup):
    loaded: BoolProperty(default=False)
    kind: EnumProperty(name="Colour Source", items=KIND_ITEMS, default='ITEM',
                       update=_kind_update,
                       description="Where this material's colour comes from. Detected on "
                                   "load; change it if the guess is wrong")
    source: StringProperty(name="Texture Milo")
    target: StringProperty(name="Milo Material")
    cfg: StringProperty(name="Outfit Config")
    uses1: BoolProperty(default=False)
    uses2: BoolProperty(default=False)
    link2: BoolProperty(default=False,
                        description="Colour 2 follows colour 1's index (one cfg choice "
                                    "drives both palettes, e.g. hair + hair_secondary)")
    cfg_palette1: StringProperty()
    cfg_palette2: StringProperty()
    cfg_index1: IntProperty(default=0)
    cfg_index2: IntProperty(default=0)
    sel_palette1: StringProperty()
    sel_palette2: StringProperty()
    sel_index1: IntProperty(default=0, min=0)
    sel_index2: IntProperty(default=0, min=0)
    palette1: EnumProperty(name="Palette", items=_palette_items, get=_pg1, set=_ps1,
                           description="Palette for colour 1 (the item's .cfg default is "
                                       "selected on load)")
    palette2: EnumProperty(name="Palette", items=_palette_items, get=_pg2, set=_ps2,
                           description="Palette for colour 2 (the item's .cfg default is "
                                       "selected on load)")
    color1: EnumProperty(name="Color 1", items=_ci1, get=_cg1, set=_cs1,
                         description="Colour 1 - the item's main colour")
    color2: EnumProperty(name="Color 2", items=_ci2, get=_cg2, set=_cs2,
                         description="Colour 2 - where the diffuse alpha is white")


def color_count(pg):
    return 2 if pg.uses2 else (1 if pg.uses1 else 0)


def _palette_milo_update(self, context):
    path = bpy.path.abspath(self.palette_milo) if self.palette_milo else ""
    if not path or not os.path.isfile(path):
        return
    try:
        store_palettes(context.scene, read_palette_milo(path), path)
    except Exception as e:
        _log(f"RB2 creator: couldn't load palettes from {path}: {e}")
        return
    for mat in bpy.data.materials:
        pg = getattr(mat, 'rb2cc', None)
        if pg is not None and pg.loaded:
            for ch in (1, 2):
                if not getattr(pg, f"sel_palette{ch}"):
                    pg[f"sel_palette{ch}"] = getattr(pg, f"cfg_palette{ch}")
            apply_material_colors(mat, context)


def _character_color_accessors(kind):
    prop = 'skin_index' if kind == 'SKIN' else 'eye_index'
    palette = CHARACTER_PALETTES[kind]

    def items(self, context):
        return _color_items(palette)

    def get(self):
        cols = _palettes().get(palette)
        i = getattr(self, prop)
        return i if cols and 0 <= i < len(cols) else 0

    def set_(self, value):
        self[prop] = value
        for mat in bpy.data.materials:
            pg = getattr(mat, 'rb2cc', None)
            if pg is not None and pg.loaded and pg.kind == kind:
                apply_material_colors(mat)
    return items, get, set_


_sk_i, _sk_g, _sk_s = _character_color_accessors('SKIN')
_ey_i, _ey_g, _ey_s = _character_color_accessors('EYES')


class RB2CCSceneSettings(bpy.types.PropertyGroup):
    skin_index: IntProperty(default=0, min=0)
    eye_index: IntProperty(default=0, min=0)
    skin_color: EnumProperty(name="Skin Tone", items=_sk_i, get=_sk_g, set=_sk_s,
                             description="The character's skin tone (skin.pal), applied to "
                                         "every material using Skin")
    eye_color: EnumProperty(name="Eye Colour", items=_ey_i, get=_ey_g, set=_ey_s,
                            description="The character's eye colour (eyes.pal), applied to "
                                        "every material using Eyes")
    skin_texture_dir: StringProperty(
        name="Skin Textures Folder", subtype='DIR_PATH',
        description="Optional folder searched (deeply) for skin textures that a milo only "
                    "references. Folders next to and above the loaded milo are always "
                    "searched too")
    palette_milo: StringProperty(
        name="Color Palettes Milo", subtype='FILE_PATH', update=_palette_milo_update,
        description="RB2's colorpalettes.milo_xbox. Found automatically next to (or above) "
                    "the texture milo when left empty")


# ---------------------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------------------

def _base_name(name):
    return _BLENDER_DUP_RE.sub('', name)


def channels_for(target, choices):
    """{channel: (palette, index, choice number)} for one material from the cfg choices.
    The first choice to claim a channel keeps it."""
    out = {}
    for k, c in enumerate(choices):
        if target not in c['mats']:
            continue
        for ch in (1, 2):
            pal = c.get(f"palette{ch}")
            if pal and ch not in out:
                out[ch] = (pal, c['index'], k)
    return out


def dress_jobs(context, info, jobs, filepath, flip_green=True, choice_indices=None,
               skin_dir=None, fur_shells=True, milo_shader=False):
    """Builds and colours the materials in `jobs` [(material, slot)] from one milo's
    MiloTextures. `choice_indices` (a character's '(colors a b)') overrides the cfg
    defaults, one index per cfg colour choice in cfg order. Returns (dressed material
    names, [warnings])."""
    slots, texs, choices = info.slots, info.texs, info.choices
    kinds = {sl['target']: classify_slot(sl, choices, info.rendered) for sl in slots}
    pal_ok, pal_msg = True, ""
    if choices or any(kinds[sl['target']] != 'ITEM' for _m, sl in jobs):
        pal_ok, pal_msg = ensure_palettes(context, filepath)
        if not pal_ok:
            _log(f"  {pal_msg}")

    if skin_dir is None:
        skin_dir = bpy.path.abspath(context.scene.rb2cc.skin_texture_dir) \
            if context.scene.rb2cc.skin_texture_dir else ""
    missing = []

    def resolve(sl, role):
        """(tex name, parsed tex) for a slot's texture role. Skin and eye materials
        fall back to candidate names, here and then in milos on disk."""
        name = sl.get(role)
        if isinstance(name, str) and name and _has_pixels(texs.get(name)):
            return name, texs[name]
        if kinds[sl['target']] == 'ITEM':
            if isinstance(name, str) and name:
                if name in info.rendered:
                    _log(f"  '{name}' is drawn at runtime (a patch/render target) - "
                         f"slot left empty")
                else:
                    _log(f"  '{name}' isn't in this milo (it's probably in the item's "
                         f"resource milo) - slot left empty")
            return None, None
        cands = candidate_names(sl, role)
        if role == 'diffuse_tex' and kinds[sl['target']] == 'SKIN' and info.skin_source \
                and (name in info.rendered or not name):
            cands.insert(0, info.skin_source)
        for n in cands:
            if _has_pixels(texs.get(n)):
                return n, texs[n]
        if kinds[sl['target']] == 'EYES' and role != 'diffuse_tex' and not name:
            return None, None           # retail eyes have no normal or spec map
        hit = find_textures_on_disk(cands, filepath, skin_dir)
        if hit:
            n, tex, path = hit
            _log(f"  '{n}' found in {path}")
            texs[n] = tex
            return n, tex
        if role == 'diffuse_tex':
            missing.append((sl['target'], cands))
        _log(f"  {sl['target']}: no {role.replace('_', ' ')} found (looked for "
             f"{', '.join(cands)})")
        return None, None

    images = {}

    def image(sl, role, img_role):
        name, tex = resolve(sl, role)
        if tex is None:
            return None
        key = (name, img_role)
        if key not in images:
            try:
                img = build_image(name, tex, img_role, flip_green)
                if img_role == 'diffuse':
                    img.alpha_mode = 'CHANNEL_PACKED'   # alpha is data, not coverage
                images[key] = img
            except (TextureImportError, ValueError) as e:
                _log(f"  couldn't decode '{name}': {e}")
                images[key] = None
        return images[key]

    done = []
    for m, sl in jobs:
        kind = kinds[sl['target']]
        build_creator_material(m, sl, dict(
            diffuse=image(sl, 'diffuse_tex', 'diffuse'),
            mask=image(sl, 'two_color_mask', 'specular')
            if sl.get('two_color_mask') else None,
            normal=image(sl, 'normal_map', 'normal'),
            specular=image(sl, 'specular_map', 'specular')),
            keep_alpha=kind in CHARACTER_PALETTES, milo_shader=milo_shader)

        chans = channels_for(sl['target'], choices)
        pg = m.rb2cc
        pg.loaded = True
        pg.source = filepath
        pg.target = sl['target']
        pg.cfg = ", ".join(info.cfgs)
        pg["kind"] = [k for k, _n, _d in KIND_ITEMS].index(kind)
        pg.uses1 = 1 in chans
        pg.uses2 = 2 in chans
        pg.link2 = 1 in chans and 2 in chans and chans[1][2] == chans[2][2]
        for ch in (1, 2):
            pal, default, k = chans.get(ch, ("", 0, -1))
            idx = default
            if pal and choice_indices is not None and 0 <= k < len(choice_indices):
                idx = int(choice_indices[k])
            pg[f"cfg_palette{ch}"] = pal
            pg[f"cfg_index{ch}"] = default        # the item's own .cfg default, always
            pg[f"sel_palette{ch}"] = pal
            pg[f"sel_index{ch}"] = idx
            if pal and pal_ok and pal not in _palettes(context):
                _log(f"  '{m.name}': palette '{pal}' isn't in the palettes milo")
        wire_diffuse_alpha(m, alpha_keeps(pg))
        if kind == 'SKIN' and info.makeup_rects:
            m["rb2_makeup_rects"] = json.dumps(info.makeup_rects)
            rebuild_makeup(m, context)
        apply_material_colors(m, context)
        m["milo_material"] = sl['target']
        if sl.get('fur'):
            from .rb2_fur import apply_fur
            users = [o for o in bpy.data.objects if o.type == 'MESH'
                     and any(s.material is m for s in o.material_slots)]
            if apply_fur(m, users, sl['fur'], fur_shells):
                f = sl['fur']
                _log(f"  '{m.name}': fur ({f['name']}) - "
                     + (f"{f['passes']} shells, thickness {f['thickness']:g}, gravity "
                        f"{f['gravity']:g}" if fur_shells else "flat (2D)"))
        sync_milo_shader(m)
        done.append(m.name)

        if kind != 'ITEM':
            desc = "character " + ("skin tone" if kind == 'SKIN' else "eye colour")
        else:
            desc = {0: "no colour options", 1: "1 colour", 2: "2 colours"}[color_count(pg)]
        extra = "".join(f", colour {ch} = {getattr(pg, f'sel_palette{ch}')}"
                        f"[{getattr(pg, f'sel_index{ch}')}]" for ch in sorted(chans))
        if pg.link2:
            extra += " (one choice drives both)"
        if 'two_color' in sl:
            extra += f", two_color flag {sl['two_color']}"
        if sl.get('blend') in (BLEND_ADD, BLEND_SRC_ALPHA, BLEND_PREMULT):
            extra += ", blended"
        _log(f"  '{m.name}' <- {sl['target']}: {desc}{extra}")

    if missing:
        warn_skin = (f"No skin diffuse found for {', '.join(t for t, _c in missing)} - set "
                     f"'Skin Textures Folder' and load again")
    else:
        warn_skin = ""
    return done, [w for w in (pal_msg, warn_skin) if w]


class MATERIAL_OT_rb2cc_load_textures(bpy.types.Operator, ImportHelper):
    """Load an RB2 texture or item milo (armyjacket_dirty, malerings_chief, ...) into this
    material, replacing the placeholder. Its .cfg decides whether the item takes one
    colour, two, or none"""
    bl_idname = "material.rb2cc_load_textures"
    bl_label = "Load Customizable Textures"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".milo_xbox"
    filter_glob: StringProperty(default="*.milo_xbox;*.milo_ps3;*.milo", options={'HIDDEN'})
    material_name: StringProperty(options={'HIDDEN'})
    apply_to_all_targets: BoolProperty(
        name="Also Dress Other Targets",
        description="Apply every material this milo defines to the matching materials in "
                    "the scene (e.g. the goggles' lenses), not just this one",
        default=True)
    flip_normal_green: BoolProperty(
        name="Flip Normal Map Green",
        description="Convert the normal maps' Y from DirectX (down) to Blender's OpenGL (up)",
        default=True)
    use_milo_shader: BoolProperty(
        name="Use Milo Shader",
        description="Shade with the game-style Milo Character Shader (Blinn-Phong specular "
                    "with the specular map's gloss, as the GDRB/TBRB importers do) instead of "
                    "a plain Principled BSDF. Experimental for Rock Band 2; it can be "
                    "switched per material afterwards in this panel",
        default=False)

    def execute(self, context):
        mat = bpy.data.materials.get(self.material_name) or context.material
        if mat is None:
            self.report({'ERROR'}, "No material to load into")
            return {'CANCELLED'}
        try:
            info = read_texture_milo(self.filepath)
        except Exception as e:
            self.report({'ERROR'}, f"Couldn't read texture milo: {e}")
            return {'CANCELLED'}
        slots, texs, choices = info.slots, info.texs, info.choices
        if not slots:
            self.report({'ERROR'}, "This milo defines no materials (no 'target' properties "
                                   "and no Mat entries) - is it an RB2 item milo?")
            return {'CANCELLED'}

        _log(f"===== RB2 creator textures: '{info.dir_name}' ({self.filepath}) =====")
        for n in info.notes:
            _log(f"  {n}")

        kinds = {sl['target']: classify_slot(sl, choices, info.rendered) for sl in slots}

        def has_content(sl):
            return (sl.get('from_mat') or kinds[sl['target']] != 'ITEM'
                    or any(sl.get(k) in texs for k in _TEX_KEYS))

        by_target = {sl['target']: sl for sl in slots}
        own = by_target.get(_base_name(mat.name))
        if own is None:
            usable = [sl for sl in slots if has_content(sl)]
            if len(usable) != 1:
                names = ", ".join(by_target)
                self.report({'ERROR'}, f"'{mat.name}' isn't one of this milo's materials "
                                       f"({names})")
                return {'CANCELLED'}
            own = usable[0]
            _log(f"  '{mat.name}' doesn't match by name; dressing it as '{own['target']}'")

        jobs = [(mat, own)]
        if self.apply_to_all_targets:
            for sl in slots:
                if sl is own:
                    continue
                if not has_content(sl):
                    _log(f"  '{sl['target']}': its textures aren't in this milo (probably "
                         f"in the item's resource milo) - not touched")
                    continue
                for m in bpy.data.materials:
                    if m is not mat and _base_name(m.name) == sl['target']:
                        jobs.append((m, sl))

        done, warn = dress_jobs(context, info, jobs, self.filepath,
                                flip_green=self.flip_normal_green,
                                milo_shader=self.use_milo_shader)
        msg = f"Loaded '{info.dir_name}' into {len(done)} material(s): {', '.join(done)}"
        self.report({'WARNING'} if warn else {'INFO'}, ". ".join([msg] + warn))
        _log(f"===== {msg} =====")
        return {'FINISHED'}


class MATERIAL_OT_rb2cc_milo_shader(bpy.types.Operator):
    """Switch this dressed material between the Milo Character Shader (game-style
    specular) and a plain Principled BSDF"""
    bl_idname = "material.rb2cc_milo_shader"
    bl_label = "Milo Shader"
    bl_options = {'REGISTER', 'UNDO'}

    material_name: StringProperty(options={'HIDDEN'})
    enable: BoolProperty(options={'HIDDEN'})
    all_materials: BoolProperty(
        name="All Dressed Materials",
        description="Apply to every Rock Band 2 dressed material in the file",
        default=False)

    def execute(self, context):
        mats = [m for m in bpy.data.materials if m.rb2cc.loaded] if self.all_materials \
            else [bpy.data.materials.get(self.material_name) or context.material]
        n = 0
        for m in mats:
            if m is None or m.node_tree is None or _principled(m.node_tree) is None:
                continue
            set_milo_shader(m, self.enable)
            n += 1
        self.report({'INFO'}, f"{'Milo shader' if self.enable else 'Principled BSDF'} on "
                              f"{n} material(s)")
        return {'FINISHED'}


class MATERIAL_OT_rb2cc_load_makeup(bpy.types.Operator, ImportHelper):
    """Load an RB2 eye or lip makeup milo (head_1_lips_1, head_5_eyes_5 ...) onto this head
    material - drawn over the face where the head's skin proxy places makeup"""
    bl_idname = "material.rb2cc_load_makeup"
    bl_label = "Load Makeup"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".milo_xbox"
    filter_glob: StringProperty(default="*.milo_xbox;*.milo_ps3;*.milo", options={'HIDDEN'})
    material_name: StringProperty(options={'HIDDEN'})
    kind: EnumProperty(name="Makeup", items=(('AUTO', "From the Milo", "Eyes or lips, from "
                                                     "the milo's names"),
                                             ('eyes', "Eyes", ""), ('lips', "Lips", "")),
                       default='AUTO')

    def execute(self, context):
        mat = bpy.data.materials.get(self.material_name) or context.material
        if mat is None or not mat.get("rb2_makeup_rects"):
            self.report({'ERROR'}, "Makeup goes on a head skin material loaded from its "
                                   "head milo")
            return {'CANCELLED'}
        try:
            kind = set_makeup(mat, None if self.kind == 'AUTO' else self.kind,
                              self.filepath, context)
        except Exception as e:
            self.report({'ERROR'}, f"Couldn't load makeup: {e}")
            return {'CANCELLED'}
        if kind not in json.loads(mat["rb2_makeup_rects"]):
            self.report({'WARNING'}, f"This head has no {kind} makeup area")
        else:
            self.report({'INFO'}, f"{kind.capitalize()} makeup loaded")
        return {'FINISHED'}


class MATERIAL_OT_rb2cc_fur_mode(bpy.types.Operator):
    """Switch this fur material between shell fur and a single flat layer"""
    bl_idname = "material.rb2cc_fur_mode"
    bl_label = "Fur Mode"
    bl_options = {'REGISTER', 'UNDO'}

    material_name: StringProperty(options={'HIDDEN'})
    shells: BoolProperty(default=True)

    def execute(self, context):
        from .rb2_fur import set_fur_mode
        mat = bpy.data.materials.get(self.material_name)
        if mat is None or not mat.get("rb2_fur"):
            return {'CANCELLED'}
        set_fur_mode(mat, self.shells)
        return {'FINISHED'}


class MATERIAL_OT_rb2cc_bake_all(bpy.types.Operator):
    """Write each Rock Band 2 material's live colours - palette tint, keep masks, makeup,
    flat fur tint - into one plain diffuse texture and use it, for exporting. Computed from
    the texture pixels directly (no Cycles bake, no UV margins), at the game texture's size"""
    bl_idname = "material.rb2cc_bake_all"
    bl_label = "Bake All Character Diffuse Textures"
    bl_options = {'REGISTER', 'UNDO'}

    save_dir: StringProperty(
        name="Also Save PNGs To", subtype='DIR_PATH',
        description="Optional folder to save the baked textures to as PNG files (for "
                    "exporters that don't embed images). Leave empty to only pack them "
                    "into the .blend")
    use_in_materials: BoolProperty(
        name="Use Baked Textures in Materials", default=True,
        description="Connect each material's base colour to its baked texture, so "
                    "exporters see a plain textured material ('Restore Live Colours' "
                    "undoes it)")

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=420)

    def execute(self, context):
        import os
        mats = [m for m in bpy.data.materials
                if getattr(m, 'rb2cc', None) is not None and m.rb2cc.loaded]
        if not mats:
            self.report({'WARNING'}, "No Rock Band 2 creator materials loaded")
            return {'CANCELLED'}
        out_dir = bpy.path.abspath(self.save_dir) if self.save_dir else ""
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        baked, skipped = [], []
        for m in mats:
            if m.get("rb2_baked"):
                restore_live(m)              # bake from the live colours, not the old bake
            try:
                img, why = bake_material(m)
            except Exception as e:
                img, why = None, str(e)
            if img is None:
                skipped.append(f"{m.name}: {why}")
                continue
            if out_dir:
                path = os.path.join(out_dir, bpy.path.clean_name(img.name) + ".png")
                img.filepath_raw = path
                img.file_format = 'PNG'
                try:
                    img.save()
                except RuntimeError as e:
                    skipped.append(f"{m.name}: baked, but saving {path} failed ({e})")
            img.pack()
            if self.use_in_materials:
                use_baked(m, img)
            baked.append(m.name)
        _log(f"RB2 bake: {len(baked)} material(s) baked"
             + (f" and saved to {out_dir}" if out_dir else "") + ": " + ", ".join(baked))
        for sk in skipped:
            _log(f"RB2 bake: skipped {sk}")
        self.report({'INFO'}, f"Baked {len(baked)} diffuse texture(s)"
                    + (f"; {len(skipped)} skipped (see the log)" if skipped else ""))
        return {'FINISHED'}


class MATERIAL_OT_rb2cc_restore_live(bpy.types.Operator):
    """Reconnect the live palette colours on every baked Rock Band 2 material"""
    bl_idname = "material.rb2cc_restore_live"
    bl_label = "Restore Live Colours"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        n = 0
        for m in bpy.data.materials:
            if m.get("rb2_baked"):
                restore_live(m)
                n += 1
        self.report({'INFO'}, f"Live colours restored on {n} material(s)")
        return {'FINISHED'}


class MATERIAL_OT_rb2cc_clear_makeup(bpy.types.Operator):
    """Remove this head's eye or lip makeup"""
    bl_idname = "material.rb2cc_clear_makeup"
    bl_label = "Remove Makeup"
    bl_options = {'REGISTER', 'UNDO'}

    material_name: StringProperty(options={'HIDDEN'})
    kind: StringProperty(options={'HIDDEN'})

    def execute(self, context):
        mat = bpy.data.materials.get(self.material_name)
        if mat is not None:
            clear_makeup(mat, self.kind, context)
        return {'FINISHED'}


# ---------------------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------------------

class MATERIAL_PT_rb2cc(bpy.types.Panel):
    bl_label = "Rock Band 2 Character Creator Texture"
    bl_idname = "MATERIAL_PT_rb2cc"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "material"

    @classmethod
    def poll(cls, context):
        return context.material is not None

    def draw(self, context):
        layout = self.layout
        mat = context.material
        pg = mat.rb2cc
        scn = context.scene

        op = layout.operator(MATERIAL_OT_rb2cc_load_textures.bl_idname,
                             text="Load Customizable Textures", icon='TEXTURE')
        op.material_name = mat.name
        row = layout.row(align=True)
        row.operator(MATERIAL_OT_rb2cc_bake_all.bl_idname, icon='RENDER_STILL')
        if any(m.get("rb2_baked") for m in bpy.data.materials):
            row.operator(MATERIAL_OT_rb2cc_restore_live.bl_idname, text="", icon='LOOP_BACK')
        if mat.get("rb2_baked"):
            layout.label(text="Using its baked texture - colour changes won't show until "
                              "you restore live colours or bake again", icon='INFO')
        if pg.loaded:
            on = bool(mat.get(MILO_SHADER_KEY))
            row = layout.row(align=True)
            row.label(text="Shading:", icon='SHADING_RENDERED')
            op = row.operator(MATERIAL_OT_rb2cc_milo_shader.bl_idname, text="Milo Shader",
                              depress=on)
            op.material_name, op.enable, op.all_materials = mat.name, True, False
            op = row.operator(MATERIAL_OT_rb2cc_milo_shader.bl_idname, text="Principled",
                              depress=not on)
            op.material_name, op.enable, op.all_materials = mat.name, False, False
            op = row.operator(MATERIAL_OT_rb2cc_milo_shader.bl_idname, text="",
                              icon='WORLD')
            op.material_name, op.enable, op.all_materials = mat.name, not on, True
        layout.prop(scn.rb2cc, "palette_milo")

        info = layout.column(align=True)
        info.enabled = pg.loaded
        if pg.loaded:
            info.label(text=f"Source: {os.path.basename(pg.source)}", icon='FILE')
            info.label(text=f"Material: {pg.target}   Config: {pg.cfg or '-'}")
        else:
            info.label(text="Not in use - load an RB2 texture milo to enable", icon='INFO')

        src = layout.row()
        src.enabled = pg.loaded
        src.prop(pg, "kind")

        have_pals = bool(_palettes(context))
        if pg.loaded and pg.kind in CHARACTER_PALETTES:
            box = layout.box()
            skin = pg.kind == 'SKIN'
            box.label(text="Skin Tone" if skin else "Eye Colour", icon='COLOR')
            if not have_pals:
                box.label(text="Set 'Color Palettes Milo' to pick colours", icon='ERROR')
                return
            box.template_icon_view(scn.rb2cc, "skin_color" if skin else "eye_color",
                                   show_labels=True, scale=4.0, scale_popup=3.0)
            box.label(text="Shared by the whole character - also under Character Skin & "
                           "Eyes", icon='INFO')
            if skin and mat.get("rb2_makeup_rects"):
                rects = json.loads(mat["rb2_makeup_rects"])
                state = json.loads(mat.get("rb2_makeup", "{}"))
                mk = layout.box()
                mk.label(text="Makeup", icon='BRUSH_DATA')
                for kind in MAKEUP_KINDS:
                    row = mk.row(align=True)
                    row.enabled = kind in rects
                    st = state.get(kind)
                    row.label(text=f"{kind.capitalize()}: "
                                   + (os.path.splitext(os.path.basename(st['source']))[0]
                                      if st else "none"))
                    op = row.operator(MATERIAL_OT_rb2cc_load_makeup.bl_idname, text="",
                                      icon='FILEBROWSER')
                    op.material_name, op.kind = mat.name, kind
                    if st:
                        op = row.operator(MATERIAL_OT_rb2cc_clear_makeup.bl_idname, text="",
                                          icon='X')
                        op.material_name, op.kind = mat.name, kind
            return

        if mat.get("rb2_fur"):
            fb = layout.box()
            shells = bool(mat.get("rb2_fur_shells", True))
            fb.label(text="Fur: " + ("shells (.fur)" if shells else "flat (2D)"), icon='OUTLINER_DATA_CURVES')
            row = fb.row(align=True)
            op = row.operator(MATERIAL_OT_rb2cc_fur_mode.bl_idname, text="Shells",
                              depress=shells)
            op.material_name, op.shells = mat.name, True
            op = row.operator(MATERIAL_OT_rb2cc_fur_mode.bl_idname, text="Flat",
                              depress=not shells)
            op.material_name, op.shells = mat.name, False

        count = color_count(pg)
        box = layout.box()
        box.enabled = pg.loaded and count > 0
        head = box.row()
        head.label(text="Color Palette", icon='COLOR')
        if pg.loaded and count == 0:
            head.label(text="(this item has no colour options)")
        if pg.loaded and count and not have_pals:
            box.label(text="Set 'Color Palettes Milo' to pick colours", icon='ERROR')

        row = box.row()
        for ch in (1, 2):
            col = row.column()
            used = pg.loaded and getattr(pg, f"uses{ch}") and have_pals
            linked = ch == 2 and pg.link2
            col.enabled = used and not linked
            col.label(text=f"Color {ch}")
            col.prop(pg, f"palette{ch}", text="")
            if used:
                col.template_icon_view(pg, f"color{ch}", show_labels=True, scale=4.0,
                                       scale_popup=3.0)
                if linked:
                    col.label(text="Follows Color 1")
                else:
                    col.label(text=f"Index {getattr(pg, f'sel_index{ch}')} "
                                   f"(default {getattr(pg, f'cfg_index{ch}')})")
            else:
                col.label(text="Not used" if pg.loaded else "-")


class MATERIAL_PT_rb2cc_character(bpy.types.Panel):
    bl_label = "Character Skin & Eyes"
    bl_idname = "MATERIAL_PT_rb2cc_character"
    bl_parent_id = "MATERIAL_PT_rb2cc"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "material"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        scn = context.scene
        layout.prop(scn.rb2cc, "skin_texture_dir")
        used = {mat.rb2cc.kind for mat in bpy.data.materials
                if getattr(mat, 'rb2cc', None) is not None and mat.rb2cc.loaded}
        have_pals = bool(_palettes(context))
        row = layout.row()
        for kind, prop, label in (('SKIN', "skin_color", "Skin Tone"),
                                  ('EYES', "eye_color", "Eye Colour")):
            col = row.column()
            col.enabled = have_pals and kind in used
            col.label(text=label)
            if col.enabled:
                col.template_icon_view(scn.rb2cc, prop, show_labels=True, scale=4.0,
                                       scale_popup=3.0)
                n = sum(1 for mat in bpy.data.materials
                        if getattr(mat, 'rb2cc', None) is not None and mat.rb2cc.loaded
                        and mat.rb2cc.kind == kind)
                col.label(text=f"{n} material(s)")
            else:
                col.label(text="No materials yet" if have_pals else "No palettes loaded")


# ---------------------------------------------------------------------------------------
# Registration (called from the add-on's register / unregister)
# ---------------------------------------------------------------------------------------

_classes = (
    RB2CCMaterialSettings,
    RB2CCSceneSettings,
    MATERIAL_OT_rb2cc_load_textures,
    MATERIAL_OT_rb2cc_milo_shader,
    MATERIAL_OT_rb2cc_load_makeup,
    MATERIAL_OT_rb2cc_clear_makeup,
    MATERIAL_OT_rb2cc_fur_mode,
    MATERIAL_OT_rb2cc_bake_all,
    MATERIAL_OT_rb2cc_restore_live,
    MATERIAL_PT_rb2cc,
    MATERIAL_PT_rb2cc_character,
)


def register():
    global _PREVIEWS
    _PREVIEWS = bpy.utils.previews.new()
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Material.rb2cc = PointerProperty(type=RB2CCMaterialSettings)
    bpy.types.Scene.rb2cc = PointerProperty(type=RB2CCSceneSettings)


def unregister():
    global _PREVIEWS
    del bpy.types.Scene.rb2cc
    del bpy.types.Material.rb2cc
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
    _ENUM_CACHE.clear()
    if _PREVIEWS is not None:
        bpy.utils.previews.remove(_PREVIEWS)
        _PREVIEWS = None
