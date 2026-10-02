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
MATERIALS
===========================================================================================
RndMat revision 56 (GDRB) or 55 (TBRB), which share this slot layout. After a 13-byte
object header come blend, colour, flags and a texture
transform, then the diffuse slot; three specular-colour floats and a few more fields later,
the normal, emissive and specular slots. All 14 retail materials checked decode cleanly,
and slots are read by POSITION rather than by name, because the data doesn't always follow
the naming - 21st_shoes.mat puts 'billiejoe shoes spec.tex' in both its diffuse and
specular slots and never uses 'billiejoe shoes_diff.tex'. That's reproduced as the game has
it.

===========================================================================================
KNOWN GAPS
===========================================================================================
1. Alpha isn't wired up, so hair, brows and lashes render solid rather than cut out.
2. Normal-detail maps (skin_detail_normals, hair_detail_normals) and the eyes' reflection
   pass aren't used.
4. TBRB's animated wrinkles aren't reproduced. Its head normal map is a render target that a
   TexBlender fills each frame, blending wrinkle maps into the base map in regions defined by
   the Blend_* helper meshes, as bone pairs move apart. The static base map is used instead.
3. The specular map feeds Specular Tint as a straightforward stand-in; the game's shader
   uses it differently.
"""

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


def parse_mat(data, name):
    """RndMat revision 56 - returns the texture slots by position."""
    r = _R(data)
    rev = r.u32()
    if rev not in MAT_REVISIONS:
        raise TextureImportError(f"'{name}' is material revision {rev}; only 55 (TBRB) and "
                                 f"56 (GDRB) are supported")
    r.p = MAT_OBJECT_HEADER_END
    r.p += 4 + 16 + 1 + 1 + 4 + 1 + 4 + 1 + 4 + 4 + 48   # blend .. texture transform
    diffuse = r.sym()
    r.sym()                                             # next pass
    r.p += 1 + 1 + 4 + 12 + 4                           # intensify, cull, emissive mult,
                                                        # specular colour, specular power
    normal = r.sym()
    emissive = r.sym()
    specular = r.sym()
    return dict(diffuse=diffuse, normal=normal, emissive=emissive, specular=specular)


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


# ---------------------------------------------------------------------------------------
# Block decoding (numpy - Blender bundles it)
# ---------------------------------------------------------------------------------------

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

    An existing image of the same name is overwritten in place rather than reused as-is,
    so re-importing picks up changes (such as the normal-map green flip) while any node
    already pointing at the image keeps working. One texture can also be needed in two
    colour spaces - 21st_shoes.mat uses 'billiejoe shoes spec.tex' as both diffuse (sRGB)
    and specular (Non-Color) - so a second role that disagrees on colour space gets its
    own image rather than silently sharing the first."""
    colorspace = 'sRGB' if role == 'diffuse' else 'Non-Color'
    if role == 'normal' and tex['encoding'] == ENC_ATI2:
        rgba = decode_normal(tex, flip_green)
    else:
        rgba = decode_rgba(tex)
    h, w = rgba.shape[:2]

    base = _image_name(tex_name)
    name = base
    img = bpy.data.images.get(name)
    if img is not None and img.get("milo_texture") == tex_name \
            and img.get("milo_colorspace", colorspace) != colorspace:
        name = f"{base}_{'color' if colorspace == 'sRGB' else 'data'}"
        img = bpy.data.images.get(name)
    if img is None or tuple(img.size) != (w, h):
        if img is not None:
            bpy.data.images.remove(img)
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
    return img


def _input(node, *names):
    for n in names:
        sock = node.inputs.get(n)
        if sock is not None:
            return sock
    return None


def build_material_tree(mat, diffuse_img=None, normal_img=None, specular_img=None):
    """Replaces a material's nodes with the preview tree: diffuse -> Base Color, normal ->
    Normal Map -> Normal, specular -> Specular Tint."""
    mat.use_nodes = True
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    for n in list(nodes):
        nodes.remove(n)

    out = nodes.new('ShaderNodeOutputMaterial')
    out.location = (600, 300)
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (250, 300)
    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    for sock_name, value in (('Metallic', 0.0), ('Roughness', 1.0), ('IOR', 1.5)):
        sock = _input(bsdf, sock_name)
        if sock is not None:
            sock.default_value = value

    def tex_node(img, y):
        n = nodes.new('ShaderNodeTexImage')
        n.image = img
        n.location = (-500, y)
        return n

    if diffuse_img is not None:
        d = tex_node(diffuse_img, 500)
        base = _input(bsdf, 'Base Color')
        if base is not None:
            links.new(d.outputs['Color'], base)
    if normal_img is not None:
        nm = tex_node(normal_img, 150)
        nmap = nodes.new('ShaderNodeNormalMap')
        nmap.location = (-100, 150)
        nmap.space = 'TANGENT'
        links.new(nm.outputs['Color'], nmap.inputs['Color'])
        normal = _input(bsdf, 'Normal')
        if normal is not None:
            links.new(nmap.outputs['Normal'], normal)
    if specular_img is not None:
        sp = tex_node(specular_img, -200)
        tint = _input(bsdf, 'Specular Tint')
        if tint is not None:
            links.new(sp.outputs['Color'], tint)
    mat["milo_textured"] = True


def apply_textures(filepath, material_names, flip_green=True):
    """Builds textured node trees for the named materials from a milo's own Tex and Mat
    entries. Returns (textured count, image count, [notes])."""
    texs, mats, failed = read_milo_materials(filepath)
    notes = [f"skipped {n}: {r}" for n, r in failed]
    images = {}

    def image_for(tex_name, role):
        if not tex_name:
            return None
        if tex_name not in texs:
            notes.append(f"texture '{tex_name}' isn't in this milo")
            return None
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
        key = (tex_name, role)
        if key not in images:
            try:
                images[key] = build_image(tex_name, texs[tex_name], role, flip_green)
            except (TextureImportError, ValueError) as e:
                notes.append(f"could not decode '{tex_name}': {e}")
                images[key] = None
        return images[key]

    done = 0
    for mat_name in material_names:
        slots = mats.get(mat_name)
        mat = bpy.data.materials.get(mat_name)
        if slots is None or mat is None:
            if slots is None:
                notes.append(f"material '{mat_name}' isn't defined in this milo")
            continue
        build_material_tree(mat,
                            image_for(slots['diffuse'], 'diffuse'),
                            image_for(slots['normal'], 'normal'),
                            image_for(slots['specular'], 'specular'))
        done += 1
    return done, len([i for i in images.values() if i is not None]), notes
