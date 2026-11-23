"""
Rock Band 3 character textures: the character creator's tinting, the skin, and the skin
normal maps.

===========================================================================================
WHERE EVERYTHING IS
===========================================================================================
    item milo           its swap materials (*_swap.mat), diffuse / interp / mask / normal /
                        spec textures and its OutfitConfig (torso.cfg, hair.cfg, glasses.cfg)
    head/<g>/head.milo  skin.cfg (torso / legs / head skin), eyes.cfg, eyes and teeth
                        textures, norm.texblend + its norm_* controllers (head normal map)
    shared/colorpalettes.milo   every ColorPalette (skin.pal, cloth_dulled.pal, hair.pal ...),
                        the skin textures (<g>_<part>_diff / _interp_gw / _spec / _norm and
                        _norm_<body type>), <g>_head00_norm + <g>_head_norm01..18, the
                        head-normal region meshes (<g>_norm_chin ...), hair_shared_*, makeup
    shared/char_shared.milo     the skin materials (torso_naked, legs_skin, feet_skin,
                        feet_socks_skin, head_naked)

===========================================================================================
TINTING (OutfitConfig::MatSwap, revision 27; decomp bandobj/OutfitConfig.cpp)
===========================================================================================
Each MatSwap: mat (the swap material, whose diffuse is a runtime render target
*_diffuse_output.tex), resource_mat (the material the meshes are authored with - swapped
for 'mat' at load), two_color_diffuse / _interp / _mask textures, color1 palette + option,
color2 palette + option, and a texture list. An option picks one of the piece's three
colour indices (BandCharDesc OutfitPiece colours; skin and eyes fill index 0 with the skin
tone / eye colour), which indexes the palette (modulo its length).

The composite (MatSwap::Compose renders it on the GPU; the shader isn't in the executable,
so this is the formula that fits the retail textures - see the docs of the RB3 character
importer's planning notes):

    tint   = mix(colour1, colour2, interp)        colour2 = white without a second palette
    result = mix(diffuse * tint, diffuse, mask)   in display (gamma) values, like the game

interp and mask are the textures' alpha (RGB is white). A MatSwap with no textures and a
palette (glasses lenses) colours the material itself instead.

The skin uses the same thing through skin.cfg with skin.pal. The game also swaps the skin
normal maps (OutfitConfig::SetSkinTextures, 360 @0x8228fc50): torso, legs, feet and socks
get <g>_<torso|legs>_norm[_<body type>].tex by the dominant body group (see
rb3_proportions.body_type); the head's comes from norm.texblend: <g>_head00_norm with each
feature's region (<g>_norm_chin / eye / mouth / nose / shape .mesh, drawn in UV space,
alpha 0 on its border vertices) replaced by <g>_head_norm<type + 1>.tex.

Everything is computed with numpy into Blender images, so the RB3 shader (or the Milo
Character Shader) uses them like any other texture, and the material panel's colour
options recompute them.
"""

import hashlib
import json
import os
import struct

import bpy
import numpy as np
from bpy.props import IntProperty, PointerProperty

from .io import read_milo_container_body
from .utilities import _log
from . import texture_importer as ti


MARKER = b'\xAD\xDE\xAD\xDE'
SKIN_MATS = {                     # material -> (skin.cfg MatSwap index, texture part)
    'torso_naked.mat': (0, 'torso'),
    'legs_skin.mat': (1, 'legs'),
    'feet_skin.mat': (1, 'legs'),
    'feet_socks_skin.mat': (1, 'legs'),
    'head_naked.mat': (2, 'head'),
}
SKIN_OUTPUTS = ('torso_skin_diffuse_output.tex', 'legs_skin_diffuse_output.tex',
                'head_skin_diffuse_output.tex')
HEAD_FEATURES = ('chin', 'eye', 'mouth', 'nose', 'shape')


# ---------------------------------------------------------------------------------------
# Reading milos
# ---------------------------------------------------------------------------------------

_BODIES = {}


def _index(path):
    """(body, {name: (type, start, end)}) for every object in a milo, nested included."""
    from .mesh_importer import walk_milo_entries
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    hit = _BODIES.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    with open(path, 'rb') as f:
        body = read_milo_container_body(f.read())
    idx = {}
    for _d, _p, etype, ename, s, e in walk_milo_entries(body):
        if body[e - 4:e] == MARKER:
            e -= 4
        idx.setdefault(ename, (etype, s, e))
    _BODIES[path] = (mtime, (body, idx))
    return body, idx


def normalize_header(c):
    """An object whose type properties are set carries a DataArray after its type symbol
    (torso_admiraljacket_scarf_swap.mat: diffuse_tex2 ...). Drops it so every reader can
    take the usual 17-byte object header."""
    if len(c) > 17 and c[12] == 1:
        r = ti._R(c)
        r.p = 13
        ti._skip_data_array(r)
        return c[:12] + b'\x00' + c[r.p:]
    return c


class Library:
    """Objects by name from milos in priority order (the item first, then the head, then
    the shared milos)."""

    def __init__(self, paths):
        self.paths = [p for p in paths if p and os.path.isfile(p)]
        self._tex, self._norm, self._pal = {}, {}, {}

    def find(self, name, etype=None):
        for p in self.paths:
            got = _index(p)
            if not got:
                continue
            body, idx = got
            hit = idx.get(name)
            if hit and (etype is None or hit[0] == etype):
                return p, normalize_header(body[hit[1]:hit[2]])
        return None, None

    def objects(self, etype, path=None):
        out = []
        for p in ([path] if path else self.paths):
            got = _index(p)
            if got:
                body, idx = got
                out += [(n, normalize_header(body[s:e])) for n, (t, s, e) in idx.items()
                        if t == etype]
        return out

    # textures -------------------------------------------------------------------------
    def tex(self, name):
        """parse_tex dict, or None (missing, a render target or a dummy)."""
        if not name:
            return None
        _p, c = self.find(name, 'Tex')
        if c is None:
            return None
        try:
            t = ti.parse_tex(c, name)
        except Exception as e:
            _log(f"RB3 textures: '{name}' not decoded ({e})")
            return None
        if t.get('render_target') or not t.get('width'):
            return None
        return t

    def rgba(self, name):
        if name not in self._tex:
            t = self.tex(name)
            self._tex[name] = ti.decode_rgba(t) if t else None
        return self._tex[name]

    def normal(self, name):
        if name not in self._norm:
            t = self.tex(name)
            if t is None:
                self._norm[name] = None
            elif t['encoding'] == ti.ENC_ATI2:
                self._norm[name] = ti.decode_normal(t)
            else:
                self._norm[name] = ti.decode_rgba(t)
        return self._norm[name]

    # palettes -------------------------------------------------------------------------
    def palette(self, name):
        if not name:
            return None
        if name not in self._pal:
            _p, c = self.find(name, 'ColorPalette')
            self._pal[name] = parse_palette(c) if c else None
        return self._pal[name]


def parse_palette(c):
    """ColorPalette: 17-byte header, u32 count, count x (r, g, b, a) floats - display
    values; the fourth float is junk."""
    n = struct.unpack_from('>I', c, 17)[0]
    return [struct.unpack_from('>3f', c, 21 + 16 * i) for i in range(n)]


def parse_outfit_config(c):
    """{'colors': [3 default indices], 'swaps': [MatSwap dicts]} (OutfitConfig revision 27;
    only the part up to the MatSwaps is read)."""
    r = ti._R(c)
    r.p = 17
    colors = [struct.unpack_from('>i', c, r.p + 4 * i)[0] for i in range(3)]
    r.p += 12
    n = r.u32()
    swaps = []
    for _ in range(n):
        s = dict(mat=r.sym(), resource=r.sym(), diffuse=r.sym(), interp=r.sym(),
                 mask=r.sym(), pal1=r.sym())
        s['opt1'] = struct.unpack_from('>i', c, r.p)[0]; r.p += 4
        s['pal2'] = r.sym()
        s['opt2'] = struct.unpack_from('>i', c, r.p)[0]; r.p += 4
        s['textures'] = [r.sym() for _ in range(r.u32())]
        swaps.append(s)
    out = dict(colors=colors, swaps=swaps, patches=[])
    try:
        out['patches'] = _read_cfg_patches(r, c)
    except (ValueError, struct.error, IndexError, UnicodeDecodeError) as e:
        _log(f"RB3 textures: OutfitConfig patches not read ({e})")
    return out


def _read_cfg_patches(r, c):
    """The BandPatchMeshes after the MatSwaps (OutfitConfig rev 27): MeshAO list (name,
    coeffs, seams, name), computeAO byte, then the patches - rev, src transform, meshes,
    render_to byte, category. [{src, meshes, render_to, category}]"""
    for _ in range(r.u32()):                       # MeshAO
        r.sym()
        n = r.u32(); r.p += 4 * n                  # coeffs
        n = r.u32(); r.p += 8 * n                  # seams {index, coeff}
        r.sym()
    r.p += 1                                       # computeAO
    out = []
    n = r.u32()
    if n > 64:
        raise ValueError("bad patch count")
    for _ in range(n):
        rev = r.u32()
        if rev != 4:
            raise ValueError(f"BandPatchMesh revision {rev}")
        src = r.sym()
        meshes = [r.sym() for _ in range(r.u32())]
        render_to = bool(c[r.p]); r.p += 1
        cat = struct.unpack_from('>i', c, r.p)[0]; r.p += 4
        out.append(dict(src=src, meshes=meshes, render_to=render_to, category=cat))
    return out


def parse_mat(c, name):
    return ti.parse_mat(c, name)


# ---------------------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------------------

def _resize(a, h, w):
    if a.shape[0] == h and a.shape[1] == w:
        return a
    ys = (np.arange(h) * a.shape[0] // h).clip(0, a.shape[0] - 1)
    xs = (np.arange(w) * a.shape[1] // w).clip(0, a.shape[1] - 1)
    return a[ys][:, xs]


def data_channel(rgba):
    """An interp / mask texture's data: its alpha, or its red channel if the alpha is
    flat (a DXT1 mask)."""
    a = rgba[..., 3]
    return a if a.max() - a.min() > 0.5 else rgba[..., 0]


def compose_two_color(diffuse, interp, mask, c1, c2):
    """diffuse: H x W x 4 (0-255); interp / mask: 2-D arrays (0-255) or None; c1 / c2: display
    RGB 0-1 (c2 None = white). Returns H x W x 4 (0-255)."""
    h, w = diffuse.shape[:2]
    d = diffuse[..., :3] / 255.0
    t = np.zeros((h, w, 1)) if interp is None else _resize(interp, h, w)[..., None] / 255.0
    c1 = np.array(c1 if c1 is not None else (1.0, 1.0, 1.0))
    c2 = np.array(c2 if c2 is not None else (1.0, 1.0, 1.0))
    tint = c1 * (1.0 - t) + c2 * t
    out = d * tint
    if mask is not None:
        m = _resize(mask, h, w)[..., None] / 255.0
        out = out * (1.0 - m) + d * m
    res = diffuse.copy()
    res[..., :3] = np.clip(out, 0.0, 1.0) * 255.0
    return res


def colour_for(lib, pal, option, colors):
    p = lib.palette(pal)
    if not p:
        return None
    idx = colors[option] if 0 <= option < len(colors) else 0
    return p[idx % len(p)]


def rasterize_uv_mask(mesh, h, w):
    """A mesh drawn into UV space as an alpha mask (vertex colour alpha, interpolated) -
    how a TexBlendController's region mesh weights its texture."""
    mask = np.zeros((h, w))
    uv = np.array([v['uv'] for v in mesh['verts']], dtype=np.float64)
    px = np.c_[uv[:, 0] * w - 0.5, (1.0 - uv[:, 1]) * h - 0.5]   # rows top-first
    alpha = np.array([((v.get('colour') or 0xFFFFFFFF) >> 24) & 0xFF
                      for v in mesh['verts']]) / 255.0
    for f in mesh['faces']:
        a, b, c = (int(i) for i in f[:3])
        P = px[[a, b, c]]
        x0, y0 = np.floor(P.min(0)).astype(int)
        x1, y1 = np.ceil(P.max(0)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, w - 1), min(y1, h - 1)
        if x1 < x0 or y1 < y0:
            continue
        ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        (ax, ay), (bx, by), (cx, cy) = P
        det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(det) < 1e-12:
            continue
        l1 = ((by - cy) * (xs - cx) + (cx - bx) * (ys - cy)) / det
        l2 = ((cy - ay) * (xs - cx) + (ax - cx) * (ys - cy)) / det
        l3 = 1.0 - l1 - l2
        inside = (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6)
        val = l1 * alpha[a] + l2 * alpha[b] + l3 * alpha[c]
        sub = mask[y0:y1 + 1, x0:x1 + 1]
        sub[inside] = np.maximum(sub[inside], val[inside])
    return mask


def _raster(target, px, uv, faces, source, color=(1.0, 1.0, 1.0, 1.0), premult=False,
            border=False):
    """Draws triangles into `target` (top-row-first H x W x 4, 0-255, changed in place):
    px - per-vertex target pixel coordinates (col, row); uv - per-vertex source coordinates
    (u right, v UP, as parse_mesh_entry gives them); faces - vertex index triples.
    Blending: source alpha (kBlendSrcAlpha) or premultiplied (kBlendPreMultAlpha); border:
    texels outside 0..1 are transparent black (kTexBorderBlack), else clamped."""
    h, w = target.shape[:2]
    sh, sw = source.shape[:2]
    col = np.array(color[:3])
    for f in faces:
        a, b, c = (int(i) for i in f[:3])
        P = px[[a, b, c]]
        x0, y0 = np.floor(P.min(0)).astype(int)
        x1, y1 = np.ceil(P.max(0)).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w - 1), min(y1, h - 1)
        if x1 < x0 or y1 < y0:
            continue
        ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        (ax, ay), (bx, by), (cx, cy) = P
        det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(det) < 1e-12:
            continue
        l1 = ((by - cy) * (xs - cx) + (cx - bx) * (ys - cy)) / det
        l2 = ((cy - ay) * (xs - cx) + (ax - cx) * (ys - cy)) / det
        l3 = 1.0 - l1 - l2
        inside = (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6)
        if not inside.any():
            continue
        u = l1 * uv[a, 0] + l2 * uv[b, 0] + l3 * uv[c, 0]
        v = l1 * uv[a, 1] + l2 * uv[b, 1] + l3 * uv[c, 1]
        fx = u * sw - 0.5
        fy = (1.0 - v) * sh - 0.5
        sx = np.clip(fx.round().astype(int), 0, sw - 1)
        sy = np.clip(fy.round().astype(int), 0, sh - 1)
        texel = source[sy, sx]
        alpha = texel[..., 3] / 255.0 * color[3]
        rgb = texel[..., :3] * col
        if premult:
            rgb = rgb * color[3]
        if border:
            out = (u < 0.0) | (u > 1.0) | (v < 0.0) | (v > 1.0)
            alpha = np.where(out, 0.0, alpha)
            rgb = np.where(out[..., None], 0.0, rgb)
        alpha = alpha[..., None]
        sub = target[y0:y1 + 1, x0:x1 + 1]
        if premult:
            blended = rgb + sub[..., :3] * (1.0 - alpha)
        else:
            blended = rgb * alpha + sub[..., :3] * (1.0 - alpha)
        sub[..., :3][inside] = np.clip(blended[inside], 0.0, 255.0)
    return target


def draw_render_to_mesh(target, mesh, source, color=(1.0, 1.0, 1.0, 1.0), premult=False,
                        border=False):
    """Draws a 'render to' patch mesh into a texture the way BandPatchMesh::Render does:
    its vertex positions ARE the target's coordinates (x, y in -1..1 - Render scales y by
    height / width and the camera by the same, which cancels; +y runs DOWN the texture, as
    the retail makeup quads show: eyes at y ~0, lips at y ~0.4), its UVs sample `source`,
    blended over with the source alpha (makeup.mat: kBlendSrcAlpha) times `color`.
    target / source: top-row-first H x W x 4 arrays (0-255); target is changed in place."""
    h, w = target.shape[:2]
    pos = np.array([v['co'][:2] for v in mesh['verts']], dtype=np.float64)
    uv = np.array([v['uv'] for v in mesh['verts']], dtype=np.float64)
    px = np.c_[(pos[:, 0] + 1.0) * 0.5 * w - 0.5, (pos[:, 1] + 1.0) * 0.5 * h - 0.5]
    return _raster(target, px, uv, mesh['faces'], source, color, premult, border)


def apply_makeup(lib, rgba, mesh_names):
    """Draws the character's makeup patches (BandCharDesc patches of category makeup,
    texture -1: <g>_makeup_eyes_N / _lips_N quads from colorpalettes, their own material
    makeup.mat -> the makeup.tex atlas) onto the head skin texture, in patch order."""
    from .mesh_importer import parse_mesh_entry
    out = rgba
    for name in mesh_names:
        _p, c = lib.find(name, 'Mesh')
        if c is None:
            _log(f"RB3 textures: makeup '{name}' not found")
            continue
        try:
            mesh = parse_mesh_entry(c + MARKER, name)
        except Exception as e:
            _log(f"RB3 textures: makeup '{name}' not read ({e})")
            continue
        color = (1.0, 1.0, 1.0, 1.0)
        atlas_name = 'makeup.tex'
        _pm, cm = lib.find(mesh.get('mat') or 'makeup.mat', 'Mat')
        if cm is not None:
            try:
                ms = parse_mat(cm, mesh.get('mat') or 'makeup.mat')
                color = ms.get('color', color)
                atlas_name = ms.get('diffuse') or atlas_name
            except Exception:
                pass
        atlas = lib.rgba(atlas_name)
        if atlas is None:
            _log(f"RB3 textures: makeup atlas '{atlas_name}' not found")
            continue
        if out is rgba:
            out = rgba.copy()
        draw_render_to_mesh(out, mesh, atlas, color)
    return out


PATCH_MAKEUP = 0x20


def makeup_meshes(patches):
    """The makeup patch meshes of a BandCharDesc's patch list, in order."""
    return [p['mesh'] for p in patches or ()
            if (p.get('category', 0) & PATCH_MAKEUP) and p.get('mesh')]


# ---------------------------------------------------------------------------------------
# Patches: an item's own artwork, tattoos, art and decals (bandobj/BandPatchMesh.cpp)
# ---------------------------------------------------------------------------------------
#
# Every OutfitConfig carries BandPatchMeshes: {src transform, target meshes, render_to,
# category}. At PreRender the game draws, into each target mesh's diffuse output:
#   * src set: the item's own patch - '<target>_patch.mesh', kept in the item (AddPatch
#     finds the permanent one by name), a render-to mesh whose positions are the target
#     texture's coordinates and whose material (blend, colour, border wrap) samples the
#     artwork: the baseball tee's muscle car is baseballtee_resource_patch.mesh /
#     baseballtee_patch.mat -> baseballtee_musclecar_patch.tex.
#   * the character's patches whose category matches: texture -1 maps its placement mesh
#     with that mesh's own material (makeup, face paint); otherwise the art (prefab_artNN
#     of the prefab milo, or the save's art slot) is projected onto the target: FindXfm
#     puts an orthonormal frame on the placement mesh at the patch's UV, its axes half the
#     UV gradient's lengths there; RotateAboutZ(rotation), axes x scale * 0.5 - so the art
#     spans half the placement region per unit of scale (Kim's freckles at scale 1 on
#     head_front come out ~3 inches across nose and cheeks) - and
#     ProjectPatches spreads decal coordinates over the target's surface from that point,
#     carrying the frame along the surface (WorkVerts::AddUV / Normalize). Drawn with
#     premultiplied alpha and a black border (BandPatchMesh::Render's defaults).
# Categories (BandCharDesc::Patch): torso 0x1, leg 0x2, feet 0x4, hair 0x8, tattoo 0x10,
# makeup 0x20, face paint 0x40, torso / leg overlay 0x80 / 0x100, guitar 0x200, bass 0x400,
# drum 0x800, mic 0x1000, keyboard 0x2000. Tattoos, makeup and face paint target the
# head milo's skin.cfg meshes <g>_tattoo_torso / _legs / _head (colorpalettes), which
# share the naked body's UVs - so they land in the shared skin textures.

PATCH_SIZE = 0.5                  # PreRender: frame axes x scale * 0.5
PATCH_INSTRUMENTS = 0x2E00          # GetPatchMesh: placement mesh from the instrument
_MESH_CACHE = {}
_PROJ_CACHE = {}


def _mesh(lib, name):
    """A parsed mesh from the library (cached), or None."""
    from .mesh_importer import parse_mesh_entry
    key = (tuple(lib.paths), name)
    if key not in _MESH_CACHE:
        _p, c = lib.find(name, 'Mesh')
        m = None
        if c is not None:
            try:
                m = parse_mesh_entry(c + MARKER, name)
            except Exception as e:
                _log(f"RB3 textures: mesh '{name}' not read ({e})")
        _MESH_CACHE[key] = m
    return _MESH_CACHE[key]


def find_xfm(place, uv):
    """BandPatchMesh::FindXfm: (origin, dP/du, dP/dv) on the placement mesh at `uv` (game
    UV: v runs down the texture). None if the mesh has no triangle there."""
    P = np.array([v['co'] for v in place['verts']], dtype=np.float64)
    U = np.array([v['uv'] for v in place['verts']], dtype=np.float64)
    U[:, 1] = 1.0 - U[:, 1]                          # back to the game's v
    q = np.array(uv, dtype=np.float64)
    best = None
    for f in place['faces']:
        a, b, c = (int(i) for i in f[:3])
        e1, e2 = U[b] - U[a], U[c] - U[a]
        det = e1[0] * e2[1] - e1[1] * e2[0]
        if abs(det) < 1e-12:
            continue
        d = q - U[a]
        l1 = (d[0] * e2[1] - d[1] * e2[0]) / det
        l2 = (e1[0] * d[1] - e1[1] * d[0]) / det
        out = max(0.0, -l1) + max(0.0, -l2) + max(0.0, l1 + l2 - 1.0)
        if best is None or out < best[0]:
            # P = Pa + (Pb - Pa) * l1 + (Pc - Pa) * l2; gradient from the UV edges
            E = np.array([[e1[0], e2[0]], [e1[1], e2[1]]])
            G = np.c_[P[b] - P[a], P[c] - P[a]] @ np.linalg.inv(E)
            best = (out, P[a] + (P[b] - P[a]) * l1 + (P[c] - P[a]) * l2, G[:, 0], G[:, 1])
            if out == 0.0:
                break
    return None if best is None else best[1:]


def _rot_between(a, b):
    """The smallest rotation taking unit vector a onto unit vector b (3x3)."""
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1.0 + c)


def project_patch(target, origin, ax, ay, limit=1.15):
    """BandPatchMesh::ProjectPatches: decal coordinates (-1..1 across the art) spread over
    `target`'s surface from the point nearest `origin`, the frame (ax, ay: the patch's
    half-extent axes) carried vertex to vertex in the surface's tangent planes. Returns
    (per-vertex decal coords (n x 2, NaN where not reached), faces to draw) or None."""
    import heapq
    P = np.array([v['co'] for v in target['verts']], dtype=np.float64)
    N = np.array([v.get('normal') or (0.0, 0.0, 1.0) for v in target['verts']],
                 dtype=np.float64)
    F = np.array([f[:3] for f in target['faces']], dtype=np.int64)
    if not len(P) or not len(F):
        return None
    # weld the UV-seam twins (same position) so the spread crosses seams
    key = np.round(P, 4)
    _u, weld, inv_count = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    weld = weld.reshape(-1)
    nw = int(weld.max()) + 1
    WP = np.zeros((nw, 3)); WN = np.zeros((nw, 3))
    np.add.at(WP, weld, P); np.add.at(WN, weld, N)
    WP /= np.bincount(weld, minlength=nw)[:, None]
    ln = np.linalg.norm(WN, axis=1, keepdims=True); ln[ln < 1e-9] = 1.0
    WN /= ln
    WF = weld[F]
    nbr = [set() for _ in range(nw)]
    for a, b, c in WF:
        nbr[a].update((b, c)); nbr[b].update((a, c)); nbr[c].update((a, b))

    B = np.c_[ax, ay]                                       # 3 x 2
    M0 = np.linalg.pinv(B)                                  # 2 x 3: offset -> decal coords
    nd = np.cross(ax, ay)
    if np.linalg.norm(nd) < 1e-9:
        return None
    nd /= np.linalg.norm(nd)
    # seed: the nearest vertex facing the patch whose footprint holds it
    rel = WP - origin
    coords = rel @ M0.T
    depth = np.abs(rel @ nd)
    reach = max(np.linalg.norm(ax), np.linalg.norm(ay))
    # placement meshes float a few inches off the body (torso_front ~5): follow the
    # patch's axis in to the surface - facing vertices near the art's centre line, the
    # nearest of them along it
    ok = (WN @ nd > 0.2) & (np.abs(coords).max(1) <= 1.0) & (depth <= 2.0 * reach + 12.0)
    if not ok.any():
        return None
    cand = np.where(ok)[0]
    off = np.linalg.norm(coords[cand], axis=1)
    near = cand[off <= max(0.15, off.min() * 1.5)]
    seed = int(near[np.argmin(depth[near])])
    C = np.full((nw, 2), np.nan)
    Ms = [None] * nw
    C[seed] = coords[seed]
    Ms[seed] = M0 @ _rot_between(nd, WN[seed]).T
    done = np.zeros(nw, bool)
    dist = {seed: 0.0}
    heap = [(0.0, seed)]
    while heap:
        dv, v = heapq.heappop(heap)
        if done[v]:
            continue
        if v != seed:
            # from every finished neighbour: its coords plus the offset in its tangent plane
            cs, mats = [], []
            for n in nbr[v]:
                if not done[n]:
                    continue
                d = WP[v] - WP[n]
                dt = d - np.dot(d, WN[n]) * WN[n]
                l, lt = np.linalg.norm(d), np.linalg.norm(dt)
                if lt > 1e-9:
                    dt *= l / lt
                cs.append(C[n] + Ms[n] @ dt)
                mats.append(Ms[n] @ _rot_between(WN[n], WN[v]).T)
            if not cs:
                continue
            C[v] = np.mean(cs, axis=0)
            Ms[v] = np.mean(mats, axis=0)
        done[v] = True
        if np.abs(C[v]).max() > limit:
            continue                                        # past the art: stop spreading
        for n in nbr[v]:
            if done[n]:
                continue
            nd_ = dv + float(np.linalg.norm(WP[n] - WP[v]))
            if nd_ < dist.get(n, np.inf):
                dist[n] = nd_
                heapq.heappush(heap, (nd_, n))
    VC = C[weld]
    # the spread can run on round an armpit onto the torso: keep what the patch would see
    # along its axis - faces far behind another reached surface at the same spot are cut
    reached = ~np.isnan(VC).any(1)
    cand = [i for i, (a, b, c) in enumerate(F) if reached[a] and reached[b] and reached[c]]
    Q = (P - origin) @ M0.T
    D = (P - origin) @ nd
    res, lo, span = 96, -1.6, 3.2
    zb = np.full((res, res), -np.inf)

    def cells(pts):
        ij = np.floor((pts - lo) / span * res).astype(int)
        ok = (ij >= 0).all(1) & (ij < res).all(1)
        return ij[ok], ok

    for i in cand:
        tri = F[i]
        w = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 0], [0, 1, 1],
                      [1, 0, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2]], dtype=np.float64)
        w /= w.sum(1, keepdims=True)
        pts, dep = w @ Q[tri], w @ D[tri]
        ij, ok = cells(pts)
        np.maximum.at(zb, (ij[:, 1], ij[:, 0]), dep[ok])
    tol = 1.0 + 0.1 * reach
    visible = set()
    for i in cand:
        tri = F[i]
        ij, ok = cells(Q[tri].mean(0)[None])
        if not ok.any() or D[tri].mean() >= zb[ij[0, 1], ij[0, 0]] - tol:
            visible.add(i)
    faces = []
    unit = 1.0 / (np.linalg.norm(ax) * np.linalg.norm(ay))     # decal area per surface area
    for fi, (a, b, c) in enumerate(F):
        if fi not in visible:
            continue
        t = VC[[a, b, c]]
        if np.isnan(t).any():
            continue
        # where the carried frame has collapsed (round an armpit) the art smears: drop
        # triangles whose decal area is far below what their surface area should give
        area3 = 0.5 * np.linalg.norm(np.cross(P[b] - P[a], P[c] - P[a]))
        e1, e2 = t[1] - t[0], t[2] - t[0]
        area2 = 0.5 * abs(e1[0] * e2[1] - e1[1] * e2[0])
        if area3 > 1e-9 and area2 < 0.2 * area3 * unit:
            continue
        # all three past the same edge of the art: nothing to draw
        if (t[:, 0] < -1).all() or (t[:, 0] > 1).all() or (t[:, 1] < -1).all() \
                or (t[:, 1] > 1).all():
            continue
        faces.append((int(a), int(b), int(c)))
    return VC, faces


def _patch_frame(place, patch):
    got = find_xfm(place, patch['uv'])
    if got is None:
        return None
    origin, du, dv = got
    dv = -dv            # FindXfm negates the v axis: y runs UP the art, z out of the body
    # FindXfm (360 @0x82333eb0): an orthonormal frame on the surface, its axes scaled to
    # half the UV gradient's lengths (PreRender @0x82339ad8 then x scale * 0.5)
    lu, lv = np.linalg.norm(du), np.linalg.norm(dv)
    n = np.cross(du, dv)
    if lu < 1e-9 or lv < 1e-9 or np.linalg.norm(n) < 1e-9:
        return None
    xd = du / lu
    n /= np.linalg.norm(n)
    yd = np.cross(n, xd)
    du, dv = xd * lu * 0.5, yd * lv * 0.5
    r = float(patch.get('rotation') or 0.0)
    c, s = np.cos(r), np.sin(r)
    # Matrix3::RotateAboutZ, then the axes scaled by the patch size
    ax = c * du + s * dv
    ay = -s * du + c * dv
    sx, sy = patch.get('scale') or (1.0, 1.0)
    return origin, ax * (sx * PATCH_SIZE), ay * (sy * PATCH_SIZE)


def _projected(lib, place_lib, job):
    key = (tuple(lib.paths), tuple(place_lib.paths), job['target'], job['place'],
           tuple(job['uv']), job['rotation'], tuple(job['scale']))
    if key in _PROJ_CACHE:
        return _PROJ_CACHE[key]
    res = None
    target = _mesh(lib, job['target'])
    place = _mesh(place_lib, job['place'])
    if target is None or place is None:
        _log(f"RB3 textures: patch on '{job['target']}' via '{job['place']}': "
             f"{'target' if target is None else 'placement'} mesh not found")
    else:
        fr = _patch_frame(place, job)
        if fr is not None:
            got = project_patch(target, *fr)
            if got is not None and got[1]:
                VC, faces = got
                tuv = np.array([v['uv'] for v in target['verts']], dtype=np.float64)
                # decal coords (y up the art) -> art UV (u right, v up)
                duv = np.c_[(VC[:, 0] + 1.0) * 0.5, (VC[:, 1] + 1.0) * 0.5]
                res = (tuv, np.nan_to_num(duv), faces)
        if res is None:
            _log(f"RB3 textures: couldn't project onto '{job['target']}' from "
                 f"'{job['place']}' at {tuple(round(x, 2) for x in job['uv'])}")
    _PROJ_CACHE[key] = res
    return res


_ART = {}          # image name -> rgba, for art that came from a save


def register_art(name, rgba):
    _ART[name] = rgba


def _art_rgba(ref):
    """The art for a patch: {'tex': name, 'lib': paths} (prefab_artNN in the prefab milo)
    or {'image': name} (a save's art slot, kept as a packed Blender image)."""
    if not ref:
        return None
    if ref.get('tex'):
        return Library(ref.get('lib') or []).rgba(ref['tex'])
    name = ref.get('image')
    if name in _ART:
        return _ART[name]
    img = bpy.data.images.get(name) if name else None
    if img is None:
        return None
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    rgba = np.flipud(px.reshape(h, w, 4)) * 255.0
    _ART[name] = rgba
    return rgba


def patch_jobs(lib, cfgs, mat_names, char_patches, art, target_filter=None):
    """What to draw into one output texture: for each BandPatchMesh of `cfgs` with a target
    mesh drawn with one of `mat_names` (or accepted by target_filter(mesh name)), its own
    patch and the character's patches of its category. art: {texture index: art ref}."""
    jobs = []
    for cfg in cfgs or ():
        for bpm in cfg.get('patches') or ():
            for tname in bpm['meshes']:
                if target_filter is not None:
                    if not target_filter(tname):
                        continue
                else:
                    tm = _mesh(lib, tname)
                    if tm is None or tm.get('mat') not in mat_names:
                        continue
                if bpm['src']:
                    own = tname.rsplit('.', 1)[0] + '_patch.mesh'
                    jobs.append(dict(kind='own', mesh=own))
                if not bpm['category']:
                    continue
                for p in char_patches or ():
                    if not (p.get('category', 0) & bpm['category']) or not p.get('mesh'):
                        continue
                    if p['category'] & PATCH_MAKEUP:
                        continue                         # drawn by apply_makeup
                    if p.get('texture', -1) < 0:
                        jobs.append(dict(kind='own', mesh=p['mesh']))
                        continue
                    ref = (art or {}).get(p['texture'])
                    if ref is None:
                        continue
                    jobs.append(dict(kind='project', target=tname, place=p['mesh'],
                                     instrument=bool(p['category'] & PATCH_INSTRUMENTS),
                                     uv=list(p['uv']), rotation=p.get('rotation', 0.0),
                                     scale=list(p.get('scale') or (1.0, 1.0)), art=ref))
    return jobs


def apply_patches(lib, rgba, jobs, place_paths=None):
    """Draws the patch jobs over a composited texture."""
    out = rgba
    place_lib = Library(place_paths) if place_paths else lib
    for job in jobs or ():
        try:
            if job['kind'] == 'own':
                mesh = _mesh(lib, job['mesh'])
                if mesh is None:
                    _log(f"RB3 textures: patch mesh '{job['mesh']}' not found")
                    continue
                mname = mesh.get('mat')
                ms = {}
                if mname:
                    _p, cm = lib.find(mname, 'Mat')
                    if cm is not None:
                        ms = parse_mat(cm, mname)
                src = lib.rgba(ms.get('diffuse')) if ms.get('diffuse') else None
                if src is None:
                    _log(f"RB3 textures: patch '{job['mesh']}' has no texture")
                    continue
                if out is rgba:
                    out = rgba.copy()
                draw_render_to_mesh(out, mesh, src, ms.get('color', (1.0, 1.0, 1.0, 1.0)),
                                    premult=ms.get('blend') == ti.BLEND_PREMULT,
                                    border=ms.get('tex_wrap') in (2, 3))
            else:
                src = _art_rgba(job['art'])
                if src is None:
                    _log(f"RB3 textures: art {job['art']} not found")
                    continue
                pl = lib if job.get('instrument') else place_lib
                got = _projected(lib, pl, job)
                if got is None:
                    continue
                tuv, duv, faces = got
                if out is rgba:
                    out = rgba.copy()
                h, w = out.shape[:2]
                px = np.c_[tuv[:, 0] * w - 0.5, (1.0 - tuv[:, 1]) * h - 0.5]
                _raster(out, px, duv, faces, src, premult=True, border=True)
        except Exception as e:
            _log(f"RB3 textures: patch {job.get('mesh') or job.get('place')} failed ({e})")
    return out


_REGION_CACHE = {}


def region_mask(lib, mesh_name, h, w):
    key = (tuple(lib.paths), mesh_name, h, w)
    if key in _REGION_CACHE:
        return _REGION_CACHE[key]
    from .mesh_importer import parse_mesh_entry
    _p, c = lib.find(mesh_name, 'Mesh')
    m = None
    if c is not None:
        try:
            m = rasterize_uv_mask(parse_mesh_entry(c + MARKER, mesh_name), h, w)
        except Exception as e:
            _log(f"RB3 textures: region '{mesh_name}' not drawn ({e})")
    _REGION_CACHE[key] = m
    return m


def head_normal(lib, gender, types, controllers=None):
    """The head normal map norm.texblend produces: <g>_head00_norm with each feature's
    region blended to <g>_head_norm<type + 1>. types: {feature: type index}. controllers:
    [(feature, region mesh)] from the head milo's norm.texblend (default: the usual five)."""
    base = lib.normal(f"{gender}_head00_norm.tex")
    if base is None:
        return None
    out = base.astype(np.float64).copy()
    h, w = out.shape[:2]
    regions = controllers or [(f, f"{gender}_norm_{f}.mesh") for f in HEAD_FEATURES]
    for feat, mesh_name in regions:
        t = int(types.get(feat, 0))
        typed = lib.normal(f"{gender}_head_norm{t + 1:02d}.tex")
        m = region_mask(lib, mesh_name, h, w)
        if typed is None or m is None:
            continue
        typed = _resize(typed, h, w)
        out[..., :3] = out[..., :3] * (1.0 - m[..., None]) + typed[..., :3] * m[..., None]
    # renormalise the blended vectors
    v = out[..., :3] / 127.5 - 1.0
    ln = np.linalg.norm(v, axis=-1, keepdims=True)
    ln[ln < 1e-6] = 1.0
    out[..., :3] = (v / ln + 1.0) * 127.5
    return out


# ---------------------------------------------------------------------------------------
# Blender images
# ---------------------------------------------------------------------------------------

def image_from_rgba(name, rgba, colorspace='sRGB'):
    """A packed Blender image from a top-row-first H x W x 4 (0-255) array, reused by name."""
    h, w = rgba.shape[:2]
    img = bpy.data.images.get(name)
    if img is not None and (tuple(img.size) != (w, h) or img.get("rb3_generated") is None):
        if img.get("rb3_generated") is None:
            name = name + ".rb3"
            img = bpy.data.images.get(name)
        else:
            bpy.data.images.remove(img)
            img = None
    if img is None:
        img = bpy.data.images.new(name, width=w, height=h, alpha=True)
    img.colorspace_settings.name = colorspace
    try:
        img.alpha_mode = 'CHANNEL_PACKED'
    except (AttributeError, TypeError):
        pass
    px = np.ascontiguousarray(np.flipud(rgba) / 255.0, dtype=np.float32)
    img.pixels.foreach_set(px.ravel())
    img.update()
    try:
        img.pack()
    except RuntimeError:
        pass
    img["rb3_generated"] = True
    return img


def texture_image(lib, name, role):
    """A Blender image for a stored texture (ti.build_image keeps one per texture)."""
    t = lib.tex(name)
    if t is None:
        return None
    try:
        return ti.build_image(name, t, role)
    except Exception as e:
        _log(f"RB3 textures: '{name}' -> image failed ({e})")
        return None


# ---------------------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------------------

_STATE = "rb3_tint_state"


_COMPOSITES = {}       # state -> image name, so a texture shared by several materials
                      # (the skin) is drawn once per import


def _composite_image(lib, st):
    """Builds / refreshes the composite image for a material's tint state."""
    key = json.dumps(st, sort_keys=True, default=str)
    name = _COMPOSITES.get(key)
    if name and bpy.data.images.get(name) is not None:
        return bpy.data.images[name]
    colors = st['colors']
    jobs = st.get('patch_jobs')
    if st.get('textures'):
        name = st['textures'][colors[st['opt1']] % len(st['textures'])
                              if 0 <= st['opt1'] < len(colors) else 0]
        if not jobs:
            return texture_image(lib, name, 'diffuse')
        rgba = lib.rgba(name)
        if rgba is None:
            return None
    else:
        diff = lib.rgba(st['diffuse'])
        if diff is None:
            return None
        interp = lib.rgba(st['interp'])
        mask = lib.rgba(st['mask'])
        c1 = colour_for(lib, st['pal1'], st['opt1'], colors)
        c2 = colour_for(lib, st['pal2'], st['opt2'], colors) if st['pal2'] else None
        rgba = compose_two_color(diff, None if interp is None else data_channel(interp),
                                 None if mask is None else data_channel(mask), c1, c2)
    if st.get('makeup'):
        rgba = apply_makeup(lib, rgba, st['makeup'])
    if jobs:
        rgba = apply_patches(lib, rgba, jobs)
    img = image_from_rgba(st['output'], rgba, 'sRGB')
    if img is not None:
        _COMPOSITES[key] = img.name
    return img


def recompose(mat):
    """Recomputes a material's tinted diffuse from its stored state and its colour options
    (the material panel's colour indices)."""
    raw = mat.get(_STATE)
    if not raw:
        return False
    st = json.loads(raw)
    tp = mat.rb3_tint
    st['colors'] = [tp.color0, tp.color1, tp.color2]
    lib = Library(st['lib'])
    if st.get('lens'):
        c = colour_for(lib, st['pal1'], st['opt1'], st['colors'])
        s = getattr(mat, 'gltfmilo_settings', None)
        if c is not None and s is not None:
            a = s.base_color[3]
            s.base_color = (c[0], c[1], c[2], a)
        return True
    img = _composite_image(lib, st)
    if img is None:
        return False
    s = getattr(mat, 'gltfmilo_settings', None)
    if s is not None:
        s.diffuse_tex = img
    node = mat.node_tree.nodes.get("Milo_Diffuse") if mat.node_tree else None
    if node is not None:
        node.image = img
    try:
        from . import rb3_shader
        rb3_shader.sync_rb3_material(mat)
    except Exception:
        pass
    return True


def _swap_for(lib, cfgs, mat_name, mat_slots):
    """The MatSwap that dresses a material: by its own name (as mat or resource_mat), else
    by the render target its diffuse names - the swap material with the same diffuse, or
    failing that the swap whose diffuse shares the render target's first word (the female
    head's eyesmale.mat draws 'eye_diffuse_output', eyes.cfg composites 'eyes_diffuse')."""
    for cfg in cfgs:
        for s in cfg['swaps']:
            if mat_name in (s['mat'], s['resource']):
                return cfg, s
    out = (mat_slots or {}).get('diffuse', '') or ''
    if not out.endswith('_output.tex'):
        return None, None
    for cfg in cfgs:
        for s in cfg['swaps']:
            _p, c = lib.find(s['mat'], 'Mat')
            if c is None:
                continue
            try:
                if parse_mat(c, s['mat']).get('diffuse') == out:
                    return cfg, s
            except Exception:
                pass
    word = out.split('_')[0]
    for cfg in cfgs:
        for s in cfg['swaps']:
            d = (s['diffuse'] or '').split('_')[0]
            if d and (d.startswith(word) or word.startswith(d)):
                return cfg, s
    return None, None


def dress_material(mat, lib, cfgs, piece_colors, gender, skin=None, output_prefix=""):
    """Gives one imported material its textures and tint. skin: dict with 'skin_color',
    'normal_suffix', 'head_types', 'skin_cfg', 'head_controllers' for the skin materials.
    Returns a short note for the log."""
    game = mat.get("milo_material") or mat.name
    _p, c = lib.find(game, 'Mat')
    slots = None
    if c is not None:
        try:
            slots = parse_mat(c, game)
        except Exception as e:
            _log(f"RB3 textures: '{game}' not read ({e})")
    cfg, swap = _swap_for(lib, cfgs, game, slots)
    colors = list(piece_colors)
    is_skin = skin is not None and game in SKIN_MATS
    if is_skin:
        idx, part = SKIN_MATS[game]
        scfg = skin.get('skin_cfg')
        if scfg and idx < len(scfg['swaps']):
            cfg, swap = scfg, scfg['swaps'][idx]
        colors = [skin['skin_color'], 0, 0]
    elif swap is not None and swap['mat'] != game:
        # the swap material (in the item) has the textures; the resource one is a stub
        _p2, c2 = lib.find(swap['mat'], 'Mat')
        if c2 is not None:
            try:
                slots = parse_mat(c2, swap['mat'])
            except Exception as e:
                _log(f"RB3 textures: '{swap['mat']}' not read ({e})")
    if slots is None:
        return f"{game}: no material data"
    if cfg is not None and not is_skin:
        colors = [c if c is not None and c >= 0 else d
                  for c, d in zip((list(colors) + [None] * 3)[:3], cfg['colors'])]

    images = {}
    note = []
    # normal / specular / detail
    if is_skin:
        _i, part = SKIN_MATS[game]
        if part == 'head':
            nrm = head_normal(lib, gender, skin.get('head_types') or {},
                              skin.get('head_controllers'))
            if nrm is not None:
                images['normal'] = image_from_rgba(f"{gender}_head_rb3_normal", nrm,
                                                   'Non-Color')
                note.append("head normal composite")
        else:
            suf = skin.get('normal_suffix') or ''
            name = f"{gender}_{part}_norm{'_' + suf if suf else ''}.tex"
            images['normal'] = texture_image(lib, name, 'normal') or \
                texture_image(lib, f"{gender}_{part}_norm.tex", 'normal')
            note.append(name)
        images['specular'] = texture_image(lib, f"{gender}_{part}_spec.tex", 'specular')
        slots = dict(slots)
        slots['normal'] = slots.get('normal') or 'skin'
    else:
        images['normal'] = texture_image(lib, slots.get('normal'), 'normal')
        images['specular'] = texture_image(lib, slots.get('specular'), 'specular')
    if slots.get('detail_normal'):
        images['detail'] = texture_image(lib, slots['detail_normal'], 'normal')

    # diffuse
    state = None
    diff_name = slots.get('diffuse') or ''
    if swap is not None and (swap['diffuse'] or swap['textures']):
        state = dict(lib=lib.paths, diffuse=swap['diffuse'], interp=swap['interp'],
                     mask=swap['mask'], pal1=swap['pal1'], opt1=swap['opt1'],
                     pal2=swap['pal2'], opt2=swap['opt2'], textures=swap['textures'],
                     output=f"{output_prefix}{(swap['diffuse'] or game).rsplit('.', 1)[0]}"
                            f"_rb3_output", colors=colors)
        if is_skin and '_head_' in (swap['diffuse'] or ''):
            state['makeup'] = list(skin.get('makeup') or [])
        if is_skin:
            jobs = _skin_jobs(lib, skin, SKIN_MATS[game][1])
        else:
            jobs = patch_jobs(lib, cfgs, {game, swap['mat'], swap['resource']},
                              (skin or {}).get('patches'), (skin or {}).get('art'))
        if jobs:
            state['patch_jobs'] = jobs
            note.append(f"{len(jobs)} patch(es)")
        images['diffuse'] = _composite_image(lib, state)
        note.append(f"tinted ({swap['pal1'] or '-'} / {swap['pal2'] or '-'}, colours "
                    f"{colors})")
    elif swap is not None and swap['pal1']:
        state = dict(lib=lib.paths, pal1=swap['pal1'], opt1=swap['opt1'], lens=True,
                     colors=colors)
        c = colour_for(lib, swap['pal1'], swap['opt1'], colors)
        if c is not None:
            a = slots.get('color', (1, 1, 1, 1))[3]
            slots = dict(slots)
            slots['color'] = (c[0], c[1], c[2], a)
        note.append(f"colour from {swap['pal1']}")
    elif skin is not None and diff_name in SKIN_OUTPUTS:
        part = diff_name.split('_')[0]
        sc = skin.get('skin_cfg')
        sw = None
        if sc:
            for s in sc['swaps']:
                if f"_{part}_" in (s['diffuse'] or ''):
                    sw = s
        if sw is not None:
            state = dict(lib=lib.paths, diffuse=sw['diffuse'], interp=sw['interp'],
                         mask=sw['mask'], pal1=sw['pal1'], opt1=sw['opt1'], pal2=sw['pal2'],
                         opt2=sw['opt2'], textures=[],
                         output=f"{(sw['diffuse']).rsplit('.', 1)[0]}_rb3_output",
                         colors=[skin['skin_color'], 0, 0])
            if '_head_' in (sw['diffuse'] or ''):
                state['makeup'] = list(skin.get('makeup') or [])
            jobs = _skin_jobs(lib, skin, part)
            if jobs:
                state['patch_jobs'] = jobs
                note.append(f"{len(jobs)} patch(es)")
            images['diffuse'] = _composite_image(lib, state)
            note.append("skin")
    else:
        jobs = patch_jobs(lib, cfgs, {game}, (skin or {}).get('patches'),
                          (skin or {}).get('art')) if diff_name else []
        if jobs:
            state = dict(lib=lib.paths, textures=[diff_name], opt1=0, colors=[0, 0, 0],
                         pal1='', pal2='', opt2=0, diffuse='', interp='', mask='',
                         output=f"{output_prefix}{diff_name.rsplit('.', 1)[0]}_rb3_output",
                         patch_jobs=jobs)
            images['diffuse'] = _composite_image(lib, state)
            note.append(f"{len(jobs)} patch(es)")
        else:
            images['diffuse'] = texture_image(lib, diff_name, 'diffuse')

    ti.fill_material_settings(mat, slots, images)
    try:
        ti.build_material_tree(mat, slots, images)
    except Exception as e:
        _log(f"RB3 textures: '{mat.name}' node tree not built ({e})")
    if is_skin:
        mat["rb3_skin"] = json.dumps(dict(part=SKIN_MATS[game][1], lib=lib.paths,
                                          controllers=skin.get('head_controllers')))
    if state is not None:
        mat[_STATE] = json.dumps(state)
        tp = mat.rb3_tint
        cols = (list(state['colors']) + [0, 0, 0])[:3]
        tp["color0"], tp["color1"], tp["color2"] = (int(cols[0]), int(cols[1]),
                                                     int(cols[2]))
    return f"{game}: " + (", ".join(note) if note else "textures")


def _skin_jobs(lib, skin, part):
    """Tattoos / face paint for one skin part: skin.cfg's patch meshes for that part's
    tattoo mesh (<g>_tattoo_<part>.mesh, the skin texture's UVs)."""
    scfg = skin.get('skin_cfg')
    if not scfg or not skin.get('patches'):
        return []
    suffix = f"_tattoo_{part}.mesh"
    return patch_jobs(lib, [scfg], set(), skin.get('patches'), skin.get('art'),
                      target_filter=lambda n: n.endswith(suffix))


def item_configs(path):
    out = []
    got = _index(path)
    if not got:
        return out
    body, idx = got
    for n, (t, s, e) in idx.items():
        if t == 'OutfitConfig':
            try:
                out.append(parse_outfit_config(normalize_header(body[s:e])))
            except Exception as ex:
                _log(f"RB3 textures: {os.path.basename(path)} {n} not read ({ex})")
    return out


def head_controllers(head_path, gender):
    """[(feature, region mesh)] from the head milo's norm.texblend controllers."""
    out = []
    got = _index(head_path) if head_path else None
    if not got:
        return None
    body, idx = got
    for f in HEAD_FEATURES:
        hit = idx.get(f"norm_{f}.texblendctl")
        if hit:
            c = body[hit[1]:hit[2]]
            mesh = f"{gender}_norm_{f}.mesh"
            try:
                n = struct.unpack_from('>I', c, 17)[0]
                cand = c[21:21 + n].decode('latin-1')
                if cand.endswith('.mesh'):
                    mesh = cand
            except (struct.error, UnicodeDecodeError):
                pass
            out.append((f, mesh))
    return out or None


# ---------------------------------------------------------------------------------------
# Live updates from the proportions / face panels
# ---------------------------------------------------------------------------------------

def _skin_materials(arm_obj):
    mats = set()
    for o in bpy.data.objects:
        if o.type != 'MESH' or not any(m.type == 'ARMATURE' and m.object is arm_obj
                                       for m in o.modifiers):
            continue
        for s in o.material_slots:
            if s.material is not None and s.material.get("rb3_skin"):
                mats.add(s.material)
    return mats


def update_body_normals(arm_obj, gender, suffix):
    """Swaps the torso / legs / feet normal maps for a new body type."""
    if arm_obj.get("rb3_norm_suffix") == suffix:
        return 0
    n = 0
    for mat in _skin_materials(arm_obj):
        info = json.loads(mat["rb3_skin"])
        part = info['part']
        if part == 'head':
            continue
        lib = Library(info['lib'])
        name = f"{gender}_{part}_norm{'_' + suffix if suffix else ''}.tex"
        img = texture_image(lib, name, 'normal') or texture_image(
            lib, f"{gender}_{part}_norm.tex", 'normal')
        if img is None:
            continue
        _set_role(mat, 'normal', img)
        n += 1
    arm_obj["rb3_norm_suffix"] = suffix
    return n


def update_head_normal(arm_obj, gender, types):
    key = json.dumps(types, sort_keys=True)
    if arm_obj.get("rb3_head_norm_types") == key:
        return 0
    n = 0
    for mat in _skin_materials(arm_obj):
        info = json.loads(mat["rb3_skin"])
        if info['part'] != 'head':
            continue
        lib = Library(info['lib'])
        nrm = head_normal(lib, gender, types, info.get('controllers'))
        if nrm is None:
            continue
        img = image_from_rgba(f"{gender}_head_rb3_normal", nrm, 'Non-Color')
        _set_role(mat, 'normal', img)
        n += 1
    arm_obj["rb3_head_norm_types"] = key
    return n


def _set_role(mat, role, img):
    s = getattr(mat, 'gltfmilo_settings', None)
    if s is not None:
        setattr(s, f"{role}_tex", img)
    node = mat.node_tree.nodes.get("Milo_Normal" if role == 'normal' else "Milo_Diffuse") \
        if mat.node_tree else None
    if node is not None:
        node.image = img
    try:
        from . import rb3_shader
        rb3_shader.sync_rb3_material(mat)
    except Exception:
        pass


# ---------------------------------------------------------------------------------------
# Material panel
# ---------------------------------------------------------------------------------------

def _update(self, context):
    mat = self.id_data
    if isinstance(mat, bpy.types.Material):
        recompose(mat)


class RB3TintSettings(bpy.types.PropertyGroup):
    color0: IntProperty(name="Colour 1", min=0, max=255, default=0, update=_update,
                        description="The piece's first colour index (palette entry)")
    color1: IntProperty(name="Colour 2", min=0, max=255, default=1, update=_update,
                        description="The piece's second colour index (palette entry)")
    color2: IntProperty(name="Colour 3", min=0, max=255, default=2, update=_update,
                        description="The piece's third colour index (palette entry)")


class MATERIAL_PT_rb3_tint(bpy.types.Panel):
    bl_label = "Rock Band 3 Colours"
    bl_idname = "MATERIAL_PT_rb3_tint"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "material"

    @classmethod
    def poll(cls, context):
        m = context.material
        return m is not None and bool(m.get(_STATE))

    def draw(self, context):
        layout = self.layout
        mat = context.material
        st = json.loads(mat[_STATE])
        lib = Library(st['lib'])
        tp = mat.rb3_tint
        cols = [tp.color0, tp.color1, tp.color2]
        used = sorted({o for o, p in ((st.get('opt1'), st.get('pal1')),
                                      (st.get('opt2'), st.get('pal2'))) if p and o is not None})
        if not used and st.get('textures'):
            used = [st.get('opt1', 0)]
        for o in used:
            if not 0 <= o < 3:
                continue
            row = layout.row(align=True)
            row.prop(tp, f"color{o}")
            for p, opt in ((st.get('pal1'), st.get('opt1')), (st.get('pal2'), st.get('opt2'))):
                if p and opt == o:
                    pal = lib.palette(p)
                    if pal:
                        c = pal[cols[o] % len(pal)]
                        row.label(text=f"{p} [{cols[o] % len(pal)}/{len(pal)}]  "
                                       f"#{int(c[0]*255):02X}{int(c[1]*255):02X}"
                                       f"{int(c[2]*255):02X}")
        if st.get('textures'):
            layout.label(text=f"{len(st['textures'])} texture options")


_classes = (RB3TintSettings, MATERIAL_PT_rb3_tint)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Material.rb3_tint = PointerProperty(type=RB3TintSettings)


def unregister():
    del bpy.types.Material.rb3_tint
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
