"""
Rock Band 3 preview shader - Rock Band 3's material shading, lit by Blender.

===========================================================================================
WHERE THIS COMES FROM
===========================================================================================
Rebuilt from Rock Band 3's own Xbox 360 shaders: the compiled `standard` shader variants
in the game's shader cache (xbox_shaders), disassembled from Xenos microcode and checked
formula by formula against an emulation of the microcode, plus the code in default.xex
that picks a variant for a material, uploads its constants and builds the environment's
colour transform.

===========================================================================================
HOW IT'S PUT TOGETHER
===========================================================================================
In the game a character is lit by a six-colour ambient cube built from the venue's lights.
Here Blender's own lighting (an HDRI, lamps) takes that role, as with the Milo Character
Shader, through ordinary BSDFs - no baking. What's reproduced from the game is everything
the MATERIAL contributes: the colour of each lighting term, computed in the game's display
(gamma) space and converted to linear for Blender at the end.

    Normal map   x' = 2R - 1 (tangent), y' = 2G - 1 (binormal) on the plugin's decoded maps;
                 z = sat(1 - x'^2 - y'^2), no square root; the detail map adds strength x its
                 own decoded vector; de-normal scales x/y by (1 - deNormal). Skin's specular
                 uses the full-strength normal, everything else the softened one.
    Diffuse      tex x matColor x AO [x2 with intensify]                 -> Diffuse BSDF
    Ambient      the same x the environment's ambient colour             -> emission
    Specular     specRGB [x map.rgb] x F x AO, power p = max(power [x map.a], 0.5)
                 F = 0.25 + (1 - sat(n.v)) x (0.5 + 0.5 n_z); skin: ((1 - sat(n.v)) x
                 (0.5 + 0.5 n_z))^2; hair: specular colour treated as white -> Glossy BSDF
                 The game's lobe is (R.L)^p in display values, i.e. (R.L)^(2.2 p) in linear,
                 normalised by (p + 2)/(2 pi); the Glossy's roughness and colour are matched
                 to that lobe.
    Rim          rimRGB x ((1 - sat(n.v)) x (0.5 + 0.5 n_z))^rimPower x AO, lit by light
                 from behind the surface (the game reads its cube along -view) - a Diffuse
                 BSDF facing away from the camera. The game also boosts the diffuse by 30%
                 of the rim; that cross term is folded in as 0.3 x albedo.
    Env map      the reflection, faded by (1 - sat(n.v)) with falloff and by the specular
                 map's alpha with specmask, x matColor x AO              -> mirror Glossy
    Glow         emissive map x emissive multiplier                      -> emission
    Colour adj.  with the material's Color Adjust on and the environment's colour adjust
                 enabled, the environment's 3x4 colour transform (hue, saturation,
                 lightness, contrast, brightness, levels) is applied to every term's colour;
                 its offset is added as emission. Exact under neutral light - the game
                 applies it to the lit result.
    Alpha        matColor.a x ambient alpha x tex.a.

Per-pixel lighting off drops the normal map, specular map, detail map and rim (the game
falls back to vertex lighting). Without Use Environment the game picks an unlit variant:
texture x colour, no specular - drawn here as emission.

===========================================================================================
WHAT ISN'T REPRODUCED
===========================================================================================
The game's light rig and its cube approximation (Blender lights the surfaces instead),
point-light-only features (specular2, anisotropy, point-light rim), pre-lit vertex
lighting, fog, bloom, and the per-vertex spherical-harmonic occlusion RB3 keeps in vertex
colours (AO is 1 here).
"""

import math

import bpy
import numpy as np

RB3_GROUP_NAME = "Milo RB3 Shader"
RB3_GROUP_VERSION = 2
ENV_GROUP_NAME = "Milo RB3 Environment"
ENV_GROUP_VERSION = 1
ENCODE_GROUP_NAME = "Milo sRGB Encode"
DECODE_GROUP_NAME = "Milo sRGB Decode"
HELPER_GROUP_VERSION = 1
# Groups from the first (baked ambient cube) version, removed once nothing uses them.
OBSOLETE_GROUPS = ("Milo RB3 Lighting", "Milo RB3 Cube Lookup")

NODE_PREFIX = "MiloRB3_"
SHADER_NODE_NAME = NODE_PREFIX + "Shader"
MAT_KEY = "milo_rb3_shader"
PREV_SURFACE_KEY = "milo_rb3_prev_surface"

# The game works on display values; Blender's lighting on linear ones. A display-space
# product a x b is the linear product decode(a) x decode(b) under a power-law gamma, so a
# display-space lobe (R.L)^p becomes (R.L)^(GAMMA p) in linear.
GAMMA = 2.2

NON_COLOR_SPACES = {'Non-Color', 'Raw', 'Linear', 'Linear Rec.709', 'Linear CIE-XYZ D65',
                    'Linear FilmLight E-Gamut', 'ACEScg', 'ACES2065-1', 'XYZ', 'Utility - Raw'}


# ---------------------------------------------------------------------------------------
# Small node-building helper
# ---------------------------------------------------------------------------------------

class _Builder:
    """Wraps a node tree with terse maths helpers. Every helper returns an output socket;
    arguments may be sockets or constants (floats, or 3-tuples for vectors)."""

    def __init__(self, nt, x0=-2000, dx=180):
        self.nt = nt
        self.nodes = nt.nodes
        self.links = nt.links
        self._x = x0
        self._dx = dx
        self._y = 0

    def node(self, kind, **attrs):
        n = self.nodes.new(kind)
        n.location = (self._x, self._y)
        self._y -= 160
        if self._y < -3000:
            self._y = 0
            self._x += self._dx
        for k, v in attrs.items():
            setattr(n, k, v)
        return n

    def _set(self, sock, v, vector=False):
        if v is None:
            return
        if isinstance(v, (int, float)):
            sock.default_value = (float(v),) * 3 if vector else float(v)
        elif isinstance(v, tuple):
            sock.default_value = v
        else:
            self.links.new(v, sock)

    def math(self, op, a, b=None, c=None, clamp=False):
        n = self.node('ShaderNodeMath', operation=op, use_clamp=clamp)
        for i, v in enumerate((a, b, c)):
            self._set(n.inputs[i], v)
        return n.outputs[0]

    def vmath(self, op, a, b=None, c=None, scale=None):
        n = self.node('ShaderNodeVectorMath', operation=op)
        for i, v in enumerate((a, b, c)):
            self._set(n.inputs[i], v, vector=True)
        if scale is not None:
            self._set(n.inputs[3], scale)
        return n.outputs[1] if op in ('DOT_PRODUCT', 'LENGTH', 'DISTANCE') else n.outputs[0]

    def sep(self, v):
        n = self.node('ShaderNodeSeparateXYZ')
        self._set(n.inputs[0], v, vector=True)
        return n.outputs[0], n.outputs[1], n.outputs[2]

    def comb(self, x, y, z):
        n = self.node('ShaderNodeCombineXYZ')
        for i, v in enumerate((x, y, z)):
            self._set(n.inputs[i], v)
        return n.outputs[0]

    def scale(self, v, s):
        return self.vmath('SCALE', v, scale=s)

    def add(self, a, b):
        return self.vmath('ADD', a, b)

    def mul(self, a, b):
        return self.vmath('MULTIPLY', a, b)

    def lerp(self, a, b, t):
        """a + (b - a) x t for floats."""
        return self.math('MULTIPLY_ADD', self.math('SUBTRACT', b, a), t, a)

    def vlerp(self, a, b, t):
        """a + (b - a) x t for vectors (t a float socket or constant)."""
        return self.add(a, self.scale(self.vmath('SUBTRACT', b, a), t))

    def group(self, tree, **inputs):
        n = self.node('ShaderNodeGroup')
        n.node_tree = tree
        for name, v in inputs.items():
            self._set(n.inputs[name], v, vector=isinstance(v, tuple) and len(v) == 3)
        return n


def _new_group(name):
    ng = bpy.data.node_groups.get(name)
    if ng is None:
        ng = bpy.data.node_groups.new(name, 'ShaderNodeTree')
    ng.interface.clear()
    ng.nodes.clear()
    return ng


def _socket(ng, name, in_out, kind, default=None, lo=None, hi=None):
    it = ng.interface.new_socket(name=name, in_out=in_out, socket_type=kind)
    if default is not None and hasattr(it, 'default_value'):
        it.default_value = default
    if lo is not None and hasattr(it, 'min_value'):
        it.min_value, it.max_value = lo, hi
    return it


def _io(ng):
    gin = ng.nodes.new('NodeGroupInput')
    gin.location = (-2400, 0)
    gout = ng.nodes.new('NodeGroupOutput')
    gout.location = (1600, 0)
    return gin, gout


def _current(name, version):
    ng = bpy.data.node_groups.get(name)
    return ng is not None and ng.get("milo_group_version") == version


# ---------------------------------------------------------------------------------------
# Helper groups
# ---------------------------------------------------------------------------------------

def _build_srgb(name, encode):
    """Exact sRGB transfer, per channel: encode (linear -> display) or decode."""
    ng = _new_group(name)
    _socket(ng, "Color", 'INPUT', 'NodeSocketColor', (0.0, 0.0, 0.0, 1.0))
    _socket(ng, "Color", 'OUTPUT', 'NodeSocketColor')
    gin, gout = _io(ng)
    b = _Builder(ng)
    chans = []
    for c in b.sep(gin.outputs[0]):
        c = b.math('MAXIMUM', c, 0.0)
        if encode:
            low = b.math('MULTIPLY', c, 12.92)
            high = b.math('SUBTRACT', b.math('MULTIPLY', b.math('POWER', c, 1.0 / 2.4), 1.055),
                          0.055)
            is_low = b.math('LESS_THAN', c, 0.0031308)
        else:
            low = b.math('DIVIDE', c, 12.92)
            high = b.math('POWER', b.math('DIVIDE', b.math('ADD', c, 0.055), 1.055), 2.4)
            is_low = b.math('LESS_THAN', c, 0.04045)
        chans.append(b.lerp(high, low, is_low))
    ng.links.new(b.comb(*chans), gout.inputs[0])
    ng["milo_group_version"] = HELPER_GROUP_VERSION
    return ng


def ensure_encode_group():
    if _current(ENCODE_GROUP_NAME, HELPER_GROUP_VERSION):
        return bpy.data.node_groups[ENCODE_GROUP_NAME]
    return _build_srgb(ENCODE_GROUP_NAME, True)


def ensure_decode_group():
    if _current(DECODE_GROUP_NAME, HELPER_GROUP_VERSION):
        return bpy.data.node_groups[DECODE_GROUP_NAME]
    return _build_srgb(DECODE_GROUP_NAME, False)


def ensure_decode_signed_group():
    """decode() that keeps a negative value's sign: sign(c) x decode(|c|). For the colour
    transform's offset, which darkens when negative."""
    name = DECODE_GROUP_NAME + " Signed"
    if _current(name, HELPER_GROUP_VERSION):
        return bpy.data.node_groups[name]
    ng = _new_group(name)
    _socket(ng, "Color", 'INPUT', 'NodeSocketVector', (0.0, 0.0, 0.0))
    _socket(ng, "Color", 'OUTPUT', 'NodeSocketVector')
    gin, gout = _io(ng)
    b = _Builder(ng)
    dec = ensure_decode_group()
    pos = b.group(dec, Color=b.vmath('MAXIMUM', gin.outputs[0], 0.0)).outputs[0]
    neg = b.group(dec, Color=b.vmath('MAXIMUM', b.scale(gin.outputs[0], -1.0), 0.0)).outputs[0]
    ng.links.new(b.vmath('SUBTRACT', pos, neg), gout.inputs[0])
    ng["milo_group_version"] = HELPER_GROUP_VERSION
    return ng


# ---------------------------------------------------------------------------------------
# Shared environment group: ambient colour and the colour transform (RndEnviron)
# ---------------------------------------------------------------------------------------

_IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def ensure_environment_group():
    """One group every RB3 material reads - the scene-wide part of the game's RndEnviron:
    its ambient colour and colour transform. push_environment() writes its values."""
    if _current(ENV_GROUP_NAME, ENV_GROUP_VERSION):
        return bpy.data.node_groups[ENV_GROUP_NAME]
    ng = _new_group(ENV_GROUP_NAME)
    for name in ("Ambient", "Row R", "Row G", "Row B", "Offset"):
        _socket(ng, name, 'OUTPUT', 'NodeSocketVector')
    _socket(ng, "Ambient Alpha", 'OUTPUT', 'NodeSocketFloat')
    _socket(ng, "Color Adjust On", 'OUTPUT', 'NodeSocketFloat')
    gin, gout = _io(ng)
    b = _Builder(ng)
    defaults = {"Ambient": (0.0, 0.0, 0.0), "Row R": _IDENTITY[0], "Row G": _IDENTITY[1],
                "Row B": _IDENTITY[2], "Offset": (0.0, 0.0, 0.0)}
    for name, val in defaults.items():
        n = b.node('ShaderNodeCombineXYZ')
        n.name = n.label = name
        for i in range(3):
            n.inputs[i].default_value = val[i]
        ng.links.new(n.outputs[0], gout.inputs[name])
    for name, val in (("Ambient Alpha", 1.0), ("Color Adjust On", 0.0)):
        n = b.node('ShaderNodeValue')
        n.name = n.label = name
        n.outputs[0].default_value = val
        ng.links.new(n.outputs[0], gout.inputs[name])
    ng["milo_group_version"] = ENV_GROUP_VERSION
    return ng


def push_environment(ambient=None, ambient_alpha=None, color_xfm=None, color_adjust_on=None):
    """Writes the scene's environment into the shared group. color_xfm: (A, t) from
    build_color_xfm."""
    ng = ensure_environment_group()
    nodes = ng.nodes

    def vec(name, v):
        n = nodes[name]
        for i in range(3):
            n.inputs[i].default_value = float(v[i])
    if ambient is not None:
        vec("Ambient", ambient)
    if ambient_alpha is not None:
        nodes["Ambient Alpha"].outputs[0].default_value = float(ambient_alpha)
    if color_xfm is not None:
        a, t = color_xfm
        for i, row in enumerate(("Row R", "Row G", "Row B")):
            vec(row, a[i])
        vec("Offset", t)
    if color_adjust_on is not None:
        nodes["Color Adjust On"].outputs[0].default_value = 1.0 if color_adjust_on else 0.0


def build_color_xfm(hue=0.0, saturation=0.0, lightness=0.0, contrast=0.0, brightness=0.0,
                    in_lo=(0.0, 0.0, 0.0), in_hi=(1.0, 1.0, 1.0),
                    out_lo=(0.0, 0.0, 0.0), out_hi=(1.0, 1.0, 1.0)):
    """The environment's colour transform as RB3 builds it (ColorXfm::AdjustColorXfm in
    default.xex): identity, then hue, saturation, lightness, contrast, brightness and levels
    applied in that order. Hue is in degrees, levels are colours, the rest are percentages.
    Returns (A, t) with colour' = A colour + t, on display values."""
    def step(state, m, off):
        a, t = state
        return m @ a, m @ t + np.asarray(off, dtype=np.float64)

    state = (np.eye(3), np.zeros(3))

    # Hue: blends each channel towards the next in 120-degree segments, with the cosine and
    # sine of the position within the segment (x 90 degrees) - so in-between hues brighten
    # by up to sqrt(2), as in the game.
    h = float(hue)
    if h != 0.0:
        m = np.zeros((3, 3))
        if h > 0.0:
            seg = h >= 120.0
            ang = math.radians(((h - 120.0) if seg else h) / 120.0 * 90.0)
        else:
            seg = h <= -120.0
            ang = math.radians(((-h - 120.0) if seg else -h) / 120.0 * 90.0)
        c, s = math.cos(ang), math.sin(ang)
        for i in range(3):
            i1, i2 = (i + 1) % 3, (i + 2) % 3
            if h > 0.0 and not seg:
                m[i, i], m[i, i1] = c, s
            elif h > 0.0:
                m[i, i1], m[i, i2] = c, s
            elif not seg:
                m[i, i], m[i, i2] = c, s
            else:
                m[i, i1], m[i, i2] = s, c
        state = step(state, m, (0.0, 0.0, 0.0))

    # Saturation: diagonal k, off-diagonal (1 - k)/2 (equal channel weights).
    x = float(saturation) * 0.01
    k = 1.0 + x if x > 0.0 else 1.0 + x * (2.0 / 3.0)
    m = np.full((3, 3), (1.0 - k) * 0.5)
    np.fill_diagonal(m, k)
    state = step(state, m, (0.0, 0.0, 0.0))

    # Lightness: towards white when positive, towards black when negative.
    x = float(lightness) * 0.01
    if x < 0.0:
        state = step(state, np.eye(3) * (1.0 + x), (0.0, 0.0, 0.0))
    else:
        state = step(state, np.eye(3) * (1.0 - x), (x, x, x))

    # Contrast: around mid-grey.
    x = float(contrast) * 0.01
    k = 1.0 / (1.0 - 0.9921875 * x) if x > 0.0 else 1.0 + 0.992126 * x
    off = (1.0 - k) * 0.5
    state = step(state, np.eye(3) * k, (off, off, off))

    # Brightness: an offset of brightness/200.
    x = float(brightness) * 0.005
    state = step(state, np.eye(3), (x, x, x))

    # Levels, per channel: out_lo + (c - in_lo) x (out_hi - out_lo)/(in_hi - in_lo).
    sc = np.zeros(3)
    off = np.zeros(3)
    for i in range(3):
        span = float(in_hi[i]) - float(in_lo[i])
        sc[i] = (float(out_hi[i]) - float(out_lo[i])) / span if span != 0.0 else 0.0
        off[i] = float(out_lo[i]) - float(in_lo[i]) * sc[i]
    state = step(state, np.diag(sc), off)
    return state


# ---------------------------------------------------------------------------------------
# The material group
# ---------------------------------------------------------------------------------------

# (name, socket type, default, min, max)
_RB3_INPUTS = (
    ("Diffuse", 'NodeSocketColor', (1.0, 1.0, 1.0, 1.0), None, None),
    ("Diffuse Alpha", 'NodeSocketFloat', 1.0, 0.0, 1.0),
    ("Material Color", 'NodeSocketColor', (1.0, 1.0, 1.0, 1.0), None, None),
    ("Material Alpha", 'NodeSocketFloat', 1.0, 0.0, 1.0),
    ("Intensify", 'NodeSocketFloat', 0.0, 0.0, 1.0),
    ("Lit", 'NodeSocketFloat', 1.0, 0.0, 1.0),
    ("Per Pixel", 'NodeSocketFloat', 1.0, 0.0, 1.0),
    ("Normal Map", 'NodeSocketColor', (0.5, 0.5, 1.0, 1.0), None, None),
    ("Detail Normal", 'NodeSocketColor', (0.5, 0.5, 1.0, 1.0), None, None),
    ("Detail Strength", 'NodeSocketFloat', 0.0, 0.0, 10.0),
    ("Normal Scale", 'NodeSocketFloat', 1.0, 0.0, 2.0),
    ("Specular On", 'NodeSocketFloat', 0.0, 0.0, 1.0),
    ("Specular Color", 'NodeSocketColor', (0.0, 0.0, 0.0, 1.0), None, None),
    ("Specular Power", 'NodeSocketFloat', 10.0, 0.0, 10000.0),
    ("Specular Map", 'NodeSocketColor', (1.0, 1.0, 1.0, 1.0), None, None),
    ("Specular Map Alpha", 'NodeSocketFloat', 1.0, 0.0, 1.0),
    ("Has Specular Map", 'NodeSocketFloat', 0.0, 0.0, 1.0),
    ("Rim On", 'NodeSocketFloat', 0.0, 0.0, 1.0),
    ("Rim Color", 'NodeSocketColor', (0.0, 0.0, 0.0, 1.0), None, None),
    ("Rim Power", 'NodeSocketFloat', 4.0, 0.0, 1000.0),
    ("Emissive", 'NodeSocketColor', (0.0, 0.0, 0.0, 1.0), None, None),
    ("Emissive Multiplier", 'NodeSocketFloat', 1.0, 0.0, 100.0),
    ("Environment Map", 'NodeSocketFloat', 0.0, 0.0, 1.0),
    ("Env Falloff", 'NodeSocketFloat', 0.0, 0.0, 1.0),
    ("Env Specmask", 'NodeSocketFloat', 0.0, 0.0, 1.0),
    ("Skin", 'NodeSocketFloat', 0.0, 0.0, 1.0),
    ("Hair", 'NodeSocketFloat', 0.0, 0.0, 1.0),
    ("Color Adjust", 'NodeSocketFloat', 0.0, 0.0, 1.0),
    ("AO", 'NodeSocketFloat', 1.0, 0.0, 1.0),
    ("Use Alpha", 'NodeSocketFloat', 0.0, 0.0, 1.0),
    ("Alpha Cut", 'NodeSocketFloat', 0.0, 0.0, 1.0),
    ("Alpha Threshold", 'NodeSocketFloat', 0.0, 0.0, 1.0),
    ("Additive", 'NodeSocketFloat', 0.0, 0.0, 1.0),
)

# Node labels the tests (and anyone inspecting the group) can find the closures by.
DIFFUSE_LABEL = "RB3 Diffuse"
RIM_LABEL = "RB3 Rim"
SPECULAR_LABEL = "RB3 Specular"
REFLECTION_LABEL = "RB3 Environment Map"
EMISSION_LABEL = "RB3 Ambient + Glow"


def ensure_rb3_group():
    if _current(RB3_GROUP_NAME, RB3_GROUP_VERSION):
        return bpy.data.node_groups[RB3_GROUP_NAME]
    ng = _new_group(RB3_GROUP_NAME)
    for name, kind, default, lo, hi in _RB3_INPUTS:
        _socket(ng, name, 'INPUT', kind, default, lo, hi)
    _socket(ng, "Shader", 'OUTPUT', 'NodeSocketShader')
    _socket(ng, "Alpha", 'OUTPUT', 'NodeSocketFloat')
    gin, gout = _io(ng)
    I = gin.outputs
    b = _Builder(ng)
    env = b.group(ensure_environment_group())
    dec = ensure_decode_group()

    geo = b.node('ShaderNodeNewGeometry')
    view = geo.outputs["Incoming"]                    # unit vector towards the eye
    lit = I["Lit"]
    ao = I["AO"]
    pp = I["Per Pixel"]

    # ---- normals ----
    nm = b.vmath('MULTIPLY_ADD', I["Normal Map"], 2.0, -1.0)
    nm = b.vlerp((0.0, 0.0, 1.0), nm, pp)             # vertex lighting ignores the map
    x, y, _z = b.sep(nm)
    zb = b.math('SUBTRACT', 1.0, b.math('ADD', b.math('MULTIPLY', x, x),
                                        b.math('MULTIPLY', y, y)), clamp=True)
    dn = b.vmath('MULTIPLY_ADD', I["Detail Normal"], 2.0, -1.0)
    dx, dy, _dz = b.sep(dn)
    zd = b.math('SUBTRACT', 1.0, b.math('ADD', b.math('MULTIPLY', dx, dx),
                                        b.math('MULTIPLY', dy, dy)), clamp=True)
    ds = b.math('MULTIPLY', I["Detail Strength"], pp)
    X = b.math('MULTIPLY_ADD', dx, ds, x)
    Y = b.math('MULTIPLY_ADD', dy, ds, y)
    Z = b.math('MULTIPLY_ADD', zd, ds, zb)
    k = I["Normal Scale"]

    def to_world(tx, ty, tz, label):
        enc = b.vmath('MULTIPLY_ADD', b.comb(tx, ty, tz), 0.5, 0.5)
        n = b.node('ShaderNodeNormalMap', space='TANGENT')
        n.label = label
        ng.links.new(enc, n.inputs["Color"])
        return n.outputs["Normal"]

    n_soft = to_world(b.math('MULTIPLY', X, k), b.math('MULTIPLY', Y, k), Z, "Normal (De-Normal)")
    n_full = to_world(X, Y, Z, "Normal (Full)")
    n_spec = b.vmath('NORMALIZE', b.vlerp(n_soft, n_full, I["Skin"]))

    # ---- shared factors (display values) ----
    vn = b.math('MAXIMUM', b.math('MINIMUM', b.vmath('DOT_PRODUCT', view, n_spec), 1.0), 0.0)
    _nx, _ny, nz = b.sep(n_spec)
    edge = b.math('MULTIPLY', b.math('SUBTRACT', 1.0, vn),
                  b.math('MULTIPLY_ADD', nz, 0.5, 0.5))
    boost = b.math('MULTIPLY', ao, b.math('ADD', I["Intensify"], 1.0))
    albedo = b.mul(I["Diffuse"], I["Material Color"])
    has_sm = b.math('MULTIPLY', I["Has Specular Map"], pp)

    # Colour transform: applied to each term's display colour, when enabled.
    adj_on = b.math('MULTIPLY', I["Color Adjust"], env.outputs["Color Adjust On"])

    def adjust(col):
        rows = [b.vmath('DOT_PRODUCT', env.outputs[r], col) for r in ("Row R", "Row G", "Row B")]
        return b.vlerp(col, b.comb(*rows), adj_on)

    def linear(col):
        """Display colour -> the linear colour a BSDF or emission takes."""
        return b.group(dec, Color=adjust(col)).outputs[0]

    # ---- diffuse ----
    diff_disp = b.scale(albedo, b.math('MULTIPLY', boost, lit))
    diffuse = b.node('ShaderNodeBsdfDiffuse')
    diffuse.label = DIFFUSE_LABEL
    ng.links.new(linear(diff_disp), diffuse.inputs["Color"])
    ng.links.new(n_soft, diffuse.inputs["Normal"])

    # ---- ambient (the environment's ambient colour; white when unlit) + glow ----
    ambient = b.vlerp((1.0, 1.0, 1.0), env.outputs["Ambient"], lit)
    amb_disp = b.scale(b.mul(albedo, ambient), boost)
    glow_disp = b.scale(I["Emissive"], I["Emissive Multiplier"])
    offset = b.group(ensure_decode_signed_group(),
                     Color=b.scale(env.outputs["Offset"], adj_on)).outputs[0]
    emit_col = b.add(b.add(linear(amb_disp), linear(glow_disp)), offset)
    emission = b.node('ShaderNodeEmission')
    emission.label = EMISSION_LABEL
    ng.links.new(emit_col, emission.inputs["Color"])
    emission.inputs["Strength"].default_value = 1.0

    # ---- specular ----
    sm_alpha = b.lerp(1.0, I["Specular Map Alpha"], has_sm)
    power = b.math('MAXIMUM', b.math('MULTIPLY', I["Specular Power"], sm_alpha), 0.5)
    spec_col = b.vlerp(I["Specular Color"], (1.0, 1.0, 1.0), I["Hair"])
    spec_col = b.mul(spec_col, b.vlerp((1.0, 1.0, 1.0), I["Specular Map"], has_sm))
    fres = b.lerp(b.math('ADD', edge, 0.25), b.math('MULTIPLY', edge, edge), I["Skin"])
    spec_disp = b.scale(spec_col, b.math('MULTIPLY', b.math('MULTIPLY', fres, ao),
                                         b.math('MULTIPLY', I["Specular On"], lit)))
    # Lobe: (R.L)^p on display values is (R.L)^q, q = GAMMA p, on linear ones. Phong
    # exponent q ~ Blinn 4q; Blinn n -> Beckmann/GGX alpha^2 = 2/(n + 2); Blender's
    # roughness is sqrt(alpha). Colour: the game's peak (p + 2)/(2 pi), raised to GAMMA,
    # over the peak of a normalised lobe of exponent q.
    q = b.math('MULTIPLY', power, GAMMA)
    rough = b.math('POWER', b.math('DIVIDE', 2.0, b.math('MULTIPLY_ADD', q, 4.0, 2.0)), 0.25)
    peak_game = b.math('POWER', b.math('MULTIPLY_ADD', power, 1.0 / (2.0 * math.pi),
                                       1.0 / math.pi), GAMMA)
    peak_lobe = b.math('MULTIPLY_ADD', q, 1.0 / (2.0 * math.pi), 1.0 / math.pi)
    spec_lin = b.scale(linear(spec_disp), b.math('DIVIDE', peak_game, peak_lobe))
    specular = b.node('ShaderNodeBsdfGlossy')
    specular.label = SPECULAR_LABEL
    ng.links.new(spec_lin, specular.inputs["Color"])
    ng.links.new(rough, specular.inputs["Roughness"])
    ng.links.new(n_spec, specular.inputs["Normal"])

    # ---- rim: lit from behind the surface ----
    rim_on = b.math('MULTIPLY', b.math('MULTIPLY', I["Rim On"], pp), lit)
    rim_f = b.math('POWER', edge, b.math('MAXIMUM', I["Rim Power"], 0.5))
    rim_k = b.math('MULTIPLY', b.math('MULTIPLY', rim_f, ao), rim_on)
    rim_mix = b.vmath('MULTIPLY_ADD', b.scale(albedo, boost), 0.3, 0.7)
    rim_disp = b.scale(b.mul(I["Rim Color"], rim_mix), rim_k)
    rim = b.node('ShaderNodeBsdfDiffuse')
    rim.label = RIM_LABEL
    ng.links.new(linear(rim_disp), rim.inputs["Color"])
    ng.links.new(b.scale(view, -1.0), rim.inputs["Normal"])

    # ---- environment map: a mirror reflection of the scene ----
    env_f = b.math('MULTIPLY', I["Environment Map"], b.lerp(1.0, b.math('SUBTRACT', 1.0, vn),
                                                            I["Env Falloff"]))
    env_f = b.math('MULTIPLY', env_f, b.lerp(1.0, I["Specular Map Alpha"],
                                             b.math('MULTIPLY', I["Env Specmask"], has_sm)))
    env_disp = b.scale(I["Material Color"], b.math('MULTIPLY', b.math('MULTIPLY', env_f, boost),
                                                   lit))
    reflection = b.node('ShaderNodeBsdfGlossy')
    reflection.label = REFLECTION_LABEL
    ng.links.new(linear(env_disp), reflection.inputs["Color"])
    reflection.inputs["Roughness"].default_value = 0.0
    ng.links.new(n_spec, reflection.inputs["Normal"])

    surface = None
    for sh in (diffuse, rim, specular, reflection, emission):
        if surface is None:
            surface = sh.outputs[0]
            continue
        add = b.node('ShaderNodeAddShader')
        ng.links.new(surface, add.inputs[0])
        ng.links.new(sh.outputs[0], add.inputs[1])
        surface = add.outputs[0]

    # ---- alpha ----
    amb_alpha = b.lerp(1.0, env.outputs["Ambient Alpha"], lit)
    a = b.math('MULTIPLY', b.math('MULTIPLY', I["Diffuse Alpha"], I["Material Alpha"]),
               amb_alpha)
    cut = b.math('GREATER_THAN', a, I["Alpha Threshold"])
    a_cut = b.lerp(a, cut, I["Alpha Cut"])
    use = b.math('MAXIMUM', I["Use Alpha"], I["Alpha Cut"])
    alpha = b.lerp(1.0, a_cut, use)
    ng.links.new(alpha, gout.inputs["Alpha"])

    transp = b.node('ShaderNodeBsdfTransparent')
    blended = b.node('ShaderNodeMixShader')
    ng.links.new(alpha, blended.inputs[0])
    ng.links.new(transp.outputs[0], blended.inputs[1])
    ng.links.new(surface, blended.inputs[2])
    # Additive: the surface on top of what's behind, scaled by alpha.
    add_sh = b.node('ShaderNodeAddShader')
    ng.links.new(transp.outputs[0], add_sh.inputs[0])
    ng.links.new(blended.outputs[0], add_sh.inputs[1])
    final = b.node('ShaderNodeMixShader')
    ng.links.new(I["Additive"], final.inputs[0])
    ng.links.new(blended.outputs[0], final.inputs[1])
    ng.links.new(add_sh.outputs[0], final.inputs[2])
    ng.links.new(final.outputs[0], gout.inputs["Shader"])
    ng["milo_group_version"] = RB3_GROUP_VERSION
    ng["milo_shader_version"] = RB3_GROUP_VERSION     # lets Auto-Detect Textures find it
    for old in OBSOLETE_GROUPS:
        g = bpy.data.node_groups.get(old)
        if g is not None and getattr(g, 'users', 0) == 0:
            bpy.data.node_groups.remove(g)
    return ng


# ---------------------------------------------------------------------------------------
# Material settings -> shader
# ---------------------------------------------------------------------------------------

def effective_rb3_lighting(pre_lit, use_environment):
    """(use_environ, prelit) as an RB3 material is written.

    RB3's rev-68 material stores use_environ BEFORE prelit (RndMat::Load in default.xex);
    the writer used to store them the other way round, so the panel's long-standing
    defaults - Pre-Lit on, Use Environment off - have always reached the game as
    use_environ on / prelit off: the lit setup every retail RB3 character material uses.
    That pair is kept meaning exactly that (also covering materials imported before the
    reader was fixed, which show the same pair); any other choice is written as labelled."""
    if pre_lit and not use_environment:
        return True, False
    return bool(use_environment), bool(pre_lit)


def _material_output(nt):
    outs = [n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL']
    return next((n for n in outs if getattr(n, 'is_active_output', False)), None) \
        or (outs[0] if outs else None)


def _is_color_image(img):
    try:
        return img.colorspace_settings.name not in NON_COLOR_SPACES
    except AttributeError:
        return True


def _set_render_method(mat, method):
    from .texture_importer import _set_render_method as srm
    srm(mat, method)


def _image_slot(mat, role, img, x, y, color_target=None, alpha_target=None, vector_from=None):
    """Creates/updates the image node for one texture role and wires its raw (display)
    value into the shader group. Colour images go through the sRGB encoder so the shader
    sees the stored byte values, as the game does."""
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    tex_name = NODE_PREFIX + role
    enc_name = NODE_PREFIX + role + "Encode"
    tex = nodes.get(tex_name)
    enc = nodes.get(enc_name)
    if img is None:
        for n in (tex, enc):
            if n is not None:
                nodes.remove(n)
        return False
    if tex is None:
        tex = nodes.new('ShaderNodeTexImage')
        tex.name = tex_name
        tex.label = f"RB3 {role}"
        tex.location = (x, y)
    tex.image = img
    if vector_from is not None:
        links.new(vector_from, tex.inputs["Vector"])
    if _is_color_image(img):
        if enc is None:
            enc = nodes.new('ShaderNodeGroup')
            enc.name = enc_name
            enc.label = "sRGB Encode"
            enc.location = (x + 300, y)
        enc.node_tree = ensure_encode_group()
        links.new(tex.outputs["Color"], enc.inputs[0])
        src = enc.outputs[0]
    else:
        if enc is not None:
            nodes.remove(enc)
        src = tex.outputs["Color"]
    if color_target is not None:
        links.new(src, color_target)
    if alpha_target is not None:
        links.new(tex.outputs["Alpha"], alpha_target)
    return True


def _unlink(nt, sock):
    for l in list(sock.links):
        nt.links.remove(l)


def sync_rb3_material(mat):
    """Pushes the material's glTFMilo settings into its RB3 shader nodes. No-op unless the
    material has the RB3 shader enabled."""
    if not mat.get(MAT_KEY) or mat.node_tree is None:
        return
    st = getattr(mat, 'gltfmilo_settings', None)
    if st is None:
        return
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    grp = nodes.get(SHADER_NODE_NAME)
    if grp is None or grp.node_tree is None:
        return
    tree = ensure_rb3_group()
    if grp.node_tree is not tree:
        grp.node_tree = tree
    gi = grp.inputs

    def put(name, v):
        sock = gi.get(name)
        if sock is not None:
            sock.default_value = v

    use_env, _prelit = effective_rb3_lighting(st.pre_lit, st.use_environment)
    per_pixel = bool(getattr(st, 'per_pixel_lit', True))
    col = tuple(st.base_color)
    put("Material Color", (col[0], col[1], col[2], 1.0))
    put("Material Alpha", float(col[3]))
    put("Intensify", 1.0 if st.intensify else 0.0)
    put("Lit", 1.0 if use_env else 0.0)
    put("Per Pixel", 1.0 if per_pixel else 0.0)
    put("Normal Scale", 1.0 - float(st.de_normal))
    spec = tuple(st.specular_rgb)
    put("Specular On", 1.0 if max(spec) > 0.0 else 0.0)
    put("Specular Color", (*spec, 1.0))
    put("Specular Power", float(st.specular_power))
    rim = tuple(st.rim_rgb)
    put("Rim On", 1.0 if max(rim) > 0.0 else 0.0)
    put("Rim Color", (*rim, 1.0))
    put("Rim Power", float(st.rim_power))
    put("Emissive Multiplier", float(st.emissive_multiplier))
    put("Environment Map", 1.0 if st.environment_map else 0.0)
    put("Env Falloff", 1.0 if getattr(st, 'environment_map_falloff', False) else 0.0)
    put("Env Specmask", 1.0 if getattr(st, 'environment_map_specmask', False) else 0.0)
    put("Skin", 1.0 if st.shader_variation == 'kShaderVariationSkin' else 0.0)
    put("Hair", 1.0 if st.shader_variation == 'kShaderVariationHair' else 0.0)
    put("Color Adjust", 1.0 if getattr(st, 'color_adjust', False) else 0.0)

    # ---- textures ----
    x0, y0 = grp.location.x - 700, grp.location.y
    for name in ("Diffuse", "Diffuse Alpha", "Normal Map", "Specular Map",
                 "Specular Map Alpha", "Emissive", "Detail Normal"):
        _unlink(nt, gi[name])
    put("Diffuse", (1.0, 1.0, 1.0, 1.0))
    put("Diffuse Alpha", 1.0)
    put("Normal Map", (0.5, 0.5, 1.0, 1.0))
    put("Specular Map", (1.0, 1.0, 1.0, 1.0))
    put("Specular Map Alpha", 1.0)
    put("Emissive", (0.0, 0.0, 0.0, 1.0))
    put("Detail Normal", (0.5, 0.5, 1.0, 1.0))
    _image_slot(mat, "Diffuse", st.diffuse_tex, x0, y0 + 300,
                gi["Diffuse"], gi["Diffuse Alpha"])
    _image_slot(mat, "Normal", st.normal_tex, x0, y0, gi["Normal Map"])
    has_sm = _image_slot(mat, "Specular", st.specular_tex, x0, y0 - 300,
                         gi["Specular Map"], gi["Specular Map Alpha"])
    put("Has Specular Map", 1.0 if has_sm else 0.0)
    _image_slot(mat, "Emissive", st.emissive_tex, x0, y0 - 600, gi["Emissive"])

    detail_img = bpy.data.images.get(mat.get("milo_rb3_detail_image", "") or "")
    uv = nodes.get(NODE_PREFIX + "DetailUV")
    mp = nodes.get(NODE_PREFIX + "DetailMapping")
    if detail_img is not None:
        if uv is None:
            uv = nodes.new('ShaderNodeTexCoord')
            uv.name = NODE_PREFIX + "DetailUV"
            uv.location = (x0 - 600, y0 - 900)
        if mp is None:
            mp = nodes.new('ShaderNodeMapping')
            mp.name = NODE_PREFIX + "DetailMapping"
            mp.location = (x0 - 400, y0 - 900)
        t = float(mat.get("milo_normal_detail_tiling", 1.0) or 1.0)
        mp.inputs['Scale'].default_value = (t, t, 1.0)
        links.new(uv.outputs['UV'], mp.inputs['Vector'])
        _image_slot(mat, "DetailNormal", detail_img, x0, y0 - 900, gi["Detail Normal"],
                    vector_from=mp.outputs['Vector'])
        put("Detail Strength", float(st.normal_detail_strength))
    else:
        for n in (uv, mp, nodes.get(NODE_PREFIX + "DetailNormal")):
            if n is not None:
                nodes.remove(n)
        put("Detail Strength", 0.0)

    # ---- alpha / blending / culling ----
    blend = st.blend
    use_alpha = blend in ('kBlendSrcAlpha', 'kPreMultAlpha', 'kBlendSrcAlphaAdd')
    additive = blend in ('kBlendAdd', 'kBlendSrcAlphaAdd')
    put("Use Alpha", 1.0 if use_alpha else 0.0)
    put("Alpha Cut", 1.0 if st.alpha_cut else 0.0)
    put("Alpha Threshold", min(1.0, st.alpha_threshold / 255.0))
    put("Additive", 1.0 if additive else 0.0)
    if st.alpha_cut and not additive:
        _set_render_method(mat, 'CLIP')
    elif use_alpha or additive:
        _set_render_method(mat, 'BLEND')
    else:
        _set_render_method(mat, 'OPAQUE')
    mat.use_backface_culling = bool(st.cull)


def enable_rb3_shader(mat):
    """Switches a material's surface to the RB3 shader. Existing nodes are left in place
    (only the Material Output's Surface link moves), so disabling restores them."""
    if mat.node_tree is None:
        try:
            mat.use_nodes = True
        except AttributeError:
            pass
    nt = mat.node_tree
    if nt is None:
        return False
    nodes, links = nt.nodes, nt.links
    out = _material_output(nt)
    if out is None:
        out = nodes.new('ShaderNodeOutputMaterial')
        out.location = (600, 300)
    surf = out.inputs['Surface']
    grp = nodes.get(SHADER_NODE_NAME)
    if not mat.get(MAT_KEY):
        prev = surf.links[0] if surf.links else None
        mat[PREV_SURFACE_KEY] = ([prev.from_node.name, prev.from_socket.identifier]
                                 if prev is not None and prev.from_node.name != SHADER_NODE_NAME
                                 else [])
        # An imported material's detail map lives on its Milo Character Shader tree.
        det = nodes.get("Milo_DetailNormal")
        if det is not None and getattr(det, 'image', None) is not None:
            mat["milo_rb3_detail_image"] = det.image.name
    if grp is None:
        grp = nodes.new('ShaderNodeGroup')
        grp.name = SHADER_NODE_NAME
        grp.location = (out.location.x - 400, out.location.y - 500)
        grp.width = 260
    tree = ensure_rb3_group()
    if grp.node_tree is not tree:
        grp.node_tree = tree
    grp.label = RB3_GROUP_NAME
    links.new(grp.outputs["Shader"], surf)
    mat[MAT_KEY] = True
    sync_rb3_material(mat)
    return True


def disable_rb3_shader(mat):
    nt = mat.node_tree
    mat[MAT_KEY] = False
    if nt is None:
        return
    nodes, links = nt.nodes, nt.links
    for n in [n for n in nodes if n.name.startswith(NODE_PREFIX)]:
        nodes.remove(n)
    prev = list(mat.get(PREV_SURFACE_KEY, []) or [])
    out = _material_output(nt)
    if out is not None and len(prev) == 2:
        node = nodes.get(prev[0])
        if node is not None:
            sock = next((s for s in node.outputs if s.identifier == prev[1]), None)
            if sock is not None:
                links.new(sock, out.inputs['Surface'])


# ---------------------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------------------

def srgb_encode(x):
    x = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def srgb_decode(x):
    x = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))
