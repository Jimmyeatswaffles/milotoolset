"""
Rock Band 2 fur (shell fur, with a flat fallback).

Some items carry a fur layer: furbikini_cheetah adds furbikini_fur.mesh (481 vertices)
over the bikini cups, drawn with torso_furbikini_fur.mat - alpha-blended, diffuse
cheetah_fur.tex, and a pointer to a Fur object. RB2's RndFur (field names from the RB2
symbol dump, doc/rb2_dump/rockband2/system/src/rndobj/Fur.cpp; the load order from RB3's
Fur.cpp, whose Wii build reads the object and throws it away):

    revision, 13-byte object header,
    int NumPasses, float Thickness, float Curvature,
    [revision > 1] float ShellOut, float AlphaFalloff,
    float Stretch, float Slide, float Gravity, float Fluidity,
    Color BaseTint, Color EndTint, symbol FurDetail (texture), float FurTiling,
    [revision > 2] symbol Wind

furbikini_cheetah's Fur.fur (revision 2): 6 passes, thickness 0.5, shell-out 0.5, alpha
fall-off 0.5, gravity 0.5, curvature 0, base tint grey 0.49 -> end tint white, no detail
texture - the strand mask is the fur material's own diffuse alpha (fine noise, 97-255).

That is classic shell fur: the mesh drawn NumPasses times, each pass pushed further out
along its normals, keeping only the stronger strands of the noise and fading toward the
tips, tinted from BaseTint at the root to EndTint at the tip. Here a Geometry Nodes
modifier (after the Armature modifier, so the shells follow the body) makes the passes
and stores each one's 0..1 layer as 'rb2_fur_layer'; the material reads it. The motion
fields (Stretch, Slide, Fluidity) have no still equivalent and are left out; Gravity
droops the outer passes.

Flat: the same mesh drawn once - the root pass, tinted half-way between the two tints -
for anything that can't use the shells.
"""

import struct

import bpy

from .utilities import _log


GROUP_NAME = "RB2 Fur Shells"
GROUP_VERSION = 2
MODIFIER_NAME = "RB2 Fur"
LAYER_ATTR = "rb2_fur_layer"
FUR_KEY = "rb2_fur"


class FurError(Exception):
    pass


def parse_fur(data):
    """The Fur object's fields as a dict."""
    rev = struct.unpack_from('>I', data, 0)[0]
    if not 1 <= rev <= 3:
        raise FurError(f"Fur revision {rev} isn't handled")
    o = 17
    out = dict(revision=rev)
    out['passes'] = struct.unpack_from('>i', data, o)[0]; o += 4
    out['thickness'], out['curvature'] = struct.unpack_from('>2f', data, o); o += 8
    if rev > 1:
        out['shell_out'], out['alpha_falloff'] = struct.unpack_from('>2f', data, o); o += 8
    else:
        out['shell_out'], out['alpha_falloff'] = 0.5, 0.5
    (out['stretch'], out['slide'], out['gravity'],
     out['fluidity']) = struct.unpack_from('>4f', data, o); o += 16
    out['base_tint'] = list(struct.unpack_from('>4f', data, o)); o += 16
    out['end_tint'] = list(struct.unpack_from('>4f', data, o)); o += 16
    n = struct.unpack_from('>I', data, o)[0]
    out['detail'] = data[o + 4:o + 4 + n].decode('latin-1'); o += 4 + n
    out['tiling'] = struct.unpack_from('>f', data, o)[0]
    if not 1 <= out['passes'] <= 64:
        raise FurError(f"implausible pass count {out['passes']}")
    return out


# ---------------------------------------------------------------------------------------
# Geometry Nodes: the passes
# ---------------------------------------------------------------------------------------

def _socket(ng, name, kind, in_out='INPUT', default=None, lo=None, hi=None):
    it = ng.interface.new_socket(name=name, in_out=in_out, socket_type=kind)
    if default is not None:
        it.default_value = default
    if lo is not None:
        it.min_value = lo
    if hi is not None:
        it.max_value = hi
    return it


def _group_name(fur):
    return (f"{GROUP_NAME} {int(fur['passes'])}x {float(fur['thickness']):g} "
            f"g{float(fur['gravity']):g}")


def ensure_shell_group(fur):
    """Passes 2..N as face duplicates pushed out along the normal by thickness x layer and
    down by thickness x gravity x layer^2; every pass stores its layer (root 0, tip 1).

    The shell count, thickness and gravity are value nodes INSIDE the group (labelled, so
    they can be edited there), one group per setting: Blender 5 no longer takes
    modifier['Socket_N'] = value for a Geometry Nodes modifier's inputs, so the group
    needs none beyond its geometry."""
    name = _group_name(fur)
    ng = bpy.data.node_groups.get(name)
    if ng is not None and ng.get("rb2_version") == GROUP_VERSION:
        return ng
    if ng is None:
        ng = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    ng.interface.clear()
    ng.nodes.clear()
    _socket(ng, "Geometry", 'NodeSocketGeometry')
    _socket(ng, "Geometry", 'NodeSocketGeometry', in_out='OUTPUT')

    N, L = ng.nodes, ng.links
    gin = N.new('NodeGroupInput'); gin.location = (-1000, 0)
    gout = N.new('NodeGroupOutput'); gout.location = (900, 0)

    def value(label, v, y):
        n = N.new('ShaderNodeValue'); n.label = label; n.name = "RB2Fur_" + label
        n.location = (-1000, y)
        n.outputs[0].default_value = float(v)
        return n
    shells_v = value("Shells", int(fur['passes']), -250)
    thick_v = value("Thickness", fur['thickness'], -400)
    grav_v = value("Gravity", fur['gravity'], -550)

    def math(op, x, y, a=None, b=None):
        n = N.new('ShaderNodeMath'); n.operation = op; n.location = (x, y)
        if a is not None:
            n.inputs[0].default_value = a
        if b is not None:
            n.inputs[1].default_value = b
        return n

    # root pass: layer 0
    root = N.new('GeometryNodeStoreNamedAttribute'); root.location = (-600, 250)
    root.data_type = 'FLOAT'; root.domain = 'POINT'
    root.inputs['Name'].default_value = LAYER_ATTR
    root.inputs['Value'].default_value = 0.0
    L.new(gin.outputs['Geometry'], root.inputs['Geometry'])

    # duplicates: amount = shells - 1
    amount = math('SUBTRACT', -800, -150, b=1.0)
    L.new(shells_v.outputs[0], amount.inputs[0])
    dup = N.new('GeometryNodeDuplicateElements'); dup.location = (-600, -100)
    dup.domain = 'FACE'
    L.new(gin.outputs['Geometry'], dup.inputs['Geometry'])
    L.new(amount.outputs[0], dup.inputs['Amount'])

    # layer = (index + 1) / (shells - 1)
    idx1 = math('ADD', -400, -300, b=1.0)
    L.new(dup.outputs['Duplicate Index'], idx1.inputs[0])
    layer = math('DIVIDE', -250, -300)
    L.new(idx1.outputs[0], layer.inputs[0])
    L.new(amount.outputs[0], layer.inputs[1])

    store = N.new('GeometryNodeStoreNamedAttribute'); store.location = (-100, -100)
    store.data_type = 'FLOAT'; store.domain = 'POINT'
    store.inputs['Name'].default_value = LAYER_ATTR
    L.new(dup.outputs['Geometry'], store.inputs['Geometry'])
    L.new(layer.outputs[0], store.inputs['Value'])

    # offset = normal * thickness * layer + (0, 0, -thickness * gravity * layer^2)
    normal = N.new('GeometryNodeInputNormal'); normal.location = (-100, -400)
    out_len = math('MULTIPLY', -100, -550)
    L.new(thick_v.outputs[0], out_len.inputs[0])
    L.new(layer.outputs[0], out_len.inputs[1])
    push = N.new('ShaderNodeVectorMath'); push.operation = 'SCALE'; push.location = (100, -400)
    L.new(normal.outputs['Normal'], push.inputs[0])
    L.new(out_len.outputs[0], push.inputs['Scale'])
    sq = math('MULTIPLY', -100, -700)
    L.new(layer.outputs[0], sq.inputs[0]); L.new(layer.outputs[0], sq.inputs[1])
    droop = math('MULTIPLY', 100, -700)
    L.new(sq.outputs[0], droop.inputs[0]); L.new(thick_v.outputs[0], droop.inputs[1])
    droop2 = math('MULTIPLY', 250, -700)
    L.new(droop.outputs[0], droop2.inputs[0]); L.new(grav_v.outputs[0], droop2.inputs[1])
    neg = math('MULTIPLY', 400, -700, b=-1.0)
    L.new(droop2.outputs[0], neg.inputs[0])
    down = N.new('ShaderNodeCombineXYZ'); down.location = (550, -700)
    L.new(neg.outputs[0], down.inputs['Z'])
    total = N.new('ShaderNodeVectorMath'); total.operation = 'ADD'; total.location = (550, -400)
    L.new(push.outputs['Vector'], total.inputs[0]); L.new(down.outputs['Vector'], total.inputs[1])

    setpos = N.new('GeometryNodeSetPosition'); setpos.location = (350, -100)
    L.new(store.outputs['Geometry'], setpos.inputs['Geometry'])
    L.new(total.outputs['Vector'], setpos.inputs['Offset'])

    join = N.new('GeometryNodeJoinGeometry'); join.location = (650, 0)
    L.new(root.outputs['Geometry'], join.inputs['Geometry'])
    L.new(setpos.outputs['Geometry'], join.inputs['Geometry'])
    L.new(join.outputs['Geometry'], gout.inputs['Geometry'])
    ng["rb2_version"] = GROUP_VERSION
    return ng


def set_shells(obj, fur, enabled):
    """Adds or updates the shell modifier (enabled) or turns it off (flat)."""
    mod = obj.modifiers.get(MODIFIER_NAME)
    if enabled:
        if mod is None:
            mod = obj.modifiers.new(MODIFIER_NAME, 'NODES')
        ng = mod.node_group
        if ng is None or ng.get("rb2_version") != GROUP_VERSION \
                or ng.name != _group_name(fur):
            mod.node_group = ensure_shell_group(fur)     # also replaces a stale group
        mod.show_viewport = mod.show_render = True
        # after the Armature modifier, so the passes follow the posed, deformed body
        if obj.modifiers.find(MODIFIER_NAME) < len(obj.modifiers) - 1:
            _move_last(obj, MODIFIER_NAME)
    elif mod is not None:
        mod.show_viewport = mod.show_render = False


def _move_last(obj, name):
    try:
        obj.modifiers.move(obj.modifiers.find(name), len(obj.modifiers) - 1)
    except (AttributeError, RuntimeError, TypeError):
        pass


# ---------------------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------------------

def _srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def setup_fur_material(mat, fur, shells):
    """Adds the fur shading to a dressed material: the palette colour (the two-colour
    group's output) x a root-to-tip tint, and an alpha that keeps the stronger strands of
    the diffuse alpha's noise on the outer passes, fading toward the tips."""
    from .rb2_creator_textures import GROUP_NODE_NAME, wire_diffuse_alpha
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    for n in [n for n in nodes if n.name.startswith("RB2Fur_")]:
        nodes.remove(n)
    grp = nodes.get(GROUP_NODE_NAME)
    bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
    diff = nodes.get("RB2CC_Diffuse")
    if grp is None or bsdf is None or diff is None:
        _log(f"    '{mat.name}': fur needs a dressed material with a diffuse - skipped")
        return False
    # The diffuse alpha is the strand mask here - coverage only, never a keep mask.
    mat[FUR_KEY] = True
    wire_diffuse_alpha(mat, False)

    def node(t, name, x, y):
        n = nodes.new(t); n.name = "RB2Fur_" + name; n.location = (x, y)
        return n

    attr = node('ShaderNodeAttribute', "Layer", -300, -500)
    attr.attribute_type = 'GEOMETRY'
    attr.attribute_name = LAYER_ATTR
    flat = node('ShaderNodeValue', "Flat", -300, -650)
    flat.label = "Fur flat (1) / shells (0)"
    flat.outputs[0].default_value = 0.0 if shells else 1.0

    # layer used for the alpha (flat: the root pass) and for the tint (flat: half way)
    lay_a = node('ShaderNodeMix', "LayerAlpha", -100, -500)
    lay_a.data_type = 'FLOAT'
    links.new(flat.outputs[0], lay_a.inputs[0])
    links.new(attr.outputs['Fac'], lay_a.inputs[2])
    lay_a.inputs[3].default_value = 0.0
    lay_t = node('ShaderNodeMix', "LayerTint", -100, -700)
    lay_t.data_type = 'FLOAT'
    links.new(flat.outputs[0], lay_t.inputs[0])
    links.new(attr.outputs['Fac'], lay_t.inputs[2])
    lay_t.inputs[3].default_value = 0.5

    # tint: base -> end, times the palette colour
    tint = node('ShaderNodeMix', "Tint", 100, -700)
    tint.data_type = 'RGBA'
    links.new(lay_t.outputs[0], tint.inputs[0])
    tint.inputs[6].default_value = (*[_srgb_to_linear(c) for c in fur['base_tint'][:3]], 1.0)
    tint.inputs[7].default_value = (*[_srgb_to_linear(c) for c in fur['end_tint'][:3]], 1.0)
    shade = node('ShaderNodeMix', "Shade", 300, 450)
    shade.data_type = 'RGBA'; shade.blend_type = 'MULTIPLY'
    shade.inputs[0].default_value = 1.0
    links.new(grp.outputs["Color"], shade.inputs[6])
    links.new(tint.outputs[2], shade.inputs[7])
    base = bsdf.inputs.get('Base Color')
    if base is not None:
        links.new(shade.outputs[2], base)

    # strands: noise remapped to 0..1, kept where it beats the layer, fading to the tips
    strand = node('ShaderNodeMapRange', "Strand", 100, -300)
    strand.inputs['From Min'].default_value = 0.35
    strand.inputs['From Max'].default_value = 1.0
    links.new(diff.outputs['Alpha'], strand.inputs['Value'])
    # threshold rises to 0.85 at the tip pass, so the last pass keeps its strongest strands
    lo = node('ShaderNodeMath', "Lo", 100, -500); lo.operation = 'MULTIPLY_ADD'
    links.new(lay_a.outputs[0], lo.inputs[0])
    lo.inputs[1].default_value = 0.85; lo.inputs[2].default_value = -0.15
    hi = node('ShaderNodeMath', "Hi", 100, -600); hi.operation = 'MULTIPLY_ADD'
    links.new(lay_a.outputs[0], hi.inputs[0])
    hi.inputs[1].default_value = 0.85; hi.inputs[2].default_value = 0.15
    keep = node('ShaderNodeMapRange', "Keep", 300, -300)
    keep.interpolation_type = 'SMOOTHSTEP'
    links.new(strand.outputs['Result'], keep.inputs['Value'])
    links.new(lo.outputs[0], keep.inputs['From Min'])
    links.new(hi.outputs[0], keep.inputs['From Max'])
    fall = node('ShaderNodeMath', "Falloff", 300, -500); fall.operation = 'MULTIPLY'
    links.new(lay_a.outputs[0], fall.inputs[0])
    fall.inputs[1].default_value = float(fur['alpha_falloff'])
    fade = node('ShaderNodeMath', "Fade", 450, -500); fade.operation = 'SUBTRACT'
    fade.inputs[0].default_value = 1.0
    links.new(fall.outputs[0], fade.inputs[1])
    alpha = node('ShaderNodeMath', "Alpha", 600, -300); alpha.operation = 'MULTIPLY'
    alpha.use_clamp = True
    links.new(keep.outputs['Result'], alpha.inputs[0])
    links.new(fade.outputs[0], alpha.inputs[1])
    mat_alpha = float(mat.get("rb2_fur_mat_alpha", 1.0))
    alpha2 = node('ShaderNodeMath', "AlphaMat", 750, -300); alpha2.operation = 'MULTIPLY'
    links.new(alpha.outputs[0], alpha2.inputs[0]); alpha2.inputs[1].default_value = mat_alpha
    a_in = bsdf.inputs.get('Alpha')
    if a_in is not None:
        for l in list(a_in.links):
            links.remove(l)
        links.new(alpha2.outputs[0], a_in)
    for n in [n for n in nodes if n.type == 'MATH' and not n.name.startswith("RB2Fur_")
              and not n.outputs[0].is_linked]:
        nodes.remove(n)                  # the plain alpha-blend multiply this replaced
    _set_render_method(mat, shells)
    mat["rb2_fur_params"] = repr(fur)
    from .rb2_creator_textures import sync_milo_shader
    sync_milo_shader(mat)
    return True


def _set_render_method(mat, shells):
    # Stacked passes sort badly when alpha-blended; dithering handles the overlap.
    for attr, value in (('surface_render_method', 'DITHERED' if shells else 'BLENDED'),
                        ('blend_method', 'HASHED' if shells else 'BLEND')):
        try:
            setattr(mat, attr, value)
            return
        except (AttributeError, TypeError):
            continue


def set_fur_mode(mat, shells):
    """Switches one fur material, and the meshes using it, between shells and flat."""
    import ast
    nt = mat.node_tree
    flat = nt.nodes.get("RB2Fur_Flat") if nt else None
    if flat is not None:
        flat.outputs[0].default_value = 0.0 if shells else 1.0
    try:
        fur = ast.literal_eval(mat.get("rb2_fur_params", "{}"))
    except (ValueError, SyntaxError):
        fur = {}
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and any(s.material is mat for s in obj.material_slots):
            if fur:
                set_shells(obj, fur, shells)
    mat["rb2_fur_shells"] = bool(shells)
    _set_render_method(mat, shells)


def apply_fur(mat, objects, fur, shells):
    """Sets up a fur material and its meshes. Returns True when done."""
    if not setup_fur_material(mat, fur, shells):
        return False
    for obj in objects:
        if obj.type == 'MESH':
            if shells:
                set_shells(obj, fur, True)
    mat["rb2_fur_shells"] = bool(shells)
    return True
