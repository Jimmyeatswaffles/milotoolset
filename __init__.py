bl_info = {
    "name": "Milo Toolset",
    "author": "jimmyeatwaffles",
    "version": (0, 9, 0),
    "blender": (5, 2, 0),
    "location": "File > Export > Milo Scene (.milo)",
    "description": "3D model exporter for Harmonix games using the milo archive format. "
                   "Based on the gltfmilo CLI program by ihatecompvir.",
    "category": "Import-Export",
}

import os
import math as _math

import bpy
import bpy.utils.previews
from bpy.props import (
    StringProperty,
    BoolProperty,
    EnumProperty,
    IntProperty,
    FloatProperty,
    FloatVectorProperty,
    PointerProperty,
    CollectionProperty,
)
from bpy_extras.io_utils import ExportHelper, ImportHelper

from .utilities import *
from .io import *
from .physics_exporter import *
from .texture_exporter import *
from .skeleton_exporter import *
from .model_exporter import *

# `import *` skips names starting with underscore, so anything private that this file
# calls directly (rather than through a public wrapper) has to be imported explicitly.
from .utilities import _log, _resolve_stock_reference_armature
from .io import _build_armature_from_skeleton
from .physics_exporter import _hair_bone_names, hair_profile_for_game
from .skeleton_exporter import _read_skeleton_milo_trans, _trans_world_translation


# =====================================================================================
# Custom icons (game/console thumbnails + the export menu icon)
#
# Loaded into a bpy.utils.previews collection at register() time - the icon_id numbers
# it hands out are only valid once that's happened, which is why the Platform/Game
# EnumProperty items below are built by FUNCTIONS (_milo_platform_items/_milo_game_items)
# rather than static tuples: a static tuple is evaluated once at class-definition time
# (i.e. at addon import, before register() runs), so it could never see real icon ids.
#
# Icon files live in this addon's icons/ folder. A couple of provided icons (RB1, GH2,
# GDRB) aren't wired to any game/console yet - those games don't have export/import
# support in this plugin yet, so the icons just sit here unused until that lands.
# =====================================================================================

_ICON_FILES = {
    'RB3': "RB3.jpg",
    'DC1': "DC1.png",
    'DC3': "DC3.png",
    'RB2': "RB2.jpg",
    'TBRB': "TBRB.jpg",
    'PS3': "PS3.png",
    'XBOX360': "x360.png",
    'MILO_EXPORT': "miloexportandinporticon.png",
    # Provided, but not yet wired to a game/console - no export/import path exists for
    # these yet. Kept here so they're one line away from use once support is added.
    'RB1': "RB1.jpg",
    'GH2': "GH2.png",
    'GDRB': "GDRB.jpg",
}

_preview_collections = {}


def _register_icons():
    icons_dir = os.path.join(os.path.dirname(__file__), "icons")
    pcoll = bpy.utils.previews.new()
    for key, filename in _ICON_FILES.items():
        path = os.path.join(icons_dir, filename)
        if os.path.exists(path):
            pcoll.load(key, path, 'IMAGE')
        else:
            _log(f"Milo Toolset: icon '{filename}' not found in icons/ - '{key}' will "
                 f"show the default blank icon instead")
    _preview_collections["main"] = pcoll


def _unregister_icons():
    for pcoll in _preview_collections.values():
        bpy.utils.previews.remove(pcoll)
    _preview_collections.clear()





MAT_BLEND_ITEMS = (
    ('kBlendDest', "Dest", ""),
    ('kBlendSrc', "Src", "Default - opaque/cutout materials"),
    ('kBlendAdd', "Add", ""),
    ('kBlendSrcAlpha', "Src Alpha", "Standard alpha blending"),
    ('kBlendSrcAlphaAdd', "Src Alpha Add", ""),
    ('kBlendSubtract', "Subtract", ""),
    ('kBlendMultiply', "Multiply", ""),
    ('kPreMultAlpha', "Premultiplied Alpha", ""),
)


MAT_ZMODE_ITEMS = (
    ('kZModeDisable', "Disable", ""),
    ('kZModeNormal', "Normal", "Default"),
    ('kZModeTransparent', "Transparent", ""),
    ('kZModeForce', "Force", ""),
    ('kZModeDecal', "Decal", ""),
)


MAT_SHADER_VARIATION_ITEMS = (
    ('kShaderVariationNone', "None", "Default - no special shader variation"),
    ('kShaderVariationSkin', "Skin", "Skin shading variation"),
    ('kShaderVariationHair', "Hair", "Hair shading variation"),
)


class GltfMiloMaterialSettings(bpy.types.PropertyGroup):
    blend: EnumProperty(
        name="Blend Mode",
        description="RndMat.blend - how to blend this material's poly into the screen",
        items=MAT_BLEND_ITEMS,
        default='kBlendSrc',
    )
    base_color: FloatVectorProperty(
        name="Base Color",
        description="RndMat.color - base material color",
        subtype='COLOR',
        size=4,
        min=0.0, max=1.0,
        default=(1.0, 1.0, 1.0, 1.0),
    )
    pre_lit: BoolProperty(
        name="Pre-Lit",
        description="RndMat.preLit - use vertex color and alpha for base or ambient",
        default=True,
    )
    intensify: BoolProperty(
        name="Intensify Base Color",
        description="RndMat.intensify - boosts (doubles) the base color, pushing bright "
                     "materials further into the venue's bloom range. This is the trick "
                     "RB3's neon-sign guitar uses on its bulbs: every untextured bulb "
                     "material has this enabled on top of a bright base color and Pre-Lit, "
                     "which is what makes them read as glowing neon rather than just brightly "
                     "lit. Pair with Pre-Lit + a bright base color + no diffuse texture",
        default=False,
    )
    use_environment: BoolProperty(
        name="Use Environment",
        description="RndMat.useEnviron - modulate with environment ambient and lights",
        default=False,
    )
    z_mode: EnumProperty(
        name="Z-Buffer Mode",
        description="RndMat.zMode - how to read and write the z-buffer",
        items=MAT_ZMODE_ITEMS,
        default='kZModeNormal',
    )
    alpha_cut: BoolProperty(
        name="Alpha Cut",
        description="RndMat.alphaCut - cut zero-alpha pixels from the z-buffer",
        default=False,
    )
    alpha_threshold: IntProperty(
        name="Alpha Threshold",
        description="RndMat.alphaThreshold - alpha level below which gets cut (0-255)",
        min=0, max=255,
        default=0,
    )
    diffuse_tex: PointerProperty(
        name="Diffuse Texture",
        description="RndMat.diffuseTex - base texture map, modulated with color and alpha",
        type=bpy.types.Image,
    )
    normal_tex: PointerProperty(
        name="Normal Map",
        description="RndMat.normalMap - texture map that defines lighting normals",
        type=bpy.types.Image,
    )
    specular_tex: PointerProperty(
        name="Specular Map",
        description="RndMat.specularMap - texture map for specular color and power",
        type=bpy.types.Image,
    )
    emissive_tex: PointerProperty(
        name="Emissive Map",
        description="RndMat.emissiveMap - map for self illumination",
        type=bpy.types.Image,
    )
    specular_rgb: FloatVectorProperty(
        name="Specular RGB",
        description="RndMat.specularRGB - color to use when not driven by a specular map",
        subtype='COLOR',
        size=3,
        min=0.0, max=1.0,
        default=(0.255, 0.255, 0.255),
    )
    specular_power: FloatProperty(
        name="Specular Power",
        description="RndMat.specularPower - power to use when not driven by a specular map",
        default=0.0,
    )
    emissive_multiplier: FloatProperty(
        name="Emissive Multiplier",
        description="RndMat.emissiveMultiplier - multiplier applied to emission",
        default=1.0,
    )
    environment_map: BoolProperty(
        name="Environment Map?",
        description='RndMat.environMap - when enabled, uses the game\'s base cube map '
                     '("instruments.cube") for reflections. No support yet for custom '
                     'cube maps',
        default=False,
    )
    cull: BoolProperty(
        name="Cull (Backface)",
        description="RndMat.cull - cull backfaces of this material's polygons "
                     "(equivalent to the CLI tool's !DoubleSided)",
        default=False,
    )
    de_normal: EnumProperty(
        name="De-Normal",
        description="RndMat.deNormal - stored as a float, but only -1, 0, or 1 are "
                     "meaningful values, so it's exposed as a 3-way choice here",
        items=(
            ('-1', "-1", ""),
            ('0', "0", ""),
            ('1', "1", ""),
        ),
        default='0',
    )
    anisotropy: FloatProperty(
        name="Anisotropy",
        description="RndMat.anisotropy",
        default=0.0,
    )
    normal_detail_strength: FloatProperty(
        name="Normal Detail Strength",
        description="RndMat.normalDetailStrength",
        default=0.0,
    )
    rim_rgb: FloatVectorProperty(
        name="Rim RGB",
        description="RndMat.rimRGB - rim lighting color",
        subtype='COLOR',
        size=3,
        min=0.0, max=1.0,
        default=(0.0, 0.0, 0.0),
    )
    rim_power: FloatProperty(
        name="Rim Power",
        description="RndMat.rimPower - rim lighting falloff power",
        default=4.0,
    )
    shader_variation: EnumProperty(
        name="Shader Variation",
        description="RndMat.shaderVariation - special shading path (skin/hair)",
        items=MAT_SHADER_VARIATION_ITEMS,
        default='kShaderVariationNone',
    )
    meta_material: StringProperty(
        name="MetaMaterial (DC3)",
        description="Dance Central 3 only. The .mmat material template this material derives "
                     "from - it drives the actual shading path, so picking the right one matters. "
                     "DC3 resolves it by name against the game's own globally-loaded template "
                     "library (it is NOT shipped inside your character milo) and pulls default/"
                     "forced property values from it. Templates retail Emilia uses, by surface: "
                     "skin/body/face -> 'char_basic_skin.mmat' (or 'char_basic_skin_rim.mmat' for "
                     "rim lighting); clothing -> 'char_basic_rim.mmat'; hair -> "
                     "'char_basic_hair.mmat'; eyes/teeth -> 'char_basic_skin_nospec.mmat'. AVOID "
                     "'char_basic_color.mmat' on a visible surface - it is a flat self-lit color "
                     "template and renders the character as a uniform glowing color instead of a "
                     "properly lit, textured surface. Ignored for RB3 and DC1, which use the flat "
                     "rev-68 material format",
        default=DC3_DEFAULT_META_MATERIAL,
    )

    # --- Refract settings (own collapsible sub-panel in the UI) ---
    refract_enabled: BoolProperty(
        name="Refract Enabled",
        description="RndMat.refractEnabled - enable screen-space refraction for this material",
        default=False,
    )
    refract_strength: FloatProperty(
        name="Refract Strength",
        description="RndMat.refractStrength - how strongly the material refracts",
        default=0.0,
    )
    refract_normal_map: PointerProperty(
        name="Refract Normal Map",
        description="RndMat.refractNormalMap - normal map that drives refraction "
                     "distortion (separate from the main Normal Map)",
        type=bpy.types.Image,
    )


class MATERIAL_OT_gltfmilo_autodetect_textures(bpy.types.Operator):
    """Scans this material's node tree for a Principled BSDF and fills the texture
    slots above from whatever Image Texture nodes feed its Base Color, Normal,
    Specular, and Emission inputs."""
    bl_idname = "material.gltfmilo_autodetect_textures"
    bl_label = "Auto-Detect Textures From Nodes"
    bl_options = {'REGISTER', 'UNDO'}

    @staticmethod
    def _trace_image(socket):
        """Walk backward from a shader input socket through Reroute/Normal Map
        nodes (and, as a fallback, any other single-input node) to find the
        Image Texture node feeding it."""
        if socket is None or not socket.is_linked:
            return None
        node = socket.links[0].from_node
        visited = set()
        while node is not None and id(node) not in visited:
            visited.add(id(node))
            if node.type == 'TEX_IMAGE':
                return node.image
            if node.type == 'REROUTE':
                in0 = node.inputs[0] if node.inputs else None
                node = in0.links[0].from_node if (in0 and in0.is_linked) else None
                continue
            if node.type == 'NORMAL_MAP':
                color_in = node.inputs.get('Color')
                node = color_in.links[0].from_node if (color_in and color_in.is_linked) else None
                continue
            # Fallback: follow the first linked input on any other node type
            next_node = None
            for inp in node.inputs:
                if inp.is_linked:
                    next_node = inp.links[0].from_node
                    break
            node = next_node
        return None

    def execute(self, context):
        mat = context.material
        if mat is None or mat.node_tree is None:
            self.report({'WARNING'}, "This material has no node tree to scan.")
            return {'CANCELLED'}

        principled = next((n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
        if principled is None:
            self.report({'WARNING'}, "No Principled BSDF node found in this material.")
            return {'CANCELLED'}

        settings = mat.gltfmilo_settings
        found = []

        img = self._trace_image(principled.inputs.get('Base Color'))
        if img:
            settings.diffuse_tex = img
            found.append("Diffuse")

        img = self._trace_image(principled.inputs.get('Normal'))
        if img:
            settings.normal_tex = img
            found.append("Normal")

        # "Specular Tint" is where the CLI tool's own convention plugs a specular
        # map in (confirmed against a real rig) - check that before "IOR Level"/
        # the older pre-4.0 "Specular" input name
        specular_input = (principled.inputs.get('Specular Tint')
                           or principled.inputs.get('Specular IOR Level')
                           or principled.inputs.get('Specular'))
        img = self._trace_image(specular_input)
        if img:
            settings.specular_tex = img
            found.append("Specular")

        emission_input = principled.inputs.get('Emission Color') or principled.inputs.get('Emission')
        img = self._trace_image(emission_input)
        if img:
            settings.emissive_tex = img
            found.append("Emissive")

        if found:
            self.report({'INFO'}, f"Auto-detected: {', '.join(found)}")
        else:
            self.report({'WARNING'}, "No connected image textures found on the Principled BSDF.")
        return {'FINISHED'}


class MATERIAL_PT_gltfmilo_settings(bpy.types.Panel):
    bl_label = "glTFMilo Material Settings"
    bl_idname = "MATERIAL_PT_gltfmilo_settings"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "material"

    @classmethod
    def poll(cls, context):
        return context.material is not None

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        settings = context.material.gltfmilo_settings

        layout.prop(settings, "blend")
        layout.prop(settings, "base_color")
        layout.prop(settings, "pre_lit")
        layout.prop(settings, "intensify")
        layout.prop(settings, "use_environment")
        layout.prop(settings, "z_mode")
        layout.prop(settings, "alpha_cut")
        layout.prop(settings, "alpha_threshold")

        box = layout.box()
        box.label(text="Textures", icon='TEXTURE')
        row = box.row(align=True)
        row.label(text="Diffuse")
        row.template_ID(settings, "diffuse_tex", open="image.open")
        row = box.row(align=True)
        row.label(text="Normal Map")
        row.template_ID(settings, "normal_tex", open="image.open")
        row = box.row(align=True)
        row.label(text="Specular")
        row.template_ID(settings, "specular_tex", open="image.open")
        row = box.row(align=True)
        row.label(text="Emissive")
        row.template_ID(settings, "emissive_tex", open="image.open")
        box.operator("material.gltfmilo_autodetect_textures", icon='VIEWZOOM')

        layout.prop(settings, "specular_rgb")
        layout.prop(settings, "specular_power")
        layout.prop(settings, "emissive_multiplier")
        layout.prop(settings, "environment_map")

        box = layout.box()
        box.label(text="Advanced", icon='MODIFIER')
        box.prop(settings, "cull")
        box.prop(settings, "de_normal")
        box.prop(settings, "anisotropy")
        box.prop(settings, "normal_detail_strength")
        box.prop(settings, "rim_rgb")
        box.prop(settings, "rim_power")
        box.prop(settings, "shader_variation")
        box.prop(settings, "meta_material")


class MATERIAL_PT_gltfmilo_refract(bpy.types.Panel):
    """Refract settings - a separate, collapsed-by-default sub-panel so the
    (rarely-used, easily-misunderstood) screen-space refraction options don't
    clutter the main material settings or confuse users."""
    bl_label = "Refract Settings"
    bl_idname = "MATERIAL_PT_gltfmilo_refract"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "material"
    bl_parent_id = "MATERIAL_PT_gltfmilo_settings"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return context.material is not None

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        settings = context.material.gltfmilo_settings

        layout.prop(settings, "refract_enabled")
        layout.prop(settings, "refract_strength")
        row = layout.row(align=True)
        row.label(text="Refract Normal Map")
        row.template_ID(settings, "refract_normal_map", open="image.open")


class GltfMiloHairBoneItem(bpy.types.PropertyGroup):
    """One bone in a strand's ordered root-to-tip chain."""
    name: StringProperty(name="Bone")


class GltfMiloHairStrand(bpy.types.PropertyGroup):
    """One hair strand - an ordered chain of bones (root first) plus per-strand
    starting-flip angle. Physics params live on the armature-level settings, matching
    the .hair format (they're global to the CharHair asset, not per-strand)."""
    name: StringProperty(name="Strand Name", default="Strand")
    angle: FloatProperty(
        name="Angle",
        description="CharHair.Strand.angle - starting flip angle in degrees",
        default=0.0,
    )
    hookup_flags: IntProperty(
        name="Hookup Flags",
        description="CharHair.Strand.hookup_flags - a collision-group BITMASK. This strand's "
                     "points collide with a CharCollide volume only if this value shares at "
                     "least one bit with that volume's 'Collision Group Flags' "
                     "(hookup_flags & collide.flags != 0). Set this to the OR of the group "
                     "bits of every collision volume you want this strand to bump against. "
                     "e.g. 63 (= 1+2+4+8+16+32) collides with volumes in groups 1 through 6. "
                     "If 0, this strand collides with nothing",
        default=1, min=0,
    )
    bones: CollectionProperty(type=GltfMiloHairBoneItem)


class GltfMiloHairSettings(bpy.types.PropertyGroup):
    """Armature-level CharHair physics settings + the list of authored strands.
    Float defaults match the values seen in real RB3 .hair files (see the reference
    screenshot: stiffness/torsion 0.1, inertia 0.65, gravity 1.0, weight 0.5)."""
    enabled: BoolProperty(
        name="Export CharHair (.hair) for this Armature",
        description="When on (and 'Export Hair Physics' is enabled in the exporter), "
                     "embeds a CharHair asset built from the strands below",
        default=False,
    )
    simulate: BoolProperty(
        name="Simulate Physics?",
        description="CharHair.simulate - run the hair/cloth physics simulation",
        default=True,
    )
    stiffness: FloatProperty(name="Stiffness", description="CharHair.stiffness - stiffness of each strand", default=0.1)
    torsion: FloatProperty(name="Torsion", description="CharHair.torsion - rotational stiffness of each strand", default=0.1)
    inertia: FloatProperty(name="Inertia", description="CharHair.inertia - inertia of the hair, zero means none", default=0.65)
    gravity: FloatProperty(name="Gravity", description="CharHair.gravity - gravity of the hair, one is normal", default=1.0)
    weight: FloatProperty(name="Weight", description="CharHair.weight - weight of the hair, one is normal", default=0.5)
    friction: FloatProperty(name="Friction", description="CharHair.friction - hair friction against each other", default=0.0)
    min_slack: FloatProperty(name="Min Slack", description="CharHair.minSlack - if using sides, how far in it could go", default=0.0)
    max_slack: FloatProperty(name="Max Slack", description="CharHair.maxSlack - if using sides, how far out it could go", default=0.0)

    strands: CollectionProperty(type=GltfMiloHairStrand)
    active_strand_index: IntProperty(default=0)


def _selected_bone_names(context, armature_obj):
    """Returns the set of selected bone names, working across Blender modes. As of
    Blender 5.0 the `.select` attribute was removed from Bone (the object/pose-mode
    rest datablock), so iterating `armature_obj.data.bones` and reading `b.select`
    raises AttributeError. The mode-appropriate context selection lists are the
    supported way to read selection:
      - Pose mode:   context.selected_pose_bones (each has a .name)
      - Edit mode:   context.selected_bones / context.selected_editable_bones
      - Object mode: nothing is "selected" at the bone level; fall back to any pose
                     bones flagged, else empty.
    We collect just the NAMES here (never holding bone pointers across a mode switch,
    per Blender's own guidance) and resolve them against data.bones afterwards for the
    parent/child chain walk."""
    names = set()

    # Prefer whichever context list is populated for the current mode.
    for attr in ("selected_pose_bones", "selected_bones", "selected_editable_bones"):
        seq = getattr(context, attr, None)
        if seq:
            for b in seq:
                # Only keep bones that belong to THIS armature (context lists can span
                # multiple selected armatures).
                name = getattr(b, "name", None)
                if name and name in armature_obj.data.bones:
                    names.add(name)
            if names:
                return names

    return names


def _ordered_bones_from_selection(context, armature_obj):
    """Given an armature with some bones selected (in pose or edit mode), returns the
    selected bones (as data.bones entries) ordered root-first along their parent chain.
    Raises ValueError if the selection isn't a single connected chain (a strand must be
    one linear chain - the engine walks TransChildren().front() from the root, i.e. a
    single path)."""
    selected_set = _selected_bone_names(context, armature_obj)
    if not selected_set:
        raise ValueError(
            "No bones selected. Enter Pose or Edit mode and select the bones of a "
            "single hair strand first."
        )

    bones = armature_obj.data.bones
    selected = [bones[name] for name in selected_set]

    roots = [b for b in selected if (b.parent is None or b.parent.name not in selected_set)]
    if len(roots) != 1:
        raise ValueError(
            f"Selection has {len(roots)} root bones - a strand must be a single "
            f"connected chain with exactly one root (a bone whose parent is unselected)."
        )

    ordered = []
    current = roots[0]
    while current is not None:
        ordered.append(current)
        sel_children = [c for c in current.children if c.name in selected_set]
        if len(sel_children) > 1:
            raise ValueError(
                f"Bone '{current.name}' has {len(sel_children)} selected children - "
                f"a strand must be a single linear chain, not a branching tree."
            )
        current = sel_children[0] if sel_children else None

    if len(ordered) != len(selected):
        raise ValueError(
            "Selected bones don't form one connected chain - some selected bones "
            "aren't part of the root's parent chain."
        )
    return ordered


class ARMATURE_OT_gltfmilo_create_hair_strand(bpy.types.Operator):
    """Create a hair strand from the currently selected armature bones. The bones must
    form a single connected chain (one root, no branching); they're recorded root-first
    so the exporter knows the physics bone order and parenting chain."""
    bl_idname = "armature.gltfmilo_create_hair_strand"
    bl_label = "Create Hair Strand from Selected Bones"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and obj.type == 'ARMATURE'

    def execute(self, context):
        armature_obj = context.object
        try:
            ordered = _ordered_bones_from_selection(context, armature_obj)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        hair = armature_obj.data.gltfmilo_hair
        strand = hair.strands.add()
        strand.name = f"Strand {len(hair.strands)}"
        for bone in ordered:
            item = strand.bones.add()
            item.name = bone.name
        hair.active_strand_index = len(hair.strands) - 1

        self.report({'INFO'}, f"Created '{strand.name}' from {len(ordered)} bone(s): "
                              f"{' -> '.join(b.name for b in ordered)}")
        return {'FINISHED'}


class ARMATURE_OT_gltfmilo_remove_hair_strand(bpy.types.Operator):
    """Remove the selected hair strand."""
    bl_idname = "armature.gltfmilo_remove_hair_strand"
    bl_label = "Remove Hair Strand"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.object
        if obj is None or obj.type != 'ARMATURE':
            return False
        return len(obj.data.gltfmilo_hair.strands) > 0

    def execute(self, context):
        hair = context.object.data.gltfmilo_hair
        idx = hair.active_strand_index
        if 0 <= idx < len(hair.strands):
            hair.strands.remove(idx)
            hair.active_strand_index = max(0, idx - 1)
        return {'FINISHED'}


class DATA_UL_gltfmilo_hair_strands(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.label(text=item.name, icon='STRANDS')
        row.label(text=f"{len(item.bones)} bone(s)")


class DATA_PT_gltfmilo_hair(bpy.types.Panel):
    bl_label = "glTFMilo Hair Setup"
    bl_idname = "DATA_PT_gltfmilo_hair"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "data"

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == 'ARMATURE'

    def draw(self, context):
        layout = self.layout
        hair = context.object.data.gltfmilo_hair

        layout.prop(hair, "enabled")

        col = layout.column()
        col.enabled = hair.enabled

        col.operator("armature.gltfmilo_create_hair_strand", icon='ADD')

        row = col.row()
        row.template_list("DATA_UL_gltfmilo_hair_strands", "", hair, "strands",
                          hair, "active_strand_index", rows=3)
        side = row.column(align=True)
        side.operator("armature.gltfmilo_remove_hair_strand", icon='REMOVE', text="")

        if 0 <= hair.active_strand_index < len(hair.strands):
            strand = hair.strands[hair.active_strand_index]
            box = col.box()
            box.prop(strand, "name")
            box.prop(strand, "angle")
            box.prop(strand, "hookup_flags")
            box.label(text="Bones (root -> tip):")
            for item in strand.bones:
                box.label(text=f"    {item.name}", icon='BONE_DATA')

        col.separator()
        box = col.box()
        box.label(text="Physics Settings", icon='PHYSICS')
        box.prop(hair, "simulate")
        box.prop(hair, "stiffness")
        box.prop(hair, "torsion")
        box.prop(hair, "inertia")
        box.prop(hair, "gravity")
        box.prop(hair, "weight")
        box.prop(hair, "friction")
        box.prop(hair, "min_slack")
        box.prop(hair, "max_slack")
        box.label(text="Wind: world.wind (fixed)", icon='FORCE_WIND')


def _on_milo_type_update(self, context):
    """When the user switches export Type, set sensible per-type defaults for the
    armature/hair toggles so they don't have to remember to tick them. Instruments
    need their bone rig embedded (that's what makes a guitar load and hold correctly),
    so armature export defaults ON for them; hair physics is enabled by default too so
    it can be experimented with on instruments. Switching away from instrument doesn't
    force anything back off - the user stays in control after the initial nudge."""
    if self.milo_type == 'instrument':
        self.export_armature = True
        self.export_hair = True


def _milo_icon_id(key):
    """Look up a loaded custom icon's id by key, falling back to 0 (no icon) if the
    icon wasn't found/loaded - e.g. running before register() or if a source image is
    missing from the icons/ folder."""
    pcoll = _preview_collections.get("main")
    if pcoll is not None and key in pcoll:
        return pcoll[key].icon_id
    return 0


def _milo_platform_items(self, context):
    """Dynamic EnumProperty items for Platform, with per-console icons. Built as a
    function (rather than a static tuple) because the custom icon ids from
    bpy.utils.previews are only known once register() has loaded them - see
    _milo_icon_id. String literals below are safe to return fresh each call (Blender's
    known dynamic-enum gotcha is about NOT fabricating new strings per call; these are
    constants baked into the function, so their identity is stable)."""
    return (
        ('xbox360', "Xbox 360", "Export for Xbox 360", _milo_icon_id('XBOX360'), 0),
        ('ps3', "PS3", "Export for PlayStation 3 (different vertex packing, no texture "
         "byte-swap, and BC1 normal maps - otherwise identical to Xbox)",
         _milo_icon_id('PS3'), 1),
    )


def _milo_game_items(self, context):
    """Dynamic EnumProperty items for Game, with per-game icons (see
    _milo_platform_items for why this is a function). Dance Central 3 has no icon
    supplied yet, so it falls back to Blender's default blank enum icon (icon id 0)."""
    return (
        ('rb3', "Rock Band 3", "Currently the most thoroughly tested path",
         _milo_icon_id('RB3'), 0),
        ('dc1', "Dance Central 1", "EXPERIMENTAL. DC1 uses byte-identical format "
         "revisions to RB3 (DirectoryMeta 28 / Mesh 38 / Mat 68 / Tex 11 / "
         "CharCollide 7), so the same asset writers are used. Xbox 360 only. "
         "Everything is written flat into the milo root - the game resolves objects "
         "by name (ObjectDir::FindObject checks its own entries before recursing into "
         "subdirectories), so the subdirectory layout real DC1 files use is a "
         "load-time sharing optimisation rather than a requirement",
         _milo_icon_id('DC1'), 1),
        ('dc3', "Dance Central 3", "EXPERIMENTAL. Same asset writers as DC1/RB3, but the "
         "container's DirectoryMeta is written at revision 32 (with the extra rev-32 "
         "header byte) so it is revision-consistent with a DC3 CharClipSet. Use a DC3 "
         "donor to copy that game's win/intro animation clips - injecting them into a "
         "rev-28 (DC1) container instead makes DC3 misread the clips and crash. Xbox 360 "
         "only", _milo_icon_id('DC3'), 2),
        ('rb2', "Rock Band 2", "Not yet implemented", _milo_icon_id('RB2'), 3),
        ('tbrb', "The Beatles: Rock Band", "Not yet implemented", _milo_icon_id('TBRB'), 4),
    )


class EXPORT_OT_milo_scene(bpy.types.Operator, ExportHelper):
    """Export the current scene to a Milo (.milo) file"""
    bl_idname = "export_scene.milo"
    bl_label = "Export Milo Scene"
    bl_options = {'PRESET'}

    filename_ext = ".milo_xbox"
    filter_glob: StringProperty(
        default="*.milo_xbox;*.milo_ps3;*.milo",
        options={'HIDDEN'},
    )

    platform: EnumProperty(
        name="Platform",
        description="Target console for the exported Milo file",
        items=_milo_platform_items,
        default=0,
    )

    game: EnumProperty(
        name="Game",
        description="Target game / revision set",
        items=_milo_game_items,
        default=0,
    )

    donor_mode: EnumProperty(
        name="Donor Mode",
        description="What to do with the donor milo",
        items=(
            ('inject_meshes', "Inject Meshes Into Donor",
             "Rebuild the DONOR milo with its meshes replaced by the Blender meshes, "
             "leaving everything else byte-for-byte untouched (directory body, LOD lists, "
             "CharClipSet animations, materials, textures, bones, collision, hair). The "
             "output is the donor with new geometry - materials are NOT exported, so mesh "
             "material references should match the donor's existing material names"),
            ('charclipset', "Copy CharClipSet Only",
             "Build the milo from scratch as usual, but copy the CharClipSet (animation "
             "clip set) verbatim out of the donor as entry 0"),
        ),
        default='inject_meshes',
    )

    donor_milo: StringProperty(
        name="Donor Milo (CharClipSet)",
        description="EXPERIMENTAL, Dance Central only. Path to a real DC1 or DC3 character "
                     "milo to use as the donor. What gets taken from it depends on Donor "
                     "Mode above (DC3 supports CharClipSet copy only). The CharClipSet "
                     "extractor auto-detects the donor's revision (DC1 = 28, DC3 = 32). "
                     "Leave empty to export a fresh milo with no donor data",
        subtype='FILE_PATH',
        default="",
    )

    milo_type: EnumProperty(
        name="Type",
        description="Intended use of the exported Milo directory",
        items=(
            ('character', "Character", "Implemented, but not yet verified against a real "
             "reference file the way the static mesh path was - test carefully"),
            ('venue', "Venue", "Not yet implemented (needs RndGroup support)"),
            ('instrument', "Instrument", "For guitars/basses/etc. Uses the "
             "colorpalettes.milo shared subdir (not the character skeleton). The engine "
             "also expects the instrument's animation bone rig (bone_target_*, "
             "bone_pos_string*, spot_neck_fret*, etc.) to be present - export those from "
             "your guitar armature. Selecting this auto-enables armature + hair export"),
            ('custom_song_asset', "Custom Song Asset",
             "The Beatles: Rock Band ONLY. Instead of building a milo, export each mesh/"
             "material/texture as a LOOSE standalone .mesh, .mat and .tex file into the "
             "chosen folder. TBRB custom songs can load these from an 'extras' folder and "
             "wire them into the world at runtime via song.dta scripting (set_trans_parent, "
             "add_object, set diffuse_tex...). Use the per-object 'Milo Object > Trans "
             "Parent' setting to bake a bone parent into the .mesh so it rides an existing "
             "in-game object without a set_trans_parent call. No container, no LOD groups, "
             "no subdirectory references - just the raw assets"),
            ('other', "Other", "Plain RndDir - the most thoroughly validated path"),
        ),
        default='other',
        update=_on_milo_type_update,
    )

    instrument_type: EnumProperty(
        name="Instrument",
        description="Which instrument this milo is, for Type=Instrument. This selects the "
                     "OutfitConfig (.cfg) name the game looks up when you select the item - "
                     "it MUST match or the customize/color panel hard-crashes on select. "
                     "Confirmed from the RB3 decomp (GetConfigNameFromAssetType) and by "
                     "byte-diffing real vanilla instruments: guitar->guitar.cfg, "
                     "bass->bass.cfg, drum->drum.cfg, mic->mic.cfg, keyboard->keyboard.cfg",
        items=(
            ('guitar', "Guitar", "Emits guitar.cfg"),
            ('bass', "Bass", "Emits bass.cfg"),
            ('drum', "Drums", "Emits drum.cfg"),
            ('mic', "Microphone", "Emits mic.cfg"),
            ('keyboard', "Keyboard", "Emits keyboard.cfg"),
        ),
        default='guitar',
    )

    tbrb_target: EnumProperty(
        name="TBRB Milo",
        description="The Beatles: Rock Band splits a character across THREE milos - a "
                     "skeleton milo (bones + collision), a base head/hands milo, and an "
                     "outfit+hair milo. This picks which one to export",
        items=(
            ('skeleton', "Skeleton (bones + collision)",
             "Export <beatle>_skeleton.milo - the Trans bone hierarchy taken from the export "
             "armature plus any CharCollide (.coll) volumes. This is the piece the original "
             "glTFMilo CLI never exported. IMPLEMENTED and byte-verified against a real PS3 "
             "george_skeleton"),
            ('meshes', "Meshes (head+hands / outfit)",
             "Export the mesh milo: meshes + materials + textures, written flat with no LOD "
             "groups. Meshes are written at revision 33 (glTFMilo's target, confirmed working "
             "in-game) rather than retail's 36 - Harmonix loaders read older mesh revisions, "
             "and rev 33 stores tangents as plain floats so the RB3 flat-normal-map bug "
             "cannot occur. Export the skeleton milo separately and install both"),
        ),
        default='skeleton',
    )

    tbrb_beatle: EnumProperty(
        name="Beatle",
        description="Which Beatle this skeleton is for. This ONLY sets the output file name "
                     "(<beatle>_skeleton) - the skeleton content comes from your export "
                     "armature regardless. Pick 'Custom' to keep whatever name you typed",
        items=(
            ('george', "George", "Output george_skeleton"),
            ('john', "John", "Output john_skeleton"),
            ('paul', "Paul", "Output paul_skeleton"),
            ('ringo', "Ringo", "Output ringo_skeleton"),
            ('custom', "Custom (keep typed name)",
             "Use the filename you typed in the dialog instead of forcing <beatle>_skeleton"),
        ),
        default='george',
    )

    tbrb_mesh_kind: EnumProperty(
        name="Mesh Milo Kind",
        description="Which kind of mesh milo to write. This sets the subdirectory references, "
                     "which differ: an outfit milo points at the base head/hands milo plus a "
                     "shared outfit milo (and inherits the skeleton through them), while a "
                     "head/hands milo points at the skeleton milo directly",
        items=(
            ('outfit', "Outfit", "References base head/hands milo + shared outfit milo"),
            ('headhands', "Head / Hands", "References the skeleton milo directly"),
        ),
        default='outfit',
    )

    tbrb_base_milo: StringProperty(
        name="Base Head/Hands Milo",
        description="Outfit only: relative path to the base head/hands milo this outfit sits "
                     "on. Retail cavern02 uses '../george_headhands_short.milo'. Leave empty "
                     "to default to '../george_headhands_long.milo'",
        default="../george_headhands_long.milo",
    )

    tbrb_share_milo: StringProperty(
        name="Shared Outfit Milo",
        description="Outfit only: relative path to the shared outfit milo. Retail cavern02 "
                     "uses '../../shared/outfit/cavern02_share.milo'. Leave empty to omit it",
        default="",
    )

    tbrb_test_hardcode_subdirs: BoolProperty(
        name="TEST: Missing Sub Directory Theory",
        description="Isolation test. When on, ignores the fields above and writes the EXACT "
                     "two subdirectories from the retail cavern02 outfit milo "
                     "(../george_headhands_short.milo and "
                     "../../shared/outfit/cavern02_share.milo). Lets you confirm whether the "
                     "missing/wrong subdirs were the crash without changing anything else",
        default=False,
    )

    tbrb_skeleton_milo: StringProperty(
        name="Skeleton Milo Name",
        description="Name of the companion skeleton milo this mesh milo references as its "
                     "subdirectory. Retail george_headhands_long points at "
                     "'george_skeleton.milo', and that reference is how the mesh milo's bone "
                     "names resolve at runtime. Leave empty to derive it from the Beatle "
                     "dropdown (<beatle>_skeleton.milo)",
        default="",
    )

    tbrb_armature: StringProperty(
        name="Skeleton Armature",
        description="Name of the armature object to build the skeleton from. Leave empty to "
                     "use the active object (or the scene's only armature). Set this when the "
                     "scene holds more than one rig - the exporter refuses to merge armatures, "
                     "because merging de-duplicates by bone name and lets a stale rig silently "
                     "shadow the live one",
        default="",
    )

    tbrb_standard_collisions: BoolProperty(
        name="Retail Head/Neck Collisions",
        description="Emit the four collision volumes the retail TBRB skeleton ships "
                     "(face.coll, forehead.coll, head.coll, neck.coll) as byte-exact copies "
                     "of george's, including their local transforms. Hair and cloth in the "
                     "outfit milo collide against these BY NAME, so a skeleton without them "
                     "is missing something the rest of the character expects. This is "
                     "separate from the per-bone Milo Collision panel because TBRB puts "
                     "THREE volumes on bone_head.mesh, which the one-volume-per-bone UI "
                     "can't express. Turn off to use your own per-bone volumes instead",
        default=True,
    )

    tbrb_repair_degenerate_bones: BoolProperty(
        name="Repair Collapsed Driver Bones",
        description="When exporting a TBRB skeleton, detect non-deforming driver bones "
                     "(spot_*, *-crease, head_lookat) that imported collapsed onto the "
                     "armature origin and restore their rest transform from a reference "
                     "skeleton milo. Some retargeted GLB rigs stack these helper bones on one "
                     "identical wrong transform, which drags the face skin in-game (mouth/cheek "
                     "pull) and breaks the look-at. Only bones whose OWN world position is the "
                     "origin AND that the reference places off-origin are touched - bones you "
                     "moved on purpose are never overridden. Needs a Reference Skeleton Milo set",
        default=True,
    )

    tbrb_reference_skeleton_milo: StringProperty(
        name="Reference Skeleton Milo",
        description="Path to a pristine retail skeleton milo (e.g. the vanilla "
                     "george_skeleton.milo_ps3) used ONLY as the authoritative rest for "
                     "collapsed driver bones when 'Repair Collapsed Driver Bones' is on. The "
                     "mesh geometry, weights, and every bone you positioned still come from your "
                     "armature; this file just supplies correct positions for the origin-"
                     "collapsed helper bones. Leave empty to skip the repair",
        subtype='FILE_PATH',
        default="",
    )

    outfit_config: StringProperty(name="Outfit Config", subtype='FILE_PATH', default="")
    ignore_tex_size_limits: BoolProperty(
        name="Ignore Texture Size Limits",
        description="When off (default), textures are downscaled to fit within 512x512 "
                     "before export, matching the original CLI tool's behavior - this saves "
                     "space and console memory on real hardware. Turn this on to export "
                     "textures at up to 2048x2048 instead of 512x512 (2048 is a hard engine "
                     "limit, not a style choice - textures larger than that crash the game "
                     "regardless of platform, so this will still downscale anything bigger "
                     "than 2048x2048). Large textures near that ceiling can also take a "
                     "noticeably long time to compress during export.",
        default=False,
    )
    only_selected: BoolProperty(
        name="Selected Objects Only",
        description="Export only the currently selected mesh objects. If nothing is "
                     "selected, the whole scene is exported as usual",
        default=False,
    )
    bone_axis_correction: EnumProperty(
        name="Bone Axis Correction (experimental)",
        description="Rotates every bone's local frame by a fixed amount before computing "
                     "its skin offset. This is a debugging aid, not a real setting - if a "
                     "skinned mesh follows its bone correctly but is rotated by a clean "
                     "90/180 degrees, that points to a bone-local axis convention mismatch "
                     "rather than a position bug, and this lets you test corrections quickly "
                     "without re-rigging in Blender first. NOTE: reset to 'None' as the "
                     "default - the old -90 X default was tuned against the global Z-up/"
                     "Y-up conversion this version removes, so it no longer applies as a "
                     "starting point and needs to be re-tested from scratch if needed at all",
        items=(
            ('none', "None", ""),
            ('x90', "+90 X", ""),
            ('xneg90', "-90 X", ""),
            ('y90', "+90 Y", ""),
            ('yneg90', "-90 Y", ""),
            ('z90', "+90 Z", ""),
            ('zneg90', "-90 Z", ""),
        ),
        default='none',
    )
    offset_multiply_order: EnumProperty(
        name="Bone Offset Multiply Order (experimental)",
        description="The decompiled game engine's own SetBone() and glTFMilo's C# "
                     "implementation disagree on the multiply order for computing each "
                     "bone's skin offset matrix. 'Mesh First' (the default) matches the "
                     "decompiled engine; 'Bone First' matches glTFMilo's own CLI tool. "
                     "If skinned meshes deform incorrectly, try switching this alongside "
                     "the Bone Axis Correction above",
        items=(
            ('mesh_first', "Mesh First (engine order)", ""),
            ('bone_first', "Bone First (glTFMilo order)", ""),
        ),
        default='mesh_first',
    )
    export_armature: BoolProperty(
        name="Export Character Armature (Experimental)",
        description="Writes every armature bone as a real top-level Trans entry in the "
                     "milo, instead of relying purely on RB3's external shared-armature "
                     "system. Debugging aid for testing whether custom characters need "
                     "their own embedded skeleton. Only applies to Type=Character",
        default=False,
    )
    stock_rest_offsets: BoolProperty(
        name="Bind Offsets To Stock Rest (Experimental)",
        description="Compute each mesh's bone offset matrices against a pristine "
                     "reference copy of the base RB3 skeleton, instead of the deform "
                     "armature you actually rigged to. RB3 always skins against the "
                     "fixed shared skeleton, so if you moved or rotated bones to fit "
                     "your character's proportions, offsets taken from your edited rig "
                     "make the mesh deform wildly in-game. This keeps the REST pose "
                     "correct no matter how you edited the rig; the tradeoff is mild "
                     "animation drift (bones pivot about their stock positions), like "
                     "the GHWT Definitive Edition mod. Leave OFF if every bone is still "
                     "at its stock rest - it makes no difference then. Type=Character only",
        default=False,
    )
    reference_armature: StringProperty(
        name="Stock Reference Armature",
        description="Name of the pristine, UNEDITED base-skeleton armature to read stock "
                     "bone rest transforms from (for 'Bind Offsets To Stock Rest'). Keep "
                     "it at the same world transform as your deform rig. Leave blank to "
                     "auto-detect: an armature named like a reference "
                     "('stock'/'reference'/'_ref'), or the one armature in the scene that "
                     "deforms no mesh",
        default="",
    )
    hide_head: BoolProperty(
        name="Hide Head? (Experimental)",
        description="Embeds a CharMeshHide asset in the milo with its top-level "
                     "HIDE_HEAD flag set (the same field MiloEditor's own 'Hide Head' "
                     "checkbox edits) and an empty hides list. UNTESTED IN RB3: "
                     "searching the RB3 Wii decompilation (DarkRTA/rb3) turns up no "
                     "code anywhere that reads this flag - only RB4, a different "
                     "engine, is known to use head-hiding accessories. This option "
                     "exists to test on real hardware/Xenia whether RB3 secretly "
                     "honors it after all, or whether it's a vestigial field from "
                     "shared Milo tooling that RB3 never consumes. Only applies to "
                     "Type=Character",
        default=False,
    )
    export_hair: BoolProperty(
        name="Export Hair Physics (.hair) (Experimental)",
        description="Embeds a CharHair (.hair) asset for any armature that has "
                     "'Export CharHair' enabled in its glTFMilo Hair Setup (in the "
                     "armature Data properties). Format is confirmed against MiloLib "
                     "and the RB3 decomp, but whether hair physics actually work from "
                     "a generated .hair alone is untested - collision (.coll) files "
                     "aren't implemented yet, so collisions won't work either way. "
                     "Only applies to Type=Character",
        default=False,
    )
    export_tangents: BoolProperty(
        name="Export Vertex Tangents",
        description="Writes real per-vertex tangents instead of zeros. Normal maps need a "
                     "tangent basis (TBN); with zero tangents the basis is degenerate and "
                     "the normal map collapses back to the geometric normal, so surfaces "
                     "render FLAT - which is why normal maps appeared to do nothing while "
                     "specular maps (which don't need tangents) worked fine. The engine will "
                     "NOT generate tangents itself at RB3's mesh revision (38): the "
                     "decompiled engine only auto-generates them for revision < 30, so they "
                     "must be supplied in the file. Confirmed working in-game on Xbox 360 "
                     "(fabric wrinkles reappear). Leave ON; turn off only to A/B compare "
                     "against the old flat output. Also adds the tangent to the vertex dedup "
                     "key, which can raise the vertex count slightly at UV seams. PS3 NOTE: "
                     "the PS3 tangent format (11-11-10) has no W, so per-vertex handedness "
                     "isn't stored there - detail may invert on mirrored UV islands",
        default=True,
    )
    tbrb_force_mat_defaults: BoolProperty(
        name="Force TBRB Material Defaults (testing)",
        description="TBRB only. The Blender material UI defaults were authored for Rock "
                     "Band 3; several of those values render a TBRB character solid black "
                     "even though the .mat file is structurally correct. When ON, the "
                     "render-affecting material fields (cull, specular power, point lights, "
                     "color adjust, rim lighting, secondary specular) are overridden on "
                     "export with byte-verified known-good TBRB values, so you can confirm "
                     "the material writer itself works without hand-syncing every field in "
                     "the UI. Texture names, base color, blend mode and z-mode still come "
                     "from your material. This is a temporary testing switch - once the UI "
                     "exposes proper TBRB material values it can be turned off. Has no "
                     "effect on non-TBRB exports",
        default=True,
    )
    dc1_generate_mipmaps: BoolProperty(
        name="Generate Mipmaps (DC1)",
        description="Writes a full mip chain (down to a 16px smaller dimension, matching "
                     "every real DC1 texture - confirmed by byte-parsing retail emilia01) "
                     "for every texture instead of a single base level. Dance Central 1 "
                     "requires this for ATI2/BC5 normal "
                     "maps: a vanilla DC1 normal map ships a mip chain, and DC1 renders a "
                     "mip-less ATI2/DXN surface BLACK - the exact 'character turns black when "
                     "a normal map is present' bug. RB3 renders single-level ATI2 normal maps "
                     "fine, and DXT1/DXT5 (diffuse/spec) render mip-less on both, so this is "
                     "DC1-specific. Leave ON for DC1; turn OFF only to A/B confirm that the "
                     "mip chain is what fixes the black rendering. Has no effect for RB3",
        default=True,
    )
    dc3_force_mat_defaults: BoolProperty(
        name="Force DC3 Material Defaults",
        description="Dance Central 3 only. DC3 uses its own rev-70 material format (a "
                     "RndMat/BaseMaterial/MetaMaterial hierarchy) rather than the flat rev-68 "
                     "material RB3 and DC1 share - this is why a custom character with a normal "
                     "map renders solid black on DC3 even though the same export works on DC1. "
                     "The dedicated DC3 material writer fixes the format; this switch additionally "
                     "snaps the two render-affecting fields whose RB3 UI defaults are inverted "
                     "relative to DC3 (Pre-Lit and Use Environment) to the values every retail "
                     "Emilia material uses (prelit OFF, use-environment ON). Texture names, base "
                     "color, blend, z-mode, specular, rim and the MetaMaterial name still come "
                     "from your material. Leave ON unless you are deliberately hand-authoring "
                     "these bits. Has no effect on non-DC3 exports",
        default=True,
    )
    custom_song_texture_format: EnumProperty(
        name="Texture Format",
        description="TBRB Custom Song Asset only. Which loose file format textures are "
                     "written in",
        items=(
            ('tex', "Compiled .tex", "Write the compiled RndTex binary format directly, "
             "same as every other export path - the loose file is a ready-to-load .tex"),
            ('png', "Source .png", "Write a plain, uncompressed .png of the (resized) "
             "image instead of a compiled .tex. Matches the custom-song compiler's "
             "expected input: it takes loose .png source files and does its own PNG -> "
             ".tex conversion when it packages the song, so this skips this exporter's "
             "own DXT/BCn compression and just hands over the source image. Materials "
             "still reference the texture by its eventual '.tex' name, since that's what "
             "the finished, compiled asset the compiler produces will be called"),
        ),
        default='tex',
    )
    custom_song_mat_revision: EnumProperty(
        name="Material Revision",
        description="TBRB Custom Song Asset only. Which RndMat revision loose .mat files "
                     "are written at",
        items=(
            ('rev28', "Revision 28 (proven for custom songs)",
             "Older, simpler material format - diffuse/normal/specular map support only, "
             "no rim lighting/environment map/refraction. Byte-verified against a real "
             "reference material confirmed to load correctly through the custom-song "
             "'extras' folder pipeline. Recommended until revision 55 is confirmed working "
             "through that same pipeline"),
            ('rev55', "Revision 55 (matches vanilla game data)",
             "The same material format every other TBRB export path in this plugin uses, "
             "and byte-verified field-for-field against a real vanilla PS3 instrument "
             "material dumped from inside an actual retail milo - but NOT yet confirmed to "
             "load through the loose custom-song 'extras' folder pipeline itself. Supports "
             "rim lighting, environment map and refraction on top of diffuse/normal/"
             "specular"),
        ),
        default='rev28',
    )
    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        box = layout.box()
        box.label(text="Target", icon='CONSOLE')
        box.prop(self, "platform")
        box.prop(self, "game")
        box.prop(self, "milo_type")
        if self.milo_type == 'instrument':
            box.prop(self, "instrument_type")
        if self.milo_type == 'custom_song_asset':
            if self.game != 'tbrb':
                box.label(text="Custom Song Asset is TBRB-only - set Game to TBRB.",
                          icon='ERROR')
            else:
                sub = box.box()
                sub.label(text="Loose asset export (no milo)", icon='FILE_BLANK')
                sub.label(text="Writes .mesh/.mat/.tex into the chosen folder for the",
                          icon='INFO')
                sub.label(text="custom-song 'extras' + song.dta workflow.")
                sub.label(text="Set per-mesh parent in Object Properties > Milo Object.",
                          icon='INFO')
                sub.prop(self, "custom_song_mat_revision")
                if self.custom_song_mat_revision == 'rev28':
                    sub.label(text="Materials written at rev 28 - diffuse/normal/specular",
                              icon='INFO')
                    sub.label(text="map only, matches a confirmed-working reference file.")
                else:
                    sub.label(text="Materials written at rev 55 - matches vanilla game",
                              icon='INFO')
                    sub.label(text="data, but unconfirmed through this loose-file pipeline.")
                sub.prop(self, "custom_song_texture_format")
                if self.custom_song_texture_format == 'png':
                    sub.label(text="Textures written as loose .png source files - the",
                              icon='INFO')
                    sub.label(text="song compiler converts these to .tex at package time.")
                else:
                    sub.label(text="Textures written as compiled .tex (default, matches",
                              icon='INFO')
                    sub.label(text="every other export path).")
        if self.game in ('dc1', 'dc3'):
            sub = box.box()
            label = "Dance Central 1" if self.game == 'dc1' else "Dance Central 3"
            sub.label(text=f"{label} (experimental)", icon='ARMATURE_DATA')
            sub.prop(self, "dc1_generate_mipmaps")
            if not self.dc1_generate_mipmaps:
                sub.label(text="Mip-less ATI2 normal maps render BLACK in DC1.", icon='ERROR')
            sub.prop(self, "donor_milo")
            if self.game == 'dc3':
                sub.label(text="Container written at rev 32 to match DC3 clips.", icon='INFO')
                sub.prop(self, "dc3_force_mat_defaults")
                sub.label(text="DC3 uses its own rev-70 material format (fixes black",
                          icon='INFO')
                sub.label(text="normal-mapped surfaces). Set each material's MetaMaterial")
                sub.label(text="(.mmat) in Material Properties. Default char_basic_skin;")
                sub.label(text="clothing=char_basic_rim, hair=char_basic_hair,")
                sub.label(text="eyes/teeth=char_basic_skin_nospec.")
                sub.label(text="Avoid char_basic_color - it glows flat (no lighting).",
                          icon='ERROR')
                if not self.export_tangents:
                    sub.label(text="Tangents OFF - normal maps will be flat.", icon='ERROR')
                if self.donor_milo:
                    sub.label(text="Copies the DC3 donor's CharClipSet (win/intro anims).",
                              icon='INFO')
            elif self.donor_milo:
                sub.prop(self, "donor_mode")
                if self.donor_mode == 'inject_meshes':
                    sub.label(text="Output = donor with its meshes swapped for yours.",
                              icon='INFO')
                    sub.label(text="Name Blender objects to match donor mesh names.",
                              icon='INFO')
                else:
                    sub.label(text="Fresh milo + donor's CharClipSet as entry 0.",
                              icon='INFO')

        if self.game == 'tbrb' and self.milo_type != 'custom_song_asset':
            sub = box.box()
            sub.label(text="The Beatles: Rock Band (PS3 first)", icon='ARMATURE_DATA')
            sub.prop(self, "tbrb_target")
            if self.tbrb_target == 'skeleton':
                sub.prop(self, "tbrb_beatle")
                sub.prop(self, "tbrb_armature")
                sub.prop(self, "tbrb_standard_collisions")
                sub.prop(self, "tbrb_repair_degenerate_bones")
                if self.tbrb_repair_degenerate_bones:
                    sub.prop(self, "tbrb_reference_skeleton_milo")
                    if not self.tbrb_reference_skeleton_milo:
                        sub.label(text="Set a vanilla skeleton milo to enable the repair.",
                                  icon='ERROR')
                    else:
                        sub.label(text="Collapsed driver bones (spot_*/crease/lookat) restored "
                                       "from it.", icon='INFO')
                sub.label(text="Every bone in that armature is exported, weighted or not.",
                          icon='INFO')
                sub.label(text="Bones come from the export armature; missing retail bones "
                               "are reported.", icon='INFO')
                if self.platform != 'ps3':
                    sub.label(text="Start with PS3 - TBRB isn't testable on Xenia/360 yet.",
                              icon='INFO')
            else:
                sub.prop(self, "tbrb_mesh_kind")
                if self.tbrb_mesh_kind == 'outfit':
                    sub.prop(self, "tbrb_base_milo")
                    sub.prop(self, "tbrb_share_milo")
                else:
                    sub.prop(self, "tbrb_beatle")
                    sub.prop(self, "tbrb_skeleton_milo")
                sub.prop(self, "tbrb_test_hardcode_subdirs")
                sub.label(text="Flat milo: meshes + materials + textures, no LOD groups.",
                          icon='INFO')
                sub.label(text="sphereBase is set to bone_pelvis.mesh (matches retail).",
                          icon='INFO')
                if not self.export_tangents:
                    sub.label(text="Tangents OFF - normal maps will be flat.", icon='ERROR')

        box = layout.box()
        box.label(text="Scene", icon='SCENE_DATA')
        box.prop(self, "only_selected")

        box = layout.box()
        box.label(text="Materials & Textures", icon='MATERIAL')
        box.prop(self, "ignore_tex_size_limits")
        box.prop(self, "export_tangents")
        if self.export_tangents and self.platform == 'ps3':
            box.label(text="PS3: no per-vertex handedness (mirrored UVs may invert)",
                      icon='INFO')
        if self.game == 'tbrb':
            box.prop(self, "tbrb_force_mat_defaults")
            if self.tbrb_force_mat_defaults:
                box.label(text="TBRB render values forced; UI material fields partly ignored.",
                          icon='INFO')

        box = layout.box()
        box.label(text="Character", icon='ARMATURE_DATA')
        box.prop(self, "outfit_config")
        box.prop(self, "bone_axis_correction")
        row = box.row()
        row.enabled = (self.milo_type in ('character', 'instrument'))
        row.prop(self, "export_armature")
        if self.game in ('dc1', 'dc3'):
            row.enabled = False
            box.label(text="Dance Central always exports the full armature (no shared "
                           "skeleton).", icon='INFO')
        row = box.row()
        row.enabled = (self.milo_type == 'character')
        row.prop(self, "hide_head")
        row = box.row()
        row.enabled = (self.milo_type in ('character', 'instrument'))
        row.prop(self, "export_hair")
        row = box.row()
        row.enabled = (self.milo_type == 'character')
        row.prop(self, "stock_rest_offsets")
        row = box.row()
        row.enabled = (self.milo_type == 'character' and self.stock_rest_offsets)
        row.prop(self, "reference_armature")

    def _collect_armatures(self, context):
        """Return {name: armature_object} for every armature in the export scope. Unlike
        the mesh path (which infers armatures from mesh Armature modifiers), a skeleton-only
        export may have no meshes at all, so armature objects are gathered directly."""
        if self.only_selected and context.selected_objects:
            objs = context.selected_objects
        else:
            objs = context.scene.objects
        return {o.name: o for o in objs if o.type == 'ARMATURE'}

    def _export_tbrb_meshes(self, context):
        """Export a TBRB MESH milo: meshes + materials + textures, flat (no LOD groups).
        The companion skeleton milo must be exported separately and installed alongside -
        this milo's only subdir reference points at it, and that's how bone names resolve."""
        import os

        base, ext = os.path.splitext(self.filepath)
        if self.platform == 'ps3':
            ext = '.milo_ps3'
        elif ext.lower() != '.milo_xbox':
            ext = '.milo_xbox'
        self.filepath = base + ext
        root_name = os.path.splitext(os.path.basename(self.filepath))[0]

        skeleton_milo = (self.tbrb_skeleton_milo or "").strip()
        if not skeleton_milo:
            skeleton_milo = f"{self.tbrb_beatle}_skeleton.milo" \
                if self.tbrb_beatle != 'custom' else "george_skeleton.milo"

        # Resolve the subdirectory list by milo kind. An OUTFIT milo references the base
        # head/hands milo + a shared outfit milo (and inherits the skeleton transitively);
        # a HEAD/HANDS milo references the skeleton milo directly. sphereBase is a real bone
        # in every retail mesh milo (bone_pelvis.mesh), never the project name.
        sphere_base = "bone_pelvis.mesh"
        if self.tbrb_test_hardcode_subdirs:
            # Isolation test: emit the EXACT subdirs from the retail cavern02 outfit milo,
            # so "are the subdirs the crash?" can be answered without any other variable.
            sub_dirs = ["../george_headhands_short.milo",
                        "../../shared/outfit/cavern02_share.milo"]
            _log("  TEST MODE: hardcoding retail cavern02 subdirs "
                 "(../george_headhands_short.milo, ../../shared/outfit/cavern02_share.milo).")
        elif self.tbrb_mesh_kind == 'outfit':
            base_milo = (self.tbrb_base_milo or "").strip() or "../george_headhands_long.milo"
            share_milo = (self.tbrb_share_milo or "").strip()
            sub_dirs = [base_milo] + ([share_milo] if share_milo else [])
            _log(f"  Outfit subdirs: {sub_dirs}")
        else:
            sub_dirs = [skeleton_milo]
            _log(f"  Head/hands subdir: {sub_dirs}")

        _log(f"===== Starting TBRB MESH export: '{root_name}' -> {self.filepath} "
             f"(platform={self.platform}, kind={self.tbrb_mesh_kind}) =====")
        _log(f"  sphereBase='{sphere_base}'")
        if self.platform != 'ps3':
            _log("  NOTE: PS3 is the verified TBRB target so far.")

        _log("--- Meshes ---")
        try:
            entries, materials_by_name, armatures_used = collect_mesh_entries(
                context, root_name, only_selected=self.only_selected,
                bone_axis_correction=self.bone_axis_correction,
                offset_multiply_order=self.offset_multiply_order,
                stock_reference_armature=None,
                platform=self.platform,
                include_tangents=self.export_tangents)
        except BoneLimitExceeded as e:
            details = "; ".join(f"'{n}': {c} bones (max {MAX_BONES_PER_MESH})"
                                 for n, c in e.offenders)
            _log(f"EXPORT FAILED: bone limit exceeded - {details}")
            self.report({'ERROR'},
                        f"Export blocked - exceeds the {MAX_BONES_PER_MESH}-bone-per-mesh "
                        f"limit (this crashes the game): {details}.")
            return {'CANCELLED'}
        except ValueError as e:
            _log(f"EXPORT FAILED: {e}")
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        if not entries:
            _log("EXPORT FAILED: no mesh objects found.")
            self.report({'ERROR'}, "No mesh objects found to export.")
            return {'CANCELLED'}

        if not self.export_tangents:
            _log("  WARNING: tangents are DISABLED. At mesh revision 33 tangents are plain "
                 "floats and are written correctly, so there's no reason to turn them off "
                 "here - normal maps will render FLAT without them.")

        _log("--- Materials & Textures ---")
        # mip_floor=16: every retail TBRB texture bottoms out at a 16px smaller dimension,
        # never 4. generate_mips is forced on - retail ships full chains for every texture.
        materials, textures, _texture_images = gather_materials_and_textures(
            materials_by_name, max_tex_size=512,
            ignore_tex_size_limits=self.ignore_tex_size_limits,
            platform=self.platform, generate_mips=True,
            mip_floor=TBRB_TEX_MIP_FLOOR)
        for (tname, tw, th, enc, bpp, _blk, nmips) in textures:
            _log(f"  Tex '{tname}': {tw}x{th} bpp={bpp} enc={enc} mipMaps={nmips}")

        _log("--- Writing mesh container ---")
        data = build_tbrb_mesh_milo_bytes(
            root_name, entries, materials, textures,
            skeleton_milo_name=skeleton_milo,
            platform=self.platform, write_tangents=self.export_tangents,
            sub_dirs=sub_dirs, sphere_base=sphere_base,
            force_tbrb_mat_defaults=self.tbrb_force_mat_defaults)

        with open(self.filepath, 'wb') as f:
            f.write(data)

        total_verts = sum(len(e[4]) for e in entries)
        total_faces = sum(len(e[5]) for e in entries)
        summary = (f"Exported TBRB mesh milo '{root_name}': {len(entries)} mesh(es), "
                   f"{len(materials)} material(s), {len(textures)} texture(s), "
                   f"{total_verts} verts, {total_faces} tris -> {self.filepath} "
                   f"({len(data)} bytes)")
        _log(f"===== SUCCESS: {summary} =====")
        self.report({'INFO'}, summary)
        return {'FINISHED'}

    def _export_tbrb_custom_song_assets(self, context):
        """Export loose standalone .mesh / .mat / .tex files (no milo container).

        The TBRB custom-song community loads models by dropping loose milo assets into an
        'extras' folder that the packaging tool mounts into $world, then wiring them in with
        song.dta scripting (set_trans_parent onto a bone, add_object into a draw group,
        set diffuse_tex to rebind textures). This export produces exactly those loose files:
        one .mesh per Blender mesh, one .mat per material, one .tex per texture, each a
        complete standalone asset (revision + body + 0xADDEADDE), written into the folder the
        user picked in the export dialog. No directory container, no LOD groups, no subdir
        references - the packaging tool and the DTA do the assembly.

        Assets are written at the TBRB revisions (mesh 34, mat 55, tex 10) so they match what
        a TBRB milo carries and load the same way when mounted. The per-object 'Milo Object >
        Trans Parent' setting is honoured: if set, that mesh's RndTrans parentObj is baked to
        the named in-game bone (e.g. bone_piano_base.mesh), so the asset rides that bone the
        instant it loads without the DTA needing a set_trans_parent call."""
        import os

        out_dir = os.path.dirname(self.filepath)
        if not out_dir:
            out_dir = os.getcwd()

        _log(f"===== Starting TBRB CUSTOM SONG ASSET export (loose files) -> {out_dir} "
             f"(platform={self.platform}) =====")
        if self.platform != 'ps3':
            _log("  NOTE: PS3 is the verified TBRB target so far.")

        _log("--- Meshes ---")
        # root_name is only used as the default Trans parent for meshes with no per-object
        # override. For a loose asset there is no milo root dir to parent to, so default to
        # empty - a mesh with no explicit parent stays unparented until the DTA wires it in,
        # exactly like the community's existing set_trans_parent workflow. A per-object Trans
        # Parent override (e.g. bone_piano_base.mesh) replaces this.
        loose_root = ""
        try:
            entries, materials_by_name, armatures_used = collect_mesh_entries(
                context, loose_root, only_selected=self.only_selected,
                bone_axis_correction=self.bone_axis_correction,
                offset_multiply_order=self.offset_multiply_order,
                stock_reference_armature=None,
                platform=self.platform,
                include_tangents=self.export_tangents)
        except BoneLimitExceeded as e:
            details = "; ".join(f"'{n}': {c} bones (max {MAX_BONES_PER_MESH})"
                                 for n, c in e.offenders)
            _log(f"EXPORT FAILED: bone limit exceeded - {details}")
            self.report({'ERROR'},
                        f"Export blocked - exceeds the {MAX_BONES_PER_MESH}-bone-per-mesh "
                        f"limit (this crashes the game): {details}.")
            return {'CANCELLED'}
        except ValueError as e:
            _log(f"EXPORT FAILED: {e}")
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        if not entries:
            _log("EXPORT FAILED: no mesh objects found.")
            self.report({'ERROR'}, "No mesh objects found to export.")
            return {'CANCELLED'}

        if not self.export_tangents:
            _log("  WARNING: tangents are DISABLED - normal maps will render FLAT.")

        _log("--- Materials & Textures ---")
        materials, textures, texture_images = gather_materials_and_textures(
            materials_by_name, max_tex_size=512,
            ignore_tex_size_limits=self.ignore_tex_size_limits,
            platform=self.platform, generate_mips=True,
            mip_floor=TBRB_TEX_MIP_FLOOR)

        # --- write each asset as its own loose file ---
        written = []

        def _safe_filename(name):
            # Entry names already carry the extension (.mesh/.mat/.tex) and are sanitized to
            # milo-symbol-safe characters, which are also filesystem-safe. Guard anyway.
            return name.replace("/", "_").replace("\\", "_")

        _log("--- Writing loose assets ---")
        # Textures first (materials reference them by name; order on disk doesn't matter to
        # the game, but writing them first keeps the log readable).
        write_png = (self.custom_song_texture_format == 'png')
        for (tname, tw, th, enc, bpp, blk, nmips) in textures:
            if write_png:
                # Source .png: hand the compiled-name entry a .png extension instead, and
                # write real image bytes via Blender's own PNG writer rather than this
                # exporter's compiled RndTex/DXT format. The .mat still references
                # '<name>.tex' (unchanged) - that's the name the song compiler's own
                # PNG -> .tex conversion is expected to produce.
                png_name = tname[:-4] + ".png" if tname.lower().endswith(".tex") else tname + ".png"
                path = os.path.join(out_dir, _safe_filename(png_name))
                image = texture_images.get(tname)
                if image is None:
                    _log(f"  Tex  '{tname}': FAILED - no source image found for PNG export, "
                         f"skipping.")
                    self.report({'WARNING'}, f"Could not export '{tname}' as PNG - "
                                              f"source image not found.")
                    continue
                pw, ph = export_texture_as_png(image, path, max_size=512,
                                                ignore_size_limits=self.ignore_tex_size_limits)
                written.append(path)
                _log(f"  Tex  '{tname}': {pw}x{ph} -> '{os.path.basename(path)}' (png, "
                     f"{os.path.getsize(path)} bytes)")
            else:
                w = MiloWriter(big_endian=True)
                write_tbrb_rnd_tex(w, tw, th, enc, bpp, blk,
                                    external_path="", platform=self.platform, num_mips=nmips)
                path = os.path.join(out_dir, _safe_filename(tname))
                with open(path, 'wb') as f:
                    f.write(w.buf)
                written.append(path)
                _log(f"  Tex  '{tname}': {tw}x{th} bpp={bpp} enc={enc} mipMaps={nmips} "
                     f"-> {len(w.buf)} bytes")

        use_rev28_mat = (self.custom_song_mat_revision == 'rev28')
        for (mat_entry_name, settings) in materials:
            w = MiloWriter(big_endian=True)
            if use_rev28_mat:
                write_tbrb_custom_song_rnd_mat(w, settings, standalone=True)
            else:
                write_tbrb_rnd_mat(w, settings, standalone=True,
                                    force_tbrb_defaults=self.tbrb_force_mat_defaults)
            path = os.path.join(out_dir, _safe_filename(mat_entry_name))
            with open(path, 'wb') as f:
                f.write(w.buf)
            written.append(path)
            rev_note = "rev 28" if use_rev28_mat else "rev 55"
            _log(f"  Mat  '{mat_entry_name}' ({rev_note}): -> {len(w.buf)} bytes")

        for (entry_name, local_xfm, world_xfm, parent_obj, vertices, faces,
             bone_transforms, mat_name, needs_ao_calc) in entries:
            w = MiloWriter(big_endian=True)
            write_tbrb_rnd_mesh(w, entry_name, local_xfm, world_xfm, parent_obj,
                                 vertices, faces, mat_name=mat_name,
                                 bone_transforms=bone_transforms,
                                 force_white_vertex_color=needs_ao_calc,
                                 write_tangents=self.export_tangents)
            path = os.path.join(out_dir, _safe_filename(entry_name))
            with open(path, 'wb') as f:
                f.write(w.buf)
            written.append(path)
            parent_note = f", parent '{parent_obj}'" if parent_obj else ", no parent (wire in via DTA)"
            _log(f"  Mesh '{entry_name}': {len(vertices)} verts, {len(faces)} tris"
                 f"{parent_note} -> {len(w.buf)} bytes")

        summary = (f"Exported {len(entries)} loose .mesh, {len(materials)} .mat, "
                   f"{len(textures)} .tex file(s) -> {out_dir}")
        _log(f"===== SUCCESS: {summary} =====")
        self.report({'INFO'}, summary)
        return {'FINISHED'}

    def _resolve_tbrb_armature(self, context):
        """Resolve EXACTLY ONE armature to build the skeleton from, and say which.

        A skeleton milo is one skeleton. Silently merging every armature in the scene is how
        bones go missing: the merge de-duplicates by name, so a stale `Armature_old` sitting
        in the file can shadow the live rig's bones. Preference order is explicit name ->
        active object -> selected -> the scene's only armature. Anything ambiguous is an
        error the user resolves, not a guess.

        Returns (armature_object, error_message)."""
        scene_arms = [o for o in context.scene.objects if o.type == 'ARMATURE']
        if not scene_arms:
            return None, ("No armature found in the scene. A TBRB skeleton milo is built "
                          "from an armature's bones.")

        wanted = (self.tbrb_armature or "").strip()
        if wanted:
            for o in scene_arms:
                if o.name == wanted:
                    return o, None
            return None, (f"Armature '{wanted}' not found. Available: "
                          f"{', '.join(o.name for o in scene_arms)}")

        active = getattr(context.view_layer.objects, "active", None)
        if active is not None and active.type == 'ARMATURE':
            return active, None

        selected = [o for o in getattr(context, "selected_objects", []) or []
                    if o.type == 'ARMATURE']
        if len(selected) == 1:
            return selected[0], None
        if len(selected) > 1:
            return None, (f"{len(selected)} armatures selected. Select just one, make it "
                          f"active, or name it in 'Skeleton Armature'.")

        if len(scene_arms) == 1:
            return scene_arms[0], None
        return None, (f"The scene has {len(scene_arms)} armatures "
                      f"({', '.join(o.name for o in scene_arms)}). Make the one you want "
                      f"active, or name it in 'Skeleton Armature' - merging them would "
                      f"silently drop bones.")

    def _export_tbrb(self, context):
        """The Beatles: Rock Band export dispatch. Currently only the SKELETON milo
        (bones + collision) is implemented; the mesh milo is the next phase."""
        import os

        if self.platform not in ('xbox360', 'ps3'):
            self.report({'ERROR'}, "TBRB export supports Platform = PS3 (preferred) or "
                                    "Xbox 360.")
            return {'CANCELLED'}

        # Custom Song Asset export produces loose .mesh/.mat/.tex files regardless of the
        # TBRB Milo (skeleton/meshes) selector - it never builds a container.
        if self.milo_type == 'custom_song_asset':
            return self._export_tbrb_custom_song_assets(context)

        if self.tbrb_target == 'meshes':
            return self._export_tbrb_meshes(context)

        # Normalize output extension to the platform, then apply the Beatle filename.
        base, ext = os.path.splitext(self.filepath)
        if self.platform == 'ps3':
            ext = '.milo_ps3'
        elif ext.lower() not in ('.milo_xbox',):
            ext = '.milo_xbox'
        if self.tbrb_beatle != 'custom':
            base = os.path.join(os.path.dirname(base), f"{self.tbrb_beatle}_skeleton")
        self.filepath = base + ext
        root_name = os.path.splitext(os.path.basename(self.filepath))[0]

        _log(f"===== Starting TBRB SKELETON export: '{root_name}' -> {self.filepath} "
             f"(platform={self.platform}, beatle={self.tbrb_beatle}) =====")
        if self.platform != 'ps3':
            _log("  NOTE: exporting for Xbox 360, but TBRB isn't testable on Xenia/360 yet - "
                 "PS3 is the verified target.")

        armature_obj, err = self._resolve_tbrb_armature(context)
        if armature_obj is None:
            _log(f"EXPORT FAILED: {err}")
            self.report({'ERROR'}, err)
            return {'CANCELLED'}
        armatures_used = {armature_obj.name: armature_obj}
        _log(f"Skeleton armature: '{armature_obj.name}'")

        _log("--- Skeleton bones (Trans) ---")
        # EVERY bone, unconditionally. A skeleton milo IS the skeleton: unweighted driver and
        # facial bones (creases, footik, head_lookat, eyes) are resolved by name at runtime and
        # a missing one crashes on song start.
        try:
            bone_trans_entries = build_all_bone_trans_entries(
                armature_obj, root_name, label="TBRB skeleton")
        except ValueError as e:
            _log(f"EXPORT FAILED: {e}")
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        # Sort alphabetically to match how the retail skeleton is laid out (order is
        # name-resolved at runtime, so this is convention/tidiness, not correctness).
        bone_trans_entries.sort(key=lambda e: e[0])
        if not bone_trans_entries:
            _log("EXPORT FAILED: armature has no bones to export.")
            self.report({'ERROR'}, "The armature has no bones to export.")
            return {'CANCELLED'}

        # --- Bone completeness check against the retail skeleton -------------------
        # A skeleton milo missing bones LOADS FINE and then crashes once a song starts,
        # because the animation/look-at systems resolve driver bones by name and the retail
        # meshes' inverse-bind matrices are baked against the full bone set. Warn loudly.
        exported_bone_names = {b[0] for b in bone_trans_entries}
        missing_bones = sorted(TBRB_REFERENCE_SKELETON_BONES - exported_bone_names)
        if missing_bones:
            _log(f"  WARNING: {len(missing_bones)} bone(s) present in the retail TBRB "
                 f"skeleton are MISSING from this armature:")
            for n in missing_bones:
                _log(f"    - {n}")
            _log("  Missing driver bones (footik / head_lookat / eyes) are a known cause of "
                 "a crash on song start. Re-import the armature so it carries every bone.")
            self.report({'WARNING'},
                        f"{len(missing_bones)} retail bone(s) missing from the armature "
                        f"(e.g. {', '.join(missing_bones[:3])}) - this commonly crashes on "
                        f"song start. See the system console for the full list.")
        else:
            _log("  Bone check: armature contains every bone the retail TBRB skeleton has.")
        extra_bones = sorted(exported_bone_names - TBRB_REFERENCE_SKELETON_BONES)
        # A bone whose name is one ".mesh" away from a retail bone is almost always a typo,
        # not a custom bone. It's a silent double failure: the vertex group (named with the
        # suffix) never binds to the bone, AND the exported Trans has a name nothing in the
        # game looks up - so the bone is effectively still missing.
        suffix_typos = [n for n in extra_bones
                        if f"{n}.mesh" in TBRB_REFERENCE_SKELETON_BONES]
        if suffix_typos:
            _log(f"  WARNING: {len(suffix_typos)} bone(s) look like they are missing the "
                 f"'.mesh' suffix:")
            for n in suffix_typos:
                _log(f"    - '{n}'  should probably be  '{n}.mesh'")
            _log("  Rename these in Blender. As-is the vertex group (which DOES carry the "
                 "suffix) won't bind to the bone, and the game won't find the bone either.")
            self.report({'WARNING'},
                        f"{len(suffix_typos)} bone(s) appear to be missing the '.mesh' "
                        f"suffix (e.g. '{suffix_typos[0]}' -> '{suffix_typos[0]}.mesh'). "
                        f"See the system console.")
        other_extra = [n for n in extra_bones if n not in suffix_typos]
        if other_extra:
            _log(f"  NOTE: {len(other_extra)} bone(s) are not in the retail skeleton "
                 f"(custom bones are fine, but typos here silently do nothing): "
                 f"{', '.join(other_extra)}")

        # --- Repair origin-collapsed driver bones from a reference skeleton ---------
        if self.tbrb_repair_degenerate_bones and self.tbrb_reference_skeleton_milo:
            ref_path = bpy.path.abspath(self.tbrb_reference_skeleton_milo)
            _log(f"--- Collapsed-bone repair (reference: {ref_path}) ---")
            reference_trans = _read_skeleton_milo_trans(ref_path)
            if not reference_trans:
                _log("  WARNING: could not parse the reference skeleton milo (wrong file, "
                     "compressed, or non-PS3). No repair applied.")
                self.report({'WARNING'},
                            "Reference skeleton milo could not be parsed - collapsed-bone "
                            "repair was skipped. See the system console.")
            else:
                before = {b[0]: _trans_world_translation(b[1:]) for b in bone_trans_entries}
                bone_trans_entries, repaired = repair_degenerate_skeleton_bones(
                    bone_trans_entries, reference_trans, root_name)
                if repaired:
                    _log(f"  Repaired {len(repaired)} origin-collapsed driver bone(s) from the "
                         f"reference skeleton:")
                    after = {b[0]: _trans_world_translation(b[1:]) for b in bone_trans_entries}
                    for n in repaired:
                        bx = before.get(n, (0, 0, 0)); ax = after.get(n, (0, 0, 0))
                        _log(f"    {n:28s} world ({bx[0]:+.3f},{bx[1]:+.3f},{bx[2]:+.3f}) "
                             f"-> ({ax[0]:+.3f},{ax[1]:+.3f},{ax[2]:+.3f})")
                    self.report({'INFO'},
                                f"Repaired {len(repaired)} collapsed driver bone(s) from the "
                                f"reference skeleton.")
                else:
                    _log("  No origin-collapsed bones found - nothing to repair (rig is clean "
                         "or bones already positioned).")
        elif self.tbrb_repair_degenerate_bones:
            _log("--- Collapsed-bone repair: ENABLED but no Reference Skeleton Milo set - "
                 "skipped. Point it at the vanilla george_skeleton.milo_ps3 to enable. ---")

        # --- Collision volumes ------------------------------------------------------
        if self.tbrb_standard_collisions:
            char_collide_entries = build_tbrb_standard_collision_entries(exported_bone_names)
            _log(f"--- Collision volumes (CharCollide) ---\n"
                 f"Emitting the {len(char_collide_entries)} retail TBRB head/neck volume(s) "
                 f"(byte-exact copies of george's, including local transforms):")
        else:
            char_collide_entries = build_char_collide_entries(armatures_used, root_name)
            _log("--- Collision volumes (CharCollide) ---\n"
                 "Using per-bone authored volumes (retail standard volumes are OFF).")
        char_collide_entries.sort(key=lambda e: e[0])
        if char_collide_entries:
            for ename, coll in char_collide_entries:
                _log(f"  {ename}: shape={coll['shape']} r0={coll['radius0']} "
                     f"-> parent '{coll['parent']}'")
        else:
            _log("  NONE. The retail george skeleton ships 4 (face/forehead/head/neck); "
                 "hair and cloth in the outfit milo collide against these by name.")

        _log("--- Writing skeleton container ---")
        data = build_tbrb_skeleton_milo_bytes(root_name, bone_trans_entries,
                                               char_collide_entries)
        with open(self.filepath, 'wb') as f:
            f.write(data)

        summary = (f"Exported TBRB skeleton '{root_name}': {len(bone_trans_entries)} bone(s), "
                   f"{len(char_collide_entries)} collision volume(s) -> {self.filepath} "
                   f"({len(data)} bytes)")
        _log(f"===== SUCCESS: {summary} =====")
        self.report({'INFO'}, summary)
        return {'FINISHED'}

    def execute(self, context):
        if self.game == 'tbrb':
            return self._export_tbrb(context)
        if self.game not in ('rb3', 'dc1', 'dc3'):
            self.report(
                {'ERROR'},
                "Only Game=Rock Band 3, Dance Central 1 and Dance Central 3 are "
                "implemented so far."
            )
            return {'CANCELLED'}
        if self.platform not in ('xbox360', 'ps3'):
            self.report(
                {'ERROR'},
                "Only Platform=Xbox 360 and PS3 are implemented so far."
            )
            return {'CANCELLED'}
        if self.game in ('dc1', 'dc3') and self.platform != 'xbox360':
            self.report(
                {'ERROR'},
                "Dance Central was an Xbox 360 exclusive - set Platform to Xbox 360."
            )
            return {'CANCELLED'}
        if self.milo_type == 'custom_song_asset':
            self.report(
                {'ERROR'},
                "Type=Custom Song Asset is for The Beatles: Rock Band only - set Game to "
                "The Beatles: Rock Band."
            )
            return {'CANCELLED'}
        if self.milo_type not in ('other', 'character', 'instrument'):
            self.report(
                {'ERROR'},
                "Type=Venue isn't implemented yet - only Other, Character, and Instrument."
            )
            return {'CANCELLED'}

        import os
        # Normalize the output extension to match the target platform. ExportHelper's
        # fixed filename_ext is .milo_xbox; swap it to .milo_ps3 for PS3 so the file
        # name reflects what it actually is (and so MiloLib/tools pick the right platform).
        base, ext = os.path.splitext(self.filepath)
        if self.platform == 'ps3' and ext.lower() in ('.milo_xbox', ''):
            self.filepath = base + '.milo_ps3'
        elif self.platform == 'xbox360' and ext.lower() == '.milo_ps3':
            self.filepath = base + '.milo_xbox'

        root_name = os.path.splitext(os.path.basename(self.filepath))[0]

        _log(f"===== Starting Milo export: '{root_name}' -> {self.filepath} "
             f"(type={self.milo_type}, platform={self.platform}, game={self.game}) =====")

        stock_ref_arm = None
        if self.stock_rest_offsets:
            stock_ref_arm = _resolve_stock_reference_armature(
                context, self.reference_armature, self.only_selected)
            if stock_ref_arm is not None:
                _log(f"Stock-rest offsets ON - reading bind rest from reference armature "
                     f"'{stock_ref_arm.name}'")
            else:
                _log("Stock-rest offsets ON but no reference armature could be resolved - "
                     "falling back to the deform rig's own rest (no change). Name one in "
                     "'Stock Reference Armature', or add a pristine base-skeleton copy.")
                self.report({'WARNING'},
                            "Stock-rest offsets: no reference armature found; used the "
                            "deform rig's rest instead (see the export log).")

        if self.export_tangents:
            if self.platform == 'ps3':
                _log("Vertex tangents: ENABLED - real per-vertex tangents will be written. "
                     "PS3's 11-11-10 tangent has no W, so per-vertex handedness isn't stored; "
                     "normal-map detail may invert on mirrored UV islands.")
            else:
                _log("Vertex tangents: ENABLED - real per-vertex tangents (with raw Blender "
                     "handedness) will be written; tangent + handedness included in the vertex "
                     "dedup key.")
        else:
            _log("Vertex tangents: DISABLED - writing zero tangents. Normal maps will render "
                 "FLAT (degenerate TBN basis). Only use this to A/B compare.")

        _log("--- Meshes ---")
        try:
            entries, materials_by_name, armatures_used = collect_mesh_entries(
                context, root_name, only_selected=self.only_selected,
                bone_axis_correction=self.bone_axis_correction,
                offset_multiply_order=self.offset_multiply_order,
                stock_reference_armature=stock_ref_arm,
                platform=self.platform,
                include_tangents=self.export_tangents)
        except BoneLimitExceeded as e:
            details = "; ".join(f"'{name}': {count} bones (max {MAX_BONES_PER_MESH})"
                                 for name, count in e.offenders)
            _log(f"EXPORT FAILED: bone limit exceeded - {details}")
            self.report(
                {'ERROR'},
                f"Export blocked - exceeds the {MAX_BONES_PER_MESH}-bone-per-mesh "
                f"limit (this crashes the game): {details}. Split the offending "
                f"mesh(es) into separate objects so each stays under the limit."
            )
            return {'CANCELLED'}
        except ValueError as e:
            _log(f"EXPORT FAILED: {e}")
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        if not entries:
            _log("EXPORT FAILED: no mesh objects found in the scene to export.")
            self.report({'WARNING'}, "No mesh objects found in the scene to export.")
            return {'CANCELLED'}

        bone_trans_entries = []
        emitted_bone_names = set()
        # DC1 has NO shared skeleton milo - the character must carry its own complete bone
        # hierarchy - so armature export is forced on and nothing is filtered out. This
        # matters a lot: RB3_SKELETON_BONES contains 490 RB3 bone names, and DC1 uses the
        # same Harmonix naming convention, so the RB3 filter would silently drop ~100 of a
        # DC1 character's ~136 bones and ship a nearly-empty skeleton.
        force_full_armature = (self.game in ('dc1', 'dc3'))
        # Hair physics bones: whether their local rotation gets flattened to identity.
        # Driven by the per-game table in physics_exporter (HAIR_GAME_PROFILES) so each
        # game's .hair quirks live in one place - see that table for the invariant this
        # has to preserve and which retail files it was checked against.
        zero_hair_rotation = hair_profile_for_game(self.game)['zero_rotation']
        want_armature = (self.export_armature or force_full_armature)
        if want_armature and self.milo_type in ('character', 'instrument') and armatures_used:
            _log("--- Armature ---")
            if force_full_armature and not self.export_armature:
                _log("  DC1: forcing armature export on (DC1 characters must embed their "
                     "own skeleton).")
            # IMPORTANT: RB3_SKELETON_BONES is the COMBINED RB3 skeleton and it INCLUDES
            # the guitar rig bones (bone_target_strum.mesh, spot_neck_fret01.mesh, etc.).
            # For an RB3 character we skip those names because char_shared.milo already
            # provides them - but an instrument does NOT get char_shared, and neither does
            # a DC1 character, so for those we must NOT skip or the rig comes out empty.
            skip_shared = (self.milo_type == 'character' and not force_full_armature)
            # Hair physics bones are emitted by a separate pass further down so that the
            # rotation treatment applied to them stays in sync with the matrices written
            # into the .hair file. The hard rule (byte-verified against retail files) is
            # that a strand's base/root matrix must EQUAL its root bone's Trans local
            # rotation. Two settings satisfy that; mixing them is what twists strands:
            #   RB3/TBRB - zero the bones' rotation and write identity matrices.
            #   DC1/DC3  - keep the bones' real rotation and write the real matrix, which
            #              is what retail Dance Central ships and what preserves the
            #              orientations the rig was authored with. Zeroing here instead
            #              visibly collapses every physics bone onto its parent's
            #              direction, which is wrong for these games.
            # Either way the bones must not be claimed by the general pass below with
            # untracked rotation, or the two sides silently disagree.
            hair_bones_reserved = set()
            if (self.export_hair and self.milo_type in ('character', 'instrument')
                    and armatures_used):
                hair_bones_reserved = _hair_bone_names(armatures_used)
                if hair_bones_reserved:
                    _log(f"  Holding back {len(hair_bones_reserved)} hair physics bone(s) "
                         f"from the general armature pass - they are emitted separately "
                         f"with rotation handling matched to the .hair matrices "
                         f"({'zeroed' if zero_hair_rotation else 'real rotations kept'}).")
            bone_trans_entries = build_armature_trans_entries(
                armatures_used, root_name, skip_skeleton_bones=skip_shared,
                exclude_bones=hair_bones_reserved,
                label=("DC1 armature (all bones)" if force_full_armature else
                       "instrument armature" if self.milo_type == 'instrument' else "armature"))
            emitted_bone_names = {b[0] for b in bone_trans_entries}
            if force_full_armature:
                total_bones = sum(len(a.data.bones) for a in armatures_used.values())
                # Hair bones held back above are emitted by the zero_rotation pass, so
                # they aren't missing - just counted in the other pass.
                expected = total_bones - len(hair_bones_reserved)
                _log(f"  DC1: emitted {len(bone_trans_entries)} of {expected} armature "
                     f"bone(s) as Trans entries (no skeleton filtering applied"
                     f"{f'; {len(hair_bones_reserved)} hair bone(s) deferred' if hair_bones_reserved else ''}).")
                if len(bone_trans_entries) != expected:
                    _log(f"  WARNING: {expected - len(bone_trans_entries)} bone(s) were "
                         f"not emitted - duplicates across armatures are only written once.")

        char_mesh_hide_entries = []
        if self.hide_head and self.milo_type == 'character':
            hide_entry_name = f"{sanitize_milo_name(root_name)}.hide"
            _log(f"--- CharMeshHide ---\nExporting '{hide_entry_name}' "
                 f"(HIDE_HEAD flag, untested in RB3, hides list empty)...")
            char_mesh_hide_entries = [
                (hide_entry_name, CharMeshHideOptions.HIDE_HEAD, [])
            ]
            _log(f"  Exported '{hide_entry_name}'.")

        char_hair_entries = []
        if self.export_hair and self.milo_type in ('character', 'instrument') and armatures_used:
            _log("--- CharHair (.hair) ---")
            char_hair_entries = build_char_hair_entries(armatures_used, root_name,
                                                        zero_rotation=zero_hair_rotation)
            for (hair_entry_name, hair_settings, strands) in char_hair_entries:
                total_points = sum(len(s['points']) for s in strands)
                _log(f"Created CharHair '{hair_entry_name}': {len(strands)} strand(s), "
                     f"{total_points} point(s) total, simulate={hair_settings['simulate']}, "
                     f"wind={hair_settings['wind']}.")
                for s in strands:
                    _log(f"  Strand root '{s['root']}': "
                         f"{' -> '.join(p['bone'] for p in s['points'])}")
            if not char_hair_entries:
                _log("  No CharHair exported (no armature had 'Export CharHair' enabled "
                     "with at least one strand).")
            else:
                # A .hair file drives bones by name, so those bones MUST exist as real
                # Trans entries in the milo or the hair has nothing to move. Emit Trans
                # entries for exactly the hair physics bones (skipping any bone already
                # emitted above by Export Armature). This is what the glTFMilo CLI does
                # automatically when it imports the armature.
                hair_bones = _hair_bone_names(armatures_used)
                hair_bones_to_emit = hair_bones - emitted_bone_names
                if hair_bones_to_emit:
                    # For RB3 characters, base-skeleton hair bones come from
                    # char_shared.milo, so they're skipped (skip_skeleton_bones=True).
                    # Instruments get colorpalettes.milo and DC1 characters get no shared
                    # skeleton at all, so both must emit every hair bone verbatim - even
                    # ones whose names appear in the combined RB3_SKELETON_BONES list.
                    skip_shared = (self.milo_type == 'character' and not force_full_armature)
                    _log(f"--- Hair physics bones (Trans) ---\n"
                         f"Emitting up to {len(hair_bones_to_emit)} hair bone(s) as Trans "
                         f"entries so the .hair file has bones to drive.")
                    hair_bone_entries = build_armature_trans_entries(
                        armatures_used, root_name, only_bones=hair_bones_to_emit,
                        skip_skeleton_bones=skip_shared, label="hair bones",
                        zero_rotation=zero_hair_rotation)
                    for entry in hair_bone_entries:
                        _log(f"  Bone '{entry[0]}' -> parent '{entry[3]}'")
                    bone_trans_entries = bone_trans_entries + hair_bone_entries
                    emitted_bone_names |= {b[0] for b in hair_bone_entries}
                    if skip_shared:
                        # Only relevant for characters - flag base-skeleton hair bones that
                        # weren't re-emitted (they should already exist in the shared armature).
                        skipped = hair_bones & RB3_SKELETON_BONES
                        if skipped:
                            _log(f"  NOTE: {len(skipped)} hair bone(s) match base-skeleton "
                                 f"names and were not re-emitted (they should already exist "
                                 f"in the shared armature): {', '.join(sorted(skipped))}")

        outfit_config_entries = []
        if self.milo_type == 'instrument':
            # An instrument MUST carry an OutfitConfig or the game hard-asserts (crashes)
            # when you select it and the customize/color panel loads
            # (ChooseColorPanel::Load -> MILO_ASSERT(mCurrentOutfitConfig, 0x21)). The game
            # looks it up by an INSTRUMENT-SPECIFIC name, not always "guitar.cfg": ClosetMgr
            # calls GetConfigNameFromAssetType, which maps guitar->guitar.cfg, bass->bass.cfg,
            # drum->drum.cfg (confirmed in the RB3 decomp and by byte-diffing a real vanilla
            # bass, which carries bass.cfg). Emitting the wrong name (e.g. guitar.cfg on a
            # bass) reproduces the exact crash-on-select. We emit a minimal dummy config for
            # now; real color palettes / mat swaps can be authored into it later.
            cfg_name = INSTRUMENT_CFG_NAMES.get(self.instrument_type, "guitar.cfg")
            outfit_config_entries = [(cfg_name, (0, 1, 2), True)]
            _log(f"--- OutfitConfig (.cfg) ---\nEmitting dummy '{cfg_name}' for "
                 f"instrument_type='{self.instrument_type}' (colors=[0,1,2], computeAO=True, "
                 f"all lists empty) so the instrument doesn't crash on select.")

        # CharCollide (.coll) volumes - gathered from any bone with collision enabled in
        # the Bone properties "Milo Collision" panel. Independent of the export type; if a
        # bone has a volume authored, it gets emitted.
        char_collide_entries = []
        if armatures_used:
            char_collide_entries = build_char_collide_entries(armatures_used, root_name)
            if char_collide_entries:
                _log(f"--- CharCollide (.coll) ---\nEmitting {len(char_collide_entries)} "
                     f"collision volume(s):")
                for ename, coll in char_collide_entries:
                    _log(f"  {ename}: shape={coll['shape']} r0={coll['radius0']} "
                         f"r1={coll['radius1']} l0={coll['length0']} l1={coll['length1']} "
                         f"-> parent '{coll['parent']}'")

        # DC1 donor handling. Two modes:
        #   inject_meshes - rebuild the DONOR with its meshes swapped for ours, keeping
        #                   everything else byte-identical (this bypasses build_milo_bytes
        #                   entirely, so none of our directory-body assumptions apply).
        #   charclipset   - build fresh as usual, copying only the donor's CharClipSet.
        raw_entries = []
        donor_path = None
        if self.game in ('dc1', 'dc3') and self.donor_milo:
            import os as _os
            donor_path = bpy.path.abspath(self.donor_milo)
            if not _os.path.isfile(donor_path):
                _log(f"EXPORT FAILED: donor milo not found: {donor_path}")
                self.report({'ERROR'}, f"Donor milo not found: {donor_path}")
                return {'CANCELLED'}
        elif self.game in ('dc1', 'dc3'):
            _log("No donor milo set - exporting a fresh milo with no donor data.")

        if self.game == 'dc3' and donor_path and self.donor_mode == 'inject_meshes':
            self.report(
                {'ERROR'},
                "DC3 donor support currently covers 'Copy CharClipSet Only'. Mesh "
                "injection into a DC3 donor isn't implemented yet (its rev-32 container "
                "isn't handled by the donor walker). Set Donor Mode to 'Copy CharClipSet "
                "Only'.")
            return {'CANCELLED'}

        if donor_path and self.donor_mode == 'inject_meshes':
            _log("--- Donor mesh injection (DC1, experimental) ---")
            try:
                data = inject_meshes_into_donor(
                    donor_path, entries, platform=self.platform,
                    write_tangents=self.export_tangents)
            except DonorInjectError as e:
                _log(f"EXPORT FAILED: donor mesh injection failed - {e}")
                self.report({'ERROR'}, f"Donor mesh injection failed: {e}")
                return {'CANCELLED'}
            with open(self.filepath, 'wb') as f:
                f.write(data)
            summary = (f"Injected {len(entries)} mesh(es) into donor -> {self.filepath} "
                       f"({len(data)} bytes). Materials/textures NOT exported.")
            _log(f"===== SUCCESS: {summary} =====")
            self.report({'INFO'}, summary)
            return {'FINISHED'}

        if donor_path:
            _log("--- Donor CharClipSet (DC1/DC3, experimental) ---")
            try:
                clip_name, clip_bytes, clip_boundaries = extract_charclipset_from_donor(donor_path)
            except DonorExtractionError as e:
                _log(f"EXPORT FAILED: donor CharClipSet extraction failed - {e}")
                self.report({'ERROR'}, f"Donor CharClipSet extraction failed: {e}")
                return {'CANCELLED'}
            raw_entries.append(("CharClipSet", clip_name, clip_bytes, clip_boundaries))

        _log("--- Materials & Textures ---")
        # mip_floor: DC1's real mip chains stop the instant either dimension would drop
        # below 16 (confirmed by byte-parsing every one of 24 real textures embedded in
        # retail emilia01.milo_xbox's nested texture sub-milos - 128/256/512 bases and
        # both square and non-square textures all bottom out at a 16px smaller dimension,
        # NEVER at 4x4 - e.g. 256x256 ATI2 normal maps carry exactly mipMaps=4, landing at
        # 16x16, and 512x128 lands at 64x16). The function's own default (mip_floor=4)
        # was going two levels deeper than any real DC1 file ever does (down to 4x4/8x4),
        # which is a real, confirmed mismatch - left unfixed pending the same byte-level
        # check for DC3, whose real mip depth hasn't been independently verified yet.
        game_mip_floor = DC1_TEX_MIP_FLOOR if self.game == 'dc1' else 4
        materials, textures, _texture_images = gather_materials_and_textures(
            materials_by_name, max_tex_size=512, ignore_tex_size_limits=self.ignore_tex_size_limits,
            platform=self.platform,
            generate_mips=(self.game in ('dc1', 'dc3') and self.dc1_generate_mipmaps),
            mip_floor=game_mip_floor)

        _log("--- Writing container ---")
        data = build_milo_bytes(root_name, entries, milo_type=self.milo_type,
                                 materials=materials, textures=textures,
                                 bone_trans_entries=bone_trans_entries,
                                 char_mesh_hide_entries=char_mesh_hide_entries,
                                 char_hair_entries=char_hair_entries,
                                 platform=self.platform,
                                 write_tangents=self.export_tangents,
                                 outfit_config_entries=outfit_config_entries,
                                 char_collide_entries=char_collide_entries,
                                 raw_entries=raw_entries,
                                 dir_revision=(DC3_MILO_REVISION if self.game == 'dc3'
                                               else RB3_MILO_REVISION),
                                 force_dc3_mat_defaults=(self.game == 'dc3'
                                                         and self.dc3_force_mat_defaults),
                                 is_dc1=(self.game == 'dc1'))

        with open(self.filepath, 'wb') as f:
            f.write(data)

        total_verts = sum(len(e[4]) for e in entries)
        total_faces = sum(len(e[5]) for e in entries)
        bone_note = f", {len(bone_trans_entries)} bone(s) (experimental)" if bone_trans_entries else ""
        hide_note = ", 1 CharMeshHide (experimental, untested in RB3)" if char_mesh_hide_entries else ""
        hair_note = (f", {len(char_hair_entries)} CharHair (experimental, untested in RB3)"
                     if char_hair_entries else "")
        tangent_note = "" if self.export_tangents else ", tangents OFF (normal maps will be flat)"
        cfg_note = ", 1 OutfitConfig (.cfg)" if outfit_config_entries else ""
        clip_note = (f", CharClipSet '{raw_entries[0][1]}' from donor" if raw_entries else "")
        summary = (
            f"Exported {len(entries)} mesh(es), {len(materials)} material(s), "
            f"{len(textures)} texture(s){bone_note}{hide_note}{hair_note}{cfg_note}{clip_note}{tangent_note}, "
            f"{total_verts} verts, {total_faces} tris to {self.filepath}"
        )
        _log(f"===== SUCCESS: {summary} ({len(data)} bytes total) =====")
        self.report({'INFO'}, summary)
        return {'FINISHED'}


def menu_func_export(self, context):
    self.layout.operator(EXPORT_OT_milo_scene.bl_idname, text="Milo Scene (.milo)",
                          icon_value=_milo_icon_id('MILO_EXPORT'))


def _coll_redraw(self, context):
    """Tag all 3D viewports for redraw when a collision property changes, so the
    wireframe preview updates live as the user drags a slider."""
    if context is None:
        return
    wm = getattr(context, "window_manager", None)
    if wm is None:
        return
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


class GltfMiloObjectSettings(bpy.types.PropertyGroup):
    """Per-object Milo settings. Stored on bpy.types.Object.

    mesh_parent exposes the RndTrans `parentObj` field of the exported .mesh. In a
    normal milo the exporter forces every mesh to parent to the milo root dir, because
    that's how a self-contained scene hangs together. But the TBRB "Custom Song Asset"
    workflow ships a LOOSE .mesh that a song.dta wires into the world at runtime via
    set_trans_parent - and some assets (a keyboard replacement, a prop that rides a
    character bone) want to be parented to a bone that already exists in-game (e.g.
    bone_piano_base.mesh, bone_neck.mesh) so they inherit that bone's animation/placement
    without the DTA having to call set_trans_parent at all.

    Leave BLANK to keep the exporter's default parent (the milo root for a normal export).
    This value is written verbatim as the Trans parentObj symbol, so it must match the
    exact in-game object name including any .mesh suffix (bones in these skeletons are
    named like 'bone_piano_base.mesh')."""
    mesh_parent: StringProperty(
        name="Mesh Trans Parent",
        description="Optional .mesh trans bone parenting. The name of the object/bone this "
                     "mesh should be parented to (its RndTrans parentObj), e.g. "
                     "'bone_piano_base.mesh' or 'bone_neck.mesh'. Used by the TBRB Custom "
                     "Song Asset workflow so the loose .mesh rides an existing in-game bone "
                     "the moment it loads, instead of needing a set_trans_parent call in the "
                     "song.dta. Leave BLANK to keep whatever parent the exporter uses by "
                     "default (the milo root for a normal export)",
        default="",
    )


class GltfMiloCollisionSettings(bpy.types.PropertyGroup):
    """Per-bone CharCollide (.coll) settings. Stored on bpy.types.Bone."""
    enabled: BoolProperty(
        name="Export Collision Volume",
        description="Emit a CharCollide (.coll) volume attached to this bone. Also shows a "
                     "live wireframe preview of the volume in the viewport",
        default=False,
        update=_coll_redraw,
    )
    shape: EnumProperty(
        name="Shape",
        description="CharCollide.shape - the collision volume type",
        items=COLL_SHAPE_ITEMS,
        default='sphere',
        update=_coll_redraw,
    )
    radius0: FloatProperty(
        name="Radius 0",
        description="CharCollide.radius0 - radius of the sphere, or of the length0 hemisphere "
                     "if this is a cigar/capsule",
        default=0.5, min=0.0, soft_max=5.0,
        update=_coll_redraw,
    )
    radius1: FloatProperty(
        name="Radius 1",
        description="CharCollide.radius1 - cigar only: radius of the length1 hemisphere",
        default=0.5, min=0.0, soft_max=5.0,
        update=_coll_redraw,
    )
    length0: FloatProperty(
        name="Length 0",
        description="CharCollide.length0 - cigar only: placement of the radius0 hemisphere "
                     "along the bone's local X axis. Must be <= length1",
        default=0.0, soft_min=-5.0, soft_max=5.0,
        update=_coll_redraw,
    )
    length1: FloatProperty(
        name="Length 1",
        description="CharCollide.length1 - cigar only: placement of the radius1 hemisphere "
                     "along the bone's local X axis. Must be >= length0",
        default=1.0, soft_min=-5.0, soft_max=5.0,
        update=_coll_redraw,
    )
    flags: IntProperty(
        name="Collision Group Flags",
        description="CharCollide.flags - a collision-group BITMASK. This is the linchpin of "
                     "collision: a hair strand only collides with this volume when the "
                     "strand's own 'Hookup Flags' share at least one bit with this value "
                     "(strand.hookup_flags & collide.flags != 0). Vanilla characters give "
                     "each volume a distinct power-of-two bit (1, 2, 4, 8, 16, ...) so hair "
                     "can opt into specific volumes. If this is 0 the volume collides with "
                     "NOTHING (that's why an all-zero setup silently does nothing). Default "
                     "1 = collision group 1",
        default=1, min=0,
        update=_coll_redraw,
    )
    mesh_y_bias: BoolProperty(
        name="Mesh Y Bias",
        description="CharCollide.mesh_y_bias - for spheres/cigars, biases the fit toward the "
                     "bone's local +Y (green) axis. Used by the game for chest/back volumes",
        default=False,
        update=_coll_redraw,
    )


_coll_draw_handle = None


def _coll_line_shader():
    import gpu
    # 3.x/4.x compatible uniform-color line shader
    try:
        return gpu.shader.from_builtin('UNIFORM_COLOR')
    except Exception:
        return gpu.shader.from_builtin('3D_UNIFORM_COLOR')


def _circle_points(center, axis_u, axis_v, radius, segments=24):
    """Return a list of points forming a circle in the plane spanned by axis_u/axis_v."""
    from mathutils import Vector
    pts = []
    for i in range(segments):
        a = (i / segments) * 2.0 * _math.pi
        pts.append(center + axis_u * (radius * _math.cos(a)) + axis_v * (radius * _math.sin(a)))
    return pts


def _loop_segments(points):
    """Convert a closed loop of points into GPU LINE segment pairs."""
    segs = []
    n = len(points)
    for i in range(n):
        segs.append(points[i])
        segs.append(points[(i + 1) % n])
    return segs


def _sphere_wire_segments(center, rx, ry, rz, radius, segments=24):
    """Three orthogonal great-circle rings (XY, XZ, YZ) approximating a sphere."""
    segs = []
    segs += _loop_segments(_circle_points(center, rx, ry, radius, segments))
    segs += _loop_segments(_circle_points(center, rx, rz, radius, segments))
    segs += _loop_segments(_circle_points(center, ry, rz, radius, segments))
    return segs


def _capsule_wire_segments(world_mtx, radius0, radius1, length0, length1, segments=16):
    """Cigar/capsule wireframe along local X: a hemisphere of radius0 at x=length0 and a
    hemisphere of radius1 at x=length1, joined by tube lines. Mirrors UtilDrawCigar's
    intent (two radii, two placements along X)."""
    from mathutils import Vector
    # Basis vectors (world) for the bone-local axes.
    origin = world_mtx.to_translation()
    m = world_mtx.to_3x3()
    x_axis = m @ Vector((1.0, 0.0, 0.0))
    y_axis = m @ Vector((0.0, 1.0, 0.0))
    z_axis = m @ Vector((0.0, 0.0, 1.0))
    for v in (x_axis, y_axis, z_axis):
        if v.length > 1e-8:
            v.normalize()

    c0 = origin + x_axis * length0
    c1 = origin + x_axis * length1
    segs = []
    # End rings (perpendicular to X)
    segs += _loop_segments(_circle_points(c0, y_axis, z_axis, radius0, segments))
    segs += _loop_segments(_circle_points(c1, y_axis, z_axis, radius1, segments))
    # Connecting tube lines (4 around)
    for ax in (y_axis, z_axis, -y_axis, -z_axis):
        segs.append(c0 + ax * radius0)
        segs.append(c1 + ax * radius1)
    # Hemisphere caps: half-rings in the XY and XZ planes at each end.
    # radius0 cap bulges toward -X, radius1 cap toward +X.
    def half_ring(center, axis_long, axis_side, radius, sign):
        pts = []
        steps = max(4, segments // 2)
        for i in range(steps + 1):
            a = (i / steps) * _math.pi
            pts.append(center + axis_long * (sign * radius * _math.sin(a))
                       + axis_side * (radius * _math.cos(a)))
        out = []
        for i in range(len(pts) - 1):
            out.append(pts[i]); out.append(pts[i + 1])
        return out
    segs += half_ring(c0, x_axis, y_axis, radius0, -1.0)
    segs += half_ring(c0, x_axis, z_axis, radius0, -1.0)
    segs += half_ring(c1, x_axis, y_axis, radius1, +1.0)
    segs += half_ring(c1, x_axis, z_axis, radius1, +1.0)
    return segs


def _plane_wire_segments(world_mtx, size=1.0):
    """A square plane facing bone-local +X, with a short normal stub, mirroring how
    CharCollide::Highlight draws a plane oriented by WorldXfm.m.x."""
    from mathutils import Vector
    origin = world_mtx.to_translation()
    m = world_mtx.to_3x3()
    x_axis = m @ Vector((1.0, 0.0, 0.0))
    y_axis = m @ Vector((0.0, 1.0, 0.0))
    z_axis = m @ Vector((0.0, 0.0, 1.0))
    for v in (x_axis, y_axis, z_axis):
        if v.length > 1e-8:
            v.normalize()
    corners = [
        origin + y_axis * size + z_axis * size,
        origin - y_axis * size + z_axis * size,
        origin - y_axis * size - z_axis * size,
        origin + y_axis * size - z_axis * size,
    ]
    segs = _loop_segments(corners)
    # normal stub (local +X)
    segs.append(origin)
    segs.append(origin + x_axis * size)
    return segs


def _bone_world_matrix(armature_obj, bone):
    """World matrix of a bone's rest position (armature world @ bone.matrix_local)."""
    return armature_obj.matrix_world @ bone.matrix_local


def _draw_collision_volumes():
    import gpu
    from gpu_extras.batch import batch_for_shader
    context = bpy.context
    if context is None:
        return
    # Gather segments per color; enabled volumes get a red-ish wire.
    all_segs = []
    for obj in context.view_layer.objects:
        if obj.type != 'ARMATURE' or not obj.visible_get():
            continue
        arm = obj.data
        for bone in arm.bones:
            coll = getattr(bone, "gltfmilo_collision", None)
            if coll is None or not coll.enabled:
                continue
            wm = _bone_world_matrix(obj, bone)
            origin = wm.to_translation()
            m = wm.to_3x3()
            from mathutils import Vector
            rx = m @ Vector((1.0, 0.0, 0.0)); ry = m @ Vector((0.0, 1.0, 0.0)); rz = m @ Vector((0.0, 0.0, 1.0))
            for v in (rx, ry, rz):
                if v.length > 1e-8:
                    v.normalize()
            shape = coll.shape
            if shape in ('sphere', 'inside_sphere'):
                all_segs += _sphere_wire_segments(origin, rx, ry, rz, coll.radius0)
            elif shape in ('cigar', 'inside_cigar'):
                lo, hi = coll.length0, coll.length1
                if lo > hi:
                    lo, hi = hi, lo
                all_segs += _capsule_wire_segments(wm, coll.radius0, coll.radius1, lo, hi)
            elif shape == 'plane':
                all_segs += _plane_wire_segments(wm, size=max(coll.radius0, 0.5))

    if not all_segs:
        return

    shader = _coll_line_shader()
    batch = batch_for_shader(shader, 'LINES', {"pos": all_segs})
    gpu.state.line_width_set(1.5)
    gpu.state.blend_set('ALPHA')
    try:
        gpu.state.depth_test_set('LESS_EQUAL')
    except Exception:
        pass
    shader.bind()
    shader.uniform_float("color", (1.0, 0.15, 0.15, 0.9))
    batch.draw(shader)
    gpu.state.blend_set('NONE')
    try:
        gpu.state.depth_test_set('NONE')
    except Exception:
        pass


def _register_coll_draw_handler():
    global _coll_draw_handle
    if _coll_draw_handle is None:
        _coll_draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw_collision_volumes, (), 'WINDOW', 'POST_VIEW')


def _unregister_coll_draw_handler():
    global _coll_draw_handle
    if _coll_draw_handle is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_coll_draw_handle, 'WINDOW')
        except Exception:
            pass
        _coll_draw_handle = None


class BONE_PT_gltfmilo_collision(bpy.types.Panel):
    bl_label = "Milo Collision (.coll)"
    bl_idname = "BONE_PT_gltfmilo_collision"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "bone"

    @classmethod
    def poll(cls, context):
        return context.bone is not None or context.edit_bone is not None

    def draw_header(self, context):
        bone = context.bone
        if bone is not None and hasattr(bone, "gltfmilo_collision"):
            self.layout.prop(bone.gltfmilo_collision, "enabled", text="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        bone = context.bone
        if bone is None:
            layout.label(text="Select a bone in Pose or Object mode to edit collision.",
                         icon='INFO')
            return

        coll = bone.gltfmilo_collision
        col = layout.column()
        col.active = coll.enabled
        col.prop(coll, "shape")

        shape = coll.shape
        if shape in ('sphere', 'inside_sphere'):
            col.prop(coll, "radius0", text="Radius")
        elif shape in ('cigar', 'inside_cigar'):
            col.prop(coll, "radius0")
            col.prop(coll, "radius1")
            col.prop(coll, "length0")
            col.prop(coll, "length1")
            if coll.length0 > coll.length1:
                col.label(text="length0 > length1 (will be swapped for preview)",
                          icon='ERROR')
            col.prop(coll, "mesh_y_bias")
        elif shape == 'plane':
            col.prop(coll, "radius0", text="Size")

        col.prop(coll, "flags")
        layout.label(text="Preview only - .coll export not wired up yet.", icon='INFO')


class OBJECT_PT_gltfmilo_object(bpy.types.Panel):
    """Object Properties panel exposing per-object Milo export settings - currently the
    optional .mesh trans parent used by the TBRB Custom Song Asset workflow."""
    bl_label = "Milo Object"
    bl_idname = "OBJECT_PT_gltfmilo_object"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "object"

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == 'MESH'

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        obj = context.object
        settings = getattr(obj, "gltfmilo_object", None)
        if settings is None:
            layout.label(text="No Milo object settings on this object.", icon='INFO')
            return

        col = layout.column()
        col.prop(settings, "mesh_parent", text="Trans Parent")
        if (settings.mesh_parent or "").strip():
            col.label(text="Mesh will parent to this existing in-game object/bone.",
                      icon='INFO')
        else:
            col.label(text="Blank = default parent (milo root on a normal export).",
                      icon='INFO')


class IMPORT_OT_tbrb_skeleton(bpy.types.Operator, ImportHelper):
    """Import a TBRB skeleton milo as a Blender armature"""
    bl_idname = "import_scene.tbrb_skeleton"
    bl_label = "Import TBRB Skeleton Milo"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".milo_ps3"
    filter_glob: StringProperty(
        default="*.milo_ps3;*.milo_xbox;*.milo",
        options={'HIDDEN'},
    )

    bone_display_length: FloatProperty(
        name="Bone Display Length",
        description="Length given to bones that have no child to point at. Purely cosmetic - "
                     "Milo bones are transforms, so only the head position and orientation "
                     "are exported. Does NOT affect exported data",
        default=0.5, min=0.01, max=20.0,
    )

    connect_bones: BoolProperty(
        name="Auto-Connect Bones",
        description="Snap each bone's tail to its single child's head. Leave OFF unless you "
                     "want prettier viewport display: connecting bones makes Blender move a "
                     "child's head whenever the parent's tail moves, which is exactly how a "
                     "faithful rest pose gets silently corrupted during editing",
        default=False,
    )

    import_collisions: BoolProperty(
        name="Import Collision Volumes",
        description="Copy each CharCollide's shape/radius/length/flags onto its parent bone's "
                     "Milo Collision panel. NOTE: the panel holds ONE volume per bone, and "
                     "TBRB puts three on bone_head.mesh - only the first is stored. Export "
                     "with 'Retail Head/Neck Collisions' on to emit all four correctly",
        default=True,
    )

    def execute(self, context):
        try:
            dir_name, bones, collisions = parse_tbrb_skeleton_milo(self.filepath)
        except Exception as e:
            _log(f"IMPORT FAILED: {e}")
            self.report({'ERROR'}, f"Could not parse skeleton milo: {e}")
            return {'CANCELLED'}

        if not bones:
            self.report({'ERROR'}, "No Trans bones found - is this a skeleton milo?")
            return {'CANCELLED'}

        _log(f"===== Importing TBRB skeleton '{dir_name}' from {self.filepath} =====")
        _log(f"  {len(bones)} bone(s), {len(collisions)} collision volume(s)")

        try:
            _arm, applied = _build_armature_from_skeleton(
                context, dir_name, bones, collisions,
                bone_display_length=self.bone_display_length,
                connect_bones=self.connect_bones,
                import_collisions=self.import_collisions)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        summary = (f"Imported skeleton '{dir_name}': {len(bones)} bone(s), "
                   f"{applied} collision volume(s) applied")
        _log(f"===== SUCCESS: {summary} =====")
        self.report({'INFO'}, summary)
        return {'FINISHED'}


class _IMPORT_OT_milo_skeleton_base(bpy.types.Operator, ImportHelper):
    """Shared base for the RB3/DC1/DC3 skeleton importers. Each subclass only sets its
    bl_idname/bl_label, the display name shown in the File > Import menu, and its default
    file extension/filter. All of them use the general parse_milo_skeleton (which handles
    compressed containers and both CharCollide revisions) plus the shared armature builder,
    so the only real difference between games is the label and the filename filter."""
    bl_options = {'REGISTER', 'UNDO'}

    # Subclasses override these three:
    _game_label = "Milo"
    filename_ext = ".milo_xbox"
    filter_glob: StringProperty(
        default="*.milo_xbox;*.milo_ps3;*.milo_wii;*.milo",
        options={'HIDDEN'},
    )

    bone_display_length: FloatProperty(
        name="Bone Display Length",
        description="Length given to bones that have no child to point at. Purely cosmetic - "
                     "Milo bones are transforms, so only the head position and orientation "
                     "are exported. Does NOT affect exported data",
        default=0.5, min=0.01, max=20.0,
    )

    connect_bones: BoolProperty(
        name="Auto-Connect Bones",
        description="Snap each bone's tail to its single child's head. Leave OFF unless you "
                     "want prettier viewport display: connecting bones makes Blender move a "
                     "child's head whenever the parent's tail moves, which is exactly how a "
                     "faithful rest pose gets silently corrupted during editing",
        default=False,
    )

    import_collisions: BoolProperty(
        name="Import Collision Volumes",
        description="Copy each CharCollide's shape/radius/length/flags onto its parent bone's "
                     "Milo Collision panel. The panel holds ONE volume per bone; if a milo "
                     "puts more than one on the same bone only the first is stored",
        default=True,
    )

    def execute(self, context):
        try:
            dir_name, bones, collisions = parse_milo_skeleton(self.filepath)
        except Exception as e:
            _log(f"IMPORT FAILED: {e}")
            self.report({'ERROR'}, f"Could not parse {self._game_label} milo: {e}")
            return {'CANCELLED'}

        if not bones:
            self.report({'ERROR'},
                        "No Trans bones found - is this a skeleton/character milo with a "
                        "shared armature? (A mesh-only milo has no bones to import.)")
            return {'CANCELLED'}

        _log(f"===== Importing {self._game_label} skeleton '{dir_name}' "
             f"from {self.filepath} =====")
        _log(f"  {len(bones)} bone(s), {len(collisions)} collision volume(s)")

        try:
            _arm, applied = _build_armature_from_skeleton(
                context, dir_name, bones, collisions,
                bone_display_length=self.bone_display_length,
                connect_bones=self.connect_bones,
                import_collisions=self.import_collisions)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        summary = (f"Imported {self._game_label} skeleton '{dir_name}': "
                   f"{len(bones)} bone(s), {applied} collision volume(s) applied")
        _log(f"===== SUCCESS: {summary} =====")
        self.report({'INFO'}, summary)
        return {'FINISHED'}


class IMPORT_OT_rb3_skeleton(_IMPORT_OT_milo_skeleton_base):
    """Import a Rock Band 3 skeleton/character milo as a Blender armature.

    Use this to pull the authoritative RB3 shared skeleton (the fixed rig every RB3
    character is weighted to) straight out of a retail milo, so your character template's
    rest pose matches the game exactly. Handles compressed retail milos automatically."""
    bl_idname = "import_scene.rb3_skeleton"
    bl_label = "Import RB3 Skeleton Milo"
    _game_label = "RB3"
    filename_ext = ".milo_xbox"
    filter_glob: StringProperty(
        default="*.milo_xbox;*.milo_ps3;*.milo_wii;*.milo",
        options={'HIDDEN'},
    )


class IMPORT_OT_dc1_skeleton(_IMPORT_OT_milo_skeleton_base):
    """Import a Dance Central 1 skeleton/character milo as a Blender armature.

    DC1 shares RB3's DirectoryMeta revision (28) and CharCollide revision (7); the parser
    is identical to RB3's path. Use it to recover DC1's dancer skeleton for a correct
    character template rest pose."""
    bl_idname = "import_scene.dc1_skeleton"
    bl_label = "Import DC1 Skeleton Milo"
    _game_label = "DC1"
    filename_ext = ".milo_xbox"
    filter_glob: StringProperty(
        default="*.milo_xbox;*.milo;*.milo_ps3",
        options={'HIDDEN'},
    )


class IMPORT_OT_dc3_skeleton(_IMPORT_OT_milo_skeleton_base):
    """Import a Dance Central 3 skeleton/character milo as a Blender armature.

    DC3 uses DirectoryMeta revision 32 (an extra header byte vs RB3/DC1); the parser
    accounts for that automatically. Bone and CharCollide layouts are otherwise the same
    as RB3. Use it to recover DC3's dancer skeleton for a correct template rest pose."""
    bl_idname = "import_scene.dc3_skeleton"
    bl_label = "Import DC3 Skeleton Milo"
    _game_label = "DC3"
    filename_ext = ".milo_xbox"
    filter_glob: StringProperty(
        default="*.milo_xbox;*.milo",
        options={'HIDDEN'},
    )


class IMPORT_OT_gh2_skeleton(bpy.types.Operator, ImportHelper):
    """Import a Guitar Hero 2 (Xbox 360) skeleton/character milo as a Blender armature.

    GH2 X360's object-stream body is little-endian (unlike RB3/DC1/DC3/TBRB, which are
    all big-endian) so it needs its own reader - parse_gh2_skeleton in io.py - but the
    Trans (bone) layout itself turned out to be byte-identical to the other four games',
    just endian-flipped, and both feed the exact same _build_armature_from_skeleton.

    A standalone operator rather than a subclass of _IMPORT_OT_milo_skeleton_base:
    that base's execute() hardcodes a call to parse_milo_skeleton (the RB3/DC1/DC3
    reader), so sharing it would mean either modifying that already-relied-on class or
    adding a branch to it for a format it was never written against. Duplicating its
    ~30 lines of glue here keeps the existing importers untouched. If more early-
    Xbox-360-era games end up sharing GH2's reader later, that's the point to
    reconsider factoring out a common base with a pluggable parser function.

    No mesh/texture import yet - GH2's RndMesh/RndMat/RndTex are all lower, differently-
    laid-out revisions than RB3/DC1/TBRB and need their own (larger) effort. Skeleton
    only, for now.
    """
    bl_idname = "import_scene.gh2_skeleton"
    bl_label = "Import GH2 Skeleton Milo"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".milo_xbox"
    filter_glob: StringProperty(
        default="*.milo_xbox",
        options={'HIDDEN'},
    )

    bone_display_length: FloatProperty(
        name="Bone Display Length",
        description="Length given to bones that have no child to point at. Purely "
                     "cosmetic - Milo bones are transforms, so only the head position "
                     "and orientation are exported. Does NOT affect exported data",
        default=0.5, min=0.01, max=20.0,
    )

    connect_bones: BoolProperty(
        name="Auto-Connect Bones",
        description="Snap each bone's tail to its single child's head. Leave OFF unless "
                     "you want prettier viewport display: connecting bones makes Blender "
                     "move a child's head whenever the parent's tail moves, which is "
                     "exactly how a faithful rest pose gets silently corrupted during "
                     "editing",
        default=False,
    )

    import_collisions: BoolProperty(
        name="Import Collision Volumes",
        description="No-op for GH2 right now - the one retail file this reader was "
                     "developed against (goth2.milo_xbox) has zero CharCollide entries, "
                     "so there's nothing yet to decode. Left here for UI consistency "
                     "with the other importers and so it's one flag away from working "
                     "once a GH2 CharCollide layout is confirmed against a file that "
                     "actually has one",
        default=True,
    )

    def execute(self, context):
        try:
            dir_name, bones, collisions = parse_gh2_skeleton(self.filepath)
        except Exception as e:
            _log(f"IMPORT FAILED: {e}")
            self.report({'ERROR'}, f"Could not parse GH2 milo: {e}")
            return {'CANCELLED'}

        if not bones:
            self.report({'ERROR'},
                        "No Trans bones found - is this a skeleton/character milo with a "
                        "shared armature? (A mesh-only milo has no bones to import.)")
            return {'CANCELLED'}

        _log(f"===== Importing GH2 skeleton '{dir_name}' from {self.filepath} =====")
        _log(f"  {len(bones)} bone(s), {len(collisions)} collision volume(s)")

        try:
            _arm, applied = _build_armature_from_skeleton(
                context, dir_name, bones, collisions,
                bone_display_length=self.bone_display_length,
                connect_bones=self.connect_bones,
                import_collisions=self.import_collisions)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        summary = (f"Imported GH2 skeleton '{dir_name}': {len(bones)} bone(s), "
                   f"{applied} collision volume(s) applied")
        _log(f"===== SUCCESS: {summary} =====")
        self.report({'INFO'}, summary)
        return {'FINISHED'}


def _build_gh2_mesh_object(mesh_dict, collection):
    """Build one Blender mesh object from a parse_gh2_meshes() entry: geometry + UV
    only, matching that function's deliberate scope (no weights, no real materials -
    see its docstring).

    Positioning: uses the mesh's own embedded world_xfm as the object's matrix_world,
    with vertex positions left completely raw (no coordinate-axis conversion, same as
    _build_armature_from_skeleton applies none either - meshes and the armature need to
    stay in the same convention as each other, not necessarily "true" Z-up).

    This choice isn't fully cross-checked against a reference render yet: of the meshes
    inspected during development, local_xfm came out identity for the large per-bone-
    group body chunks but a real non-identity offset for at least one rigidly-parented
    standalone piece (an eye), which reads as the standard "vertices in the mesh's own
    local space, world_xfm the one placement transform to apply" model - and is why
    world_xfm (not local_xfm, and no parent-chain walk through the `parent` field) is
    what's applied here. If meshes come in offset or misrotated relative to the
    imported armature, this is the first place to revisit.

    `material` and `parent` (which can name either a bone or another mesh - see
    parse_gh2_meshes' docstring) are stashed as custom properties only; nothing here
    creates a Blender material or reparents anything.
    """
    verts_in = mesh_dict['vertices']
    verts = [(v['x'], v['y'], v['z']) for v in verts_in]
    faces = mesh_dict['faces']

    me = bpy.data.meshes.new(mesh_dict['name'])
    me.from_pydata(verts, [], faces)
    me.update()

    uv_layer = me.uv_layers.new(name="UVMap")
    for loop in me.loops:
        v = verts_in[loop.vertex_index]
        # Stored exactly as found in the file - no V-flip applied. If a texture ends
        # up vertically mirrored once material import lands, GH2 may follow the usual
        # DirectX top-down V convention and need this to become (v['u'], 1.0 - v['v'])
        # - flagging now since it's unconfirmed either way.
        uv_layer.data[loop.index].uv = (v['u'], v['v'])

    obj = bpy.data.objects.new(mesh_dict['name'], me)
    collection.objects.link(obj)
    obj.matrix_world = _milo_to_blender_matrix(mesh_dict['world_xfm'])
    obj['gh2_material'] = mesh_dict['material']
    obj['gh2_parent'] = mesh_dict['parent']
    return obj


class IMPORT_OT_gh2_meshes(bpy.types.Operator, ImportHelper):
    """Import every mesh in a Guitar Hero 2 (Xbox 360) character milo as separate
    Blender mesh objects - geometry and UVs only, no weights or materials yet (see
    parse_gh2_meshes' docstring in io.py for why that's the deliberate scope of this
    first pass). Each of the file's Mesh entries becomes its own object, matching how
    GH2 actually splits a character into many small per-bone-group chunks rather than
    one skinned mesh - the same structure the eventual weights pass will need to key
    off of, so keeping that 1-object-per-entry shape now avoids a re-import later.

    A standalone operator, not layered onto IMPORT_OT_gh2_skeleton, so mesh-only or
    skeleton-only imports both stay simple single-purpose actions - matching how this
    addon generally keeps import/export concerns as separate operators rather than one
    operator with a "what to import" toggle pile.
    """
    bl_idname = "import_scene.gh2_meshes"
    bl_label = "Import GH2 Meshes"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".milo_xbox"
    filter_glob: StringProperty(
        default="*.milo_xbox",
        options={'HIDDEN'},
    )

    def execute(self, context):
        try:
            dir_name, meshes = parse_gh2_meshes(self.filepath)
        except Exception as e:
            _log(f"IMPORT FAILED: {e}")
            self.report({'ERROR'}, f"Could not parse GH2 milo: {e}")
            return {'CANCELLED'}

        _log(f"===== Importing {len(meshes)} GH2 mesh(es) from {self.filepath} =====")

        collection = bpy.data.collections.new(dir_name)
        context.scene.collection.children.link(collection)

        imported = 0
        for m in meshes:
            try:
                _build_gh2_mesh_object(m, collection)
                imported += 1
            except Exception as ex:
                _log(f"  WARNING: failed to build Blender object for mesh "
                     f"'{m['name']}': {ex}")

        summary = (f"Imported {imported}/{len(meshes)} GH2 mesh(es) into collection "
                   f"'{dir_name}'")
        _log(f"===== SUCCESS: {summary} =====")
        self.report({'INFO'}, summary)
        return {'FINISHED'}


class TOPBAR_MT_milo_skeleton_import(bpy.types.Menu):
    """File > Import > Milo Skeleton Importer submenu grouping every game's skeleton
    importer in one place instead of scattering four entries across the Import menu."""
    bl_idname = "TOPBAR_MT_milo_skeleton_import"
    bl_label = "Milo Skeleton Importer"

    def draw(self, context):
        layout = self.layout
        layout.operator(IMPORT_OT_rb3_skeleton.bl_idname,
                        text="Rock Band 3 (.milo_xbox)",
                        icon_value=_milo_icon_id('RB3'))
        layout.operator(IMPORT_OT_tbrb_skeleton.bl_idname,
                        text="The Beatles: Rock Band (.milo_ps3)",
                        icon_value=_milo_icon_id('TBRB'))
        layout.operator(IMPORT_OT_dc1_skeleton.bl_idname,
                        text="Dance Central 1 (.milo_xbox)",
                        icon_value=_milo_icon_id('DC1'))
        layout.operator(IMPORT_OT_dc3_skeleton.bl_idname,
                        text="Dance Central 3 (.milo_xbox)",
                        icon_value=_milo_icon_id('DC3'))
        layout.operator(IMPORT_OT_gh2_skeleton.bl_idname,
                        text="Guitar Hero 2 (.milo_xbox)",
                        icon_value=_milo_icon_id('GH2'))


def menu_func_import(self, context):
    self.layout.menu(TOPBAR_MT_milo_skeleton_import.bl_idname,
                      icon_value=_milo_icon_id('MILO_EXPORT'))
    # Not nested inside the skeleton submenu above - this imports meshes, not a
    # skeleton, and there's only one game's mesh importer so far, so a second
    # submenu would be premature. Revisit if more games get mesh import later.
    self.layout.operator(IMPORT_OT_gh2_meshes.bl_idname,
                          text="Guitar Hero 2 Meshes (.milo_xbox)",
                          icon_value=_milo_icon_id('GH2'))


classes = (
    GltfMiloMaterialSettings,
    MATERIAL_OT_gltfmilo_autodetect_textures,
    MATERIAL_PT_gltfmilo_settings,
    MATERIAL_PT_gltfmilo_refract,
    GltfMiloHairBoneItem,
    GltfMiloHairStrand,
    GltfMiloHairSettings,
    ARMATURE_OT_gltfmilo_create_hair_strand,
    ARMATURE_OT_gltfmilo_remove_hair_strand,
    DATA_UL_gltfmilo_hair_strands,
    DATA_PT_gltfmilo_hair,
    GltfMiloObjectSettings,
    OBJECT_PT_gltfmilo_object,
    GltfMiloCollisionSettings,
    BONE_PT_gltfmilo_collision,
    EXPORT_OT_milo_scene,
    IMPORT_OT_tbrb_skeleton,
    IMPORT_OT_rb3_skeleton,
    IMPORT_OT_dc1_skeleton,
    IMPORT_OT_dc3_skeleton,
    IMPORT_OT_gh2_skeleton,
    IMPORT_OT_gh2_meshes,
    TOPBAR_MT_milo_skeleton_import,
)


def register():
    _register_icons()
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Material.gltfmilo_settings = PointerProperty(type=GltfMiloMaterialSettings)
    bpy.types.Armature.gltfmilo_hair = PointerProperty(type=GltfMiloHairSettings)
    bpy.types.Bone.gltfmilo_collision = PointerProperty(type=GltfMiloCollisionSettings)
    bpy.types.Object.gltfmilo_object = PointerProperty(type=GltfMiloObjectSettings)
    _register_coll_draw_handler()
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    _unregister_coll_draw_handler()
    del bpy.types.Object.gltfmilo_object
    del bpy.types.Bone.gltfmilo_collision
    del bpy.types.Armature.gltfmilo_hair
    del bpy.types.Material.gltfmilo_settings
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    _unregister_icons()


if __name__ == "__main__":
    register()


