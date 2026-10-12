"""
Rock Band 2 character importer (EXPERIMENTAL).

Builds a whole character from the game's prefab list - the decrypted prefabs.dta - and
the extracted game files, set once in the add-on's preferences:

    (guitar32
       (selectable TRUE)
       (gender male)
       (skin_color_index 1) (eye_color_index 2) (eyebrow_color_index 47)
       (height 0.75) (weight 0.5)
       (outfit
          (heads head_2 (colors 4 0))
          (torso strapjacket_pleather (colors 45 12))
          (hands naked)
          ...))

For each outfit part it finds the part's milo among the extracted files, imports its
meshes (from the part's resource milo when the part is a texture variant that points at
one, e.g. strapjacket_pleather -> strapjacket_resource), then dresses the materials
through the RB2 character-creator texture system:

    (colors a b)           one palette index per .cfg colour choice, in cfg order
    skin_color_index       skin.pal, the whole character's skin tone
    eye_color_index        eyes.pal
    eyebrow_color_index    hair.pal, applied to the head's eyebrows

    height, weight         the body sliders, applied to the armature from the gender's
                           deform milo (see rb2_proportions)

Not supported yet, and skipped with a note in the log: eye/lip makeup, tattoos and
instruments.

Selectable characters are the ones with (selectable TRUE); their display names come from
the locale file's (prefab_<name> "Display Name") entries.
"""

import os
import re
import time

import bpy
from bpy.props import BoolProperty, EnumProperty, StringProperty
from bpy_extras.io_utils import ImportHelper

from .io import read_milo_container_body
from .utilities import _log
from . import rb2_creator_textures as cc
from . import rb2_proportions
from . import rb2_outfit_fit


# ---------------------------------------------------------------------------------------
# DTA (the game's plain-text data format)
# ---------------------------------------------------------------------------------------

_TOKEN = re.compile(r'''
    (?P<ws>\s+) | (?P<comment>;[^\n]*) | (?P<open>[(\[{]) | (?P<close>[)\]}]) |
    "(?P<str>(?:[^"\\]|\\.)*)" | '(?P<qsym>[^']*)' | (?P<atom>[^\s()\[\]{};"']+)
''', re.VERBOSE | re.DOTALL)


class DtaString(str):
    """A quoted string, as opposed to a bare symbol."""


def parse_dta(text):
    """Nested lists of symbols (str), quoted strings (DtaString), ints and floats."""
    stack = [[]]
    for m in _TOKEN.finditer(text):
        kind = m.lastgroup
        if kind in ('ws', 'comment'):
            continue
        if kind == 'open':
            stack.append([])
        elif kind == 'close':
            if len(stack) > 1:
                node = stack.pop()
                stack[-1].append(node)
        elif kind == 'str':
            stack[-1].append(DtaString(m.group('str')))
        elif kind == 'qsym':
            stack[-1].append(m.group('qsym'))
        else:
            atom = m.group('atom')
            try:
                value = int(atom)
            except ValueError:
                try:
                    value = float(atom)
                except ValueError:
                    value = atom
            stack[-1].append(value)
    while len(stack) > 1:
        node = stack.pop()
        stack[-1].append(node)
    return stack[0]


def _read_text(path):
    with open(path, 'rb') as f:
        return f.read().decode('latin-1')


def _fields(entry):
    """{key: [values]} for a DTA entry's (key value ...) children."""
    out = {}
    for x in entry[1:]:
        if isinstance(x, list) and x and isinstance(x[0], str):
            out[x[0]] = x[1:]
    return out


def _character_from_entry(entry, key, default_selectable=False, default_gender='male'):
    f = _fields(entry)
    outfit = []
    for part in f.get('outfit', []):
        if isinstance(part, list) and len(part) >= 2 and isinstance(part[0], str):
            colors = []
            for x in part[2:]:
                if isinstance(x, list) and x and x[0] == 'colors':
                    colors = [int(c) for c in x[1:] if isinstance(c, (int, float))]
            outfit.append((part[0], str(part[1]), colors))

    def first(k, default=None):
        v = f.get(k)
        return v[0] if v else default
    sel = first('selectable')
    return dict(
        name=key,
        selectable=default_selectable if sel is None
        else str(sel).upper() in ('TRUE', '1'),
        gender=str(first('gender', default_gender)),
        instrument=str(first('instrument', '')),
        skin_color_index=int(first('skin_color_index', 0)),
        eye_color_index=int(first('eye_color_index', 0)),
        eyebrow_color_index=first('eyebrow_color_index'),
        height=float(first('height', 0.5)),
        weight=float(first('weight', 0.5)),
        outfit=outfit,
        has_tattoo='tattoo' in f and bool(f['tattoo']))


def parse_prefabs(path):
    """{name: character dict} from prefabs.dta - (prefab (guitar32 ...) ...) - or from
    characters.dta, whose (creator (male (contestant_number_one ...)) (female ...)) holds
    the character creator's starting looks. Those are keyed 'male/contestant_number_one'
    since each gender reuses the names, and count as selectable."""
    chars = {}
    for top in parse_dta(_read_text(path)):
        if not (isinstance(top, list) and top and isinstance(top[0], str)):
            continue
        if top[0] == 'prefab':
            for entry in top[1:]:
                if isinstance(entry, list) and entry and isinstance(entry[0], str):
                    chars[entry[0]] = _character_from_entry(entry, entry[0])
        elif top[0] == 'creator':
            for group in top[1:]:
                if not (isinstance(group, list) and group and isinstance(group[0], str)):
                    continue
                for entry in group[1:]:
                    if isinstance(entry, list) and entry and isinstance(entry[0], str):
                        key = f"{group[0]}/{entry[0]}"
                        chars[key] = _character_from_entry(entry, key, True, group[0])
    return chars


def parse_locale_prefab_names(path):
    """{checksum name: display name} from a locale file's (prefab_<name> "Name") entries."""
    names = {}
    for x in parse_dta(_read_text(path)):
        if isinstance(x, list) and len(x) >= 2 and isinstance(x[0], str) \
                and x[0].startswith('prefab_') and isinstance(x[1], DtaString):
            names[x[0][len('prefab_'):]] = str(x[1])
    return names


_PREFAB_CACHE = {}
_LOCALE_CACHE = {}


def _cached(cache, path, loader):
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    hit = cache.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        value = loader(path)
    except Exception as e:
        _log(f"RB2 character: couldn't read {path}: {e}")
        value = None
    cache[path] = (mtime, value)
    return value


def find_locale_files(near_path, root):
    """locale*.dta files beside the prefabs file or under the extracted files that name
    prefabs - the game splits its strings across several of these."""
    found = []
    dirs = [os.path.dirname(os.path.abspath(near_path))] if near_path else []
    for d in dirs:
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if f.lower().startswith('locale') and f.lower().endswith('.dta'):
                found.append(os.path.join(d, f))
    if root and os.path.isdir(root):
        for r, sub, files in os.walk(root):
            if r[len(root):].count(os.sep) >= 5:
                sub[:] = []
            for f in files:
                if f.lower().startswith('locale') and f.lower().endswith('.dta'):
                    full = os.path.join(r, f)
                    if full not in found:
                        found.append(full)
    return found


def display_names(near_path, root, explicit=""):
    names = {}
    paths = [explicit] if explicit else find_locale_files(near_path, root)
    for p in paths:
        got = _cached(_LOCALE_CACHE, p, parse_locale_prefab_names)
        if got:
            names.update(got)
    return names


# ---------------------------------------------------------------------------------------
# The extracted game files
# ---------------------------------------------------------------------------------------

_MILO_EXTS = ('.milo_xbox', '.milo_ps3', '.milo_wii', '.milo')
_INDEX = {}            # root -> (built at, {basename: [paths]})
_RESOURCE_RE = re.compile(rb'([A-Za-z0-9_\-./]+?_resource)\.milo')

SUPPORTED_PARTS = ('hair', 'heads', 'facehair', 'glasses', 'torso', 'hands', 'legs', 'feet',
                   'wrist', 'rings', 'earrings', 'piercings')
MAKEUP_PARTS = ('eyes', 'lips')
INSTRUMENT_PARTS = ('guitar', 'bass', 'drum', 'mic', 'keys')


def _strip_ext(fname):
    low = fname.lower()
    for ext in _MILO_EXTS:
        if low.endswith(ext):
            return low[:-len(ext)], ext
    return None, None


def milo_index(root, rebuild=False):
    """{lowercase basename: [paths]} for every milo under the extracted files."""
    if not rebuild and root in _INDEX:
        return _INDEX[root][1]
    t = time.time()
    index = {}
    for r, _dirs, files in os.walk(root):
        for f in files:
            base, ext = _strip_ext(f)
            if base:
                index.setdefault(base, []).append(os.path.join(r, f))
    _INDEX[root] = (time.time(), index)
    _log(f"RB2 character: indexed {sum(len(v) for v in index.values())} milo(s) under "
         f"{root} in {time.time() - t:.1f}s")
    return index


def find_milo(root, name, category=None, gender=None):
    """The best path for a milo named `name`: the same name can exist in several folders
    (hands/naked and torso/naked), so the part's category and the character's gender
    folder decide, then the platform (Xbox 360 first, since that's what's verified)."""
    paths = milo_index(root).get(name.lower())
    if not paths and time.time() - _INDEX.get(root, (0,))[0] > 30:
        paths = milo_index(root, rebuild=True).get(name.lower())   # files added since
    if not paths:
        return None

    def score(p):
        parts = [x.lower() for x in os.path.normpath(p).split(os.sep)]
        ext = _strip_ext(os.path.basename(p))[1]
        s = 0
        if gender and gender.lower() in parts:
            s += 8
        if category and category.lower() in parts:
            s += 4
        if 'gen' in parts:
            s += 1
        s += {'.milo_xbox': 3, '.milo_ps3': 2, '.milo': 1}.get(ext, 0)
        return s
    return max(paths, key=score)


def milo_summary(path):
    """(has top-level meshes, [resource milo basenames the directory references])."""
    with open(path, 'rb') as f:
        body = read_milo_container_body(f.read())
    objs, metas = cc.walk_milo(body)
    top = metas[0][0]
    has_mesh = any(t == 'Mesh' and d == top for d, t, _n, _a, _z in objs)
    resources = []
    for m in _RESOURCE_RE.finditer(metas[0][1]):
        base = os.path.basename(m.group(1).decode('latin-1')).lower()
        if base not in resources:
            resources.append(base)
    return has_mesh, resources


def find_deform(root, gender, instrument):
    """<gender>_<instrument>_deform, then any deform milo of that gender. Only the vocal
    ones have been checked; the others are assumed to hold the same five samples."""
    index = milo_index(root)
    inst = (instrument or '').lower()
    for cand in (inst, inst.rstrip('s'), 'vocal', 'guitar', 'bass', 'drum'):
        name = f"{gender}_{cand}_deform"
        if cand and name in index:
            return find_milo(root, name, None, gender)
    picks = sorted(k for k in index if k.startswith(gender.lower() + '_')
                   and k.endswith('_deform'))
    return find_milo(root, picks[0], None, gender) if picks else None


def find_skeleton(root, gender):
    index = milo_index(root)
    for name in (f"{gender}_skeleton", f"{gender}skeleton", "skeleton"):
        if name in index:
            return find_milo(root, name, None, gender)
    picks = [k for k in index if 'skeleton' in k and gender.lower() in k]
    if picks:
        return find_milo(root, sorted(picks, key=len)[0], None, gender)
    return None


# ---------------------------------------------------------------------------------------
# Preferences
# ---------------------------------------------------------------------------------------

class MiloToolsetPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    rb2_extracted_dir: StringProperty(
        name="Rock Band 2 Extracted Files",
        subtype='DIR_PATH',
        description="Folder holding Rock Band 2's extracted game files (the one containing "
                    "'char'). The Rock Band 2 character importer finds every outfit milo, "
                    "the colour palettes and the skin textures here")

    def draw(self, context):
        layout = self.layout
        layout.label(text="Location of Rock Band 2 extracted files (for the Rock Band 2 "
                          "character importer):")
        layout.prop(self, "rb2_extracted_dir", text="")


def extracted_root(context):
    addon = context.preferences.addons.get(__package__)
    path = addon.preferences.rb2_extracted_dir if addon else ""
    path = bpy.path.abspath(path) if path else ""
    return path.rstrip("\\/") if path else ""


# ---------------------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------------------

def _new_collection(context, name):
    coll = bpy.data.collections.new(name)
    context.scene.collection.children.link(coll)
    layer = context.view_layer.layer_collection.children.get(coll.name)
    if layer is not None:
        context.view_layer.active_layer_collection = layer
    return coll


def _activate(context, obj):
    for o in context.view_layer.objects:
        o.select_set(False)
    if obj is not None and obj.name in context.view_layer.objects:
        obj.select_set(True)
        context.view_layer.objects.active = obj
    else:
        context.view_layer.objects.active = None


class IMPORT_OT_rb2_character(bpy.types.Operator, ImportHelper):
    """(Experimental) Import a whole Rock Band 2 character - meshes, textures, colours,
    skin tone and proportions - from the game's prefabs.dta (or characters.dta, for the
    creator's starting looks) and the extracted files set in the add-on's preferences"""
    bl_idname = "import_scene.rb2_character"
    bl_label = "Import RB2 Character (Experimental)"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".dta"
    filter_glob: StringProperty(default="*.dta", options={'HIDDEN'})

    character: StringProperty(
        name="Character",
        description="The character's checksum name from prefabs.dta (e.g. guitar32). Its "
                    "display name (The Duke of Gravity) or prefab_guitar32 work too")
    allow_unselectable: BoolProperty(
        name="Allow Non-Selectable",
        description="Also import characters the game doesn't offer in its menus "
                    "(selectable is not TRUE). These are less tested",
        default=False)
    skeleton: EnumProperty(
        name="Skeleton",
        items=(('AUTO', "Selected or Auto", "Use the selected armature; if none is "
                "selected, look for the gender's skeleton milo in the extracted files"),
               ('ACTIVE', "Selected Armature", "Use the selected armature only"),
               ('NONE', "None", "Import the meshes without an armature")),
        default='AUTO')
    import_file_bones: BoolProperty(
        name="Add Bones From Each Part",
        description="Add bones a part defines itself (the head's face rig, hair physics "
                    "bones) to the armature so its meshes bind to them",
        default=True)
    apply_proportions: BoolProperty(
        name="Apply Proportions",
        description="Pose the armature for the character's height and weight from the "
                    "gender's deform milo. Adjustable afterwards in the armature's "
                    "Rock Band 2 Proportions panel",
        default=True)
    snap_proportions: BoolProperty(
        name="Snap to Slider Steps",
        description="Round height and weight to the creator's five slider steps. Off "
                    "keeps the prefab's own values (a few use values in between)",
        default=False)
    import_fur: BoolProperty(
        name="Import .fur (Shell Fur)",
        description="Build fur items (furbikini ...) as shell fur from their Fur object, "
                    "with a Geometry Nodes modifier. Off: a single flat (2D) fur layer, for "
                    "exports or renderers that can't use the shells. Switchable later in the "
                    "material panel",
        default=True)
    fit_outfit: BoolProperty(
        name="Fit Outfit Parts",
        description="Apply the parts' CharMeshHide (hide pieces another part covers) and "
                    "CharCuff (squeeze e.g. pants inside boots, into an 'RB2 Cuff' shape "
                    "key). The cuff squeeze is reconstructed, not taken from the game",
        default=True)
    load_textures: BoolProperty(
        name="Load Textures and Colours",
        description="Dress every part with its textures, its (colors a b) and the "
                    "character's skin tone and eye colour",
        default=True)
    eyebrow_override: BoolProperty(
        name="Use eyebrow_color_index",
        description="Colour the eyebrows from the character's eyebrow_color_index instead "
                    "of the head's (colors a b). Which one the game uses isn't confirmed",
        default=True)
    locale_file: StringProperty(
        name="Locale File", subtype='FILE_PATH',
        description="Optional locale .dta with the characters' display names. Found "
                    "automatically beside the prefabs file or in the extracted files")

    # -- file browser sidebar ----------------------------------------------------------

    def draw(self, context):
        layout = self.layout
        root = extracted_root(context)
        box = layout.box()
        box.label(text="Experimental", icon='EXPERIMENTAL')
        if not root or not os.path.isdir(root):
            box.label(text="Set the Rock Band 2 extracted files", icon='ERROR')
            box.label(text="folder in the add-on's preferences")
        else:
            box.label(text=os.path.basename(root) or root, icon='FILE_FOLDER')

        layout.prop(self, "character")
        layout.prop(self, "allow_unselectable")
        layout.prop(self, "skeleton")
        layout.prop(self, "import_file_bones")
        layout.prop(self, "apply_proportions")
        sub = layout.column()
        sub.enabled = self.apply_proportions
        sub.prop(self, "snap_proportions")
        layout.prop(self, "import_fur")
        layout.prop(self, "fit_outfit")
        layout.prop(self, "load_textures")
        sub = layout.column()
        sub.enabled = self.load_textures
        sub.prop(self, "eyebrow_override")
        layout.prop(self, "locale_file")

        path = self.filepath
        if path and path.lower().endswith('.dta') and os.path.isfile(path):
            prefabs = _cached(_PREFAB_CACHE, path, parse_prefabs)
            if prefabs:
                names = display_names(path, root, bpy.path.abspath(self.locale_file)
                                      if self.locale_file else "")
                box = layout.box()
                box.label(text="Selectable characters:")
                col = box.column(align=True)
                for key, pf in prefabs.items():
                    if pf['selectable']:
                        disp = names.get(key) or ("creator preset" if '/' in key else "?")
                        col.label(text=f"{key}  -  {disp}")

    # -- import --------------------------------------------------------------------------

    def _resolve_character(self, prefabs, names):
        want = self.character.strip()
        if want.lower().startswith('prefab_'):
            want = want[len('prefab_'):]
        if want in prefabs:
            return want
        low = {k.lower(): k for k in prefabs}
        if want.lower() in low:
            return low[want.lower()]
        for key, disp in names.items():
            if disp.lower() == want.lower() and key in prefabs:
                return key
        return None

    def _variant_fur_meshes(self, context, milo, armature):
        """A texture variant can bring its own fur layer (furbikini_cheetah ->
        furbikini_fur.mesh); the resource milo doesn't have it. Imports the variant's
        meshes and keeps the ones drawn with a fur material - not patches and the like."""
        try:
            probe = cc.read_texture_milo(milo)
        except Exception:
            return []
        fur_mats = {sl['target'] for sl in probe.slots if sl.get('fur')}
        if not fur_mats:
            return []
        before = set(bpy.data.objects)
        _activate(context, armature)
        try:
            bpy.ops.import_scene.rb2_meshes(
                filepath=milo, exclude_lod=True, exclude_shadow=True, exclude_helpers=True,
                import_file_bones=self.import_file_bones,
                parent_to_armature=armature is not None)
        except Exception as e:
            _log(f"    fur mesh import failed: {e}")
            return []
        keep = []
        for o in [o for o in bpy.data.objects if o not in before]:
            mats = {cc._base_name(s.material.name) for s in getattr(o, 'material_slots', ())
                    if s.material is not None}
            if o.type == 'MESH' and mats & fur_mats:
                keep.append(o)
            else:
                bpy.data.objects.remove(o, do_unlink=True)
        if keep:
            _log(f"    fur layer from {os.path.basename(milo)}: "
                 + ", ".join(o.name for o in keep))
            if self.apply_proportions:
                try:
                    rb2_proportions.attach_meshdeforms(keep, milo)
                except Exception as e:
                    _log(f"    fur girth data unreadable: {e}")
        return keep

    def execute(self, context):
        root = extracted_root(context)
        if not root or not os.path.isdir(root):
            self.report({'ERROR'}, "Set 'Rock Band 2 Extracted Files' in the Milo Toolset "
                                   "add-on preferences first")
            return {'CANCELLED'}
        prefabs = _cached(_PREFAB_CACHE, self.filepath, parse_prefabs)
        if not prefabs:
            self.report({'ERROR'}, "No (prefab ...) characters found in this file - is it "
                                   "the decrypted prefabs.dta?")
            return {'CANCELLED'}
        names = display_names(self.filepath, root, bpy.path.abspath(self.locale_file)
                              if self.locale_file else "")
        key = self._resolve_character(prefabs, names)
        if key is None:
            self.report({'ERROR'}, f"No character '{self.character}' in this file (use its "
                                   f"checksum name, e.g. guitar32)")
            return {'CANCELLED'}
        pf = prefabs[key]
        if not pf['selectable'] and not self.allow_unselectable:
            self.report({'ERROR'}, f"'{key}' isn't a selectable character - enable 'Allow "
                                   f"Non-Selectable' to import it anyway")
            return {'CANCELLED'}

        title = names.get(key, key)
        gender = pf['gender']
        _log(f"===== RB2 character (experimental): {key} '{title}' ({gender}) =====")
        skipped, warnings = [], []

        prev_active = context.view_layer.objects.active
        coll = _new_collection(context, title)
        coll["rb2_prefab"] = key
        coll["rb2_height"] = pf['height']
        coll["rb2_weight"] = pf['weight']

        # Armature
        armature = None
        if self.skeleton in ('AUTO', 'ACTIVE') and prev_active is not None \
                and prev_active.type == 'ARMATURE':
            armature = prev_active
            _log(f"  armature: using selected '{armature.name}'")
        elif self.skeleton == 'AUTO':
            skel = find_skeleton(root, gender)
            if skel:
                _activate(context, None)
                try:
                    bpy.ops.import_scene.tbrb_skeleton(filepath=skel)
                    a = context.view_layer.objects.active
                    armature = a if a is not None and a.type == 'ARMATURE' else None
                except Exception as e:
                    _log(f"  skeleton import failed ({e})")
                _log(f"  armature: {'imported' if armature else 'could not import'} "
                     f"{skel}")
                # Put the new skeleton's rest pose on the neutral body every outfit milo is
                # bound at, before any part adds its own bones (hair physics bones are
                # placed relative to the armature's rest - off the head otherwise).
                if armature is not None:
                    deform = find_deform(root, gender, pf['instrument'])
                    if deform is None:
                        _log("  armature: no deform milo to conform the skeleton to the "
                             "neutral body - left as imported")
                    else:
                        try:
                            n_moved, med = rb2_proportions.conform_rest_to_neutral(
                                context, armature, rb2_proportions.read_deform_milo(deform))
                            _log(f"  armature: rest pose conformed to the neutral body from "
                                 f"{os.path.basename(deform)} ({n_moved} bone(s), median "
                                 f"move {med:.2f})")
                        except Exception as e:
                            _log(f"  armature: couldn't conform the rest pose ({e})")
            else:
                warnings.append(f"no {gender} skeleton milo found - select an armature and "
                                f"import again to rig the meshes")
                _log(f"  armature: no '{gender}_skeleton' milo found in the extracted files")

        # Character-wide colours, and the palettes, before anything is dressed
        if self.load_textures:
            scn = context.scene
            scn.rb2cc["skin_index"] = pf['skin_color_index']
            scn.rb2cc["eye_index"] = pf['eye_color_index']
            if not cc._palettes(context):
                pal = find_milo(root, "colorpalettes")
                if pal:
                    try:
                        cc.store_palettes(scn, cc.read_palette_milo(pal), pal)
                        scn.rb2cc["palette_milo"] = pal
                    except Exception as e:
                        _log(f"  palettes: couldn't read {pal}: {e}")

        imported_parts = 0
        fit_parts = []
        makeup = []
        head_skin = []
        for category, item, colors in pf['outfit']:
            if item.lower() == 'none':
                continue
            if category in MAKEUP_PARTS:
                makeup.append((category, item))         # once the head is dressed
                continue
            if category in INSTRUMENT_PARTS:
                skipped.append(f"{category} ({item})")
                continue
            if category not in SUPPORTED_PARTS:
                skipped.append(f"{category} ({item})")
                continue
            milo = find_milo(root, item, category, gender)
            if milo is None:
                warnings.append(f"{category}: '{item}' not found in the extracted files")
                _log(f"  {category}: '{item}' NOT FOUND")
                continue
            try:
                has_mesh, resources = milo_summary(milo)
            except Exception as e:
                warnings.append(f"{category}: couldn't read '{item}' ({e})")
                continue

            mesh_src = None
            for res in resources:
                mesh_src = find_milo(root, res, category, gender)
                if mesh_src:
                    break
            if mesh_src is None and has_mesh:
                mesh_src = milo
            if mesh_src is None:
                warnings.append(f"{category}: no meshes for '{item}'"
                                + (f" (resource {resources[0]} not found)" if resources
                                   else ""))
                continue
            _log(f"  {category}: {item} -> meshes from {os.path.basename(mesh_src)}"
                 + (f", textures from {os.path.basename(milo)}" if mesh_src != milo else ""))

            before = set(bpy.data.objects)
            _activate(context, armature)
            try:
                result = bpy.ops.import_scene.rb2_meshes(
                    filepath=mesh_src, exclude_lod=True, exclude_shadow=True,
                    exclude_helpers=True, import_file_bones=self.import_file_bones,
                    parent_to_armature=armature is not None)
            except Exception as e:
                result = {'CANCELLED'}
                _log(f"    mesh import failed: {e}")
            new_objs = [o for o in bpy.data.objects if o not in before]
            if 'FINISHED' not in result or not new_objs:
                warnings.append(f"{category}: meshes from '{os.path.basename(mesh_src)}' "
                                f"didn't import")
                continue
            imported_parts += 1
            if mesh_src != milo:
                new_objs += self._variant_fur_meshes(context, milo, armature)
            for o in new_objs:
                o["rb2_part"] = category
                o["rb2_item"] = item
            if self.fit_outfit:
                fit = dict(hides=[], cuffs=[])
                for src in {mesh_src, milo}:
                    try:
                        got = rb2_outfit_fit.read_fit(src)
                        fit['hides'] += got['hides']
                        fit['cuffs'] += got['cuffs']
                    except Exception as e:
                        _log(f"    hide/cuff data unreadable in {os.path.basename(src)}: {e}")
                if fit['hides'] or fit['cuffs']:
                    _log(f"    fitting: {len(fit['hides'])} CharMeshHide, "
                         f"{len(fit['cuffs'])} CharCuff")
                fit_parts.append((category, fit, new_objs))
            if self.apply_proportions:
                try:
                    girth = rb2_proportions.attach_meshdeforms(new_objs, mesh_src)
                    if girth:
                        _log(f"    girth (MeshDeform) data on {', '.join(girth)}")
                except Exception as e:
                    _log(f"    girth data unreadable: {e}")

            if not self.load_textures:
                continue
            mats = []
            for o in new_objs:
                for slot in getattr(o, 'material_slots', ()):
                    if slot.material is not None and slot.material not in mats:
                        mats.append(slot.material)
            try:
                info = cc.read_texture_milo(milo)
            except Exception as e:
                warnings.append(f"{category}: textures unreadable ({e})")
                continue
            res_slots = {}
            if mesh_src != milo:
                # A variant can name textures that only its resource milo stores
                # (hotpants_camo -> hotpants_camo_legs_diff.tex in hotpants_resource).
                try:
                    res_info = cc.read_texture_milo(mesh_src)
                    res_slots = {sl['target']: sl for sl in res_info.slots
                                 if any(sl.get(k) for k in cc._TEX_KEYS)}
                    res_texs = res_info.texs
                    added = [n for n, t in res_texs.items() if cc._has_pixels(t)
                             and not cc._has_pixels(info.texs.get(n))]
                    for n in added:
                        info.texs[n] = res_texs[n]
                except Exception as e:
                    _log(f"    resource textures unreadable: {e}")
            by_target = {sl['target']: sl for sl in info.slots}
            jobs = []
            for m in mats:
                base = cc._base_name(m.name)
                sl = by_target.get(base)
                if sl is None and base in res_slots:
                    # Defined only by the resource milo, with its own textures there
                    # (tighterpants' alpha-cut hoop cards: legs_tighterpants_hoop.mat).
                    sl = res_slots[base]
                    info.slots.append(sl)
                    by_target[base] = sl
                    _log(f"    material '{m.name}': defined by the resource milo - dressed "
                         f"from it")
                if sl is None:
                    # The bare skin a pair of pants or a jacket leaves showing is its own
                    # mesh with a skin material the outfit's textures don't define.
                    sl = cc.skin_slot_for(base)
                    if sl is None:
                        _log(f"    material '{m.name}' isn't defined by {item} - left as is")
                        continue
                    info.slots.append(sl)
                    by_target[base] = sl
                    _log(f"    material '{m.name}': skin the outfit doesn't define - dressed "
                         f"as character skin")
                jobs.append((m, sl))
            if not jobs:
                continue
            done, warn = cc.dress_jobs(context, info, jobs, milo, choice_indices=colors,
                                       fur_shells=self.import_fur)
            warnings.extend(f"{category}: {w}" for w in warn)
            head_skin.extend(m for m, _sl in jobs if m.get("rb2_makeup_rects"))

            if category == 'heads' and self.eyebrow_override \
                    and pf['eyebrow_color_index'] is not None:
                for m, _sl in jobs:
                    pg = m.rb2cc
                    if pg.kind == 'ITEM' and pg.uses1 and pg.sel_palette1 == 'hair.pal':
                        pg["sel_index1"] = int(pf['eyebrow_color_index'])
                        pg["cfg_index1"] = int(pf['eyebrow_color_index'])
                        cc.apply_material_colors(m, context)
                        _log(f"    '{m.name}': eyebrows -> hair.pal"
                             f"[{pf['eyebrow_color_index']}] (eyebrow_color_index)")

        # Eye and lip makeup, drawn into the head where its skin proxy places makeup.
        for category, item in makeup:
            if not self.load_textures:
                break
            if not head_skin:
                skipped.append(f"{category} makeup ({item}) - no head skin to put it on")
                continue
            path = find_milo(root, item, category, gender)
            if path is None:
                warnings.append(f"{category}: makeup '{item}' not found in the extracted "
                                f"files")
                continue
            for m in head_skin:
                try:
                    kind = cc.set_makeup(m, 'eyes' if category == 'eyes' else 'lips', path,
                                         context)
                    _log(f"  {category}: makeup {item} on '{m.name}' ({kind})")
                except Exception as e:
                    warnings.append(f"{category}: makeup '{item}' ({e})")

        # Any skin/eye material dressed above follows the character's indices.
        for m in bpy.data.materials:
            pg = getattr(m, 'rb2cc', None)
            if pg is not None and pg.loaded and pg.kind in cc.CHARACTER_PALETTES:
                cc.apply_material_colors(m, context)

        if self.apply_proportions:
            if armature is None:
                skipped.append(f"proportions (height {pf['height']}, weight "
                               f"{pf['weight']}) - no armature")
            else:
                deform = find_deform(root, gender, pf['instrument'])
                if deform is None:
                    warnings.append(f"no {gender} deform milo found - proportions not "
                                    f"applied")
                else:
                    try:
                        armature.rb2_proportions["snap_to_steps"] = self.snap_proportions
                        applied, unmatched = rb2_proportions.load_and_apply(
                            armature, deform, pf['height'], pf['weight'])
                        _log(f"  proportions: height {pf['height']}, weight "
                             f"{pf['weight']} from {os.path.basename(deform)} "
                             f"({applied} channel(s))")
                        if unmatched and not applied:
                            warnings.append("proportions: no deform channel matched a "
                                            "bone of the armature")
                    except Exception as e:
                        warnings.append(f"proportions: {e}")
        if self.fit_outfit and fit_parts:
            try:
                hidden, combined = rb2_outfit_fit.apply_hides(fit_parts)
                if combined:
                    _log(f"  hides: combined flags {combined:#x}; hidden: "
                         + (", ".join(f"{n} ({c})" for n, c, _f in hidden) or "nothing - no "
                            "part lists a mesh for these flags"))
            except Exception as e:
                warnings.append(f"hides: {e}")
            if armature is not None:
                try:
                    for cuff, obj, count in rb2_outfit_fit.apply_cuffs(fit_parts, armature):
                        _log(f"  cuffs: {cuff} squeezed {count} vertex(es) of '{obj}'")
                except Exception as e:
                    warnings.append(f"cuffs: {e}")
            else:
                _log("  cuffs: skipped - no armature")

        if pf['has_tattoo']:
            skipped.append("tattoos")
        _log(f"  not supported yet, skipped: {', '.join(skipped)}")
        for w in warnings:
            _log(f"  WARNING: {w}")
        msg = (f"Imported '{title}' ({key}): {imported_parts} part(s)"
               + (f"; {len(warnings)} warning(s), see the log" if warnings else ""))
        _log(f"===== {msg} =====")
        self.report({'WARNING'} if warnings else {'INFO'}, msg)
        if armature is not None:
            _activate(context, armature)
        return {'FINISHED'}


# ---------------------------------------------------------------------------------------
# Registration (called from the add-on's register / unregister)
# ---------------------------------------------------------------------------------------

_classes = (
    MiloToolsetPreferences,
    IMPORT_OT_rb2_character,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
