"""
Rock Band 3 character importer (EXPERIMENTAL).

Builds a character from one of the game's prefabs (char/main/prefabs.milo - BandCharDesc
objects) or one the players made, from a save (see 'Saves' below), and the Xbox 360 dump set in the add-on's preferences ("Rock Band 3 Xbox 360
Dump", the folder containing 'char'). For now it brings in the armature, every outfit and
instrument mesh with its vertex groups, placeholder material slots, and the body
proportions (height / weight / muscle). Textures, colours, face sculpting and makeup come
later.

===========================================================================================
WHERE THINGS ARE (BandCharDesc::MakeOutfitPath / MakeInstrumentPath, decomp)
===========================================================================================
    outfit piece   char/main/<piece>/<gender>/<item>.milo   eyebrows, facehair, hair,
                   earrings, glasses, piercings, feet, hands, legs, rings, torso, wrist
    head           char/main/head/<gender>/head.milo        (unless the head is hidden)
    instrument     char/main/<inst>/<item>.milo             guitar, bass, mic, keyboard
                   char/main/drum/<item>_<venue>.milo       drums add a venue suffix
    skeleton       char/main/skeleton.milo   the one rig every RB3 character shares
    deform         char/main/deform.milo     CharClipSet with the 'male' / 'female' clips

Files are found by name anywhere under the dump, preferring copies whose folders match the
piece and gender, so a dump laid out as char/main/torso/male/gen/... works as well.

RB3 items keep their meshes, MeshDeforms and collisions in an inline
'<item>/<item>_resource.milo' subdirectory; the texture variant (admiraljacket_ornate) is
the outer directory. The meshes are read with the nested walker (mesh_importer.
parse_rb3_meshes), the MeshDeforms likewise (rb3_proportions.read_meshdeforms).

===========================================================================================
PREFABS (BandCharDesc revision 0x11, decomp bandobj/BandCharDesc.cpp)
===========================================================================================
    prefab symbol, gender symbol, skin colour int,
    head: hide byte, eye colour, shape, chin, chin w/h, jaw w/h, nose, nose w/h, eye,
          eye separation/height/rotation, mouth, mouth w/h, brow separation/height
    outfit: 12 x (symbol name, u8 colour x3)  eyebrows earrings facehair glasses hair
          piercings feet hands legs rings torso wrist
    height, weight, muscle floats
    instruments: 5 x (symbol name, u8 colour x3)  guitar bass drum mic keyboard
    patches: count x (texture int, category int, mesh string, uv, rotation, scale)
The prefabs at the top of prefabs.milo are the selectable ones; the inline subdirectories
hold the creator presets (charcreator_*, facetype_*) and debug characters. Display names
come from the locale files (ui/locale/eng/*.dta), keyed by the object's own name.
"""

import os
import re
import struct
import time

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, StringProperty
from bpy_extras.io_utils import ImportHelper

from .io import read_milo_container_body
from .utilities import _log
from . import rb3_proportions as rb3p
from . import rb3_face
from . import rb3_textures
from .rb2_character_importer import (parse_dta, DtaString, _read_text, _cached,
                                     _new_collection, _activate, _strip_ext)


# ---------------------------------------------------------------------------------------
# Prefabs
# ---------------------------------------------------------------------------------------

OUTFIT_PIECES = ('eyebrows', 'earrings', 'facehair', 'glasses', 'hair', 'piercings', 'feet',
                 'hands', 'legs', 'rings', 'torso', 'wrist')
INSTRUMENTS = ('guitar', 'bass', 'drum', 'mic', 'keyboard')


class _R:
    def __init__(self, b, p):
        self.b, self.p = b, p

    def u8(self):
        v = self.b[self.p]; self.p += 1; return v

    def i32(self):
        v = struct.unpack_from('>i', self.b, self.p)[0]; self.p += 4; return v

    def f32(self):
        v = struct.unpack_from('>f', self.b, self.p)[0]; self.p += 4; return v

    def sym(self):
        n = self.i32()
        if not 0 <= n < 256:
            raise ValueError("bad symbol")
        v = self.b[self.p:self.p + n].decode('latin-1'); self.p += n
        return v


def _piece(r):
    return dict(name=r.sym(), colors=[r.u8(), r.u8(), r.u8()])


def parse_band_char_desc(b):
    """One BandCharDesc object (revision 0x11) as a dict, or None. The object header's
    length varies (debug prefabs carry a property array), so the fields are found by
    trying header lengths until the whole object parses and ends on its end marker."""
    for h in range(8, 320):
        try:
            r = _R(b, h)
            prefab = r.sym()
            gender = r.sym()
            if gender not in ('male', 'female'):
                continue
            skin = r.i32()
            head = dict(hide=r.u8(), eye_color=r.i32(), shape=r.i32(), chin=r.i32(),
                        chin_width=r.f32(), chin_height=r.f32(), jaw_width=r.f32(),
                        jaw_height=r.f32(), nose=r.i32(), nose_width=r.f32(),
                        nose_height=r.f32(), eye=r.i32(), eye_separation=r.f32(),
                        eye_height=r.f32(), eye_rotation=r.f32(), mouth=r.i32(),
                        mouth_width=r.f32(), mouth_height=r.f32(), brow_separation=r.f32(),
                        brow_height=r.f32())
            outfit = {k: _piece(r) for k in OUTFIT_PIECES}
            height, weight, muscle = r.f32(), r.f32(), r.f32()
            inst = {k: _piece(r) for k in INSTRUMENTS}
            patches = []
            for _ in range(r.i32()):
                patches.append(dict(texture=r.i32(), category=r.i32(), mesh=r.sym(),
                                    uv=(r.f32(), r.f32()), rotation=r.f32(),
                                    scale=(r.f32(), r.f32())))
            if r.p not in (len(b), len(b) - 4):
                continue
            if not all(0.0 <= v <= 1.0 for v in (height, weight, muscle)):
                continue
            return dict(prefab=prefab, gender=gender, skin_color=skin, head=head,
                        outfit=outfit, height=height, weight=weight, muscle=muscle,
                        instruments=inst, patches=patches)
        except (ValueError, struct.error, IndexError, UnicodeDecodeError):
            continue
    return None


def parse_prefabs_milo(path):
    """{object name: character dict} from prefabs.milo; each dict also gets 'group':
    'prefab' (the selectable ones at the top), 'creator' (charcreator_* / facetype_*
    presets) or 'debug'."""
    from .mesh_importer import walk_milo_entries
    with open(path, 'rb') as f:
        body = read_milo_container_body(f.read())
    chars = {}
    for depth, _p, etype, ename, s, e in walk_milo_entries(body):
        if etype != 'BandCharDesc':
            continue
        c = parse_band_char_desc(body[s:e])
        if c is None:
            _log(f"RB3 character: couldn't read prefab '{ename}'")
            continue
        if depth == 0:
            c['group'] = 'prefab'
        elif ename.startswith(('charcreator_', 'facetype_')):
            c['group'] = 'creator'
        else:
            c['group'] = 'debug'
        chars[ename] = c
    return chars


# ---------------------------------------------------------------------------------------
# Saves
# ---------------------------------------------------------------------------------------
#
# RB3 saves are FixedSizeSaveable streams (decomp meta/FixedSizeSaveable.cpp): every
# record is padded to a fixed size. Same layout on Xbox 360 and PS3, little-endian, no
# encryption over the characters (only the song scores block is scrambled):
#
#     u32 version (152), symbol table: u32 count, then max-symbols x 0x36
#         {u32 length + text padded to 0x32, u32 id}  - symbols saved by id refer here
#     TourProgress ...                               (PS3: 0x74 bytes longer)
#     u32 character count, then 10 TourChar slots (band3/tour/TourChar.cpp):
#         name     u32 length + text, padded to 0x80
#         guid     0x14 bytes
#         BandCharDesc::SaveFixed (bandobj/BandCharDesc.cpp):
#             gender symbol id, skin colour i32
#             head    19 x 4 bytes (eye colour, shape, chin, ... brow height - no 'hide')
#             outfit  12 x {name u32 length + text padded to 0x32, colours 3 x i32}
#             height, weight, muscle floats
#             instruments 5 x the same piece record
#             patches u32 count, 16 x 0x9C {texture i32, category i32, mesh string padded
#                     to 0x80, uv 2f, rotation f, scale 2f}
#         portrait RndBitmap, 256x512 DXT5 (0x20 header + 0x20000)
# The character list's position shifts between platforms, so it's found by structure.

class RB3SaveError(Exception):
    pass


_HEAD_FIELDS = ('eye_color', 'shape', 'chin', 'chin_width', 'chin_height', 'jaw_width',
                'jaw_height', 'nose', 'nose_width', 'nose_height', 'eye', 'eye_separation',
                'eye_height', 'eye_rotation', 'mouth', 'mouth_width', 'mouth_height',
                'brow_separation', 'brow_height')
_HEAD_INTS = {'eye_color', 'shape', 'chin', 'nose', 'eye', 'mouth'}
_SAVE_MAX_CHARS = 10
_SAVE_DESC_SIZE = 8 + 0x4C + 12 * 0x3E + 12 + 5 * 0x3E + 4 + 16 * 0x9C     # 0xE42
_SAVE_CHAR_HEAD = 0x80 + 0x14                                              # name + guid
_SAVE_ID_SPAN = 0x36


def _fixed_text(b, o, size):
    n = struct.unpack_from('<I', b, o)[0]
    if n > size - 4:
        raise RB3SaveError("bad fixed string")
    if any(b[o + 4 + n:o + size]):
        raise RB3SaveError("fixed string not padded")
    return b[o + 4:o + 4 + n].decode('utf-8')


def _save_symbols(b):
    """(version, {id: symbol}, end of the used entries) from the save's symbol table."""
    if len(b) < 16:
        raise RB3SaveError("file too small")
    version, count = struct.unpack_from('<II', b, 0)
    if not 100 <= version < 1000 or not 0 < count < 100000:
        raise RB3SaveError("not a Rock Band 3 save")
    syms, o = {}, 8
    for _ in range(count):
        s = _fixed_text(b, o, 0x32)
        syms[struct.unpack_from('<I', b, o + 0x32)[0]] = s
        o += _SAVE_ID_SPAN
    return version, syms, o


def _save_piece(b, o):
    name = _fixed_text(b, o, 0x32)
    if name and not re.fullmatch(r'[A-Za-z0-9_.\-]+', name):
        raise RB3SaveError("bad item name")
    cols = list(struct.unpack_from('<3i', b, o + 0x32))
    # -1 marks a colour slot the item doesn't use
    return dict(name=name, colors=[c if c >= 0 else 0 for c in cols])


def _save_char(b, o, syms):
    """One TourChar slot at o as a character dict (the prefab dict's shape), or raise."""
    name = _fixed_text(b, o, 0x80)
    if not name.strip():
        raise RB3SaveError("no name")
    p = o + _SAVE_CHAR_HEAD
    gender = syms.get(struct.unpack_from('<i', b, p)[0])
    if gender not in ('male', 'female'):
        raise RB3SaveError("not a character")
    skin = struct.unpack_from('<i', b, p + 4)[0]
    p += 8
    head = {'hide': 0}
    for k in _HEAD_FIELDS:
        head[k] = struct.unpack_from('<i' if k in _HEAD_INTS else '<f', b, p)[0]
        p += 4
    outfit = {}
    for k in OUTFIT_PIECES:
        outfit[k] = _save_piece(b, p)
        p += 0x3E
    height, weight, muscle = struct.unpack_from('<3f', b, p)
    p += 12
    if not all(0.0 <= v <= 1.0 for v in (height, weight, muscle)) or not 0 <= skin < 64:
        raise RB3SaveError("bad body values")
    inst = {}
    for k in INSTRUMENTS:
        inst[k] = _save_piece(b, p)
        p += 0x3E
    n = struct.unpack_from('<i', b, p)[0]
    p += 4
    if not 0 <= n <= 16:
        raise RB3SaveError("bad patch count")
    patches = []
    for i in range(n):
        q = p + i * 0x9C
        tex, cat = struct.unpack_from('<2i', b, q)
        mesh = _fixed_text(b, q + 8, 0x80)
        u, v, rot, sx, sy = struct.unpack_from('<5f', b, q + 0x88)
        patches.append(dict(texture=tex, category=cat, mesh=mesh, uv=(u, v), rotation=rot,
                            scale=(sx, sy)))
    return dict(prefab='', gender=gender, skin_color=skin, head=head, outfit=outfit,
                height=height, weight=weight, muscle=muscle, instruments=inst,
                patches=patches, save_name=name.strip(), group='save')


def _save_slot_size(b, o):
    """The slot stride: the descriptor plus the portrait bitmap whose header follows it
    (bpp at +1, width / height u16 at +7 / +9; 0x20-byte header)."""
    bm = o + _SAVE_CHAR_HEAD + _SAVE_DESC_SIZE
    bpp = b[bm + 1]
    w, h = struct.unpack_from('<HH', b, bm + 7)
    if b[bm] != 1 or bpp not in (4, 8, 16, 24, 32) or not 0 < w <= 1024 or not 0 < h <= 1024:
        raise RB3SaveError("no portrait after the character")
    return _SAVE_CHAR_HEAD + _SAVE_DESC_SIZE + 0x20 + w * h * bpp // 8


_ART_SLOTS = 8
_ART_RECORD = 0x10234          # PatchDir: layer data, then the art's 256x256 DXT5 bitmap
_ART_BITMAP = 0x214            # bitmap offset in the record (after the list's count)


def _bitmap_at(b, o):
    """(width, height, encoding, data) of a saved RndBitmap (0x20-byte header) or None."""
    if o + 0x20 > len(b) or b[o] != 1:
        return None
    enc = struct.unpack_from('<I', b, o + 2)[0]
    w, h = struct.unpack_from('<HH', b, o + 7)
    if enc not in (8, 24) or not 0 < w <= 1024 or not 0 < h <= 1024:
        return None
    size = w * h // 2 if enc == 8 else w * h
    if o + 0x20 + size > len(b):
        return None
    return w, h, enc, b[o + 0x20:o + 0x20 + size]


def _decode_bitmap(bm, swapped):
    """RGBA (top row first, 0-255) of a saved bitmap: Xbox 360 saves keep the DXT blocks
    byte-swapped like its textures, PS3 saves don't."""
    from . import texture_importer as ti
    w, h, enc, data = bm
    if not swapped:
        data = ti._byteswap16(data)          # decode_rgba swaps; this undoes it
    return ti.decode_rgba(dict(width=w, height=h, encoding=enc, data=data))


def _roughness(rgba):
    a = rgba[..., 3]
    return float(np.abs(np.diff(a, axis=0)).mean() + np.abs(np.diff(a, axis=1)).mean())


def _save_art(b, list_end, probe):
    """{art slot: RGBA} - the profile's PatchDirs after the 10 character slots (the art a
    patch's texture index names). probe: a bitmap (the first portrait) to tell whether the
    save's blocks are byte-swapped - garbled DXT is far noisier."""
    try:
        swapped = _roughness(_decode_bitmap(probe, True)) <= \
            _roughness(_decode_bitmap(probe, False))
    except Exception:
        swapped = True
    art = {}
    for k in range(_ART_SLOTS):
        bm = _bitmap_at(b, list_end + _ART_BITMAP + 4 + k * _ART_RECORD)
        if bm is None:
            continue
        try:
            art[k] = _decode_bitmap(bm, swapped)
        except Exception as e:
            _log(f"RB3 save: art slot {k} not decoded ({e})")
    return art, swapped


def _save_characters(b):
    """[(slot, character dict)] from a plain RB3 save.dat / SAVE.DAT."""
    _version, syms, start = _save_symbols(b)
    # the character count (1..10) followed by a character's name: scan the profile for the
    # first spot where both parse; the slots then follow at the portrait-derived stride
    pat = re.compile(rb'[\x01-\x0a]\x00\x00\x00[\x01-\x7c]\x00\x00\x00', re.S)
    for m in pat.finditer(b, start, min(len(b), start + 0x200000)):
        o = m.start() + 4
        try:
            first = _save_char(b, o, syms)
            stride = _save_slot_size(b, o)
        except (RB3SaveError, struct.error, UnicodeDecodeError, IndexError):
            continue
        count = b[m.start()]
        out = [(0, first)]
        for slot in range(1, _SAVE_MAX_CHARS):
            q = o + slot * stride
            try:
                out.append((slot, _save_char(b, q, syms)))
            except (RB3SaveError, struct.error, UnicodeDecodeError, IndexError):
                continue
        if len(out) != count:
            _log(f"RB3 save: the save lists {count} character(s), {len(out)} read")
        # the profile's art (tattoos, decals): shared by all its characters
        art, swapped = {}, True
        probe = _bitmap_at(b, o + _SAVE_CHAR_HEAD + _SAVE_DESC_SIZE)
        if probe is not None:
            art, swapped = _save_art(b, o + _SAVE_MAX_CHARS * stride, probe)
        for _slot, c in out:
            c['art'] = art
            c['platform'] = 'xbox' if swapped else 'ps3'
        return out
    return []


_SAVE_KIND_CACHE = {}


def is_rb3_save(path):
    """An RB3 save rather than prefabs.milo: an Xbox 360 package, or a plain save whose
    start is the version and a symbol table."""
    try:
        st = os.stat(path)
    except OSError:
        return False
    key = (path, st.st_mtime, st.st_size)
    if key not in _SAVE_KIND_CACHE:
        try:
            with open(path, 'rb') as f:
                head = f.read(0x100)
        except OSError:
            return False
        # .dat / .sav: save.dat, SAVE.DAT - also when still encrypted, so the reader can
        # say so
        result = (head[:4] in (b'CON ', b'LIVE', b'PIRS')
                  or path.lower().endswith(('.dat', '.sav')))
        if not result and len(head) >= 0x50:
            try:
                _version, count = struct.unpack_from('<II', head, 0)
                n = struct.unpack_from('<I', head, 8)[0]
                result = (100 <= _version < 1000 and 0 < count < 100000 and 0 < n < 0x2E
                          and re.fullmatch(rb'[ -~]+', head[12:12 + n]) is not None)
            except struct.error:
                result = False
        _SAVE_KIND_CACHE[key] = result
    return _SAVE_KIND_CACHE[key]


def parse_rb3_save(path):
    """{'save<slot>': character dict} from an RB3 save: an Xbox 360 CON package (its
    save.dat) or a plain save.dat / PS3 SAVE.DAT."""
    from .rb2_character_importer import _stfs_files, SaveError
    with open(path, 'rb') as f:
        data = f.read()
    if data[:4] in (b'CON ', b'LIVE', b'PIRS'):
        try:
            files = _stfs_files(data)
        except SaveError as e:
            raise RB3SaveError(str(e))
        candidates = files.get('save.dat', [])
        if not candidates:
            raise RB3SaveError("no save.dat in this package")
    else:
        candidates = [data]
    for raw in candidates:
        try:
            chars = _save_characters(raw)
        except (RB3SaveError, struct.error, UnicodeDecodeError):
            chars = []
        if chars:
            return {f"save{slot}": c for slot, c in chars}
    if data[:4] not in (b'CON ', b'LIVE', b'PIRS') and not _plain_save_start(data):
        raise RB3SaveError("this looks like a PS3 save still under the console's encryption "
                           "- decrypt it first with a PS3 save tool")
    raise RB3SaveError("no characters found - is this a Rock Band 3 save?")


def _plain_save_start(data):
    try:
        _save_symbols(data[:0x1000] if len(data) > 0x1000 else data)
        return True
    except (RB3SaveError, struct.error, UnicodeDecodeError):
        # the table runs past the first page: only its first entry has to be readable
        try:
            version, count = struct.unpack_from('<II', data, 0)
            return 100 <= version < 1000 and 0 < count < 100000 \
                and bool(_fixed_text(data, 8, 0x32))
        except (RB3SaveError, struct.error, UnicodeDecodeError):
            return False


def load_characters(path):
    """Characters from prefabs.milo or from a save."""
    if is_rb3_save(path):
        return parse_rb3_save(path)
    return parse_prefabs_milo(path)


def parse_locale_names(path):
    """{symbol: string} for every (symbol "string") entry of a locale .dta."""
    names = {}
    for x in parse_dta(_read_text(path)):
        if isinstance(x, list) and len(x) >= 2 and isinstance(x[0], str) \
                and isinstance(x[1], DtaString):
            names[x[0]] = str(x[1])
    return names


_PREFAB_CACHE = {}
_LOCALE_CACHE = {}


def locale_files(root, explicit=""):
    if explicit:
        return [explicit]
    out = []
    for d in (os.path.join(root, 'ui', 'locale', 'eng'),
              os.path.join(root, 'ui', 'locale', 'eng', 'gen')):
        if os.path.isdir(d):
            out += [os.path.join(d, f) for f in sorted(os.listdir(d))
                    if f.lower().endswith('.dta')]
    return out


def display_names(root, explicit=""):
    names = {}
    for p in locale_files(root, explicit):
        got = _cached(_LOCALE_CACHE, p, parse_locale_names)
        if got:
            names.update(got)
    return names


def display_name(names, key):
    for k in (key, 'prefab_' + key):
        if k in names:
            return names[k]
    return key


# ---------------------------------------------------------------------------------------
# The dump
# ---------------------------------------------------------------------------------------

_INDEX = {}


def milo_index(root, rebuild=False):
    """{lowercase basename without extension: [paths]} for every milo under the dump."""
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
    _log(f"RB3 character: indexed {sum(len(v) for v in index.values())} milo(s) under "
         f"{root} in {time.time() - t:.1f}s")
    return index


def _score(p, folders):
    parts = [x.lower() for x in os.path.normpath(p).split(os.sep)]
    ext = _strip_ext(os.path.basename(p))[1]
    s = sum(8 for f in folders if f and f.lower() in parts)
    if 'main' in parts:
        s += 2
    s += {'.milo_xbox': 3, '.milo_ps3': 2, '.milo': 1}.get(ext, 0)
    return s


def find_milo(root, name, *folders):
    index = milo_index(root)
    paths = index.get(name.lower())
    if not paths and time.time() - _INDEX.get(root, (0,))[0] > 30:
        paths = milo_index(root, rebuild=True).get(name.lower())
    if not paths:
        return None
    return max(paths, key=lambda p: _score(p, folders))


def find_drum(root, name):
    """char/main/drum/<item>_<venue>.milo - any venue will do for the meshes."""
    direct = find_milo(root, name, 'drum')
    if direct:
        return direct
    index = milo_index(root)
    picks = sorted(k for k in index if k.startswith(name.lower() + '_'))
    return find_milo(root, picks[0], 'drum') if picks else None


def find_skeleton(root):
    return find_milo(root, 'skeleton', 'char', 'main')


def find_deform(root):
    return find_milo(root, 'deform', 'char', 'main')


def find_shared(root, name):
    """char/main/shared/<name>.milo (colorpalettes, char_shared)."""
    return find_milo(root, name, 'char', 'main', 'shared')


def find_head_shaper(root, gender):
    """BandCharDesc's head_male_path / head_female_path: the head-shape milo."""
    return find_milo(root, f'head_{gender}', 'char', 'main', 'head', gender)


def dump_root(context):
    addon = context.preferences.addons.get(__package__)
    path = getattr(addon.preferences, 'rb3_extracted_dir', "") if addon else ""
    path = bpy.path.abspath(path) if path else ""
    return path.rstrip("\\/") if path else ""


# ---------------------------------------------------------------------------------------
# An item's own bones (hair / hat physics bones)
# ---------------------------------------------------------------------------------------

def read_item_bones(path):
    """({Trans name: (local 12 floats, parent name)}, has CharHair) for every Trans in an
    item milo, nested directories included. Hair and hats bring their physics bones this
    way (male_hair_crazyhawk_resource: 19 bone_hair_* bones under bone_hair, driven by its
    CharHair 'crazyhawk.hair'); every other piece's Trans entries are skeleton bones."""
    from .mesh_importer import walk_milo_entries
    with open(path, 'rb') as f:
        body = read_milo_container_body(f.read())
    bones, has_hair = {}, False
    for _depth, _p, etype, ename, s, e in walk_milo_entries(body):
        if etype == 'CharHair':
            has_hair = True
        elif etype == 'Trans':
            got = rb3p._decode_trans(body[s:e])
            if got:
                bones[ename] = (got[0], got[1])
    return bones, has_hair


def merge_item_bones(context, armature, path):
    """Adds the item's bones the armature lacks, each chained from its parent's rest pose
    as the game does (a bone_hair_* bone sits at bone_hair's rest times its own local
    transform). Returns ([added names], has CharHair)."""
    from .mesh_importer import add_file_bones
    bones, has_hair = read_item_bones(path)
    missing = {n: v for n, v in bones.items() if n not in armature.data.bones}
    if not missing:
        return [], has_hair
    # parents the file defines that the armature already has stay as they are; only
    # the missing bones (and their in-file parents) are added
    added = add_file_bones(context, armature, bones)
    names = [n for n, _p in added]
    for n in names:
        b = armature.data.bones.get(n)
        if b is not None:
            b["rb3_item_bone"] = os.path.basename(path)
            if has_hair:
                b["rb3_physics"] = True
    return names, has_hair


# ---------------------------------------------------------------------------------------
# Face bones: where the head is skinned to them
# ---------------------------------------------------------------------------------------

def head_bind_worlds(path):
    """{bone name: Blender world matrix} where the head milo's meshes are bound to each bone
    (the inverse of their bone-table matrices). head.mesh wins where meshes disagree."""
    from .mesh_importer import parse_rb3_meshes
    from .io import _milo_to_blender_matrix
    _dir, meshes, _failed = parse_rb3_meshes(path)
    meshes = [m for m in meshes if not m['is_lod'] and not m['is_shadow']]
    meshes.sort(key=lambda m: m['name'] != 'head.mesh')
    out = {}
    for m in meshes:
        for bn, bm in zip(m.get('bone_names') or (), m.get('bone_mats') or ()):
            if bn not in out:
                try:
                    out[bn] = _milo_to_blender_matrix(bm).inverted()
                except Exception:
                    pass
    return out


def conform_face_bones(context, armature, path, tolerance=0.01):
    """Moves the face bones (everything below bone_head that the head milo's meshes are
    skinned to) to where the head is bound to them; their children keep their offsets.

    The shared skeleton carries ONE face layout. Each head is skinned to its own: the
    male head's eyes, lids and tongue sit about 0.5 units from the skeleton's, and the
    female head's whole face is narrower and lower - the female rest pose (moved onto the
    female body by conform_rest) only shifts the face bones with the head, so without this
    they float around the female face. Returns (bones moved, median distance)."""
    from mathutils import Matrix
    binds = head_bind_worlds(path)
    bones = armature.data.bones
    head = bones.get('bone_head.mesh') or bones.get('bone_head')
    if head is None or not binds:
        return 0, 0.0

    def under_head(b):
        p = b.parent
        while p is not None:
            if p == head:
                return True
            p = p.parent
        return False

    targets = {}
    for name, m in binds.items():
        b = bones.get(name)
        if b is None or b == head or not under_head(b):
            continue
        loc, rot, _sca = m.decompose()
        tm = Matrix.Translation(loc) @ rot.to_matrix().to_4x4()
        if (tm.to_translation() - b.matrix_local.to_translation()).length > tolerance or \
                rot.rotation_difference(b.matrix_local.to_quaternion()).angle > 1e-3:
            targets[name] = tm
    if not targets:
        return 0, 0.0

    prev_active = context.view_layer.objects.active
    prev_mode = armature.mode
    context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    moved = []
    try:
        ebs = armature.data.edit_bones
        old = {eb.name: eb.matrix.copy() for eb in ebs}
        new = {}

        def visit(eb):
            m = old[eb.name]
            if eb.name in targets:
                nm = targets[eb.name]
                moved.append((nm.to_translation() - m.to_translation()).length)
            elif eb.parent is not None and eb.parent.name in new:
                nm = new[eb.parent.name] @ (old[eb.parent.name].inverted() @ m)
            else:
                nm = m
            new[eb.name] = nm
            for c in eb.children:
                visit(c)

        for eb in [e for e in ebs if e.parent is None]:
            visit(eb)
        for eb in ebs:
            if (new[eb.name].to_translation() - old[eb.name].to_translation()).length > 1e-6 \
                    or eb.name in targets:
                length = eb.length
                eb.matrix = new[eb.name]
                eb.length = length
    finally:
        bpy.ops.object.mode_set(mode=prev_mode if prev_mode != 'EDIT' else 'OBJECT')
        context.view_layer.objects.active = prev_active
    moved.sort()
    return len(moved), (moved[len(moved) // 2] if moved else 0.0)


# ---------------------------------------------------------------------------------------
# Bone filtering
# ---------------------------------------------------------------------------------------

BONE_FILTERS = (
    ('FULL', "Full Skeleton", "Keep every bone of the shared RB3 skeleton (character "
     "meshes only)"),
    ('CHARACTER', "Character Only", "Character meshes only; delete bones none of them use"),
    ('GUITAR', "Character w/ Guitar Only", "Character and its guitar; delete bones none of "
     "them use"),
    ('DRUMS', "Character w/ Drums Only", "Character and its drum kit; delete bones none of "
     "them use"),
    ('MIC', "Character w/ Mic Only", "Character and its microphone; delete bones none of "
     "them use"),
    ('KEYBOARD', "Character w/ Keyboard Only", "Character and its keyboard; delete bones "
     "none of them use"),
)
FILTER_INSTRUMENT = {'GUITAR': 'guitar', 'DRUMS': 'drum', 'MIC': 'mic', 'KEYBOARD': 'keyboard'}


def _stem(n):
    return n[:-5] if n.endswith('.mesh') else n


def prune_bones(context, arm_obj, mesh_objs, helper_stems):
    """Deletes bones no imported mesh uses. Kept: every bone a mesh has weight on or hangs
    from, every helper bone (exo_*, every bone the deform clip drives - the proportions need
    them), and every ancestor of a kept bone. Returns the number removed."""
    used = set()
    for obj in mesh_objs:
        if obj.type != 'MESH':
            continue
        if obj.parent is arm_obj and obj.parent_type == 'BONE' and obj.parent_bone:
            used.add(obj.parent_bone)
        idx = {vg.index: vg.name for vg in obj.vertex_groups}
        live = set()
        for v in obj.data.vertices:
            for g in v.groups:
                if g.weight > 1e-6:
                    live.add(g.group)
        used.update(idx[i] for i in live if i in idx)
    bones = arm_obj.data.bones
    keep = set()
    for b in bones:
        st = _stem(b.name)
        if b.name in used or st.startswith('exo_') or st in helper_stems:
            keep.add(b.name)
    for name in list(keep):
        b = bones.get(name)
        while b is not None and b.parent is not None:
            b = b.parent
            keep.add(b.name)
    remove = [b.name for b in bones if b.name not in keep]
    if not remove:
        return 0
    prev_active = context.view_layer.objects.active
    context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')
    try:
        ebs = arm_obj.data.edit_bones
        for name in remove:
            eb = ebs.get(name)
            if eb is not None:
                ebs.remove(eb)
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')
        context.view_layer.objects.active = prev_active
    return len(remove)


# ---------------------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------------------

_ENUM_ITEMS = []      # kept alive for Blender's dynamic enum


def _prefab_items(self, context):
    global _ENUM_ITEMS
    path = bpy.path.abspath(self.filepath) if self.filepath else ""
    items = []
    if path and os.path.isfile(path):
        chars = _cached(_PREFAB_CACHE, path, load_characters) or {}
        root = dump_root(context)
        names = display_names(root, bpy.path.abspath(self.locale_file)
                              if self.locale_file else "") if root else {}
        tags = {'prefab': "", 'creator': "  [creator preset]", 'debug': "  [debug]",
                'save': ""}
        order = {'save': 0, 'prefab': 0, 'creator': 1, 'debug': 2}
        for key in sorted(chars, key=lambda k: (order[chars[k]['group']],
                                                int(k[4:]) if chars[k]['group'] == 'save'
                                                else 0, k)):
            c = chars[key]
            if c['group'] == 'debug' and not self.show_debug:
                continue
            if c['group'] == 'save':
                label = c['save_name']
            else:
                disp = display_name(names, key)
                label = (f"{disp} ({key})" if disp != key else key) + tags[c['group']]
            items.append((key, label, f"{c['gender']}, height {c['height']:g}, weight "
                                      f"{c['weight']:g}, muscle {c['muscle']:g}"))
    if not items:
        items = [('NONE', "(select prefabs.milo or a save)", "")]
    _ENUM_ITEMS = items
    return items


class IMPORT_OT_rb3_character(bpy.types.Operator, ImportHelper):
    """(Experimental) Import a Rock Band 3 character from prefabs.milo or a save (Xbox 360
    package / save.dat, decrypted PS3 SAVE.DAT) and the Xbox 360 dump set in the add-on's
    preferences: the armature, every outfit (and optionally instrument)
    mesh with vertex groups and material slots, and the body proportions"""
    bl_idname = "import_scene.rb3_character"
    bl_label = "Import RB3 Character (Experimental)"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ""
    # '*': Xbox 360 save packages ('band3') have no extension
    filter_glob: StringProperty(default="*", options={'HIDDEN'})

    prefab: EnumProperty(name="Character", items=_prefab_items,
                         description="The character to import: a prefab from the selected prefabs.milo, "
                                     "or one the players made, from a save")
    show_debug: BoolProperty(name="Show Debug Prefabs", default=False,
                             description="Also list the debug characters prefabs.milo keeps")
    bone_filter: EnumProperty(
        name="Bones", items=BONE_FILTERS, default='FULL',
        description="Which meshes to import, and whether to delete the bones they don't use "
                    "afterwards. Helper bones the proportions need are always kept")
    apply_proportions: BoolProperty(
        name="Apply Proportions", default=True,
        description="Pose the armature for the character's height, weight and muscle from "
                    "char/main/deform. Adjustable afterwards in the armature's Rock Band 3 "
                    "Proportions panel")
    apply_face: BoolProperty(
        name="Apply Face", default=True,
        description="Sculpt the head with the character's face types and sliders from the "
                    "head-shape milo (head_male / head_female). Adjustable afterwards in "
                    "the armature's Rock Band 3 Face panel")
    load_patches: BoolProperty(
        name="Tattoos and Art", default=True,
        description="Project the character's tattoos and art (the prefab's prefab_art "
                    "textures, or the art made in the game's art maker, from a save) onto "
                    "the skin, clothes and instruments that take them")
    load_makeup: BoolProperty(
        name="Makeup", default=True,
        description="Draw the character's eye and lip makeup (the makeup.tex designs in "
                    "colorpalettes) onto the head skin texture")
    load_textures: BoolProperty(
        name="Load Textures and Colours", default=True,
        description="Dress every part with its textures, tinted with the character's colours "
                    "(the creator's palettes), the skin tone and eye colour, the body-type "
                    "skin normal maps and the face-type head normal map")
    use_rb3_shader: BoolProperty(
        name="Use RB3 Shader", default=True,
        description="Turn the Rock Band 3 shader on for the imported materials (they're "
                    "placeholder slots for now - textures come later)")
    locale_file: StringProperty(
        name="Locale File", subtype='FILE_PATH',
        description="Optional locale .dta with the characters' display names. Default: "
                    "every .dta in ui/locale/eng of the dump")

    def draw(self, context):
        layout = self.layout
        root = dump_root(context)
        box = layout.box()
        box.label(text="Experimental", icon='EXPERIMENTAL')
        if not root or not os.path.isdir(root):
            box.label(text="Set the Rock Band 3 Xbox 360 dump", icon='ERROR')
            box.label(text="folder in the add-on's preferences")
        else:
            box.label(text=os.path.basename(root) or root, icon='FILE_FOLDER')
        path = bpy.path.abspath(self.filepath) if self.filepath else ""
        save = bool(path) and os.path.isfile(path) and is_rb3_save(path)
        if save:
            layout.label(text="Characters in this save:", icon='FILE')
        layout.prop(self, "prefab")
        if not save:
            layout.prop(self, "show_debug")
        layout.prop(self, "bone_filter")
        layout.prop(self, "apply_proportions")
        layout.prop(self, "apply_face")
        layout.prop(self, "load_textures")
        sub = layout.column()
        sub.enabled = self.load_textures
        sub.prop(self, "load_makeup")
        sub.prop(self, "load_patches")
        layout.prop(self, "use_rb3_shader")
        layout.prop(self, "locale_file")
        if self.prefab and self.prefab != 'NONE':
            c = (_cached(_PREFAB_CACHE, path, load_characters) or {}).get(self.prefab)
            if c:
                col = layout.box().column(align=True)
                col.label(text=f"{c['gender']} - height {c['height']:g}, weight "
                               f"{c['weight']:g}, muscle {c['muscle']:g}")
                g, suffix = rb3p.body_type(c['gender'], c['height'], c['weight'], c['muscle'])
                col.label(text=f"Body type: {rb3p.GROUP_LABELS[g]}")
                for k in OUTFIT_PIECES:
                    if c['outfit'][k]['name']:
                        col.label(text=f"{k}: {c['outfit'][k]['name']}")

    def _art_refs(self, root, key, c, path, warnings):
        """{texture index: art reference} for the character's patches: a prefab's
        prefab_artNN.tex (char/main/prefab/<prefab>.milo), or a save's art slots, kept as
        packed images."""
        used = sorted({p['texture'] for p in c.get('patches') or ()
                       if p.get('texture', -1) >= 0})
        if not used:
            return {}
        refs = {}
        if c.get('group') == 'save':
            base = os.path.splitext(os.path.basename(path))[0]
            for k in used:
                rgba = (c.get('art') or {}).get(k)
                if rgba is None:
                    warnings.append(f"art slot {k} is empty in the save")
                    continue
                name = f"RB3 art {base} {k}"
                rb3_textures.image_from_rgba(name, rgba, 'sRGB')
                rb3_textures.register_art(name, rgba)
                refs[k] = {'image': name}
            return refs
        pmilo = find_milo(root, key, 'prefab')
        if pmilo is None:
            warnings.append(f"char/main/prefab/{key}.milo not found - no tattoos / art")
            return {}
        for k in used:
            refs[k] = {'tex': f"prefab_art{k:02d}.tex", 'lib': [pmilo]}
        return refs

    def _import_meshes(self, context, armature, milo, game_rest=True):
        before = set(bpy.data.objects)
        _activate(context, armature)
        try:
            # The item's own bones are merged beforehand (merge_item_bones, nested-aware);
            # game_rest: place the meshes through their bone tables against the armature,
            # as the game skins them - crazyhawk's hair is bound with bone_hair 1.7 units
            # to the side and is drawn 1.7 units off-centre otherwise. Not for the head
            # (see IMPORT_OT_rb3_meshes: its face bones are the head's own).
            result = bpy.ops.import_scene.rb3_meshes(
                filepath=milo, exclude_lod=True, exclude_shadow=True, exclude_helpers=True,
                import_file_bones=False, parent_to_armature=armature is not None,
                rest_from_armature=game_rest and armature is not None)
        except Exception as e:
            _log(f"    mesh import failed: {e}")
            result = {'CANCELLED'}
        new = [o for o in bpy.data.objects if o not in before]
        # instrument patch placement quads (patch.mat) are art-maker helpers, and an item's
        # own '<mesh>_patch.mesh' (<item>_patch.mat) is a render-to mesh drawn into its
        # texture (rb3_textures.apply_patches) - neither is geometry
        for o in [o for o in new if o.type == 'MESH'
                  and (o.get("milo_material") or '').endswith('patch.mat')]:
            new.remove(o)
            bpy.data.objects.remove(o, do_unlink=True)
        return ('FINISHED' in result), new

    def execute(self, context):
        root = dump_root(context)
        if not root or not os.path.isdir(root):
            self.report({'ERROR'}, "Set 'Rock Band 3 Xbox 360 Dump' in the Milo Toolset "
                                   "add-on preferences first")
            return {'CANCELLED'}
        path = bpy.path.abspath(self.filepath)
        save = is_rb3_save(path)
        if save:
            try:
                chars = parse_rb3_save(path)
            except (RB3SaveError, OSError) as e:
                self.report({'ERROR'}, f"Couldn't read the save: {e}")
                return {'CANCELLED'}
            _log(f"RB3 save: {len(chars)} character(s): "
                 f"{', '.join(c['save_name'] for c in chars.values())}")
        else:
            chars = _cached(_PREFAB_CACHE, path, parse_prefabs_milo)
        if not chars:
            self.report({'ERROR'}, "No characters (BandCharDesc) found - is this RB3's "
                                   "char/main/prefabs.milo?")
            return {'CANCELLED'}
        key = self.prefab
        if key not in chars:
            self.report({'ERROR'}, "Pick a character")
            return {'CANCELLED'}
        c = chars[key]
        gender = c['gender']
        names = display_names(root, bpy.path.abspath(self.locale_file)
                              if self.locale_file else "")
        title = c.get('save_name') or display_name(names, key)
        _log(f"===== RB3 character (experimental): {key} '{title}' ({gender}) =====")
        warnings = []

        skel = find_skeleton(root)
        if skel is None:
            self.report({'ERROR'}, "char/main/skeleton.milo not found in the dump")
            return {'CANCELLED'}
        deform_path = find_deform(root) if self.apply_proportions else None
        clips = hier = None
        if self.apply_proportions:
            if deform_path is None:
                warnings.append("char/main/deform.milo not found - no proportions")
            else:
                try:
                    clips = rb3p.read_deform_milo(deform_path)
                    hier = rb3p.hierarchy_from_skeleton(skel)
                except Exception as e:
                    warnings.append(f"couldn't read the deform milo: {e}")
                    clips = None

        coll = _new_collection(context, title)
        coll["rb3_prefab"] = key
        if c.get('save_name'):
            coll["rb3_save_character"] = c['save_name']
            coll["rb3_save_file"] = path
        for k in ('gender', 'height', 'weight', 'muscle', 'skin_color'):
            coll[f"rb3_{k}"] = c[k]

        # Armature: the shared skeleton
        _activate(context, None)
        try:
            bpy.ops.import_scene.rb3_skeleton(filepath=skel)
        except Exception as e:
            self.report({'ERROR'}, f"Couldn't import {skel}: {e}")
            return {'CANCELLED'}
        armature = context.view_layer.objects.active
        if armature is None or armature.type != 'ARMATURE':
            self.report({'ERROR'}, f"No armature from {skel}")
            return {'CANCELLED'}
        armature.name = title
        armature.data.name = title
        if armature.name not in coll.objects:
            coll.objects.link(armature)
        for other in list(armature.users_collection):
            if other is not coll:
                other.objects.unlink(armature)
        armature.rb3_proportions.gender = gender
        _log(f"  armature: {os.path.relpath(skel, root)} ({len(armature.data.bones)} bones)")
        if clips and gender == 'female' and 'male' in clips and 'female' in clips:
            try:
                n, med = rb3p.conform_rest(context, armature, hier, clips['female'],
                                           clips['male'])
                _log(f"  armature: rest pose moved onto the female neutral body ({n} bones, "
                     f"median {med:.2f})")
            except Exception as e:
                warnings.append(f"couldn't move the rest pose to the female body: {e}")

        # Parts
        parts = []
        if not c['head']['hide']:
            parts.append(('head', 'head', find_milo(root, 'head', 'head', gender)))
        for k in OUTFIT_PIECES:
            name = c['outfit'][k]['name']
            if name:
                parts.append((k, name, find_milo(root, name, k, gender)))
        inst = FILTER_INSTRUMENT.get(self.bone_filter)
        if inst:
            name = c['instruments'][inst]['name']
            if name:
                milo = find_drum(root, name) if inst == 'drum' else find_milo(root, name, inst)
                parts.append((inst, name, milo))
            else:
                warnings.append(f"this character has no {inst}")

        # Textures: the shared milos, the head milo (skin.cfg, eyes) and the skin settings
        tex_ctx = None
        if self.load_textures:
            pal = find_shared(root, 'colorpalettes')
            shared = find_shared(root, 'char_shared')
            head_path = next((m for cat, _n, m in parts if cat == 'head' and m), None) \
                or find_milo(root, 'head', 'head', gender)
            if pal is None:
                warnings.append("char/main/shared/colorpalettes.milo not found - no "
                                "palettes or skin textures")
            skin_cfg = None
            if head_path:
                for cfg in rb3_textures.item_configs(head_path):
                    if cfg['swaps'] and cfg['swaps'][0]['pal1'] == 'skin.pal':
                        skin_cfg = cfg
            _g, suffix = rb3p.body_type(gender, c['height'], c['weight'], c['muscle'])
            hd = c['head']
            head_types = {f: hd.get(f, 0) for f in rb3_textures.HEAD_FEATURES}
            # char_shared before colorpalettes: both have a head_naked.mat, and the
            # character's is char_shared's (skin shader variation, the warm skin colour)
            art = self._art_refs(root, key, c, path, warnings) if self.load_patches else {}
            tex_ctx = dict(shared=[head_path, shared, pal], skin=dict(
                skin_color=c['skin_color'], normal_suffix=suffix, head_types=head_types,
                skin_cfg=skin_cfg,
                patches=c.get('patches') if self.load_patches else [], art=art,
                makeup=rb3_textures.makeup_meshes(c.get('patches'))
                if self.load_makeup else [],
                head_controllers=rb3_textures.head_controllers(head_path, gender)))
            armature["rb3_norm_suffix"] = suffix
            import json as _json
            armature["rb3_head_norm_types"] = _json.dumps(head_types, sort_keys=True)
        dressed = set()

        imported = []
        n_md = 0
        for category, name, milo in parts:
            if milo is None:
                warnings.append(f"{category}: '{name}' not found in the dump")
                _log(f"  {category}: '{name}' NOT FOUND")
                continue
            _log(f"  {category}: {name} -> {os.path.relpath(milo, root)}")
            try:
                added, has_hair = merge_item_bones(context, armature, milo)
                if added:
                    _log(f"    merged {len(added)} bone(s) from the item"
                         + (" (physics bones of its CharHair)" if has_hair else "") + ": "
                         + ", ".join(_stem(n) for n in added[:8])
                         + (" ..." if len(added) > 8 else ""))
            except Exception as e:
                warnings.append(f"{category}: couldn't merge the item's bones ({e})")
            ok, new = self._import_meshes(context, armature, milo,
                                          game_rest=category != 'head')
            if not ok or not new:
                warnings.append(f"{category}: no meshes imported from '{name}'")
                continue
            for o in new:
                o["rb3_part"] = category
            imported += new
            if category == 'head':
                try:
                    n, med = conform_face_bones(context, armature, milo)
                    if n:
                        _log(f"    face bones: {n} moved to where the head is skinned to "
                             f"them (median {med:.2f})")
                except Exception as e:
                    warnings.append(f"head: couldn't place the face bones ({e})")
            if tex_ctx is not None:
                if category == 'head':
                    piece_colors = [c['head']['eye_color'], 0, 0]
                elif category in INSTRUMENTS:
                    piece_colors = list(c['instruments'][category]['colors'])
                else:
                    piece_colors = list(c['outfit'][category]['colors'])
                lib = rb3_textures.Library([milo] + tex_ctx['shared'])
                cfgs = rb3_textures.item_configs(milo)
                prefix = "" if category == 'head' else f"{category}_"
                for o in new:
                    for slot in (o.material_slots if o.type == 'MESH' else ()):
                        m = slot.material
                        if m is None or m.name in dressed:
                            continue
                        dressed.add(m.name)
                        try:
                            note = rb3_textures.dress_material(
                                m, lib, cfgs, piece_colors, gender, skin=tex_ctx['skin'],
                                output_prefix=prefix)
                            _log(f"    material {note}")
                        except Exception as e:
                            warnings.append(f"{category}: '{m.name}' not textured ({e})")
            try:
                got = rb3p.attach_meshdeforms(new, milo)
                n_md += len(got)
            except Exception as e:
                _log(f"    girth data not read: {e}")

        # Face sculpting: the head's types and sliders (rb3_face)
        if self.apply_face and any(o.get("rb3_part") == 'head' for o in imported):
            shaper = find_head_shaper(root, gender)
            if shaper is None:
                warnings.append(f"head_{gender}.milo (the head-shape milo) not found - "
                                f"face not sculpted")
            else:
                try:
                    _log(f"  face: {os.path.relpath(shaper, root)}")
                    if not rb3_face.set_from_head(armature, shaper, c['head']):
                        warnings.append("face: no head mesh matches the head-shape milo's "
                                        "vertex count")
                except Exception as e:
                    warnings.append(f"face sculpting failed: {e}")

        # Material slots: placeholders from the mesh importer; RB3 shader on request
        if self.use_rb3_shader:
            mats = {s.material for o in imported if o.type == 'MESH'
                    for s in o.material_slots if s.material}
            for m in mats:
                st = getattr(m, 'gltfmilo_settings', None)
                if st is None:
                    continue
                try:
                    st.rb3_shader = True
                except Exception as e:
                    _log(f"  RB3 shader not set on '{m.name}': {e}")

        # Proportions
        if clips and gender in clips:
            try:
                rb3p.store_deform(armature, gender, clips[gender], hier, deform_path, skel)
                pg = armature.rb3_proportions
                pg["height"], pg["weight"], pg["muscle"] = c['height'], c['weight'], c['muscle']
                placed, missing = rb3p.apply_proportions(armature, report=True)
            except Exception as e:
                warnings.append(f"proportions failed: {e}")
        elif self.apply_proportions and clips:
            warnings.append(f"the deform milo has no '{gender}' clip")

        # Bones
        if self.bone_filter != 'FULL':
            helpers = set()
            if clips and gender in clips:
                for blk in (clips[gender]['full'], clips[gender].get('one')):
                    if blk:
                        helpers.update(n.rsplit('.', 1)[0] for n in blk['names'])
            try:
                n = prune_bones(context, armature, imported, helpers)
                label = next(i[1] for i in BONE_FILTERS if i[0] == self.bone_filter)
                _log(f"  bones: removed {n} unused, {len(armature.data.bones)} left ({label})")
                if clips and gender in clips:
                    rb3p.apply_proportions(armature)
            except Exception as e:
                warnings.append(f"bone filter failed: {e}")

        _activate(context, armature)
        summary = (f"RB3 character '{title}': {len(imported)} mesh(es), "
                   f"{len(armature.data.bones)} bones, girth data on {n_md} mesh(es)")
        for w in warnings:
            _log(f"  WARNING: {w}")
        _log(f"===== {summary} =====")
        self.report({'WARNING'} if warnings else {'INFO'},
                    summary + (f"; {len(warnings)} warning(s) - see the log"
                               if warnings else ""))
        return {'FINISHED'}


_classes = (IMPORT_OT_rb3_character,)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
