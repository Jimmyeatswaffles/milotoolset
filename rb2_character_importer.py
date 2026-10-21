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

Saves work too - Xbox 360 (a CON package such as 'band', or its extracted save.dat) and
PS3 (RB2.SAV as RPCS3 keeps it, which has no console encryption): the characters players
made are read from it and picked by the names they were given. Both platforms store the
same thing - the same Rand2 encryption, little-endian on PS3 as well, the same character
records - so one reader serves both; only the PS3 header after the seed differs, which
doesn't matter as records are found by their structure. RB2.SAV is a fixed 6 MB of which
~200 KB is data, so the zero padding is trimmed before decrypting.

    CON (STFS)   save.dat is one file of the package; its blocks are found through the
                 package's hash tables (two alternating copies - both are tried).
    save.dat     u32 version (14 outside, 15 inside), then a random 32-bit seed and the rest
                 XORed with the low byte of the engine's Rand2 generator (Park-Miller) from
                 that seed: BinStream's own encryption (RB3 decomp: utl/BinStream.cpp,
                 math/Rand2.cpp), little-endian.
    characters   after a u32 count, each record: u32 0x24, the name (u32 length + text), u32
                 1, a 16-byte id, a byte, then the character as a binary DataArray - u16
                 count, 4 bytes, then nodes (u32 type: 0 int, 1 float, 5 symbol, 0x12
                 string, 0x10 array) - exactly a prefab entry with the creator preset it
                 started from in place of a prefab name:
                     (contestant_number_one (gender female) (skin_color_index 0) ...
                      (outfit (hair ladylayered_resource (colors 1 0)) ...))
                 Each character's progress follows its record, so records are found by
                 their structure rather than at fixed offsets.
    colours      The (colors a b) in that array are NOT the ones in use: they are the
                 starting preset's, left as they were when items were swapped (Sylvain,
                 from contestant_number_one, keeps (4 4) and (26 11) from its bleached tee
                 and work pants). After the array each character has a closet: per item
                 used, its slot, its name, u32 2, two u32 palette indices and u32 -1 -
                 loosedapperjacket_vintage 26 0 = the slate jacket and black tie Sylvain
                 wears, bootcutpants_distressedjeans 18 0 = red-orange denim[18] and a black
                 belt. Those win; the array's numbers are only a fallback.
"""

import os
import re
import struct
import time

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty
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


# ---------------------------------------------------------------------------------------
# Xbox 360 saves
# ---------------------------------------------------------------------------------------

class SaveError(Exception):
    pass


def _stfs_files(data):
    """{name: [bytes, ...]} for the files of an STFS package (CON / LIVE / PIRS) - one
    candidate per hash-table copy whose block chain is complete."""
    if data[:4] not in (b'CON ', b'LIVE', b'PIRS'):
        raise SaveError("not an Xbox 360 package")
    header_size = struct.unpack_from('>I', data, 0x340)[0]
    first = (header_size + 0xFFF) & 0xFFFFF000
    sep = data[0x37B]
    sex = 0 if sep & 1 else 1                  # two copies of each hash table unless set
    ft_blocks = struct.unpack_from('<H', data, 0x37C)[0]
    ft_start = int.from_bytes(data[0x37E:0x381], 'little')

    def backing(b):
        n = (((b + 0xAA) // 0xAA) << sex) + b
        if b < 0xAA:
            return n
        if b < 0x70E4:
            return n + (((b + 0x70E4) // 0x70E4) << sex)
        return (1 << sex) + n + (((b + 0x70E4) // 0x70E4) << sex)

    def off(b):
        return first + (backing(b) << 12)

    step0 = 0xAC if sex else 0xAB

    def hash_entry(b, copy):
        if b < 0xAA:
            hb = 0
        else:
            hb = (b // 0xAA) * step0
            hb += ((b // 0x70E4) + 1) << sex
            if b // 0x70E4:
                hb += 1 << sex
        o = first + ((hb + copy) << 12) + (b % 0xAA) * 0x18
        return data[o:o + 0x18]

    def chain(start, count, copy, consecutive):
        out, b = [], start
        for _ in range(count):
            if b >= 0xFFFFFF or off(b) + 0x1000 > len(data):
                return None
            out.append(b)
            b = b + 1 if consecutive else int.from_bytes(hash_entry(b, copy)[0x15:0x18], 'big')
        return out

    tables = []
    for copy in range(1 << sex):
        blocks = chain(ft_start, max(ft_blocks, 1), copy, False)
        if blocks:
            tables.append(blocks)
    files = {}
    seen = set()
    for blocks in tables:
        for b in blocks:
            o = off(b)
            for k in range(0x40):
                e = data[o + k * 0x40:o + k * 0x40 + 0x40]
                if not e[0]:
                    continue
                ln = e[0x28] & 0x3F
                if e[0x28] & 0x80 or not 0 < ln <= 0x28:
                    continue
                name = e[:ln].decode('latin-1', 'replace')
                nblocks = int.from_bytes(e[0x29:0x2C], 'little')
                start = int.from_bytes(e[0x2F:0x32], 'little')
                size = struct.unpack_from('>I', e, 0x34)[0]
                for copy in range(1 << sex):
                    key = (name, start, size, copy)
                    if key in seen:
                        continue
                    seen.add(key)
                    bl = chain(start, nblocks, copy, bool(e[0x28] & 0x40))
                    if bl:
                        body = b''.join(data[off(x):off(x) + 0x1000] for x in bl)[:size]
                        files.setdefault(name, []).append(body)
    return files


def _rand2_xor(body, seed):
    s = 1 if seed == 0 else abs(seed)
    out = bytearray(body)
    for i in range(len(out)):
        t = (s % 127773) * 0x41A7 - (s // 127773) * 0xB14
        s = t if t > 0 else t + 0x7FFFFFFF
        out[i] ^= s & 0xFF
    return bytes(out)


def _decrypt_save(raw):
    if len(raw) < 16:
        raise SaveError("save too small")
    seed = struct.unpack_from('<i', raw, 4)[0]
    return _rand2_xor(raw[8:], seed)


_DTB_TEXT = {2, 3, 4, 5, 7, 0x12, 0x20, 0x21, 0x22, 0x23, 0x25}
_DTB_TREE = {0x10, 0x11, 0x13}


def _dtb_node(b, o, depth=0):
    if depth > 64:
        raise SaveError("data nested too deep")
    t = struct.unpack_from('<I', b, o)[0]; o += 4
    if t == 0:
        return struct.unpack_from('<i', b, o)[0], o + 4
    if t == 1:
        return struct.unpack_from('<f', b, o)[0], o + 4
    if t in _DTB_TEXT:
        n = struct.unpack_from('<I', b, o)[0]
        if n > 4096 or o + 4 + n > len(b):
            raise SaveError("bad text length")
        text = b[o + 4:o + 4 + n].decode('utf-8', 'replace')
        return (DtaString(text) if t == 0x12 else text), o + 4 + n
    if t in _DTB_TREE:
        return _dtb_array(b, o, depth + 1)
    if t in (6, 8, 9, 0x24):
        return None, o + 4
    raise SaveError(f"unknown data type {t:#x}")


def _dtb_array(b, o, depth=0):
    n = struct.unpack_from('<H', b, o)[0]
    if n > 4096:
        raise SaveError("bad array count")
    o += 6
    out = []
    for _ in range(n):
        v, o = _dtb_node(b, o, depth)
        out.append(v)
    return out, o


def _closet_colours(region, items):
    """{item: [colour, colour]} from a character's closet: the item's name (u32 length +
    text), then u32 2, the colours, and u32 -1. The last entry for an item wins."""
    out = {}
    for item in items:
        raw = item.encode('latin-1', 'replace')
        tag = struct.pack('<I', len(raw)) + raw
        o = 0
        while True:
            o = region.find(tag, o)
            if o < 0:
                break
            q = o + len(tag)
            try:
                n = struct.unpack_from('<I', region, q)[0]
                if 1 <= n <= 3:
                    cols = list(struct.unpack_from(f'<{n}I', region, q + 4))
                    tail = struct.unpack_from('<i', region, q + 4 + 4 * n)[0]
                    if all(c < 256 for c in cols) and tail == -1:
                        out[item] = cols
            except struct.error:
                pass
            o += 1
    return out


def _save_characters(plain):
    """[(name, entry list, {item: closet colours})] from a decrypted save: records of u32
    0x24, the name, u32 1, a 16-byte id, a byte, then the character's DataArray (preset,
    (gender ...), ...), followed by that character's progress and closet."""
    found, o = [], 0
    marker = struct.pack('<I', 0x24)
    while True:
        o = plain.find(marker, o)
        if o < 0 or o + 8 > len(plain):
            break
        try:
            ln = struct.unpack_from('<I', plain, o + 4)[0]
            if not 1 <= ln <= 64:
                raise SaveError("name")
            raw_name = plain[o + 8:o + 8 + ln]
            q = o + 8 + ln
            if struct.unpack_from('<I', plain, q)[0] != 1:
                raise SaveError("record")
            arr, end = _dtb_array(plain, q + 4 + 16 + 1)
            if len(arr) < 3 or not isinstance(arr[0], str) \
                    or not (isinstance(arr[1], list) and arr[1][:1] == ['gender']):
                raise SaveError("not a character")
            found.append([raw_name.decode('utf-8', 'replace').strip(), arr, o, end])
            o = end
            continue
        except (SaveError, struct.error, UnicodeDecodeError, IndexError):
            pass
        o += 1
    out = []
    for i, (name, arr, _start, end) in enumerate(found):
        nxt = found[i + 1][2] if i + 1 < len(found) else len(plain)
        items = []
        for field in arr[1:]:
            if isinstance(field, list) and field[:1] == ['outfit']:
                items = [p[1] for p in field[1:] if isinstance(p, list) and len(p) > 1
                         and isinstance(p[1], str) and p[1] != 'none']
        out.append((name, arr, _closet_colours(plain[end:nxt], items)))
    return out


_SAVE_EXTS = ('.dat', '.sav')
SAVE_VERSIONS = range(1, 64)          # the plain u32 version before the seed (14 seen)


def _trim(data):
    """Without the trailing zero padding (RB2.SAV is a fixed 6 MB, ~200 KB of data)."""
    end = len(data.rstrip(b'\x00'))
    return data[:min(len(data), end + 16)]


def _looks_like_save(data):
    """A plain save: a small version number, a seed, then data that decrypts into at least
    one character record within its first 64 KB."""
    if len(data) < 64 or struct.unpack_from('<I', data, 0)[0] not in SAVE_VERSIONS:
        return False
    try:
        return bool(_save_characters(_decrypt_save(_trim(data[:65536]))))
    except (SaveError, struct.error):
        return False


def _encrypted_hint(data):
    """Why a save that isn't a CON gave nothing: most likely a PS3 save still under the
    console's encryption (its own start isn't the game's version number)."""
    if struct.unpack_from('<I', data, 0)[0] not in SAVE_VERSIONS:
        return ("this looks like a PS3 save still under the console's encryption - decrypt "
                "it first with a PS3 save tool, or use the copy RPCS3 keeps (its RB2.SAV is "
                "not encrypted)")
    return "no characters found - is this a Rock Band 2 save?"


def parse_save(path):
    """{character name: character dict} from an RB2 save: an Xbox 360 CON package, its
    extracted save.dat, or a PS3 RB2.SAV (RPCS3)."""
    with open(path, 'rb') as f:
        data = f.read()
    candidates = []
    if data[:4] in (b'CON ', b'LIVE', b'PIRS'):
        files = _stfs_files(data)
        candidates = files.get('save.dat', [])
        if not candidates:
            raise SaveError("no save.dat in this package")
    else:
        if len(data) < 64:
            raise SaveError("file too small to be a save")
        candidates = [_trim(data)]
    best = []
    for raw in candidates:
        try:
            chars = _save_characters(_decrypt_save(raw))
        except (SaveError, struct.error):
            continue
        if len(chars) > len(best):
            best = chars
    if not best:
        if data[:4] in (b'CON ', b'LIVE', b'PIRS'):
            raise SaveError("no characters found in this package's save.dat")
        raise SaveError(_encrypted_hint(data))
    out = {}
    for name, arr, closet in best:
        key = name or f"character_{len(out) + 1}"
        while key in out:
            key += "_"
        ch = _character_from_entry([key] + arr[1:], key, default_selectable=True)
        ch['preset'] = arr[0]
        ch['from_save'] = True
        fixed = []
        for category, item, colors in ch['outfit']:
            if item in closet:
                colors = list(closet[item])
            fixed.append((category, item, colors))
        ch['outfit'] = fixed
        ch['closet_items'] = sorted(closet)
        out[key] = ch
    return out


def load_characters(path):
    """Characters from prefabs.dta / characters.dta, or from a save."""
    if is_save_file(path):
        return parse_save(path)
    return parse_prefabs(path)


_SAVE_KIND_CACHE = {}


def _wrap(text, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def is_save_file(path):
    """A save rather than a DTA: a CON package, a .dat / .sav file (save.dat, RB2.SAV), or
    any other file whose contents decrypt into a character record. Text DTA files never
    qualify."""
    try:
        st = os.stat(path)
    except OSError:
        return False
    key = (path, st.st_mtime, st.st_size)
    if key in _SAVE_KIND_CACHE:
        return _SAVE_KIND_CACHE[key]
    try:
        with open(path, 'rb') as f:
            head = f.read(65536)
    except OSError:
        return False
    if head[:4] in (b'CON ', b'LIVE', b'PIRS') or path.lower().endswith(_SAVE_EXTS):
        result = True
    elif path.lower().endswith('.dta') or head.lstrip()[:1] in (b'(', b';', b'#'):
        result = False
    else:
        result = _looks_like_save(head)
    _SAVE_KIND_CACHE[key] = result
    return result


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


_REF_RE = re.compile(rb'([A-Za-z0-9_\-./]+?)\.milo')


def milo_summary(path):
    """(has top-level meshes, [resource milo basenames the directory references]).

    Kept for callers that only want the *_resource references; milo_refs gives them all."""
    has_mesh, refs = milo_refs(path)
    return has_mesh, [os.path.basename(r).lower() for r in refs
                      if r.lower().endswith('_resource')]


_SOURCE_PATH_RE = re.compile(r'(^|/)(band|prj|run|char)/', re.IGNORECASE)


def milo_refs(path):
    """(has top-level meshes, [every milo the directory references, as written - e.g.
    'puffyskirt/puffyskirt_barelegs' - except the shared colour palettes and the milo's own
    source path]).

    Self-contained milos name themselves by their source path - naked (hands) ->
    'band/run/char/female/outfits/hands/naked', head_2_chops -> '.../facehair/head_2_chops',
    strapjacket_pleather -> 'prj/band/run/char/male/outfits/torso/strapjacket_pleather'.
    That's metadata, not a link: followed, it led away from the milo to another file of the
    same name (the torso's naked instead of the hands, the male head_1 for a female)."""
    with open(path, 'rb') as f:
        body = read_milo_container_body(f.read())
    objs, metas = cc.walk_milo(body)
    top = metas[0][0]
    has_mesh = any(t == 'Mesh' and d == top for d, t, _n, _a, _z in objs)
    refs = []
    for m in _REF_RE.finditer(metas[0][1]):
        ref = m.group(1).decode('latin-1').lstrip('./')
        # the symbol's length byte can sit in front of it; keep the path from its first
        # letter or slash on
        ref = re.sub(r'^[^A-Za-z0-9_/]+', '', ref)
        if 'colorpalettes' in ref.lower() or not ref or ref in refs:
            continue
        own = os.path.splitext(os.path.basename(path))[0].lower()
        if os.path.basename(ref).lower() == own and _SOURCE_PATH_RE.search(ref):
            continue                                  # its own source path
        refs.append(ref)
    return has_mesh, refs


def find_milo_ref(root, ref, category, gender, exclude=None):
    """The file a directory reference names. 'puffyskirt/puffyskirt_barelegs' shares its
    name with the variant making the reference, so the copy inside a 'puffyskirt' folder is
    preferred, and the referencing file itself never counts."""
    base = os.path.basename(ref).lower()
    paths = [p for p in milo_index(root).get(base, ())
             if not exclude or os.path.normcase(os.path.abspath(p))
             != os.path.normcase(os.path.abspath(exclude))]
    if not paths:
        return None
    folders = [x.lower() for x in re.split(r'[\\/]', ref)[:-1] if x and x != '..']

    def score(p):
        parts = [x.lower() for x in os.path.normpath(p).split(os.sep)]
        s = 16 * sum(1 for f in folders if f in parts)
        if gender and gender.lower() in parts:
            s += 8
        if category and category.lower() in parts:
            s += 4
        if 'gen' in parts:
            s += 1
        return s
    return max(paths, key=score)


def resolve_part_milos(root, milo, category, gender, item, depth=3):
    """(milo holding the meshes or None, [every milo met on the way, outermost first]).

    Usually a texture variant names its *_resource milo. Some go through another milo -
    puffyskirt_barelegs names 'puffyskirt/puffyskirt_barelegs', a second milo of the same
    name in a puffyskirt folder - so references are followed (up to `depth` steps) until a
    milo with meshes turns up. As a last resort the naming convention is tried:
    puffyskirt_barelegs -> puffyskirt_resource."""
    chain, todo, seen = [], [(milo, 0)], set()
    mesh_src = None
    while todo:
        path, d = todo.pop(0)
        key = os.path.normcase(os.path.abspath(path))
        if key in seen:
            continue
        seen.add(key)
        chain.append(path)
        try:
            has_mesh, refs = milo_refs(path)
        except Exception:
            continue
        # A resource reference beats the milo's own meshes (a variant's own meshes are
        # patches, fur and the like, handled separately).
        res = [r for r in refs if r.lower().endswith('_resource')]
        other = [r for r in refs if r not in res]
        for r in res:
            p = find_milo_ref(root, r, category, gender, exclude=path)
            if p:
                if mesh_src is None:
                    mesh_src = p
                if os.path.normcase(os.path.abspath(p)) not in seen:
                    chain.append(p)
                    seen.add(os.path.normcase(os.path.abspath(p)))
        if mesh_src is None and has_mesh:
            mesh_src = path                       # a milo's own meshes beat other refs
        if mesh_src is None and d < depth:
            for r in other:
                p = find_milo_ref(root, r, category, gender, exclude=path)
                if p:
                    todo.append((p, d + 1))
    if mesh_src is None:
        try:
            has_mesh, _r = milo_refs(milo)
        except Exception:
            has_mesh = False
        if has_mesh:
            mesh_src = milo
    if mesh_src is None and '_' in item:
        guess = find_milo(root, item.split('_')[0] + '_resource', category, gender)
        if guess:
            mesh_src = guess
            if guess not in chain:
                chain.append(guess)
    return mesh_src, chain


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


# ---------------------------------------------------------------------------------------
# Master armatures and retargeting
# ---------------------------------------------------------------------------------------
#
# char/<gender>/gen holds the game's complete character rigs - <gender>_vocal (and the
# guitar / drum versions): ~200 bones with face bones. Their body bones are exactly where
# the outfits are bound (strapjacket_resource 0.000 from male_vocal; the female hands and
# lolitatop at most 0.005 from female_vocal), so no outfit mesh needs fitting and seams
# like the female wrists stay closed. Their 34 face bones are ONE generic face, though:
# each head binds its own layout (head_2 a median 0.35 off, up to 0.70 at the lips; head_7
# 0.58) - importing a head onto the master as it is reshapes the face. So before each
# part's meshes come in, the bones they bind that aren't body bones (face, hair) are moved
# to where that part binds them, and bones they need that the rig lacks (bone_neckTwist,
# used by the outfits' skin pieces, is in neither master) are added there.

DECAL_LIFT = 0.1

MASTER_TYPES = {
    'GUITAR': ('guitar',),
    'VOCALS': ('vocal', 'vocals'),
    'DRUMS': ('drum', 'drums', 'drummer'),
}
_FACE_KEYS = ('brow', 'lid', 'lip', 'jaw', 'cheek', 'nose', 'crease', 'eye', 'tongue',
              'teeth', 'hair', 'mouth')


def master_type_for(instrument):
    inst = (instrument or '').lower()
    if inst.startswith('drum'):
        return 'DRUMS'
    if inst in ('guitar', 'bass') or inst.startswith('guitar'):
        return 'GUITAR'
    return 'VOCALS'


def find_master(root, gender, kind):
    """<gender>_<type> in char/<gender>/gen - the game's full rig for that instrument."""
    index = milo_index(root)
    for variant in MASTER_TYPES[kind]:
        name = f"{gender}_{variant}".lower()
        if name in index:
            return find_milo(root, name, None, gender)
    return None


def _stem(name):
    for ext in ('.mesh', '.trans'):
        if name.endswith(ext):
            return name[:-len(ext)]
    return name


def build_master_armature(context, path, name):
    """An armature from a master rig's own bones (its Trans hierarchy) - the same reading
    that lines up exactly with the outfits' bind poses."""
    from .mesh_importer import read_file_bones, add_file_bones
    fb = read_file_bones(path)
    if not fb:
        raise RuntimeError(f"no bones read from {os.path.basename(path)}")
    data = bpy.data.armatures.new(name)
    obj = bpy.data.objects.new(name, data)
    context.collection.objects.link(obj)
    _activate(context, obj)
    add_file_bones(context, obj, fb)
    obj["rb2_master"] = os.path.basename(path)
    return obj


def _rig_worlds(path):
    """{bone name: Blender world matrix} for a rig milo, chained through its own parents
    exactly as add_file_bones places them."""
    from .mesh_importer import read_file_bones
    from .io import _milo_to_blender_matrix
    fb = read_file_bones(path)
    world = {}

    def w(name, stack=()):
        if name in world:
            return world[name]
        local12, parent = fb[name]
        m = _milo_to_blender_matrix(local12)
        if parent and parent in fb and parent not in stack:
            m = w(parent, stack + (name,)) @ m
        world[name] = m
        return m
    for n in fb:
        w(n)
    return world


def conform_to_rig(context, armature, ref_path):
    """Puts every bone the armature shares with a reference rig where that rig has it, and
    carries every other bone along with its parent (its offset kept).

    The drum rigs store their body in a different pose from the standing bind every outfit
    is skinned to - male_drum's bone_spine1 is 9.41 units off, female_drum's breasts 5.55 -
    while the vocal rigs match those binds exactly. Conforming the drum rig to the vocal rig
    gives it the standing rest pose; the drum-only bones keep their place relative to their
    parents (the sticks stay in the hands) and the kit, which hangs from the rig's root, stays
    where it is. Drumming animation then poses the body. Returns the number of bones moved."""
    ref = {_stem(n): m for n, m in _rig_worlds(ref_path).items()}
    prev_active = context.view_layer.objects.active
    context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    moved = 0
    try:
        ebs = armature.data.edit_bones
        old = {eb.name: eb.matrix.copy() for eb in ebs}
        new = {}

        def visit(eb):
            nonlocal moved
            target = ref.get(_stem(eb.name))
            if target is not None:
                nm = target.normalized()
                if (nm.to_translation() - old[eb.name].to_translation()).length > 1e-4:
                    moved += 1
            elif eb.parent is not None:
                nm = new[eb.parent.name] @ (old[eb.parent.name].inverted() @ old[eb.name])
            else:
                nm = old[eb.name]
            new[eb.name] = nm
            for c in eb.children:
                visit(c)
        for eb in [e for e in ebs if e.parent is None]:
            visit(eb)
        for eb in ebs:
            length = eb.length
            eb.matrix = new[eb.name]
            eb.length = length
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')
        context.view_layer.objects.active = prev_active
    return moved


def retarget_to_part(context, armature, mesh_path, body_stems, hierarchy=None):
    """Before a part's meshes are imported: moves the armature's non-body bones (face) to
    where this part's meshes are bound, and adds bones the meshes need that the armature
    lacks and the part doesn't define itself. Meshes bound in a frame of their own (head_7's
    eyebrows: bone_head 86 units off, height on Y) are ignored - only meshes whose body
    bones agree with the armature count.

    Mount bones are the exception: a bone the part's file hangs from its own root (hair's
    bone_hair) is where the game ATTACHES the part, resolved by name in the character's rig.
    The rig keeps it at the head (61.31 for the female); hippybangs_maohat is bound to it
    0.78 lower (60.53) and libertyspikes to it turned 180 degrees, and the game draws both
    through the rig's bone - lifting the hat onto the head, turning the spikes off the face.
    Those bones aren't moved; instead their offset (rig rest x bind inverse) is returned for
    the part's meshes. Returns (moved, added, {mount bone name: offset matrix})."""
    from .mesh_importer import parse_rb2_meshes, read_file_bones
    from .io import _milo_to_blender_matrix
    try:
        _dn, meshes, _fb = parse_rb2_meshes(mesh_path)
    except Exception:
        return [], [], {}
    try:
        fb = read_file_bones(mesh_path)
    except Exception:
        fb = {}
    file_defined = {_stem(n) for n in fb}
    bones = armature.data.bones
    by_stem = {_stem(b.name): b.name for b in bones}

    def is_body(stem):
        if body_stems:
            return stem in body_stems
        return not any(k in stem.lower() for k in _FACE_KEYS)

    mount_stems = {_stem(n) for n, (_l, parent) in fb.items()
                   if parent not in fb and _stem(n) in by_stem and not is_body(_stem(n))}
    mounts = {}
    for m in meshes:
        if m.get('is_lod') or m.get('is_shadow') or m.get('is_helper'):
            continue
        for bn, bm in zip(m.get('bone_names') or (), m.get('bone_mats') or ()):
            st = _stem(bn)
            if st in mount_stems and by_stem[st] not in mounts:
                try:
                    bind_m = _milo_to_blender_matrix(bm).inverted()
                    mounts[by_stem[st]] = bones[by_stem[st]].matrix_local @ bind_m.inverted()
                except Exception:
                    pass

    bind = {}                                  # stem -> (bone name as the mesh has it, M)
    for m in meshes:
        if m.get('is_lod') or m.get('is_shadow') or m.get('is_helper'):
            continue
        mats = {}
        for bn, bm in zip(m.get('bone_names') or (), m.get('bone_mats') or ()):
            try:
                mats[bn] = _milo_to_blender_matrix(bm).inverted()
            except Exception:
                continue
        consistent = True
        for bn, M in mats.items():
            st = _stem(bn)
            an = by_stem.get(st)
            if an and is_body(st) and \
                    (M.to_translation() - bones[an].matrix_local.to_translation()).length > 0.5:
                consistent = False
                break
        if not consistent:
            _log(f"    retarget: '{m['name']}' is bound in a frame of its own - not used")
            continue
        for bn, M in mats.items():
            bind.setdefault(_stem(bn), (bn, M))

    # A mounted part's own bones (hair physics strands) go where its meshes are bound,
    # carried by the mount offset like the meshes themselves - not by their own transforms:
    # nightcat_resource's bone_hair-L-back-03 says (57.8, 10.8, 66.6), 62 units to the side,
    # while its mesh is bound to it at (-4.3, -2.0, 63.2), the end of the left back strand
    # (the game's hair physics re-seats it). Built from its own transform it dragged that
    # strand out sideways.
    mount_offset = next(iter(mounts.values())) if mounts else None
    file_parent = {_stem(n): _stem(p) for n, (_l, p) in fb.items()}
    move, add = {}, {}
    for st, (bn, M) in bind.items():
        an = by_stem.get(st)
        if an is None:
            if st not in file_defined:
                add[st] = (bn, M)
            elif mount_offset is not None:
                add[st] = (bn, mount_offset @ M)
        elif not is_body(st) and st not in mount_stems:
            rest = bones[an].matrix_local
            if (M.to_translation() - rest.to_translation()).length > 1e-3 or \
                    max(abs(a - b) for ra, rb in zip(M.to_3x3(), rest.to_3x3())
                        for a, b in zip(ra, rb)) > 1e-3:
                move[an] = M
    if not move and not add:
        return [], [], mounts

    prev_active = context.view_layer.objects.active
    context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    added = []
    try:
        ebs = armature.data.edit_bones
        for an, M in move.items():
            eb = ebs.get(an)
            if eb is None:
                continue
            length = eb.length
            eb.matrix = M.normalized()
            eb.length = length
        created = {}
        for st, (bn, M) in add.items():
            eb = ebs.new(bn)
            eb.head = (0.0, 0.0, 0.0)
            eb.tail = (0.0, 0.5, 0.0)
            eb.matrix = M.normalized()
            created[st] = eb
            added.append(bn)
        for st, eb in created.items():
            parent = None
            if hierarchy and st in hierarchy:
                gp = hierarchy[st][0]
                parent = ebs.get(by_stem.get(gp, '')) if gp else None
            if parent is None and st in file_parent:
                fp = file_parent[st]
                parent = created.get(fp) or ebs.get(by_stem.get(fp, ''))
            if parent is not None and parent is not eb:
                eb.parent = parent
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')
        context.view_layer.objects.active = prev_active
    return sorted(move), added, mounts


def mount_meshes(armature, objects, mounts, mesh_path):
    """Moves a part's meshes by their mount bone's offset (see retarget_to_part) - only meshes
    still where the file stores them, so one the mesh importer already fitted isn't moved
    twice. Returns the names moved."""
    import numpy as np
    from .mesh_importer import parse_rb2_meshes
    if not mounts:
        return []
    try:
        _dn, raw, _fb = parse_rb2_meshes(mesh_path)
    except Exception:
        return []
    raw_by_name = {m['name']: m for m in raw}
    done = []
    for obj in objects:
        if obj.type != 'MESH':
            continue
        groups = {vg.name for vg in obj.vertex_groups}
        hits = [b for b in mounts if b in groups]
        if not hits:
            continue
        src = raw_by_name.get(re.sub(r'\.\d{3}$', '', obj.name)) or raw_by_name.get(obj.data.name)
        me = obj.data
        if src is None or len(src['verts']) != len(me.vertices):
            continue
        n = min(16, len(me.vertices))
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get('co', co)
        co = co.reshape(-1, 3)[:n]
        stored = np.array([v['co'] for v in src['verts'][:n]])
        if np.abs(co - stored).max() > 1e-3:
            continue                          # already fitted by the mesh importer
        weight = {b: 0.0 for b in hits}
        idx = {obj.vertex_groups[b].index: b for b in hits}
        for v in me.vertices:
            for g in v.groups:
                if g.group in idx:
                    weight[idx[g.group]] += g.weight
        bone = max(weight, key=weight.get)
        to_obj = obj.matrix_world.inverted() @ armature.matrix_world
        me.transform(to_obj @ mounts[bone] @ to_obj.inverted())
        me.update()
        done.append(obj.name)
    return done


def frame_correct_meshes(armature, objects, mesh_path, tolerance=0.05):
    """Meshes stored in a frame of their own, which the game's skinning undoes and the
    mesh importer (finding every bone off the rig) leaves as stored.

    modavengerboots_resource.mesh carries a Y mirror (its own transform is diag(1, -1, 1))
    and is bound to bones mirrored the same way - bone_L-toe at y -2.86 where the rig has
    +2.86, the front - so imported as stored the boots face backwards. When all of a
    mesh's bones relate to the armature by the SAME transform T (rest @ bind^-1), the
    game draws the mesh moved by T; so is it here - faces flipped back outward when T is a
    mirror. Only meshes still exactly as the file stores them. Returns
    [(object name, 'mirror' | 'moved')]."""
    import numpy as np
    from mathutils import Matrix
    from .mesh_importer import parse_rb2_meshes
    from .io import _milo_to_blender_matrix
    try:
        _dn, raw, _fb = parse_rb2_meshes(mesh_path)
    except Exception:
        return []
    raw_by_name = {m['name']: m for m in raw}
    bones = armature.data.bones
    done = []
    for obj in objects:
        if obj.type != 'MESH':
            continue
        src = raw_by_name.get(re.sub(r'\.\d{3}$', '', obj.name)) or raw_by_name.get(obj.data.name)
        me = obj.data
        if src is None or len(src['verts']) != len(me.vertices) or not src.get('bone_names'):
            continue
        n = min(16, len(me.vertices))
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get('co', co)
        if np.abs(co.reshape(-1, 3)[:n] - np.array([v['co'] for v in src['verts'][:n]])).max() > 1e-3:
            continue                                   # the importer already fitted it
        ts = []
        for bn, bm in zip(src['bone_names'], src['bone_mats']):
            b = bones.get(bn)
            if b is None:
                continue
            try:
                # rest @ bind^-1, and bind^-1 is the bone table's own matrix
                ts.append(b.matrix_local @ _milo_to_blender_matrix(bm))
            except Exception:
                continue
        if not ts:
            continue
        t0 = ts[0]
        if any(max(abs(a - c) for ra, rc in zip(t, t0) for a, c in zip(ra, rc)) > tolerance
               for t in ts[1:]):
            continue                                   # not one shared frame
        if max(abs(a - c) for ra, rc in zip(t0, Matrix.Identity(4))
               for a, c in zip(ra, rc)) < 1e-3:
            continue                                   # already where the rig wants it
        to_obj = obj.matrix_world.inverted() @ armature.matrix_world
        me.transform(to_obj @ t0 @ to_obj.inverted())
        mirrored = t0.to_3x3().determinant() < 0
        if mirrored:
            me.flip_normals()
        me.update()
        done.append((obj.name, 'mirror' if mirrored else 'moved'))
    return done


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

    filename_ext = ""
    filter_glob: StringProperty(default="*", options={'HIDDEN'})

    character: StringProperty(
        name="Character",
        description="prefabs.dta: the checksum name (guitar32), its display name (The Duke of "
                    "Gravity) or prefab_guitar32. A save - Xbox 360 CON file such as 'band', "
                    "its save.dat, or a PS3 RB2.SAV: the name the character was given in "
                    "the game (Heather)")
    allow_unselectable: BoolProperty(
        name="Allow Non-Selectable",
        description="Also import characters the game doesn't offer in its menus "
                    "(selectable is not TRUE). These are less tested",
        default=False)
    skeleton: EnumProperty(
        name="Armature",
        items=(('AUTO', "Selected or by Instrument", "Use the selected armature; if none is "
                "selected, the game's full rig for the character's instrument "
                "(char/<gender>/gen/<gender>_guitar / _vocal / _drum)"),
               ('GUITAR', "Guitar Rig", "The game's guitar rig, <gender>_guitar"),
               ('VOCALS', "Vocals Rig", "The game's vocals rig, <gender>_vocal"),
               ('DRUMS', "Drums Rig", "The game's drums rig, <gender>_drum"),
               ('ACTIVE', "Selected Armature", "Use the selected armature only"),
               ('NONE', "None", "Import the meshes without an armature")),
        default='AUTO')
    retarget: BoolProperty(
        name="Match Face & Hair Bones",
        description="Before each part comes in, move the armature's face and hair bones to "
                    "where that part's meshes are bound (each head has its own face "
                    "layout), and add bones the meshes need that the rig lacks - so no mesh "
                    "is reshaped to fit the rig",
        default=True)
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
    decal_lift: FloatProperty(
        name="Decal Lift",
        description="How far decals that lie on clothing (a shirt's printed logo) are lifted "
                    "off it so the viewport draws them on top. The game keeps them ~0.005 "
                    "units off, too close for Blender's depth precision at a normal viewing "
                    "distance. Raise it if a logo still disappears into the clothing",
        default=DECAL_LIFT, min=0.0, max=2.0, step=1, precision=3)
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
    use_milo_shader: BoolProperty(
        name="Use Milo Shader",
        description="Shade the character with the game-style Milo Character Shader "
                    "(Blinn-Phong specular with the specular map's gloss, as the GDRB/TBRB "
                    "importers do) instead of a plain Principled BSDF. Experimental for "
                    "Rock Band 2; each material can be switched back in the Rock Band 2 "
                    "Character Creator Texture panel",
        default=False)
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
        layout.prop(self, "retarget")
        layout.prop(self, "import_file_bones")
        layout.prop(self, "apply_proportions")
        sub = layout.column()
        sub.enabled = self.apply_proportions
        sub.prop(self, "snap_proportions")
        layout.prop(self, "import_fur")
        layout.prop(self, "fit_outfit")
        layout.prop(self, "decal_lift")
        layout.prop(self, "load_textures")
        sub = layout.column()
        sub.enabled = self.load_textures
        sub.prop(self, "use_milo_shader")
        sub.prop(self, "eyebrow_override")
        layout.prop(self, "locale_file")

        path = self.filepath
        if path and os.path.isfile(path) and (path.lower().endswith('.dta')
                                              or is_save_file(path)):
            prefabs = _cached(_PREFAB_CACHE, path, load_characters)
            if not prefabs and is_save_file(path):
                try:
                    parse_save(path)
                except Exception as e:
                    box = layout.box()
                    box.label(text="No characters read:", icon='ERROR')
                    for line in _wrap(str(e), 44):
                        box.label(text=line)
                    return
            if prefabs:
                save = is_save_file(path)
                names = {} if save else display_names(
                    path, root, bpy.path.abspath(self.locale_file) if self.locale_file else "")
                box = layout.box()
                box.label(text="Characters in this save:" if save else "Selectable characters:")
                col = box.column(align=True)
                for key, pf in prefabs.items():
                    if not pf['selectable']:
                        continue
                    if save:
                        col.label(text=f"{key}  -  {pf['gender']}, from {pf.get('preset', '?')}")
                    else:
                        disp = names.get(key) or ("creator preset" if '/' in key else "?")
                        col.label(text=f"{key}  -  {disp}")
            elif is_save_file(path):
                layout.label(text="No characters read from this file", icon='ERROR')

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
        """A texture variant can bring meshes of its own that the resource milo doesn't
        have: a fur layer (furbikini_cheetah -> furbikini_fur.mesh) or a decal
        (thermalundertee_moped -> thermalundertee_patch.1.mesh, the moped logo on the chest,
        alpha-blended, its artwork in thermalundertee_moped_patch.tex). Imports the variant's
        meshes and keeps the ones whose material is fur or has a real texture in the
        variant - not patches whose texture the game paints at runtime
        (strapjacket_patch_diffuse.tex is an empty render target)."""
        try:
            probe = cc.read_texture_milo(milo)
        except Exception:
            return []
        fur_mats = {sl['target'] for sl in probe.slots if sl.get('fur')}
        fur_mats |= {sl['target'] for sl in probe.slots
                     if sl.get('from_mat') and isinstance(sl.get('diffuse_tex'), str)
                     and cc._has_pixels(probe.texs.get(sl['diffuse_tex']))
                     and sl['diffuse_tex'] not in probe.rendered}
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
        # A decal sits a hair off the clothing it lies on (the moped patch 0.005 units in
        # front of the tee) - the game draws it after, with a depth test; Blender's viewport
        # can't resolve that gap at a normal distance and buries it in the shirt (with the
        # default clip start of 0.01, surfaces ~100 units away need ~0.06 between them).
        # Lift transparent-z-mode decals (not cut-outs, not fur) along their normals by the
        # importer's Decal Lift - 0.1 by default, about 2-3 mm at real scale.
        slot_of = {sl['target']: sl for sl in probe.slots}
        for o in keep:
            mats = [cc._base_name(sl.material.name) for sl in o.material_slots
                    if sl.material is not None]
            sl = next((slot_of[m] for m in mats if m in slot_of), None)
            if sl and not sl.get('fur') and sl.get('z_mode') == cc.ZMODE_TRANSPARENT \
                    and not sl.get('alpha_cut'):
                me = o.data
                lift = float(getattr(self, 'decal_lift', DECAL_LIFT))
                for v in me.vertices:
                    v.co = v.co + v.normal * lift
                me.update()
                o["rb2_decal_lift"] = lift
                _log(f"    '{o.name}': decal lifted {lift:g} off the surface below")
        if keep:
            _log(f"    the variant's own meshes from {os.path.basename(milo)}: "
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
        save = is_save_file(self.filepath)
        if save:
            try:
                prefabs = parse_save(self.filepath)
            except Exception as e:
                self.report({'ERROR'}, f"Couldn't read the save: {e}")
                return {'CANCELLED'}
            _log(f"RB2 save: {len(prefabs)} character(s): {', '.join(prefabs)}")
        else:
            prefabs = _cached(_PREFAB_CACHE, self.filepath, parse_prefabs)
        if not prefabs:
            self.report({'ERROR'}, "No (prefab ...) characters found in this file - is it "
                                   "the decrypted prefabs.dta?")
            return {'CANCELLED'}
        names = {} if save else display_names(
            self.filepath, root, bpy.path.abspath(self.locale_file) if self.locale_file else "")
        key = self._resolve_character(prefabs, names)
        if key is None:
            hint = (f"characters in this save: {', '.join(prefabs)}" if save
                    else "use its checksum name, e.g. guitar32")
            self.report({'ERROR'}, f"No character '{self.character}' in this file ({hint})")
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
        is_master = False
        if self.skeleton in ('AUTO', 'ACTIVE') and prev_active is not None \
                and prev_active.type == 'ARMATURE':
            armature = prev_active
            _log(f"  armature: using selected '{armature.name}'")
        master_kind = self.skeleton if self.skeleton in MASTER_TYPES else (
            master_type_for(pf['instrument']) if self.skeleton == 'AUTO' and armature is None
            else None)
        if master_kind:
            master = find_master(root, gender, master_kind)
            if master is None:
                _log(f"  armature: no {gender} {master_kind.lower()} rig found in "
                     f"char/{gender}/gen" + (" - falling back to a skeleton search"
                                             if self.skeleton == 'AUTO' else ""))
                if self.skeleton != 'AUTO':
                    warnings.append(f"no {gender} {master_kind.lower()} rig found")
            else:
                try:
                    armature = build_master_armature(
                        context, master, f"{gender}_{master_kind.lower()}")
                    is_master = True
                    _log(f"  armature: the game's {master_kind.lower()} rig "
                         f"{os.path.basename(master)} ({len(armature.data.bones)} bones)")
                except Exception as e:
                    warnings.append(f"couldn't build the {master_kind.lower()} rig: {e}")
                if armature is not None and master_kind == 'DRUMS':
                    vocal = find_master(root, gender, 'VOCALS')
                    if vocal is None:
                        warnings.append("drum rig left in its stored (drumming) pose - the "
                                        f"{gender} vocal rig, which gives the standing pose "
                                        "the outfits are bound in, wasn't found")
                    else:
                        try:
                            n_moved = conform_to_rig(context, armature, vocal)
                            _log(f"  armature: drum rig put in the standing bind pose from "
                                 f"{os.path.basename(vocal)} ({n_moved} bone(s) moved; "
                                 f"sticks and kit kept with their parents)")
                        except Exception as e:
                            warnings.append(f"couldn't put the drum rig in the standing "
                                            f"pose: {e}")
        if armature is None and self.skeleton == 'AUTO':
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
                if armature is not None and not is_master:
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

        body_stems, hierarchy = None, None
        if armature is not None and self.retarget:
            dpath = find_deform(root, gender, pf['instrument'])
            if dpath:
                try:
                    hierarchy = rb2_proportions.read_deform_milo(dpath)['hierarchy']
                    body_stems = set(hierarchy)
                except Exception as e:
                    _log(f"  retarget: body bones from name only ({e})")

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
                mesh_src, chain = resolve_part_milos(root, milo, category, gender, item)
            except Exception as e:
                warnings.append(f"{category}: couldn't read '{item}' ({e})")
                continue
            if mesh_src is None:
                warnings.append(f"{category}: no meshes for '{item}' (looked through "
                                f"{', '.join(os.path.basename(c) for c in chain)})")
                continue
            _log(f"  {category}: {item} -> meshes from {os.path.basename(mesh_src)}"
                 + (f", textures from {os.path.basename(milo)}" if mesh_src != milo else "")
                 + (f" (via {' -> '.join(os.path.relpath(c, root) for c in chain[1:-1])})"
                    if len(chain) > 2 else ""))

            mounts = {}
            if armature is not None and self.retarget:
                try:
                    moved, added, mounts = retarget_to_part(context, armature, mesh_src,
                                                            body_stems, hierarchy)
                    if moved or added:
                        _log(f"    retarget: {len(moved)} bone(s) matched to "
                             f"{os.path.basename(mesh_src)}"
                             + (f" ({', '.join(_stem(n) for n in moved[:6])}"
                                f"{' ...' if len(moved) > 6 else ''})" if moved else "")
                             + (f"; added {', '.join(added)}" if added else ""))
                except Exception as e:
                    _log(f"    retarget failed: {e}")

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
            if armature is not None and self.retarget and mounts:
                try:
                    mounted = mount_meshes(armature, new_objs, mounts, mesh_src)
                    if mounted:
                        _log(f"    attached to the rig's {', '.join(_stem(b) for b in mounts)}: "
                             f"{', '.join(mounted)}")
                except Exception as e:
                    _log(f"    attaching to the rig failed: {e}")
            if armature is not None:
                try:
                    for name, how in frame_correct_meshes(armature, new_objs, mesh_src):
                        _log(f"    '{name}': stored in a frame of its own "
                             f"({'mirrored' if how == 'mirror' else 'shifted'}) - placed "
                             f"through its bones as the game does")
                except Exception as e:
                    _log(f"    frame check failed: {e}")
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
            # A variant can name textures - and leave materials - to the milos behind it:
            # its resource milo (hotpants_camo -> hotpants_camo_legs_diff.tex in
            # hotpants_resource) or an intermediate one (puffyskirt_barelegs ->
            # puffyskirt/puffyskirt_barelegs). Nearer milos win.
            for src in [c for c in chain if c != milo] + ([mesh_src] if mesh_src not in chain
                                                           else []):
                if src == milo:
                    continue
                try:
                    res_info = cc.read_texture_milo(src)
                except Exception as e:
                    _log(f"    textures in {os.path.basename(src)} unreadable: {e}")
                    continue
                for sl in res_info.slots:
                    if any(sl.get(k) for k in cc._TEX_KEYS):
                        res_slots.setdefault(sl['target'], sl)
                for n, t in res_info.texs.items():
                    if cc._has_pixels(t) and not cc._has_pixels(info.texs.get(n)):
                        info.texs[n] = t
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
                                       fur_shells=self.import_fur,
                                       milo_shader=self.use_milo_shader)
            warnings.extend(f"{category}: {w}" for w in warn)
            head_skin.extend(m for m, _sl in jobs if m.get("rb2_makeup_rects"))

            if category == 'heads' and self.eyebrow_override \
                    and pf['eyebrow_color_index'] is not None:
                for m, _sl in jobs:
                    pg = m.rb2cc
                    if pg.kind == 'ITEM' and pg.uses1 and pg.sel_palette1 == 'hair.pal':
                        pg["sel_index1"] = int(pf['eyebrow_color_index'])
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
