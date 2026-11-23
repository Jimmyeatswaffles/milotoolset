"""
Green Day: Rock Band venue importer - static meshes and textures.

Pick a venue's main milo (world/americanidiot/gen/americanidiot.milo_xbox) and this follows
everything it pulls in, placing every visible static mesh where the game draws it, with the
same textured materials the character importers build. With "Import Cameras" on, the camera
director's shots come in as well (see gdrb_venue_cameras). Each part can be turned off on its
own. Lights, crowds, particles, triggers and audio are not imported.

===========================================================================================
HOW A VENUE IS PUT TOGETHER - verified on the retail American Idiot venue (Xbox 360)
===========================================================================================
A venue is a chain of milos, each naming the next as a sub-directory in its root object:

    americanidiot.milo (WorldDir)         -> americanidiot_mocams.milo, americanidiot_bank.milo
      americanidiot_mocams.milo           -> xbx/americanidiot_lighting.milo
        americanidiot_lighting.milo       -> ../americanidiot_char.milo
          americanidiot_char.milo         -> americanidiot_geom.milo, ../shared/chars.milo
            americanidiot_geom.milo       -> (WorldInstances, below)

The chain is followed from the selected file, and every milo in it contributes its meshes.
Almost all of the geometry is in *_geom (stage, field, roof, trailers, banners, trusses); *_char
adds the static crowd cards and video-screen surfaces.

Props are shared between venues. A WorldInstance entry places one shared milo
('../shared/amps/monitor_crappy01.milo', '../../char/mic_static.milo') at its own transform -
nine monitors, twenty fire cans and so on. Paths are written from the venue's source folder,
and the built file sits in a 'gen' folder below it, so the reference above resolves to
world/shared/amps/gen/monitor_crappy01.milo_xbox (see texture_importer.find_shared_milo). The
first instance of each prop also carries a copy of the prop's objects in an inline directory
('monitor_crappy01.inst'); that copy is used when the shared milo isn't on disk.

An entry of type RndDir can also be an inline copy of a shared milo (geom's 'greendaysign'),
placed by its own transform. Its objects are imported through that transform. The particle
proxies in the main milo (Flame, Fireworks, Camera_Flash...) are RndDirs of this kind under
shared/fx and are skipped.

===========================================================================================
PLACEMENT
===========================================================================================
Vertices are mesh-local. Every Mesh, Group, RndDir and WorldInstance carries an RndTrans:
revision 9, a local and a world 4x3 row-vector matrix, a constraint, a target symbol, a
preserve-scale byte and the parent's name. These are located by that structure rather than
mesh_importer._find_trans, because venue transforms are routinely non-uniformly scaled
(monitor instances: 0.48 x 0.54 x 0.73) and that reader only accepts uniform scale - it missed
37 of geom's meshes. On every mesh both readers find, they agree exactly.

The world matrix is rebuilt from the local ones (local x parent's world, up to the directory,
whose own placement - identity, an instance's transform, an inline dir's transform - ends the
chain). That reproduces the stored world on 180 geom meshes, but not on objects in an inline
directory: greendaysign's sign mesh stores a world 124 units higher than its directory puts
it, which would push it through the stage roof. The stored world there is stale, so the chain
is what's used everywhere.

Meshes are drawn through Blender object transforms (not baked), so every copy of a prop and
every geometry instance shares one mesh datablock.

===========================================================================================
WHAT'S SKIPPED, AND WHY
===========================================================================================
* Geometry instances: a Mesh with no vertices whose geometry owner is another mesh (98 of
  geom's 211) borrows that mesh's data - not skipped, linked.
* Hidden meshes (the drawable's 'showing' flag is 0): in *_char these are exactly the 24
  crowd-placement areas the WorldCrowds fill. Optional.
* Skinned meshes (with bones), LODs and shadow casters, as in the character importers.
* flare_fake* meshes: camera-facing light flares, oriented at runtime.
* Light rig fixtures: in a milo with Spotlights, meshes with no placement of their own (every
  light_cannon_base, the strobes) are drawn by the spotlights and would pile up at the origin.
"""

import os
import re
import math
import struct
import time

import bpy
from bpy.props import StringProperty, BoolProperty
from bpy_extras.io_utils import ImportHelper

from .io import read_milo_container_body
from .utilities import _log
from . import mesh_importer as mi


_TRANS_REVISION = b'\x00\x00\x00\x09'
_MILO_REF_RE = re.compile(rb'[A-Za-z0-9_ ./\\-]+\.milo(?![A-Za-z0-9_])')
_MESH_REF_RE = re.compile(rb'[A-Za-z0-9_ .-]+\.mesh(?![A-Za-z0-9_])')
_IDENTITY4 = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
# Recursion guard for prop milos that place other prop milos.
MAX_INSTANCE_DEPTH = 6
LIGHT_RIG_TYPES = ('Spotlight', 'SpotlightDrawer')
# Sub-directories in the venue chain that are never scenery: the main milo also names a HUD
# preview ('../../test/hud/hudpre_4p_1p_wide.milo') and the audio banks.
_NOT_SCENERY_RE = re.compile(r'(^|/)(hud|sfx|test)/|_bank\.milo$|audio_mixer')


class VenueImportError(Exception):
    pass


# ---------------------------------------------------------------------------------------
# Directory walk
# ---------------------------------------------------------------------------------------
#
# mesh_importer.walk_milo_entries accepts any table-shaped bytes as a nested directory,
# including empty ones. A venue's camera shots are full of look-at targets - a tag, then
# 'billiejoe', 'bone_neck.mesh' and zeros - which read exactly like an empty directory, so
# that walk recursed through them until it gave up, and the main milos (where the shots live)
# couldn't be split at all. This copy only accepts real directory tables: a directory class
# (…Dir, Character, WorldInstance, CharClipSet) with at least one entry. With it every venue
# milo in world/ - americanidiot, dookie, twentyfirst and the shared folder - walks end to end.

_DIR_CLASSES = ('Character', 'WorldInstance', 'CharClipSet', 'BandCharacter')


def _strict_table(body, o):
    t = mi._dir_table_at(body, o)
    if t is None or not t[3]:
        return None
    if not (t[1].endswith('Dir') or t[1] in _DIR_CLASSES):
        return None
    return t


def _walk_dir(body, o, depth, path, out):
    if depth > 16:
        raise mi.MeshImportError("directories nested implausibly deep")
    t = _strict_table(body, o)
    if t is None:
        raise mi.MeshImportError(f"expected a directory table at {o:#x}")
    _rev, _dtype, dname, ents, p = t
    path = path + (dname,)
    _end, p = _walk_object(body, p, depth, path, out, None)
    for etype, ename in ents:
        start = p
        end, p = _walk_object(body, p, depth, path, out, ename)
        out.append((depth, path, etype, ename, start, end))
    return p


def _walk_object(body, p, depth, path, out, owner):
    while True:
        m = body.find(mi.MILO_END_MARKER, p)
        if m < 0:
            raise mi.MeshImportError(f"object at {p:#x} has no end marker")
        q, nested = p, None
        while True:
            q = body.find(b'\x00\x00\x00', q, m)
            if q < 0:
                break
            if _strict_table(body, q) is not None:
                nested = q
                break
            q += 1
        if nested is None:
            break
        p = _walk_dir(body, nested, depth + 1, path, out)
    p = m + 4
    if owner is not None and p < len(body):
        t = _strict_table(body, p)
        if t is not None and mi._names_owner(t[2], owner):
            p = _walk_dir(body, p, depth + 1, path, out)
    return m, p


def walk_entries(body):
    """Every entry of a venue milo as (depth, dir path, type, name, start, end). An empty
    directory (dookie_mocams) gives an empty list."""
    t = mi._dir_table_at(body, 0)
    if t is not None and not t[3]:
        return []
    out = []
    end = _walk_dir(body, 0, 0, (), out)
    if end != len(body):
        raise mi.MeshImportError(f"directory walk stopped at {end:#x} of {len(body):#x}")
    return out


# ---------------------------------------------------------------------------------------
# Low-level readers
# ---------------------------------------------------------------------------------------

def _sym(b, o, max_len=256):
    """(string, offset after it) for a length-prefixed printable symbol, or (None, o)."""
    if o + 4 > len(b):
        return None, o
    n = mi._u32(b, o)
    if n > max_len or o + 4 + n > len(b):
        return None, o
    s = b[o + 4:o + 4 + n]
    if not all(32 <= c < 127 for c in s):
        return None, o
    return s.decode('latin-1'), o + 4 + n


def _trans_at(b, o):
    """The RndTrans at o as dict(local, world, parent, end), or None. Structural: revision 9,
    two finite 4x3 matrices with non-degenerate axes (any scale, any skew), a small
    constraint, a printable target, a 0/1 flag and a printable parent name."""
    if b[o:o + 4] != _TRANS_REVISION:
        return None
    try:
        vals = struct.unpack_from('>24f', b, o + 4)
    except struct.error:
        return None
    if not all(math.isfinite(x) and abs(x) < 1e6 for x in vals):
        return None
    for m in (vals[:12], vals[12:]):
        for r in range(3):
            ln = math.sqrt(m[r * 3] ** 2 + m[r * 3 + 1] ** 2 + m[r * 3 + 2] ** 2)
            if not 1e-5 < ln < 1e4:
                return None
    q = o + 100
    if q + 13 > len(b) or mi._u32(b, q) > 16:
        return None
    _target, q = _sym(b, q + 4, 128)
    if _target is None or q >= len(b) or b[q] not in (0, 1):
        return None
    parent, q = _sym(b, q + 1, 256)
    if parent is None:
        return None
    return dict(local=vals[:12], world=vals[12:], parent=parent, end=q)


def find_trans(chunk, last=False):
    """First RndTrans in an object (meshes and groups put theirs near the start), or with
    last=True the last one - RndDir and WorldInstance put theirs at the end, after their
    directory data, so only the tail is searched."""
    lo = max(0, len(chunk) - 2048) if last else 0
    hit = None
    o = chunk.find(_TRANS_REVISION, lo)
    while o >= 0 and o + 113 <= len(chunk):
        r = _trans_at(chunk, o)
        if r is not None:
            if not last:
                return r
            hit = r
        o = chunk.find(_TRANS_REVISION, o + 1)
    return hit


def _showing(chunk, trans):
    """The RndDrawable 'showing' flag that follows a mesh's transform (draw revision, then
    the flag). True when it can't be read."""
    q = trans['end']
    if q + 5 > len(chunk) or mi._u32(chunk, q) > 32:
        return True
    return chunk[q + 4] != 0


def _m4(m12):
    return mi._m4(m12)


def _mul(a, b):
    return mi._mul4(a, b)


def _is_identity4(m, tol=1e-4):
    return all(abs(m[i][j] - (1.0 if i == j else 0.0)) < tol for i in range(4) for j in range(4))


def parse_instance_mesh(chunk, name):
    """A Mesh that draws another mesh's geometry: no vertices, no faces, and a geometry
    owner naming the mesh that has them. Returns dict(name, mat, geom_owner) or None."""
    trans = find_trans(chunk)
    if trans is None:
        return None
    # mat and geomOwner follow the drawable data: two consecutive printable symbols, the
    # second a .mesh, then mutable (4), volume (4), bsp (1) and a vertex count of 0. The
    # material needn't end in .mat - geom's curtains use 'BlackCanvas_NoPP'.
    o = trans['end']
    while o < min(len(chunk), trans['end'] + 256):
        m, o1 = _sym(chunk, o, 256)
        if m is not None:
            g, o2 = _sym(chunk, o1, 256)
            if g and g.lower().endswith('.mesh') and o2 + 14 <= len(chunk) \
                    and mi._u32(chunk, o2 + 9) == 0 and chunk[o2 + 13] in (0, 1):
                if g == name:
                    # Its own geometry, and it has none: an emptied mesh (dookie's
                    # WoodPlank18, the screens_01 tour buses). Nothing to draw.
                    return dict(name=name, mat=m, geom_owner=None)
                return dict(name=name, mat=m, geom_owner=g)
        o += 1
    return None


# ---------------------------------------------------------------------------------------
# One milo file, split into its directories
# ---------------------------------------------------------------------------------------

class _Dir:
    """One directory: the file's root, or an inline directory inside an entry (an RndDir
    copy of a shared milo, or a WorldInstance's copy of its prop)."""

    def __init__(self, milo, name, owner=None, parent=None):
        self.milo = milo
        self.name = name
        self.owner = owner              # (type, name, start, end) of the entry holding it
        self.parent = parent
        self.entries = []               # (type, name, start, end)
        self.children = []
        self._trans = {}
        self._pool = None

    @property
    def body(self):
        return self.milo.body

    def chunk(self, ent):
        return self.body[ent[2]:ent[3]]

    def by_name(self, name):
        if not hasattr(self, '_names'):
            self._names = {e[1]: e for e in self.entries}
        return self._names.get(name)

    def trans(self, name):
        if name not in self._trans:
            ent = self.by_name(name)
            if ent is None:
                self._trans[name] = None
            elif ent[0] in ('RndDir', 'WorldInstance'):
                self._trans[name] = find_trans(self.chunk(ent), last=True)
            else:
                self._trans[name] = find_trans(self.chunk(ent))
        return self._trans[name]

    def world(self, name, placement, _depth=0):
        """4x4 row-vector world matrix of an entry: its local matrix down the parent chain to
        this directory, which sits at `placement`."""
        t = self.trans(name)
        if t is None:
            return placement
        p = t['parent']
        if p and p != name and p != self.name and self.by_name(p) is not None and _depth < 32:
            parent_world = self.world(p, placement, _depth + 1)
        else:
            parent_world = placement
        return _mul(_m4(t['local']), parent_world)

    def pool(self):
        """(texs, mats, source) defined in this directory."""
        if self._pool is None:
            from .texture_importer import parse_tex, parse_mat, TextureImportError
            texs, mats = {}, {}
            for etype, ename, s, e in self.entries:
                try:
                    if etype == 'Tex' and ename not in texs:
                        texs[ename] = parse_tex(self.body[s:e], ename)
                    elif etype == 'Mat' and ename not in mats:
                        mats[ename] = parse_mat(self.body[s:e], ename)
                except (TextureImportError, struct.error, IndexError, ValueError) as ex:
                    self.milo.notes.append(f"{self.milo.basename}: skipped {ename}: {ex}")
            self._pool = (texs, mats, self.milo.basename)
        return self._pool


class _Milo:
    """A parsed milo file: its directories and the sub-directory milos its root names."""

    def __init__(self, path):
        self.path = path
        self.basename = os.path.basename(path)
        self.notes = []
        with open(path, 'rb') as f:
            self.body = read_milo_container_body(f.read())
        table = mi._dir_table_at(self.body, 0)
        if table is None:
            raise VenueImportError(f"'{self.basename}' doesn't start with a directory table")
        _rev, self.dir_type, self.dir_name, _ents, root_data = table

        # Sub-directory milos: named in the root object's own data, before its end marker.
        m = self.body.find(mi.MILO_END_MARKER, root_data)
        self.subdirs = []
        for ref in _MILO_REF_RE.findall(self.body[root_data:m if m > 0 else len(self.body)]):
            r = ref.decode('latin-1').replace('\\', '/').strip()
            if r and r not in self.subdirs:
                self.subdirs.append(r)

        self.root = _Dir(self, self.dir_name)
        self.table_types = {t for t, _n in _ents}
        try:
            walked = walk_entries(self.body)
        except mi.MeshImportError:
            walked = None
        if walked is None:
            # Flat directory the structural walk can't settle: the marker split. The main
            # WorldDir defeats both (1563 markers for 780 entries: particle proxies and
            # camera shots nest deeper than the walker follows) - it holds no static
            # geometry, so only its sub-directory list is used and its entries are left out.
            _r, _t, _n, ents = mi._read_dir_entries(self.body)
            self.table_types = {t for t, _n in ents}
            try:
                spans = mi._entry_spans(self.body, len(ents))
            except mi.MeshImportError:
                if 'Mesh' in self.table_types:
                    self.notes.append(f"{self.basename}: entries couldn't be split; its own "
                                      f"meshes are skipped, the milos it loads are not")
                return
            self.root.entries = [(t, n, s, e) for (t, n), (s, e) in zip(ents, spans)]
            return
        # Hang each nested entry on the entry that owns its directory. Paths alone can't
        # tell directories apart: geom's rack_tall_01 and rack_tall_02 instances both carry
        # a copy named 'rack_tall_01'. A directory is owned by the entry whose data contains
        # it (a WorldInstance's prop copy) or, failing that, by the entry it directly follows
        # and is named after (an inline RndDir: 'greendaysign' comes after its marker).
        recs = [(depth, dpath, (t, n, s, e)) for depth, dpath, t, n, s, e in walked]
        dirs = {None: self.root}
        owner_of = {}
        for depth, dpath, ent in sorted(recs, key=lambda r: r[0]):
            if depth == 0:
                self.root.entries.append(ent)
                continue
            s = ent[2]
            cands = [(pe, pp) for pd, pp, pe in recs if pd == depth - 1 and pp == dpath[:-1]]
            inside = [c for c in cands if c[0][2] <= s < c[0][3]]
            if inside:
                owner, opath = min(inside, key=lambda c: c[0][3] - c[0][2])
            else:
                before = [c for c in cands if c[0][3] <= s
                          and mi._names_owner(dpath[-1], c[0][1])]
                if not before:
                    continue
                owner, opath = max(before, key=lambda c: c[0][3])
            key = (owner[2], owner[3])
            if key not in dirs:
                parent = dirs.get(owner_of.get((opath, owner[2])), self.root)
                d = _Dir(self, dpath[-1], owner=owner, parent=parent)
                parent.children.append(d)
                dirs[key] = d
            dirs[key].entries.append(ent)
            owner_of[(dpath, s)] = key
        for d in dirs.values():
            d.entries.sort(key=lambda en: en[2])

    def has_type(self, *types):
        return any(t in self.table_types for t in types)


# ---------------------------------------------------------------------------------------
# Planning: what goes where (no Blender calls - testable on its own)
# ---------------------------------------------------------------------------------------

class VenuePlan:
    """Walks a venue and records every mesh placement.

    placements: dict(key, mesh, world, collection, object_name, mat, mat_pool, hidden)
      key        - identifies the geometry, shared by every placement drawing it
      mesh       - the parsed mesh dict (mesh_importer.parse_mesh_entry)
      world      - 4x4 row-vector matrix
      mat        - material name for this placement (an instance can differ from its owner)
      mat_pools  - [(texs, mats, source)] in lookup order
    """

    def __init__(self, filepath, follow_shared=True, folder_milos=True, skip_hidden=True,
                 skip_lod=True, skip_shadow=True, meshes=True):
        self.filepath = os.path.abspath(filepath)
        self.meshes = meshes
        self.follow_shared = follow_shared
        self.skip_hidden = skip_hidden
        self.skip_lod = skip_lod
        self.skip_shadow = skip_shadow
        self.placements = []
        self.notes = []
        self.skipped = {}               # reason -> count
        self.missing_refs = []
        self.chain = []                 # venue milos, in visit order
        self.props = {}                 # shared milo path -> instance count
        self._milos = {}
        self._resolved = {}
        self._parsed = {}               # (milo path, dir id, mesh name) -> dict or None
        self._global_pools = []

        main = self._load(self.filepath)
        if main is None:
            raise VenueImportError(f"couldn't read '{filepath}'")
        self.venue_name = main.dir_name
        self._visit_chain(main)
        if folder_milos:
            self._visit_folder()

    # -- files -----------------------------------------------------------------------------

    def _load(self, path):
        key = os.path.normcase(os.path.abspath(path))
        if key not in self._milos:
            try:
                self._milos[key] = _Milo(path)
            except Exception as e:
                self.notes.append(f"couldn't read '{os.path.basename(path)}': {e}")
                self._milos[key] = None
        return self._milos[key]

    def _resolve(self, from_path, ref):
        """A referenced milo on disk, or None. Cached: a venue names the same prop dozens of
        times, and a miss ends in a bounded folder search."""
        from .texture_importer import find_shared_milo
        key = (os.path.dirname(os.path.abspath(from_path)), ref)
        if key not in self._resolved:
            try:
                self._resolved[key] = find_shared_milo(from_path, ref)
            except Exception:
                self._resolved[key] = None
        return self._resolved[key]

    def _skip(self, reason, n=1):
        self.skipped[reason] = self.skipped.get(reason, 0) + n

    def _library_pools(self, milo, _seen=None):
        """Texture/material pools of a milo's sub-directory milos (amps_shared.milo for an
        amp), followed recursively - where a shared prop keeps materials it names."""
        _seen = _seen if _seen is not None else set()
        out = []
        for ref in milo.subdirs:
            p = self._resolve(milo.path, ref)
            if p is None:
                continue
            k = os.path.normcase(os.path.abspath(p))
            if k in _seen:
                continue
            _seen.add(k)
            sub = self._load(p)
            if sub is None:
                continue
            out += self._all_pools(sub.root)
            out += self._library_pools(sub, _seen)
        return out

    def _all_pools(self, d):
        out = [d.pool()]
        for c in d.children:
            if c.owner is None or c.owner[0] != 'WorldInstance':
                out += self._all_pools(c)
        return out

    def _pools_for(self, d, extra=()):
        chain = []
        x = d
        while x is not None:
            chain.append(x.pool())
            x = x.parent
        return chain + list(extra)

    # -- the venue chain -------------------------------------------------------------------

    def _visit_chain(self, milo):
        seen = set()
        queue = [milo]
        while queue:
            m = queue.pop(0)
            k = os.path.normcase(os.path.abspath(m.path))
            if k in seen:
                continue
            seen.add(k)
            self.chain.append(m)
            if m.dir_type == 'Character':
                self.notes.append(f"'{m.basename}' is a Character (crowd/band) - not static "
                                  f"geometry, skipped")
                continue
            if self.meshes:
                self._global_pools += self._all_pools(m.root)
                libs = self._library_pools(m)
                self._visit_dir(m.root, _IDENTITY4, collection=m.dir_name, prefix='',
                                libs=libs, depth=0, stack=(k,))
            for ref in m.subdirs:
                if _NOT_SCENERY_RE.search(ref.replace('\\', '/').lower()):
                    continue
                p = self._resolve(m.path, ref)
                if p is None:
                    if ref not in self.missing_refs:
                        self.missing_refs.append(ref)
                    continue
                sub = self._load(p)
                if sub is not None:
                    queue.append(sub)
        self._chain_keys = seen

    def _visit_folder(self):
        """Milos beside the main one that the chain never reached (exterior_props sits in
        the venue folder but no venue milo names it). Only those holding static meshes."""
        folder = os.path.dirname(self.filepath)
        ext = os.path.splitext(self.filepath)[1].lower()
        try:
            names = sorted(os.listdir(folder))
        except OSError:
            return
        for fn in names:
            if os.path.splitext(fn)[1].lower() != ext:
                continue
            p = os.path.join(folder, fn)
            k = os.path.normcase(os.path.abspath(p))
            if k in self._chain_keys:
                continue
            m = self._load(p)
            if m is None or m.dir_type == 'Character':
                continue
            has_meshes = self.meshes and m.has_type('Mesh')
            if not (has_meshes or m.has_type('BandCamShot')):
                continue
            self.chain.append(m)
            self._chain_keys.add(k)
            self.notes.append(f"'{fn}' isn't referenced by the venue but sits beside it - "
                              f"imported too")
            if has_meshes:
                self._global_pools += self._all_pools(m.root)
                self._visit_dir(m.root, _IDENTITY4, collection=m.dir_name, prefix='',
                                libs=self._library_pools(m), depth=0, stack=(k,))

    # -- one directory ---------------------------------------------------------------------

    def _parsed_mesh(self, d, ent):
        key = (d.milo.path, id(d), ent[1])
        if key not in self._parsed:
            chunk = d.chunk(ent)
            res = None
            try:
                res = mi.parse_mesh_entry(chunk, ent[1])
                res['instance_of'] = None
            except (mi.MeshImportError, struct.error, IndexError):
                inst = parse_instance_mesh(chunk, ent[1])
                if inst is not None and inst['geom_owner'] is None:
                    res = dict(name=ent[1], mat=inst['mat'], instance_of=None, verts=[],
                               faces=[], bone_names=[])
                elif inst is not None:
                    res = dict(name=ent[1], mat=inst['mat'], instance_of=inst['geom_owner'])
            if res is not None:
                t = d.trans(ent[1])
                res['hidden'] = (t is not None and not _showing(chunk, t))
            self._parsed[key] = res
        return self._parsed[key]

    def _visit_dir(self, d, placement, collection, prefix, libs, depth, stack):
        light_rig = d.milo.has_type(*LIGHT_RIG_TYPES)
        pools = self._pools_for(d, libs)

        for ent in d.entries:
            etype, ename = ent[0], ent[1]
            if etype != 'Mesh':
                continue
            m = self._parsed_mesh(d, ent)
            if m is None:
                self._skip('unreadable')
                self.notes.append(f"{d.milo.basename}: couldn't read mesh '{ename}'")
                continue
            if self.skip_lod and mi.is_lod_name(ename):
                self._skip('LOD'); continue
            if self.skip_shadow and mi.is_shadow_name(ename):
                self._skip('shadow'); continue
            if ename.lower().startswith('flare_fake'):
                self._skip('fake flare'); continue
            if self.skip_hidden and m['hidden']:
                self._skip('hidden'); continue
            geo = m
            if m['instance_of']:
                owner = d.by_name(m['instance_of'])
                geo = self._parsed_mesh(d, owner) if owner and owner[0] == 'Mesh' else None
                if geo is None or geo.get('instance_of') or not geo.get('verts'):
                    self._skip('instance of a missing mesh')
                    self.notes.append(f"{d.milo.basename}: '{ename}' draws "
                                      f"'{m['instance_of']}', which isn't readable here")
                    continue
            if not geo.get('verts') or not geo.get('faces'):
                self._skip('empty'); continue
            if not m['mat'] and not geo.get('mat'):
                self._skip('helper (no material)'); continue
            # A bone table of empty names is a rigid mesh (mic_static's mic and stand).
            skinned = any(geo.get('bone_names') or ())
            if skinned and depth == 0:
                # At venue level a skinned mesh is a character piece - not scenery.
                self._skip('skinned'); continue
            if skinned:
                # Inside a prop: a jiggle rig (speaker_cone_2x2's cones, bones driven by
                # kick/snare TransAnims). Its bind matrices invert the bones' rest poses, so
                # at rest the vertices are already where they're drawn, in the prop's space.
                world = placement
            else:
                world = d.world(ename, placement)
            if light_rig and _is_identity4(world):
                self._skip('light rig fixture'); continue
            self.placements.append(dict(
                key=(d.milo.path, id(d), geo['name']),
                mesh=geo, world=world, collection=collection,
                object_name=(prefix + ename) if prefix else ename,
                mat=m['mat'] or geo.get('mat', ''), mat_pools=pools,
                source_file=d.milo.basename))

        for c in d.children:
            otype = c.owner[0] if c.owner else None
            if otype != 'RndDir':
                continue                # WorldInstance copies are a fallback; others aren't geometry
            refs = _MILO_REF_RE.findall(d.chunk(c.owner))
            ref = refs[0].decode('latin-1') if refs else ''
            if '/fx/' in ref.replace('\\', '/').lower():
                self._skip('particle effect dir'); continue
            cw = d.world(c.owner[1], placement)
            self._visit_dir(c, cw, collection, prefix, libs, depth, stack)

        for ent in d.entries:
            if ent[0] == 'WorldInstance':
                self._visit_instance(d, ent, placement, collection, prefix, libs, depth, stack)

    def _visit_instance(self, d, ent, placement, collection, prefix, libs, depth, stack):
        chunk = d.chunk(ent)
        ref, _o = _sym(chunk, 4, 512)
        if not ref:
            # Path left empty - seen throughout americanidiot_screens_01 (a screenshot build).
            # Nothing to place.
            self._skip('unbound prop instance (no milo path)')
            return
        world = d.world(ent[1], placement)
        label = os.path.splitext(os.path.basename(ref.replace('\\', '/')))[0]
        name_prefix = f"{prefix}{ent[1]}:"

        path = self._resolve(d.milo.path, ref) if self.follow_shared else None
        if path is not None and depth < MAX_INSTANCE_DEPTH:
            k = os.path.normcase(os.path.abspath(path))
            if k in stack:
                self.notes.append(f"'{ref}' places itself - stopped")
                return
            prop = self._load(path)
            if prop is not None:
                self.props[path] = self.props.get(path, 0) + 1
                self._visit_dir(prop.root, world, collection=f"Props/{label}",
                                prefix=name_prefix, libs=self._library_pools(prop)
                                + self._global_pools, depth=depth + 1, stack=stack + (k,))
                return
        # Not on disk (or not followed): the copy the venue keeps with the prop's first
        # instance.
        embedded = self._embedded_copy(d, ref)
        if embedded is None:
            if ref not in self.missing_refs:
                self.missing_refs.append(ref)
            self._skip('prop not found')
            return
        self.props[ref + ' (embedded copy)'] = self.props.get(ref + ' (embedded copy)', 0) + 1
        self._visit_dir(embedded, world, collection=f"Props/{label}", prefix=name_prefix,
                        libs=libs + self._global_pools, depth=depth + 1, stack=stack)

    def _embedded_copy(self, d, ref):
        for c in d.children:
            if c.owner and c.owner[0] == 'WorldInstance':
                r, _o = _sym(d.chunk(c.owner), 4, 512)
                if r == ref:
                    return c
        return None


# ---------------------------------------------------------------------------------------
# Blender construction
# ---------------------------------------------------------------------------------------

def _blender_matrix(m4):
    from mathutils import Matrix
    return Matrix(((m4[0][0], m4[1][0], m4[2][0], m4[3][0]),
                   (m4[0][1], m4[1][1], m4[2][1], m4[3][1]),
                   (m4[0][2], m4[1][2], m4[2][2], m4[3][2]),
                   (0.0, 0.0, 0.0, 1.0)))


def _collection(root, path, cache):
    if path in cache:
        return cache[path]
    parent = root
    built = ''
    for part in path.split('/'):
        built = f"{built}/{part}" if built else part
        if built not in cache:
            col = bpy.data.collections.new(part)
            parent.children.link(col)
            cache[built] = col
        parent = cache[built]
    return parent


def _build_mesh_data(mesh):
    verts = mesh['verts']
    me = bpy.data.meshes.new(mesh['name'])
    me.from_pydata([v['co'] for v in verts], [], [list(f) for f in mesh['faces']])
    uv = me.uv_layers.new(name="UVMap")
    vi = [0] * len(me.loops)
    me.loops.foreach_get("vertex_index", vi)
    flat = []
    for i in vi:
        flat += verts[i]['uv']
    uv.data.foreach_set("uv", flat)
    me.validate(clean_customdata=False)
    me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    try:
        me.normals_split_custom_set_from_vertices([v['normal'] for v in verts])
    except (RuntimeError, AttributeError, ValueError):
        pass
    me.update()
    return me


class _Texturer:
    """Builds each material once, from the first pool in its lookup order that defines it,
    taking its textures from that pool or, failing that, from any later pool."""

    def __init__(self, flip_green):
        self.flip_green = flip_green
        self.done = {}
        self.notes = []
        self.missing_tex = set()
        self.missing_mat = set()
        self.textured = 0
        self.images = 0

    def material(self, mat_name, pools, fallback_source, with_textures):
        from .mesh_importer import milo_material
        if not mat_name:
            return milo_material('', fallback_source)
        hit = next(((texs, mats, src) for texs, mats, src in pools if mat_name in mats), None)
        if hit is None:
            self.missing_mat.add(mat_name)
            return milo_material(mat_name, fallback_source)
        _texs, mats, src = hit
        key = (src, mat_name)
        if key in self.done:
            return self.done[key]
        mat = milo_material(mat_name, src)
        self.done[key] = mat
        if with_textures:
            try:
                self._texture(mat, mat_name, mats[mat_name], pools, hit)
                self.textured += 1
            except Exception as e:
                self.notes.append(f"'{mat_name}': textures failed ({e})")
        return mat

    def _texture(self, mat, mat_name, slots, pools, hit):
        from .texture_importer import (build_image, build_material_tree,
                                       fill_material_settings, _viewport_color,
                                       TextureImportError)
        ordered = [hit] + [p for p in pools if p is not hit]

        def image_for(tex_name, role):
            if not tex_name:
                return None
            tex = next((texs[tex_name] for texs, _m, _s in ordered if tex_name in texs), None)
            if tex is None:
                self.missing_tex.add(tex_name)
                return None
            if tex.get('render_target'):
                return None             # filled at runtime (video screens)
            img_role = 'normal' if role == 'detail' else role
            try:
                img = build_image(tex_name, tex, img_role, self.flip_green)
                self.images += 1
                return img
            except (TextureImportError, ValueError) as e:
                self.notes.append(f"could not decode '{tex_name}': {e}")
                return None

        imgs = {role: image_for(slots.get(key), role)
                for role, key in (('diffuse', 'diffuse'), ('normal', 'normal'),
                                  ('specular', 'specular'), ('emissive', 'emissive'),
                                  ('detail', 'detail_normal'))}
        build_material_tree(mat, slots, imgs)
        fill_material_settings(mat, slots, imgs)
        _viewport_color(mat, slots, imgs.get('diffuse'))


def venue_collection(context, plan):
    root = bpy.data.collections.new(f"Venue {plan.venue_name}")
    context.scene.collection.children.link(root)
    return root


def build_venue(context, plan, import_textures=True, flip_green=True, root=None):
    """Creates the plan's objects under one collection named after the venue (or under
    root). Returns (objects created, mesh datablocks, texturer)."""
    if root is None:
        root = venue_collection(context, plan)
    cols = {}
    meshes = {}
    texturer = _Texturer(flip_green)
    created = 0
    for pl in plan.placements:
        me = meshes.get(pl['key'])
        if me is None:
            me = _build_mesh_data(pl['mesh'])
            mat = texturer.material(pl['mesh'].get('mat', ''), pl['mat_pools'],
                                    pl['source_file'], import_textures)
            me.materials.append(mat)
            meshes[pl['key']] = me
        name = pl['object_name']
        if name.lower().endswith('.mesh'):
            name = name[:-5]
        obj = bpy.data.objects.new(name, me)
        obj.matrix_world = _blender_matrix(pl['world'])
        obj["milo_source"] = pl['source_file']
        obj["milo_mesh"] = pl['object_name']
        obj["milo_material"] = pl['mat']
        if pl['mat'] and pl['mat'] != pl['mesh'].get('mat', ''):
            # A geometry instance with a material of its own.
            obj.material_slots[0].link = 'OBJECT'
            obj.material_slots[0].material = texturer.material(
                pl['mat'], pl['mat_pools'], pl['source_file'], import_textures)
        _collection(root, pl['collection'], cols).objects.link(obj)
        created += 1
    return created, len(meshes), texturer


# ---------------------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------------------

class IMPORT_OT_gdrb_venue(bpy.types.Operator, ImportHelper):
    """Import a Green Day: Rock Band venue's static meshes and textures. Select the venue's
    main milo (e.g. americanidiot.milo_xbox); the milos it loads and the shared props it
    places (world/shared/...) are found and imported automatically"""
    bl_idname = "import_scene.gdrb_venue"
    bl_label = "Import GDRB Venue"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".milo_xbox"
    filter_glob: StringProperty(default="*.milo_xbox;*.milo_ps3;*.milo", options={'HIDDEN'})

    follow_shared: BoolProperty(
        name="Import Shared Props",
        description="Place the shared prop milos the venue instances (amps, speaker towers, "
                     "jumbotrons, mics...) from world/shared. When one isn't on disk, the "
                     "copy the venue keeps with its first instance is used instead",
        default=True,
    )
    folder_milos: BoolProperty(
        name="Include Unreferenced Venue Milos",
        description="Also import static meshes from milos in the same folder that the venue "
                     "chain doesn't name (exterior_props: tour bus, tents, fences)",
        default=True,
    )
    import_textures: BoolProperty(
        name="Import Textures",
        description="Decode the textures and rebuild each material, as the character "
                     "mesh importers do",
        default=True,
    )
    flip_normal_green: BoolProperty(
        name="Flip Normal Map Green",
        description="Convert the normal maps' Y from DirectX to Blender's OpenGL",
        default=True,
    )
    skip_hidden: BoolProperty(
        name="Skip Hidden Meshes",
        description="Leave out meshes the game doesn't draw (the crowd placement areas)",
        default=True,
    )
    exclude_lod: BoolProperty(
        name="Exclude LOD Meshes", default=True,
        description="Skip reduced-detail copies (names containing LOD01, LOD02...)",
    )
    exclude_shadow: BoolProperty(
        name="Exclude Shadow Meshes", default=True,
        description="Skip shadow-caster meshes",
    )

    import_meshes: BoolProperty(
        name="Import Static Meshes",
        description="Import the venue's scenery and shared props",
        default=True,
    )
    import_cameras: BoolProperty(
        name="Import Cameras",
        description="Import the camera director's shots (BandCamShots) as Blender cameras, "
                     "keyed through their keyframes. Shots framed on a band member are "
                     "parented to an Empty at his stage position",
        default=True,
    )
    include_mocap: BoolProperty(
        name="Include Mocap Cameras",
        description="Also import the venue's motion-captured camera tracks "
                     "(americanidiot_mocams). They're relative to the performer, so they "
                     "come in under one Empty to move into place",
        default=True,
    )
    set_scene_range: BoolProperty(
        name="Extend Scene Frame Range",
        description="Extend the scene's end frame to cover the longest camera shot",
        default=False,
    )

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.prop(self, "import_meshes")
        col = box.column()
        col.enabled = self.import_meshes
        for p in ("follow_shared", "folder_milos", "import_textures", "flip_normal_green",
                  "skip_hidden", "exclude_lod", "exclude_shadow"):
            col.prop(self, p)
        box = layout.box()
        box.prop(self, "import_cameras")
        col = box.column()
        col.enabled = self.import_cameras
        col.prop(self, "include_mocap")
        col.prop(self, "set_scene_range")

    def execute(self, context):
        t0 = time.time()
        if not (self.import_meshes or self.import_cameras):
            self.report({'ERROR'}, "Nothing to import - turn on meshes or cameras")
            return {'CANCELLED'}
        _log(f"===== Importing GDRB venue from '{self.filepath}' =====")
        try:
            plan = VenuePlan(self.filepath, follow_shared=self.follow_shared,
                             folder_milos=self.folder_milos, skip_hidden=self.skip_hidden,
                             skip_lod=self.exclude_lod, skip_shadow=self.exclude_shadow,
                             meshes=self.import_meshes)
        except Exception as e:
            _log(f"GDRB VENUE IMPORT FAILED: {e}")
            self.report({'ERROR'}, f"Could not read venue: {e}")
            return {'CANCELLED'}

        _log(f"  venue chain: " + ", ".join(m.basename for m in plan.chain))
        parts = []
        warn = False
        root = None
        if self.import_meshes:
            for p, n in sorted(plan.props.items()):
                _log(f"  prop x{n}: {p}")
            for n in plan.notes:
                _log(f"  {n}")
            for m in plan._milos.values():
                if m is not None:
                    for n in m.notes:
                        _log(f"  {n}")
            if plan.skipped:
                _log("  skipped: " + ", ".join(f"{v} {k}"
                                               for k, v in sorted(plan.skipped.items())))
            if plan.missing_refs:
                _log(f"  {len(plan.missing_refs)} referenced milo(s) not found on disk: "
                     + ", ".join(plan.missing_refs))
                warn = True
            if plan.placements:
                root = venue_collection(context, plan)
                created, n_meshes, tx = build_venue(context, plan, self.import_textures,
                                                    self.flip_normal_green, root=root)
                for n in tx.notes:
                    _log(f"  {n}")
                if tx.missing_mat:
                    _log(f"  {len(tx.missing_mat)} material(s) not defined anywhere loaded: "
                         + ", ".join(sorted(tx.missing_mat)))
                if tx.missing_tex:
                    _log(f"  {len(tx.missing_tex)} texture(s) not found: "
                         + ", ".join(sorted(tx.missing_tex)))
                parts.append(f"{created} object(s), {n_meshes} mesh(es), "
                             f"{len(plan.props)} prop milo(s)"
                             + (f", {tx.textured} material(s) textured"
                                if self.import_textures else ""))
            else:
                parts.append("no static meshes found")
                warn = True

        if self.import_cameras:
            from .gdrb_venue_cameras import import_venue_cameras
            if root is None:
                root = venue_collection(context, plan)
            try:
                parts.append(import_venue_cameras(context, root, plan.chain,
                                                  include_mocap=self.include_mocap,
                                                  set_scene_range=self.set_scene_range))
            except Exception as e:
                _log(f"  cameras failed: {e}")
                parts.append("cameras failed (see log)")
                warn = True

        if root is None:
            self.report({'ERROR'}, "Nothing found to import in this venue")
            return {'CANCELLED'}
        summary = (f"Imported venue '{plan.venue_name}': " + "; ".join(parts)
                   + (f"; {len(plan.missing_refs)} milo(s) not found (see log)"
                      if self.import_meshes and plan.missing_refs else "")
                   + f" in {time.time() - t0:.1f}s")
        _log(f"===== {summary} =====")
        self.report({'WARNING' if warn else 'INFO'}, summary)
        return {'FINISHED'}
