"""
Rock Band 3 outfit fitting: CharMeshHide (hiding what another item covers) and CharCuff
(squeezing e.g. pants into boots).

The files are the same as Rock Band 2's (CharMeshHide rev 2, CharCuff rev 8), so the RB2
parsers read them (rb2_outfit_fit); RB3 items keep them in their nested '_resource'
directories, so they're found with the nested walker.

CharMeshHide (system/char/CharMeshHide.cpp)
    HideAll ORs every item's flags, plus the scene's (vignettes add 0x2000 HIDE_VIGNETTE;
    venue and closet nothing), and each Hide's drawable shows when show != !(flags hit).
    show 0: hidden when covered; show 1: shown only when covered.

CharCuff (360: Deform @0x8238cac8, DeformMesh @0x8238c568 - the Wii decomp's DeformMesh is
a stub, so this follows the executable)
    Per bone, the cuff with the largest outer radius wins (ties: the later name). It works
    on the meshes of the item in its category's slot ('legs': the pants), skipping the
    ignore lists, and only on vertices that reference a bone at or below its own (the mask:
    any of the vertex's bones, not weighted). In the cuff's frame - a along its z axis,
    d the offset from the axis, the radius measured elliptically (y scaled by
    1 / eccentricity) - with three rings (offset, radius) at rising offsets:
        a >= top ring offset          untouched
        between rings                 radius limited to the rings' linear profile: a
                                      vertex outside it is pulled in onto it
        below the bottom ring         closed cuff: put ON the bottom ring at its radius;
                                      open cuff: untouched
    then, closed cuffs only, every triangle whose three vertices all sit below the bottom
    ring (+0.01) is removed. It runs on the bind-pose meshes before body proportions, so
    the squeeze goes into an 'RB3 Cuff' shape key and the removed triangles into an
    'RB3 Cuff Hidden' vertex group masked by an 'RB3 Cuff' Mask modifier (switch either
    off to undo).
"""

import re
import struct

import bpy
import numpy as np

from .io import read_milo_container_body
from .utilities import _log
from .rb2_outfit_fit import FitError, parse_meshhide, parse_cuff

CUFF_KEY = "RB3 Cuff"
HIDDEN_GROUP = "RB3 Cuff Hidden"
MASK_NAME = "RB3 Cuff"
_DUP = re.compile(r'\.\d{3}$')


def _stem(name):
    for ext in ('.mesh', '.trans'):
        if name.endswith(ext):
            return name[:-len(ext)]
    return name


def _base(obj_name):
    return _DUP.sub('', obj_name)


def read_fit(path):
    """{'hides': [meshhide], 'cuffs': [(name, cuff)]} from an RB3 item milo (nested)."""
    from .mesh_importer import walk_milo_entries
    with open(path, 'rb') as f:
        body = read_milo_container_body(f.read())
    out = dict(hides=[], cuffs=[])
    for _depth, _p, t, n, s, e in walk_milo_entries(body):
        try:
            if t == 'CharMeshHide':
                out['hides'].append(dict(parse_meshhide(body[s:e]), name=n))
            elif t == 'CharCuff':
                out['cuffs'].append((n, parse_cuff(body[s:e])))
        except (FitError, struct.error) as ex:
            _log(f"    {t} '{n}' skipped: {ex}")
    return out


# ---------------------------------------------------------------------------------------
# Hiding
# ---------------------------------------------------------------------------------------

def apply_hides(parts, scene_flags=0):
    """parts: [(category, fit, [objects])]. Shows / hides each listed drawable by the
    combined flags. Returns ([(object, category, flags)], combined)."""
    combined = scene_flags
    for _cat, fit, _objs in parts:
        for h in fit['hides']:
            combined |= h['flags']
    changed = []
    for cat, fit, objs in parts:
        by_name = {}
        for o in objs:
            b = _base(o.name)
            by_name.setdefault(b, []).append(o)
            by_name.setdefault(_stem(b), []).append(o)
        for h in fit['hides']:
            for draw, flags, show in h['hides']:
                covered = bool(combined & flags)
                showing = covered if show else not covered     # show != !(covered)
                for o in {id(x): x for x in by_name.get(draw, []) +
                          by_name.get(_stem(draw), [])}.values():
                    if showing:
                        continue
                    o.hide_set(True)
                    o.hide_render = True
                    o["rb3_hidden_by_flags"] = int(combined & flags)
                    changed.append((o.name, cat, int(combined & flags)))
    return changed, combined


# ---------------------------------------------------------------------------------------
# Cuffs
# ---------------------------------------------------------------------------------------

def _subtree(arm_obj, stem):
    """Bone stems at or below `stem` in the armature."""
    by_stem = {_stem(b.name): b for b in arm_obj.data.bones}
    root = by_stem.get(stem)
    if root is None:
        return set()
    out = set()
    todo = [root]
    while todo:
        b = todo.pop()
        out.add(_stem(b.name))
        todo.extend(b.children)
    return out


def _frame(cu):
    """(origin, x, y, z) of the cuff in armature (game) space: its world matrix is
    row-vector (rows = axes, row 3 = position)."""
    w = np.array(cu['world'], dtype=np.float64)
    return w[3, :3], w[0, :3], w[1, :3], w[2, :3]


def squeeze(points, mask, cu):
    """CharCuff::DeformMesh on armature-space points (n x 3) for the vertices in `mask`.
    Returns (new points, moved count, below-bottom flags)."""
    o, x, y, z = _frame(cu)
    (ra, oa), (rb, ob), (rc, oc) = cu['shape']   # rings: (radius, offset)
    inv_e2 = 1.0 / max(cu['ecc'], 1e-6) ** 2
    out = points.copy()
    a = (points - o) @ z
    d = points - (o + np.outer(a, z))
    px, py = d @ x, d @ y
    planar = px * px + py * py
    dd = np.einsum('ij,ij->i', d, d)
    with np.errstate(divide='ignore', invalid='ignore'):
        rr = np.where(planar > 1e-12, (px * px + py * py * inv_e2) / planar * dd, 0.0)
    moved = np.zeros(len(points), bool)
    work = mask & (a < oc)
    below = work & (a < oa)
    if not cu['open']:
        sel = below & (rr > 1e-12)
        s = ra / np.sqrt(rr[sel])
        out[sel] = o + np.outer(np.full(sel.sum(), oa), z) + d[sel] * s[:, None]
        moved |= sel
        # a vertex sitting on the axis still lands on the ring's plane
        on_axis = below & ~(rr > 1e-12)
        out[on_axis] = o + np.outer(np.full(on_axis.sum(), oa), z) + d[on_axis]
        moved |= on_axis
    mid = work & (a >= oa)
    lo = mid & (a < ob)
    hi = mid & (a >= ob)
    r = np.zeros(len(points))
    if oa != ob:
        r[lo] = (a[lo] - ob) / (oa - ob) * (ra - rb) + rb
    else:
        r[lo] = rb
    if ob != oc:
        r[hi] = (a[hi] - oc) / (ob - oc) * (rb - rc) + rc
    else:
        r[hi] = rc
    pull = mid & (r * r < rr)
    s = r[pull] / np.sqrt(rr[pull])
    out[pull] = (o + np.outer(a[pull], z)) + d[pull] * s[:, None]
    moved |= pull
    return out, int(moved.sum()), (below if not cu['open'] else np.zeros(len(points), bool))


def winning_cuffs(parts):
    """[(category, name, cuff, ignore)] - per bone, the largest outer radius (ties: the
    later name), with every rival's ignore list."""
    cuffs = [(cat, n, cu) for cat, fit, _objs in parts for n, cu in fit['cuffs']]
    out = []
    for cat, name, cu in cuffs:
        rivals = [(n, c) for _k, n, c in cuffs if c['bone'] == cu['bone'] and c is not cu]
        if any(c['outer'] > cu['outer'] or (c['outer'] == cu['outer'] and n > name)
               for n, c in rivals):
            continue
        ignore = set(cu['ignore']) | {i for _n, c in rivals for i in c['ignore']}
        out.append((cat, name, cu, ignore))
    return out


def _vertex_mask(obj, bones):
    """Vertices referencing any of `bones` (the game checks the vertex's bone indices)."""
    groups = {vg.index for vg in obj.vertex_groups if _stem(vg.name) in bones}
    n = len(obj.data.vertices)
    mask = np.zeros(n, bool)
    if not groups:
        return mask
    for v in obj.data.vertices:
        for g in v.groups:
            if g.group in groups and g.weight > 0.0:
                mask[v.index] = True
                break
    return mask


def apply_cuffs(parts, arm_obj):
    """Squeezes each winning cuff's target meshes. Returns [(cuff, object, moved,
    triangles removed)]."""
    arm_inv = np.array(arm_obj.matrix_world.inverted())
    coords, hidden = {}, {}
    results = []
    for _cat, name, cu, ignore in winning_cuffs(parts):
        bones = _subtree(arm_obj, cu['bone'])
        if not bones:
            _log(f"    cuff {name}: bone '{cu['bone']}' not in the armature")
            continue
        targets = [o for c, _f, objs in parts if c == cu['category'] for o in objs
                   if o.type == 'MESH' and _base(o.name) not in ignore
                   and _stem(_base(o.name)) not in ignore]
        for obj in targets:
            me = obj.data
            n = len(me.vertices)
            if not n:
                continue
            mask = _vertex_mask(obj, bones)
            if not mask.any():
                continue
            co = coords.get(obj.name)
            if co is None:
                co = np.empty(n * 3)
                src = me.shape_keys.reference_key.data if me.shape_keys else me.vertices
                src.foreach_get('co', co)
                co = co.reshape(n, 3)
            to_arm = arm_inv @ np.array(obj.matrix_world)
            p = (np.c_[co, np.ones(n)] @ to_arm.T)[:, :3]
            moved, count, below = squeeze(p, mask, cu)
            coords[obj.name] = (np.c_[moved, np.ones(n)] @ np.linalg.inv(to_arm).T)[:, :3]
            removed = 0
            if below.any():
                h = hidden.setdefault(obj.name, np.zeros(len(me.polygons), bool))
                for poly in me.polygons:
                    if all(below[i] for i in poly.vertices):
                        if not h[poly.index]:
                            removed += 1
                        h[poly.index] = True
            results.append((name, obj.name, count, removed))

    for obj_name, co in coords.items():
        obj = bpy.data.objects[obj_name]
        if obj.data.shape_keys is None:
            obj.shape_key_add(name="Basis", from_mix=False)
        kb = obj.data.shape_keys.key_blocks.get(CUFF_KEY) or obj.shape_key_add(
            name=CUFF_KEY, from_mix=False)
        kb.data.foreach_set('co', co.ravel())
        kb.value = 1.0
        obj.data.update()
    for obj_name, h in hidden.items():
        _mask_faces(bpy.data.objects[obj_name], h)
    return results


def _mask_faces(obj, hidden):
    """Hides the removed triangles: their vertices that touch only removed triangles go
    into a vertex group the Mask modifier drops (faces sharing a vertex with a kept face
    stay - they're inside the boot either way)."""
    me = obj.data
    keep = np.zeros(len(me.vertices), bool)
    for poly in me.polygons:
        if not hidden[poly.index]:
            keep[list(poly.vertices)] = True
    drop = [i for i in range(len(me.vertices)) if not keep[i]]
    if not drop:
        return
    vg = obj.vertex_groups.get(HIDDEN_GROUP) or obj.vertex_groups.new(name=HIDDEN_GROUP)
    vg.add(drop, 1.0, 'REPLACE')
    mod = obj.modifiers.get(MASK_NAME)
    if mod is None:
        mod = obj.modifiers.new(MASK_NAME, 'MASK')
        try:                                   # first: drop the faces before skinning
            obj.modifiers.move(obj.modifiers.find(mod.name), 0)
        except (AttributeError, RuntimeError, TypeError):
            pass
    mod.mode = 'VERTEX_GROUP'
    mod.vertex_group = HIDDEN_GROUP
    mod.invert_vertex_group = True
