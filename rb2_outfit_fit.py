"""
Rock Band 2 outfit fitting: hiding and cuffing (EXPERIMENTAL).

Two systems keep stacked outfit parts from poking through each other. Both live in an
item's resource milo; the logic comes from the RB3 decomp (src/system/char), with the RB2
symbol dump (doc/rb2_dump/rockband2) confirming RB2 has the same classes.

CharMeshHide (exact)
    Each item may carry one: its own flags - a bitfield of body areas it covers - and a list
    of hides, each a drawable of that item plus flags. The game ORs every item's flags into
    one value, then hides each listed drawable whose flags share a bit with it
    (CharMeshHide::HideAll / HideDraws). RB2's Hide has no 'show' field (RB3 added it), so it
    is purely 'hide when covered'. Layout: revision, the 13-byte object header, u32 flags,
    u32 count, then per hide a symbol (the drawable) and u32 flags (+ a bool from revision 2).
    kissboots' boot.hide: flags 0x500, no hides - it only announces what it covers; nothing
    in kissflares listens for those bits.

CharCuff (reconstructed)
    "A cuff used to constrain colliding outfits against each other, for example boots
    against pants. The widest cuff wins." A cuff is a transform on a bone with a three-ring
    profile in its own frame - rings at (sin h, cos h, offset_i) x radius_i x Eccentricity,
    exactly as CharCuff::Highlight draws them, Eccentricity narrowing the radius along local
    Y - plus an outer radius. Among cuffs on the same bone the largest outer radius wins (ties:
    the greater name), and squeezes the meshes of its target category weighted to that bone or
    the bones below it, except any cuff's ignore list. kissboots' cuffs (outer 5.5, category
    legs) beat kissflares' (5.0, category feet) on bone_L/R-thigh, so the boots squeeze the
    flared legs inside them.

    The squeeze itself is missing from both sources (RB3's CharCuff::DeformMesh is a stub), so
    it is reconstructed: a vertex's distance from the cuff's axis is limited to the profile
    radius at its position along the axis, interpolated between the rings; past the outer ring
    nothing is limited. Past the inner ring of a closed cuff the inner radius holds, measured
    from the LEG rather than the cuff's axis: kissboots' cuff sits at the knee with its axis
    along the thigh, and that line, carried on down, drifts in front of the shin - several
    units off at the ankle - so pulling the pants onto it left blades sticking out of the
    boots. Deeper vertices are pulled toward the path from the inner ring down the cuff bone's
    chain (knee -> ankle -> toe in the armature's rest pose) instead. The cuff's
    negative side is the inside of its item - kissboots' axis points up the leg from knee
    height, kissflares' down from the hem - and measured that way kissflares' leg runs ~3-4
    units out inside the boot against a profile of 1.0. The result goes into an 'RB2 Cuff'
    shape key so it can be switched off.
"""

import json
import re
import struct

import bpy
import numpy as np

from .io import read_milo_container_body
from .utilities import _log


class FitError(Exception):
    pass


def _u32(c, o):
    return struct.unpack_from('>I', c, o)[0]


def _sym(c, o, max_len=256):
    n = _u32(c, o)
    if n > max_len or o + 4 + n > len(c):
        raise FitError(f"implausible symbol length {n}")
    raw = c[o + 4:o + 4 + n]
    if not all(32 <= x < 127 for x in raw):
        raise FitError("non-text symbol")
    return raw.decode('latin-1'), o + 4 + n


def _stem(name):
    for ext in ('.mesh', '.trans'):
        if name.endswith(ext):
            return name[:-len(ext)]
    return name


def parse_meshhide(c):
    """{'flags', 'hides': [(drawable, flags)]}."""
    rev = _u32(c, 0)
    o = 17
    flags = _u32(c, o); count = _u32(c, o + 4); o += 8
    if count > 4096:
        raise FitError(f"implausible hide count {count}")
    hides = []
    for _ in range(count):
        draw, o = _sym(c, o)
        hf = _u32(c, o); o += 4
        show = False
        if rev > 1:
            show = bool(c[o]); o += 1
        hides.append((draw, hf, show))
    return dict(flags=flags, hides=hides)


def parse_cuff(c):
    """A CharCuff revision 8: its world matrix (row-vector 4x3 -> 4x4) and fields. The
    Trans part ends with its constraint data, so the cuff's own fields are found as the run
    that reads cleanly: six floats (radius, offset x3), outer radius, open flag, bone, ...;
    the bone must be a bone_* name."""
    if _u32(c, 0) != 8:
        raise FitError(f"CharCuff revision {_u32(c, 0)} isn't handled")
    m = struct.unpack_from('>12f', c, 21 + 48)              # world (after local)
    world = np.eye(4)
    world[:3, :3] = np.array(m[:9]).reshape(3, 3)
    world[3, :3] = m[9:]
    for o in range(117, min(len(c) - 40, 200)):
        try:
            vals = struct.unpack_from('>7f', c, o)
            bone, q = _sym(c, o + 29)
            if not bone.startswith('bone_') or not all(0 < vals[i] < 100 for i in (0, 2, 4, 6)):
                continue
            ecc = struct.unpack_from('>f', c, q)[0]
            cat, q = _sym(c, q + 4)
            n = _u32(c, q); q += 4
            ignore = []
            for _ in range(n):
                s, q = _sym(c, q)
                ignore.append(_stem(s))
            if not 0 < ecc <= 1.5:
                continue
            return dict(world=world.tolist(),
                        shape=[(vals[0], vals[1]), (vals[2], vals[3]), (vals[4], vals[5])],
                        outer=vals[6], open=bool(c[o + 28]), bone=_stem(bone), ecc=ecc,
                        category=cat, ignore=ignore)
        except (FitError, struct.error):
            continue
    raise FitError("couldn't read the cuff's fields")


def read_fit(filepath):
    """{'hides': [meshhide], 'cuffs': [(name, cuff)]} from an item's milo."""
    from .rb2_creator_textures import walk_milo
    with open(filepath, 'rb') as f:
        body = read_milo_container_body(f.read())
    objs, _metas = walk_milo(body)
    out = dict(hides=[], cuffs=[])
    for _d, t, n, a, z in objs:
        try:
            if t == 'CharMeshHide':
                out['hides'].append(dict(parse_meshhide(body[a:z]), name=n))
            elif t == 'CharCuff':
                out['cuffs'].append((n, parse_cuff(body[a:z])))
        except (FitError, struct.error) as e:
            _log(f"    {t} '{n}' skipped: {e}")
    return out


# ---------------------------------------------------------------------------------------
# Applying to a Blender character
# ---------------------------------------------------------------------------------------

CUFF_KEY = "RB2 Cuff"
_BLENDER_DUP = re.compile(r'\.\d{3}$')


def apply_hides(parts):
    """parts: [(category, fit dict, [objects])]. Hides the objects the combined flags
    cover. Returns [(object name, category, flags)]."""
    combined = 0
    for _cat, fit, _objs in parts:
        for h in fit['hides']:
            combined |= h['flags']
    hidden = []
    if not combined:
        return hidden, combined
    for cat, fit, objs in parts:
        by_name = {}
        for o in objs:
            by_name.setdefault(_BLENDER_DUP.sub('', o.name), []).append(o)
        for h in fit['hides']:
            for draw, flags, show in h['hides']:
                covered = bool(combined & flags)
                if covered == show:
                    continue                      # RB2: show is always False
                for o in by_name.get(draw, ()) + by_name.get(_stem(draw), ()):
                    o.hide_set(True)
                    o.hide_render = True
                    o["rb2_hidden_by_flags"] = int(combined & flags)
                    hidden.append((o.name, cat, combined & flags))
    return hidden, combined


def _subtree(stem, hierarchy, arm_obj):
    """Bone stems at or below `stem` - the game's hierarchy when the armature carries it
    (from the deform milo), otherwise the armature's own."""
    kids = {}
    if hierarchy:
        for b, (parent, _m) in hierarchy.items():
            kids.setdefault(parent, []).append(b)
    else:
        for b in arm_obj.data.bones:
            if b.parent is not None:
                kids.setdefault(_stem(b.parent.name), []).append(_stem(b.name))
    out, todo = set(), [stem]
    while todo:
        s = todo.pop()
        if s in out or not s.startswith('bone_'):
            continue
        out.add(s)
        todo.extend(kids.get(s, ()))
    return out


def _profile(cuff, z):
    (r0, o0), (r1, o1), (r2, o2) = cuff['shape']
    r = np.interp(z, [o0, o1, o2], [r0, r1, r2])
    limited = z <= o2
    if cuff['open']:
        limited &= z >= o0
    return r, limited


def _closest_on_path(points, path):
    """Closest point on a polyline (M x 3) for each of N points."""
    best = np.repeat(path[:1], len(points), axis=0)
    best_d = np.linalg.norm(points - best, axis=1)
    for a, b in zip(path[:-1], path[1:]):
        ab = b - a
        ln = float(ab @ ab)
        if ln < 1e-12:
            continue
        t = np.clip((points - a) @ ab / ln, 0.0, 1.0)
        c = a + t[:, None] * ab
        d = np.linalg.norm(points - c, axis=1)
        closer = d < best_d
        best[closer] = c[closer]
        best_d[closer] = d[closer]
    return best, best_d


def squeeze(points, weights, cu, chain=None):
    """Armature-space points (N x 3) squeezed by one cuff; weights (N) = how much each point
    belongs to the cuff's bone subtree; chain = armature-space joint positions below the cuff
    (ankle, toe ...) for the deep, closed part. Returns (new points, count squeezed)."""
    world = np.array(cu['world'])                           # row-vector cuff -> armature
    p = np.c_[points, np.ones(len(points))] @ np.linalg.inv(world)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    d = np.hypot(x, y)
    safe = np.where(d > 1e-6, d, 1.0)
    # Eccentricity: the radius shrinks along local Y by `ecc` (CharCuff::Eccentricity)
    e2 = cu['ecc'] ** 2
    den = y * y / e2 + x * x
    ecc = np.sqrt((y * y + x * x) / np.where(den > 1e-12, den, 1.0))
    (r0, o0), _mid, _out = cu['shape']
    r, limited = _profile(cu, z)
    deep = np.zeros(len(z), dtype=bool)
    if not cu['open'] and chain:
        deep = z < o0
        limited = limited & ~deep
    allowed = r * ecc
    wts = np.clip(weights, 0.0, 1.0)
    hit = limited & (d > allowed) & (weights > 1e-4)
    k = np.where(hit, allowed / safe, 1.0)
    k = 1.0 + (k - 1.0) * wts
    p[:, 0] *= k
    p[:, 1] *= k
    out = (p @ world)[:, :3]
    count = int(hit.sum())
    if deep.any():
        # Inside the boot past the inner ring: hold the inner radius around the leg itself.
        start = (np.array([0.0, 0.0, o0, 1.0]) @ world)[:3]
        path = np.vstack([start, np.asarray(chain, dtype=float)])
        pts = points[deep]
        c, dist = _closest_on_path(pts, path)
        over = (dist > r0) & (weights[deep] > 1e-4)
        f = np.where(over, r0 / np.where(dist > 1e-6, dist, 1.0), 1.0)
        f = 1.0 + (f - 1.0) * wts[deep]
        out[deep] = c + (pts - c) * f[:, None]
        count += int(over.sum())
    return out, count


def _leg_chain(arm_obj, bone_stem):
    """Rest-pose heads of the main chain below a bone (for bone_L-thigh: knee, ankle,
    toe), following at each step the bone_* child with the most descendants."""
    by_stem = {_stem(b.name): b for b in arm_obj.data.bones}
    b = by_stem.get(bone_stem)
    out = []

    def size(x):
        return 1 + sum(size(c) for c in x.children)
    while b is not None:
        kids = [c for c in b.children if c.name.startswith('bone_')
                and not any(t in c.name.lower() for t in ('twist', 'target', 'tip'))]
        if not kids:
            break
        b = max(kids, key=size)
        out.append(list(b.matrix_local.to_translation()))
    return out


def apply_cuffs(parts, arm_obj):
    """Squeezes each winning cuff's target meshes into an 'RB2 Cuff' shape key. Returns
    [(cuff, object, vertices moved)]."""
    cuffs = []
    for cat, fit, _objs in parts:
        for name, cu in fit['cuffs']:
            cuffs.append((cat, name, cu))
    winners = []
    for cat, name, cu in cuffs:
        rivals = [(n, c) for _k, n, c in cuffs if c['bone'] == cu['bone'] and c is not cu]
        if any(c['outer'] > cu['outer'] or (c['outer'] == cu['outer'] and n > name)
               for n, c in rivals):
            continue
        ignore = {i for _n, c in rivals for i in c['ignore']} | set(cu['ignore'])
        winners.append((cat, name, cu, ignore))

    hierarchy = None
    raw = arm_obj.get("rb2_deform_data")
    if raw:
        try:
            hierarchy = json.loads(raw).get('hierarchy')
        except ValueError:
            hierarchy = None
    arm_inv = np.array(arm_obj.matrix_world.inverted())
    results = []
    deltas = {}
    for _cat, name, cu, ignore in winners:
        targets = [o for c, _f, objs in parts if c == cu['category'] for o in objs
                   if o.type == 'MESH' and _BLENDER_DUP.sub('', o.name) not in ignore
                   and _stem(_BLENDER_DUP.sub('', o.name)) not in ignore]
        if not targets:
            continue
        bones = _subtree(cu['bone'], hierarchy, arm_obj)
        # Joints below the cuff's position (the knee sits at the cuff, so skip joints above
        # or at its inner ring).
        chain = []
        world = np.array(cu['world'])
        inv = np.linalg.inv(world)
        for j in _leg_chain(arm_obj, cu['bone']):
            zj = (np.array([*j, 1.0]) @ inv)[2]
            if zj < cu['shape'][0][1]:
                chain.append(j)
        for obj in targets:
            me = obj.data
            n = len(me.vertices)
            groups = {vg.index for vg in obj.vertex_groups if _stem(vg.name) in bones}
            if not groups:
                continue
            w = np.zeros(n)
            for v in me.vertices:
                for g in v.groups:
                    if g.group in groups:
                        w[v.index] += g.weight
            w = np.clip(w, 0.0, 1.0)
            if not (w > 1e-4).any():
                continue
            co = deltas.get(obj.name)
            if co is None:
                co = np.empty(n * 3)
                src = me.shape_keys.reference_key.data if me.shape_keys else me.vertices
                src.foreach_get('co', co)
                co = co.reshape(n, 3)
            to_arm = arm_inv @ np.array(obj.matrix_world)
            p_arm = (np.c_[co, np.ones(n)] @ to_arm.T)[:, :3]
            moved, count = squeeze(p_arm, w, cu, chain)
            back = np.c_[moved, np.ones(n)] @ np.linalg.inv(to_arm).T
            deltas[obj.name] = back[:, :3]
            results.append((name, obj.name, count))

    for obj_name, co in deltas.items():
        obj = bpy.data.objects[obj_name]
        if obj.data.shape_keys is None:
            obj.shape_key_add(name="Basis", from_mix=False)
        kb = obj.data.shape_keys.key_blocks.get(CUFF_KEY) or obj.shape_key_add(
            name=CUFF_KEY, from_mix=False)
        kb.data.foreach_set('co', co.ravel())
        kb.value = 1.0
        obj.data.update()
    return results
