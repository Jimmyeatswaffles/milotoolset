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


SHELL_MARGIN = 0.08          # how far inside the squeezing item's inner wall to stay
_Z_BIN = 0.5
FADE = 0.75                  # units over which the squeeze fades out at the cuff's ends
SOFT = 0.75                  # the soft limit starts compressing at this fraction of it


def _cuff_frame(points, world):
    return np.c_[points, np.ones(len(points))] @ np.linalg.inv(world)


def shell_profile(shell_points, cu):
    """The squeezing item's wall along this cuff: (bin heights, distances) - one robust
    value per height (the 15th percentile of the item's own vertices' distance from the
    axis, on this cuff's side), so the limit doesn't jump from vertex to vertex on a
    low-poly boot. None when the item doesn't reach the cuff."""
    if shell_points is None or not len(shell_points):
        return None
    p = _cuff_frame(shell_points, np.array(cu['world']))
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    d = np.hypot(x, y)
    (r0, o0), _m, (r2, o2) = cu['shape']
    reach = max(float(cu.get('outer') or 0.0), r2) * 1.15
    sel = (d < reach) & (z > o0 - 1.0) & (z < o2 + 1.0)
    if sel.sum() < 4:
        return None
    zb = np.floor(z[sel] / _Z_BIN).astype(int)
    heights, walls = [], []
    for k in range(zb.min(), zb.max() + 1):
        m = zb == k
        if m.sum() >= 2:
            heights.append((k + 0.5) * _Z_BIN)
            walls.append(float(np.percentile(d[sel][m], 15)))
    if not heights:
        return None
    walls = np.array(walls)
    if len(walls) >= 3:                       # light smoothing along the height
        walls = np.convolve(np.pad(walls, 1, mode='edge'), [0.25, 0.5, 0.25], 'valid')
    return np.array(heights), walls


def _soft_limit(r, limit):
    """Distances above SOFT x limit are compressed smoothly toward the limit, never past
    it and never out of order - a clamp folds cloth where neighbours fall either side."""
    a0 = SOFT * limit
    span = np.maximum(limit - a0, 1e-6)
    over = np.maximum(r - a0, 0.0)
    return np.where(r > a0, a0 + span * (1.0 - np.exp(-over / span)), r)


def squeeze(points, weights, cu, chain=None, shell=None):
    """Armature-space points (N x 3) squeezed by one cuff; weights (N) = how much each point
    belongs to the cuff's bone subtree; chain = armature-space joint positions below the cuff
    (ankle, toe ...) for the deep, closed part; shell = the squeezing item's own vertices
    (armature space) - its wall. Returns (new points, count moved).

    Within the cuff's length a point's distance from the axis is held under the profile -
    and, where the item's wall is known, just inside that wall. The cuff's numbers alone
    can't serve both items seen: studdedcowboyboots' middle ring (2.4) is wider than its
    own rim (~2.1), so the profile leaves the jeans poking through, while kissboots' outer
    radius (5.5) is so much wider than its profile (2.5) that scaling by the ratio pinches
    the flares inside a wide boot top. The wall fits both. The limit is soft (no folding)
    and fades over the cuff's last FADE units, so the cloth doesn't tear where the squeeze
    starts and stops. Past the inner ring of a closed cuff the inner radius holds, around
    the leg (chain) rather than the cuff's straight axis."""
    world = np.array(cu['world'])                           # row-vector cuff -> armature
    p = _cuff_frame(points, world)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    d = np.hypot(x, y)
    safe = np.where(d > 1e-6, d, 1.0)
    # Eccentricity: the radius shrinks along local Y by `ecc` (CharCuff::Eccentricity)
    e2 = cu['ecc'] ** 2
    den = y * y / e2 + x * x
    ecc = np.sqrt((y * y + x * x) / np.where(den > 1e-12, den, 1.0))
    (r0, o0), _mid, (_r2, o2) = cu['shape']
    r, limited = _profile(cu, z)
    deep = np.zeros(len(z), dtype=bool)
    if not cu['open'] and chain:
        deep = z < o0                 # profile squeeze too, then handed over to the leg
    allowed = r.copy()
    wall = shell_profile(shell, cu)
    if wall is not None:
        heights, walls = wall
        w_at = np.interp(z, heights, walls, left=np.inf, right=np.inf)
        allowed = np.minimum(allowed, np.maximum(w_at - SHELL_MARGIN, 0.05))
    allowed = allowed * ecc
    wts = np.clip(weights, 0.0, 1.0)
    # fade in from the cuff's top end (o2) downward
    fade = np.clip((o2 - z) / FADE, 0.0, 1.0)
    target = _soft_limit(d, allowed)
    new_d = d + (target - d) * fade * wts
    moved = limited & (weights > 1e-4) & (new_d < d - 1e-5)
    k = np.where(moved, new_d / safe, 1.0)
    p[:, 0] *= k
    p[:, 1] *= k
    out = (p @ world)[:, :3]
    count = int(moved.sum())
    if deep.any():
        # Inside the boot past the inner ring: hold the inner radius around the leg itself,
        # blending in over FADE below the ring so the two regions meet smoothly.
        start = (np.array([0.0, 0.0, o0, 1.0]) @ world)[:3]
        path = np.vstack([start, np.asarray(chain, dtype=float)])
        pts = points[deep]
        c, dist = _closest_on_path(pts, path)
        tgt = _soft_limit(dist, np.full(len(dist), r0))
        blend = np.clip((o0 - z[deep]) / FADE, 0.0, 1.0)
        # what the profile part would give at the ring, faded into the leg-following hold
        f = np.where(dist > 1e-6, tgt / np.where(dist > 1e-6, dist, 1.0), 1.0)
        f = 1.0 + (f - 1.0) * wts[deep]
        deep_pts = c + (pts - c) * f[:, None]
        out[deep] = out[deep] * (1 - blend[:, None]) + deep_pts * blend[:, None]
        count += int(((f < 1.0) & (weights[deep] > 1e-4)).sum())
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
    def shell_of(category):
        """The squeezing item's own vertices, in armature space."""
        pts = []
        for c, _f, objs in parts:
            if c != category:
                continue
            for o in objs:
                if o.type != 'MESH' or not len(o.data.vertices):
                    continue
                me = o.data
                co = np.empty(len(me.vertices) * 3)
                src = me.shape_keys.reference_key.data if me.shape_keys else me.vertices
                src.foreach_get('co', co)
                m = arm_inv @ np.array(o.matrix_world)
                pts.append((np.c_[co.reshape(-1, 3), np.ones(len(me.vertices))] @ m.T)[:, :3])
        return np.vstack(pts) if pts else None
    shells = {}

    for _cat, name, cu, ignore in winners:
        if _cat not in shells:
            shells[_cat] = shell_of(_cat)
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
            moved, count = squeeze(p_arm, w, cu, chain, shells.get(_cat))
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
