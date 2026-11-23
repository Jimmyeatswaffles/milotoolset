"""
Rock Band 3 face sculpting (the character creator's face types and sliders).

The game reshapes only head.mesh (char/main/head/<gender>/head.milo), from the head-shape
milo named by BandCharDesc's head_male_path / head_female_path (head_male.milo,
head_female.milo): one MeshAnim 'base.msnm' (the neutral head, 2592 vertices male, 2999
female) and a BandFaceDeform per feature - chin, eye, jaw, mouth, nose, shape .fdm.

===========================================================================================
FORMAT - verified on both retail head-shape milos (every frame parses to the byte)
===========================================================================================
BandFaceDeform: 17-byte object header, u32 frame count, then per frame u32 byte size and
that many bytes of runs: u16 first vertex, u16 count, count x 3 signed bytes. An offset is
byte x 1/127 units (the constant AddFrame loads, 0x8201adc0 = 0.015748031). Vertex numbers
index base.msnm. Frame 0 is unused - AddFrame reads frame n + 1.

===========================================================================================
BLENDING - recovered from the retail Xbox 360 executable
===========================================================================================
BandCharDesc::Head::SetShape (0x82321938) calls, in this order:

    AddDegrees("nose",  nose,  [width, height])
    AddDegrees("mouth", mouth, [width, height])
    AddDegrees("eye",   eye,   [separation, height, rotation])
    AddFrame  ("shape", shape, 1.0)
    AddDegrees("jaw",   0,     [width, height])
    AddDegrees("chin",  chin,  [width, height])
    AddDegrees("brow",  0,     [separation, height])     - no brow.fdm in retail: no-op

AddDegrees(name, type, sliders, n) (0x8229e2b0): a type owns 2n + 1 frames starting at
type * (2n + 1). For slider k (helper 0x8229e228): below 0.5 it adds frame base + 1 + 2k
at weight 1 - 2s, otherwise frame base + 2 + 2k at weight 2s - 1; whatever weight is left
of 1 goes to the type's own frame, base. So a feature type holds 5 frames (2 sliders) or 7
(the eye's 3): nose 18 types, mouth 12, eye 18, chin 9, jaw 1, and shape 18 single frames.

BandHeadShaper::Start first sets head.mesh's vertices to base.msnm, matched up once by
nearest position (SetMeshAnim's greedy search, reproduced here), then each AddFrame adds
its offsets. AddFrame would also blend bone poses from a CharClip of the feature's name;
the retail head-shape milos have none, so only the vertices change.

In Blender the result goes into an 'RB3 Face' shape key on the head, recomputed whenever
the face settings on the armature change.
"""

import math
import os
import struct

import bpy
from bpy.props import FloatProperty, IntProperty, PointerProperty, StringProperty
from bpy_extras.io_utils import ImportHelper

from .io import read_milo_container_body
from .utilities import _log


SCALE = 0.015748031437397003
SHAPE_KEY = "RB3 Face"

# (feature, type attribute or None (type 0), slider attributes) in SetShape's order
DEGREES = (
    ('nose', 'nose', ('nose_width', 'nose_height')),
    ('mouth', 'mouth', ('mouth_width', 'mouth_height')),
    ('eye', 'eye', ('eye_separation', 'eye_height', 'eye_rotation')),
    ('jaw', None, ('jaw_width', 'jaw_height')),
    ('chin', 'chin', ('chin_width', 'chin_height')),
    ('brow', None, ('brow_separation', 'brow_height')),
)
TYPE_FEATURES = ('shape', 'chin', 'nose', 'eye', 'mouth')
SLIDERS = tuple(a for _f, _t, sl in DEGREES for a in sl)
FIELDS = TYPE_FEATURES + SLIDERS


class FaceError(Exception):
    pass


# ---------------------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------------------

def _u32(b, o):
    return struct.unpack_from('>I', b, o)[0]


def _parse_fdm(b):
    """[frame -> (vertex indices, offsets N x 3)] (offsets already scaled)."""
    import numpy as np
    n = _u32(b, 17)
    p = 21
    frames = []
    for _ in range(n):
        size = _u32(b, p); p += 4
        fr = b[p:p + size]; p += size
        idx, off = [], []
        q = 0
        while q < len(fr):
            start, cnt = struct.unpack_from('>HH', fr, q)
            d = np.frombuffer(fr, dtype=np.int8, count=cnt * 3, offset=q + 4)
            idx.append(np.arange(start, start + cnt))
            off.append(d.reshape(cnt, 3).astype(np.float64) * SCALE)
            q += 4 + cnt * 3
        if q != len(fr):
            raise FaceError("a face-deform frame doesn't end on a run boundary")
        frames.append((np.concatenate(idx) if idx else np.zeros(0, dtype=np.int64),
                       np.concatenate(off) if off else np.zeros((0, 3))))
    return frames


def _parse_msnm_base(b):
    """The first key of a MeshAnim's vertex positions (N x 3)."""
    import numpy as np
    for o in [37] + list(range(17, 96)):
        n = _u32(b, o)
        if 100 <= n <= 50000 and o + 4 + n * 12 <= len(b):
            v = np.array(struct.unpack_from(f'>{n * 3}f', b, o + 4)).reshape(n, 3)
            if np.isfinite(v).all() and np.abs(v).max() < 10000:
                return v
    raise FaceError("couldn't find base.msnm's vertices")


def read_shaper(path):
    """{'base': N x 3, 'features': {name: [frames]}} from a head-shape milo."""
    from .mesh_importer import walk_milo_entries
    with open(path, 'rb') as f:
        body = read_milo_container_body(f.read())
    base, feats = None, {}
    for _d, _p, etype, ename, s, e in walk_milo_entries(body):
        if etype == 'MeshAnim' and ename == 'base.msnm':
            base = _parse_msnm_base(body[s:e])
        elif etype == 'BandFaceDeform':
            feats[ename.rsplit('.', 1)[0]] = _parse_fdm(body[s:e])
    if base is None or not feats:
        raise FaceError(f"{os.path.basename(path)} has no base.msnm / .fdm entries - is it "
                        f"a head-shape milo (head_male / head_female)?")
    return dict(base=base, features=feats)


_CACHE = {}


def shaper(path):
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    hit = _CACHE.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    data = read_shaper(path)
    _CACHE[path] = (mtime, data)
    return data


def type_counts(data):
    """{feature: number of types} as BandHeadShaper::Init counts them."""
    per = {'nose': 5, 'mouth': 5, 'eye': 7, 'chin': 5, 'jaw': 5, 'shape': 1}
    return {k: (len(v) - 1) // per[k] for k, v in data['features'].items() if k in per}


# ---------------------------------------------------------------------------------------
# Blending
# ---------------------------------------------------------------------------------------

def frame_weights(head):
    """[(feature, frame number as AddFrame takes it, weight)] for a head dict with the
    BandCharDesc::Head field names."""
    out = []
    for feat, type_attr, sliders in DEGREES:
        t = int(head.get(type_attr, 0)) if type_attr else 0
        n = len(sliders)
        base = t * (2 * n + 1)
        rest = 1.0
        for k, attr in enumerate(sliders):
            s = float(head.get(attr, 0.5))
            if s < 0.5:
                w, fr = 1.0 - 2.0 * s, base + 1 + 2 * k
            else:
                w, fr = (s - 0.5) * 2.0, base + 2 + 2 * k
            out.append((feat, fr, w))
            rest -= w
        out.append((feat, base, rest))
        if feat == 'eye':
            out.append(('shape', int(head.get('shape', 0)), 1.0))   # SetShape's order
    return out


def compute_face(data, head):
    """The sculpted head as base.msnm-ordered positions (N x 3)."""
    v = data['base'].copy()
    for feat, fr, w in frame_weights(head):
        frames = data['features'].get(feat)
        if not frames or not w:
            continue
        k = fr + 1                                  # AddFrame reads frame n + 1
        if not 0 <= k < len(frames):
            continue
        idx, off = frames[k]
        ok = idx < len(v)
        v[idx[ok]] += w * off[ok]
    return v


def greedy_mapping(base, verts):
    """SetMeshAnim's matching: base vertex i -> head.mesh vertex mapping[i], each taken
    greedily as the nearest not yet claimed."""
    import numpy as np
    n = len(base)
    if len(verts) != n:
        raise FaceError(f"head mesh has {len(verts)} vertices, the head-shape milo "
                        f"{n} - they must match")
    vec = np.arange(n)
    for i in range(n):
        d = ((verts[vec[i:]] - base[i]) ** 2).sum(1)
        j = i + int(np.argmin(d))
        vec[i], vec[j] = vec[j], vec[i]
    return vec


# ---------------------------------------------------------------------------------------
# Blender
# ---------------------------------------------------------------------------------------

_MAP_KEY = "rb3_face_mapping"


def head_objects(arm_obj, n_verts):
    out = []
    for o in bpy.data.objects:
        if o.type != 'MESH' or len(o.data.vertices) != n_verts:
            continue
        if not any(m.type == 'ARMATURE' and m.object is arm_obj for m in o.modifiers):
            continue
        if o.get("rb3_part") == 'head' or o.data.name.startswith('head'):
            out.append(o)
    return out


def _mapping_for(obj, base):
    import numpy as np
    raw = obj.get(_MAP_KEY)
    if raw is not None and len(raw) == len(base):
        return np.array(raw, dtype=np.int64)
    me = obj.data
    keys = me.shape_keys
    co = np.empty(len(me.vertices) * 3)
    (keys.reference_key.data if keys else me.vertices).foreach_get('co', co)
    m = greedy_mapping(base, co.reshape(-1, 3))
    obj[_MAP_KEY] = [int(x) for x in m]
    return m


def _head_dict(fs):
    return {k: getattr(fs, k) for k in FIELDS}


def apply_face(arm_obj, report=False):
    """Writes the 'RB3 Face' shape key on the armature's head mesh(es). Returns the count."""
    import numpy as np
    fs = arm_obj.rb3_face
    path = bpy.path.abspath(fs.source) if fs.source else ""
    if not path:
        return 0
    try:
        data = shaper(path)
    except Exception as e:
        _log(f"RB3 face: couldn't read {path}: {e}")
        return 0
    if data is None:
        return 0
    base = data['base']
    face = compute_face(data, _head_dict(fs))
    done = 0
    for obj in head_objects(arm_obj, len(base)):
        try:
            mapping = _mapping_for(obj, base)
        except FaceError as e:
            _log(f"RB3 face: '{obj.name}': {e}")
            continue
        me = obj.data
        if me.shape_keys is None:
            obj.shape_key_add(name="Basis", from_mix=False)
        keys = me.shape_keys
        kb = keys.key_blocks.get(SHAPE_KEY) or obj.shape_key_add(name=SHAPE_KEY,
                                                                 from_mix=False)
        co = np.empty(len(me.vertices) * 3)
        keys.reference_key.data.foreach_get('co', co)
        co = co.reshape(-1, 3)
        co[mapping] = face
        kb.data.foreach_set('co', co.ravel())
        kb.value = 1.0
        me.update()
        done += 1
    try:
        from . import rb3_textures
        types = {f: int(getattr(fs, f)) for f in rb3_textures.HEAD_FEATURES}
        rb3_textures.update_head_normal(arm_obj, arm_obj.rb3_proportions.gender, types)
    except Exception as e:
        _log(f"RB3 face: head normal map not updated ({e})")
    if report:
        _log(f"RB3 face: '{arm_obj.name}' shape {fs.shape}, chin {fs.chin}, nose {fs.nose}, "
             f"eye {fs.eye}, mouth {fs.mouth} -> {done} head mesh(es)")
    return done


def set_from_head(arm_obj, path, head):
    """Store the head-shape milo and a prefab's head values on the armature, then apply."""
    fs = arm_obj.rb3_face
    fs.source = path
    for k in FIELDS:
        if k in head:
            fs[k] = int(head[k]) if k in TYPE_FEATURES else float(head[k])
    return apply_face(arm_obj, report=True)


def _update(self, context):
    obj = self.id_data
    if obj is not None and obj.type == 'ARMATURE':
        apply_face(obj)


def _slider(name, desc):
    return FloatProperty(name=name, min=0.0, max=1.0, default=0.5, update=_update,
                         description=desc)


def _type(name):
    return IntProperty(name=name, min=0, max=63, default=0, update=_update,
                       description=f"The creator's {name.lower()} type (0-based; the "
                                   f"head-shape milo says how many there are)")


class RB3FaceSettings(bpy.types.PropertyGroup):
    shape: _type("Face Shape")
    chin: _type("Chin")
    nose: _type("Nose")
    eye: _type("Eyes")
    mouth: _type("Mouth")
    chin_width: _slider("Chin Width", "Chin width slider")
    chin_height: _slider("Chin Height", "Chin height slider")
    jaw_width: _slider("Jaw Width", "Jaw width slider")
    jaw_height: _slider("Jaw Height", "Jaw height slider")
    nose_width: _slider("Nose Width", "Nose width slider")
    nose_height: _slider("Nose Height", "Nose height slider")
    eye_separation: _slider("Eye Separation", "Eye separation slider")
    eye_height: _slider("Eye Height", "Eye height slider")
    eye_rotation: _slider("Eye Rotation", "Eye rotation (tilt) slider")
    mouth_width: _slider("Mouth Width", "Mouth width slider")
    mouth_height: _slider("Mouth Height", "Mouth height slider")
    brow_separation: _slider("Brow Separation", "Brow separation slider (the retail "
                                                "head-shape milos have no brow data)")
    brow_height: _slider("Brow Height", "Brow height slider (the retail head-shape milos "
                                        "have no brow data)")
    source: StringProperty(name="Head-Shape Milo", subtype='FILE_PATH')


class POSE_OT_rb3_load_face(bpy.types.Operator, ImportHelper):
    """Load Rock Band 3's head-shape milo (head_male / head_female) for this armature's
    head and apply the face settings"""
    bl_idname = "pose.rb3_load_face"
    bl_label = "Load RB3 Head-Shape Milo"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".milo_xbox"
    filter_glob: StringProperty(default="*.milo_xbox;*.milo_ps3;*.milo", options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == 'ARMATURE'

    def execute(self, context):
        arm = context.object
        try:
            data = read_shaper(self.filepath)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        arm.rb3_face.source = self.filepath
        _CACHE.pop(self.filepath, None)
        n = apply_face(arm, report=True)
        if not n:
            self.report({'WARNING'}, f"No head mesh with {len(data['base'])} vertices on "
                                     f"this armature")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Face applied to {n} head mesh(es)")
        return {'FINISHED'}


class POSE_OT_rb3_reset_face(bpy.types.Operator):
    """Set every face slider back to the middle (types are kept)"""
    bl_idname = "pose.rb3_reset_face"
    bl_label = "Center Face Sliders"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == 'ARMATURE'

    def execute(self, context):
        fs = context.object.rb3_face
        for k in SLIDERS:
            fs[k] = 0.5
        apply_face(context.object)
        return {'FINISHED'}


class DATA_PT_rb3_face(bpy.types.Panel):
    bl_label = "Rock Band 3 Face"
    bl_idname = "DATA_PT_rb3_face"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "data"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == 'ARMATURE'

    def draw(self, context):
        layout = self.layout
        arm = context.object
        fs = arm.rb3_face
        layout.operator(POSE_OT_rb3_load_face.bl_idname,
                        text="Load Head-Shape Milo" if not fs.source else
                        "Reload Head-Shape Milo", icon='SHAPEKEY_DATA')
        if not fs.source:
            layout.label(text="Not in use - load head_male / head_female", icon='INFO')
            return
        layout.label(text=bpy.path.basename(fs.source), icon='FILE')
        counts = {}
        try:
            data = shaper(bpy.path.abspath(fs.source))
            counts = type_counts(data) if data else {}
            n_heads = len(head_objects(arm, len(data['base']))) if data else 0
        except Exception:
            n_heads = 0
        if not n_heads:
            layout.label(text="No matching head mesh on this armature", icon='ERROR')

        def row(attr, feat):
            r = layout.row(align=True)
            r.prop(fs, attr)
            if feat in counts:
                r.label(text=f"of {counts[feat]}")
                if getattr(fs, attr) >= counts[feat]:
                    layout.label(text=f"{feat}: only {counts[feat]} types - not applied",
                                 icon='ERROR')

        row("shape", 'shape')
        layout.separator()
        for feat, type_attr, sliders in DEGREES:
            if feat == 'brow':
                continue
            box = layout.box()
            col = box.column(align=True)
            if type_attr:
                r = col.row(align=True)
                r.prop(fs, type_attr)
                if feat in counts:
                    r.label(text=f"of {counts[feat]}")
            else:
                col.label(text=feat.capitalize())
            for s in sliders:
                col.prop(fs, s, slider=True)
        layout.operator(POSE_OT_rb3_reset_face.bl_idname, icon='LOOP_BACK')


_classes = (RB3FaceSettings, POSE_OT_rb3_load_face, POSE_OT_rb3_reset_face, DATA_PT_rb3_face)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.rb3_face = PointerProperty(type=RB3FaceSettings)


def unregister():
    del bpy.types.Object.rb3_face
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
