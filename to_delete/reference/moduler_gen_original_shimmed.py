import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
# import os
import cv2

import trimesh
# import pyrender
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter
import random


def get_material_index(name):
    # bus
    if name == 'LCP': return 1
    if name == 'Thermalbright_N': return 2
    if name == 'Melinex': return 3
    if name == 'Mylar': return 4
    if name == 'CORIN_XLS': return 5
    if name == 'Kapton-MT': return 6
    if name == 'Kapton-HN': return 7
    if name == 'Solar_cell': return 8
    if name == 'Black paint': return 9
    if name == 'Kapton_CRC': return 10
    if name == 'Kapton_CS)': return 11
    if name == 'Kapton-WS-glossy': return 12
    if name == 'Kapton-FMT': return 13
    if name == 'Kapton-PV9101': return 14
    if name == 'Kapton-PV9103': return 15
    if name == 'Mylar_aluminized': return 16
    if name == 'Aluminum_Foil': return 17
    if name == 'Kapton HN 1 mil': return 18
    if name == 'AZW-LAII': return 19
    if name == 'AZ-2000': return 20
    if name == 'AZ-400': return 21
    if name == 'AZJ-4020': return 22
    if name == 'AZ-3700': return 23
    if name == 'AZ-2100': return 24
    if name == 'AZ-93': return 25
    if name == 'Stainless_steel_1': return 26
    if name == 'Stainless_Steel_2': return 27
    if name == 'BetaCloth': return 28
    if name == 'Aluminum 7075T6': return 29
    if name == 'Aluminum 6061': return 30

    # wings
    if name == 'Fused_silica': return 31
    if name == 'CMX': return 32
    if name == '0214': return 33
    if name == 'Glass_Fiber': return 34
    if name == 'CMG_with_Al': return 35
    if name == 'CMG_with_Cu': return 36
    if name == 'Si solar cell': return 37
    if name == 'Ge solar cell': return 38

    # antenna
    if name == 'GFRP': return 39
    if name == 'CFRP': return 40 
    if name == 'Elgiloy': return 41
    if name == 'Aluminum': return 42
    if name == 'Titanium': return 43
    if name == 'Copper_foil': return 44
    if name == 'Copper': return 45
    if name == 'Gold_foil': return 46

    return None

def get_material_name(idx):
    # bus
    if idx == 1: return 'LCP'
    if idx == 2: return 'Thermalbright_N'
    if idx == 3: return 'Melinex'
    if idx == 4: return 'Mylar'
    if idx == 5: return 'CORIN_XLS'
    if idx == 6: return 'Kapton-MT'
    if idx == 7: return 'Kapton-HN'
    if idx == 8: return 'Solar_cell'
    if idx == 9: return 'Black paint'
    if idx == 10: return 'Kapton_CRC'
    if idx == 11: return 'Kapton_CS)'
    if idx == 12: return 'Kapton-WS-glossy'
    if idx == 13: return 'Kapton-FMT'
    if idx == 14: return 'Kapton-PV9101'
    if idx == 15: return 'Kapton-PV9103'
    if idx == 16: return 'Mylar_aluminized'
    if idx == 17: return 'Aluminum_Foil'
    if idx == 18: return 'Kapton HN 1 mil'
    if idx == 19: return 'AZW-LAII'
    if idx == 20: return 'AZ-2000'
    if idx == 21: return 'AZ-400'
    if idx == 22: return 'AZJ-4020'
    if idx == 23: return 'AZ-3700'
    if idx == 24: return 'AZ-2100'
    if idx == 25: return 'AZ-93'
    if idx == 26: return 'Stainless_steel_1'
    if idx == 27: return 'Stainless_Steel_2'
    if idx == 28: return 'BetaCloth'
    if idx == 29: return 'Aluminum 7075T6'
    if idx == 30: return 'Aluminum 6061'

    # wings
    if idx == 31: return 'Fused_silica'
    if idx == 32: return 'CMX'
    if idx == 33: return '0214'
    if idx == 34: return 'Glass_Fiber'
    if idx == 35: return 'CMG_with_Al'
    if idx == 36: return 'CMG_with_Cu'
    if idx == 37: return 'Si solar cell'
    if idx == 38: return 'Ge solar cell'

    # antenna
    if idx == 39: return 'GFRP'
    if idx == 40: return 'CFRP'   # ⚠️ note trailing space in your dict
    if idx == 41: return 'Elgiloy'
    if idx == 42: return 'Aluminum'
    if idx == 43: return 'Titanium'
    if idx == 44: return 'Copper_foil'
    if idx == 45: return 'Copper'
    if idx == 46: return 'Gold_foil'

    return None

def generate_material(n=18, end=1000):
    """
    n: target sample count
    end: target wavelength upper bound (nm)

    Behavior change:
      For target wavelengths > max(measured λ), values are set to 0.
      (i.e., np.interp(..., right=0.0))
    """
    target = np.linspace(350, end, n)  # uniform wavelength grid
    material_dict = {"bus": {}, "wings": {}, "antenna": {}}

    # load all sheets
    bus_material = pd.read_excel('/home/claude/rebuild/data/material_refined/Bus.xlsx',
                                 sheet_name=None)
    wing_material = pd.read_excel('/home/claude/rebuild/data/material_refined/Wings.xlsx',
                                  sheet_name=None)
    antenna_material = pd.read_excel('/home/claude/rebuild/data/material_refined/antenna.xlsx',
                                     sheet_name=None)

    datasets = {"bus": bus_material, "wings": wing_material, "antenna": antenna_material}

    for category, sheets in datasets.items():
        for sheet_name, df in sheets.items():
            if df is None or df.empty:
                continue

            # keep up to first 3 columns (λ, pristine, irradiated?) but don't require 3rd to exist
            df = df.iloc[:, :3].copy()

            # coerce numeric
            for c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

            # require only the first two columns to be valid (λ & pristine)
            df = df.dropna(subset=[df.columns[0], df.columns[1]])
            if df.empty:
                continue

            # arrays
            x = df.iloc[:, 0].to_numpy(dtype=float)
            y_pristine = df.iloc[:, 1].to_numpy(dtype=float)

            # sort by wavelength
            order = np.argsort(x)
            x = x[order]
            y_pristine = y_pristine[order]

            # handle degenerate single-point case
            if x.size == 1:
                # <= single λ: use that value; > single λ: 0
                y_pristine_interp = np.where(target <= x[0], y_pristine[0], 0.0)
            else:
                # interpolate; clip right side to 0
                y_pristine_interp = np.interp(
                    target, x, y_pristine,
                    left=y_pristine[0],
                    right=0.0
                )

            entry = {
                "wavelength": target.copy(),
                "pristine": y_pristine_interp
            }

            # optional irradiated column
            has_irr = df.shape[1] >= 3
            if has_irr:
                y_irr_raw = df.iloc[:, 2].to_numpy(dtype=float)[order]
                valid = np.isfinite(y_irr_raw) & np.isfinite(x)
                if valid.sum() >= 2:
                    xi = x[valid]
                    yi = y_irr_raw[valid]
                    # clip right side to 0
                    y_irr_interp = np.interp(
                        target, xi, yi,
                        left=yi[0],
                        right=0.0
                    )
                    entry["irradiated"] = y_irr_interp
                elif valid.sum() == 1:
                    # one valid point: <= that λ use its value, > that λ set 0
                    xv, yv = x[valid][0], y_irr_raw[valid][0]
                    entry["irradiated"] = np.where(target <= xv, yv, 0.0)

            material_dict[category][sheet_name] = entry

    return material_dict

def generate_distinct_color_list(materials: dict):
#     color_list = []
    seen_colors = set()
    color_to_material = {cat: {} for cat in materials.keys()}

    for category, subdict in materials.items():
        for sheet in subdict.keys():
            # generate unique color
            while True:
                r, g, b = np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)
                color_key = (r, g, b)
                if color_key not in seen_colors:
                    seen_colors.add(color_key)
                    break

#             rgba = [r, g, b, 255]
#             color_list.append(rgba)

            # nested dict same shape as materials
            color_to_material[category][color_key] = sheet

    return color_to_material


def get_intrinsics(fov_y, image_size):
    f = (0.5 * image_size) / np.tan(fov_y / 2)
    cx, cy = image_size / 2, image_size / 2
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

def rotate_mesh_vertices(mesh, rotation_matrix):
    rotated_vertices = (rotation_matrix @ mesh.vertices.T).T
    mesh_rotated = mesh.copy()
    mesh_rotated.vertices = rotated_vertices
    return mesh_rotated

def project_vertices(vertices, camera_pose, intrinsics):
    verts_hom = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
    verts_cam = (camera_pose @ verts_hom.T).T[:, :3]
    verts_proj = (intrinsics @ verts_cam.T).T
    verts_proj[:, 0] /= verts_proj[:, 2]
    verts_proj[:, 1] /= verts_proj[:, 2]
    return verts_proj[:, :2]

def rasterize_components_with_depth(components, image_size=256, camera_pos=[0,0,25], angles=(0, 0, 0)):
    depth_buffer = np.full((image_size, image_size), np.inf)
    material_mask = np.zeros((image_size, image_size), dtype=np.uint8)
    
    rotation = R.from_euler('xyz', angles, degrees=True)
    rotation_matrix = rotation.as_matrix()

    camera_pose = np.eye(4)
    # camera_pose[:3, 3] = [0, 0, camera_h]
    camera_pose[:3, 3] = camera_pos
    intrinsics = get_intrinsics(np.radians(60), image_size)

    for mesh, material_id in components:
        rotated_mesh = rotate_mesh_vertices(mesh, rotation_matrix)
        verts_3d = (camera_pose[:3, :3] @ rotated_mesh.vertices.T + camera_pose[:3, 3:4]).T
        verts_2d = project_vertices(rotated_mesh.vertices, camera_pose, intrinsics).astype(np.int32)

        for face in rotated_mesh.faces:
            pts_2d = verts_2d[face]
            pts_3d = verts_3d[face]
            if np.any(pts_2d[:, 0] < 0) or np.any(pts_2d[:, 0] >= image_size):
                continue
            if np.any(pts_2d[:, 1] < 0) or np.any(pts_2d[:, 1] >= image_size):
                continue
            
            mask_poly = np.zeros((image_size, image_size), dtype=np.uint8)
            cv2.fillConvexPoly(mask_poly, pts_2d, 1)
            
            z_avg = np.mean(pts_3d[:, 2])
            
            update_mask = (mask_poly == 1) & (z_avg < depth_buffer)
            material_mask[update_mask] = material_id
            depth_buffer[update_mask] = z_avg
    
    return material_mask

def _find_material(fake_material: dict, name: str):
    """
    Search across 'bus', 'wings', 'antenna' for an exact material name.
    Returns (group_name, material_name, entry_dict), or raises ValueError if not found.
    """
    for group in ("bus", "wings", "antenna"):
        group_dict = fake_material.get(group, {})
        if name in group_dict:
            return group, name, group_dict[name]
    # optional: try a case-insensitive match if exact not found
    for group in ("bus", "wings", "antenna"):
        for k in fake_material.get(group, {}):
            if k.lower() == name.lower():
                return group, k, fake_material[group][k]
    raise ValueError(f"Material '{name}' not found in any of ('bus','wings','antenna').")

    
def sample_material(fake_material: dict, name: str, weight: float = None):
    """
    Pick ONE material by its name (across bus/wings/antenna).
    If irradiated exists, blend: y = (1-w)*pristine + w*irradiated, with w in [0,1].
    If irradiated doesn't exist, keep pristine. If weight is None, w is random in [0,1].

    Returns:
      {
        'part': <'bus'|'wings'|'antenna'>,
        'item': {
          'name': <material_name>,
          'wavelength': np.ndarray,
          'spectrum': np.ndarray,      # blended or pristine-only
          'pristine': np.ndarray,
          'irradiated': np.ndarray|None,
          'weight': float|None
        }
      }
    """
    group, mat_name, entry = _find_material(fake_material, name)

    wl = np.asarray(entry["wavelength"])
    y_pris = np.asarray(entry["pristine"])
    y_irr = np.asarray(entry["irradiated"]) if "irradiated" in entry else None

    if y_irr is not None and y_irr.shape == y_pris.shape and np.isfinite(y_irr).any():
        w_use = random.random() if weight is None else float(np.asarray(weight).reshape(-1)[0])  # [shim]
        if not (0.0 <= w_use <= 1.0):
            raise ValueError("weight must be in [0,1].")
        y_syn = (1.0 - w_use) * y_pris + w_use * y_irr
    else:
        y_syn = y_pris.copy()
        w_use = None
        y_irr = None

#     return {
#         "part": group,
#         "item": {
#             "name": mat_name,
#             "wavelength": wl,
#             "spectrum": y_syn,
#             "pristine": y_pris,
#             "irradiated": y_irr,
#             "weight": w_use
#         }
#     }
    return y_syn

# def create_spectral_cube(material_mask, fake_material,  freq_count):
#     h, w = material_mask.shape
    
#     spectral_cube = np.zeros((freq_count, h, w))

#     unique_ids = np.unique(material_mask)
# #     print(f"Unique material IDs in mask: {unique_ids}")
#     weight = np.random.rand(1)
#     for x in range(h):
#         for y in range(w):
#             material_id = material_mask[x, y]
#             material_name = get_material_name(material_id)
#             if material_id != 0:
# #                 spectral_cube[:, x, y] = get_subdict(fake_material, material_name)['DHR']
#                 spectral_cube[:,x,y] = sample_material(fake_material,material_name,weight=weight)
#             else:
#                 spectral_cube[:, x, y] = 0.0

#     return spectral_cube, unique_ids[1:]

def create_spectral_cube(material_mask, fake_material,  freq_count):
    h, w = material_mask.shape
    unique_id = np.unique(material_mask)[1:]
    
    spectral_cube = np.zeros((freq_count, h, w))
    spectral_cube_noise_spec = np.zeros((freq_count, h, w))
    
   

    unique_ids = np.unique(material_mask)
    weight = np.random.rand(1)
    
    noise_spec = {}
    spec = {}
    for spc_id in unique_id:                       # e.g., array([8,17,26,34,35,42], dtype=uint8)
        material_name = get_material_name(spc_id)  # was material_id; use spc_id here
        sp_weighted = sample_material(fake_material, material_name, weight=weight)
        sp_noise = sp_weighted + np.random.randn(*sp_weighted.shape)*sp_weighted.std()/2
        noise_spec[int(spc_id)] = sp_noise        # keys as ints is convenient
        spec[int(spc_id)] = sp_weighted
    
    
    for x in range(h):
        for y in range(w):
            material_id = material_mask[x, y]
            if material_id != 0:
                spectral_cube[:,x,y] = spec[material_mask[x, y]]
                spectral_cube_noise_spec[:,x,y] = noise_spec[material_mask[x, y]]
            else:
                spectral_cube[:, x, y] = 0.0
                spectral_cube_noise_spec[:,x,y] = 0.0

    return spectral_cube,spectral_cube_noise_spec, unique_ids[1:]


def generate_sample(fake_material, num_spectrum=18, zoom_level=3, image_size=128, camera_pos=[0,0,25]):
    
    color_to_material = generate_distinct_color_list(fake_material)
    
    idx = np.random.randint(0,2)
    if idx == 0:
        satellite, components = make_satellite_with_ids(color_to_material)
    else:
        satellite, components = make_satellite_with_ids2(color_to_material)
    
    angles = np.array([90., 0., 0.])
    angles[0] = np.random.randint(45,135,1)[0] # vertical  [shim]
    # angles[1] = np.random.randint(-45,45,1) # horizontal
    angles[2] = np.random.randint(-180,180,1)[0] # face direction  [shim]

    material_mask = rasterize_components_with_depth(
        components, image_size=image_size, camera_pos=camera_pos, angles=angles)
    
    spectral_cube,spectral_cube_noise_spec, labels = create_spectral_cube(material_mask, fake_material, num_spectrum)
    
    #---for a larger same model
    zoomed_material_mask = rasterize_components_with_depth(
        components, image_size=image_size*zoom_level, camera_pos=camera_pos, angles=angles)
    zoomed_spectral_cube,_, _ = create_spectral_cube(zoomed_material_mask, fake_material, num_spectrum)

    
    return zoomed_material_mask, spectral_cube, spectral_cube_noise_spec, labels, zoomed_spectral_cube



def make_satellite_with_ids(color_to_material):
    components = []
    
    #--------------
    # Body split into up, mid, down
    bus_colors = list(color_to_material['bus'].keys()) # colors
    bus_names = list(color_to_material['bus'].values()) # material names
    bus_idx = np.random.choice(np.arange(len(bus_names)), 3, replace=False)
    
    h_total = 3.0
    h_third = h_total / 2.0

    # Down part
    this_id = get_material_index(bus_names[bus_idx[0]])
    body_down = trimesh.creation.cylinder(radius=0.8, height=h_third, sections=20)
    body_down.apply_translation([0, 0, -h_total / 2 + h_third / 2])
    body_down.visual.vertex_colors = np.tile(bus_colors[bus_idx[0]], (len(body_down.vertices), 1))
    components.append((body_down, this_id))

#     # Mid part
#     this_id = get_material_index(bus_names[bus_idx[1]])
#     body_mid = trimesh.creation.cylinder(radius=0.8, height=h_third, sections=50)
#     body_mid.visual.vertex_colors = np.tile(bus_colors[bus_idx[1]], (len(body_mid.vertices), 1))
#     components.append((body_mid, this_id))

    # Up part
    this_id = get_material_index(bus_names[bus_idx[1]])
    body_up = trimesh.creation.cylinder(radius=0.8, height=h_third, sections=20)
    body_up.apply_translation([0, 0, h_total / 2 - h_third / 2])
    body_up.visual.vertex_colors = np.tile(bus_colors[bus_idx[1]], (len(body_up.vertices), 1))
    components.append((body_up, this_id))
    
    # Connectors
    this_id = get_material_index(bus_names[bus_idx[2]])
    connector1 = trimesh.creation.box(extents=[2.0, 0.3, 0.3])
    connector1.apply_translation([1.5 + 0.5, 0, 0])
    connector2 = trimesh.creation.box(extents=[2.0, 0.3, 0.3])
    connector2.apply_translation([-1.5 - 0.5, 0, 0])
    connectors = trimesh.util.concatenate([connector1, connector2])
    connectors.visual.vertex_colors = np.tile(bus_colors[bus_idx[2]], (len(connectors.vertices), 1))
    components.append((connectors, this_id))
    
    
    #------------
    # Antenna
    antenna_colors = list(color_to_material['antenna'].keys())
    antenna_names = list(color_to_material['antenna'].values())
    antenna_idx = np.random.choice(np.arange(len(antenna_names)), 1, replace=False)
    
    
    
    this_id = get_material_index(antenna_names[antenna_idx[0]])
    antenna = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
    antenna.apply_translation([0, 0, h_total / 2 + 0.8])
    antenna.visual.vertex_colors = np.tile(antenna_colors[antenna_idx[0]], (len(antenna.vertices), 1))
    components.append((antenna, this_id))
    
    
    
    #-------------------
    # Panels
    wing_colors = list(color_to_material['wings'].keys())
    wing_names = list(color_to_material['wings'].values())
    wing_idx = np.random.choice(np.arange(len(wing_names)), 1, replace=False)
    
    this_id = get_material_index(wing_names[wing_idx[0]])
    panel1 = trimesh.creation.box(extents=[4.0, 0.01, 2.0])
    panel1.apply_translation([6., 0, 0])
    panel2 = trimesh.creation.box(extents=[4.0, 0.01, 2.0])
    panel2.apply_translation([-6., 0, 0])
    solar_panels = trimesh.util.concatenate([panel1, panel2])
    solar_panels.visual.vertex_colors = np.tile(wing_colors[wing_idx[0]], (len(solar_panels.vertices), 1))
    components.append((solar_panels, this_id))

    # Combine all
    satellite = trimesh.util.concatenate([
        body_down, body_up,
        antenna, connectors, solar_panels
    ])

    return satellite, components


# def make_satellite_with_ids2(color_to_material):
#     components = []

#     # ---------- Bus (split up/down like yours) ----------
#     bus_colors = list(color_to_material['bus'].keys())
#     bus_names  = list(color_to_material['bus'].values())
#     bus_idx = np.random.choice(np.arange(len(bus_names)), 2, replace=False)

#     h_total = 3.5
#     h_half  = h_total / 2.0
#     radius  = 0.8

#     # lower half
#     this_id = get_material_index(bus_names[bus_idx[0]])
#     body_low = trimesh.creation.cylinder(radius=radius, height=h_half, sections=24)
#     body_low.apply_translation([0, 0, -h_half/2])
#     body_low.visual.vertex_colors = np.tile(bus_colors[bus_idx[0]], (len(body_low.vertices), 1))
#     components.append((body_low, this_id))

#     # upper half
#     this_id = get_material_index(bus_names[bus_idx[1]])
#     body_up = trimesh.creation.cylinder(radius=radius, height=h_half, sections=24)
#     body_up.apply_translation([0, 0,  h_half/2])
#     body_up.visual.vertex_colors = np.tile(bus_colors[bus_idx[1]], (len(body_up.vertices), 1))
#     components.append((body_up, this_id))

#     # ---------- Solar panels (wings) ----------
#     wing_colors = list(color_to_material['wings'].keys())
#     wing_names  = list(color_to_material['wings'].values())
#     wing_idx = np.random.choice(np.arange(len(wing_names)), 1, replace=False)

#     this_id = get_material_index(wing_names[wing_idx[0]])
#     p1 = trimesh.creation.box(extents=[4.5, 0.01, 2.0]); p1.apply_translation([ 4.5, 0, 0])
#     p2 = trimesh.creation.box(extents=[4.5, 0.01, 2.0]); p2.apply_translation([-4.5, 0, 0])
#     panels = trimesh.util.concatenate([p1, p2])
#     panels.visual.vertex_colors = np.tile(wing_colors[wing_idx[0]], (len(panels.vertices), 1))
#     components.append((panels, this_id))

#     # ---------- Boom ----------
#     boom_idx = np.random.choice(np.arange(len(bus_names)), 1, replace=False)
#     this_id = get_material_index(bus_names[boom_idx[0]])
#     boom = trimesh.creation.box(extents=[0.15, 0.15, 6.0])
#     boom.apply_translation([0, 0, h_half + 3.0])
#     boom.visual.vertex_colors = np.tile(bus_colors[boom_idx[0]], (len(boom.vertices), 1))
#     components.append((boom, this_id))

#     # ---------- Reflector dish ----------
#     dish_idx = np.random.choice(np.arange(len(wing_names)), 1, replace=False)
#     this_id = get_material_index(wing_names[dish_idx[0]])
#     dish = trimesh.creation.cylinder(radius=3.2, height=0.03, sections=64)
#     dish.apply_translation([0, 0, h_half + 6.0])
#     dish.visual.vertex_colors = np.tile(wing_colors[dish_idx[0]], (len(dish.vertices), 1))
#     components.append((dish, this_id))

#     # ---------- Feed ----------
#     ant_colors = list(color_to_material['antenna'].keys())
#     ant_names  = list(color_to_material['antenna'].values())
#     ant_idx = np.random.choice(np.arange(len(ant_names)), 1, replace=False)

#     this_id = get_material_index(ant_names[ant_idx[0]])
#     feed = trimesh.creation.icosphere(subdivisions=2, radius=0.25)
#     feed.apply_translation([0.0, 0.0, h_half + 5.3])
#     feed.visual.vertex_colors = np.tile(ant_colors[ant_idx[0]], (len(feed.vertices), 1))
#     components.append((feed, this_id))

#     # ---------- center the WHOLE object ----------
#     # compute global AABB from all parts
#     mins = np.array([ np.inf,  np.inf,  np.inf])
#     maxs = np.array([-np.inf, -np.inf, -np.inf])
#     for m, _ in components:
#         bmin, bmax = m.bounds
#         mins = np.minimum(mins, bmin)
#         maxs = np.maximum(maxs, bmax)
#     center = (mins + maxs) / 2.0

#     # shift every part so the overall center is at the origin
#     for m, _ in components:
#         m.apply_translation(-center)

#     satellite = trimesh.util.concatenate([m for (m, _) in components])
#     return satellite, components

def make_satellite_with_ids2(color_to_material):
    components = []

    # ---------- Bus (split up/down) ----------
    bus_colors = list(color_to_material['bus'].keys())
    bus_names  = list(color_to_material['bus'].values())
    bus_idx = np.random.choice(np.arange(len(bus_names)), 2, replace=False)

    h_total = 3.5
    h_half  = h_total / 2.0
    body_radius = 1.25
    body_sections = 24

    # lower half
    this_id = get_material_index(bus_names[bus_idx[0]])
    body_low = trimesh.creation.cylinder(radius=body_radius, height=h_half, sections=body_sections)
    body_low.apply_translation([0, 0, -h_half/2])
    body_low.visual.vertex_colors = np.tile(bus_colors[bus_idx[0]], (len(body_low.vertices), 1))
    components.append((body_low, this_id))

    # upper half
    this_id = get_material_index(bus_names[bus_idx[1]])
    body_up = trimesh.creation.cylinder(radius=body_radius, height=h_half, sections=body_sections)
    body_up.apply_translation([0, 0,  h_half/2])
    body_up.visual.vertex_colors = np.tile(bus_colors[bus_idx[1]], (len(body_up.vertices), 1))
    components.append((body_up, this_id))

    # ---------- Solar panels (wings) ----------
    wing_colors = list(color_to_material['wings'].keys())
    wing_names  = list(color_to_material['wings'].values())
    wing_idx = np.random.choice(np.arange(len(wing_names)), 1, replace=False)

    this_id = get_material_index(wing_names[wing_idx[0]])

    panel_len_x, panel_thick_y, panel_height_z = 6.0, 0.01, 3.0
    panel_gap = 0.12
    panel_center_x = body_radius + panel_gap + panel_len_x / 2.0

    p1 = trimesh.creation.box(extents=[panel_len_x, panel_thick_y, panel_height_z])
    p1.apply_translation([+panel_center_x, 0, 0])
    p2 = trimesh.creation.box(extents=[panel_len_x, panel_thick_y, panel_height_z])
    p2.apply_translation([-panel_center_x, 0, 0])
    panels = trimesh.util.concatenate([p1, p2])
    panels.visual.vertex_colors = np.tile(wing_colors[wing_idx[0]], (len(panels.vertices), 1))
    components.append((panels, this_id))

    # ---------- Boom ----------
    boom_idx = np.random.choice(np.arange(len(bus_names)), 1, replace=False)
    this_id = get_material_index(bus_names[boom_idx[0]])
    boom = trimesh.creation.box(extents=[0.15, 0.15, 6.0])
    boom.apply_translation([0, 0, h_half + 3.0])
    boom.visual.vertex_colors = np.tile(bus_colors[boom_idx[0]], (len(boom.vertices), 1))
    components.append((boom, this_id))

    # ---------- Antenna (dish + feed share one material) ----------
    ant_colors = list(color_to_material['antenna'].keys())
    ant_names  = list(color_to_material['antenna'].values())
    ant_idx = np.random.choice(np.arange(len(ant_names)), 1, replace=False)

    ant_color = ant_colors[ant_idx[0]]
    ant_name  = ant_names[ant_idx[0]]
    ant_id    = get_material_index(ant_name)

    # dish
    dish = trimesh.creation.cylinder(radius=3.2, height=0.03, sections=64)
    dish.apply_translation([0, 0, h_half + 6.0])
    dish.visual.vertex_colors = np.tile(ant_color, (len(dish.vertices), 1))

    # feed
    feed = trimesh.creation.icosphere(subdivisions=2, radius=0.25)
    feed.apply_translation([0.0, 0.0, h_half + 5.3])
    feed.visual.vertex_colors = np.tile(ant_color, (len(feed.vertices), 1))

    # combine antenna parts as one component (same material)
    antenna = trimesh.util.concatenate([dish, feed])
    components.append((antenna, ant_id))

    # ---------- Center the whole assembly ----------
    mins = np.array([np.inf, np.inf, np.inf])
    maxs = np.array([-np.inf, -np.inf, -np.inf])
    for m, _ in components:
        bmin, bmax = m.bounds
        mins = np.minimum(mins, bmin)
        maxs = np.maximum(maxs, bmax)
    center = (mins + maxs) / 2.0

    for m, _ in components:
        m.apply_translation(-center)

    satellite = trimesh.util.concatenate([m for (m, _) in components])
    return satellite, components


def simulator(num_spectrum, image_size, fake_material, zoom_level=3, camera_pos=[0,0,25], noise_std=0.05, seed=None):
    
    material_mask, spectral_cube, spectral_cube_noise_spec,labels, zoomed_spectral_cube = generate_sample(
        fake_material=fake_material,
        num_spectrum = num_spectrum,
        zoom_level=zoom_level,
        image_size=image_size, 
        camera_pos=camera_pos
    )

    n_slices = spectral_cube.shape[0]
    blur_cube = np.zeros([n_slices,image_size,image_size])

    sigmas = np.linspace(3.5e-7, 1.0e-6, n_slices)
    sigmas = 0.25*sigmas/(3.6*5e-6)*120
    blurred_cube = np.stack(
        [gaussian_filter(spectral_cube_noise_spec[j], 
                                       sigma=sigmas[j], 
                                       mode="mirror") for j in range(n_slices)], 
        axis=0
    ) 
    # blurred_cube += np.random.randn(*blurred_cube.shape)*noise_std
    rng = np.random.default_rng(seed)  # create once outside your loop/function
    blurred_cube += rng.normal(0.0, noise_std, size=blurred_cube.shape).astype(np.float32)


    blur_cube = blurred_cube
        
    return material_mask,spectral_cube, blur_cube, labels, zoomed_spectral_cube


#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------simulator for random shape

def get_subdict(fake_spectral, subdict_name):
    for _, sub in fake_spectral.items():
        if subdict_name in sub:
            return sub[subdict_name]
    return None

def random_complex_region_mask(n=32, max_regions=5, seed=None,
                               min_radius=2, min_side=2):
    """
    Generate a mask divided into up to max_regions regions.
    Each region has a unique label between 1 and 16.
    Shapes can be rotated for more complexity.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((n, n), dtype=np.uint8)

    num_regions = rng.integers(1, max_regions + 1)

    def rint(low, high):
        if high < low:
            return low
        return int(rng.integers(low, high + 1))

    used_labels = set()
    shape_types = [
        "circle", "rectangle", "ellipse", "polygon",
        "triangle", "line", "diamond", "cross", "ring"
    ]

    for _ in range(num_regions):
        label = int(rng.integers(1, 47))
        while label in used_labels:
            label = int(rng.integers(1, 47))
        used_labels.add(label)

        shape_type = random.choice(shape_types)
        
        if shape_type == "circle":
            cx, cy = rint(0, n-1), rint(0, n-1)
            radius = rint(min_radius, max(min_radius, n//6))
            cv2.circle(img, (cx, cy), radius, label, -1)

        elif shape_type == "ring":
            cx, cy = rint(0, n-1), rint(0, n-1)
            outer = rint(min_radius+3, max(min_radius+3, n//5))
            inner = rint(1, max(1, outer-2))
            cv2.circle(img, (cx, cy), outer, label, -1)
            cv2.circle(img, (cx, cy), inner, 0, -1)  # hollow

        elif shape_type == "rectangle":
            w, h = rint(min_side, n//4), rint(min_side, n//4)
            rect = ((rint(w, n-w), rint(h, n-h)), (w, h), rint(0, 180)) # (center, (w,h), angle)
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.fillPoly(img, [box], label)

        elif shape_type == "line":
            cx, cy = rint(0, n-1), rint(0, n-1)
            length = rint(min_radius*2, n//2)
            angle = np.deg2rad(rint(0, 180))
            dx, dy = int(length*np.cos(angle)), int(length*np.sin(angle))
            cv2.line(img, (cx-dx, cy-dy), (cx+dx, cy+dy), label, thickness=2)

        elif shape_type == "ellipse":
            cx, cy = rint(0, n-1), rint(0, n-1)
            ax, ay = rint(min_radius, n//5), rint(min_radius, n//5)
            angle = rint(0, 180)
            cv2.ellipse(img, (cx, cy), (ax, ay), angle, 0, 360, label, -1)

        elif shape_type == "polygon":
            k = rint(4, 8)
            pts = np.column_stack([rng.integers(0, n, size=k), rng.integers(0, n, size=k)]).astype(np.int32)
            M = cv2.getRotationMatrix2D((n//2, n//2), rint(0,180), 1.0)
            pts = cv2.transform(pts.reshape(-1,1,2), M)
            cv2.fillPoly(img, [pts], label)

        elif shape_type == "triangle":
            pts = np.column_stack([rng.integers(0, n, size=2), rng.integers(0, n, size=2)]).astype(np.int32)
            M = cv2.getRotationMatrix2D((np.mean(pts[:,0]), np.mean(pts[:,1])), rint(0,180), 1.0)
            pts = cv2.transform(pts.reshape(-1,1,2), M)
            cv2.fillPoly(img, [pts], label)

        elif shape_type == "diamond":
            cx, cy = rint(0, n-1), rint(0, n-1)
            size = rint(min_radius, n//6)
            pts = np.array([[cx, cy-size],[cx+size, cy],[cx, cy+size],[cx-size, cy]], np.int32)
            M = cv2.getRotationMatrix2D((cx,cy), rint(0,180), 1.0)
            pts = cv2.transform(pts.reshape(-1,1,2), M)
            cv2.fillPoly(img, [pts], label)

        elif shape_type == "cross":
            cx, cy = rint(0, n-1), rint(0, n-1)
            size = rint(min_radius*2, n//6)
            thickness = max(1, size//4)
            cross = np.zeros((n,n), np.uint8)
            cv2.rectangle(cross, (cx-thickness, cy-size), (cx+thickness, cy+size), 255, -1)
            cv2.rectangle(cross, (cx-size, cy-thickness), (cx+size, cy+thickness), 255, -1)
            M = cv2.getRotationMatrix2D((cx,cy), rint(0,180), 1.0)
            rotated = cv2.warpAffine(cross, M, (n,n))
            img[rotated>0] = label

    return img

def shape_simulator(fake_material, size=20, num_channels=18, max_regions=3, seed=None,
                    noise_std=0.08, zoom=2):
    """
    Matches the 'shared noise per material' logic from create_spectral_cube:
      - For each material ID, compute sp_weighted (length==num_channels)
      - Build noise_spec[id] = sp_weighted + N(0, std(sp_weighted)^2)  # one draw per material
      - Assign by region using those precomputed vectors (shared across all pixels of that ID)
      - Low-res cube is formed from noise_spec, then blurred; optional global sensor noise after blur
      - High-res cube uses the CLEAN spec (non-noised), piecewise constant

    Returns:
        mask_hr   : [H,W] int32 mask at high-res (H=W=size*zoom)
        hsi_hr    : [C,H,W] HR cube (CLEAN spectra per material)
        blurred_lr: [C,size,size] LR cube built from shared-noise then blurred (+ optional post-blur noise)
    """
    rng = np.random.default_rng(seed)

    # 1) Low-res mask
    mask_lr = random_complex_region_mask(size, max_regions=max_regions, seed=seed).astype(np.int32)
    unique_ids = np.unique(mask_lr)
    unique_ids = unique_ids[unique_ids != 0]

    # same global weight behavior as your create_spectral_cube
    weight = np.random.rand(1)

    # helper: match spectrum length to num_channels
    def _match_len(spectrum, target_len):
        spectrum = np.asarray(spectrum, dtype=np.float32)
        L = len(spectrum)
        if L == target_len:
            return spectrum
        if L > target_len:
            step = L / target_len
            return np.array([spectrum[int(i * step)] for i in range(target_len)], dtype=np.float32)
        return np.pad(spectrum, (0, target_len - L), mode="edge").astype(np.float32)

    # 2) Precompute clean + shared-noise spectra per material id
    spec = {}
    noise_spec = {}
    for spc_id in unique_ids:
        material_name = get_material_name(int(spc_id))
        sp_weighted = sample_material(fake_material, material_name, weight=weight)
        sp_weighted = _match_len(sp_weighted, num_channels)

        sigma = float(sp_weighted.std())/2
        eps_vec = rng.normal(0.0, sigma, size=sp_weighted.shape).astype(np.float32) if sigma > 0 else 0.0
        sp_noise = (sp_weighted + eps_vec).astype(np.float32)

        spec[int(spc_id)] = sp_weighted.astype(np.float32)  # CLEAN spectrum
        noise_spec[int(spc_id)] = sp_noise                  # shared noisy spectrum

    # 3) Build low-res cube from shared-noise spectra
    hsi_lr = np.zeros((num_channels, size, size), dtype=np.float32)
    for spc_id in unique_ids:
        region = (mask_lr == spc_id)
        sp = noise_spec[int(spc_id)]
        for b in range(num_channels):
            hsi_lr[b][region] = sp[b]

    # 4) Blur LR band-wise
    sigmas = np.linspace(3.5e-7, 1.0e-6, num_channels)
    sigmas = 0.25 * sigmas / (3.6 * 5e-6) * 120
    blurred_lr = np.stack(
        [gaussian_filter(hsi_lr[j], sigma=float(sigmas[j]), mode="mirror") for j in range(num_channels)],
        axis=0
    ).astype(np.float32)

    # Optional global sensor/read noise after blur (unchanged)
    if noise_std and noise_std > 0:
        blurred_lr += rng.normal(0.0, noise_std, size=blurred_lr.shape).astype(np.float32)

    # 5) Upsample mask and build HR cube using the CLEAN spectra (change is here)
    H, W = size * zoom, size * zoom
    mask_hr = cv2.resize(mask_lr, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)

    hsi_hr = np.zeros((num_channels, H, W), dtype=np.float32)
    for spc_id in np.unique(mask_hr):
        if spc_id == 0:
            continue
        region = (mask_hr == spc_id)
        sp = spec[int(spc_id)]  # CLEAN spectrum for HR (this line changed)
        for b in range(num_channels):
            hsi_hr[b][region] = sp[b]

    return mask_hr, hsi_hr, blurred_lr


#-----another random shape--------------
#-----another random shape--------------
#-----another random shape--------------
#-----another random shape--------------
#-----another random shape--------------
#-----another random shape--------------
#-----another random shape--------------
#-----another random shape--------------

# --- new: N×N mask made of small n×n patches ---
def random_patch_region_mask2(
    N=32,
    patch_sizes=(1, 2, 3, 4),
    num_patches_range=(5, 10),
    seed=None,
    max_trials_per_patch=80
):
    """
    Create an N×N label mask composed of several non-overlapping n×n square patches.
    - n is sampled from `patch_sizes`
    - number of patches k ~ Uniform(num_patches_range[0], num_patches_range[1]) inclusive
    - each patch gets a unique label randomly drawn from 1..47
    """
    rng = np.random.default_rng(seed)
    mask = np.zeros((N, N), dtype=np.uint8)

    k = int(rng.integers(num_patches_range[0], num_patches_range[1] + 1))
    # unique labels in [1..47]
    all_labels = np.arange(1, 47, dtype=np.uint8)
    if k > len(all_labels):
        k = len(all_labels)
    labels = rng.choice(all_labels, size=k, replace=False)

    for lbl in labels:
        placed = False
        for _ in range(max_trials_per_patch):
            n = int(rng.choice(patch_sizes))
            if n > N:
                continue
            x0 = int(rng.integers(0, N - n + 1))
            y0 = int(rng.integers(0, N - n + 1))
            region = mask[y0:y0 + n, x0:x0 + n]
            if np.all(region == 0):  # no overlap
                region[:] = np.random.randint(1,47)
                placed = True
                break
        # if not placed after trials, we skip this label
    return mask

def shape_simulator2(fake_material, size=20, num_channels=18, seed=None,
                     noise_std=0.08, num_patches=(5,10), zoom=2):
    """
    - Build a patchy low-res mask via random_patch_region_mask2.
    - For each material ID:
        spec[id]       = clean spectrum (length==num_channels)
        noise_spec[id] = spec[id] + N(0, std(spec[id])^2)  # one draw per material, shared by all its pixels
    - Low-res cube (LR): filled from noise_spec (shared per material) -> blur bandwise.
      (Optional additive image noise after blur controlled by noise_std; set to 0 to disable.)
    - High-res cube (HR): filled from CLEAN spec (non-noised), piecewise constant by region.
    """
    rng = np.random.default_rng(seed)

    # 1) Low-res region mask (size×size) with small square patches
    mask_lr = random_patch_region_mask2(
        N=size,
        patch_sizes=(1, 2, 3, 4),
        num_patches_range=num_patches,
        seed=seed
    ).astype(np.int32)

    unique_ids = np.unique(mask_lr)
    unique_ids = unique_ids[unique_ids != 0]

    # One global weight (to mirror your prior functions)
    weight = np.random.rand(1)

    # Helper: match spectrum length to num_channels
    def _match_len(spectrum, target_len):
        spectrum = np.asarray(spectrum, dtype=np.float32)
        L = len(spectrum)
        if L == target_len:
            return spectrum
        if L > target_len:
            step = L / target_len
            return np.array([spectrum[int(i * step)] for i in range(target_len)], dtype=np.float32)
        return np.pad(spectrum, (0, target_len - L), mode='edge').astype(np.float32)

    # 2) Precompute per-ID spectra (clean + shared-noise vector)
    spec = {}
    noise_spec = {}
    for spc_id in unique_ids:
        material_name = get_material_name(int(spc_id))
        sp_clean = sample_material(fake_material, material_name, weight=weight)
        sp_clean = _match_len(sp_clean, num_channels)
        spec[int(spc_id)] = sp_clean.astype(np.float32)

        sigma = float(sp_clean.std())/2
        if sigma > 0:
            # Shared noise vector per material (same for all pixels of that material)
            eps_vec = rng.normal(0.0, sigma, size=sp_clean.shape).astype(np.float32)
            sp_noisy = (sp_clean + eps_vec).astype(np.float32)
        else:
            sp_noisy = sp_clean.copy()
        noise_spec[int(spc_id)] = sp_noisy

    # 3) Build LR cube from *noisy* spectra (shared per material), then blur
    hsi_lr = np.zeros((num_channels, size, size), dtype=np.float32)
    for spc_id in unique_ids:
        region = (mask_lr == spc_id)
        sp = noise_spec[int(spc_id)]
        for b in range(num_channels):
            hsi_lr[b][region] = sp[b]

    # Blur band-wise (PSF)
    sigmas = np.linspace(3.5e-7, 1.0e-6, num_channels)
    sigmas = 0.25 * sigmas / (3.6 * 5e-6) * 120
    blurred_lr = np.stack(
        [gaussian_filter(hsi_lr[j], sigma=float(sigmas[j]), mode="mirror") for j in range(num_channels)],
        axis=0
    ).astype(np.float32)

    # Optional post-blur image noise (leave as-is; set noise_std=0 to disable)
    if noise_std and noise_std > 0:
        blurred_lr += rng.normal(0.0, noise_std, size=blurred_lr.shape).astype(np.float32)

    # 4) Upsample mask and build HR cube from CLEAN spectra (non-noised)
    H, W = size * zoom, size * zoom
    mask_hr = cv2.resize(mask_lr, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)

    hsi_hr = np.zeros((num_channels, H, W), dtype=np.float32)
    for spc_id in np.unique(mask_hr):
        if spc_id == 0:
            continue
        region = (mask_hr == spc_id)
        sp = spec[int(spc_id)]  # CLEAN spectrum for HR
        for b in range(num_channels):
            hsi_hr[b][region] = sp[b]

    return mask_hr, hsi_hr, blurred_lr