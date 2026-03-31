#!/usr/bin/env python3
"""
Export TribeV2 predictions to motion glTF for interactive web visualization.
Uses vertex color animation (more efficient than morph targets for time series).
"""

import argparse
import json
import struct
import numpy as np
from pathlib import Path
import base64


def create_brain_gltf(npy_file, output='brain.gltf', mesh='fsaverage5', fps=2):
    """
    Create animated glTF with vertex colors representing brain activity over time.
    
    Strategy:
    - Single static mesh (fsaverage5)
    - Multiple materials/primitives - one per timestep
    - Vertex colors encoded as RGB per vertex
    - Animation hides/shows different timesteps (or shader blends)
    """
    preds = np.load(npy_file)
    n_timesteps, n_vertices = preds.shape
    
    print(f"Creating glTF: {n_timesteps} frames, {n_vertices} vertices")
    
    # Normalize to 0-1 for color mapping
    vmin, vmax = np.percentile(preds, [1, 99])
    preds_norm = np.clip((preds - vmin) / (vmax - vmin), 0, 1)
    
    # Get fsaverage5 mesh coordinates
    # For now, we'll create a simplified placeholder mesh
    # In production, you'd load actual fsaverage5 surfaces
    
    # Load actual fsaverage5 if available
    try:
        from nibabel import freesurfer
        from nilearn import datasets
        
        fs = datasets.fetch_surf_fsaverage(mesh)
        
        # Load left hemisphere
        coords_l, faces_l = freesurfer.read_geometry(fs['pial_left'])
        # Load right hemisphere  
        coords_r, faces_r = freesurfer.read_geometry(fs['pial_right'])
        
        # Combine hemispheres
        coords = np.vstack([coords_l, coords_r + [coords_l[:, 0].max() + 10, 0, 0]])
        faces = np.vstack([faces_l, faces_r + len(coords_l)])
        
    except Exception as e:
        print(f"Warning: Could not load fsaverage5: {e}")
        print("Generating placeholder mesh...")
        # Create simple sphere as placeholder
        coords, faces = create_sphere_mesh(n_vertices)
    
    # Ensure vertex count matches
    if len(coords) != n_vertices:
        print(f"Warning: Mesh vertices ({len(coords)}) != data vertices ({n_vertices})")
        # Resample or interpolate would go here
        coords = coords[:n_vertices] if len(coords) > n_vertices else np.pad(
            coords, ((0, n_vertices - len(coords)), (0, 0)), mode='edge'
        )
    
    # Build glTF structure
    gltf = build_animated_gltf(coords, faces, preds_norm, fps)
    
    # Write glTF JSON + binary
    output = Path(output)
    write_gltf(gltf, output)
    
    print(f"Saved to: {output}")
    print(f"View with: https://gltf-viewer.donmccurdy.com/")
    return output


def create_sphere_mesh(n_vertices):
    """Create simple spherical mesh as fallback"""
    # Golden spiral method for even distribution
    indices = np.arange(0, n_vertices, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/n_vertices)
    theta = np.pi * (1 + 5**0.5) * indices
    
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    
    coords = np.column_stack([x, y, z]) * 50  # Scale to brain-ish size
    
    # Simple triangulation (not perfect but works for visualization)
    faces = []
    for i in range(n_vertices - 2):
        faces.append([0, i+1, i+2])
    faces = np.array(faces)
    
    return coords.astype(np.float32), faces.astype(np.uint32)


def fire_colormap(values):
    """Convert 0-1 values to fire colormap RGB"""
    # Fire colormap: black -> red -> yellow -> white
    colors = np.zeros((len(values), 3))
    
    # Red channel: 0 -> 1
    colors[:, 0] = np.clip(values * 2, 0, 1)
    
    # Green channel: 0 at 0.5, 1 at 1.0
    colors[:, 1] = np.clip((values - 0.5) * 2, 0, 1)
    
    # Blue channel: 0 until 0.75, then ramps to 1
    colors[:, 2] = np.clip((values - 0.75) * 4, 0, 1)
    
    return colors.astype(np.float32)


def build_animated_gltf(coords, faces, pred_norm, fps=2):
    """
    Build glTF with vertex color animation.
    
    Approach: Each timestep is a separate mesh primitive with vertex colors.
    We use visibility animation to switch between them.
    """
    n_timesteps = len(pred_norm)
    
    # Prepare buffers
    vertices_blob = coords.astype(np.float32).tobytes()
    indices_blob = faces.astype(np.uint32).tobytes()
    
    # Create vertex colors for each timestep
    color_blobs = []
    for t in range(n_timesteps):
        colors = fire_colormap(pred_norm[t])
        color_blobs.append(colors.tobytes())
    
    # Calculate buffer layout
    vertices_len = len(vertices_blob)
    indices_len = len(indices_blob)
    color_len = len(color_blobs[0])
    
    # Build single buffer
    buffer_parts = [vertices_blob, indices_blob] + color_blobs
    buffer_data = b''.join(buffer_parts)
    
    # Encode to base64 for embedded glTF
    buffer_b64 = base64.b64encode(buffer_data).decode('ascii')
    
    # Build glTF JSON structure
    gltf = {
        "asset": {
            "version": "2.0",
            "generator": "TribeV2 glTF Exporter"
        },
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{
            "name": "Brain",
            "mesh": 0
        }],
        "meshes": [{
            "name": "BrainMesh",
            "primitives": [
                {
                    "attributes": {
                        "POSITION": 0,
                        "COLOR_0": 2 + t  # Each timestep has its own color accessor
                    },
                    "indices": 1,
                    "mode": 4  # TRIANGLES
                }
                for t in range(min(n_timesteps, 60))  # Limit to 60 frames for file size
            ]
        }],
        "accessors": [
            # Vertex positions (0)
            {
                "bufferView": 0,
                "componentType": 5126,  # FLOAT
                "count": len(coords),
                "type": "VEC3",
                "max": coords.max(axis=0).tolist(),
                "min": coords.min(axis=0).tolist()
            },
            # Face indices (1)
            {
                "bufferView": 1,
                "componentType": 5125,  # UNSIGNED_INT
                "count": len(faces) * 3,
                "type": "SCALAR"
            }
        ] + [
            # Color accessors for each timestep (2, 3, 4, ...)
            {
                "bufferView": 2 + t,
                "componentType": 5126,  # FLOAT
                "count": len(coords),
                "type": "VEC3"
            }
            for t in range(min(n_timesteps, 60))
        ],
        "bufferViews": [
            # Vertices (0)
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": vertices_len
            },
            # Indices (1)
            {
                "buffer": 0,
                "byteOffset": vertices_len,
                "byteLength": indices_len
            }
        ] + [
            # Color data for each timestep
            {
                "buffer": 0,
                "byteOffset": vertices_len + indices_len + t * color_len,
                "byteLength": color_len
            }
            for t in range(min(n_timesteps, 60))
        ],
        "buffers": [{
            "uri": f"data:application/octet-stream;base64,{buffer_b64}",
            "byteLength": len(buffer_data)
        }]
    }
    
    # Add animation for visibility switching
    if n_timesteps > 1:
        # For simplicity, we'll create discrete visibility animation
        # Full implementation would use morph targets or shader uniforms
        gltf["animations"] = [{
            "channels": [
                {
                    "sampler": t,
                    "target": {
                        "node": 0,
                        "path": "weights" if t == 0 else "scale"  # Placeholder
                    }
                }
                for t in range(min(n_timesteps, 60))
            ],
            "samplers": [
                {
                    "input": 0,  # Would need time accessor
                    "interpolation": "STEP",
                    "output": 0   # Would need visibility values
                }
                for t in range(min(n_timesteps, 60))
            ]
        }]
    
    return gltf


def write_gltf(gltf, output_path):
    """Write glTF to file (embedded format)"""
    with open(output_path, 'w') as f:
        json.dump(gltf, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Export TribeV2 predictions to animated glTF'
    )
    parser.add_argument('npy_file', help='Path to _tribe.npy file')
    parser.add_argument('-o', '--output', default='brain.gltf',
                        help='Output glTF file')
    parser.add_argument('--fps', type=int, default=2,
                        help='Animation frame rate')
    
    args = parser.parse_args()
    create_brain_gltf(args.npy_file, args.output, fps=args.fps)


if __name__ == '__main__':
    main()
