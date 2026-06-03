#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_weights.py — HDF5 Weight File Inspector
=================================================
Utility to inspect the layer names and shapes stored in a Keras
HDF5 weight file. Used to verify layer-name compatibility between
the original TF1/Keras2 UMSI++ weights and our TF2/Keras3 port.

Usage:
    python3 saliency/inspect_weights.py saliency/weights/umsi++.hdf5
"""

import sys
import h5py


def inspect_hdf5_weights(filepath: str) -> None:
    """Print all groups/datasets in an HDF5 weight file."""
    with h5py.File(filepath, 'r') as f:
        print(f"=== HDF5 Weight File: {filepath} ===")
        print(f"Top-level keys: {list(f.keys())}")
        print()

        def _visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  [{name}] shape={obj.shape} dtype={obj.dtype}")

        # Check if it's a Keras weight file
        if 'model_weights' in f:
            root = f['model_weights']
        elif 'layer_names' in f.attrs:
            root = f
        else:
            root = f

        # List layer names
        if 'layer_names' in root.attrs:
            layer_names = [n.decode('utf-8') if isinstance(n, bytes) else n
                          for n in root.attrs['layer_names']]
            print(f"Layer count: {len(layer_names)}")
            print(f"\nLayer names:")
            for i, name in enumerate(layer_names):
                print(f"  [{i:3d}] {name}")
                # Check for weight arrays
                if name in root:
                    grp = root[name]
                    if 'weight_names' in grp.attrs:
                        wnames = grp.attrs['weight_names']
                        for wn in wnames:
                            wn_str = wn.decode('utf-8') if isinstance(wn, bytes) else wn
                            if wn_str in grp:
                                ds = grp[wn_str]
                                print(f"        {wn_str}: shape={ds.shape}")
            print()
        else:
            print("No 'layer_names' attribute. Listing all datasets:")
            root.visititems(_visit)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 inspect_weights.py <weights.hdf5>")
        sys.exit(1)
    inspect_hdf5_weights(sys.argv[1])
