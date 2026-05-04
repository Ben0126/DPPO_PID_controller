"""
Merge multiple expert_demos*.h5 files into a single mixed dataset.

Used by Run 13 to combine hover-only demos (expert_demos_v4.h5) with
approach demos (expert_demos_v4_approach.h5) so BC loss anchors the policy
to both regimes simultaneously.

Episode groups are renumbered consecutively in the output file. Per-episode
attributes (e.g. initial_pos_range) are preserved.

Usage:
    python -m scripts.merge_expert_demos \
        --inputs data/expert_demos_v4.h5 data/expert_demos_v4_approach.h5 \
        --output data/expert_demos_v4_mixed.h5
"""

import os
import sys
import argparse
import numpy as np
import h5py


def copy_episode(src_grp: h5py.Group, dst_grp: h5py.Group):
    for key in src_grp.keys():
        ds = src_grp[key]
        kwargs = {}
        if ds.compression is not None:
            kwargs['compression'] = ds.compression
            if ds.compression_opts is not None:
                kwargs['compression_opts'] = ds.compression_opts
        dst_grp.create_dataset(key, data=ds[...], **kwargs)
    for k, v in src_grp.attrs.items():
        dst_grp.attrs[k] = v


def merge(inputs, output):
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    total_episodes = 0
    total_steps    = 0
    versions       = []

    with h5py.File(output, 'w') as out_hf:
        attrs_done = False
        for src_path in inputs:
            print(f"Reading {src_path} ...")
            with h5py.File(src_path, 'r') as in_hf:
                ep_keys = sorted(
                    [k for k in in_hf.keys() if k.startswith('episode_')],
                    key=lambda s: int(s.split('_')[1]),
                )
                print(f"  {len(ep_keys)} episodes")
                versions.append(in_hf.attrs.get('version', 'unknown'))
                if not attrs_done:
                    for k in ('image_size', 'state_dim', 'action_dim', 'action_space'):
                        if k in in_hf.attrs:
                            out_hf.attrs[k] = in_hf.attrs[k]
                    attrs_done = True
                else:
                    for k in ('image_size', 'state_dim', 'action_dim', 'action_space'):
                        if k in in_hf.attrs and k in out_hf.attrs:
                            assert in_hf.attrs[k] == out_hf.attrs[k], \
                                f"Mismatch on {k}: {in_hf.attrs[k]} vs {out_hf.attrs[k]}"
                for ep_key in ep_keys:
                    src_grp = in_hf[ep_key]
                    new_name = f'episode_{total_episodes}'
                    dst_grp  = out_hf.create_group(new_name)
                    copy_episode(src_grp, dst_grp)
                    total_episodes += 1
                    if 'actions' in src_grp:
                        total_steps += src_grp['actions'].shape[0]

        out_hf.attrs['n_episodes']  = total_episodes
        out_hf.attrs['total_steps'] = total_steps
        out_hf.attrs['version']     = 'v4_mixed'
        out_hf.attrs['source_versions'] = np.array(
            [str(v) for v in versions], dtype=h5py.string_dtype())
        out_hf.attrs['source_files']    = np.array(
            [os.path.basename(p) for p in inputs], dtype=h5py.string_dtype())

    print(f"\n=== Merge complete ===")
    print(f"  Inputs:      {len(inputs)} files")
    print(f"  Episodes:    {total_episodes}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Output:      {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge expert demo HDF5 files")
    parser.add_argument('--inputs', nargs='+', required=True,
                        help='Input HDF5 files (in priority order)')
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    merge(args.inputs, args.output)
