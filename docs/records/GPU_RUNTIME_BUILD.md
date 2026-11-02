# GPU runtime candidate build: 2026-10-28

Status: **built locally; offline CPU/import/CLI smoke checks passed**. GPU execution,
reference-environment parity, full evaluation, and registry publication have not been performed.
This record is build evidence, not a scientific run or a supported environment release.

## Artifact

| Field | Observed value |
|---|---|
| Local tag | `nextt-runtime:cu128-candidate` |
| Local Docker image ID | `sha256:47099b4c9b0409554c6755b270099825205cdb1f0acbee2cba00ecb9e6331994` |
| Docker-reported image size | 9,425,366,640 bytes |
| Platform | Linux amd64 |
| Source base commit | `c2ef35a2484508e0ece7c2927527f247bd523685` |
| Packaged source manifest SHA256 | `97fd2ea97c58f81f925b7f3b3cbdcfb563b30815c654b78896162312c2d3523c` |
| Packaged files checked against the checkout | 176 |
| Python | 3.10.19 |
| PyTorch / CUDA runtime | 2.9.1+cu128 / 12.8 |
| Triton / fla-core / einops | 3.5.1 / 0.5.2 / 0.8.1 |
| Installed Python distributions | 107 |

The image was built while the new build tooling was uncommitted. The source base commit is therefore
not sufficient to reproduce its entire build context: the image's source-file manifest records the
actual packaged bytes. Documentation added or revised after the build is not part of that image. The build predates the
plan relocation and scope clarification on main (`8ba20de`); the manifest describes the build checkout,
not the later integration commit. This image ID
identifies the local artifact; no public registry pull location is provided or implied.

## Downloads and installation

- CUDA base: DaoCloud's Docker Hub mirror, pinned linux/amd64 manifest for CUDA 12.8.1 devel / Ubuntu 22.04.
- uv 0.9.5: DaoCloud's GHCR mirror, pinned linux/amd64 manifest.
- Python standalone: npmmirror.
- Ubuntu packages: Tsinghua mirror.
- Python dependencies: Huawei Cloud PyPI mirror. Tsinghua also resolved all 105 root-lock packages,
  but its large downloads were slower in this attempt, so the installation was restarted against Huawei.
- Root packages were exported from the unchanged `uv.lock`, with the GPU, upstream, and tracking extras.
  Installation retained artifact hashes, followed by the separately hash-pinned linear overlay.
- `uv pip check` reported all 107 installed packages compatible. No dependency versions were changed.

The shared Docker daemon had an unavailable proxy endpoint. A dedicated `nextt-env-builder` was used
with direct mirror access and an already cached BuildKit image (builder reported v0.31.2).
The shared daemon configuration and other jobs were not changed. No image was uploaded to a registry.

## Verification actually performed

All runtime checks used `docker run --network none` **without** GPU allocation or dataset/checkpoint mounts:

1. Package consistency check passed inside the loaded image.
2. Torch, Triton, FLA KDA/simple-GLA modules, and the project's attention wrapper imported successfully.
3. A small CPU matrix multiplication returned the expected result.
4. `nexttransformer.evaluate --help` and `nexttransformer.train --help` loaded successfully.
5. The image's default command printed evaluator help successfully.
6. All 176 packaged source-file hashes matched the build checkout.
7. Documentation links, shell examples, the build script's Bash syntax, and Git whitespace checks passed.

The no-GPU import check reported `cuda_visible=false` and `fa3_available=false`. FLA warned that it
was using CPU without Triton device support and recommended Python 3.11+. The pinned Python 3.10.19
was retained. These checks establish package/entry-point loading, not GPU kernel execution.

## Remaining acceptance work

No reference FA3 cache is included. Capture the intended reference environment and kernel artifacts,
validate host driver compatibility and every allocated device, exercise BF16/FLA/sparse kernels and NCCL,
and compare the same checkpoint in the reference environment and container before formal use.
The full evaluation launcher, result validator, and Harbor adapter remain pending. Training also needs
update, accounting, and resume validation. Do not interpret this build as completion of those tasks.

See the [build instructions](../../environments/eval/README.md) and [implementation plan](../plans/REPRODUCIBLE_EVAL.md).
The image retains complete build inventory under `/opt/nextt-manifest/`; local build and smoke logs
are under `.agent-work/docker-build/` in the build checkout.
