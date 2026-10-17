# Current supervisor-provided dataset — 12 October 2026

The user designated the dataset supplied by co-supervisor **Ms. Anna-Louise Brun**
as the data source for subsequent TRDP2 work. The preceding LIDC-IDRI expansion
is deferred; previous datasets, protocols, splits and results are retained.

## Download and integrity

- Source service: RENATER FileSender (`filesender.renater.fr`), file ID 79364118.
- Original filename: `sino_selected.zip`.
- Local immutable archive: `~/espaces/travail/trdp2/worm_data.zip`.
- Downloaded bytes: **1,682,413,324**, matching the HTTP Content-Length.
- SHA-256: `dbdc04ffed5a6616eea5e72da843ca725af14f8b8e09477b870e1b4dbd0ddc8b`.
- ZIP CRC verification: **passed for all members**.
- Contents: 144 TIFF files and one directory entry.
- Total uncompressed member size: **18,226,205,856 bytes** (18.23 GB).
- TIFF files are extracted directly into `~/espaces/travail/trdp2/worm_data/`; raw data remain outside Git.
- Local detailed provenance/inventory: `~/espaces/travail/trdp2/worm_data/download_manifest.json`.

The bearer download token is deliberately absent from tracked documentation and
provenance. No permission to redistribute this dataset is inferred from the link;
this intake preserves the supervisor-supplied archive on university storage.

## Initial first-file header evidence (before full-array audit)

The first TIFF, `sino_selected/worm_004_gt.tiff`, is a 510-page array with
shape **(510, 100, 310)**, dtype **float64**, and TIFF axes **QYX**. Its description
records only the array shape. Resolution tags have unit NONE and values 1/1.
Pixels were not decoded for the initial header check. The subsequent full-array
audit is recorded in `worm_data/array_audit.json`; see the completed exploration
summary below.

All filenames end in `_gt.tiff`, but that naming does not establish whether these
are full-angle sinograms, reconstruction targets, or another reference array.
The archive contains no separate JSON/YAML/readme geometry file. Generic TIFF
axes QYX do not establish acquisition/physical axes. Do not transpose or convert
these arrays to the chest-volume format on filename evidence alone.

## Completed full-array exploration

All 144 TIFFs were decoded and checked with bounded memory, and every extracted
file hash matches its extraction inventory. The audit found:

| Property | Result |
|---|---|
| Array shape | (510, 100, 310) for all 144 files |
| Stored dtype / generic axes | float64 / QYX for all files |
| Nonfinite values | 0 |
| Negative values | 0 |
| Global range | 0 to 51,017.07421875 |
| Zero elements | 83.1789% of all stored elements |
| Exact byte-identical TIFF duplicates | None |

The array shape is not yet a verified physical voxel/angle/detector shape. The
user confirmed that no accompanying explanatory notes are available. The three
representative stored-axis plane sets show structured image content, but neither
the physical coordinate meanings nor independent-specimen identities are proven.

The complete per-file report is `~/espaces/travail/trdp2/worm_data/array_audit.json`.
A fixed-scale example image is `~/espaces/travail/trdp2/worm_data/preview.png`;
it uses the first, middle and last numeric filename IDs and central planes of
all three array axes, with one shared display range. Display saturation does not
change the TIFFs or numeric audit. The dataset README explains the local records.

The superseded `data/trdp2/anna_louise_brun_20261012/` folder was removed only after
successful extraction and relocation. The redundant nesting `sino_selected/`
was flattened; two small duplicate metadata files were merged into the retained
download manifest before removal. The raw archive and all 144 TIFFs are preserved.

## Next intake tasks

1. Establish the meaning/order of each axis, angle list/range/units, detector
   sampling, projection scaling, and beam geometry. Obtain source calibration or
   generation code if it is not encoded in the TIFFs.
2. Determine how worm IDs relate to independent specimens and whether selected
   files are slices/crops/repeats. Freeze splits at specimen level before tuning.
3. Determine which reference data exist and which reconstruction metrics are
   justified. A sinogram reference alone is not a known 3D volume target.
4. Record preprocessing and a new dataset-specific acquisition protocol, then
   validate tiny geometry/operator cases before reconstruction experiments.

The frozen TV 0.05 result is a reference for the old supplied chest convention.
Transfer to this dataset's intensity/geometry scale remains a research question.
No model training or reconstruction on this new dataset has been launched.

## Supervisor clarification received 13 October; recorded 17 October 2026

This supersedes the initial unknown-axis/geometry assessment above. Anna-Louise
confirmed synthetic noiseless parallel-beam sinogram stacks, axes (z, angle,
detector) = (510, 100, 310), axial/detector sampling 1 micrometre. The angular
count and step imply 0, 1.8, ..., 178.2 degrees (180 excluded). No flips were
reported, but numerical orientation/rotation-centre checks remain necessary.
All stacks represent the same worm deformed into different shapes: these are
not independent specimens. Ground-truth projection data are not voxel targets.

She describes values as global densities in mg/cm^3, to convert to g/cm^3.
The projection integration/normalization convention remains unresolved: physical
line integrals have different dimensions from density. Preserve stored values
until the custom simulator JSON or forward formula clarifies path-length factors.
All structures and quantitative mass-density accuracy matter, including edges
and interiors. Matching deformed density volumes have not yet been supplied.

The active direction is uncertainty across acquisition configurations and held-out
deformations, with controlled additive-noise stress tests. See
[the preparation protocol](../experiments/worm_uncertainty/README.md).
