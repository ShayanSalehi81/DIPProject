# Micro-Q-Instruct Phase 1 Dataset Card

## Version and provenance

- Configuration version: `1.0`
- Master seed: `20260724`
- Micro-Bench dataset ID: `jnirschl/uBench`
- Micro-Bench immutable revision: `3f2c5b590bc7a208d5b60f3527ce4c76a331aa2b`
- Pool policy: strict external-train / official-evaluation separation

## Composition

- Unique clean source images: 1701
- Materialized recipe records: 15480
- Source roles: `{'development': 147, 'evaluation': 750, 'training': 804}`
- Source modalities: `{'electron': 183, 'fluorescence': 893, 'light': 625}`
- Source licenses: `{'CC-BY-4.0': 733, 'CC-BY-SA-4.0': 50, 'CC0': 50, 'CC0-1.0': 868}`
- Degradation records: `{'chromatic_misregistration': 203, 'chromatic_misregistration+defocus': 6, 'chromatic_misregistration+motion': 4, 'chromatic_misregistration+periodic': 1, 'chromatic_misregistration+quantization': 5, 'chromatic_misregistration+sampling': 4, 'chromatic_misregistration+saturation': 2, 'clean': 171, 'defocus': 1398, 'defocus+chromatic_misregistration': 8, 'defocus+jpeg': 38, 'defocus+motion': 16, 'defocus+periodic': 16, 'defocus+photobleaching': 13, 'defocus+poisson_gaussian': 8, 'defocus+quantization': 11, 'defocus+sampling': 30, 'defocus+saturation': 28, 'jpeg': 1621, 'jpeg+chromatic_misregistration': 3, 'jpeg+defocus': 26, 'jpeg+motion': 26, 'jpeg+periodic': 8, 'jpeg+photobleaching': 13, 'jpeg+poisson_gaussian': 25, 'jpeg+quantization': 32, 'jpeg+sampling': 28, 'jpeg+saturation': 30, 'motion': 1691, 'motion+chromatic_misregistration': 2, 'motion+defocus': 7, 'motion+jpeg': 18, 'motion+periodic': 17, 'motion+photobleaching': 17, 'motion+poisson_gaussian': 18, 'motion+quantization': 33, 'motion+sampling': 20, 'motion+saturation': 31, 'periodic': 1610, 'periodic+chromatic_misregistration': 7, 'periodic+defocus': 10, 'periodic+jpeg': 24, 'periodic+motion': 6, 'periodic+photobleaching': 7, 'periodic+poisson_gaussian': 24, 'periodic+quantization': 24, 'periodic+sampling': 32, 'periodic+saturation': 20, 'photobleaching': 762, 'photobleaching+chromatic_misregistration': 1, 'photobleaching+defocus': 19, 'photobleaching+motion': 6, 'photobleaching+periodic': 5, 'photobleaching+poisson_gaussian': 10, 'photobleaching+quantization': 16, 'photobleaching+sampling': 12, 'photobleaching+saturation': 9, 'poisson_gaussian': 1708, 'poisson_gaussian+chromatic_misregistration': 4, 'poisson_gaussian+defocus': 16, 'poisson_gaussian+jpeg': 17, 'poisson_gaussian+motion': 15, 'poisson_gaussian+periodic': 26, 'poisson_gaussian+photobleaching': 24, 'poisson_gaussian+quantization': 14, 'poisson_gaussian+sampling': 25, 'poisson_gaussian+saturation': 29, 'quantization': 1596, 'quantization+chromatic_misregistration': 3, 'quantization+defocus': 15, 'quantization+jpeg': 35, 'quantization+motion': 38, 'quantization+periodic': 27, 'quantization+photobleaching': 10, 'quantization+poisson_gaussian': 18, 'quantization+sampling': 7, 'quantization+saturation': 21, 'sampling': 1641, 'sampling+defocus': 49, 'sampling+jpeg': 23, 'sampling+motion': 25, 'sampling+periodic': 37, 'sampling+photobleaching': 11, 'sampling+poisson_gaussian': 25, 'sampling+quantization': 23, 'sampling+saturation': 14, 'saturation': 1577, 'saturation+defocus': 21, 'saturation+jpeg': 17, 'saturation+motion': 42, 'saturation+periodic': 18, 'saturation+poisson_gaussian': 40, 'saturation+quantization': 21, 'saturation+sampling': 16}`

## Intended uses

This release is intended for classical microscopy degradation analysis, severity calibration,
restoration evaluation, and later degradation-aware vision-language instruction construction.
Official Micro-Bench-derived records are evaluation-only.

## Out-of-scope uses

The release is not a diagnostic medical device, does not establish biological ground truth, and
must not be used to claim independent Micro-Bench performance after benchmark images have entered
training.

## Known limitations

Degradations are controlled approximations. Photobleaching is represented as retained signal,
defocus uses a disk PSF, periodic restoration uses the known synthetic frequency, and reference
morphology is available only for catalog records with aligned masks. Severity parameters are
empirical and require recalibration when the source or acquisition distribution changes.

## Licensing

Code licensing does not relicense source images. Each record retains its original source dataset,
license, and metadata. Consult the source license before redistributing generated derivatives.

## Reproduction

Archive this card with the frozen YAML configuration, generated manifest, pool-overlap report,
validation report, structured logs, and run-provenance JSON.
