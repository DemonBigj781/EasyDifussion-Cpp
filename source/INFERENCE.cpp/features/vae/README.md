# VAE inference feature

INFERENCE.cpp owns the meaning, execution, and sequencing of VAE encode and decode. The
Common inference layer currently validates image/latent shapes, scaling policy,
and tiling policy and produces an encode or decode plan.

DiffUser does not own the VAE algorithm. It must supply Common handlers for the
VAE model resource, image resource, latent resource, storage placement,
transfers, and normalized compute primitives requested by the independently
developed INFERENCE implementation. Each claimed GPU backend needs matching
DiffUser definitions and translations for those administrative/primitives
routes; the VAE algorithm itself remains in INFERENCE.cpp.

Execution remains blocked until those resource handlers and primitives exist.
No SDKIT3 VAE implementation is linked or copied here.
