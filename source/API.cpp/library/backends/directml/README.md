# DirectML backend

The Library exposes hardware-adapter discovery through `DirectmlHandler`.
Native DXGI, D3D12, and DirectML calls remain in
`features/detect/directml/definition/gpu/`. Workflow `121` provides
compiler-only coverage with the pinned Microsoft DirectML header package.
