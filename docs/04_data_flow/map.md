config ─► HSIDataset (main process, once): SimConfig + material library
              └─ split(name, run) ─► HSISplit: size, seed keys, fresh?

per step:

SeededBatchSampler (main process)
   knows: split size, batch size, sampler_seed, epoch (set by accelerate before each epoch)
   yields: (epoch, indices[B])
      │
      │   DataLoader sends the item to a worker; with several GPUs, accelerate deals items out to ranks
      │
      ▼
HSISplit.__getitem__ (worker)
   for each index: seed = scene_seed(keys, [epoch if train], index) ─► generate_sample
   returns {index: {input [36,15,15], label [15,15], index}}
      ▼
collate_volumes (worker) ─► (inputs [B,36,15,15], masks [B,15,15], indices [B])
      ▼
handle_batch (GPU)
   inputs  ─► model ─► logits [B,47,15,15]
   masks   ─► loss, SegmentationMeter
   indices ─► test_preds (test loader only) ─► test_predictions.npz