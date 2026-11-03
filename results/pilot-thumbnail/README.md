# Superseded thumbnail pilot

These files document an earlier pipeline check. Its inputs came from the ABO `images/small/` service, where the longest edge was 256 pixels. Those small JPEGs were enlarged to make nominal 512×512 references; therefore the original high-frequency content at 512 pixels was unavailable. Do not use these figures as the native-resolution restoration benchmark.

`formal-summary.json` gives **23.697 dB** for full diffusion and **30.428 dB** for GPU SwinIR in that thumbnail pilot. `classical-summary.json` and `clean-clone-thumbnail-stage1.json` document the separate thumbnail/CPU check. The primary experiment uses the original JPEGs in `examples/originals/` and is indexed under `results/formal/`.

