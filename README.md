# LADMG - Latent audio diffusion for music generation with expressive control

Recent advances in generative deep learning models provide exciting new tools for music generation. In particular, the conditioning capabilities of diffusion models offer interesting possibilities to add expressive control to the generation process, which helps make it a more accessible tool. In this paper, we apply the novel diffusion method Iterative $\alpha$-(de)Blending, which simplifies the usual formalism of stochastic diffusion models to a deterministic one, so as to generate audio loops from pure noise. We use a high fidelity neural autoencoder to generate latent codes of a compressed audio representation. Conditioning is applied using either beats-per-minute information or high-level audio concepts, and reinforced using classifier-free guidance. The latent codes are then inverted back to a waveform with the decoder. Finally, we assess the quality of our method on a large dataset of minimal electronic music.
