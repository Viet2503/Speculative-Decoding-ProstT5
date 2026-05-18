# Speculative Decoding for ProstT5

## Background

Proteins can be represented either by their primary amino acid (AA) sequence or their three-dimensional structure. [Foldseek][1] introduced a discretization of 3D structural features into a 1D "structural alphabet" known as 3Di tokens. This allows protein modeling to be framed as a machine translation problem for which [ProstT5][2] implements the translations in both directions:

1. **Inverse folding (3Di → AA):** Designing a functional amino acid sequence that will fold into a specific, pre-defined 3D backbone.
2. **Folding (AA → 3Di):** Predicting the structural states (3Di) from a raw amino acid sequence—a task analogous to secondary structure prediction but with higher structural resolution.

The ProstT5 encoder-decoder (enc-dec) architecture handles these "translations" with high accuracy through autoregressive decoding. However, this process is computationally expensive: to generate a sequence of length *L*, the model must perform *L* sequential passes, where each step depends on the previous output. This creates a significant bottleneck for high-throughput protein design.

To mitigate this, one-shot **enc-CNN** models were developed. These models use the ProstT5 encoder embeddings but replace the heavy transformer decoder with a lightweight Convolutional Neural Network (CNN) that predicts the entire sequence in a single parallel pass. While the CNN is approximately 4000× faster, it lacks the global context and consistency of the autoregressive decoder, leading to slightly decreased sequence recovery.

This project proposes to combine the strengths of both approaches using [speculative decoding][3]. In this framework, a fast **draft** model (such as the enc-CNN, a Profile HMM, or structural profiles) suggests a block of *K* candidate tokens for the different translation directions. The heavy enc-dec model then verifies these tokens in a single forward pass. Because verification is parallelizable, this "assisted generation" can achieve the high accuracy of the transformer with a latency closer to that of the fast CNN, provided the draft model's **acceptance rate** is sufficiently high. Output generated with this approach is exactly the same as running the enc-dec.

## Exploration Steps

We will focus on the **inverse folding (3Di → AA)** direction first, and then work on the **folding (AA → 3Di)** direction depending on time.

### 1. Establish baseline performance

- Establish a baseline of the inverse folding and folding direction for a range of protein lengths.
- Measure latency (seconds/protein), throughput (tokens/second), and vRAM usage.
- Report this for the **enc-dec** (the full ProstT5 model) and **enc-CNN** (encoding using ProstT5 and then running the CNN for the direction; you find the CNNs in the [ProstT5 GitHub repo](https://github.com/mheinzinger/ProstT5)) models.
- How well do enc-dec and enc-CNN agree?

### 2. Establish the draft models

Ideas for draft models:

- **Using the enc-CNN model** (inverse folding and folding)
  - The enc-CNN model gives you a single output for the full sequence, independent of any already generated prefix.
- **Using evolutionary information from an MSA** (inverse folding direction only)
  - Using a Multiple Sequence Alignment (MSA) of a specific protein family (e.g., Glycosyltransferases), prune the HMM to fit the length of the template protein and build a Profile HMM. Can you make this drafter **prefix-aware**? It should be able to look at the residues already verified by the main model and "sharpen" its guess for the next block of amino acids.
  - For the folding direction, the same approach might be used for aligned 3Di sequences.
- **Using predicted 3Di-Flex profiles** (folding direction only)
  - [ProtProfileMD][4] provides a predicted profile over 3Di states.
- **(Optional)** Explore other drafting methods.

### 3. Implement assisted generation

- Implement different drafters for inverse folding and folding directions, optimally using the `assistant_model` parameter of the Transformers `generate` method.

### 4. Optimization & analysis

- Experiment with different window sizes (*K*) for drafting. This might depend on your methods (and is not really applicable for enc-CNNs as drafters) and also, e.g., conservation of the amino acids when you have the MSA. Implement fixed and dynamic choice of *K*.
- Report the final speedup achieved and the **acceptance rate** (how often the big model agreed with the draft). How much overhead does the draft model use (runtime of draft, vRAM usage)?

### 5. (Optional) Generalization to other pLMs

- Can your draft models be applied to other protein language models (e.g., ProFam)?

## Literature

- **Speculative decoding:** Leviathan, Y., Kalman, M. & Matias, Y. (2023). *Fast Inference from Transformers via Speculative Decoding.* Proceedings of the 40th International Conference on Machine Learning, in *Proceedings of Machine Learning Research* 202:19274–19286. [https://proceedings.mlr.press/v202/leviathan23a.html](https://proceedings.mlr.press/v202/leviathan23a.html)
- **ProstT5:** Heinzinger, M., Weissenow, K., Sanchez, J. G., Henkel, A., Mirdita, M., Steinegger, M., & Rost, B. (2024). Bilingual language model for protein sequence and structure. *NAR Genomics and Bioinformatics*, 6(4). [https://doi.org/10.1093/nargab/lqae150](https://doi.org/10.1093/nargab/lqae150)
- **Assisted generation (Hugging Face):** [https://huggingface.co/blog/assisted-generation](https://huggingface.co/blog/assisted-generation)

## References

[1]: https://doi.org/10.1038/s41587-023-01773-0 "van Kempen et al. (2023) — Foldseek"
[2]: https://doi.org/10.1093/nargab/lqae150 "Heinzinger et al. (2024) — ProstT5"
[3]: https://proceedings.mlr.press/v202/leviathan23a.html "Leviathan et al. (2023) — Speculative Decoding"
[4]: https://doi.org/10.64898/2026.01.21.700698 "Lüth et al. (2026) — ProtProfileMD"

**[1]** van Kempen, M., Kim, S. S., Tumescheit, C., Mirdita, M., Lee, J., Gilchrist, C. L. M., Söding, J., & Steinegger, M. (2023). Fast and accurate protein structure search with Foldseek. *Nature Biotechnology*, 42(2), 243–246. https://doi.org/10.1038/s41587-023-01773-0

**[2]** Heinzinger, M., Weissenow, K., Sanchez, J. G., Henkel, A., Mirdita, M., Steinegger, M., & Rost, B. (2024). Bilingual language model for protein sequence and structure. *NAR Genomics and Bioinformatics*, 6(4). https://doi.org/10.1093/nargab/lqae150

**[3]** Leviathan, Y., Kalman, M. & Matias, Y. (2023). Fast Inference from Transformers via Speculative Decoding. *Proceedings of Machine Learning Research* 202:19274–19286.

**[4]** Lüth, F. H., Mihaila, V., Mirdita, M., Steinegger, M., Rost, B., & Heinzinger, M. (2026). Protein Language Modeling beyond static folds reveals sequence-encoded flexibility. *openRxiv*. https://doi.org/10.64898/2026.01.21.700698
