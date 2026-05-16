# Audio Embedding Exploration Task using YAMNet
Estimated effort: 4–6 hours maximum.

## Objective

Use YAMNet to analyze the provided audio clips and explore how the model represents different acoustic events over time through its embeddings.

Rather than relying solely on YAMNet’s predicted class labels, focus on understanding the structure and behavior of the embeddings themselves.

We are particularly interested in how you investigate, reason about, and interpret real-world audio data.

## Resources
- `sample_audio_clips` is a folder containing 3 audio clips for you to work with.
    - `sample_audio.mp3`: Acoustic events are clear with no background noise.
    - `busy_street.mp3` and `classroom.mp3`: Many overlapping acoustic events occur, background noise is omnipresent.

- `get_started_yamnet.py`: Contains basic functions that loads an input audio clip, runs this audio clip through the model, and outputs frame-level class prediction scores and embeddings from the model (you can use this code to get started, expand upon this, or otherwise choose to proceed with your own implementation).

You may use this script directly, modify it, or implement your own pipeline.

## Your Task

Analyze the embeddings and investigate questions such as:
- Do different temporal regions form distinct patterns in embedding space?
- Are there repeated or acoustically similar regions?
- How stable are embeddings across noisy or overlapping events?
- Do embedding transitions align with predicted class labels?
- Where does YAMNet appear reliable, uncertain, or misleading?
- Bonus: explore methods for improving robustness or temporal consistency of predictions.

### You are encouraged to use
- dimensionality reduction (PCA, t-SNE, UMAP),
- clustering,
- similarity analysis,
- temporal visualization,
- or any other method you find useful.

### Deliverables
Please submit:
- Your code/notebook
- Short report (1–2 pages preferred) including:
    - methodology, visualizations and observations
    - Interpretation of results
    - Limitations/questions you encountered
    - Any assumptions or experimental choices you made

## Notes
- There are no ground-truth labels.
- We care more about your reasoning and exploration process than “correct” answers.
- If embeddings do not form clear structures, investigate and discuss possible reasons.
- Keep the implementation lightweight and reproducible.


