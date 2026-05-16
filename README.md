# Audio Embedding Exploration Task (YAMNet)
Objective

Use YAMNet to analyze the provided audio clips and explore how the model represents different acoustic events over time through its embeddings.

Rather than relying solely on YAMNet’s predicted class labels, focus on understanding the structure and behavior of the embeddings themselves.

We are particularly interested in how you investigate, reason about, and interpret real-world audio data.

Provided Files
sample_audio.mp3
Clear acoustic events with minimal background noise
busy_street.mp3
Multiple overlapping sound sources and continuous background noise
classroom.mp3
Complex real-world acoustic environment with overlapping events
get_started_yamnet.py
Starter script that loads audio, runs YAMNet, and outputs:
frame-level prediction scores
embedding vectors

You may use this script directly, modify it, or implement your own pipeline.

Your Task

Analyze the embeddings and investigate questions such as:

Do different temporal regions form distinct patterns in embedding space?
Are there repeated or acoustically similar regions?
How stable are embeddings across noisy or overlapping events?
Do embedding transitions align with predicted class labels?
Where does YAMNet appear reliable, uncertain, or misleading?

You are encouraged to use:

dimensionality reduction (PCA, t-SNE, UMAP),
clustering,
similarity analysis,
temporal visualization,
or any other method you find useful.
Deliverables

Please submit:

Your code/notebook
Short report (1–2 pages preferred) including:
methodology,
visualizations,
observations,
interpretation of results,
limitations/questions you encountered
Any assumptions or experimental choices you made
Notes
There are no ground-truth labels.
We care more about your reasoning and exploration process than “correct” answers.
If embeddings do not form clear structures, investigate and discuss possible reasons.
Keep the implementation lightweight and reproducible.
