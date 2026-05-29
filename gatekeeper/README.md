# Gatekeeper

Gatekeeper is the core of MemoSight: an always-on, ultra-lightweight visual binary classifier that runs on the wearable device and decides whether a frame is worth capturing ("content present" vs. "nothing to record"). Only when it fires does the system wake up the expensive downstream pipeline — high-res capture, OCR, and memory card generation. This gate-then-process design is what makes continuous, on-device capture power-feasible.

The model is designed around three hard constraints: low-power continuous operation, on-device inference, and int8 quantization support. It takes low-resolution grayscale frames as input and is trained with standard frameworks to stay portable. The first target scene class is whiteboards, documents, and printed text — a well-scoped domain with clear positive/negative boundaries and available public datasets.

This directory contains all training and evaluation code for the gatekeeper model, along with scripts that produce the power-vs-recall Pareto trade-off curve — the primary quantitative deliverable of the project. The goal is to validate the model on a Raspberry Pi prototype, with MVP frozen by end of August 2026.
