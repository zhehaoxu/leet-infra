"""Frozen reference modules for the ``build-a-simple-agent`` project.

These files are the *official* implementations of earlier milestone steps. They
are injected read-only into the judging sandbox (see project.md section 2), so a
learner can start from any step without replaying their own earlier code.

Rules for anything in this package:
  * pure functions only; no globals holding state, no caches, no randomness;
  * every value that would come from an LLM or the outside world is an argument;
  * no imports beyond the Python standard library.
"""
