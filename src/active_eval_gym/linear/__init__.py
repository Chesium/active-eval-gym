"""Exact analysis of known linear plant families under shared linear feedback.

This layer supports the ESE 6180 certificate and gain-update study. It works
with known matrices rather than Gymnasium environments, and it never trains,
samples, or writes artifacts. Import from the submodules: ``family`` (plants),
``exact`` (closed-loop quantities), and ``certificate`` (LMI search and
verification, which imports cvxpy).
"""
