# Thematic Signal Comparison Project

This project compares two ways of building a thematic signal from company summaries:

1. TF-IDF similarity to a theme description
2. Embedding similarity to the same theme description

Both signals are passed into the same convex optimizer.
A simple equal-weight portfolio is used as a benchmark.

This design isolates the effect of the signal-generation method while 
keeping the portfolio construction layer fixed.
