## Exercises

1. Consider the maximum element algorithm from Chapter 1. Prove, by induction,
   that this algorithm does indeed produce the maximum element when it
   finishes.
2. A general polynomial of degree $n$ is usually written in the following
   form:

   $$
   p(x) = a_0 + a_1x + a_2x^2 + \dots + a_nx^n = \sum_{i=0}^n a_ix^i.
   $$

   Here the $a_j$ are constants and $x$ is an unknown value (an
   indeterminate). This exercise introduces **Horner's method** for evaluating
   a polynomial.

    1. Prove, using induction on the degree of the polynomial, that a degree
       $n\geq 1$ polynomial can be rewritten recursively as

       $$
       q_n(x) = a_0 + xq_{n-1}(x), q_{n-1}(x) = a_1 + q_{n-2}(x),\dots,
       q_1(x)=a_{n-1} + a_nx.
       $$

    2. How many operations are involved in evaluating this polynomial by
       writing it in this way? Why is this better than evaluating the
       polynomial from its standard form?
