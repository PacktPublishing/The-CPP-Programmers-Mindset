## Solutions to exercises

1. As described the proof is by induction on the length of the list. Recall
   that the setup of the algorithm is as follows. We have a list $L$ conprised
   of $n$ ordered elements. Let $P(k)$ be the statement that if $M$ denotes the
   output of the algorithm applied to a list of $k>0$ elements, then $M$ is the
   maximum element of this list. At this point it is worth reminding ourselves
   that the algorithm is iterative in nature, so the algorithm applied to $L$
   is the same as applying the algorithm applied to the intial sublist of $L$
   consisting of all but the final element, plus one additional step.

   The base case for the induction is $n = 1$. In this case, the output of the
   algorithm is the only value in the list which is clearly the maximum value.

   Suppose now that, for some $r > 0$, $P(r)$ is true and consider a list $L$ of
   $n = r + 1$ elements. As mentioned, applying the algorithm to $L$ is the same
   as applying the algorithm to the first $r$ elements of $L$ followed by one
   additional step. Let $M'$ be the output of the algorithm applied to the the
   first $r$ elements of $L$. By the inductive hypothesis $P(r)$, this $M'$ is
   indeed the maximum element of the first $r$ elements of $L$. Now apply the
   one additional step, which compares the $(r+1)$st element ($L[r]$ assuming
   $0$-based indexing) of $L$ with $M'$, and sets $M$ with the larger of $L[r]$
   and $M'$. Thus, after applying this step, we are left with $M$ which is at
   least as large as both $M'$ and $L[r]$ and is either equal to $L[j]$ for $j <
   r$ or is equal to $L[r]$. Thus $M$ is the maximum of $L$. This proves the
   inductive step, and completes the proof.

2. This question has multiple parts. We solve each separately.

    1. We first need to prove, by induction, that we can write any polynomial
       $p(x)$ of degree $n > 0$ in recursive form as polynomials $q_j(x) =
       a_{n-j} + q_{j-1}(x)$ ($j > 1$) and $q_1(x) = a_{n-1} + a_nx$. This is a
       very straightforward exercise.

       For $n = 1$ the proof is quite trivial. Indeed, the polynomial
       $p(x) = a_0 + a_1x$ is already in the desired form $q_1(x)$.

       Now assume that a polynomial of degree $n\geq 1$ can be written in the as
       such a system and consider a degree $n+1$ polynomial $p(x) = a_0 + a_1x +
       \dot + a_{n+1}x^{n+1}$. Factorizing this as

       $$
       p(x) = a_0 + x(p'(x)) = a_0 + x(a_1 + a_2x^2 + \dots + a_{n+1}x^{n}).
       $$

       This inner polynomial $p'(x)$ as degree $n$, so by the inductive
       hypothesis can be written recrusively as polynomials $q_j'(x)$
       ($j=1,2,\dots, n$). Further inspection yields that $q_1'(x) = a_n +
       a_{n+1}x$ and $q_j'(x) = a_{n+1-j} + xq_{j-1}'(x)$ for $j > 1$.
       Setting $q_j(x) = q_j'(x)$ for $j=1,2,\dots n$ and setting $q_{n+1}(x) =
       a_0 + xq_{n}'(x)$ gives us the system we need. This completes the proof.

    2. Evaluating one of the polynomials $q_n(x)$ in Hörner's method requires
       one addition and one multiplication, assuming $q_{n-1}(x)$ has already
       been evaluated. (Or a single fused multiply addition!) Thus the
       complete evaluation requires $n$ additions and $n$ multiplications.
       Compare this to evaluating a single power $x^n$, which would require
       $O(\log n)$ multiplications, and that's before we consider the
       coefficients. A secondary effect is that this recursive evaluation
       has better numerical stability. When we add two floating point numbers
       of greatly different magnitude, the result is usually not exactly
       equal to the sum of the two numbers. In particular, if $x$ is small
       then $x^n$ will be small compared to $x$ so, when aggregating, the sum
       the lower-order terms will be larger in magnitude than the
       higher-order terms so will dominate the sum. In contrast, each of the
       sums in the polynomials $q_n(x)$ will have operands of more comparable
       magnitudes. This means there is a smaller chance that one of the terms
       dominates the sum. (Of course this is not impossible if, for instance,
       the coefficients themselves have greatly different magnitudes.)