Compute the echoing period Δ of the Choptuik critical solution in every whole number of spacetime dimensions D from 6 to 200, each correct to six decimal places.

The critical solution is the discretely self-similar solution of general relativity with a spherically symmetric, minimally coupled, massless real scalar field that sits at the threshold of black hole formation. Δ is its period in logarithmic time.

Work from first principles in the current directory: derive the equations, write the solver, and test your own results for convergence. Do not look the values up, and do not quote them from memory. Python 3 with numpy, scipy, mpmath and sympy is installed. You have 4 CPU threads and 12 hours.

Write two files, and keep both up to date as you go, because the work is stopped at 12 hours whether or not it is finished:
- results.csv, with the header D,Delta,uncertainty and one row for each D you have a value for;
- METHOD.md, which says how each value was computed and what its convergence test showed.
