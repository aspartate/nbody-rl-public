# Physics-informed deep Q-learning for $n$-body simulations

## Overview
The $n$-body problem in astronomy is a challenging task that involves predicting the motion of celestial bodies under gravitational attraction. As the number of bodies increases, simulating and accurately predicting their motion becomes computationally expensive. To address this issue, various optimization schemes have been developed, including symplectic integrators, tree-based methods, fast multipole solvers, and adaptive timestepping.

In this project, we propose a novel approach to tackle the $n$-body problem by leveraging reinforcement learning techniques. Specifically, we utilize deep Q-learning to optimize the step size selection in adaptive timestepping. Our method explicitly encodes physical constraints in the training paradigm, enabling the reinforcement learning algorithm to learn optimal step sizes for efficient and accurate simulations.

Please note that the code in this project to date is primarily intended as a proof of concept and a starting point for further development. However, it showcases the feasibility of using physics-informed reinforcement learning for $n$-body systems, opening up possibilities for further advancements and improvements in the field.
