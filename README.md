# Dynamic Workshop Scheduling Considering Rework Path based on State Space Updates
## Abstract
Improving product quality is crucial in aviation product manufacturing. Compared with directly scrapping products, rework of intermediate products helps to ensure that the final delivered product meets the predetermined quality standards with 
less cost. To address the uncertainty of the machining process caused by rework tasks and the optimization problem of rework paths, a dynamic workshop scheduling problem (DJSP-RP) considering rework paths is established. Firstly, based on the 
collaborative virtual workflow model, a dynamic scheduling framework considering rework based on state space updates is proposed with several state space variables such as rework progress are introduced, which can coordinate scheduling tasks and 
decision-making mechanisms; Secondly, for the problem of deterministic machining tasks in different scenarios, an improved multi-objective evolutionary algorithm is used to solve a reliable scheduling scheme. A certain aviation wall panel laser 
processing section is taken as the research object, selecting maximum completion time and total quality cost as optimization objectives. In static production scheduling without considering rework tasks, multi-objective evolutionary algorithm based on 
decomposition is used to find non-dominated solution-sets. In a dynamic environment considering rework, taking robustness indicators into account, it is verified that the proposed rescheduling method is closer to the ideal value point in the target domain 
compared to rightward shift rescheduling.

![image](https://github.com/user-attachments/assets/9242c3f0-b98b-444b-8413-7fa5e89f937b)

![image](https://github.com/user-attachments/assets/1e77ec52-ef66-4969-9fd3-51efd488fb0b)

![image](https://github.com/user-attachments/assets/32706b2b-1f69-4432-a157-15d99c05948d)

## Context
The working mechanism of collaborative task flow with the introduction of state space update rules is shown in Figure 3. First, an initial scheduling scheme is generated according to the order requirements and available machines in the work section, and a
real-time task flow is formed according to the scheduling scheme. Compared with the virtual workflow, events such as tight post-process processing and rework are triggered when passing the quality detection point.Update the status space according to the 
attributes of the nodes in the real-time task flow (critical/non-critical processes), integrate the status of the task nodes to feed back to the production environment, and the scheduling system updates the unfinished processes, determines whether to 
reschedule them, and continues to update the real-time task flow.The relationship between task flow working mechanism and decision variable encoding and decoding is shown in Figure 4.Since the decomposition-based multi-objective evolutionary algorithm (MOEA/D)
has shown remarkable performance in solving multi-objective optimization problems, this paper uses MOEA/D for static scheduling to obtain the initial scheduling strategy.Each rescheduling triggers the reading of the state space and the initiation of MOEA/D, 
and the newly generated scheduling scheme also affects the evolution of the state space.In this paper, the individual with the highest cosine similarity to the neutral weight vector [1,1] on the initial Pareto solution set is selected for event simulation. 
Since a better solution has been achieved based on static scheduling, the chromosome corresponding to the right shift strategy is taken as the initial starting point, the population size is initialized near the starting point, and then the MOEA/D algorithm is 
used to solve it.The scheme achieves better performance in the three objectives of maximum completion time, quality loss and robustness.

![image](https://github.com/user-attachments/assets/a7fff49c-61cb-471e-8bc8-a6a096a8f5e2)

![image](https://github.com/user-attachments/assets/540f3e6d-5b0d-4690-9c17-962c9b624f5f)

![image](https://github.com/user-attachments/assets/cb4c92e7-57cb-44d7-b552-9d41646c3aa9)

## Conclusion
Aiming at the dynamic job-shop scheduling problem considering rework tasks, this paper proposes a reactive rescheduling method based on state space update. Firstly, the non-dominated solution set in static production environment is solved, and then the rework 
situation of key nodes is detected in dynamic environment, the state space is updated, and the new resource set to be scheduled is rescheduled using a decomposition-based multi-objective evolutionary algorithm.The uncertainty of machining process and the 
assignment of rework equipment caused by rework tasks were effectively solved, and the two optimization objectives of maximum completion time and total quality cost were optimized, which provided certain reference significance for the production scheduling of 
aviation structural parts considering rework.In the dynamic events related to product quality, this paper only considers the impact of rework on the production process, and the method will be further discussed in combination with the scenarios of product scrapping 
and repair.
