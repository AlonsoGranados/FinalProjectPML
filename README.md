# FinalProjectPML
To install Mujoco follow the steps in: https://github.com/openai/mujoco-py

To install Cliffwalking follow the steps in: https://github.com/caburu/gym-cliffwalking

To install Rlkit follow the steps in: https://github.com/vitchyr/rlkit

Once these environments are setup replace the files:

batch_rl_algorthms.py in rlkit/rlkit/core/, sac.py in rlkit/torch/sac/ and cliffwalking_env in gym-cliffwalking/gym_cliffwalking/envs/

And include files:
Adaptive_entropy.py in rlkit/examples/ and Q-learning.py in rlkit/examples/

To execute Mujoco:

Alpha decay,
include the parameter use_automatic_entropy_tuning=False in the trainer arguments for rlkit/examples/sac.py.

SAC, 
set use_automatic_entropy_tuning=True 

To execute cliffwalking:

Q-learning,
Run rlkit/examples/Q-learning.py

Soft Q-learning
Run rlkit/examples/Adaptive_entropy.py comment convergence rate

Alpha decay
Run rlkit/examples/Adaptive_entropy.py uncoment convergence rate

State decay
Run rlkit/examples/Adaptive_entropy.py uncoment state converge rate

Sidenote:
The training for mujoco took about 8 hours per model in a GPU GEforce 1660 ti.
