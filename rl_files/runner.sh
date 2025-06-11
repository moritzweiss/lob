# need to run this script from the root directory of the project !!
#!/bin/bash

echo "starting the script now" 

PROJECT_ROOT="${PWD}"
echo $PROJECT_ROOT
PYTHON_SCRIPT="$PROJECT_ROOT/rl_files/actor_critic.py"
echo $PYTHON_SCRIPT

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
  echo "Error: Python script not found at $PYTHON_SCRIPT"
  exit 1
fi

declare -a ARGS=(
  "noise 20"
  "noise 60"
  "flow 20"
  "flow 60"
  "strategic 20"
  "strategic 60"
)

NUM_STEPS=100
NUM_ENVS=128 
NUM_ITERATIONS=400
# NUM_ITERATIONS=3
TIMESTEPS=$((NUM_ITERATIONS * $NUM_ENVS * $NUM_STEPS))
echo "time steps: "
echo $TIMESTEPS
echo "num steps: "
echo $NUM_STEPS
echo "num envs: "
echo $NUM_ENVS
echo "num iterations: "
echo $NUM_ITERATIONS

NUM_EVALUATION_EPISODES=10000
# NUM_EVALUATION_EPISODES=10 
echo "num evaluation episodes: "
echo $NUM_EVALUATION_EPISODES

# for exp_name in "log_normal" "dirichlet"; 
for exp_name in "log_normal" 
# normal distribution performs more or less the same as dirichlet. therefore we are ignoring the normal distribution for the moment 
do
for args in "${ARGS[@]}"; 
do
  set -- $args 
  ARG1=$1
  ARG2=$2

  echo "#####"
  echo "#####" 
  echo "STARTING A RUN"  
  echo "Running experiment: $exp_name with $ARG1 $ARG2"
  python3 "$PYTHON_SCRIPT" --env_type "$ARG1" --num_lots "$ARG2" --total_timesteps "$((TIMESTEPS))" --num_envs "$((NUM_ENVS))" --num_steps "$((NUM_STEPS))" --n_eval_episodes "$((NUM_EVALUATION_EPISODES))" --exp_name "$exp_name" --num_iterations "$((NUM_ITERATIONS))" --tag "raw_rewards"
   
done
done 


echo "All processes completed."