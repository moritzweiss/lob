# need to run this script from the root directory of the project !!
#!/bin/bash

echo "starting the script now" 

PYTHON_SCRIPT="/u/weim/lob/rl_files/actor_critic.py"

declare -a ARGS=(
  "noise 20"
  "noise 60"
  "flow 20"
  "flow 60"
  "strategic 20"
  "strategic 60"
)

for args in "${ARGS[@]}"
do
  set -- $args 
  ARG1=$1
  ARG2=$2
  TIMESTEPS=$((500 * 128 * 100))

  echo "######"
  echo "#####" 
  echo "STARTING A RUN"  
  echo "Running $PYTHON_SCRIPT with arguments: $ARG1 $ARG2"
  python3 "$PYTHON_SCRIPT" --env_type "$ARG1" --num_lots "$ARG2" --total_timesteps "$TIMESTEPS" --num_envs "128" --num_steps "100" --n_evalutation_episodes "10000" --tag "best_price_new"
  # var reduction is to variance of 0.1
done

echo "All processes completed."