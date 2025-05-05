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
NUM_ITERATIONS=500
TIMESTEPS=$((NUM_ITERATIONS * $NUM_ENVS * $NUM_STEPS))
echo "time steps: "
echo $TIMESTEPS

NUM_EVALUATION_EPISODES=10000
echo "num evaluation episodes: "
echo $NUM_EVALUATION_EPISODES

for args in "${ARGS[@]}"
do
  set -- $args 
  ARG1=$1
  ARG2=$2

  echo "######"
  echo "#####" 
  echo "STARTING A RUN"  
  echo "Running $PYTHON_SCRIPT with arguments: $ARG1 $ARG2"
  python3 "$PYTHON_SCRIPT" --env_type "$ARG1" --num_lots "$ARG2" --total_timesteps "$((TIMESTEPS))" --num_envs "$((NUM_ENVS))" --num_steps "$((NUM_STEPS))" --n_evalutation_episodes "100" 
done

#   # Run the script and log output
#   python3 "$PYTHON_SCRIPT" --env_type "$ARG1" --num_lots "$ARG2" --total_timesteps "$TIMESTEPS" --num_envs "128" --num_steps "100" --n_evalutation_episodes "10000" --tag "best_price_new" >> script_output.log 2>&1

#   # Check if the last command succeeded
#   if [ $? -ne 0 ]; then
#     echo "Error: Execution failed for args $ARG1 $ARG2. Check script_output.log for details."
#   fi
# done


echo "All processes completed."