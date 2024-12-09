LAUNCH_CMD="bash scripts/training_bash/multi_node.sh"
jbsub -cores 4x32+8 -mem 400g -require 'a100_80gb,h100' -queue x86_1h /hlt/exec/mpiwrap.sh ${LAUNCH_CMD}



jbsub -cores 4x32+8 -mem 400g -require 'a100_80gb,h100' -queue x86_1h /hlt/exec/mpiwrap.sh bash scripts/training_bash/multi_node.sh
