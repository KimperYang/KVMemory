jbsub -cores 2x64+8 mpiwrap.sh -mem 400g -require 'a100_80gb,h100' -queue x86_6h bash scripts/training_bash/finetune_bias.sh


/hlt/exec/mpiwrap.sh [<flags>] <cmd> <arg1> <arg2> ...


jbsub -cores 2x64+8 -mem 400g -require 'a100_80gb,h100' -queue x86_6h mpiwrap.sh "bash scripts/training_bash/finetune_bias.sh"

