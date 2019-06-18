# Used to enable bash autocomplete on `apertools` commands
# Usage: 
#	. .apertools-complete.sh
#
# To automatically enable, add the above line to your ~/.bashrc

_apertools_completion() {
    COMPREPLY=( $( env COMP_WORDS="${COMP_WORDS[*]}" \
                   COMP_CWORD=$COMP_CWORD \
                   _apertools_COMPLETE=complete $1 ) )
    return 0
}

complete -F _apertools_completion -o default apertools;
