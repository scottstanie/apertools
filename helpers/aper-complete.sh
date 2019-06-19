# Used to enable bash autocomplete on `aper` commands
# Usage: 
#	. .aper-complete.sh
#
# To automatically enable, add the above line to your ~/.bashrc

_aper_completion() {
    COMPREPLY=( $( env COMP_WORDS="${COMP_WORDS[*]}" \
                   COMP_CWORD=$COMP_CWORD \
                   _aper_COMPLETE=complete $1 ) )
    return 0
}

complete -F _aper_completion -o default aper;
