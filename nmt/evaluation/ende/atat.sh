perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < ${1} > ${1}.atat
