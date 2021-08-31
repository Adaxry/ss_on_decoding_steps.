sh ./delbpe.sh $1
sh ./atat.sh $1
./multi-bleu.perl ./test.fr.delbpe.atat < ${1}.delbpe.atat > ${1}.delbpe.atat.eval
cat ${1}.delbpe.atat.eval  
