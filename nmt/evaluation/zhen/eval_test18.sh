./delbpe.sh $1
./detokenizer.perl -u -l en < $1.delbpe > $1.delbpe.detok
./wrap-xml.perl en newstest2018-zhen-src.zh.sgm wc_test < $1.delbpe.detok > $1.delbpe.detok.sgm
./mteval-v13a.pl -c -r newstest2018-zhen-ref.en.sgm -s newstest2018-zhen-src.zh.sgm -t $1.delbpe.detok.sgm > $1.eval 
