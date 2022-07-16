#index () {
#  n=$1
#  m=$((n-1))
#  shahid=""
#  for (( i=0 ; i<=$n-2 ; i++ ));
#  do
##    echo $shahid
#    shahid+="python ReadRequest.py $i & "
##      python ReadRequest.py $i
#  done
#  shahid+="python ReadRequest.py $m"
#  echo $shahid
#  eval $shahid
#
#
#}

index () {
  n=$1
  m=$((n))
  shahid=""
  for (( i=1 ; i<=$n-1 ; i++ ));
  do
#    echo $shahid
    shahid+="python ReadRequest.py $i & "
#      python ReadRequest.py $i
  done
  shahid+="python ReadRequest.py $m"
  echo $shahid
  eval $shahid


}
index 2

