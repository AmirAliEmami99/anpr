index () {
if [ $1 == "IMG" ]; then
  echo 1
  python Salam1.py & python Salam2.py
elif [ $1 == "CON" ]; then
  echo 2
elif [ $1 == "TrustedCON" ]; then
  echo 3
elif [ $1 == "Number" ];  then
  echo 4
fi
}

index "IMG"

