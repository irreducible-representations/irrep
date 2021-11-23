for dir in */; do
  echo $dir
  cd $dir
  bash run.sh
  cd ..
done
